from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _event_segments(mask: np.ndarray) -> List[tuple]:
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _max_consecutive_below(values: np.ndarray, thr: float) -> int:
    max_run = 0
    run = 0
    for v in values:
        if np.isnan(v):
            run = 0
            continue
        if v < thr:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _median_reliable(values: pd.Series, reliable: pd.Series):
    if values is None or reliable is None:
        return None
    mask = reliable.astype(bool) & values.notna()
    if mask.sum() == 0:
        return None
    return float(values[mask].median())


def _resolve_speed_col(fused: pd.DataFrame, cfg: Dict[str, float]) -> str:
    cand = cfg.get("obd_speed_col") or "obd_Vehicle Speed Sensor [km/h]"
    cols = set(fused.columns)
    if cand in cols:
        return cand
    if not cand.startswith("obd_") and f"obd_{cand}" in cols:
        return f"obd_{cand}"
    if cand.startswith("obd_") and f"obd_{cand}" in cols:
        return f"obd_{cand}"
    if cand.startswith("obd_") and cand[4:] in cols:
        return cand[4:]
    for col in cols:
        if "Vehicle Speed Sensor" in col:
            return col
    return cand


def decide_events(fused: pd.DataFrame, cfg: Dict[str, float]) -> pd.DataFrame:
    if fused is None or fused.empty:
        return pd.DataFrame()

    tlc_col = "road_lane_tlc_s"
    dist_col = "road_lane_dist_to_edge_m"
    lane_conf_col = "road_lane_lane_conf"
    gaze_col = "driver_gaze_on_road_prob"
    yaw_col = "phyphox_phy_yaw_rate_rads"
    vehicles_col = "road_scene_vehicles_count"
    speed_col = _resolve_speed_col(fused, cfg)

    polar_hr_z_col = "polar_polar_hr_z"
    polar_rmssd_z_col = "polar_polar_rmssd_z"
    polar_rel_col = "polar_polar_reliable"

    muse_beta_alpha_z_col = "muse_muse_beta_alpha_ratio_z"
    muse_rel_col = "muse_muse_reliable"

    tlc = fused.get(tlc_col, pd.Series([np.nan] * len(fused)))
    dist = fused.get(dist_col, pd.Series([np.nan] * len(fused)))
    lane_conf = fused.get(lane_conf_col, pd.Series([0] * len(fused))).fillna(0)
    lane_conf_ok = lane_conf >= 0.5
    speed_kmh = fused.get(speed_col, pd.Series([np.nan] * len(fused)))
    smooth_n = int(cfg.get("obd_decel_smooth_n", 3) or 3)
    if smooth_n < 1:
        smooth_n = 1
    speed_ms = speed_kmh / 3.6
    speed_ms = speed_ms.rolling(window=smooth_n, center=True, min_periods=1).mean()
    accel_ms2 = speed_ms.diff() * cfg["resample_hz"]

    is_event = lane_conf_ok & ((tlc < cfg["tlc_thr"]) | (dist < cfg["edge_thr"]))
    segments = _event_segments(is_event.values)

    events = []
    for idx, (start, end) in enumerate(segments, start=1):
        segment = fused.iloc[start : end + 1]
        tlc_seg = segment.get(tlc_col)
        dist_seg = segment.get(dist_col)

        if tlc_seg is not None and tlc_seg.notna().any():
            peak_idx = tlc_seg.idxmin()
        else:
            peak_idx = dist_seg.idxmin() if dist_seg is not None else segment.index[start]

        peak_t = float(fused.loc[peak_idx, "t_seconds"])
        min_tlc = float(tlc_seg.min()) if tlc_seg is not None else float("nan")
        min_dist = float(dist_seg.min()) if dist_seg is not None else float("nan")
        crossed = False
        if not np.isnan(min_dist):
            crossed = min_dist <= 0.0

        window_attn = fused[(fused["t_seconds"] >= peak_t - 2.0) & (fused["t_seconds"] <= peak_t + 0.5)]
        gaze_vals = window_attn.get(gaze_col, pd.Series(dtype=float)).to_numpy()
        max_run = _max_consecutive_below(gaze_vals, cfg["attn_thr"])
        distracted = max_run / cfg["resample_hz"] >= cfg["attn_min_s"]

        lane_conf_peak = fused.loc[peak_idx, lane_conf_col] if lane_conf_col in fused else 0
        lane_conf_peak = float(lane_conf_peak) >= 0.5

        window_context = window_attn
        yaw_vals = window_context.get(yaw_col, pd.Series(dtype=float))
        veh_vals = window_context.get(vehicles_col, pd.Series(dtype=float))
        defensive = False
        if yaw_vals.notna().any() and yaw_vals.abs().max() > cfg["yaw_thr"]:
            defensive = True
        if veh_vals.notna().any() and veh_vals.max() >= cfg["veh_thr"]:
            defensive = True

        speed_vals = speed_kmh.loc[window_context.index] if not speed_kmh.empty else pd.Series(dtype=float)
        accel_vals = accel_ms2.loc[window_context.index] if not accel_ms2.empty else pd.Series(dtype=float)
        speed_med = float(speed_vals.median()) if speed_vals.notna().any() else float("nan")
        decel_min = float(accel_vals.min()) if accel_vals.notna().any() else float("nan")
        speed_min = float(cfg.get("obd_speed_min_kmh", 0.0) or 0.0)
        decel_thr = float(cfg.get("obd_decel_mps2_thr", 0.0) or 0.0)
        obd_state = "unknown"
        hard_brake = False
        if speed_vals.notna().any() and accel_vals.notna().any():
            if speed_med < speed_min:
                obd_state = "low_speed"
            elif decel_min <= -decel_thr:
                hard_brake = True
                obd_state = "hard_brake"
            else:
                obd_state = "no_hard_brake"

        should_warn = 0
        reason = "ATTENTIVE_INTENTIONAL"
        warning_strength = "NONE"
        confidence = "LOW"

        if distracted:
            should_warn = 1
            reason = "DISTRACTED_GAZE"
            warning_strength = "MEDIUM"
            confidence = "MEDIUM"
        elif not lane_conf_peak:
            should_warn = 0
            reason = "LANE_UNCERTAIN"
            warning_strength = "NONE"
            confidence = "LOW"
        elif defensive:
            should_warn = 0
            reason = "DEFENSIVE_CONTEXT"
            warning_strength = "NONE"
            confidence = "LOW"
        else:
            should_warn = 0
            reason = "ATTENTIVE_INTENTIONAL"
            warning_strength = "NONE"
            confidence = "LOW"

        if crossed:
            if obd_state == "hard_brake":
                should_warn = 1
                reason = "EMERGENCY_DECEL"
                warning_strength = "HIGH"
                confidence = "MEDIUM"
            elif obd_state == "no_hard_brake" and should_warn == 0 and reason != "LANE_UNCERTAIN":
                reason = "DEFENSIVE_CONTEXT"
                warning_strength = "NONE"
                confidence = "LOW"

        window_bio = fused[(fused["t_seconds"] >= peak_t - 5.0) & (fused["t_seconds"] <= peak_t + 1.0)]
        polar_hr_z = window_bio.get(polar_hr_z_col)
        polar_rmssd_z = window_bio.get(polar_rmssd_z_col)
        polar_rel = window_bio.get(polar_rel_col)
        muse_beta_alpha_z = window_bio.get(muse_beta_alpha_z_col)
        muse_rel = window_bio.get(muse_rel_col)

        hr_med = _median_reliable(polar_hr_z, polar_rel)
        rmssd_med = _median_reliable(polar_rmssd_z, polar_rel)
        eeg_med = _median_reliable(muse_beta_alpha_z, muse_rel)

        reliable_ecg = hr_med is not None or rmssd_med is not None
        reliable_eeg = eeg_med is not None

        stress_ecg = False
        if hr_med is not None and hr_med > cfg["bio_hr_z_thr"]:
            stress_ecg = True
        if rmssd_med is not None and rmssd_med < cfg["bio_rmssd_z_thr"]:
            stress_ecg = True

        arousal_eeg = eeg_med is not None and eeg_med > cfg["bio_eeg_z_thr"]

        if obd_state == "low_speed":
            reason += "+OBD_LOW_SPEED"
        elif obd_state == "hard_brake":
            reason += "+OBD_HARD_BRAKE"
        elif obd_state == "no_hard_brake":
            reason += "+OBD_NO_HARD_BRAKE"

        if should_warn == 1 and stress_ecg:
            reason += "+STRESS_CONFIRM"
            warning_strength = "HIGH"
            confidence = "HIGH"
        elif should_warn == 0 and reliable_ecg and reliable_eeg and not stress_ecg and not arousal_eeg:
            reason += "+CALM_CONFIRM"
            confidence = "HIGH"
        elif should_warn == 0 and arousal_eeg:
            reason += "+HIGH_AROUSAL"
            warning_strength = "SOFT"
            confidence = "MEDIUM"
        elif not reliable_ecg and not reliable_eeg:
            reason += "+BIO_UNRELIABLE"
            confidence = "MEDIUM"

        events.append({
            "event_id": idx,
            "start_t": float(segment["t_seconds"].min()),
            "end_t": float(segment["t_seconds"].max()),
            "duration_s": float(segment["t_seconds"].max() - segment["t_seconds"].min()),
            "peak_t": peak_t,
            "min_tlc_s": min_tlc,
            "min_dist_to_edge_m": min_dist,
            "obd_speed_kmh_med": speed_med,
            "obd_decel_min_mps2": decel_min,
            "should_warn": int(should_warn),
            "reason_code": reason,
            "warning_strength": warning_strength,
            "confidence": confidence,
        })

    return pd.DataFrame(events)

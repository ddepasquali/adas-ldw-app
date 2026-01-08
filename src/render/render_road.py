from __future__ import annotations

from datetime import timedelta
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..utils.utils_video import open_video, frame_sampler, make_video_writer
from ..video.video_lane import detect_lane, DEFAULT_PARAMS, load_ufld_model, load_yolop_model, compute_color_mask
from ..video.video_scene import load_yolo, detect_objects, VEHICLE_CLASSES, VULN_CLASSES
from ..core.overlay_style import resolve_overlay_style


def _interp_lane_x(pts, y_eval: float) -> Optional[float]:
    if pts is None:
        return None
    if isinstance(pts, tuple) and len(pts) == 2 and all(isinstance(p, tuple) for p in pts):
        (x1, y1), (x2, y2) = pts
        if y2 == y1:
            return None
        t = (y_eval - y1) / (y2 - y1)
        return float(x1 + t * (x2 - x1))
    if isinstance(pts, list):
        if not pts:
            return None
        if len(pts) == 1:
            return float(pts[0][0])
        pts_sorted = sorted(pts, key=lambda p: p[1])
        ys = np.array([p[1] for p in pts_sorted], dtype=float)
        xs = np.array([p[0] for p in pts_sorted], dtype=float)
        if y_eval <= ys[0]:
            return float(xs[0])
        if y_eval >= ys[-1]:
            return float(xs[-1])
        return float(np.interp(y_eval, ys, xs))
    return None


def _vehicle_center_px(frame_w: float, lane_width_px: float, lane_width_m: float, camera_shift_m: float) -> float:
    shift_px = 0.0
    if lane_width_m and lane_width_m > 0:
        shift_px = (camera_shift_m / lane_width_m) * lane_width_px
    return frame_w / 2.0 + shift_px


def _style_color(style: Dict[str, object], key: str, default):
    value = style.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return default
    return default


def _style_pos(style: Dict[str, object], key: str, default):
    value = style.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _lane_color_thresholds(lane_cfg: Dict[str, float]):
    red_ratio = float(lane_cfg.get("lane_color_red_ratio", 0.0))
    yellow_ratio = float(lane_cfg.get("lane_color_yellow_ratio", 0.20))
    red_m = float(lane_cfg.get("lane_color_red_m", 0.0))
    yellow_m = float(lane_cfg.get("lane_color_yellow_m", 0.15))
    if red_ratio > yellow_ratio:
        red_ratio, yellow_ratio = yellow_ratio, red_ratio
    if red_m > yellow_m:
        red_m, yellow_m = yellow_m, red_m
    return red_ratio, yellow_ratio, red_m, yellow_m


def _lane_color(
    dist_ratio: Optional[float],
    dist_to_edge_m: Optional[float],
    lane_cfg: Dict[str, float],
    style: Dict[str, object],
):
    red_ratio, yellow_ratio, red_m, yellow_m = _lane_color_thresholds(lane_cfg)
    red_color = _style_color(style, "lane_color_red", (0, 0, 255))
    yellow_color = _style_color(style, "lane_color_yellow", (0, 255, 255))
    green_color = _style_color(style, "lane_color_green", (0, 255, 0))
    if dist_ratio is not None and np.isfinite(dist_ratio):
        if dist_ratio <= red_ratio:
            return red_color
        if dist_ratio <= yellow_ratio:
            return yellow_color
        return green_color
    dist = None
    if dist_to_edge_m is not None:
        try:
            dist = float(dist_to_edge_m)
        except (TypeError, ValueError):
            dist = None
    if dist is None or not np.isfinite(dist):
        return green_color
    if dist <= red_m:
        return red_color
    if dist <= yellow_m:
        return yellow_color
    return green_color


def _box_color(label: str, style: Dict[str, object]):
    if label in VULN_CLASSES:
        return _style_color(style, "box_color_vulnerable", (0, 255, 255))
    if label in VEHICLE_CLASSES:
        return _style_color(style, "box_color_vehicle", (0, 0, 255))
    return _style_color(style, "box_color_other", (255, 0, 0))


def _lane_ratio_from_edges(
    left_x: Optional[float],
    right_x: Optional[float],
    frame_w: float,
    lane_width_m: float,
    camera_shift_m: float,
) -> Optional[float]:
    if left_x is None or right_x is None:
        return None
    if right_x <= left_x:
        return None
    lane_width_px = right_x - left_x
    if lane_width_px <= 1:
        return None

    vehicle_center_px = _vehicle_center_px(frame_w, lane_width_px, lane_width_m, camera_shift_m)

    dist_to_edge_px = min(vehicle_center_px - left_x, right_x - vehicle_center_px)
    lane_half_px = lane_width_px / 2.0
    if lane_half_px <= 0:
        return None
    return dist_to_edge_px / lane_half_px


def _lane_edges_from_mask(
    mask: np.ndarray,
    y_eval: int,
    frame_w: float,
    band_px: int,
) -> Tuple[Optional[float], Optional[float]]:
    h = mask.shape[0]
    band_px = max(1, int(band_px))
    y_min = max(0, y_eval - band_px)
    y_max = min(h, y_eval + band_px + 1)
    band = mask[y_min:y_max, :]
    _, xs = np.where(band > 0)
    if xs.size < 2:
        return None, None
    center_x = frame_w / 2.0
    left_candidates = xs[xs < center_x]
    right_candidates = xs[xs > center_x]
    if left_candidates.size == 0 or right_candidates.size == 0:
        return None, None
    left_x = float(left_candidates.max())
    right_x = float(right_candidates.min())
    return left_x, right_x


def _lane_line_centers_from_mask(
    mask: np.ndarray,
    y_eval: int,
    frame_w: float,
    band_px: int,
    cluster_gap_px: int,
):
    h = mask.shape[0]
    band_px = max(1, int(band_px))
    cluster_gap_px = max(1, int(cluster_gap_px))
    y_min = max(0, y_eval - band_px)
    y_max = min(h, y_eval + band_px + 1)
    band = mask[y_min:y_max, :]
    _, xs = np.where(band > 0)
    if xs.size == 0:
        return []
    xs_sorted = np.unique(xs)
    clusters = []
    start = xs_sorted[0]
    prev = xs_sorted[0]
    for x in xs_sorted[1:]:
        if x - prev > cluster_gap_px:
            clusters.append((start, prev))
            start = x
        prev = x
    clusters.append((start, prev))
    centers = [0.5 * (a + b) for a, b in clusters]
    centers = [c for c in centers if 0 <= c < frame_w]
    return sorted(centers)


def _lane_ratio_from_mask(
    mask: np.ndarray,
    y_eval: int,
    frame_w: float,
    lane_width_m: float,
    camera_shift_m: float,
    band_px: int,
) -> Optional[float]:
    left_x, right_x = _lane_edges_from_mask(mask, y_eval, frame_w, band_px)
    return _lane_ratio_from_edges(left_x, right_x, frame_w, lane_width_m, camera_shift_m)


def _lane_eval_info(
    debug_lines: Dict[str, object],
    lane_cfg: Dict[str, float],
    frame_shape,
    lane_width_m: float,
    camera_shift_m: float,
) -> Optional[Dict[str, object]]:
    if not debug_lines:
        return None
    h, w = frame_shape[:2]
    y_eval = int(h * lane_cfg["y_eval_ratio"])
    band_px = int(lane_cfg.get("lane_color_eval_band_px", 6))
    band_px = max(1, band_px)
    cluster_gap_px = int(lane_cfg.get("lane_line_cluster_gap_px", 12))

    info: Dict[str, object] = {
        "ratio": None,
        "left_x": None,
        "right_x": None,
        "center_x": None,
        "y_eval": int(y_eval),
        "band_px": int(band_px),
        "line_count": 0,
    }

    mask = debug_lines.get("lane_mask")
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        centers = _lane_line_centers_from_mask(mask, y_eval, w, band_px, cluster_gap_px)
        info["line_count"] = len(centers)
        left_x, right_x = _lane_edges_from_mask(mask, y_eval, w, band_px)
        info["left_x"] = left_x
        info["right_x"] = right_x
        ratio = _lane_ratio_from_edges(left_x, right_x, w, lane_width_m, camera_shift_m)
        if ratio is not None:
            info["ratio"] = ratio
            info["center_x"] = _vehicle_center_px(w, right_x - left_x, lane_width_m, camera_shift_m)
            info["source"] = "mask"
            return info

    left_x = _interp_lane_x(debug_lines.get("left"), y_eval)
    right_x = _interp_lane_x(debug_lines.get("right"), y_eval)
    info["left_x"] = left_x
    info["right_x"] = right_x
    ratio = _lane_ratio_from_edges(left_x, right_x, w, lane_width_m, camera_shift_m)
    if ratio is not None:
        info["ratio"] = ratio
        info["center_x"] = _vehicle_center_px(w, right_x - left_x, lane_width_m, camera_shift_m)
        info["source"] = "lines"
        return info

    info["source"] = "none"
    return info


def _lane_points_from_debug(debug_lines: Dict[str, object], side: str):
    raw_key = f"{side}_points"
    pts = debug_lines.get(raw_key)
    if isinstance(pts, list) and pts:
        return pts

    pts = debug_lines.get(side)
    if isinstance(pts, list) and pts:
        return pts

    if isinstance(pts, tuple) and len(pts) == 2 and all(isinstance(p, tuple) for p in pts):
        (x1, y1), (x2, y2) = pts
        steps = int(max(abs(y2 - y1), abs(x2 - x1)) // 6) + 2
        xs = np.linspace(x1, x2, steps)
        ys = np.linspace(y1, y2, steps)
        return list(zip(xs, ys))

    return []


def _lane_points_from_mask(
    mask: np.ndarray,
    roi_top: int,
    roi_bottom: int,
):
    h, w = mask.shape[:2]
    y_start = max(0, min(roi_top, h - 1))
    y_end = max(y_start + 1, min(roi_bottom, h))
    step_y = max(2, int((y_end - y_start) / 120) or 2)
    center_x = w / 2.0

    left_pts = []
    right_pts = []
    for y in range(y_start, y_end, step_y):
        row = mask[y]
        xs = np.where(row > 0)[0]
        if xs.size < 2:
            continue
        left = xs[xs < center_x]
        right = xs[xs > center_x]
        if left.size > 0:
            left_pts.append((float(left.max()), float(y)))
        if right.size > 0:
            right_pts.append((float(right.min()), float(y)))

    return left_pts, right_pts


def _draw_lane_dots(
    frame,
    pts,
    color,
    connect_thresh: Optional[float],
    dot_radius: int,
    extra_radius: int,
    extra_step: int,
    line_thickness: int,
):
    if not pts:
        return
    h, w = frame.shape[:2]
    pts_sorted = [(float(x), float(y)) for x, y in pts if 0 <= x < w and 0 <= y < h]
    if not pts_sorted:
        return
    pts_sorted.sort(key=lambda p: p[1])

    if connect_thresh is None:
        if len(pts_sorted) >= 2:
            dists = [
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(pts_sorted, pts_sorted[1:])
            ]
            if dists:
                median = float(np.median(dists))
                connect_thresh = min(25.0, max(10.0, median * 3.0))
            else:
                connect_thresh = 0.0
        else:
            connect_thresh = 0.0

    for x, y in pts_sorted:
        cv2.circle(frame, (int(x), int(y)), dot_radius, color, -1)

    for p1, p2 in zip(pts_sorted, pts_sorted[1:]):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist <= connect_thresh and dist > 0:
            cv2.line(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                color,
                line_thickness,
            )
            if extra_step > 0 and dist > extra_step:
                steps = int(dist // extra_step)
                if steps >= 2:
                    for i in range(1, steps):
                        t = i / steps
                        x = p1[0] + dx * t
                        y = p1[1] + dy * t
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (int(x), int(y)), extra_radius, color, -1)


def _force_windows_from_cfg(lane_cfg: Dict[str, float]):
    windows = []
    raw = lane_cfg.get("lane_force_windows_s")
    if isinstance(raw, (list, tuple)) and raw:
        for item in raw:
            start = None
            end = None
            pad = None
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                pad = item.get("pad")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = item[0], item[1]
                if len(item) >= 3:
                    pad = item[2]
            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue
            if start > end:
                start, end = end, start
            windows.append((start, end, pad))
        if windows:
            return windows

    start = lane_cfg.get("lane_force_red_start_s")
    end = lane_cfg.get("lane_force_red_end_s")
    if start is None or end is None:
        return []
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return []
    if start > end:
        start, end = end, start
    return [(start, end, None)]


def _forced_lane_color(t_video_s: float, lane_cfg: Dict[str, float], style: Dict[str, object]):
    windows = _force_windows_from_cfg(lane_cfg)
    if not windows:
        return None, None
    default_pad = lane_cfg.get("lane_force_yellow_pad_s", 1.0)
    try:
        default_pad = float(default_pad)
    except (TypeError, ValueError):
        default_pad = 1.0

    yellow = False
    for start, end, pad in windows:
        use_pad = default_pad
        if pad is not None:
            try:
                use_pad = float(pad)
            except (TypeError, ValueError):
                use_pad = default_pad
        if start <= t_video_s <= end:
            return _style_color(style, "lane_color_red", (0, 0, 255)), "RED"
        if (start - use_pad) <= t_video_s < start or end < t_video_s <= (end + use_pad):
            yellow = True
    if yellow:
        return _style_color(style, "lane_color_yellow", (0, 255, 255)), "YELLOW"
    return None, None


def _calibrate_camera_shift(
    left_x: Optional[float],
    right_x: Optional[float],
    frame_w: float,
    lane_width_m: float,
    target_side: str,
    target_edge_m: float,
) -> Optional[float]:
    if left_x is None or right_x is None:
        return None
    lane_width_px = right_x - left_x
    if lane_width_px <= 1 or lane_width_m <= 0:
        return None
    edge_px = (target_edge_m / lane_width_m) * lane_width_px
    if target_side == "left":
        target_center_px = left_x + edge_px
    elif target_side == "right":
        target_center_px = right_x - edge_px
    elif target_side == "center":
        target_center_px = (left_x + right_x) / 2.0 + edge_px
    else:
        return None
    return (target_center_px - frame_w / 2.0) * lane_width_m / lane_width_px


def process_road_video(
    path: str,
    video_start_dt,
    global_start_dt,
    out_lane_csv_path: str,
    out_scene_csv_path: str,
    out_video_path: Optional[str],
    lane_width_m: float,
    camera_shift_m: float,
    target_hz: float,
    max_width: Optional[int],
    yolo_model_path: str,
    yolo_conf: float,
    lane_model_path: Optional[str],
    lane_params: Optional[Dict[str, float]],
    lane_mode: str,
    draw_lane_roi: bool,
    draw_lane_mask: bool,
    draw_lane_edges: bool,
    draw_lane_color_mask: bool,
    draw_lane_hough: bool,
    draw_lane_points: bool,
    draw_lane_dots: bool,
    road_style: Optional[Dict[str, object]],
    log,
):
    cap_info = open_video(path)
    if cap_info is None:
        log(f"Road video not found or unreadable: {path}")
        return pd.DataFrame(), pd.DataFrame()
    cap, fps, width, height, _ = cap_info

    model = load_yolo(yolo_model_path, log) if yolo_model_path else None
    writer = None

    lane_rows = []
    scene_rows = []
    prev_offset = None
    prev_t = None

    lane_cfg = dict(DEFAULT_PARAMS)
    if lane_params:
        lane_cfg.update(lane_params)

    style = road_style or resolve_overlay_style({}).get("road", {})
    metrics_pos = _style_pos(style, "metrics_text_pos", (10, 60))
    metrics_scale = float(style.get("metrics_text_scale", 0.6))
    metrics_thickness = int(style.get("metrics_text_thickness", 2))
    metrics_color = _style_color(style, "metrics_text_color", (255, 255, 255))
    lane_conf_pos = _style_pos(style, "lane_conf_text_pos", (10, 30))
    lane_conf_scale = float(style.get("lane_conf_text_scale", 0.6))
    lane_conf_thickness = int(style.get("lane_conf_text_thickness", 2))
    lane_conf_color = _style_color(style, "lane_conf_text_color", (100, 100, 255))
    eval_pos = _style_pos(style, "eval_text_pos", (10, 85))
    eval_gap = int(style.get("eval_text_line_gap", 25))
    eval_scale = float(style.get("eval_text_scale", 0.5))
    eval_thickness = int(style.get("eval_text_thickness", 1))
    eval_text_color = _style_color(style, "eval_text_color", (255, 255, 255))
    lane_eval_color = _style_color(style, "lane_eval_color", (255, 255, 255))
    lane_eval_thickness = int(style.get("lane_eval_thickness", 1))
    lane_eval_band_color = _style_color(style, "lane_eval_band_color", (0, 255, 255))
    lane_eval_band_alpha = float(style.get("lane_eval_band_alpha", 0.12))
    lane_edge_marker_color = _style_color(style, "lane_edge_marker_color", (255, 255, 255))
    lane_edge_marker_thickness = int(style.get("lane_edge_marker_thickness", 2))
    lane_edge_marker_radius = int(style.get("lane_edge_marker_radius", 3))
    lane_center_color = _style_color(style, "lane_center_color", (255, 0, 255))
    lane_center_thickness = int(style.get("lane_center_thickness", 1))
    lane_center_radius = int(style.get("lane_center_radius", 3))
    dot_radius = int(style.get("lane_dot_radius", 3))
    extra_dot_radius = int(style.get("lane_extra_dot_radius", 2))
    lane_line_thickness = int(style.get("lane_line_thickness", 1))
    extra_step_detail = int(style.get("lane_extra_step_detail", 5))
    extra_step_default = int(style.get("lane_extra_step_default", 7))
    box_thickness = int(style.get("box_thickness", 2))
    box_label_scale = float(style.get("box_label_scale", 0.5))
    box_label_thickness = int(style.get("box_label_thickness", 1))
    box_label_text_color = _style_color(style, "box_label_text_color", (255, 255, 255))
    box_label_pad_x = int(style.get("box_label_pad_x", 2))
    box_label_pad_y = int(style.get("box_label_pad_y", 4))
    box_label_y_offset = int(style.get("box_label_y_offset", 6))
    box_label_text_offset = _style_pos(style, "box_label_text_offset", (1, 2))
    roi_color = _style_color(style, "roi_color", (0, 255, 255))
    roi_thickness = int(style.get("roi_thickness", 2))
    roi_text_color = _style_color(style, "roi_text_color", (0, 255, 255))
    roi_text_scale = float(style.get("roi_text_scale", 0.6))
    roi_text_thickness = int(style.get("roi_text_thickness", 2))
    roi_text_offset_x = int(style.get("roi_text_offset_x", 10))
    roi_text_offset_y = int(style.get("roi_text_offset_y", -10))
    mask_overlay_color = _style_color(style, "mask_overlay_color", (0, 255, 255))
    mask_overlay_alpha = float(style.get("mask_overlay_alpha", 0.35))
    edges_overlay_color = _style_color(style, "edges_overlay_color", (255, 0, 255))
    edges_overlay_alpha = float(style.get("edges_overlay_alpha", 0.35))
    hough_color = _style_color(style, "hough_color", (0, 128, 255))
    hough_thickness = int(style.get("hough_thickness", 2))

    calib_start = lane_cfg.get("lane_center_calib_start_s")
    calib_end = lane_cfg.get("lane_center_calib_end_s")
    calib_side = str(lane_cfg.get("lane_center_calib_side", "left")).lower()
    calib_edge_m = lane_cfg.get("lane_center_calib_edge_m", 0.0)
    calib_min_lines = int(lane_cfg.get("lane_center_calib_min_lines", 3))
    calib_apply_once = bool(lane_cfg.get("lane_center_calib_apply_once", True))
    calib_done = False

    lane_model = None
    if lane_mode == "ufld":
        lane_model = load_ufld_model(lane_model_path, log)
        if lane_model is None:
            log("UFLD model unavailable; falling back to poly mode")
            lane_mode = "poly"
    if lane_mode == "yolop":
        lane_model = load_yolop_model(lane_model_path, log)
        if lane_model is None:
            log("YOLOP model unavailable; falling back to poly mode")
            lane_mode = "poly"

    for t_video_s, _, frame in frame_sampler(cap, target_hz, max_width=max_width):
        metrics, debug_lines = detect_lane(
            frame,
            lane_width_m,
            camera_shift_m,
            params=lane_params,
            prev_offset_m=prev_offset,
            prev_t=prev_t,
            t_s=t_video_s,
            mode=lane_mode,
            model=lane_model,
        )

        if calib_start is not None and calib_end is not None:
            if not calib_apply_once or not calib_done:
                try:
                    start_s = float(calib_start)
                    end_s = float(calib_end)
                except (TypeError, ValueError):
                    start_s = None
                    end_s = None
                if start_s is not None and end_s is not None:
                    if start_s > end_s:
                        start_s, end_s = end_s, start_s
                    if start_s <= t_video_s <= end_s:
                        mask = debug_lines.get("lane_mask")
                        if mask is not None:
                            h, w = frame.shape[:2]
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            y_eval = int(h * lane_cfg["y_eval_ratio"])
                            band_px = int(lane_cfg.get("lane_color_eval_band_px", 6))
                            band_px = max(1, band_px)
                            cluster_gap_px = int(lane_cfg.get("lane_line_cluster_gap_px", 12))
                            centers = _lane_line_centers_from_mask(mask, y_eval, w, band_px, cluster_gap_px)
                            if len(centers) >= calib_min_lines:
                                left_x, right_x = _lane_edges_from_mask(mask, y_eval, w, band_px)
                                try:
                                    edge_m = float(calib_edge_m)
                                except (TypeError, ValueError):
                                    edge_m = 0.0
                                new_shift = _calibrate_camera_shift(
                                    left_x,
                                    right_x,
                                    w,
                                    lane_width_m,
                                    calib_side,
                                    edge_m,
                                )
                                if new_shift is not None:
                                    camera_shift_m = float(new_shift)
                                    calib_done = True
                                    prev_offset = None
                                    prev_t = None
                                    log(f"camera_shift_m calibrated to {camera_shift_m:.3f} at t={t_video_s:.2f}s")

        dt = video_start_dt + timedelta(seconds=t_video_s)
        t_seconds = (dt - global_start_dt).total_seconds()

        lane_rows.append({"t_seconds": t_seconds, **metrics})

        if metrics.get("lane_conf") == 1:
            prev_offset = metrics.get("lane_offset_m")
            prev_t = t_video_s

        scene = detect_objects(model, frame, conf=yolo_conf)
        scene_rows.append({
            "t_seconds": t_seconds,
            "obj_count": scene["obj_count"],
            "vehicles_count": scene["vehicles_count"],
            "vuln_count": scene["vuln_count"],
        })

        if out_video_path:
            if writer is None:
                h, w = frame.shape[:2]
                writer = make_video_writer(out_video_path, target_hz, (w, h))

            eval_info = _lane_eval_info(
                debug_lines,
                lane_cfg,
                frame.shape,
                lane_width_m,
                camera_shift_m,
            )
            lane_ratio = eval_info.get("ratio") if eval_info else None
            lane_color = _lane_color(lane_ratio, metrics.get("dist_to_edge_m"), lane_cfg, style)
            forced_color, forced_label = _forced_lane_color(t_video_s, lane_cfg, style)
            if forced_color is not None:
                lane_color = forced_color
            lane_has_estimate = (
                lane_ratio is not None
                or metrics.get("lane_conf") == 1
                or debug_lines.get("lane_mask") is not None
                or forced_color is not None
            )

            show_eval = lane_cfg.get("road_overlay_lane_eval")
            if show_eval is None:
                show_eval = lane_cfg.get("lane_color_overlay", False)
            if show_eval and eval_info is not None:
                h, w = frame.shape[:2]
                y_eval = int(eval_info.get("y_eval") or 0)
                band_px = int(eval_info.get("band_px") or 0)
                if band_px > 0:
                    overlay = frame.copy()
                    y1 = max(0, y_eval - band_px)
                    y2 = min(h - 1, y_eval + band_px)
                    cv2.rectangle(overlay, (0, y1), (w - 1, y2), lane_eval_band_color, -1)
                    frame[:] = cv2.addWeighted(overlay, lane_eval_band_alpha, frame, 1 - lane_eval_band_alpha, 0)
                cv2.line(frame, (0, y_eval), (w - 1, y_eval), lane_eval_color, lane_eval_thickness)

                left_x = eval_info.get("left_x")
                right_x = eval_info.get("right_x")
                center_x = eval_info.get("center_x")
                if left_x is not None:
                    lx = int(round(left_x))
                    cv2.line(
                        frame,
                        (lx, y_eval - 6),
                        (lx, y_eval + 6),
                        lane_edge_marker_color,
                        lane_edge_marker_thickness,
                    )
                    cv2.circle(frame, (lx, y_eval), lane_edge_marker_radius, lane_edge_marker_color, -1)
                if right_x is not None:
                    rx = int(round(right_x))
                    cv2.line(
                        frame,
                        (rx, y_eval - 6),
                        (rx, y_eval + 6),
                        lane_edge_marker_color,
                        lane_edge_marker_thickness,
                    )
                    cv2.circle(frame, (rx, y_eval), lane_edge_marker_radius, lane_edge_marker_color, -1)
                if center_x is not None:
                    cx = int(round(center_x))
                    cv2.circle(frame, (cx, y_eval), lane_center_radius, lane_center_color, -1)
                    if left_x is not None and right_x is not None:
                        edge_x = left_x if (center_x - left_x) < (right_x - center_x) else right_x
                        cv2.line(
                            frame,
                            (cx, y_eval),
                            (int(round(edge_x)), y_eval),
                            lane_center_color,
                            lane_center_thickness,
                        )

                red_ratio, yellow_ratio, red_m, yellow_m = _lane_color_thresholds(lane_cfg)
                ratio_val = float("nan") if lane_ratio is None else float(lane_ratio)
                if lane_ratio is not None and np.isfinite(lane_ratio):
                    edge_val = float(lane_ratio) * float(lane_width_m) / 2.0
                else:
                    edge_val = metrics.get("dist_to_edge_m", float("nan"))
                    try:
                        edge_val = float(edge_val)
                    except (TypeError, ValueError):
                        edge_val = float("nan")
                edge_left_m = float("nan")
                edge_right_m = float("nan")
                if left_x is not None and right_x is not None and center_x is not None:
                    lane_width_px = right_x - left_x
                    if lane_width_px > 1 and lane_width_m > 0:
                        px_to_m = lane_width_m / lane_width_px
                        edge_left_m = (center_x - left_x) * px_to_m
                        edge_right_m = (right_x - center_x) * px_to_m
                source = eval_info.get("source", "none")
                line_count = eval_info.get("line_count", 0)
                line1 = (
                    f"ratio={ratio_val:.2f} src={source} "
                    f"R<={red_ratio:.2f} Y<={yellow_ratio:.2f}"
                )
                line2 = (
                    f"edgeL={edge_left_m:+.2f}m edgeR={edge_right_m:+.2f}m "
                    f"min={edge_val:.2f}m R<={red_m:.2f} Y<={yellow_m:.2f}"
                )
                line3 = (
                    f"y={y_eval}px b=+-{band_px}px lines={line_count} shift={camera_shift_m:+.2f}m"
                )
                if forced_label:
                    line3 = f"{line3} force={forced_label}"
                cv2.putText(
                    frame,
                    line1,
                    eval_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    eval_scale,
                    eval_text_color,
                    eval_thickness,
                )
                cv2.putText(
                    frame,
                    line2,
                    (eval_pos[0], eval_pos[1] + eval_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    eval_scale,
                    eval_text_color,
                    eval_thickness,
                )
                cv2.putText(
                    frame,
                    line3,
                    (eval_pos[0], eval_pos[1] + 2 * eval_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    eval_scale,
                    eval_text_color,
                    eval_thickness,
                )

            if draw_lane_roi:
                h, w = frame.shape[:2]
                roi_top = int(h * lane_cfg["roi_top_ratio"])
                roi_bottom = int(h * lane_cfg["roi_bottom_ratio"])
                cv2.rectangle(frame, (0, roi_top), (w, roi_bottom), roi_color, roi_thickness)
                cv2.putText(
                    frame,
                    "lane ROI",
                    (roi_text_offset_x, max(roi_top + roi_text_offset_y, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    roi_text_scale,
                    roi_text_color,
                    roi_text_thickness,
                )

            if draw_lane_color_mask or draw_lane_edges or draw_lane_hough:
                h, w = frame.shape[:2]
                roi_top = int(h * lane_cfg["roi_top_ratio"])
                roi_bottom = int(h * lane_cfg["roi_bottom_ratio"])

            if draw_lane_color_mask:
                mask = compute_color_mask(frame, lane_cfg)
                mask[:roi_top, :] = 0
                mask[roi_bottom:, :] = 0
                overlay = frame.copy()
                overlay[mask > 0] = mask_overlay_color
                frame[:] = cv2.addWeighted(overlay, mask_overlay_alpha, frame, 1 - mask_overlay_alpha, 0)

            if draw_lane_edges or draw_lane_hough:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, lane_cfg["canny_low"], lane_cfg["canny_high"])
                edges[:roi_top, :] = 0
                edges[roi_bottom:, :] = 0

                if draw_lane_edges:
                    overlay = frame.copy()
                    overlay[edges > 0] = edges_overlay_color
                    frame[:] = cv2.addWeighted(overlay, edges_overlay_alpha, frame, 1 - edges_overlay_alpha, 0)

                if draw_lane_hough:
                    lines = cv2.HoughLinesP(
                        edges,
                        lane_cfg["hough_rho"],
                        lane_cfg["hough_theta"],
                        threshold=lane_cfg["hough_threshold"],
                        minLineLength=lane_cfg["hough_min_len"],
                        maxLineGap=lane_cfg["hough_max_gap"],
                    )
                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(frame, (x1, y1), (x2, y2), hough_color, hough_thickness)

            if draw_lane_mask and "lane_mask" in debug_lines:
                mask = debug_lines.get("lane_mask")
                if mask is not None:
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    overlay = frame.copy()
                    overlay[mask > 0] = lane_color
                    frame[:] = cv2.addWeighted(overlay, mask_overlay_alpha, frame, 1 - mask_overlay_alpha, 0)

            if lane_has_estimate and draw_lane_dots:
                color = lane_color
                left_pts = []
                right_pts = []
                mask = debug_lines.get("lane_mask")
                if mask is not None:
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    h, w = frame.shape[:2]
                    roi_top = int(h * lane_cfg["roi_top_ratio"])
                    roi_bottom = int(h * lane_cfg["roi_bottom_ratio"])
                    left_pts, right_pts = _lane_points_from_mask(mask, roi_top, roi_bottom)

                if not left_pts and not right_pts:
                    left_pts = _lane_points_from_debug(debug_lines, "left")
                    right_pts = _lane_points_from_debug(debug_lines, "right")

                extra_step = extra_step_detail if draw_lane_points else extra_step_default
                _draw_lane_dots(
                    frame,
                    left_pts,
                    color,
                    connect_thresh=None,
                    dot_radius=dot_radius,
                    extra_radius=extra_dot_radius,
                    extra_step=extra_step,
                    line_thickness=lane_line_thickness,
                )
                _draw_lane_dots(
                    frame,
                    right_pts,
                    color,
                    connect_thresh=None,
                    dot_radius=dot_radius,
                    extra_radius=extra_dot_radius,
                    extra_step=extra_step,
                    line_thickness=lane_line_thickness,
                )
            else:
                cv2.putText(
                    frame,
                    "lane_conf=0",
                    lane_conf_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    lane_conf_scale,
                    lane_conf_color,
                    lane_conf_thickness,
                )

            if lane_cfg.get("road_overlay_boxes", True):
                for box in scene["boxes"]:
                    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    x2 = max(0, min(x2, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    y2 = max(0, min(y2, frame.shape[0] - 1))
                    label = box["label"]
                    conf = box["conf"]
                    color = _box_color(label, style)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    label_text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, box_label_scale, box_label_thickness
                    )
                    text_x = x1
                    text_y = max(y1 - box_label_y_offset, th + box_label_y_offset)
                    cv2.rectangle(
                        frame,
                        (text_x, text_y - th - box_label_pad_y),
                        (text_x + tw + box_label_pad_x, text_y),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (text_x + box_label_text_offset[0], text_y - box_label_text_offset[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        box_label_scale,
                        box_label_text_color,
                        box_label_thickness,
                    )

            if lane_cfg.get("road_overlay_metrics", True):
                text = (
                    f"tlc={metrics.get('tlc_s', float('nan')):.2f} "
                    f"edge={metrics.get('dist_to_edge_m', float('nan')):.2f} "
                    f"veh={scene['vehicles_count']} vuln={scene['vuln_count']}"
                )
                cv2.putText(
                    frame,
                    text,
                    metrics_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    metrics_scale,
                    metrics_color,
                    metrics_thickness,
                )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    lane_df = pd.DataFrame(lane_rows)
    scene_df = pd.DataFrame(scene_rows)
    if out_lane_csv_path:
        lane_df.to_csv(out_lane_csv_path, index=False)
    if out_scene_csv_path:
        scene_df.to_csv(out_scene_csv_path, index=False)
    return lane_df, scene_df

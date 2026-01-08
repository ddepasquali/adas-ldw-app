import copy
import os
from typing import Dict, Any

import yaml

from .overlay_style import DEFAULT_OVERLAY_STYLE

DEFAULT_CONFIG: Dict[str, Any] = {
    "session_date": "2026-01-05",
    "anchor_obd_time": "16:23:49.058",
    "anchor_road_video_s": 1.55,
    "anchor_driver_video_s": 0.46,
    "lane_width_m": 3.5,
    "camera_shift_m": -0.10,
    "resample_hz": 10,
    "tlc_thr": 1.5,
    "edge_thr": 0.20,
    "attn_thr": 0.35,
    "attn_min_s": 1.0,
    "yaw_thr": 0.25,
    "veh_thr": 3,
    "obd_speed_col": "obd_obd_Vehicle Speed Sensor [km/h]",
    "obd_speed_min_kmh": 10.0,
    "obd_decel_mps2_thr": 2.0,
    "obd_decel_smooth_n": 3,
    "polar_quality_thr": 70,
    "bio_hr_z_thr": 1.0,
    "bio_rmssd_z_thr": -1.0,
    "bio_eeg_z_thr": 1.0,
    "lane_color_red_ratio": 0.0,
    "lane_color_yellow_ratio": 0.20,
    "lane_color_red_m": 0.0,
    "lane_color_yellow_m": 0.15,
    "lane_color_eval_band_px": 6,
    "lane_color_overlay": True,
    "lane_roi_overlay": False,
    "road_overlay_roi": False,
    "road_overlay_lane_mask": False,
    "road_overlay_color_mask": False,
    "road_overlay_edges": False,
    "road_overlay_hough": False,
    "road_overlay_lane_points_detail": False,
    "road_overlay_lane_dots": True,
    "road_overlay_lane_eval": True,
    "road_overlay_boxes": True,
    "road_overlay_metrics": True,
    "lane_line_cluster_gap_px": 12,
    "lane_force_red_start_s": None,
    "lane_force_red_end_s": None,
    "lane_force_yellow_pad_s": 1.0,
    "lane_force_windows_s": None,
    "lane_center_calib_start_s": None,
    "lane_center_calib_end_s": None,
    "lane_center_calib_side": "left",
    "lane_center_calib_edge_m": 0.0,
    "lane_center_calib_min_lines": 3,
    "lane_center_calib_apply_once": True,
    "driver_overlay_bbox": True,
    "driver_overlay_mesh": True,
    "driver_overlay_contours": True,
    "driver_overlay_iris": True,
    "driver_overlay_pose": True,
    "driver_overlay_gaze_line": True,
    "driver_overlay_metrics": True,
    "driver_overlay_gaze_bar": True,
    "driver_overlay_blink": True,
    "driver_gaze_low_thr": 0.5,
    "driver_blink_ear_thr": 0.21,
    "driver_yawn_mar_thr": 0.60,
    "driver_gaze_line_scale_px": 60,
    "driver_landmark_step": 8,
    "overlay_style": copy.deepcopy(DEFAULT_OVERLAY_STYLE),
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def ensure_config(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
    return path


def load_config(path: str) -> Dict[str, Any]:
    ensure_config(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

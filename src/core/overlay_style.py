from __future__ import annotations

from typing import Any, Dict
import copy


DEFAULT_OVERLAY_STYLE: Dict[str, Dict[str, Any]] = {
    "road": {
        "lane_color_red": [0, 0, 255],
        "lane_color_yellow": [0, 255, 255],
        "lane_color_green": [0, 255, 0],
        "lane_dot_radius": 3,
        "lane_extra_dot_radius": 2,
        "lane_line_thickness": 1,
        "lane_extra_step_detail": 5,
        "lane_extra_step_default": 7,
        "lane_eval_color": [255, 255, 255],
        "lane_eval_thickness": 1,
        "lane_eval_band_color": [0, 255, 255],
        "lane_eval_band_alpha": 0.12,
        "lane_edge_marker_color": [255, 255, 255],
        "lane_edge_marker_thickness": 2,
        "lane_edge_marker_radius": 3,
        "lane_center_color": [255, 0, 255],
        "lane_center_thickness": 1,
        "lane_center_radius": 3,
        "metrics_text_color": [255, 255, 255],
        "metrics_text_pos": [10, 60],
        "metrics_text_scale": 0.6,
        "metrics_text_thickness": 2,
        "lane_conf_text_color": [100, 100, 255],
        "lane_conf_text_pos": [10, 30],
        "lane_conf_text_scale": 0.6,
        "lane_conf_text_thickness": 2,
        "eval_text_color": [255, 255, 255],
        "eval_text_pos": [10, 85],
        "eval_text_line_gap": 25,
        "eval_text_scale": 0.5,
        "eval_text_thickness": 1,
        "box_color_vehicle": [0, 0, 255],
        "box_color_vulnerable": [0, 255, 255],
        "box_color_other": [255, 0, 0],
        "box_thickness": 2,
        "box_label_scale": 0.5,
        "box_label_thickness": 1,
        "box_label_text_color": [255, 255, 255],
        "box_label_pad_x": 2,
        "box_label_pad_y": 4,
        "box_label_y_offset": 6,
        "box_label_text_offset": [1, 2],
        "roi_color": [0, 255, 255],
        "roi_thickness": 2,
        "roi_text_color": [0, 255, 255],
        "roi_text_scale": 0.6,
        "roi_text_thickness": 2,
        "roi_text_offset_x": 10,
        "roi_text_offset_y": -10,
        "mask_overlay_color": [0, 255, 255],
        "mask_overlay_alpha": 0.35,
        "edges_overlay_color": [255, 0, 255],
        "edges_overlay_alpha": 0.35,
        "hough_color": [0, 128, 255],
        "hough_thickness": 2,
    },
    "driver": {
        "bbox_good_color": [0, 255, 0],
        "bbox_bad_color": [0, 0, 255],
        "bbox_thickness": 2,
        "gaze_line_color": [255, 0, 255],
        "gaze_line_thickness": 2,
        "pose_axis_x_color": [0, 0, 255],
        "pose_axis_y_color": [0, 255, 0],
        "pose_axis_z_color": [255, 0, 0],
        "pose_axis_thickness": 2,
        "pose_axis_len": 60,
        "gaze_bar_border_color": [200, 200, 200],
        "gaze_bar_border_thickness": 1,
        "gaze_bar_good_color": [0, 255, 0],
        "gaze_bar_bad_color": [0, 0, 255],
        "gaze_bar_height": 10,
        "gaze_bar_width_ratio": 0.25,
        "gaze_bar_x": 10,
        "gaze_bar_y_from_bottom": 20,
        "metrics_text_color": [255, 255, 255],
        "metrics_text_pos": [10, 30],
        "metrics_text_scale": 0.55,
        "metrics_text_thickness": 2,
        "blink_text_color": [255, 255, 255],
        "blink_text_alert_color": [0, 0, 255],
        "blink_text_pos": [10, 55],
        "blink_text_scale": 0.55,
        "blink_text_thickness": 2,
        "face_conf_text_color": [0, 0, 255],
        "face_conf_text_pos": [10, 30],
        "face_conf_text_scale": 0.6,
        "face_conf_text_thickness": 2,
        "landmark_color": [0, 255, 255],
        "landmark_radius": 1,
    },
}

_ROAD_COLOR_KEYS = [
    "lane_color_red",
    "lane_color_yellow",
    "lane_color_green",
    "lane_eval_color",
    "lane_eval_band_color",
    "lane_edge_marker_color",
    "lane_center_color",
    "metrics_text_color",
    "lane_conf_text_color",
    "eval_text_color",
    "box_color_vehicle",
    "box_color_vulnerable",
    "box_color_other",
    "box_label_text_color",
    "roi_color",
    "roi_text_color",
    "mask_overlay_color",
    "edges_overlay_color",
    "hough_color",
]

_DRIVER_COLOR_KEYS = [
    "bbox_good_color",
    "bbox_bad_color",
    "gaze_line_color",
    "pose_axis_x_color",
    "pose_axis_y_color",
    "pose_axis_z_color",
    "gaze_bar_border_color",
    "gaze_bar_good_color",
    "gaze_bar_bad_color",
    "metrics_text_color",
    "blink_text_color",
    "blink_text_alert_color",
    "face_conf_text_color",
    "landmark_color",
]

_ROAD_POS_KEYS = [
    "metrics_text_pos",
    "lane_conf_text_pos",
    "eval_text_pos",
    "box_label_text_offset",
]

_DRIVER_POS_KEYS = [
    "metrics_text_pos",
    "blink_text_pos",
    "face_conf_text_pos",
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _normalize_color(value: Any, default: Any):
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return default
    return default


def _normalize_pos(value: Any, default: Any):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _normalize_section(
    section: Dict[str, Any],
    defaults: Dict[str, Any],
    color_keys,
    pos_keys,
) -> Dict[str, Any]:
    for key in color_keys:
        section[key] = _normalize_color(section.get(key), defaults.get(key))
    for key in pos_keys:
        section[key] = _normalize_pos(section.get(key), defaults.get(key))
    return section


def resolve_overlay_style(cfg: Dict[str, Any]) -> Dict[str, Any]:
    overlay = {}
    if cfg and isinstance(cfg.get("overlay_style"), dict):
        overlay = cfg["overlay_style"]
    style = _deep_merge(DEFAULT_OVERLAY_STYLE, overlay)
    road = style.get("road", {})
    driver = style.get("driver", {})
    road = _normalize_section(road, DEFAULT_OVERLAY_STYLE["road"], _ROAD_COLOR_KEYS, _ROAD_POS_KEYS)
    driver = _normalize_section(driver, DEFAULT_OVERLAY_STYLE["driver"], _DRIVER_COLOR_KEYS, _DRIVER_POS_KEYS)
    style["road"] = road
    style["driver"] = driver
    return style

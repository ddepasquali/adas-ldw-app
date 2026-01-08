from __future__ import annotations

from datetime import timedelta
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from ..utils.utils_video import open_video, frame_sampler, make_video_writer
from ..video.video_driver import extract_driver_features
from ..core.overlay_style import resolve_overlay_style


EMA_ALPHA = 0.2


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


def _overlay_defaults(cfg: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    base = {
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
    }
    if cfg:
        for key, val in cfg.items():
            if val is not None:
                base[key] = val
    return base


def _draw_gaze_bar(frame, gaze: float, thr: float, style: Dict[str, object]):
    h, w = frame.shape[:2]
    bar_w = int(w * float(style.get("gaze_bar_width_ratio", 0.25)))
    bar_h = int(style.get("gaze_bar_height", 10))
    x0 = int(style.get("gaze_bar_x", 10))
    y0 = h - int(style.get("gaze_bar_y_from_bottom", 20))
    x1 = x0 + bar_w
    y1 = y0 + bar_h
    border_color = _style_color(style, "gaze_bar_border_color", (200, 200, 200))
    border_thickness = int(style.get("gaze_bar_border_thickness", 1))
    cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, border_thickness)
    if not np.isfinite(gaze):
        return
    fill_w = int(bar_w * max(0.0, min(1.0, gaze)))
    bad_color = _style_color(style, "gaze_bar_bad_color", (0, 0, 255))
    good_color = _style_color(style, "gaze_bar_good_color", (0, 255, 0))
    color = bad_color if gaze < thr else good_color
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y1), color, -1)


def _draw_head_pose_axes(frame, head_pose, nose, style: Dict[str, object]):
    if head_pose is None:
        return
    rvec = head_pose.get("rvec")
    tvec = head_pose.get("tvec")
    cam = head_pose.get("camera_matrix")
    if rvec is None or tvec is None or cam is None:
        return
    axis_len = float(style.get("pose_axis_len", 60.0))
    axes = np.float32([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ])
    proj, _ = cv2.projectPoints(axes, rvec, tvec, cam, np.zeros((4, 1)))
    proj = proj.reshape(-1, 2)
    nose_pt = (int(nose[0]), int(nose[1]))
    x_axis = (int(proj[0][0]), int(proj[0][1]))
    y_axis = (int(proj[1][0]), int(proj[1][1]))
    z_axis = (int(proj[2][0]), int(proj[2][1]))
    axis_thickness = int(style.get("pose_axis_thickness", 2))
    x_color = _style_color(style, "pose_axis_x_color", (0, 0, 255))
    y_color = _style_color(style, "pose_axis_y_color", (0, 255, 0))
    z_color = _style_color(style, "pose_axis_z_color", (255, 0, 0))
    cv2.line(frame, nose_pt, x_axis, x_color, axis_thickness)
    cv2.line(frame, nose_pt, y_axis, y_color, axis_thickness)
    cv2.line(frame, nose_pt, z_axis, z_color, axis_thickness)


def process_driver_video(
    path: str,
    video_start_dt,
    global_start_dt,
    out_csv_path: str,
    out_video_path: Optional[str],
    target_hz: float,
    max_width: Optional[int],
    driver_overlay: Optional[Dict[str, object]],
    driver_style: Optional[Dict[str, object]],
    log,
) -> pd.DataFrame:
    cap_info = open_video(path)
    if cap_info is None:
        log(f"Driver video not found or unreadable: {path}")
        return pd.DataFrame()
    cap, fps, width, height, _ = cap_info
    writer = None

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    overlay = _overlay_defaults(driver_overlay)
    style = driver_style or resolve_overlay_style({}).get("driver", {})
    bbox_good_color = _style_color(style, "bbox_good_color", (0, 255, 0))
    bbox_bad_color = _style_color(style, "bbox_bad_color", (0, 0, 255))
    bbox_thickness = int(style.get("bbox_thickness", 2))
    gaze_line_color = _style_color(style, "gaze_line_color", (255, 0, 255))
    gaze_line_thickness = int(style.get("gaze_line_thickness", 2))
    metrics_pos = _style_pos(style, "metrics_text_pos", (10, 30))
    metrics_scale = float(style.get("metrics_text_scale", 0.55))
    metrics_thickness = int(style.get("metrics_text_thickness", 2))
    metrics_color = _style_color(style, "metrics_text_color", (255, 255, 255))
    blink_pos = _style_pos(style, "blink_text_pos", (10, 55))
    blink_scale = float(style.get("blink_text_scale", 0.55))
    blink_thickness = int(style.get("blink_text_thickness", 2))
    blink_color = _style_color(style, "blink_text_color", (255, 255, 255))
    blink_alert_color = _style_color(style, "blink_text_alert_color", (0, 0, 255))
    face_conf_pos = _style_pos(style, "face_conf_text_pos", (10, 30))
    face_conf_scale = float(style.get("face_conf_text_scale", 0.6))
    face_conf_thickness = int(style.get("face_conf_text_thickness", 2))
    face_conf_color = _style_color(style, "face_conf_text_color", (0, 0, 255))
    landmark_color = _style_color(style, "landmark_color", (0, 255, 255))
    landmark_radius = int(style.get("landmark_radius", 1))

    data = []
    ema_gaze = None

    for t_video_s, _, frame in frame_sampler(cap, target_hz, max_width=max_width):
        metrics, debug = extract_driver_features(frame, mp_face)
        gaze = metrics.get("gaze_on_road_prob")
        if gaze is not None and not np.isnan(gaze):
            if ema_gaze is None:
                ema_gaze = gaze
            else:
                ema_gaze = EMA_ALPHA * gaze + (1 - EMA_ALPHA) * ema_gaze
            metrics["gaze_on_road_prob"] = float(ema_gaze)

        dt = video_start_dt + timedelta(seconds=t_video_s)
        t_seconds = (dt - global_start_dt).total_seconds()
        data.append({"t_seconds": t_seconds, **metrics})

        if out_video_path:
            if writer is None:
                h, w = frame.shape[:2]
                writer = make_video_writer(out_video_path, target_hz, (w, h))
            if debug is not None:
                pts = debug.get("pts")
                landmarks = debug.get("landmarks")
                bbox = debug.get("bbox")
                if overlay.get("driver_overlay_mesh") and landmarks is not None:
                    mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                if overlay.get("driver_overlay_contours") and landmarks is not None:
                    mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                    )
                if overlay.get("driver_overlay_iris") and landmarks is not None:
                    mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
                    )
                if (
                    pts is not None
                    and not overlay.get("driver_overlay_mesh")
                    and not overlay.get("driver_overlay_contours")
                ):
                    step = int(overlay.get("driver_landmark_step", 8))
                    step = max(1, step)
                    for x, y in pts[::step]:
                        cv2.circle(frame, (int(x), int(y)), landmark_radius, landmark_color, -1)

                if overlay.get("driver_overlay_bbox") and bbox is not None:
                    min_x, min_y, max_x, max_y = bbox
                    gaze_val = metrics.get("gaze_on_road_prob", float("nan"))
                    thr = float(overlay.get("driver_gaze_low_thr", 0.5))
                    box_color = bbox_good_color if np.isfinite(gaze_val) and gaze_val >= thr else bbox_bad_color
                    cv2.rectangle(
                        frame,
                        (int(min_x), int(min_y)),
                        (int(max_x), int(max_y)),
                        box_color,
                        bbox_thickness,
                    )

                if overlay.get("driver_overlay_gaze_line"):
                    iris_left = debug.get("iris_left")
                    iris_right = debug.get("iris_right")
                    eye_left = debug.get("eye_left")
                    eye_right = debug.get("eye_right")
                    scale = float(overlay.get("driver_gaze_line_scale_px", 60))
                    for iris, eye in ((iris_left, eye_left), (iris_right, eye_right)):
                        if iris is None or eye is None:
                            continue
                        vec = iris - eye
                        if np.linalg.norm(vec) < 1e-3:
                            continue
                        end = eye + vec * scale
                        cv2.line(
                            frame,
                            (int(eye[0]), int(eye[1])),
                            (int(end[0]), int(end[1])),
                            gaze_line_color,
                            gaze_line_thickness,
                        )

                if overlay.get("driver_overlay_pose"):
                    nose = pts[1] if pts is not None else None
                    if nose is not None:
                        _draw_head_pose_axes(frame, debug.get("head_pose"), nose, style)

                if overlay.get("driver_overlay_metrics"):
                    gaze_val = metrics.get("gaze_on_road_prob", float("nan"))
                    yaw = metrics.get("head_yaw_proxy", float("nan"))
                    pitch = metrics.get("head_pitch_proxy", float("nan"))
                    yaw_deg = metrics.get("head_yaw_deg", float("nan"))
                    pitch_deg = metrics.get("head_pitch_deg", float("nan"))
                    roll_deg = metrics.get("head_roll_deg", float("nan"))
                    text = (
                        f"gaze={gaze_val:.2f} yaw={yaw:.2f} pitch={pitch:.2f} "
                        f"deg(y/p/r)={yaw_deg:.1f}/{pitch_deg:.1f}/{roll_deg:.1f}"
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

                if overlay.get("driver_overlay_blink"):
                    ear = metrics.get("eye_ear", float("nan"))
                    mar = metrics.get("mouth_mar", float("nan"))
                    ear_thr = float(overlay.get("driver_blink_ear_thr", 0.21))
                    mar_thr = float(overlay.get("driver_yawn_mar_thr", 0.60))
                    blink = np.isfinite(ear) and ear < ear_thr
                    yawn = np.isfinite(mar) and mar > mar_thr
                    color = blink_alert_color if blink or yawn else blink_color
                    text = f"EAR={ear:.2f} MAR={mar:.2f} blink={int(blink)} yawn={int(yawn)}"
                    cv2.putText(
                        frame,
                        text,
                        blink_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        blink_scale,
                        color,
                        blink_thickness,
                    )

                if overlay.get("driver_overlay_gaze_bar"):
                    thr = float(overlay.get("driver_gaze_low_thr", 0.5))
                    _draw_gaze_bar(frame, metrics.get("gaze_on_road_prob", float("nan")), thr, style)

            elif overlay.get("driver_overlay_metrics"):
                text = "face_conf=0"
                cv2.putText(
                    frame,
                    text,
                    face_conf_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    face_conf_scale,
                    face_conf_color,
                    face_conf_thickness,
                )
            writer.write(frame)

    cap.release()
    mp_face.close()
    if writer:
        writer.release()

    df = pd.DataFrame(data)
    if out_csv_path:
        df.to_csv(out_csv_path, index=False)
    return df

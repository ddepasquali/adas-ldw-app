from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import cv2


YAW_SCALE = 0.35
PITCH_SCALE = 0.35


_FACE_LANDMARK_NOSE = 1
_FACE_LANDMARK_CHIN = 152
_FACE_LANDMARK_LEFT_EYE_OUTER = 33
_FACE_LANDMARK_RIGHT_EYE_OUTER = 263
_FACE_LANDMARK_MOUTH_LEFT = 61
_FACE_LANDMARK_MOUTH_RIGHT = 291

_EYE_LEFT = [33, 160, 158, 133, 153, 144]
_EYE_RIGHT = [362, 385, 387, 263, 373, 380]
_MOUTH_UPPER = 13
_MOUTH_LOWER = 14
_IRIS_LEFT = [468, 469, 470, 471, 472]
_IRIS_RIGHT = [473, 474, 475, 476, 477]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(pts: np.ndarray, idxs) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idxs)
    denom = 2.0 * _dist(p1, p4)
    if denom <= 1e-6:
        return float("nan")
    return (_dist(p2, p6) + _dist(p3, p5)) / denom


def _mouth_aspect_ratio(pts: np.ndarray) -> float:
    v = _dist(pts[_MOUTH_UPPER], pts[_MOUTH_LOWER])
    h = _dist(pts[_FACE_LANDMARK_MOUTH_LEFT], pts[_FACE_LANDMARK_MOUTH_RIGHT])
    return v / max(h, 1e-6)


def _mean_point(pts: np.ndarray, idxs) -> np.ndarray:
    return np.mean(pts[idxs], axis=0)


def _solve_head_pose(pts: np.ndarray, frame_shape):
    image_points = np.array([
        pts[_FACE_LANDMARK_NOSE],
        pts[_FACE_LANDMARK_CHIN],
        pts[_FACE_LANDMARK_LEFT_EYE_OUTER],
        pts[_FACE_LANDMARK_RIGHT_EYE_OUTER],
        pts[_FACE_LANDMARK_MOUTH_LEFT],
        pts[_FACE_LANDMARK_MOUTH_RIGHT],
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    h, w = frame_shape[:2]
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    roll_deg = float(np.degrees(roll))
    return {
        "rvec": rvec,
        "tvec": tvec,
        "camera_matrix": camera_matrix,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
    }


def extract_driver_features(frame, mesh) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "face_conf": 0.0,
            "head_yaw_proxy": float("nan"),
            "head_pitch_proxy": float("nan"),
            "gaze_on_road_prob": float("nan"),
            "eye_ear_left": float("nan"),
            "eye_ear_right": float("nan"),
            "eye_ear": float("nan"),
            "mouth_mar": float("nan"),
            "head_yaw_deg": float("nan"),
            "head_pitch_deg": float("nan"),
            "head_roll_deg": float("nan"),
        }, None

    landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    bbox_w = max(max_x - min_x, 1.0)
    bbox_h = max(max_y - min_y, 1.0)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    nose = pts[_FACE_LANDMARK_NOSE]
    yaw_proxy = (nose[0] - center_x) / bbox_w
    pitch_proxy = (nose[1] - center_y) / bbox_h

    yaw_norm = min(abs(yaw_proxy) / YAW_SCALE, 1.0)
    pitch_norm = min(abs(pitch_proxy) / PITCH_SCALE, 1.0)
    gaze_on_road_prob = max(0.0, 1.0 - (yaw_norm + pitch_norm) / 2.0)

    eye_ear_left = float("nan")
    eye_ear_right = float("nan")
    if pts.shape[0] > max(_EYE_LEFT + _EYE_RIGHT):
        eye_ear_left = _eye_aspect_ratio(pts, _EYE_LEFT)
        eye_ear_right = _eye_aspect_ratio(pts, _EYE_RIGHT)
    eye_ear = float((eye_ear_left + eye_ear_right) / 2.0) if np.isfinite(eye_ear_left) and np.isfinite(eye_ear_right) else float("nan")
    mouth_mar = _mouth_aspect_ratio(pts) if pts.shape[0] > _MOUTH_LOWER else float("nan")

    head_pose = None
    if pts.shape[0] > max(
        _FACE_LANDMARK_NOSE,
        _FACE_LANDMARK_CHIN,
        _FACE_LANDMARK_LEFT_EYE_OUTER,
        _FACE_LANDMARK_RIGHT_EYE_OUTER,
        _FACE_LANDMARK_MOUTH_LEFT,
        _FACE_LANDMARK_MOUTH_RIGHT,
    ):
        head_pose = _solve_head_pose(pts, frame.shape)
    head_yaw_deg = float("nan")
    head_pitch_deg = float("nan")
    head_roll_deg = float("nan")
    if head_pose:
        head_yaw_deg = head_pose["yaw_deg"]
        head_pitch_deg = head_pose["pitch_deg"]
        head_roll_deg = head_pose["roll_deg"]

    iris_left = None
    iris_right = None
    if pts.shape[0] > max(_IRIS_RIGHT):
        iris_left = _mean_point(pts, _IRIS_LEFT)
        iris_right = _mean_point(pts, _IRIS_RIGHT)
    eye_center_left = _mean_point(pts, _EYE_LEFT) if pts.shape[0] > max(_EYE_LEFT) else None
    eye_center_right = _mean_point(pts, _EYE_RIGHT) if pts.shape[0] > max(_EYE_RIGHT) else None

    return {
        "face_conf": 1.0,
        "head_yaw_proxy": float(yaw_proxy),
        "head_pitch_proxy": float(pitch_proxy),
        "gaze_on_road_prob": float(gaze_on_road_prob),
        "eye_ear_left": float(eye_ear_left),
        "eye_ear_right": float(eye_ear_right),
        "eye_ear": float(eye_ear),
        "mouth_mar": float(mouth_mar),
        "head_yaw_deg": head_yaw_deg,
        "head_pitch_deg": head_pitch_deg,
        "head_roll_deg": head_roll_deg,
    }, {
        "landmarks": landmarks,
        "pts": pts,
        "bbox": (min_x, min_y, max_x, max_y),
        "head_pose": head_pose,
        "iris_left": iris_left,
        "iris_right": iris_right,
        "eye_left": eye_center_left,
        "eye_right": eye_center_right,
    }

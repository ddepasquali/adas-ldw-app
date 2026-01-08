from __future__ import annotations

from typing import Generator, Tuple, Optional

import cv2


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, width, height, count


def resize_frame(frame, max_width: Optional[int]):
    if max_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_sampler(
    cap,
    target_hz: float,
    max_width: Optional[int] = None,
) -> Generator[Tuple[float, int, any], None, None]:
    if target_hz <= 0:
        target_hz = 10.0
    next_t = 0.0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_video_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_video_s + 1e-6 < next_t:
            frame_idx += 1
            continue
        frame = resize_frame(frame, max_width)
        yield t_video_s, frame_idx, frame
        next_t += 1.0 / target_hz
        frame_idx += 1


def make_video_writer(path: str, fps: float, frame_size: Tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)

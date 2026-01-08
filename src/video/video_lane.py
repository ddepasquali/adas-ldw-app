from __future__ import annotations

from typing import Dict, Optional, Tuple

import os
import math
import numpy as np
import cv2


DEFAULT_PARAMS = {
    "roi_top_ratio": 0.6,
    "roi_bottom_ratio": 1.0,
    "canny_low": 50,
    "canny_high": 150,
    "hough_rho": 1,
    "hough_theta": math.pi / 180.0,
    "hough_threshold": 50,
    "hough_min_len": 50,
    "hough_max_gap": 150,
    "slope_threshold": 0.5,
    "y_eval_ratio": 0.92,
    "min_lane_width_ratio": 0.25,
    "poly_nwindows": 9,
    "poly_margin": 80,
    "poly_minpix": 50,
    "poly_minpix_total": 300,
    "poly_hist_min": 50,
    "white_l_min": 200,
    "white_s_max": 90,
    "yellow_h_min": 15,
    "yellow_h_max": 35,
    "yellow_s_min": 80,
    "yellow_l_min": 80,
    "mask_close_kernel": 5,
    "ufld_input_w": 800,
    "ufld_input_h": 288,
    "ufld_min_points": 6,
    "yolop_input_w": 640,
    "yolop_input_h": 640,
    "yolop_min_points": 8,
    "fit_order": 2,
    "fit_resid_px": 30,
    "fit_max_iter": 2,
    "fit_y_min_ratio": 0.10,
    "fit_y_max_ratio": 0.95,
}


def _init_metrics():
    return {
        "lane_conf": 0,
        "lane_width_px": float("nan"),
        "lane_offset_m": float("nan"),
        "dist_to_edge_m": float("nan"),
        "tlc_s": float("nan"),
    }


def _average_line(lines):
    if not lines:
        return None
    slopes = []
    intercepts = []
    weights = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = math.hypot(x2 - x1, y2 - y1)
        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(length)
    if not slopes:
        return None
    slope_avg = float(np.average(slopes, weights=weights))
    intercept_avg = float(np.average(intercepts, weights=weights))
    return slope_avg, intercept_avg


def _line_points(slope, intercept, y1, y2):
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))


def _compute_metrics(
    left_x: float,
    right_x: float,
    width_px: float,
    lane_width_m: float,
    camera_shift_m: float,
    frame_w: float,
    prev_offset_m: Optional[float],
    prev_t: Optional[float],
    t_s: Optional[float],
) -> Dict[str, float]:
    metrics = _init_metrics()
    if width_px <= 0:
        return metrics
    shift_px = (camera_shift_m / lane_width_m) * width_px
    vehicle_center_px = frame_w / 2.0 + shift_px
    lane_center_px = (left_x + right_x) / 2.0
    lane_offset_px = vehicle_center_px - lane_center_px
    lane_offset_m = lane_offset_px * lane_width_m / width_px
    dist_to_edge_m = lane_width_m / 2.0 - abs(lane_offset_m)

    tlc_s = float("inf")
    if prev_offset_m is not None and prev_t is not None and t_s is not None:
        dt = max(t_s - prev_t, 1e-3)
        v_lat = (lane_offset_m - prev_offset_m) / dt
        moving_toward_edge = (
            (lane_offset_m > 0 and v_lat > 0) or (lane_offset_m < 0 and v_lat < 0)
        )
        if moving_toward_edge and abs(v_lat) > 1e-3:
            tlc_s = max(dist_to_edge_m, 0.0) / abs(v_lat)

    metrics.update({
        "lane_conf": 1,
        "lane_width_px": width_px,
        "lane_offset_m": lane_offset_m,
        "dist_to_edge_m": dist_to_edge_m,
        "tlc_s": tlc_s,
    })
    return metrics


def _color_mask(frame, cfg: Dict[str, float]) -> np.ndarray:
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    white = (l >= cfg["white_l_min"]) & (s <= cfg["white_s_max"])
    yellow = (
        (h >= cfg["yellow_h_min"]) & (h <= cfg["yellow_h_max"]) &
        (s >= cfg["yellow_s_min"]) & (l >= cfg["yellow_l_min"])
    )

    mask = np.zeros_like(l)
    mask[white | yellow] = 255

    k = int(cfg.get("mask_close_kernel", 0) or 0)
    if k >= 3:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def compute_color_mask(frame, cfg: Dict[str, float]) -> np.ndarray:
    return _color_mask(frame, cfg)


def _select_fit_points(y: np.ndarray, x: np.ndarray, y_min: float, y_max: float, min_points: int):
    mask = (y >= y_min) & (y <= y_max)
    if mask.sum() < min_points:
        mask = np.ones_like(y, dtype=bool)
    return y[mask], x[mask]


def _robust_polyfit(
    y: np.ndarray,
    x: np.ndarray,
    order: int,
    resid_thresh: float,
    max_iter: int,
    weights: Optional[np.ndarray] = None,
):
    if len(x) < order + 1:
        return None
    if weights is not None:
        fit = np.polyfit(y, x, order, w=weights)
    else:
        fit = np.polyfit(y, x, order)

    for _ in range(max_iter):
        pred = np.polyval(fit, y)
        resid = np.abs(pred - x)
        mask = resid <= resid_thresh
        if mask.sum() < order + 1:
            break
        if weights is not None:
            fit = np.polyfit(y[mask], x[mask], order, w=weights[mask])
        else:
            fit = np.polyfit(y[mask], x[mask], order)
    return fit


def detect_lane_hough(
    frame,
    lane_width_m: float,
    camera_shift_m: float,
    cfg: Dict[str, float],
    prev_offset_m: Optional[float],
    prev_t: Optional[float],
    t_s: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    h, w = frame.shape[:2]
    roi_top = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, cfg["canny_low"], cfg["canny_high"])

    mask = np.zeros_like(edges)
    polygon = np.array([
        [0, roi_bottom],
        [w, roi_bottom],
        [w, roi_top],
        [0, roi_top],
    ])
    cv2.fillPoly(mask, [polygon], 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked,
        cfg["hough_rho"],
        cfg["hough_theta"],
        threshold=cfg["hough_threshold"],
        minLineLength=cfg["hough_min_len"],
        maxLineGap=cfg["hough_max_gap"],
    )

    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < cfg["slope_threshold"]:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    left_fit = _average_line(left_lines)
    right_fit = _average_line(right_lines)

    debug_lines = {}
    metrics = _init_metrics()

    if left_fit and right_fit:
        y_eval = int(h * cfg["y_eval_ratio"])
        left_x = int((y_eval - left_fit[1]) / left_fit[0]) if left_fit[0] != 0 else None
        right_x = int((y_eval - right_fit[1]) / right_fit[0]) if right_fit[0] != 0 else None
        if left_x is not None and right_x is not None and right_x > left_x:
            lane_width_px = right_x - left_x
            min_width = cfg["min_lane_width_ratio"] * w
            if lane_width_px >= min_width:
                metrics = _compute_metrics(
                    left_x,
                    right_x,
                    lane_width_px,
                    lane_width_m,
                    camera_shift_m,
                    w,
                    prev_offset_m,
                    prev_t,
                    t_s,
                )

                left_pts = _line_points(left_fit[0], left_fit[1], roi_top, roi_bottom)
                right_pts = _line_points(right_fit[0], right_fit[1], roi_top, roi_bottom)
                if left_pts:
                    debug_lines["left"] = left_pts
                if right_pts:
                    debug_lines["right"] = right_pts

    return metrics, debug_lines


def detect_lane_poly(
    frame,
    lane_width_m: float,
    camera_shift_m: float,
    cfg: Dict[str, float],
    prev_offset_m: Optional[float],
    prev_t: Optional[float],
    t_s: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    h, w = frame.shape[:2]
    roi_top = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])
    roi_top = max(0, min(roi_top, h - 2))
    roi_bottom = max(roi_top + 1, min(roi_bottom, h))

    roi = frame[roi_top:roi_bottom, :]
    binary = _color_mask(roi, cfg)

    histogram = np.sum(binary[binary.shape[0] // 2 :, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    if histogram[left_base] < cfg["poly_hist_min"] or histogram[right_base] < cfg["poly_hist_min"]:
        return _init_metrics(), {}

    if right_base - left_base < cfg["min_lane_width_ratio"] * w:
        return _init_metrics(), {}

    nwindows = int(cfg["poly_nwindows"])
    margin = int(cfg["poly_margin"])
    minpix = int(cfg["poly_minpix"])
    window_height = max(1, binary.shape[0] // nwindows)

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = left_base
    rightx_current = right_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    if len(left_lane_inds) < cfg["poly_minpix_total"] or len(right_lane_inds) < cfg["poly_minpix_total"]:
        return _init_metrics(), {}

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) < 2 or len(rightx) < 2:
        return _init_metrics(), {}

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    y_eval = int(h * cfg["y_eval_ratio"])
    y_eval = max(roi_top + 1, min(y_eval, roi_bottom - 1))
    y_eval_roi = y_eval - roi_top

    left_x = float(left_fit[0] * y_eval_roi**2 + left_fit[1] * y_eval_roi + left_fit[2])
    right_x = float(right_fit[0] * y_eval_roi**2 + right_fit[1] * y_eval_roi + right_fit[2])

    if right_x <= left_x:
        return _init_metrics(), {}

    lane_width_px = right_x - left_x
    if lane_width_px < cfg["min_lane_width_ratio"] * w:
        return _init_metrics(), {}

    metrics = _compute_metrics(
        left_x,
        right_x,
        lane_width_px,
        lane_width_m,
        camera_shift_m,
        w,
        prev_offset_m,
        prev_t,
        t_s,
    )

    y_min = max(0.0, min(lefty.min(), righty.min()))
    y_max = min(float(roi.shape[0] - 1), max(lefty.max(), righty.max()))
    if y_max <= y_min:
        return metrics, {}
    ploty = np.linspace(y_min, y_max, max(2, int(y_max - y_min)))
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    left_pts = [(int(x), int(y + roi_top)) for x, y in zip(left_fitx, ploty) if 0 <= x < w]
    right_pts = [(int(x), int(y + roi_top)) for x, y in zip(right_fitx, ploty) if 0 <= x < w]

    debug_lines = {}
    if left_pts:
        debug_lines["left"] = left_pts
    if right_pts:
        debug_lines["right"] = right_pts

    return metrics, debug_lines


def load_ufld_model(model_path: str, log):
    if not model_path:
        log("UFLD: no model path provided")
        return None
    if not os.path.exists(model_path):
        log(f"UFLD model not found: {model_path}")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
    except Exception as exc:
        log(f"UFLD load failed: {exc}")
        return None
    return {
        "net": net,
    }


def _parse_ufld_output(out: np.ndarray) -> Optional[np.ndarray]:
    if out is None or out.ndim != 4:
        return None
    # Expect either (B, grid+1, rows, lanes) or (B, lanes, rows, grid+1)
    if out.shape[1] <= 10 and out.shape[-1] > 10:
        return out[0].transpose(2, 1, 0)
    return out[0]


def detect_lane_ufld(
    frame,
    lane_width_m: float,
    camera_shift_m: float,
    cfg: Dict[str, float],
    prev_offset_m: Optional[float],
    prev_t: Optional[float],
    t_s: Optional[float],
    model: Optional[Dict[str, object]],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    metrics = _init_metrics()
    if model is None:
        return metrics, {}

    net = model.get("net")
    if net is None:
        return metrics, {}

    h, w = frame.shape[:2]
    roi_top = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])

    masked = frame.copy()
    if roi_bottom < h:
        masked[roi_bottom:, :] = 0

    input_w = int(cfg.get("ufld_input_w", 800))
    input_h = int(cfg.get("ufld_input_h", 288))

    resized = cv2.resize(masked, (input_w, input_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    out = None
    if isinstance(outputs, list):
        for candidate in outputs:
            if isinstance(candidate, np.ndarray) and candidate.ndim == 4:
                out = candidate
                break
    elif isinstance(outputs, np.ndarray):
        out = outputs

    scores = _parse_ufld_output(out) if out is not None else None
    if scores is None:
        return metrics, {}

    griding_plus, num_rows, num_lanes = scores.shape
    griding_num = griding_plus - 1
    if griding_num <= 1:
        return metrics, {}

    loc = np.argmax(scores, axis=0)
    xs = loc.astype(float)
    xs[xs == griding_num] = -1
    xs = xs / float(griding_num - 1) * (input_w - 1)

    row_anchors = np.linspace(0, input_h - 1, num_rows)
    ys_full = row_anchors / float(input_h - 1) * (h - 1)
    xs_full = xs / float(input_w - 1) * (w - 1)

    lanes = []
    for lane_idx in range(num_lanes):
        pts = []
        for r in range(num_rows):
            if xs[r, lane_idx] < 0:
                continue
            y = float(ys_full[r])
            if y < roi_top or y > roi_bottom:
                continue
            x = float(xs_full[r, lane_idx])
            pts.append((x, y))
        if len(pts) >= int(cfg.get("ufld_min_points", 6)):
            lanes.append(pts)

    if not lanes:
        return metrics, {}

    center_x = w / 2.0
    lane_stats = []
    for pts in lanes:
        xs_lane = [p[0] for p in pts]
        lane_stats.append((float(np.median(xs_lane)), pts))

    left_candidates = [lane for lane in lane_stats if lane[0] < center_x]
    right_candidates = [lane for lane in lane_stats if lane[0] > center_x]
    if not left_candidates or not right_candidates:
        return metrics, {}

    left_lane = max(left_candidates, key=lambda x: x[0])[1]
    right_lane = min(right_candidates, key=lambda x: x[0])[1]

    left_lane = sorted(left_lane, key=lambda p: p[1])
    right_lane = sorted(right_lane, key=lambda p: p[1])

    left_y = np.array([p[1] for p in left_lane])
    left_x = np.array([p[0] for p in left_lane])
    right_y = np.array([p[1] for p in right_lane])
    right_x = np.array([p[0] for p in right_lane])

    if len(left_x) < 2 or len(right_x) < 2:
        return metrics, {}

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    y_eval = int(h * cfg["y_eval_ratio"])
    y_eval = max(roi_top + 1, min(y_eval, roi_bottom - 1))
    left_eval = np.polyval(left_fit, y_eval)
    right_eval = np.polyval(right_fit, y_eval)

    if right_eval <= left_eval:
        return metrics, {}

    lane_width_px = right_eval - left_eval
    if lane_width_px < cfg["min_lane_width_ratio"] * w:
        return metrics, {}

    metrics = _compute_metrics(
        float(left_eval),
        float(right_eval),
        float(lane_width_px),
        lane_width_m,
        camera_shift_m,
        w,
        prev_offset_m,
        prev_t,
        t_s,
    )

    plot_y_min = max(roi_top, min(left_y_fit.min(), right_y_fit.min()))
    plot_y_max = min(float(roi_bottom - 1), max(left_y_fit.max(), right_y_fit.max()))
    if plot_y_max <= plot_y_min:
        return metrics, debug_lines
    ploty = np.linspace(plot_y_min, plot_y_max, max(2, int(plot_y_max - plot_y_min)))
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    left_pts = [(int(x), int(y)) for x, y in zip(left_fitx, ploty) if 0 <= x < w]
    right_pts = [(int(x), int(y)) for x, y in zip(right_fitx, ploty) if 0 <= x < w]

    debug_lines = {}
    if left_pts:
        debug_lines["left"] = left_pts
    if right_pts:
        debug_lines["right"] = right_pts

    return metrics, debug_lines


def load_yolop_model(model_path: str, log):
    if not model_path:
        log("YOLOP: no model path provided")
        return None
    if not os.path.exists(model_path):
        log(f"YOLOP model not found: {model_path}")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
    except Exception as exc:
        log(f"YOLOP load failed: {exc}")
        return None
    output_names = net.getUnconnectedOutLayersNames()
    return {
        "net": net,
        "output_names": output_names,
    }


def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / float(h), new_w / float(w))
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h
    dw /= 2
    dh /= 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top, new_unpad_w, new_unpad_h


def _select_lane_seg(outputs, output_names):
    if output_names:
        for name, out in zip(output_names, outputs):
            if "lane" in name and isinstance(out, np.ndarray):
                return out

    seg_candidates = [out for out in outputs if isinstance(out, np.ndarray) and out.ndim == 4 and out.shape[1] == 2]
    if not seg_candidates:
        return None
    if len(seg_candidates) == 1:
        return seg_candidates[0]

    best = None
    best_score = None
    for cand in seg_candidates:
        mask = np.argmax(cand, axis=1)[0]
        positive = int((mask > 0).sum())
        if best_score is None or positive < best_score:
            best_score = positive
            best = cand
    return best


def detect_lane_yolop(
    frame,
    lane_width_m: float,
    camera_shift_m: float,
    cfg: Dict[str, float],
    prev_offset_m: Optional[float],
    prev_t: Optional[float],
    t_s: Optional[float],
    model: Optional[Dict[str, object]],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    metrics = _init_metrics()
    if model is None:
        return metrics, {}

    net = model.get("net")
    if net is None:
        return metrics, {}

    output_names = model.get("output_names", [])

    h, w = frame.shape[:2]
    roi_top = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])

    input_w = int(cfg.get("yolop_input_w", 640))
    input_h = int(cfg.get("yolop_input_h", 640))
    padded, r, dw, dh, new_unpad_w, new_unpad_h = _letterbox(frame, (input_h, input_w))

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]

    net.setInput(blob)
    outputs = net.forward(output_names)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    lane_seg = _select_lane_seg(outputs, output_names)
    if lane_seg is None:
        return metrics, {}

    lane_seg = lane_seg[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    lane_mask = np.argmax(lane_seg, axis=1)[0].astype(np.uint8)
    lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    lane_mask[:roi_top, :] = 0
    lane_mask[roi_bottom:, :] = 0

    debug_lines = {
        "lane_mask": (lane_mask * 255).astype(np.uint8),
    }

    ys, xs = np.where(lane_mask > 0)
    if len(xs) < int(cfg.get("yolop_min_points", 8)):
        return metrics, debug_lines

    center_x = w / 2.0
    left_pts = []
    right_pts = []
    for y in np.unique(ys):
        row_xs = xs[ys == y]
        left = row_xs[row_xs < center_x]
        right = row_xs[row_xs > center_x]
        if left.size > 0:
            left_pts.append((float(np.median(left)), float(y)))
        if right.size > 0:
            right_pts.append((float(np.median(right)), float(y)))

    if len(left_pts) < int(cfg.get("yolop_min_points", 8)) or len(right_pts) < int(cfg.get("yolop_min_points", 8)):
        return metrics, debug_lines

    left_pts = sorted(left_pts, key=lambda p: p[1])
    right_pts = sorted(right_pts, key=lambda p: p[1])

    left_y = np.array([p[1] for p in left_pts])
    left_x = np.array([p[0] for p in left_pts])
    right_y = np.array([p[1] for p in right_pts])
    right_x = np.array([p[0] for p in right_pts])

    if len(left_x) < 2 or len(right_x) < 2:
        return metrics, debug_lines

    fit_y_min = roi_top + (roi_bottom - roi_top) * float(cfg.get("fit_y_min_ratio", 0.10))
    fit_y_max = roi_top + (roi_bottom - roi_top) * float(cfg.get("fit_y_max_ratio", 0.95))
    min_points = int(cfg.get("yolop_min_points", 8))

    left_y_fit, left_x_fit = _select_fit_points(left_y, left_x, fit_y_min, fit_y_max, min_points)
    right_y_fit, right_x_fit = _select_fit_points(right_y, right_x, fit_y_min, fit_y_max, min_points)

    if len(left_x_fit) < 2 or len(right_x_fit) < 2:
        return metrics, debug_lines

    order = int(cfg.get("fit_order", 2))
    max_iter = int(cfg.get("fit_max_iter", 2))
    resid_thresh = max(float(cfg.get("fit_resid_px", 30)), 0.03 * w)

    left_w = left_y_fit - left_y_fit.min()
    left_w = 0.2 + 0.8 * (left_w / (left_w.ptp() + 1e-6))
    right_w = right_y_fit - right_y_fit.min()
    right_w = 0.2 + 0.8 * (right_w / (right_w.ptp() + 1e-6))

    left_fit = _robust_polyfit(left_y_fit, left_x_fit, order, resid_thresh, max_iter, left_w)
    right_fit = _robust_polyfit(right_y_fit, right_x_fit, order, resid_thresh, max_iter, right_w)

    if left_fit is None or right_fit is None:
        return metrics, debug_lines

    debug_lines["left_points"] = [(int(x), int(y)) for x, y in zip(left_x_fit, left_y_fit)]
    debug_lines["right_points"] = [(int(x), int(y)) for x, y in zip(right_x_fit, right_y_fit)]

    y_eval = int(h * cfg["y_eval_ratio"])
    y_eval = max(roi_top + 1, min(y_eval, roi_bottom - 1))
    left_eval = np.polyval(left_fit, y_eval)
    right_eval = np.polyval(right_fit, y_eval)

    if right_eval <= left_eval:
        return metrics, debug_lines

    lane_width_px = right_eval - left_eval
    if lane_width_px < cfg["min_lane_width_ratio"] * w:
        return metrics, debug_lines

    metrics = _compute_metrics(
        float(left_eval),
        float(right_eval),
        float(lane_width_px),
        lane_width_m,
        camera_shift_m,
        w,
        prev_offset_m,
        prev_t,
        t_s,
    )

    ploty = np.linspace(roi_top, roi_bottom - 1, max(2, roi_bottom - roi_top))
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    left_line = [(int(x), int(y)) for x, y in zip(left_fitx, ploty) if 0 <= x < w]
    right_line = [(int(x), int(y)) for x, y in zip(right_fitx, ploty) if 0 <= x < w]

    if left_line:
        debug_lines["left"] = left_line
    if right_line:
        debug_lines["right"] = right_line

    return metrics, debug_lines


def detect_lane(
    frame,
    lane_width_m: float,
    camera_shift_m: float,
    params: Optional[Dict[str, float]] = None,
    prev_offset_m: Optional[float] = None,
    prev_t: Optional[float] = None,
    t_s: Optional[float] = None,
    mode: str = "hough",
    model: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)

    if mode == "poly":
        return detect_lane_poly(frame, lane_width_m, camera_shift_m, cfg, prev_offset_m, prev_t, t_s)

    if mode == "ufld":
        return detect_lane_ufld(
            frame,
            lane_width_m,
            camera_shift_m,
            cfg,
            prev_offset_m,
            prev_t,
            t_s,
            model,
        )

    if mode == "yolop":
        return detect_lane_yolop(
            frame,
            lane_width_m,
            camera_shift_m,
            cfg,
            prev_offset_m,
            prev_t,
            t_s,
            model,
        )

    return detect_lane_hough(frame, lane_width_m, camera_shift_m, cfg, prev_offset_m, prev_t, t_s)

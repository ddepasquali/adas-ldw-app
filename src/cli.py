from __future__ import annotations

import argparse
import os
from datetime import timedelta

import pandas as pd

from .core import config_io
from .core.align_resample import align_and_resample
from .core.baseline import compute_baseline, apply_baseline_zscores
from .core.fuse_and_decide import decide_events
from .core.overlay_style import resolve_overlay_style
from .io.io_muse import load_muse
from .io.io_obd import load_obd
from .io.io_phyphox import load_phyphox
from .io.io_polar import load_polar
from .render.render_driver import process_driver_video
from .render.render_road import process_road_video
from .utils.sanity_checks import ensure_dir
from .utils.utils_time import parse_date, combine_date_time, add_t_seconds


DEFAULT_FILES = {
    "muse_csv": "scenario01-2026-01-05--16-23-30-muse-legacy.csv",
    "obd_csv": "scenario01-2026-01-05--16-23-48-obd-legacy.csv",
    "phyphox_csv": "scenario01-2026-01-05--16-23-00-phyphox-legacy.csv",
    "polar_csv": "scenario01-2026-01-05--16-23-15-polar-full.csv",
    "muse_baseline_csv": "muse_baseline.csv",
    "polar_baseline_csv": "polar_baseline.csv",
    "road_video": "road.mp4",
    "driver_video": "driver.mp4",
}


def make_logger(log_path: str):
    f = open(log_path, "w", encoding="utf-8")

    def _log(message: str):
        print(message)
        f.write(message + "\n")
        f.flush()

    return _log, f


def resolve_path(base_dir: str, name: str) -> str:
    if not name:
        return ""
    return name if os.path.isabs(name) else os.path.join(base_dir, name)


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument("--session_date", type=str)
    parser.add_argument("--anchor_obd_time", type=str)
    parser.add_argument("--anchor_road_video_s", type=float)
    parser.add_argument("--anchor_driver_video_s", type=float)
    parser.add_argument("--lane_width_m", type=float)
    parser.add_argument("--camera_shift_m", type=float)
    parser.add_argument("--resample_hz", type=float)
    parser.add_argument("--tlc_thr", type=float)
    parser.add_argument("--edge_thr", type=float)
    parser.add_argument("--attn_thr", type=float)
    parser.add_argument("--attn_min_s", type=float)
    parser.add_argument("--yaw_thr", type=float)
    parser.add_argument("--veh_thr", type=int)
    parser.add_argument("--polar_quality_thr", type=float)
    parser.add_argument("--bio_hr_z_thr", type=float)
    parser.add_argument("--bio_rmssd_z_thr", type=float)
    parser.add_argument("--bio_eeg_z_thr", type=float)


def apply_overrides(cfg, args):
    for key in config_io.DEFAULT_CONFIG.keys():
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def run_command(args):
    cfg = config_io.load_config(args.config)
    cfg = apply_overrides(cfg, args)
    overlay_style = resolve_overlay_style(cfg)

    in_dir = args.in_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    log_path = os.path.join(out_dir, "run_log.txt")
    log, log_file = make_logger(log_path)

    session_date = parse_date(cfg["session_date"])

    muse_path = resolve_path(in_dir, args.muse_csv)
    obd_path = resolve_path(in_dir, args.obd_csv)
    phy_path = resolve_path(in_dir, args.phyphox_csv)
    polar_path = resolve_path(in_dir, args.polar_csv)
    muse_base_path = resolve_path(in_dir, args.muse_baseline_csv)
    polar_base_path = resolve_path(in_dir, args.polar_baseline_csv)
    road_video_path = resolve_path(in_dir, args.road_video)
    driver_video_path = resolve_path(in_dir, args.driver_video)

    baseline_path = os.path.join(out_dir, "baseline_summary.json")
    if os.path.exists(muse_base_path) or os.path.exists(polar_base_path):
        baseline = compute_baseline(
            muse_base_path if os.path.exists(muse_base_path) else "",
            polar_base_path if os.path.exists(polar_base_path) else "",
            session_date,
            cfg["polar_quality_thr"],
            baseline_path,
            log,
        )
    else:
        log("Baseline files not found; skipping baseline normalization")
        baseline = compute_baseline(
            "",
            "",
            session_date,
            cfg["polar_quality_thr"],
            baseline_path,
            log,
        )

    muse_df = pd.DataFrame()
    polar_df = pd.DataFrame()
    obd_df = pd.DataFrame()
    phy_df = pd.DataFrame()

    if os.path.exists(muse_path):
        muse_df = load_muse(muse_path, session_date, log)
    else:
        log(f"Muse file missing: {muse_path}")
    if os.path.exists(polar_path):
        polar_df = load_polar(polar_path, session_date, log)
    else:
        log(f"Polar file missing: {polar_path}")
    if os.path.exists(obd_path):
        obd_df = load_obd(obd_path, session_date, log)
    else:
        log(f"OBD file missing: {obd_path}")
    if os.path.exists(phy_path):
        phy_df = load_phyphox(phy_path, session_date, log)
    else:
        log(f"Phyphox file missing: {phy_path}")

    anchor_dt = combine_date_time(session_date, cfg["anchor_obd_time"])
    road_start_dt = anchor_dt - timedelta(seconds=float(cfg["anchor_road_video_s"]))
    driver_start_dt = anchor_dt - timedelta(seconds=float(cfg["anchor_driver_video_s"]))

    candidates = []
    for df in (muse_df, polar_df, obd_df, phy_df):
        if not df.empty:
            candidates.append(df["datetime"].min())
    if os.path.exists(road_video_path):
        candidates.append(road_start_dt)
    if os.path.exists(driver_video_path):
        candidates.append(driver_start_dt)
    if not candidates:
        raise RuntimeError("No valid sources found to determine global start.")
    global_start = min(candidates)

    if not muse_df.empty:
        muse_df = add_t_seconds(muse_df, global_start)
    if not polar_df.empty:
        polar_df = add_t_seconds(polar_df, global_start)
    if not obd_df.empty:
        obd_df = add_t_seconds(obd_df, global_start)
    if not phy_df.empty:
        phy_df = add_t_seconds(phy_df, global_start)

    muse_df, polar_df = apply_baseline_zscores(
        muse_df, polar_df, baseline, cfg["polar_quality_thr"]
    )

    lane_params = {
        "roi_top_ratio": args.lane_roi_top,
        "roi_bottom_ratio": args.lane_roi_bottom,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "hough_threshold": args.hough_threshold,
        "hough_min_len": args.hough_min_len,
        "hough_max_gap": args.hough_max_gap,
        "min_lane_width_ratio": args.min_lane_width_ratio,
        "yolop_min_points": args.yolop_min_points,
        "fit_order": args.fit_order,
        "fit_resid_px": args.fit_resid_px,
        "fit_max_iter": args.fit_max_iter,
        "fit_y_min_ratio": args.fit_y_min_ratio,
        "fit_y_max_ratio": args.fit_y_max_ratio,
    }
    lane_params.update({
        "lane_color_red_ratio": cfg.get("lane_color_red_ratio"),
        "lane_color_yellow_ratio": cfg.get("lane_color_yellow_ratio"),
        "lane_color_red_m": cfg.get("lane_color_red_m"),
        "lane_color_yellow_m": cfg.get("lane_color_yellow_m"),
        "lane_color_eval_band_px": cfg.get("lane_color_eval_band_px"),
        "lane_color_overlay": cfg.get("lane_color_overlay"),
        "road_overlay_lane_eval": cfg.get("road_overlay_lane_eval"),
        "road_overlay_boxes": cfg.get("road_overlay_boxes"),
        "road_overlay_metrics": cfg.get("road_overlay_metrics"),
        "lane_line_cluster_gap_px": cfg.get("lane_line_cluster_gap_px"),
        "lane_force_red_start_s": cfg.get("lane_force_red_start_s"),
        "lane_force_red_end_s": cfg.get("lane_force_red_end_s"),
        "lane_force_yellow_pad_s": cfg.get("lane_force_yellow_pad_s"),
        "lane_force_windows_s": cfg.get("lane_force_windows_s"),
        "lane_center_calib_start_s": cfg.get("lane_center_calib_start_s"),
        "lane_center_calib_end_s": cfg.get("lane_center_calib_end_s"),
        "lane_center_calib_side": cfg.get("lane_center_calib_side"),
        "lane_center_calib_edge_m": cfg.get("lane_center_calib_edge_m"),
        "lane_center_calib_min_lines": cfg.get("lane_center_calib_min_lines"),
        "lane_center_calib_apply_once": cfg.get("lane_center_calib_apply_once"),
    })
    lane_params = {k: v for k, v in lane_params.items() if v is not None}

    debug_lane_mask = args.debug_lane_mask or args.debug_lane_all or cfg.get("road_overlay_lane_mask", False)
    debug_lane_edges = args.debug_lane_edges or args.debug_lane_all or cfg.get("road_overlay_edges", False)
    debug_lane_color = args.debug_lane_color_mask or args.debug_lane_all or cfg.get("road_overlay_color_mask", False)
    debug_lane_hough = args.debug_lane_hough or args.debug_lane_all or cfg.get("road_overlay_hough", False)
    debug_lane_points = args.debug_lane_points or args.debug_lane_all or cfg.get("road_overlay_lane_points_detail", False)
    draw_lane_roi = (
        args.debug_lane_roi
        or args.debug_lane_all
        or cfg.get("road_overlay_roi", False)
        or cfg.get("lane_roi_overlay", False)
    )
    draw_lane_dots = cfg.get("road_overlay_lane_dots", True) or args.debug_lane_points or args.debug_lane_all

    road_lane_df = pd.DataFrame()
    road_scene_df = pd.DataFrame()
    if os.path.exists(road_video_path):
        road_lane_df, road_scene_df = process_road_video(
            road_video_path,
            road_start_dt,
            global_start,
            os.path.join(out_dir, "road_lane.csv"),
            os.path.join(out_dir, "road_scene.csv"),
            os.path.join(out_dir, "road_annotated.mp4"),
            cfg["lane_width_m"],
            cfg["camera_shift_m"],
            args.video_hz or cfg["resample_hz"],
            args.max_width,
            None if args.no_yolo else args.yolo_model,
            args.yolo_conf,
            args.lane_model,
            lane_params,
            args.lane_mode,
            draw_lane_roi,
            debug_lane_mask,
            debug_lane_edges,
            debug_lane_color,
            debug_lane_hough,
            debug_lane_points,
            draw_lane_dots,
            overlay_style.get("road"),
            log,
        )
    else:
        log(f"Road video missing: {road_video_path}")
        pd.DataFrame().to_csv(os.path.join(out_dir, "road_lane.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "road_scene.csv"), index=False)

    driver_df = pd.DataFrame()
    if os.path.exists(driver_video_path):
        driver_overlay = {
            "driver_overlay_bbox": cfg.get("driver_overlay_bbox"),
            "driver_overlay_mesh": cfg.get("driver_overlay_mesh"),
            "driver_overlay_contours": cfg.get("driver_overlay_contours"),
            "driver_overlay_iris": cfg.get("driver_overlay_iris"),
            "driver_overlay_pose": cfg.get("driver_overlay_pose"),
            "driver_overlay_gaze_line": cfg.get("driver_overlay_gaze_line"),
            "driver_overlay_metrics": cfg.get("driver_overlay_metrics"),
            "driver_overlay_gaze_bar": cfg.get("driver_overlay_gaze_bar"),
            "driver_overlay_blink": cfg.get("driver_overlay_blink"),
            "driver_gaze_low_thr": cfg.get("driver_gaze_low_thr"),
            "driver_blink_ear_thr": cfg.get("driver_blink_ear_thr"),
            "driver_yawn_mar_thr": cfg.get("driver_yawn_mar_thr"),
            "driver_gaze_line_scale_px": cfg.get("driver_gaze_line_scale_px"),
            "driver_landmark_step": cfg.get("driver_landmark_step"),
        }
        driver_df = process_driver_video(
            driver_video_path,
            driver_start_dt,
            global_start,
            os.path.join(out_dir, "driver_state.csv"),
            os.path.join(out_dir, "driver_annotated.mp4"),
            args.video_hz or cfg["resample_hz"],
            args.max_width,
            driver_overlay,
            overlay_style.get("driver"),
            log,
        )
    else:
        log(f"Driver video missing: {driver_video_path}")
        pd.DataFrame().to_csv(os.path.join(out_dir, "driver_state.csv"), index=False)

    sources = {
        "muse": muse_df,
        "polar": polar_df,
        "obd": obd_df,
        "phyphox": phy_df,
        "driver": driver_df,
        "road_lane": road_lane_df,
        "road_scene": road_scene_df,
    }

    resampled, fused = align_and_resample(sources, cfg["resample_hz"])

    resampled.get("muse", pd.DataFrame()).to_csv(os.path.join(out_dir, "muse_features.csv"), index=False)
    resampled.get("polar", pd.DataFrame()).to_csv(os.path.join(out_dir, "polar_features.csv"), index=False)
    resampled.get("obd", pd.DataFrame()).to_csv(os.path.join(out_dir, "obd_features.csv"), index=False)
    resampled.get("phyphox", pd.DataFrame()).to_csv(os.path.join(out_dir, "phyphox_features.csv"), index=False)

    fused.to_csv(os.path.join(out_dir, "fused_10hz.csv"), index=False)

    events = decide_events(fused, cfg)
    events.to_csv(os.path.join(out_dir, "lane_events.csv"), index=False)

    log("Run complete.")
    log_file.close()


def init_config_command(args):
    path = config_io.ensure_config(args.config)
    print(path)


def build_parser():
    parser = argparse.ArgumentParser(description="LDW nuisance alert PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_p = subparsers.add_parser("init-config", help="Create default config if missing")
    init_p.add_argument("--config", default="config/session.yaml")
    init_p.set_defaults(func=init_config_command)

    run_p = subparsers.add_parser("run", help="Run the LDW pipeline")
    run_p.add_argument("--config", default="config/session.yaml")
    run_p.add_argument("--in_dir", default="data/raw")
    run_p.add_argument("--out_dir", default="data/feat")

    run_p.add_argument("--muse_csv", default=DEFAULT_FILES["muse_csv"])
    run_p.add_argument("--obd_csv", default=DEFAULT_FILES["obd_csv"])
    run_p.add_argument("--phyphox_csv", default=DEFAULT_FILES["phyphox_csv"])
    run_p.add_argument("--polar_csv", default=DEFAULT_FILES["polar_csv"])
    run_p.add_argument("--muse_baseline_csv", default=DEFAULT_FILES["muse_baseline_csv"])
    run_p.add_argument("--polar_baseline_csv", default=DEFAULT_FILES["polar_baseline_csv"])
    run_p.add_argument("--road_video", default=DEFAULT_FILES["road_video"])
    run_p.add_argument("--driver_video", default=DEFAULT_FILES["driver_video"])

    run_p.add_argument("--max_width", type=int, default=1280)
    run_p.add_argument("--video_hz", type=float)

    run_p.add_argument("--yolo_model", default="models/yolov8n.pt")
    run_p.add_argument("--yolo_conf", type=float, default=0.25)
    run_p.add_argument("--no_yolo", action="store_true")

    run_p.add_argument("--debug_lane_roi", action="store_true")
    run_p.add_argument("--debug_lane_mask", action="store_true")
    run_p.add_argument("--debug_lane_edges", action="store_true")
    run_p.add_argument("--debug_lane_color_mask", action="store_true")
    run_p.add_argument("--debug_lane_hough", action="store_true")
    run_p.add_argument("--debug_lane_points", action="store_true")
    run_p.add_argument("--debug_lane_all", action="store_true")

    run_p.add_argument("--lane_mode", choices=["hough", "poly", "ufld", "yolop"], default="hough")
    run_p.add_argument("--lane_model", default="models/yolop-640-640.onnx")
    run_p.add_argument("--lane_roi_top", type=float)
    run_p.add_argument("--lane_roi_bottom", type=float)
    run_p.add_argument("--canny_low", type=int)
    run_p.add_argument("--canny_high", type=int)
    run_p.add_argument("--hough_threshold", type=int)
    run_p.add_argument("--hough_min_len", type=int)
    run_p.add_argument("--hough_max_gap", type=int)
    run_p.add_argument("--min_lane_width_ratio", type=float)
    run_p.add_argument("--yolop_min_points", type=int)
    run_p.add_argument("--fit_order", type=int)
    run_p.add_argument("--fit_resid_px", type=float)
    run_p.add_argument("--fit_max_iter", type=int)
    run_p.add_argument("--fit_y_min_ratio", type=float)
    run_p.add_argument("--fit_y_max_ratio", type=float)

    add_config_args(run_p)
    run_p.set_defaults(func=run_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "bicycle"}
VULN_CLASSES = {"person", "bicycle", "motorcycle"}


def load_yolo(model_path: str, log):
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except Exception as exc:
        log(f"YOLO load failed: {exc}")
        return None


def detect_objects(model, frame, conf: float = 0.25) -> Dict[str, object]:
    if model is None:
        return {
            "obj_count": 0,
            "vehicles_count": 0,
            "vuln_count": 0,
            "boxes": [],
        }

    results = model.predict(source=frame, conf=conf, verbose=False)
    if not results:
        return {"obj_count": 0, "vehicles_count": 0, "vuln_count": 0, "boxes": []}

    res = results[0]
    boxes_out: List[Dict[str, float]] = []
    vehicles = 0
    vuln = 0
    names = res.names if hasattr(res, "names") else {}
    for box in res.boxes:
        cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
        label = names.get(cls_id, str(cls_id))
        conf_score = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        boxes_out.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "label": label,
            "conf": conf_score,
        })
        if label in VEHICLE_CLASSES:
            vehicles += 1
        if label in VULN_CLASSES:
            vuln += 1

    return {
        "obj_count": len(boxes_out),
        "vehicles_count": vehicles,
        "vuln_count": vuln,
        "boxes": boxes_out,
    }

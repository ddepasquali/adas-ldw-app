from __future__ import annotations

import os
from typing import Dict


def check_inputs(paths: Dict[str, str], log):
    ok = True
    for name, path in paths.items():
        if not path:
            continue
        if not os.path.exists(path):
            log(f"Missing input: {name} -> {path}")
            ok = False
    return ok


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

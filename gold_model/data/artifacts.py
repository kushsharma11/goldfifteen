"""CSV datasets with UTC validation and JSON provenance manifests."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_model.utils.time import parse_datetime, utc_now

DATE_COLUMNS = ("timestamp", "market_start", "market_end", "label_available_at")


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def write_dataset(frame: pd.DataFrame, path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_json(path.with_suffix(".manifest.json"), {
        "created_at": utc_now(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame), "markets": int(frame.market_id.nunique()) if len(frame) else 0,
        "provenance": provenance, "builder": frame.attrs,
        "units": {"prices": "USD", "probabilities": "0..1", "returns": "fractional", "volatility": "USD/sqrt(second)"},
    })


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Dataset not found: {path}. Collect data and run build-dataset first.")
    frame = pd.read_csv(path, dtype={"market_id": str})
    if frame.empty:
        raise ValueError("Dataset contains no eligible snapshots; inspect its manifest for skip reasons")
    missing = {"market_id", "label", *DATE_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    for name in DATE_COLUMNS:
        if name not in frame:
            raise ValueError(f"Dataset missing required timestamp column: {name}")
        frame[name] = pd.to_datetime([parse_datetime(str(value)) for value in frame[name]], utc=True)
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Dataset labels must be official binary results (0 or 1)")
    if frame.duplicated(["market_id", "timestamp"]).any():
        raise ValueError("Dataset contains duplicate market snapshots")
    if (frame.timestamp >= frame.market_end).any() or (frame.timestamp < frame.market_start).any():
        raise ValueError("Prediction timestamp falls outside the market window")
    if (frame.label_available_at < frame.market_end).any():
        raise ValueError("Settlement label availability precedes market end")
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Dataset hash differs from its provenance manifest; rebuild the dataset")
        frame.attrs["manifest"] = manifest
    return frame.sort_values(["timestamp", "market_id"]).reset_index(drop=True)

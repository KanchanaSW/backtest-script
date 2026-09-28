"""Load Dukascopy CSVs, clean them, and cache the chart timeframe as Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

OHLC = ["open", "high", "low", "close"]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def parse_timestamps(raw: pd.Series) -> pd.Series:
    if np.issubdtype(raw.dtype, np.number):
        sample = raw.dropna()
        if sample.empty:
            return pd.to_datetime(raw, utc=True)
        unit = "ms" if float(sample.iloc[0]) > 10**11 else "s"
        return pd.to_datetime(raw, unit=unit, utc=True)
    return pd.to_datetime(raw, utc=True, format="mixed")


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} is missing a timestamp column")
    missing = [column for column in OHLC if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    out = pd.DataFrame(
        {
            "timestamp": parse_timestamps(frame["timestamp"]),
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0),
        }
    )
    return out


def clean_candles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort, drop bad rows and duplicate timestamps. Do not invent missing bars."""
    stats = {"rows_in": int(len(frame))}
    cleaned = frame.dropna(subset=["timestamp", *OHLC]).copy()
    stats["dropped_na"] = stats["rows_in"] - int(len(cleaned))
    cleaned = cleaned[(cleaned[OHLC] > 0).all(axis=1)]
    cleaned = cleaned[cleaned["high"] >= cleaned["low"]]
    cleaned = cleaned[cleaned["high"] >= cleaned[["open", "close"]].max(axis=1)]
    cleaned = cleaned[cleaned["low"] <= cleaned[["open", "close"]].min(axis=1)]
    stats["dropped_invalid"] = stats["rows_in"] - stats["dropped_na"] - int(len(cleaned))
    cleaned = cleaned.sort_values("timestamp")
    duplicate_count = int(cleaned["timestamp"].duplicated(keep="last").sum())
    cleaned = cleaned.drop_duplicates("timestamp", keep="last")
    stats["dropped_duplicates"] = duplicate_count
    cleaned = cleaned.set_index("timestamp")
    if cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize("UTC")
    else:
        cleaned.index = cleaned.index.tz_convert("UTC")
    stats["rows_out"] = int(len(cleaned))
    if len(cleaned) >= 2:
        gaps = cleaned.index.to_series().diff().dt.total_seconds()
        stats["largest_gap_hours"] = float(gaps.max() / 3600.0)
        # Weekend breaks are expected. Count weekday holes longer than 15 minutes.
        weekday = cleaned.index.dayofweek < 5
        hole = (gaps > 15 * 60) & weekday
        stats["weekday_gaps_over_15m"] = int(hole.sum())
    else:
        stats["largest_gap_hours"] = 0.0
        stats["weekday_gaps_over_15m"] = 0
    return cleaned, stats


def load_raw_directory(raw_dir: Path) -> pd.DataFrame:
    files = sorted(path for path in raw_dir.glob("*.csv") if path.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Run: python scripts/download_data.py"
        )
    frames = [read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = timeframe.strip()
    grouped = frame.resample(rule, label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"])
    grouped = grouped[grouped["volume"].notna()]
    return grouped


def cache_file(processed_dir: Path, instrument: str, timeframe: str) -> Path:
    safe_tf = timeframe.replace("/", "")
    return processed_dir / f"{instrument}_{safe_tf}.parquet"


def cache_is_fresh(raw_dir: Path, parquet_path: Path) -> bool:
    if not parquet_path.exists():
        return False
    cached_at = parquet_path.stat().st_mtime
    files = list(raw_dir.glob("*.csv"))
    if not files:
        return False
    return all(path.stat().st_mtime <= cached_at for path in files)


def load_candles(config: dict, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    data = config["data"]
    raw_dir = project_path(data.get("raw_dir", "data/raw"))
    processed_dir = project_path(data.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    timeframe = str(data.get("chart_timeframe", "5min"))
    instrument = str(data.get("instrument", "xauusd"))
    parquet_path = cache_file(processed_dir, instrument, timeframe)

    if not refresh and cache_is_fresh(raw_dir, parquet_path):
        frame = pd.read_parquet(parquet_path)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        return frame, {"cache": str(parquet_path), "rows": int(len(frame)), "from_cache": True}

    raw, stats = clean_candles(load_raw_directory(raw_dir))
    chart = resample_ohlcv(raw, timeframe)
    chart.to_parquet(parquet_path)
    stats["cache"] = str(parquet_path)
    stats["rows_chart"] = int(len(chart))
    stats["from_cache"] = False
    stats["timeframe"] = timeframe
    return chart, stats


def slice_dates(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    sliced = frame
    if start:
        sliced = sliced[sliced.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        sliced = sliced[sliced.index < pd.Timestamp(end, tz="UTC")]
    return sliced

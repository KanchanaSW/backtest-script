"""Indicator math, exit path, and loader checks."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import math

import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_candles, load_candles, resample_ohlcv
from src.report import compute_metrics, rank_value, sharp_peak_notes
from src.strategy import (
    bar_path,
    bias_allows,
    pine_atr,
    pine_rma,
    pine_sma,
    pivot_high,
    plan_exits,
    protective_stop,
    resolve_params,
    run_backtest,
    update_daily_losses,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_download_module():
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rma_is_seeded_with_sma():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = pine_rma(values, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(8.0 / 3.0)
    assert out[4] == pytest.approx(31.0 / 9.0)


def test_atr_is_nan_until_wilder_can_seed():
    n = 20
    high = np.full(n, 11.0)
    low = np.full(n, 10.0)
    close = np.full(n, 10.5)
    atr = pine_atr(high, low, close, 14)
    assert np.isnan(atr[13])
    assert np.isfinite(atr[14])
    assert atr[14] == pytest.approx(1.0)


def test_sma_matches_trailing_mean():
    values = np.arange(1, 8, dtype=float)
    out = pine_sma(values, 3)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[5] == pytest.approx(5.0)


def test_pivot_high_confirms_after_right_bars():
    high = np.array([1, 2, 3, 2, 1, 2, 5, 4, 3, 2, 1], dtype=float)
    out = pivot_high(high, 2, 2)
    assert out[4] == pytest.approx(3)
    assert out[8] == pytest.approx(5)
    assert np.isnan(out[2])


def test_bar_path_goes_to_the_near_extreme_first():
    assert bar_path(100, 110, 99, 108) == [100, 99, 110, 108]
    assert bar_path(100, 110, 90, 100) == [100, 110, 90, 100]


def test_stop_beats_target_when_the_path_hits_it_first():
    fills = plan_exits(1, 100, 110, 99, 108, sl=99.5, tp1=105, tp2=112, qty1_left=5, qty2_left=5)
    assert [item[1] for item in fills] == ["stop", "stop"]
    assert fills[0][0] == pytest.approx(99.5)


def test_partial_then_runner_on_the_tradingview_path():
    fills = plan_exits(1, 100, 110, 90, 100, sl=95, tp1=105, tp2=112, qty1_left=5, qty2_left=5)
    assert [(item[1], item[2]) for item in fills] == [("tp1", "qty1"), ("stop", "qty2")]
    assert fills[0][0] == pytest.approx(105)
    assert fills[1][0] == pytest.approx(95)


def test_gap_through_stop_fills_at_the_open():
    fills = plan_exits(1, 90, 92, 88, 91, sl=95, tp1=110, tp2=120, qty1_left=5, qty2_left=5)
    assert fills[0][0] == pytest.approx(90)
    assert {item[1] for item in fills} == {"stop"}


def test_daily_loss_counts_once_per_bar_and_resets_on_a_win():
    assert update_daily_losses(0, False, [-10, -5]) == 1
    assert update_daily_losses(1, True, []) == 0
    assert update_daily_losses(2, False, [15]) == 0
    assert update_daily_losses(1, False, [-4, 6]) == 2


def test_download_chunks_are_yearly_and_exclusive():
    download = _load_download_module()
    chunks = download.year_chunks(date(2015, 1, 1), "2017-06-01")
    assert chunks[0] == (date(2015, 1, 1), "2016-01-01")
    assert chunks[1] == (date(2016, 1, 1), "2017-01-01")
    assert chunks[2] == (date(2017, 1, 1), "2017-06-01")


def test_clean_drops_duplicates_and_resamples(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:01:00Z",
                "2024-01-02T00:04:00Z",
                "2024-01-02T00:05:00Z",
            ],
            "open": [10, 11, 12, 13, 14],
            "high": [10, 11, 12, 13, 15],
            "low": [10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 15],
            "volume": [1, 9, 3, 4, 5],
        }
    )
    csv_path = tmp_path / "sample.csv"
    raw.to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path)
    frame.columns = [column.lower() for column in frame.columns]
    from src.data_loader import read_csv

    cleaned, stats = clean_candles(read_csv(csv_path))
    assert stats["dropped_duplicates"] == 1
    assert cleaned.iloc[0]["open"] == pytest.approx(11)
    resampled = resample_ohlcv(cleaned, "5min")
    assert len(resampled) == 2
    assert resampled.iloc[0]["open"] == pytest.approx(11)
    assert resampled.iloc[0]["close"] == pytest.approx(13)
    assert resampled.iloc[0]["volume"] == pytest.approx(16)


def test_loader_caches_parquet(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw_dir.mkdir()
    start = pd.Timestamp("2024-01-02T00:00:00Z")
    rows = []
    for i in range(10):
        price = 100 + i
        rows.append(
            {
                "timestamp": (start + pd.Timedelta(minutes=i)).isoformat(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 10 + i,
            }
        )
    pd.DataFrame(rows).to_csv(raw_dir / "xau.csv", index=False)
    config = {
        "data": {
            "instrument": "xauusd",
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed),
            "chart_timeframe": "5min",
        }
    }
    first, stats = load_candles(config)
    second, cached = load_candles(config)
    assert stats["from_cache"] is False
    assert cached["from_cache"] is True
    assert len(first) == len(second) == 2


def _params(**overrides):
    config = {
        "strategy": {
            "account_size": 5000.0,
            "lot_size": 0.1,
            "units_per_lot": 100.0,
            "atr_sl_mult": 2.0,
            "tp1_ratio": 1.0,
            "tp2_ratio": 2.0,
            "scale_pct": 50,
            "use_trail": True,
            "max_daily_loss": 2,
            "close_friday": True,
            "use_session": False,
            "trading_hours": "0700-1200",
            "session_tz": "America/New_York",
            "use_vol": False,
            "vol_length": 2,
            "htf": "60",
            "swing_len": 1,
            "fvg_timeout": 5,
            "show_boxes": False,
        },
        "broker": {"spread": 0.0, "slippage": 0.0, "commission_per_unit": 0.0, "margin_pct": 1.0},
        "fixed": {"atr_length": 3, "trail_atr_mult": 1.5, "chart_timezone": "UTC"},
    }
    params = resolve_params(config)
    params.update(overrides)
    return params


def _frame(rows: list[tuple]) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02T00:00:00Z")
    records = []
    for i, (o, h, l, c) in enumerate(rows):
        records.append(
            {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 100.0,
            }
        )
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(records, index=index)


def test_bias_and_sweep_stop_rules():
    assert bias_allows(1, 100, 90, "with")
    assert not bias_allows(1, 80, 90, "with")
    assert bias_allows(-1, 80, 90, "with")
    assert bias_allows(1, 80, 90, "against")
    assert bias_allows(1, 100, float("nan"), "")
    assert protective_stop(1, 100, 90, 95, "atr", 1) == pytest.approx(90)
    assert protective_stop(1, 100, 90, 95, "sweep", 1) == pytest.approx(94)
    assert math.isnan(protective_stop(1, 100, 90, 101, "sweep", 0))


def test_entry_fills_on_the_next_open_and_scales_once():
    # 24 five-minute bars make two complete hours. The next hour can sweep the
    # previous hour's low without trading through its high.
    quiet = [(105, 110, 100, 105) for _ in range(24)]
    pattern = [
        (105, 106, 98, 104),    # bullish sweep of 100
        (104, 107, 103, 106),   # pivot high at 107, confirmed next bar
        (106, 105.5, 103, 104),
        (104, 109, 104, 108),   # close crosses the pivot
        (108, 109, 106, 108.5), # bullish FVG: low above high[2]
        (108.5, 109, 107, 108), # market entry fills on this open
        (108, 109, 70, 72),     # stop the position so the trade is closed
    ]
    frame = _frame(quiet + pattern)
    result = run_backtest(frame, _params())
    trades = result["trades"]
    assert trades is not None and not trades.empty
    first = trades.iloc[0]
    assert first["signal_time"] == pd.Timestamp("2024-01-02 02:20:00", tz="UTC")
    assert first["entry_time"] == pd.Timestamp("2024-01-02 02:25:00", tz="UTC")
    assert (trades["reason"] == "tp1").sum() <= 1


def test_sharp_peak_ignores_neighbors_with_no_trades():
    def row(atr, sharpe, trades, score):
        return {
            "params": {"atr_sl_mult": atr},
            "score": score,
            "metrics": {"sharpe": sharpe, "trades": trades},
        }

    results = [
        row(1.5, 2.0, 40, 2.0),
        row(2.0, 1.8, 40, 1.8),
        row(2.5, 0.0, 0, -1e18),
    ]
    notes = sharp_peak_notes(results, {"atr_sl_mult": [1.5, 2.0, 2.5]}, 1.5)
    assert not any("sharp peak" in note for note in notes)


def test_rank_rejects_thin_samples():
    metrics = {"trades": 3, "sharpe": 4.0, "profit_factor": 2.0}
    assert rank_value(metrics, "sharpe", 30) < -1e17
    metrics["trades"] = 40
    assert rank_value(metrics, "sharpe", 30) == pytest.approx(4.0)

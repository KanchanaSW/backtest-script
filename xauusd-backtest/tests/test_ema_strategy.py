"""Tests for the 20/50 EMA + pro management strategy module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ema_strategy import resolve_ema_params, run_ema_backtest
from src.scalper_strategy import pine_ema


def test_resolve_defaults():
    params = resolve_ema_params({"strategy": {}, "broker": {}, "fixed": {}})
    assert params["fast_len"] == 20
    assert params["slow_len"] == 50
    assert params["trend_len"] == 200
    assert params["atr_mult"] == 2.0
    assert params["tp1_ratio"] == 0.8
    assert params["tp2_ratio"] == 2.0
    assert params["scale_pct"] == 50
    assert params["use_trail"] is True
    assert params["trail_atr_mult"] == 1.0
    assert params["be_offset"] == 0.30
    assert params["use_session"] is True
    assert params["trading_hours"] == "0300-1600"
    assert params["close_friday"] is True


def _params(**overrides):
    base = resolve_ema_params(
        {
            "strategy": {
                "account_size": 10000.0,
                "qty_pct": 10.0,
                "use_session": False,
                "close_friday": False,
            },
            "broker": {"spread": 0.0, "slippage": 0.0, "commission_per_unit": 0.0},
            "fixed": {},
        }
    )
    base.update(overrides)
    return base


def test_long_scales_at_tp1_then_be_stop():
    """Synthetic: pullback then resume, hit TP1 on the scaled leg."""
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.empty(n)
    close[:100] = np.linspace(1900, 2050, 100)
    close[100:160] = np.linspace(2050, 2000, 60)
    close[160:] = np.linspace(2000, 2200, n - 160)
    high = close + 3.0
    low = close - 3.0
    # Force a TP1 touch a few bars after the first valid long entry path
    high[188] = 2040.0
    frame = pd.DataFrame(
        {"open": close.copy(), "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )
    result = run_ema_backtest(
        frame,
        _params(
            atr_mult=1.0,
            tp1_ratio=1.0,
            tp2_ratio=5.0,
            scale_pct=50,
            use_trail=True,
            be_offset=0.3,
        ),
    )
    reasons = result["trades"]["reason"].tolist() if result["trades"] is not None else []
    assert "tp1" in reasons


def test_no_entry_when_against_trend_ema():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(2200, 2000, n)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    result = run_ema_backtest(frame, _params())
    sides = [] if result["trades"] is None else result["trades"]["side"].tolist()
    assert "long" not in sides


def test_backtest_runs_on_synthetic_trend():
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(2000.0, 2200.0, n) + np.sin(np.linspace(0, 20, n)) * 5
    high = close + 1.5
    low = close - 1.5
    open_ = close - 0.2
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )
    result = run_ema_backtest(frame, _params())
    assert result["initial_capital"] == 10000.0
    assert isinstance(result["final_equity"], float)
    assert np.isfinite(pine_ema(close, 50)[-1])

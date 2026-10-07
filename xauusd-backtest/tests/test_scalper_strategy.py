"""Tests for the Micro-FVG scalper strategy v3 module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scalper_strategy import (
    calc_sl_distance,
    pine_ema,
    pine_rsi,
    pine_session_vwap,
    plan_scalp_exits,
    resolve_scalper_params,
    run_scalper_backtest,
)


# ── pine_ema tests ─────────────────────────────────────────────────

def test_ema_single_value():
    vals = np.array([100.0])
    result = pine_ema(vals, 3)
    assert result[0] == 100.0


def test_ema_constant_series():
    vals = np.full(50, 42.0)
    result = pine_ema(vals, 10)
    np.testing.assert_allclose(result[-1], 42.0, atol=1e-10)


def test_ema_trending():
    vals = np.arange(1.0, 101.0)
    fast = pine_ema(vals, 5)
    slow = pine_ema(vals, 20)
    assert abs(fast[-1] - vals[-1]) < abs(slow[-1] - vals[-1])


# ── pine_rsi tests ─────────────────────────────────────────────────

def test_rsi_trending_up():
    vals = np.arange(100.0, 200.0, 1.0)
    result = pine_rsi(vals, 14)
    assert result[-1] > 70.0


def test_rsi_trending_down():
    vals = np.arange(200.0, 100.0, -1.0)
    result = pine_rsi(vals, 14)
    assert result[-1] < 30.0


def test_rsi_range_is_0_to_100():
    rng = np.random.RandomState(42)
    vals = 100.0 + np.cumsum(rng.randn(200))
    result = pine_rsi(vals, 14)
    finite = result[np.isfinite(result)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 100.0)


# ── pine_session_vwap tests ────────────────────────────────────────

def test_vwap_flat():
    day_ids = np.zeros(10, dtype=int)
    high = np.full(10, 101.0)
    low = np.full(10, 99.0)
    close = np.full(10, 100.0)
    volume = np.full(10, 100.0)
    vwap = pine_session_vwap(day_ids, high, low, close, volume)
    np.testing.assert_allclose(vwap, 100.0)


# ── calc_sl_distance tests ─────────────────────────────────────────

def test_sl_dynamic_atr():
    assert calc_sl_distance(5.0, "dynamic_atr", 1.0, 8.0) == 5.0
    assert calc_sl_distance(5.0, "dynamic_atr", 2.0, 8.0) == 10.0


def test_sl_fixed_pips():
    assert calc_sl_distance(5.0, "fixed_pips", 1.0, 6.0) == 6.0


def test_sl_tighter():
    # ATR distance (5*0.85 = 4.25) < fixed (6) -> 4.25
    assert calc_sl_distance(5.0, "tighter", 0.85, 6.0) == 4.25
    # ATR distance (5*2 = 10) > fixed (6) -> 6.0
    assert calc_sl_distance(5.0, "tighter", 2.0, 6.0) == 6.0


# ── plan_scalp_exits tests ─────────────────────────────────────────

def test_exit_tp1_hit_long():
    result = plan_scalp_exits(1, 100.0, 108.0, 99.0, 105.0,
                              sl=95.0, tp1=107.0, tp2=112.0,
                              qty1_left=6.0, qty2_left=4.0)
    assert any(r[1] == "tp1" and r[2] == "qty1" for r in result)


def test_exit_sl_hit_kills_all():
    result = plan_scalp_exits(1, 100.0, 101.0, 93.0, 94.0,
                              sl=95.0, tp1=107.0, tp2=112.0,
                              qty1_left=6.0, qty2_left=4.0)
    reasons = [r[1] for r in result]
    assert "stop" in reasons


# ── resolve_scalper_params tests ───────────────────────────────────

def test_v3_defaults():
    params = resolve_scalper_params({})
    assert params["tp_pips"] == 5.0
    assert params["sl_mode"] == "tighter"
    assert params["sl_atr_mult"] == 1.3
    assert params["sl_fixed_pips"] == 7.0
    assert params["max_daily_losses"] == 3
    assert params["use_early_be"] is True
    assert params["min_rr"] == 0.5


# ── Integration: run_scalper_backtest ──────────────────────────────

def _make_frame(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base = 2000.0
    close = base + np.cumsum(rng.randn(n) * 2.0)
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    open_ = close + rng.randn(n) * 1.0
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(100, 5000, n)
    idx = pd.date_range("2024-01-02 08:00", periods=n, freq="3min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_backtest_runs_v3():
    frame = _make_frame(300)
    params = resolve_scalper_params({})
    params["use_session"] = False
    params["use_htf_trend"] = False
    params["use_vwap_filter"] = False
    result = run_scalper_backtest(frame, params)
    assert "final_equity" in result
    assert "pnls" in result
    assert result["final_equity"] > 0


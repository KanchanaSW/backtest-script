"""Micro-FVG scalping strategy v3 and bar-by-bar backtest simulator.

This is the Python equivalent of xauusd_m3_scalper.pine v3.

v3 improvements over v2:
~~~~~~~~~~~~~~~~~~~~~~~~
- Early Break-Even trigger at 50% TP1 (stops near-misses from becoming full losses)
- Tighter SL ceiling ($6.00) and ATR multiplier (0.85) to cut loss magnitude
- Session VWAP filter (intraday institutional volume positioning)
- M15 Dual EMA (20/50) trend confirmation (eliminates counter-trend pullbacks)
- Entry candle rejection wick filter (rejects candles with large opposite wicks)
- Minimum EMA momentum spread (avoids dead squeeze / consolidation fakeouts)
- Strict daily circuit breaker (max 2 losses/day, 15-bar cooldown)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Re-use proven helpers from the existing strategy module
from src.strategy import (
    bar_path,
    execution_price,
    pine_atr,
    pine_rma,
    pine_sma,
    session_mask,
)


# ── Pine-compatible EMA ────────────────────────────────────────────
def pine_ema(values: np.ndarray, length: int) -> np.ndarray:
    """EMA seeded with the first finite value, matching ta.ema in Pine."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if length <= 0:
        return out
    alpha = 2.0 / (length + 1)
    prev = np.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        if not np.isfinite(prev):
            prev = v
            out[i] = prev
            continue
        prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


# ── Pine-compatible RSI ────────────────────────────────────────────
def pine_rsi(values: np.ndarray, length: int) -> np.ndarray:
    """RSI matching ta.rsi in Pine (uses RMA for smoothing)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if length <= 0 or n < 2:
        return out
    delta = np.diff(values)
    delta = np.concatenate(([np.nan], delta))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = np.nan
    loss[0] = np.nan
    avg_gain = pine_rma(gain, length)
    avg_loss = pine_rma(loss, length)
    for i in range(n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if not np.isfinite(ag) or not np.isfinite(al):
            continue
        if al == 0:
            out[i] = 100.0
        else:
            rs = ag / al
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


# ── HTF EMA (non-repainting) ──────────────────────────────────────
def htf_trend_signals(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    htf_timeframe: str,
    ema_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (htf_close, htf_ema) arrays aligned to the LTF index.

    Uses the *previous completed* HTF bar to avoid repainting.
    """
    from src.strategy import pine_timeframe_to_rule

    series = pd.Series(close, index=index)
    rule = pine_timeframe_to_rule(htf_timeframe)
    grouped = series.resample(rule, label="left", closed="left", origin="epoch").last()
    htf_close_series = grouped.shift(1)  # previous completed bar
    htf_ema_raw = grouped.ewm(span=ema_length, adjust=False).mean().shift(1)
    htf_close_out = htf_close_series.reindex(index, method="ffill").to_numpy(dtype=float)
    htf_ema_out = htf_ema_raw.reindex(index, method="ffill").to_numpy(dtype=float)
    return htf_close_out, htf_ema_out


# ── Session VWAP (ta.vwap equivalent) ──────────────────────────────
def pine_session_vwap(
    day_ids: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Intraday cumulative volume-weighted average price per session/day."""
    n = len(close)
    vwap_out = np.copy(close)
    if n == 0:
        return vwap_out
    tp = (high + low + close) / 3.0
    c_vol = 0.0
    c_tpv = 0.0
    cur_day = day_ids[0]
    for i in range(n):
        if day_ids[i] != cur_day:
            cur_day = day_ids[i]
            c_vol = 0.0
            c_tpv = 0.0
        v = volume[i] if np.isfinite(volume[i]) and volume[i] > 0 else 1.0
        c_vol += v
        c_tpv += tp[i] * v
        vwap_out[i] = c_tpv / c_vol if c_vol > 0 else close[i]
    return vwap_out


# ── Parameter resolver ─────────────────────────────────────────────
SCALPER_INPUT_ORDER = [
    "account_size",
    "lot_size",
    "units_per_lot",
    "min_body_pct",
    "fvg_atr_min",
    "tp_pips",
    "sl_mode",
    "sl_atr_mult",
    "sl_fixed_pips",
    "atr_length",
    "entry_confirm",
    "max_wick_pct",
    "use_early_be",
    "be_trigger_pct",
    "trail_be_offset",
    "use_partial",
    "scale_pct",
    "tp2_atr_mult",
    "use_trail_scalp",
    "trail_atr_mult",
    "use_vwap_filter",
    "use_htf_trend",
    "htf_tf",
    "htf_ema_fast_len",
    "htf_ema_slow_len",
    "htf_trend_mode",
    "use_ema_filter",
    "ema_fast_len",
    "ema_slow_len",
    "min_ema_spread",
    "use_rsi_filter",
    "rsi_length",
    "rsi_ob",
    "rsi_os",
    "use_vol_spike",
    "vol_sma_len",
    "vol_spike_mult",
    "use_atr_floor",
    "atr_floor_mult",
    "max_daily_losses",
    "max_daily_trades",
    "close_friday",
    "min_rr",
    "use_cooldown",
    "cooldown_bars",
    "use_session",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_tz",
]


def resolve_scalper_params(config: dict, overrides: dict | None = None) -> dict:
    """Merge fixed / broker / strategy sections into a flat parameter dict."""
    params: dict = {}
    params.update(config.get("fixed") or {})
    params.update(config.get("broker") or {})
    params.update(config.get("strategy") or {})
    if overrides:
        params.update(overrides)

    # Defaults matching the v3.1 Pine script
    params.setdefault("account_size", 5000.0)
    params.setdefault("lot_size", 0.1)
    params.setdefault("units_per_lot", 100.0)
    params.setdefault("min_body_pct", 45.0)
    params.setdefault("fvg_atr_min", 0.25)
    params.setdefault("tp_pips", 5.0)
    params.setdefault("sl_mode", "tighter")
    params.setdefault("sl_atr_mult", 1.3)        # v3.1: 1.3 ATR breathing room
    params.setdefault("sl_fixed_pips", 7.0)      # v3.1: $7.00 ceiling
    params.setdefault("atr_length", 14)
    params.setdefault("entry_confirm", True)
    params.setdefault("max_wick_pct", 50.0)      # v3.1: accommodates Gold wicks

    # Loss Protection & Early Break-Even
    params.setdefault("use_early_be", True)      # v3.1: early BE at 45-50% TP1
    params.setdefault("be_trigger_pct", 45.0)
    params.setdefault("trail_be_offset", 0.5)

    # Partial scale-out
    params.setdefault("use_partial", True)
    params.setdefault("scale_pct", 60)
    params.setdefault("tp2_atr_mult", 2.5)
    params.setdefault("use_trail_scalp", True)
    params.setdefault("trail_atr_mult", 0.6)

    # Institutional VWAP filter (optional)
    params.setdefault("use_vwap_filter", False)

    # HTF Trend
    params.setdefault("use_htf_trend", True)
    params.setdefault("htf_tf", "15")
    params.setdefault("htf_ema_fast_len", 20)
    params.setdefault("htf_ema_slow_len", 50)
    params.setdefault("htf_trend_mode", "single") # v3.1: single EMA to avoid lag

    # Momentum & Squeeze
    params.setdefault("use_ema_filter", True)
    params.setdefault("ema_fast_len", 8)
    params.setdefault("ema_slow_len", 21)
    params.setdefault("min_ema_spread", 0.0)

    # RSI
    params.setdefault("use_rsi_filter", True)
    params.setdefault("rsi_length", 14)
    params.setdefault("rsi_ob", 70.0)
    params.setdefault("rsi_os", 30.0)

    # Volume
    params.setdefault("use_vol_spike", True)
    params.setdefault("vol_sma_len", 20)
    params.setdefault("vol_spike_mult", 1.1)

    # ATR Floor
    params.setdefault("use_atr_floor", True)
    params.setdefault("atr_floor_mult", 0.4)

    # Risk & Limits
    params.setdefault("max_daily_losses", 3)
    params.setdefault("max_daily_trades", 15)
    params.setdefault("close_friday", True)
    params.setdefault("min_rr", 0.5)

    # Cooldown
    params.setdefault("use_cooldown", True)
    params.setdefault("cooldown_bars", 8)

    # Session
    params.setdefault("use_session", False)      # 24h active scalping
    params.setdefault("session_london", "0300-0500")
    params.setdefault("session_ny", "0830-1100")
    params.setdefault("session_overlap", "0700-0830")
    params.setdefault("session_tz", "America/New_York")

    # Broker
    params.setdefault("spread", 0.0)
    params.setdefault("slippage", 0.0)
    params.setdefault("commission_per_unit", 0.0)
    params.setdefault("margin_pct", 10.0)
    params.setdefault("chart_timezone", "UTC")
    return params


# ── Stop-loss distance ─────────────────────────────────────────────
def calc_sl_distance(atr: float, mode: str, atr_mult: float, fixed_pips: float) -> float:
    """Return the SL distance matching the Pine calc_sl_distance()."""
    atr_sl = atr * atr_mult
    if mode == "fixed_pips":
        return fixed_pips
    if mode == "tighter":
        return min(atr_sl, fixed_pips)
    return atr_sl  # dynamic_atr (default)


# ── Exit planner (supports partial scale-out) ──────────────────────
def plan_scalp_exits(
    side: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    sl: float,
    tp1: float,
    tp2: float,
    qty1_left: float,
    qty2_left: float,
) -> list[tuple[float, str, str]]:
    """Return list of (chart_price, reason, leg) for exits hit on this bar.

    leg is 'qty1', 'qty2', or 'all'.
    """
    fills: list[tuple[float, str, str]] = []
    q1 = qty1_left
    q2 = qty2_left

    def add(price: float, reason: str, leg: str) -> None:
        nonlocal q1, q2
        if leg in {"qty1", "all"} and q1 > 1e-12:
            fills.append((price, reason, "qty1"))
            q1 = 0.0
        if leg in {"qty2", "all"} and q2 > 1e-12:
            fills.append((price, reason, "qty2"))
            q2 = 0.0

    # Gap open past stops/targets
    if side == 1:
        if open_ <= sl:
            add(open_, "stop", "all")
            return fills
        if open_ >= tp1:
            add(open_, "tp1", "qty1")
        if open_ >= tp2:
            add(open_, "tp2", "qty2")
    else:
        if open_ >= sl:
            add(open_, "stop", "all")
            return fills
        if open_ <= tp1:
            add(open_, "tp1", "qty1")
        if open_ <= tp2:
            add(open_, "tp2", "qty2")

    if q1 <= 1e-12 and q2 <= 1e-12:
        return fills

    path = bar_path(open_, high, low, close)
    for left, right in zip(path, path[1:]):
        if q1 <= 1e-12 and q2 <= 1e-12:
            break
        rising = right > left
        hits: list[tuple[float, str, str]] = []
        if side == 1:
            if not rising and right <= sl < left:
                hits.append((sl, "stop", "all"))
            if rising and q1 > 1e-12 and left < tp1 <= right:
                hits.append((tp1, "tp1", "qty1"))
            if rising and q2 > 1e-12 and left < tp2 <= right:
                hits.append((tp2, "tp2", "qty2"))
        else:
            if rising and left < sl <= right:
                hits.append((sl, "stop", "all"))
            if not rising and q1 > 1e-12 and right <= tp1 < left:
                hits.append((tp1, "tp1", "qty1"))
            if not rising and q2 > 1e-12 and right <= tp2 < left:
                hits.append((tp2, "tp2", "qty2"))
        hits.sort(key=lambda item: item[0], reverse=not rising)
        for price, reason, leg in hits:
            add(price, reason, leg)
            if reason == "stop":
                return fills
    return fills


# ── Daily loss / trade counter ─────────────────────────────────────
def _update_daily(
    daily_losses: int,
    daily_trades: int,
    consec_losses: int,
    bars_since_loss: int,
    new_day: bool,
    pnl: float | None,
) -> tuple[int, int, int, int]:
    if new_day:
        daily_losses = 0
        daily_trades = 0
    if pnl is not None:
        daily_trades += 1
        if pnl < 0:
            daily_losses += 1
            consec_losses += 1
            bars_since_loss = 0
        elif pnl > 0:
            daily_losses = 0
            consec_losses = 0
    return daily_losses, daily_trades, consec_losses, bars_since_loss


# ── Main backtest loop ─────────────────────────────────────────────
def run_scalper_backtest(
    frame: pd.DataFrame,
    params: dict,
    *,
    collect_trades: bool = True,
    collect_equity: bool = True,
    trade_after: pd.Timestamp | None = None,
) -> dict:
    """Bar-by-bar scalping backtest v3."""

    if frame.empty:
        raise ValueError("No candles to backtest")

    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    index = frame.index
    n = len(frame)

    # ── Pre-compute indicators ──────────────────────────────
    atr_len = int(params["atr_length"])
    atr = pine_atr(high, low, close, atr_len)
    ema_fast = pine_ema(close, int(params["ema_fast_len"]))
    ema_slow = pine_ema(close, int(params["ema_slow_len"]))
    vol_sma = pine_sma(volume, int(params["vol_sma_len"]))
    rsi = pine_rsi(close, int(params.get("rsi_length", 14)))

    local = index.tz_convert(str(params.get("chart_timezone", "UTC")))
    hours = local.hour.to_numpy()
    dows = local.dayofweek.to_numpy()
    day_ids = local.normalize().asi8

    # Session VWAP
    use_vwap = bool(params.get("use_vwap_filter", True))
    vwap_arr = pine_session_vwap(day_ids, high, low, close, volume)

    # HTF trend (Dual EMA)
    use_htf = bool(params.get("use_htf_trend", True))
    htf_mode = str(params.get("htf_trend_mode", "double")).lower()
    if use_htf:
        htf_close_arr, htf_ema_slow_arr = htf_trend_signals(
            index, close, str(params.get("htf_tf", "15")), int(params.get("htf_ema_slow_len", 50))
        )
        _, htf_ema_fast_arr = htf_trend_signals(
            index, close, str(params.get("htf_tf", "15")), int(params.get("htf_ema_fast_len", 20))
        )
    else:
        htf_close_arr = np.full(n, np.nan)
        htf_ema_slow_arr = np.full(n, np.nan)
        htf_ema_fast_arr = np.full(n, np.nan)

    # ── Session masks ───────────────────────────────────────
    tz = str(params["session_tz"])
    use_session = bool(params["use_session"])
    mask_london = session_mask(index, str(params["session_london"]), tz, use_session)
    mask_ny = session_mask(index, str(params["session_ny"]), tz, use_session)
    mask_overlap = session_mask(index, str(params["session_overlap"]), tz, use_session)
    if use_session:
        in_session = mask_london | mask_ny | mask_overlap
    else:
        in_session = np.ones(n, dtype=bool)

    friday_block = np.zeros(n, dtype=bool)
    if bool(params["close_friday"]):
        friday_block = (dows == 4) & (hours >= 16)

    if trade_after is None:
        allowed = np.ones(n, dtype=bool)
    else:
        cutoff = pd.Timestamp(trade_after)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        allowed = index >= cutoff

    # ── Extract scalar params ───────────────────────────────
    spread = float(params["spread"])
    slippage = float(params["slippage"])
    commission = float(params["commission_per_unit"])
    margin_pct = float(params["margin_pct"])
    tp_pips = float(params["tp_pips"])
    sl_mode = str(params["sl_mode"]).lower().replace(" ", "_")
    sl_atr_mult = float(params["sl_atr_mult"])
    sl_fixed_pips = float(params["sl_fixed_pips"])
    use_trail = bool(params["use_trail_scalp"])
    trail_be_offset = float(params["trail_be_offset"])
    trail_atr_m = float(params.get("trail_atr_mult", 0.6))
    use_early_be = bool(params.get("use_early_be", True))
    be_trigger_pct = float(params.get("be_trigger_pct", 50.0))
    min_body_pct = float(params["min_body_pct"])
    fvg_atr_min = float(params["fvg_atr_min"])
    entry_confirm = bool(params.get("entry_confirm", True))
    max_wick_pct = float(params.get("max_wick_pct", 35.0))
    min_ema_spread = float(params.get("min_ema_spread", 0.12))
    use_ema = bool(params["use_ema_filter"])
    use_rsi = bool(params.get("use_rsi_filter", True))
    rsi_ob = float(params.get("rsi_ob", 70.0))
    rsi_os = float(params.get("rsi_os", 30.0))
    use_vol = bool(params["use_vol_spike"])
    vol_mult = float(params["vol_spike_mult"])
    use_atr_floor = bool(params.get("use_atr_floor", True))
    atr_floor_mult = float(params.get("atr_floor_mult", 0.5))
    max_daily_loss = int(params["max_daily_losses"])
    max_daily_trade = int(params["max_daily_trades"])
    min_rr = float(params["min_rr"])
    use_cooldown = bool(params.get("use_cooldown", True))
    cooldown_bars_limit = int(params.get("cooldown_bars", 15))
    use_partial = bool(params.get("use_partial", True))
    scale_pct = int(params.get("scale_pct", 60))
    tp2_atr_mult_param = float(params.get("tp2_atr_mult", 2.5))
    actual_qty = float(params["lot_size"]) * float(params["units_per_lot"])
    initial = float(params["account_size"])

    # ── State variables ─────────────────────────────────────
    cash = initial
    side = 0
    pos_qty1 = 0.0
    pos_qty2 = 0.0
    entry_fill = 0.0
    entry_ref = 0.0
    sl = np.nan
    tp1 = np.nan
    tp2 = np.nan
    signal_i = -1
    entry_i = -1
    tp1_hit = False
    be_activated = False
    trailing = False
    daily_losses = 0
    daily_trades = 0
    consec_losses = 0
    bars_since_loss = 100
    pending: dict | None = None
    pending_flatten = False

    trades: list[dict] = []
    pnls: list[float] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    daily_equity: list[float] = []
    peak = initial
    max_dd = 0.0
    max_dd_pct = 0.0
    day_cursor = None
    equity_at_day = initial

    def _round_trip(exit_fill: float, qty: float) -> float:
        gross = (exit_fill - entry_fill) * qty if side == 1 else (entry_fill - exit_fill) * qty
        return gross - commission * qty * 2.0

    def _mark(c: float) -> float:
        qty_open = pos_qty1 + pos_qty2
        if side == 0 or qty_open <= 1e-12:
            return cash
        fee = commission * qty_open * 2.0
        if side == 1:
            return cash + (c - entry_fill) * qty_open - fee
        return cash + (entry_fill - (c + spread)) * qty_open - fee

    def record_exit(bar: int, leg_qty: float, chart_price: float, reason: str) -> float:
        nonlocal cash
        exit_fill = execution_price(side, chart_price, False, spread, slippage)
        pnl = _round_trip(exit_fill, leg_qty)
        cash += pnl
        pnls.append(pnl)
        if collect_trades:
            trades.append({
                "signal_time": index[signal_i],
                "entry_time": index[entry_i],
                "exit_time": index[bar],
                "side": "long" if side == 1 else "short",
                "qty": leg_qty,
                "entry_price": entry_fill,
                "exit_price": exit_fill,
                "pnl": pnl,
                "reason": reason,
            })
        return pnl

    # ── Main loop ───────────────────────────────────────────
    for i in range(n):
        bar_pnls: list[float] = []

        # Weekend flatten
        if pending_flatten and side != 0:
            for leg_name, leg_qty in (("qty1", pos_qty1), ("qty2", pos_qty2)):
                if leg_qty > 1e-12:
                    bar_pnls.append(record_exit(i, leg_qty, open_[i], "friday"))
                    if leg_name == "qty1":
                        pos_qty1 = 0.0
                    else:
                        pos_qty2 = 0.0
            side = 0
            tp1_hit = False
            be_activated = False
            trailing = False
        pending_flatten = False

        # Fill pending entry
        if pending is not None:
            if side != 0:
                for leg_qty in (pos_qty1, pos_qty2):
                    if leg_qty > 1e-12:
                        bar_pnls.append(record_exit(i, leg_qty, open_[i], "reverse"))
                pos_qty1 = 0.0
                pos_qty2 = 0.0
                side = 0
                tp1_hit = False
                be_activated = False
                trailing = False
            if side == 0:
                entry_px = execution_price(pending["side"], open_[i], True, spread, slippage)
                equity_now = _mark(open_[i])
                required = abs(entry_px) * pending["qty"] * margin_pct / 100.0
                if equity_now >= required:
                    side = pending["side"]
                    pos_qty1 = pending["qty1"]
                    pos_qty2 = pending["qty2"]
                    entry_fill = entry_px
                    entry_ref = pending["entry_ref"]
                    sl = pending["sl"]
                    tp1 = pending["tp1"]
                    tp2 = pending["tp2"]
                    signal_i = pending["signal_i"]
                    entry_i = i
                    tp1_hit = False
                    be_activated = False
                    trailing = False
            pending = None

        # Exit logic on the current bar
        if side != 0 and (pos_qty1 > 1e-12 or pos_qty2 > 1e-12):
            planned = plan_scalp_exits(
                side, open_[i], high[i], low[i], close[i],
                sl, tp1, tp2, pos_qty1, pos_qty2,
            )
            for chart_price, reason, leg in planned:
                leg_qty = pos_qty1 if leg == "qty1" else pos_qty2
                if leg_qty <= 1e-12:
                    continue
                bar_pnls.append(record_exit(i, leg_qty, chart_price, reason))
                if leg == "qty1":
                    pos_qty1 = 0.0
                else:
                    pos_qty2 = 0.0
            if pos_qty1 <= 1e-12 and pos_qty2 <= 1e-12:
                side = 0
                tp1_hit = False
                be_activated = False
                trailing = False

        # Equity tracking
        equity = _mark(close[i])
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak if peak else 0.0

        if allowed[i]:
            if collect_equity:
                equity_index.append(index[i])
                equity_values.append(equity)
            if day_cursor is None:
                day_cursor = day_ids[i]
            elif day_ids[i] != day_cursor:
                daily_equity.append(equity_at_day)
                day_cursor = day_ids[i]
            equity_at_day = equity

        # Daily counters — use worst pnl on this bar
        new_day = i > 0 and day_ids[i] != day_ids[i - 1]
        worst_pnl = min(bar_pnls) if bar_pnls else None
        daily_losses, daily_trades, consec_losses, bars_since_loss = _update_daily(
            daily_losses, daily_trades, consec_losses, bars_since_loss, new_day, worst_pnl,
        )
        bars_since_loss += 1

        in_cooldown = use_cooldown and consec_losses >= 2 and bars_since_loss < cooldown_bars_limit
        can_trade = daily_losses < max_daily_loss and daily_trades < max_daily_trade and not in_cooldown

        # Friday flatten trigger
        if side != 0 and bool(params["close_friday"]) and friday_block[i]:
            pending_flatten = True

        # ── Early BE & Trailing stop management (v3) ────────
        be_dist = tp_pips * (be_trigger_pct / 100.0)
        if side == 1 and (pos_qty1 > 1e-12 or pos_qty2 > 1e-12):
            # 1. Early BE at 50% TP1
            if use_early_be and not be_activated and not tp1_hit and high[i] >= (entry_ref + be_dist):
                be_activated = True
                sl = max(sl, entry_ref + trail_be_offset)
            # 2. TP1 hit
            if not tp1_hit and high[i] >= tp1:
                tp1_hit = True
                be_activated = True
                trailing = True
                sl = max(sl, entry_ref + trail_be_offset)
            # 3. Runner trailing
            if trailing and use_trail and np.isfinite(atr[i]):
                new_trail = high[i] - atr[i] * trail_atr_m
                if new_trail > sl:
                    sl = new_trail
        elif side == -1 and (pos_qty1 > 1e-12 or pos_qty2 > 1e-12):
            # 1. Early BE at 50% TP1
            if use_early_be and not be_activated and not tp1_hit and low[i] <= (entry_ref - be_dist):
                be_activated = True
                sl = min(sl, entry_ref - trail_be_offset)
            # 2. TP1 hit
            if not tp1_hit and low[i] <= tp1:
                tp1_hit = True
                be_activated = True
                trailing = True
                sl = min(sl, entry_ref - trail_be_offset)
            # 3. Runner trailing
            if trailing and use_trail and np.isfinite(atr[i]):
                new_trail = low[i] + atr[i] * trail_atr_m
                if new_trail < sl:
                    sl = new_trail

        # ── Signal generation ───────────────────────────────
        if i < 2:
            continue
        if not (allowed[i] and side == 0 and can_trade and in_session[i]
                and not friday_block[i] and np.isfinite(atr[i]) and atr[i] > 0):
            continue

        # ATR volatility floor
        if use_atr_floor:
            atr_floor = close[i] * atr_floor_mult / 10000.0
            if atr[i] < atr_floor:
                continue

        # Micro-FVG detection
        mid_range = high[i - 1] - low[i - 1]
        mid_body = abs(close[i - 1] - open_[i - 1])
        body_pct = (mid_body / mid_range * 100.0) if mid_range > 0 else 0.0
        strong_body = body_pct >= min_body_pct

        bull_gap = low[i] - high[i - 2]
        bear_gap = low[i - 2] - high[i]
        min_gap = fvg_atr_min * atr[i]

        # Rejection wick check on current candle
        c_range = high[i] - low[i]
        u_wick = high[i] - max(open_[i], close[i])
        l_wick = min(open_[i], close[i]) - low[i]
        long_wick_ok = (u_wick / c_range * 100.0 <= max_wick_pct) if c_range > 0 else True
        short_wick_ok = (l_wick / c_range * 100.0 <= max_wick_pct) if c_range > 0 else True

        # Entry candle confirmation
        entry_bull_ok = (not entry_confirm) or (close[i] > open_[i])
        entry_bear_ok = (not entry_confirm) or (close[i] < open_[i])

        fvg_bull = (low[i] > high[i - 2]
                    and close[i - 1] > open_[i - 1]
                    and strong_body
                    and bull_gap >= min_gap
                    and entry_bull_ok
                    and long_wick_ok)
        fvg_bear = (high[i] < low[i - 2]
                    and close[i - 1] < open_[i - 1]
                    and strong_body
                    and bear_gap >= min_gap
                    and entry_bear_ok
                    and short_wick_ok)

        # Momentum & Spread filter
        if use_ema:
            ema_spread = abs(ema_fast[i] - ema_slow[i])
            momentum_ok = ema_spread >= atr[i] * min_ema_spread
            ema_bull = np.isfinite(ema_fast[i]) and np.isfinite(ema_slow[i]) and ema_fast[i] > ema_slow[i] and momentum_ok
            ema_bear = np.isfinite(ema_fast[i]) and np.isfinite(ema_slow[i]) and ema_fast[i] < ema_slow[i] and momentum_ok
        else:
            ema_bull = True
            ema_bear = True

        # Institutional VWAP filter
        if use_vwap:
            vwap_bull = close[i] > vwap_arr[i]
            vwap_bear = close[i] < vwap_arr[i]
        else:
            vwap_bull = True
            vwap_bear = True

        # HTF trend filter (Dual EMA)
        if use_htf:
            htf_s_bull = np.isfinite(htf_close_arr[i]) and np.isfinite(htf_ema_slow_arr[i]) and htf_close_arr[i] > htf_ema_slow_arr[i]
            htf_s_bear = np.isfinite(htf_close_arr[i]) and np.isfinite(htf_ema_slow_arr[i]) and htf_close_arr[i] < htf_ema_slow_arr[i]
            if htf_mode == "single":
                htf_bull = htf_s_bull
                htf_bear = htf_s_bear
            else:
                htf_bull = htf_s_bull and np.isfinite(htf_ema_fast_arr[i]) and htf_ema_fast_arr[i] >= htf_ema_slow_arr[i]
                htf_bear = htf_s_bear and np.isfinite(htf_ema_fast_arr[i]) and htf_ema_fast_arr[i] <= htf_ema_slow_arr[i]
        else:
            htf_bull = True
            htf_bear = True

        # RSI exhaustion filter
        if use_rsi:
            rsi_bull_ok = np.isfinite(rsi[i]) and rsi[i] < rsi_ob
            rsi_bear_ok = np.isfinite(rsi[i]) and rsi[i] > rsi_os
        else:
            rsi_bull_ok = True
            rsi_bear_ok = True

        # Volume spike
        if use_vol:
            vol_ok = (np.isfinite(vol_sma[i])
                      and vol_sma[i] > 0
                      and volume[i - 1] >= vol_sma[i] * vol_mult)
        else:
            vol_ok = True

        # SL distance & R:R check
        sl_dist = calc_sl_distance(atr[i], sl_mode, sl_atr_mult, sl_fixed_pips)
        if sl_dist <= 0:
            continue
        rr_ok = (tp_pips / sl_dist) >= min_rr

        if not rr_ok or not vol_ok:
            continue

        # Partial quantities
        if use_partial:
            q1 = actual_qty * scale_pct / 100.0
            q2 = actual_qty - q1
        else:
            q1 = actual_qty
            q2 = 0.0

        tp2_price_offset = atr[i] * tp2_atr_mult_param if np.isfinite(atr[i]) else tp_pips * 2.0

        if fvg_bull and ema_bull and vwap_bull and htf_bull and rsi_bull_ok:
            pending = {
                "side": 1,
                "entry_ref": close[i],
                "sl": close[i] - sl_dist,
                "tp1": close[i] + tp_pips,
                "tp2": close[i] + tp2_price_offset if use_partial else close[i] + tp_pips,
                "qty": actual_qty,
                "qty1": q1,
                "qty2": q2,
                "signal_i": i,
            }
        elif fvg_bear and ema_bear and vwap_bear and htf_bear and rsi_bear_ok:
            pending = {
                "side": -1,
                "entry_ref": close[i],
                "sl": close[i] + sl_dist,
                "tp1": close[i] - tp_pips,
                "tp2": close[i] - tp2_price_offset if use_partial else close[i] - tp_pips,
                "qty": actual_qty,
                "qty1": q1,
                "qty2": q2,
                "signal_i": i,
            }

    # ── Final accounting ────────────────────────────────────
    if day_cursor is not None:
        daily_equity.append(equity_at_day)

    final_equity = _mark(close[-1])
    trade_frame = pd.DataFrame(trades) if collect_trades and trades else None

    if collect_equity and equity_values:
        equity_frame = pd.DataFrame(
            {"equity": equity_values},
            index=pd.DatetimeIndex(equity_index, name="timestamp"),
        )
        running_peak = equity_frame["equity"].cummax()
        equity_frame["drawdown"] = equity_frame["equity"] - running_peak
        equity_frame["drawdown_pct"] = np.where(
            running_peak > 0, equity_frame["drawdown"] / running_peak, 0.0
        )
    else:
        equity_frame = None

    return {
        "pnls": pnls,
        "trades": trade_frame,
        "equity": equity_frame,
        "daily_equity": daily_equity,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "final_equity": final_equity,
        "initial_capital": initial,
        "open_qty": (pos_qty1 + pos_qty2) if side else 0.0,
        "open_side": "long" if side == 1 else "short" if side == -1 else "flat",
    }

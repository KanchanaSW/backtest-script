"""20/50 EMA + 200 trend filter with pro management.

Signals evaluate on the closed bar; market entries fill on the next open.
Stop/TP distances are locked to the signal bar's close and ATR.
After TP1, stop moves to break-even + offset and can ATR-trail the runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scalper_strategy import pine_ema
from src.strategy import execution_price, pine_atr, plan_exits, session_mask

EMA_INPUT_ORDER = [
    "account_size",
    "qty_pct",
    "fast_len",
    "slow_len",
    "trend_len",
    "atr_mult",
    "atr_length",
    "tp1_ratio",
    "tp2_ratio",
    "scale_pct",
    "use_trail",
    "trail_atr_mult",
    "be_offset",
    "use_session",
    "trading_hours",
    "session_tz",
    "close_friday",
]


def resolve_ema_params(config: dict, overrides: dict | None = None) -> dict:
    params: dict = {}
    params.update(config.get("fixed") or {})
    params.update(config.get("broker") or {})
    params.update(config.get("strategy") or {})
    if overrides:
        params.update(overrides)
    params.setdefault("account_size", 10000.0)
    params.setdefault("qty_pct", 10.0)
    params.setdefault("fast_len", 20)
    params.setdefault("slow_len", 50)
    params.setdefault("trend_len", 200)
    params.setdefault("atr_mult", 2.0)
    params.setdefault("atr_length", 14)
    params.setdefault("tp1_ratio", 0.8)
    params.setdefault("tp2_ratio", 2.0)
    params.setdefault("scale_pct", 50)
    params.setdefault("use_trail", True)
    params.setdefault("trail_atr_mult", 1.0)
    params.setdefault("be_offset", 0.30)
    params.setdefault("use_session", True)
    params.setdefault("trading_hours", "0300-1600")
    params.setdefault("session_tz", "America/New_York")
    params.setdefault("close_friday", True)
    params.setdefault("spread", 0.30)
    params.setdefault("slippage", 0.0)
    params.setdefault("commission_per_unit", 0.0)
    params.setdefault("margin_pct", 10.0)
    params.setdefault("chart_timezone", "UTC")
    return params


def run_ema_backtest(
    frame: pd.DataFrame,
    params: dict,
    *,
    collect_trades: bool = True,
    collect_equity: bool = True,
    trade_after: pd.Timestamp | None = None,
) -> dict:
    if frame.empty:
        raise ValueError("No candles to backtest")

    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    index = frame.index
    n = len(frame)

    fast_len = int(params["fast_len"])
    slow_len = int(params["slow_len"])
    trend_len = int(params["trend_len"])
    atr_len = int(params["atr_length"])
    atr_mult = float(params["atr_mult"])
    tp1_ratio = float(params["tp1_ratio"])
    tp2_ratio = float(params["tp2_ratio"])
    scale_pct = float(params["scale_pct"])
    use_trail = bool(params["use_trail"])
    trail_mult = float(params["trail_atr_mult"])
    be_offset = float(params["be_offset"])
    qty_pct = float(params["qty_pct"]) / 100.0

    fast_ema = pine_ema(close, fast_len)
    slow_ema = pine_ema(close, slow_len)
    trend_ema = pine_ema(close, trend_len)
    atr = pine_atr(high, low, close, atr_len)

    local = index.tz_convert(str(params.get("chart_timezone", "UTC")))
    hours = local.hour.to_numpy()
    dows = local.dayofweek.to_numpy()
    in_session = session_mask(
        index,
        str(params["trading_hours"]),
        str(params["session_tz"]),
        bool(params["use_session"]),
    )
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

    spread = float(params["spread"])
    slippage = float(params["slippage"])
    commission = float(params["commission_per_unit"])
    margin_pct = float(params["margin_pct"])
    initial = float(params["account_size"])

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
            trades.append(
                {
                    "signal_time": index[signal_i],
                    "entry_time": index[entry_i],
                    "exit_time": index[bar],
                    "side": "long" if side == 1 else "short",
                    "qty": leg_qty,
                    "entry_price": entry_fill,
                    "exit_price": exit_fill,
                    "pnl": pnl,
                    "reason": reason,
                }
            )
        return pnl

    for i in range(n):
        if pending_flatten and side != 0:
            for leg_name, leg_qty in (("qty1", pos_qty1), ("qty2", pos_qty2)):
                if leg_qty > 1e-12:
                    record_exit(i, leg_qty, open_[i], "friday")
                    if leg_name == "qty1":
                        pos_qty1 = 0.0
                    else:
                        pos_qty2 = 0.0
            side = 0
            tp1_hit = False
        pending_flatten = False

        if pending is not None:
            if side == 0:
                entry_px = execution_price(pending["side"], open_[i], True, spread, slippage)
                equity_now = _mark(open_[i])
                qty = (equity_now * qty_pct) / entry_px if entry_px > 0 else 0.0
                required = abs(entry_px) * qty * margin_pct / 100.0
                if qty > 1e-12 and equity_now >= required:
                    side = pending["side"]
                    pos_qty1 = qty * scale_pct / 100.0
                    pos_qty2 = qty - pos_qty1
                    entry_fill = entry_px
                    entry_ref = pending["entry_ref"]
                    sl = pending["sl"]
                    tp1 = pending["tp1"]
                    tp2 = pending["tp2"]
                    signal_i = pending["signal_i"]
                    entry_i = i
                    tp1_hit = False
            pending = None

        if side != 0 and (pos_qty1 > 1e-12 or pos_qty2 > 1e-12):
            planned = plan_exits(
                side, open_[i], high[i], low[i], close[i], sl, tp1, tp2, pos_qty1, pos_qty2
            )
            for chart_price, reason, leg in planned:
                leg_qty = pos_qty1 if leg == "qty1" else pos_qty2
                if leg_qty <= 1e-12:
                    continue
                record_exit(i, leg_qty, chart_price, reason)
                if leg == "qty1":
                    pos_qty1 = 0.0
                else:
                    pos_qty2 = 0.0
            if pos_qty1 <= 1e-12 and pos_qty2 <= 1e-12:
                side = 0
                tp1_hit = False

        if side == 1:
            if not tp1_hit and high[i] >= tp1:
                tp1_hit = True
                if use_trail:
                    sl = entry_ref + be_offset
            if tp1_hit and use_trail:
                trail = high[i] - atr[i] * trail_mult
                if np.isfinite(trail) and trail > sl:
                    sl = trail
        elif side == -1:
            if not tp1_hit and low[i] <= tp1:
                tp1_hit = True
                if use_trail:
                    sl = entry_ref - be_offset
            if tp1_hit and use_trail:
                trail = low[i] + atr[i] * trail_mult
                if np.isfinite(trail) and trail < sl:
                    sl = trail

        if side == 0 and pending is None and allowed[i] and in_session[i] and not friday_block[i]:
            f = fast_ema[i]
            s = slow_ema[i]
            t = trend_ema[i]
            a = atr[i]
            fp = fast_ema[i - 1] if i > 0 else np.nan
            sp = slow_ema[i - 1] if i > 0 else np.nan
            if (
                np.isfinite(f)
                and np.isfinite(s)
                and np.isfinite(t)
                and np.isfinite(a)
                and np.isfinite(fp)
                and np.isfinite(sp)
                and a > 0
            ):
                long_cross = fp <= sp and f > s
                short_cross = fp >= sp and f < s
                if long_cross and close[i] > t:
                    risk = a * atr_mult
                    entry_ref_sig = close[i]
                    pending = {
                        "side": 1,
                        "entry_ref": entry_ref_sig,
                        "sl": entry_ref_sig - risk,
                        "tp1": entry_ref_sig + risk * tp1_ratio,
                        "tp2": entry_ref_sig + risk * tp2_ratio,
                        "signal_i": i,
                    }
                elif short_cross and close[i] < t:
                    risk = a * atr_mult
                    entry_ref_sig = close[i]
                    pending = {
                        "side": -1,
                        "entry_ref": entry_ref_sig,
                        "sl": entry_ref_sig + risk,
                        "tp1": entry_ref_sig - risk * tp1_ratio,
                        "tp2": entry_ref_sig - risk * tp2_ratio,
                        "signal_i": i,
                    }

        if side != 0 and bool(params["close_friday"]) and friday_block[i]:
            pending_flatten = True

        equity = _mark(close[i])
        if collect_equity:
            equity_values.append(equity)
            equity_index.append(index[i])

        day = index[i].normalize()
        if day_cursor is None:
            day_cursor = day
            equity_at_day = equity
        elif day != day_cursor:
            daily_equity.append(equity_at_day)
            day_cursor = day
        equity_at_day = equity

        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

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

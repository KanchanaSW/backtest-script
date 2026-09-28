"""Pine-compatible indicators and the bar-by-bar strategy simulator.

Execution matches TradingView with process_orders_on_close off:
signals are decided on the closed bar and market orders fill on the next open.
Stop and target prices are locked to the signal bar's close. When a bar can
reach both a stop and a target, the path is open to the nearer extreme, then
the far extreme, then the close. A tie goes to the high first.

Take-profit scaling happens once. The script re-issues the TP1 exit every bar;
that second scale-out is not simulated. Each filled partial is its own closed
trade, and the daily loss counter can increase only once per bar.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

PINE_INPUT_ORDER = [
    "account_size",
    "lot_size",
    "units_per_lot",
    "atr_sl_mult",
    "tp1_ratio",
    "tp2_ratio",
    "scale_pct",
    "use_trail",
    "max_daily_loss",
    "close_friday",
    "use_session",
    "trading_hours",
    "session_tz",
    "use_vol",
    "vol_length",
    "use_bias",
    "bias_tf",
    "bias_length",
    "max_chase_atr",
    "htf",
    "swing_len",
    "fvg_timeout",
    "show_boxes",
    "sessDef",
    "sessHour",
    "sessTz",
    "vpRows",
    "vaPct",
    "useLtf",
    "ltfTf",
    "showLvl",
]

VISUAL_INPUTS = {
    "show_boxes",
    "sessDef",
    "sessHour",
    "sessTz",
    "vpRows",
    "vaPct",
    "useLtf",
    "ltfTf",
    "showLvl",
}


def resolve_params(config: dict, overrides: dict | None = None) -> dict:
    params = {}
    params.update(config.get("fixed") or {})
    params.update(config.get("broker") or {})
    params.update(config.get("strategy") or {})
    if overrides:
        params.update(overrides)
    params.setdefault("atr_length", 14)
    params.setdefault("trail_atr_mult", 1.5)
    params.setdefault("chart_timezone", "UTC")
    params.setdefault("spread", 0.30)
    params.setdefault("slippage", 0.0)
    params.setdefault("commission_per_unit", 0.0)
    params.setdefault("margin_pct", 10.0)
    params.setdefault("stop_mode", "atr")
    params.setdefault("sweep_buffer_atr", 0.15)
    params.setdefault("max_chase_atr", 0.0)
    params.setdefault("min_sweep_atr", 0.0)
    params.setdefault("use_bias", False)
    params.setdefault("bias_tf", "D")
    params.setdefault("bias_length", 20)
    params.setdefault("bias_align", "with")
    return params


def pine_rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder RMA seeded with an SMA, matching ta.rma."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if length <= 0:
        return out
    prev = np.nan
    alpha = 1.0 / length
    for i, value in enumerate(values):
        if not np.isfinite(prev):
            if i >= length - 1:
                window = values[i - length + 1 : i + 1]
                if np.all(np.isfinite(window)):
                    prev = float(window.mean())
                    out[i] = prev
            continue
        if not np.isfinite(value):
            prev = np.nan
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out[i] = prev
    return out


def pine_sma(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if length <= 0 or len(values) < length:
        return out
    totals = np.cumsum(np.where(np.isfinite(values), values, 0.0))
    finite = np.cumsum(np.isfinite(values).astype(int))
    end = finite[length - 1 :] - np.concatenate(([0], finite[:-length]))
    sum_end = totals[length - 1 :] - np.concatenate(([0.0], totals[:-length]))
    valid = end == length
    out[length - 1 :][valid] = sum_end[valid] / length
    return out


def pine_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous = np.empty_like(close, dtype=float)
    previous[0] = np.nan
    previous[1:] = close[:-1]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    true_range[0] = np.nan
    return true_range


def pine_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    return pine_rma(pine_tr(high, low, close), length)


def _pivot(values: np.ndarray, left: int, right: int, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if left < 1 or right < 1 or len(values) < left + right + 1:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, left + right + 1)
    pivot = windows[:, left]
    left_side = windows[:, :left]
    right_side = windows[:, left + 1 :]
    finite = np.isfinite(pivot) & np.all(np.isfinite(left_side), axis=1) & np.all(np.isfinite(right_side), axis=1)
    if kind == "high":
        is_pivot = finite & (pivot > left_side.max(axis=1)) & (pivot > right_side.max(axis=1))
    else:
        is_pivot = finite & (pivot < left_side.min(axis=1)) & (pivot < right_side.min(axis=1))
    idx = np.flatnonzero(is_pivot)
    out[idx + left + right] = pivot[idx]
    return out


def pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    return _pivot(high, left, right, "high")


def pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    return _pivot(low, left, right, "low")


def pine_timeframe_to_rule(timeframe: str) -> str:
    token = str(timeframe).strip()
    aliases = {
        "1": "1min",
        "3": "3min",
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "45": "45min",
        "60": "60min",
        "120": "120min",
        "180": "180min",
        "240": "240min",
        "D": "1D",
        "1D": "1D",
        "W": "1W",
        "1W": "1W",
    }
    if token in aliases:
        return aliases[token]
    if token.isdigit():
        return f"{token}min"
    return token


def higher_timeframe_ema(index: pd.DatetimeIndex, close: np.ndarray, timeframe: str, length: int) -> np.ndarray:
    """Prior completed higher-timeframe EMA, shifted so the forming bar is excluded."""
    if length <= 0 or not str(timeframe).strip():
        return np.full(len(close), np.nan)
    series = pd.Series(close, index=index)
    grouped = series.resample(pine_timeframe_to_rule(timeframe), label="left", closed="left", origin="epoch").last()
    ema = grouped.ewm(span=length, adjust=False).mean().shift(1)
    return ema.reindex(index, method="ffill").to_numpy(dtype=float)


def bias_allows(side: int, close: float, bias: float, align: str) -> bool:
    if align not in {"with", "against"} or not np.isfinite(bias) or not np.isfinite(close):
        return align not in {"with", "against"}
    above = close > bias
    if align == "with":
        return above if side == 1 else not above
    return (not above) if side == 1 else above


def protective_stop(side: int, entry: float, atr_stop: float, extreme: float, mode: str, buffer: float) -> float:
    """ATR stop, or a stop just beyond the sweep wick. NaN when the stop is not beyond entry."""
    if mode != "sweep" or not np.isfinite(extreme):
        stop = atr_stop
    elif side == 1:
        stop = extreme - buffer
    else:
        stop = extreme + buffer
    if side == 1 and stop >= entry:
        return np.nan
    if side == -1 and stop <= entry:
        return np.nan
    return stop


def previous_htf_levels(frame: pd.DataFrame, timeframe: str) -> tuple[np.ndarray, np.ndarray]:
    """Previous completed higher-timeframe high and low.

    This is request.security(htf, high[1]/low[1]) with lookahead on: the level
    updates when the new HTF bar opens, and it never includes the forming bar.
    """
    rule = pine_timeframe_to_rule(timeframe)
    grouped = frame.resample(rule, label="left", closed="left", origin="epoch").agg(
        {"high": "max", "low": "min"}
    )
    aligned = grouped.shift(1).reindex(frame.index, method="ffill")
    return aligned["high"].to_numpy(dtype=float), aligned["low"].to_numpy(dtype=float)


def parse_session(spec: str) -> tuple[int, int, set[int] | None]:
    text = str(spec).strip()
    time_part, _, day_part = text.partition(":")
    start_token, end_token = time_part.split("-")
    start = int(start_token[:2]) * 60 + int(start_token[2:4])
    end = int(end_token[:2]) * 60 + int(end_token[2:4])
    days = {int(char) for char in day_part} if day_part else None
    return start, end, days


def session_mask(index: pd.DatetimeIndex, spec: str, timezone_name: str, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(len(index), dtype=bool)
    start, end, days = parse_session(spec)
    local = index.tz_convert(timezone_name)
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    if start < end:
        inside = (minutes >= start) & (minutes < end)
    elif start == end:
        inside = np.ones(len(index), dtype=bool)
    else:
        inside = (minutes >= start) | (minutes < end)
    if not days:
        return inside
    pandas_dow = local.dayofweek.to_numpy()
    tv_day = np.where(pandas_dow == 6, 1, pandas_dow + 2)
    return inside & np.isin(tv_day, list(days))


def bar_path(open_: float, high: float, low: float, close: float) -> list[float]:
    """TradingView broker-emulator path. Equal distance to high and low goes high first."""
    if abs(open_ - low) < abs(open_ - high):
        return [open_, low, high, close]
    return [open_, high, low, close]


def update_daily_losses(daily_losses: int, new_day: bool, closed_pnls: Iterable[float]) -> int:
    """Pine increments this once per bar, and a loss on the bar wins over a win."""
    if new_day:
        daily_losses = 0
    pnls = list(closed_pnls)
    if any(pnl < 0.0 for pnl in pnls):
        return daily_losses + 1
    if any(pnl > 0.0 for pnl in pnls):
        return 0
    return daily_losses


def execution_price(side: int, chart_price: float, is_entry: bool, spread: float, slippage: float) -> float:
    """Bid-candle adjustment. Buys pay the full spread; sells trade on the bid."""
    if side == 1:
        if is_entry:
            return chart_price + spread + slippage
        return chart_price - slippage
    if is_entry:
        return chart_price - slippage
    return chart_price + spread + slippage


def plan_exits(
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
    """Chart prices hit along the TradingView path.

    Each tuple is (chart_price, reason, leg) with leg 'qty1' or 'qty2'.
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

    if side == 1:
        if open_ <= sl:
            add(open_, "stop", "all")
            return fills
        if open_ >= tp1:
            add(open_, "tp1", "qty1")
        if open_ >= tp2:
            add(open_, "tp2", "qty2")
        stop_up = False
    else:
        if open_ >= sl:
            add(open_, "stop", "all")
            return fills
        if open_ <= tp1:
            add(open_, "tp1", "qty1")
        if open_ <= tp2:
            add(open_, "tp2", "qty2")
        stop_up = True

    if q1 <= 1e-12 and q2 <= 1e-12:
        return fills

    points = bar_path(open_, high, low, close)
    for left, right in zip(points, points[1:]):
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


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    if bars < len(values):
        out[bars:] = values[:-bars]
    return out


def _round_trip(side: int, entry_fill: float, exit_fill: float, qty: float, commission: float) -> float:
    gross = (exit_fill - entry_fill) * qty if side == 1 else (entry_fill - exit_fill) * qty
    return gross - commission * qty * 2.0


def _mark(cash: float, side: int, entry_fill: float, qty: float, close: float, spread: float, commission: float) -> float:
    if side == 0 or qty <= 1e-12:
        return cash
    fee = commission * qty * 2.0
    if side == 1:
        return cash + (close - entry_fill) * qty - fee
    return cash + (entry_fill - (close + spread)) * qty - fee


def run_backtest(
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
    volume = frame["volume"].to_numpy(dtype=float)
    index = frame.index
    n = len(frame)

    atr = pine_atr(high, low, close, int(params["atr_length"]))
    vol_ma = pine_sma(volume, int(params["vol_length"]))
    swing = int(params["swing_len"])
    ph = pivot_high(high, swing, swing)
    pl = pivot_low(low, swing, swing)
    htf_high, htf_low = previous_htf_levels(frame, str(params["htf"]))
    use_bias = bool(params.get("use_bias", False))
    bias = higher_timeframe_ema(
        index,
        close,
        str(params.get("bias_tf", "D")) if use_bias else "",
        int(params.get("bias_length", 20)),
    )
    prev_close_arr = _shift(close, 1)
    prev_open_arr = _shift(open_, 1)
    high_2 = _shift(high, 2)
    low_2 = _shift(low, 2)
    fvg_bull = (low > high_2) & (prev_close_arr > prev_open_arr)
    fvg_bear = (high < low_2) & (prev_close_arr < prev_open_arr)

    local = index.tz_convert(str(params["chart_timezone"]))
    hours = local.hour.to_numpy()
    dows = local.dayofweek.to_numpy()
    day_ids = local.normalize().asi8
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
    atr_mult = float(params["atr_sl_mult"])
    tp1_ratio = float(params["tp1_ratio"])
    tp2_ratio = float(params["tp2_ratio"])
    trail_mult = float(params["trail_atr_mult"])
    use_trail = bool(params["use_trail"])
    use_vol = bool(params["use_vol"])
    timeout = int(params["fvg_timeout"])
    stop_mode = str(params.get("stop_mode", "atr"))
    sweep_buffer_atr = float(params.get("sweep_buffer_atr", 0.15))
    max_chase_atr = float(params.get("max_chase_atr", 0.0))
    min_sweep_atr = float(params.get("min_sweep_atr", 0.0))
    bias_align = str(params.get("bias_align", "with")) if use_bias else ""
    max_daily_loss = int(params["max_daily_loss"])
    actual_qty = float(params["lot_size"]) * float(params["units_per_lot"])
    qty1_target = actual_qty * float(params["scale_pct"]) / 100.0
    qty2_target = actual_qty - qty1_target
    initial = float(params["account_size"])

    last_ph = np.nan
    last_pl = np.nan
    swept_bull = False
    swept_bear = False
    sweep_low = np.nan
    sweep_high = np.nan
    sweep_depth_bull = 0.0
    sweep_depth_bear = 0.0
    armed_bull_extreme = np.nan
    armed_bear_extreme = np.nan
    armed_bull_depth = 0.0
    armed_bear_depth = 0.0
    wait_bull = False
    wait_bear = False
    bars_since = 0
    daily_losses = 0
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
    pending: list[dict] = []
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

    def record_exit(bar: int, leg_qty: float, chart_price: float, reason: str) -> float:
        nonlocal cash
        exit_fill = execution_price(side, chart_price, False, spread, slippage)
        pnl = _round_trip(side, entry_fill, exit_fill, leg_qty, commission)
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
        bar_pnls: list[float] = []

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
        pending_flatten = False

        for order in pending:
            if side != 0 and order["side"] != side:
                for leg_qty in (pos_qty1, pos_qty2):
                    if leg_qty > 1e-12:
                        bar_pnls.append(record_exit(i, leg_qty, open_[i], "reverse"))
                pos_qty1 = 0.0
                pos_qty2 = 0.0
                side = 0
                tp1_hit = False
            if side != 0:
                break
            entry_px = execution_price(order["side"], open_[i], True, spread, slippage)
            required = abs(entry_px) * order["qty"] * margin_pct / 100.0
            equity_now = _mark(cash, 0, 0.0, 0.0, open_[i], spread, commission)
            if equity_now < required:
                continue
            side = order["side"]
            pos_qty1 = order["qty1"]
            pos_qty2 = order["qty2"]
            entry_fill = entry_px
            entry_ref = order["entry_ref"]
            sl = order["sl"]
            tp1 = order["tp1"]
            tp2 = order["tp2"]
            signal_i = order["signal_i"]
            entry_i = i
            tp1_hit = False
        pending = []

        if side != 0 and (pos_qty1 > 1e-12 or pos_qty2 > 1e-12):
            planned = plan_exits(side, open_[i], high[i], low[i], close[i], sl, tp1, tp2, pos_qty1, pos_qty2)
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

        qty_open = pos_qty1 + pos_qty2
        equity = _mark(cash, side, entry_fill, qty_open, close[i], spread, commission)
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

        new_day = i > 0 and day_ids[i] != day_ids[i - 1]
        daily_losses = update_daily_losses(daily_losses, new_day, bar_pnls)
        can_trade = daily_losses < max_daily_loss

        if side != 0 and bool(params["close_friday"]) and friday_block[i]:
            pending_flatten = True

        prev_ph = last_ph
        prev_pl = last_pl
        if np.isfinite(ph[i]):
            last_ph = ph[i]
        if np.isfinite(pl[i]):
            last_pl = pl[i]

        if np.isfinite(htf_low[i]) and low[i] < htf_low[i] and close[i] > htf_low[i]:
            swept_bull = True
            swept_bear = False
            sweep_low = low[i]
            sweep_depth_bull = htf_low[i] - low[i]
        if np.isfinite(htf_high[i]) and high[i] > htf_high[i] and close[i] < htf_high[i]:
            swept_bear = True
            swept_bull = False
            sweep_high = high[i]
            sweep_depth_bear = high[i] - htf_high[i]
        if swept_bull and close[i] < sweep_low:
            swept_bull = False
        if swept_bear and close[i] > sweep_high:
            swept_bear = False

        prev_c = prev_close_arr[i]
        bullish_shift = (
            swept_bull
            and np.isfinite(prev_c)
            and np.isfinite(prev_ph)
            and np.isfinite(last_ph)
            and prev_c <= prev_ph
            and close[i] > last_ph
        )
        bearish_shift = (
            swept_bear
            and np.isfinite(prev_c)
            and np.isfinite(prev_pl)
            and np.isfinite(last_pl)
            and prev_c >= prev_pl
            and close[i] < last_pl
        )
        if bullish_shift:
            wait_bull = True
            swept_bull = False
            bars_since = 0
            armed_bull_extreme = sweep_low
            armed_bull_depth = sweep_depth_bull
        if bearish_shift:
            wait_bear = True
            swept_bear = False
            bars_since = 0
            armed_bear_extreme = sweep_high
            armed_bear_depth = sweep_depth_bear
        if wait_bull or wait_bear:
            bars_since += 1
        if bars_since > timeout:
            wait_bull = False
            wait_bear = False

        vol_ok = True
        if use_vol:
            vol_ok = bool(np.isfinite(vol_ma[i]) and volume[i] > vol_ma[i])
        sl_distance = atr[i] * atr_mult
        risk_ok = bool(np.isfinite(sl_distance) and sl_distance > 0)
        flat = side == 0
        tradable = bool(allowed[i] and flat and can_trade and vol_ok and in_session[i] and not friday_block[i] and risk_ok)

        def chase_ok(entry: float, extreme: float) -> bool:
            if max_chase_atr <= 0 or not np.isfinite(extreme) or not np.isfinite(atr[i]) or atr[i] <= 0:
                return True
            return abs(entry - extreme) <= max_chase_atr * atr[i]

        def depth_ok(depth: float) -> bool:
            if min_sweep_atr <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                return True
            return depth >= min_sweep_atr * atr[i]

        orders: list[dict] = []
        if tradable and wait_bull and bool(fvg_bull[i]):
            buffer = atr[i] * sweep_buffer_atr if np.isfinite(atr[i]) else 0.0
            stop = protective_stop(1, close[i], close[i] - sl_distance, armed_bull_extreme, stop_mode, buffer)
            risk = close[i] - stop if np.isfinite(stop) else np.nan
            qualified = (
                bias_allows(1, close[i], bias[i], bias_align)
                and chase_ok(close[i], armed_bull_extreme)
                and depth_ok(armed_bull_depth)
                and np.isfinite(risk)
                and risk > 0
            )
            if qualified:
                orders.append(
                    {
                        "side": 1,
                        "entry_ref": close[i],
                        "sl": stop,
                        "tp1": close[i] + risk * tp1_ratio,
                        "tp2": close[i] + risk * tp2_ratio,
                        "qty": actual_qty,
                        "qty1": qty1_target,
                        "qty2": qty2_target,
                        "signal_i": i,
                    }
                )
            wait_bull = False
        if tradable and wait_bear and bool(fvg_bear[i]):
            buffer = atr[i] * sweep_buffer_atr if np.isfinite(atr[i]) else 0.0
            stop = protective_stop(-1, close[i], close[i] + sl_distance, armed_bear_extreme, stop_mode, buffer)
            risk = stop - close[i] if np.isfinite(stop) else np.nan
            qualified = (
                bias_allows(-1, close[i], bias[i], bias_align)
                and chase_ok(close[i], armed_bear_extreme)
                and depth_ok(armed_bear_depth)
                and np.isfinite(risk)
                and risk > 0
            )
            if qualified:
                orders.append(
                    {
                        "side": -1,
                        "entry_ref": close[i],
                        "sl": stop,
                        "tp1": close[i] - risk * tp1_ratio,
                        "tp2": close[i] - risk * tp2_ratio,
                        "qty": actual_qty,
                        "qty1": qty1_target,
                        "qty2": qty2_target,
                        "signal_i": i,
                    }
                )
            wait_bear = False
        pending = orders

        # Stop updates are placed for the next bar. Pine moves the stop to the
        # signal close once TP1 is touched, then trails on that same close.
        if side == 1:
            if not tp1_hit and high[i] >= tp1:
                tp1_hit = True
                if use_trail:
                    sl = entry_ref
            if tp1_hit and use_trail:
                trail = high[i] - atr[i] * trail_mult
                if np.isfinite(trail) and trail > sl:
                    sl = trail
        elif side == -1:
            if not tp1_hit and low[i] <= tp1:
                tp1_hit = True
                if use_trail:
                    sl = entry_ref
            if tp1_hit and use_trail:
                trail = low[i] + atr[i] * trail_mult
                if np.isfinite(trail) and trail < sl:
                    sl = trail

    if day_cursor is not None:
        daily_equity.append(equity_at_day)

    final_equity = _mark(cash, side, entry_fill, pos_qty1 + pos_qty2, close[-1], spread, commission)
    trade_frame = pd.DataFrame(trades) if collect_trades else None
    if collect_equity:
        equity_frame = pd.DataFrame({"equity": equity_values}, index=pd.DatetimeIndex(equity_index, name="timestamp"))
        running_peak = equity_frame["equity"].cummax()
        equity_frame["drawdown"] = equity_frame["equity"] - running_peak
        equity_frame["drawdown_pct"] = np.where(running_peak > 0, equity_frame["drawdown"] / running_peak, 0.0)
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
        "open_qty": pos_qty1 + pos_qty2 if side else 0.0,
        "open_side": "long" if side == 1 else "short" if side == -1 else "flat",
    }

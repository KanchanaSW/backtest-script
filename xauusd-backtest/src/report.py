"""Metrics, charts, and the Pine input block."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.strategy import PINE_INPUT_ORDER

TITLES = {
    "account_size": "Account Size ($)",
    "lot_size": "Fixed Lot Size",
    "units_per_lot": "Units per 1 Lot (100 = Gold)",
    "atr_sl_mult": "ATR Stop Loss Multiplier",
    "tp1_ratio": "TP1 Risk/Reward Ratio",
    "tp2_ratio": "TP2 Risk/Reward Ratio",
    "scale_pct": "TP1 Scale Out %",
    "use_trail": "Trail Stop Loss after TP1 Hit",
    "max_daily_loss": "Max Consecutive Daily Losses",
    "close_friday": "Close All Trades Friday at 16:00",
    "use_session": "Use Trading Session Filter",
    "trading_hours": "Trading Hours (NY Time)",
    "session_tz": "Timezone",
    "use_vol": "Use Volume Filter",
    "vol_length": "Volume MA Length",
    "use_bias": "Trade With Higher-Timeframe Trend",
    "bias_tf": "Trend Timeframe",
    "bias_length": "Trend EMA Length",
    "max_chase_atr": "Max Chase From Sweep (ATR)",
    "htf": "Higher Timeframe (HTF) Sweep",
    "swing_len": "Swing Length for Shift",
    "fvg_timeout": "Max Bars to Wait for FVG",
    "show_boxes": "Draw Risk/Reward Boxes",
    "sessDef": "Session boundary",
    "sessHour": "Custom start hour (0-23)",
    "sessTz": "Custom hour timezone",
    "vpRows": "Row size",
    "vaPct": "Value area volume %",
    "useLtf": "Build profile from intrabars",
    "ltfTf": "Intrabar timeframe",
    "showLvl": "Show POC / VAH / VAL",
}

INT_INPUTS = {
    "scale_pct",
    "max_daily_loss",
    "vol_length",
    "bias_length",
    "swing_len",
    "fvg_timeout",
    "sessHour",
    "vpRows",
}

VISUAL_NOTE = (
    "show_boxes, sessDef, sessHour, sessTz, vpRows, vaPct, useLtf, ltfTf, and "
    "showLvl are visual. They are listed so the input list stays complete, and "
    "the backtest does not use them."
)


def sharpe_ratio(daily_equity: list[float] | np.ndarray) -> float:
    equity = np.asarray(daily_equity, dtype=float)
    if len(equity) < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    if deviation == 0.0:
        return 0.0
    return float(np.mean(returns) / deviation * math.sqrt(252))


def compute_metrics(result: dict) -> dict:
    pnls = np.asarray(result["pnls"], dtype=float)
    trades = int(len(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    win_rate = float(len(wins) / trades) if trades else 0.0
    loss_rate = float(len(losses) / trades) if trades else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    average_trade = float(pnls.mean()) if trades else 0.0
    expectancy = win_rate * avg_win - loss_rate * avg_loss
    initial = float(result["initial_capital"])
    final_equity = float(result["final_equity"])
    return {
        "net_profit": final_equity - initial,
        "realized_profit": float(pnls.sum()) if trades else 0.0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe_ratio(result["daily_equity"]),
        "max_drawdown": float(result["max_drawdown"]),
        "max_drawdown_pct": float(result["max_drawdown_pct"]),
        "trades": trades,
        "average_trade": average_trade,
        "expectancy": expectancy,
        "final_equity": final_equity,
        "open_side": result["open_side"],
        "open_qty": float(result["open_qty"]),
    }


def rank_value(metrics: dict, rank_by: str, min_trades: int) -> float:
    if int(metrics["trades"]) < int(min_trades):
        return -1e18
    value = float(metrics[rank_by])
    if math.isinf(value):
        return 1e6
    if not math.isfinite(value):
        return -1e18
    return value


def format_input_value(key: str, value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if key in INT_INPUTS:
        return str(int(value))
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def format_pine_inputs(params: dict) -> str:
    lines = ["// Paste into the strategy inputs", ""]
    for key in PINE_INPUT_ORDER:
        if key not in params:
            continue
        rendered = format_input_value(key, params[key])
        lines.append(f"{key} = {rendered}    // {TITLES.get(key, key)}")
    lines.extend(["", f"// {VISUAL_NOTE}"])
    return "\n".join(lines)


def _fmt_metric(key: str, value) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if key in {"win_rate"}:
        return f"{float(value) * 100:.2f}%"
    if key in {"max_drawdown_pct"}:
        return f"{float(value) * 100:.2f}%"
    if key in {"trades"}:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


METRIC_ROWS = [
    ("net_profit", "Net profit"),
    ("win_rate", "Win rate"),
    ("profit_factor", "Profit factor"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
    ("max_drawdown_pct", "Max drawdown %"),
    ("trades", "Trades"),
    ("average_trade", "Average trade"),
    ("expectancy", "Expectancy"),
]


def format_metrics(metrics: dict) -> str:
    lines = []
    for key, label in METRIC_ROWS:
        lines.append(f"{label:<16} {_fmt_metric(key, metrics[key])}")
    if metrics.get("open_side") not in {None, "flat"}:
        lines.append(f"{'Open position':<16} {metrics['open_side']} {metrics['open_qty']:.2f}")
    return "\n".join(lines)


def format_side_by_side(left: dict, right: dict, left_name: str = "In sample", right_name: str = "Out of sample") -> str:
    header = f"{'Metric':<16} {left_name:>16} {right_name:>16}"
    lines = [header, "-" * len(header)]
    for key, label in METRIC_ROWS:
        lines.append(
            f"{label:<16} {_fmt_metric(key, left[key]):>16} {_fmt_metric(key, right[key]):>16}"
        )
    return "\n".join(lines)


def save_chart(equity: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(equity.index, equity["equity"], color="#1f4b99", linewidth=1.0)
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Equity")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].fill_between(equity.index, equity["drawdown"], 0, color="#9a3412", alpha=0.85)
    axes[1].set_ylabel("Drawdown ($)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_heatmap(results: list[dict], x_key: str, y_key: str, score_key: str, path: Path) -> None:
    if not results:
        return
    best = max(results, key=lambda row: row["score"])
    slice_rows = []
    other_keys = [key for key in results[0]["params"] if key not in {x_key, y_key}]
    for row in results:
        if all(row["params"].get(key) == best["params"].get(key) for key in other_keys):
            slice_rows.append(row)
    if not slice_rows:
        return
    frame = pd.DataFrame(
        {
            x_key: [row["params"][x_key] for row in slice_rows],
            y_key: [row["params"][y_key] for row in slice_rows],
            score_key: [row["score"] if row["score"] > -1e17 else np.nan for row in slice_rows],
        }
    )
    table = frame.pivot_table(index=y_key, columns=x_key, values=score_key, aggfunc="mean")
    table = table.sort_index(ascending=False)
    if not np.isfinite(table.to_numpy(dtype=float)).any():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(value) for value in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(value) for value in table.index])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{score_key} at the best other inputs")
    fig.colorbar(image, ax=ax, label=score_key)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _metric_value(row: dict, metric: str) -> float | None:
    if int(row["metrics"].get("trades", 0)) <= 0:
        return None
    value = float(row["metrics"].get(metric, row["score"]))
    if math.isinf(value):
        return 1e6
    if not math.isfinite(value):
        return None
    return value


def sharp_peak_notes(results: list[dict], space: dict, ratio: float, metric: str = "sharpe") -> list[str]:
    """Flag a best point that is much stronger than the adjacent grid values."""
    if not results:
        return []
    best = max(results, key=lambda row: row["score"])
    best_value = _metric_value(best, metric)
    if best_value is None or best_value <= 0:
        return []
    notes = []
    for key, values in space.items():
        grid = list(values)
        current = best["params"].get(key)
        if current not in grid or len(grid) < 2:
            continue
        index = grid.index(current)
        if index == 0 or index == len(grid) - 1:
            notes.append(f"{key}={current} sits on the edge of the search. Widen that range.")
        neighbor_scores = []
        for neighbor in (index - 1, index + 1):
            if neighbor < 0 or neighbor >= len(grid):
                continue
            target = grid[neighbor]
            matches = [
                row
                for row in results
                if row["params"].get(key) == target
                and all(row["params"].get(other) == best["params"].get(other) for other in space if other != key)
            ]
            if not matches:
                continue
            value = _metric_value(max(matches, key=lambda row: row["score"]), metric)
            if value is not None:
                neighbor_scores.append(value)
        if not neighbor_scores:
            continue
        ceiling = max(neighbor_scores)
        if ceiling <= 0 or best_value >= ratio * ceiling:
            notes.append(
                f"{key}={current} is a sharp peak ({best_value:.3f} vs neighbor {ceiling:.3f}). Overfit risk."
            )
    return notes

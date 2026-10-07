"""Run one backtest of the 20/50 EMA + pro management strategy.

    python -m src.backtest_ema
    python -m src.backtest_ema --start 2023-10-01 --end 2026-10-01
    python -m src.backtest_ema --start 2023-10-01 --end 2026-10-01 --optimize
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product

import pandas as pd

from src.data_loader import load_candles, load_config, project_path, slice_dates
from src.ema_strategy import EMA_INPUT_ORDER, resolve_ema_params, run_ema_backtest
from src.report import compute_metrics, format_metrics, save_chart

GRID = {
    "atr_mult": [1.0, 1.2, 1.5, 2.0, 2.5],
    "tp1_ratio": [0.8, 1.0, 1.2],
    "tp2_ratio": [2.0, 2.5],
    "trail_atr_mult": [0.8, 1.0, 1.2],
}
MIN_TRADES = 80


def _jsonable(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
        else:
            out[key] = value
    return out


def _format_ema_inputs(params: dict) -> str:
    lines = ["// 20/50 EMA + Pro Management Pine Script inputs", ""]
    for key in EMA_INPUT_ORDER:
        if key not in params:
            continue
        val = params[key]
        if isinstance(val, bool):
            rendered = "true" if val else "false"
        elif isinstance(val, float):
            rendered = f"{val:.6f}".rstrip("0").rstrip(".") or "0"
        elif isinstance(val, int):
            rendered = str(val)
        else:
            rendered = str(val)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines)


def run_grid(frame: pd.DataFrame, base_params: dict) -> list[dict]:
    rows: list[dict] = []
    keys = list(GRID.keys())
    for values in product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        params = dict(base_params)
        params.update(overrides)
        params["use_session"] = True
        result = run_ema_backtest(frame, params, collect_equity=False)
        metrics = compute_metrics(result)
        rows.append({**overrides, "use_session": True, **metrics})

    # Session ablation at defaults
    abl = dict(base_params)
    abl["use_session"] = False
    result = run_ema_backtest(frame, abl, collect_equity=False)
    metrics = compute_metrics(result)
    rows.append(
        {
            "atr_mult": abl["atr_mult"],
            "tp1_ratio": abl["tp1_ratio"],
            "tp2_ratio": abl["tp2_ratio"],
            "trail_atr_mult": abl["trail_atr_mult"],
            "use_session": False,
            **metrics,
        }
    )
    return rows


def _pick_winner(rows: list[dict]) -> dict | None:
    eligible = [
        row
        for row in rows
        if int(row.get("trades", 0)) >= MIN_TRADES
        and isinstance(row.get("expectancy"), (int, float))
        and math.isfinite(float(row["expectancy"]))
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: float(row["expectancy"]))


def _write_run(out_dir, result: dict, metrics: dict, params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if result["trades"] is not None:
        result["trades"].to_csv(out_dir / "trades.csv", index=False)
    if result["equity"] is not None and not result["equity"].empty:
        result["equity"].to_csv(out_dir / "equity.csv")
        save_chart(result["equity"], out_dir / "equity_drawdown.png")
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")
    (out_dir / "ema_inputs.txt").write_text(_format_ema_inputs(params) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the 20/50 EMA + pro management strategy")
    parser.add_argument("--config", default="config_ema.yaml")
    parser.add_argument("--start", help="Inclusive UTC start, yyyy-mm-dd")
    parser.add_argument("--end", help="Exclusive UTC end, yyyy-mm-dd")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the Parquet cache")
    parser.add_argument("--optimize", action="store_true", help="Run small expectancy grid")
    args = parser.parse_args()

    config = load_config(args.config)
    frame, load_stats = load_candles(config, refresh=args.refresh)
    frame = slice_dates(frame, args.start, args.end)
    if frame.empty:
        raise SystemExit("No candles in the requested date range.")

    base_params = resolve_ema_params(config)
    out_dir = project_path(config.get("output", {}).get("directory", "results_ema"))

    print(f"Candles: {len(frame):,}  {frame.index[0]} -> {frame.index[-1]}")
    if not load_stats.get("from_cache"):
        print(
            "Cleaned raw rows: "
            f"{load_stats.get('rows_out', 0):,} "
            f"(duplicates removed {load_stats.get('dropped_duplicates', 0):,}, "
            f"largest gap {load_stats.get('largest_gap_hours', 0):.1f}h)"
        )

    if args.optimize:
        rows = run_grid(frame, base_params)
        opt_frame = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        opt_path = out_dir / "optimization.csv"
        opt_frame.to_csv(opt_path, index=False)
        winner = _pick_winner(rows)
        if winner is None:
            raise SystemExit(f"No grid row reached min_trades={MIN_TRADES}")

        ranked = sorted(
            [
                row
                for row in rows
                if int(row.get("trades", 0)) >= MIN_TRADES
                and math.isfinite(float(row.get("expectancy", float("nan"))))
            ],
            key=lambda row: float(row["expectancy"]),
            reverse=True,
        )
        print()
        print(f"Grid rows: {len(rows)}  eligible: {len(ranked)}  wrote {opt_path}")
        print("Top 5 by expectancy:")
        for row in ranked[:5]:
            print(
                f"  exp={row['expectancy']:.4f}  pf={row['profit_factor']:.3f}  "
                f"net={row['net_profit']:.2f}  wr={row['win_rate']*100:.1f}%  "
                f"n={row['trades']}  atr={row['atr_mult']} tp1={row['tp1_ratio']} "
                f"tp2={row['tp2_ratio']} trail={row['trail_atr_mult']} "
                f"sess={row['use_session']}"
            )

        win_params = dict(base_params)
        for key in ("atr_mult", "tp1_ratio", "tp2_ratio", "trail_atr_mult", "use_session"):
            win_params[key] = winner[key]
        result = run_ema_backtest(frame, win_params)
        metrics = compute_metrics(result)
        _write_run(out_dir, result, metrics, win_params)
        print()
        print("Winner:")
        print(format_metrics(metrics))
        print()
        print(_format_ema_inputs(win_params))
    else:
        result = run_ema_backtest(frame, base_params)
        metrics = compute_metrics(result)
        _write_run(out_dir, result, metrics, base_params)
        print()
        print(format_metrics(metrics))
        print()
        print(_format_ema_inputs(base_params))

    print()
    print(f"Wrote {out_dir / 'trades.csv'}")
    print(f"Wrote {out_dir / 'equity.csv'}")
    print(f"Wrote {out_dir / 'equity_drawdown.png'}")


if __name__ == "__main__":
    main()

"""Run one backtest and write the trade list, equity curve, and chart.

    python -m src.backtest
    python -m src.backtest --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data_loader import load_candles, load_config, project_path, slice_dates
from src.report import compute_metrics, format_metrics, format_pine_inputs, save_chart
from src.strategy import resolve_params, run_backtest


def _jsonable(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the XAUUSD sweep strategy")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", help="Inclusive UTC start, yyyy-mm-dd")
    parser.add_argument("--end", help="Exclusive UTC end, yyyy-mm-dd")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the Parquet cache")
    args = parser.parse_args()

    config = load_config(args.config)
    frame, load_stats = load_candles(config, refresh=args.refresh)
    frame = slice_dates(frame, args.start, args.end)
    if frame.empty:
        raise SystemExit("No candles in the requested date range.")

    params = resolve_params(config)
    result = run_backtest(frame, params)
    metrics = compute_metrics(result)
    out_dir = project_path(config.get("output", {}).get("directory", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if result["trades"] is not None:
        result["trades"].to_csv(out_dir / "trades.csv", index=False)
    if result["equity"] is not None and not result["equity"].empty:
        result["equity"].to_csv(out_dir / "equity.csv")
        save_chart(result["equity"], out_dir / "equity_drawdown.png")
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")
    (out_dir / "pine_inputs.txt").write_text(format_pine_inputs(params) + "\n")

    print(f"Candles: {len(frame):,}  {frame.index[0]} -> {frame.index[-1]}")
    if not load_stats.get("from_cache"):
        print(
            "Cleaned raw rows: "
            f"{load_stats.get('rows_out', 0):,} "
            f"(duplicates removed {load_stats.get('dropped_duplicates', 0):,}, "
            f"largest gap {load_stats.get('largest_gap_hours', 0):.1f}h)"
        )
    print()
    print(format_metrics(metrics))
    print()
    print(format_pine_inputs(params))
    print()
    print(f"Wrote {out_dir / 'trades.csv'}")
    print(f"Wrote {out_dir / 'equity.csv'}")
    print(f"Wrote {out_dir / 'equity_drawdown.png'}")


if __name__ == "__main__":
    main()

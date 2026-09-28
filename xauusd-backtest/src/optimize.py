"""Grid or random search, plus walk-forward validation.

    python -m src.optimize
    python -m src.optimize --full-only
    python -m src.optimize --walk-forward-only
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_candles, load_config, project_path, slice_dates
from src.report import (
    compute_metrics,
    format_metrics,
    format_pine_inputs,
    format_side_by_side,
    rank_value,
    save_chart,
    save_heatmap,
    sharp_peak_notes,
)
from src.strategy import resolve_params, run_backtest

_WORKER: dict = {}


def expand_grid(space: dict) -> list[dict]:
    keys = list(space)
    values = []
    for key in keys:
        value = space[key]
        values.append(list(value) if isinstance(value, (list, tuple)) else [value])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def choose_combos(space: dict, method: str, n_random: int, seed: int) -> list[dict]:
    grid = expand_grid(space)
    if method == "random" and len(grid) > n_random:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(grid), size=n_random, replace=False)
        return [grid[int(i)] for i in picked]
    return grid


def _init_worker(parquet_path: str, base_params: dict) -> None:
    frame = pd.read_parquet(parquet_path)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    _WORKER["frame"] = frame
    _WORKER["base"] = base_params


def _evaluate(task: dict) -> dict:
    overrides = task["overrides"]
    params = dict(_WORKER["base"])
    params.update(overrides)
    frame = _WORKER["frame"]
    start = task.get("start")
    end = task.get("end")
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index < pd.Timestamp(end)]
    trade_after = pd.Timestamp(task["trade_after"]) if task.get("trade_after") else None
    if len(frame) < 50:
        metrics = {
            "net_profit": 0.0,
            "realized_profit": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "average_trade": 0.0,
            "expectancy": 0.0,
            "final_equity": float(params["account_size"]),
            "open_side": "flat",
            "open_qty": 0.0,
        }
    else:
        result = run_backtest(
            frame,
            params,
            collect_trades=False,
            collect_equity=False,
            trade_after=trade_after,
        )
        metrics = compute_metrics(result)
    score = rank_value(metrics, task["rank_by"], task["min_trades"])
    return {"params": overrides, "score": score, "metrics": metrics}


def _run_tasks(tasks: list[dict], parquet_path: Path, base_params: dict, workers: int) -> list[dict]:
    if not tasks:
        return []
    if workers <= 1:
        _init_worker(str(parquet_path), base_params)
        return [_evaluate(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(parquet_path), base_params),
    ) as pool:
        return list(pool.map(_evaluate, tasks, chunksize=1))


def _window_bounds(index: pd.DatetimeIndex, spec: dict) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []
    start = index[0]
    end = index[-1]
    in_days = int(spec["in_sample_days"])
    out_days = int(spec["out_sample_days"])
    step = int(spec["step_days"])
    warmup = pd.Timedelta(days=int(spec.get("warmup_days", 20)))
    windows = []
    cursor = start
    while True:
        is_end = cursor + pd.Timedelta(days=in_days)
        oos_end = is_end + pd.Timedelta(days=out_days)
        if oos_end > end + pd.Timedelta(days=1):
            break
        windows.append((cursor - warmup, cursor, is_end, oos_end))
        cursor = cursor + pd.Timedelta(days=step)
    return windows


def _print_top(rows: list[dict], limit: int = 10) -> None:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    for rank, row in enumerate(ranked, start=1):
        metrics = row["metrics"]
        pf = metrics["profit_factor"]
        pf_text = "inf" if isinstance(pf, float) and math.isinf(pf) else f"{pf:.2f}"
        print(
            f"{rank:>2}. score {row['score']:.3f}  "
            f"sharpe {metrics['sharpe']:.2f}  pf {pf_text}  "
            f"trades {metrics['trades']}  net {metrics['net_profit']:.0f}  "
            f"{row['params']}"
        )


def _full_tasks(combos, rank_by, min_trades, start, end) -> list[dict]:
    return [
        {
            "overrides": combo,
            "rank_by": rank_by,
            "min_trades": min_trades,
            "start": start,
            "end": end,
            "trade_after": start,
        }
        for combo in combos
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the XAUUSD sweep strategy")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--walk-forward-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    frame, _stats = load_candles(config, refresh=args.refresh)
    frame = slice_dates(frame, args.start, args.end)
    if frame.empty:
        raise SystemExit("No candles in the requested date range.")

    opt = config["optimize"]
    method = str(opt.get("method", "grid"))
    combos = choose_combos(opt["ranges"], method, int(opt.get("n_random", 200)), int(opt.get("seed", 7)))
    rank_by = str(opt.get("rank_by", "sharpe"))
    min_trades = int(opt.get("min_trades", 30))
    workers = int(opt.get("workers") or 0) or os.cpu_count() or 1
    base = resolve_params(config)
    out_dir = project_path(config.get("output", {}).get("directory", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "_optimize_cache.parquet"
    frame.to_parquet(parquet_path)

    do_full = not args.walk_forward_only
    do_walk = not args.full_only
    best_params = dict(base)

    if do_full:
        print(f"Full sample: {len(combos)} combinations on {len(frame):,} bars, {workers} workers")
        rows = _run_tasks(
            _full_tasks(combos, rank_by, min_trades, None, None),
            parquet_path,
            base,
            workers,
        )
        _print_top(rows)
        best = max(rows, key=lambda row: (row["score"], row["metrics"]["trades"], row["metrics"]["sharpe"]))
        best_params.update(best["params"])
        notes = sharp_peak_notes(
            rows,
            opt["ranges"],
            float(opt.get("overfit_peak_ratio", 1.5)),
            metric=rank_by,
        )
        table = pd.DataFrame([{**row["params"], **row["metrics"], "score": row["score"]} for row in rows])
        table.sort_values("score", ascending=False).to_csv(out_dir / "optimization.csv", index=False)
        heatmap = opt.get("heatmap") or {}
        x_key = heatmap.get("x", "atr_sl_mult")
        y_key = heatmap.get("y", "tp2_ratio")
        if x_key in opt["ranges"] and y_key in opt["ranges"]:
            save_heatmap(rows, x_key, y_key, "score", out_dir / "heatmap.png")
        print()
        print("Best full-sample inputs")
        print(format_pine_inputs(best_params))
        print()
        print(format_metrics(best["metrics"]))
        if notes:
            print()
            print("Overfit checks")
            for note in notes:
                print(f"- {note}")
        (out_dir / "best_params.txt").write_text(
            format_pine_inputs(best_params) + ("\n\n" + "\n".join(notes) if notes else "") + "\n"
        )
        chosen = resolve_params(config, best["params"])
        rerun = run_backtest(frame, chosen, collect_trades=True, collect_equity=True)
        if rerun["trades"] is not None:
            rerun["trades"].to_csv(out_dir / "best_trades.csv", index=False)
        if rerun["equity"] is not None and not rerun["equity"].empty:
            rerun["equity"].to_csv(out_dir / "best_equity.csv")
            save_chart(rerun["equity"], out_dir / "best_equity_drawdown.png")

    if do_walk:
        spec = opt["walk_forward"]
        # Reuse the on-disk frame bounds. Workers slice by timestamp.
        bounds = _window_bounds(frame.index, spec)
        if not bounds:
            print("Not enough history for a walk-forward window. Lower in_sample_days or download more data.")
        else:
            print(f"Walk-forward: {len(bounds)} windows x {len(combos)} combinations")
            summaries = []
            oos_left = None
            oos_right = None
            for window_id, (warmup_start, is_start, is_end, oos_end) in enumerate(bounds, start=1):
                print(f"Window {window_id}: in-sample {is_start.date()} -> {is_end.date()}")
                is_tasks = [
                    {
                        "overrides": combo,
                        "rank_by": rank_by,
                        "min_trades": int(spec.get("min_trades", 10)),
                        "start": warmup_start.isoformat(),
                        "end": is_end.isoformat(),
                        "trade_after": is_start.isoformat(),
                    }
                    for combo in combos
                ]
                is_rows = _run_tasks(is_tasks, parquet_path, base, workers)
                winner = max(
                    is_rows,
                    key=lambda row: (row["score"], row["metrics"]["trades"], row["metrics"]["sharpe"]),
                )
                oos_task = {
                    "overrides": winner["params"],
                    "rank_by": rank_by,
                    "min_trades": 0,
                    "start": (is_end - pd.Timedelta(days=int(spec.get("warmup_days", 20)))).isoformat(),
                    "end": oos_end.isoformat(),
                    "trade_after": is_end.isoformat(),
                }
                oos = _evaluate(oos_task) if workers <= 1 else _run_tasks([oos_task], parquet_path, base, workers)[0]
                print(format_side_by_side(winner["metrics"], oos["metrics"]))
                print(f"Chosen: {winner['params']}")
                print()
                summaries.append(
                    {
                        "window": window_id,
                        "is_start": str(is_start.date()),
                        "is_end": str(is_end.date()),
                        "oos_end": str(oos_end.date()),
                        **{f"param_{key}": value for key, value in winner["params"].items()},
                        **{f"is_{key}": value for key, value in winner["metrics"].items()},
                        **{f"oos_{key}": value for key, value in oos["metrics"].items()},
                    }
                )
                oos_left = winner["metrics"]
                oos_right = oos["metrics"]
                best_params.update(winner["params"])
            pd.DataFrame(summaries).to_csv(out_dir / "walk_forward.csv", index=False)
            if oos_left and oos_right:
                print("Last window, in-sample vs out-of-sample")
                print(format_side_by_side(oos_left, oos_right))
            # Stitched out-of-sample totals from the per-window metrics.
            oos_profit = sum(row["oos_net_profit"] for row in summaries)
            oos_trades = sum(int(row["oos_trades"]) for row in summaries)
            print()
            print(f"Walk-forward OOS net profit {oos_profit:.2f} across {oos_trades} trades")
            latest_params = dict(base)
            for key, value in summaries[-1].items():
                if key.startswith("param_"):
                    latest_params[key.removeprefix("param_")] = value
            (out_dir / "walk_forward_latest.txt").write_text(format_pine_inputs(latest_params) + "\n")
            print()
            print("Latest walk-forward window")
            print(format_pine_inputs(latest_params))

    if parquet_path.exists():
        parquet_path.unlink()


if __name__ == "__main__":
    main()

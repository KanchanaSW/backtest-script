# XAUUSD backtest

Local backtest and parameter search for the TradingView strategy
`Merged 1H Sweep + Pro Management + Volume Filter` on Dukascopy gold data.

The chart timeframe defaults to 5 minutes. The sweep uses the previous completed
1-hour high and low. Signals are evaluated on the bar close and market orders
fill on the next bar's open. Stop and target distances stay locked to the signal
bar's close, which is what the Pine script does.

## Setup

Python 3.9+ and Node.js (for `npx`) are required.

```bash
cd xauusd-backtest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

`scripts/download_data.py` calls `dukascopy-node` and saves bid candles as CSV
under `data/raw`. One-minute data is the source. The loader resamples it to the
`chart_timeframe` in `config.yaml`, so you do not need separate 5m, 15m, or 1h
downloads. Flat minutes are included so bar counts stay on the clock.

A full pull from 2015 is large (several hundred MB) and can take a while.
Dukascopy is fetched in yearly files, and files that already exist are skipped.

```bash
python scripts/download_data.py
```

Shorter ranges are useful while checking the setup:

```bash
python scripts/download_data.py --from 2024-01-01 --to 2024-02-01
```

Optional raw timeframes:

```bash
python scripts/download_data.py --timeframes m1 m5 m15 h1
```

## Backtest

```bash
python -m src.backtest
python -m src.backtest --start 2020-01-01 --end 2024-01-01
```

The first run cleans the CSVs, resamples them, and writes
`data/processed/xauusd_5min.parquet`. Later runs reuse that file until a raw
CSV changes.

Outputs in `results/`:

- `trades.csv` — one row per filled partial (TP1 and the runner are separate)
- `equity.csv` — mark-to-market equity and drawdown
- `equity_drawdown.png`
- `metrics.json`
- `pine_inputs.txt` — the inputs used for this run

Reported metrics are net profit, win rate, profit factor, annualized daily
Sharpe, max drawdown, trade count, average trade, and expectancy. Win rate
counts each partial close, matching `strategy.closedtrades`.

Costs live under `broker` in `config.yaml`. The default spread is 0.30 USD on
bid data: long entries pay the spread, short exits pay the spread. Commission
and slippage default to zero.

Friday flattening and the daily loss reset use `fixed.chart_timezone` (UTC).
The session filter is off by default. When enabled, `trading_hours` is read in
`session_tz` (`America/New_York`).

## Optimize

Ranges live in `optimize.ranges`. `method` is `grid` or `random`.
Results are ranked by Sharpe (or profit factor) and discarded when the trade
count is below `min_trades`.

```bash
python -m src.optimize
python -m src.optimize --full-only
python -m src.optimize --walk-forward-only
```

Walk-forward optimizes on each in-sample window and records the next
out-of-sample window without refitting. The search uses every CPU unless
`optimize.workers` is set. On a full 2015–today history, `--full-only` is the
short run. Walk-forward repeats that search once per window, so narrow
`optimize.ranges` before leaving it running.

Extra files:

- `optimization.csv` — every full-sample combination
- `heatmap.png` — the two inputs named in `optimize.heatmap`, with the other inputs held at the best row
- `best_params.txt` — full-sample winner, plus a warning when a value sits on a sharp peak or on the edge of the grid
- `walk_forward.csv` — in-sample and out-of-sample metrics for each window
- `walk_forward_latest.txt` — inputs chosen on the most recent in-sample window
- `best_trades.csv`, `best_equity.csv`, `best_equity_drawdown.png`

A sharp peak means the best score is much higher than the neighboring grid
points (`optimize.overfit_peak_ratio`, default 1.5). Prefer a nearby plateau
over that point.

## How to move results back into TradingView

1. Run `python -m src.optimize` after the history you care about is downloaded.
2. Open `results/best_params.txt` for the full-sample winner, or
   `results/walk_forward_latest.txt` for the inputs that were chosen on the
   latest in-sample window and then checked on the following out-of-sample window.
3. Use the out-of-sample column in the walk-forward printout, and
   `walk_forward.csv`, as the performance check. The full-sample net profit
   is the in-sample number.
4. On TradingView, open this strategy on **XAUUSD, 5-minute**. The 1-hour sweep
   is an input, not the chart timeframe.
5. Paste each `name = value` line into the input with the same name, in the
   order printed. Booleans are `true` / `false`.
6. Leave the visual inputs as they are (`show_boxes` and the volume-profile
   group). They do not change orders. The confidence score on the labels is
   display-only as well.
7. If the file warns about a sharp peak, move that input to a neighbor that
   scored close to it instead of using the single best cell.
8. Match costs as closely as the TradingView strategy properties allow. This
   project charges a 0.30 USD spread on bid data. Pine will not include that
   unless you set slippage or commission on the strategy itself.

## Assumptions that differ from a raw paste of the script

- Volume-profile levels and the buy/sell confidence score are not simulated.
- After TP1 fills, the position scales out once. The script keeps calling
  `strategy.exit("TP1")`, which can scale out a second time; that is not copied.
- The daily loss limit counts each partial close, and only one step per bar.
  A winning partial on a bar resets the counter. A losing partial on the same
  bar takes priority.
- Same-bar stop and target conflicts follow the TradingView path: open to the
  nearer extreme, then the other extreme, then the close.
- Dukascopy bid candles and tick volume will not line up bar-for-bar with a
  TradingView broker feed, so trade lists will not be identical even when the
  inputs match.

## Tests

```bash
pytest
```

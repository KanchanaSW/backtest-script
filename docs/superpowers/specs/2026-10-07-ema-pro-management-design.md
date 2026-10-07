# 20/50 EMA + Pro Management Design

Date: 2026-10-07  
Status: approved for implementation planning  
Instrument / TF: XAUUSD 15-minute  
Goal: maximize net profit / expectancy (win rate secondary)

## Context

Baseline 15m backtests of the simple 20/50 EMA strategy were near break-even or slightly negative. The latest Python port diverged from Pine by dropping the 200 EMA filter and fixed RR take-profit, and by reversing on opposite crossover. Most closed trades hit the ATR stop; expectancy suffered.

Chosen approach: keep the 20/50 crossover entry core, restore trend/session filters, and add partial take-profit + break-even + ATR trail management (same family as the existing scalper / sweep strategies).

## Success criteria

On the same 3-year window (`2023-10-01` → `2026-10-01`, Dukascopy bid, $0.30 spread):

1. Net profit and expectancy beat the current EMA baseline.
2. Profit factor ≥ 1.1 preferred.
3. Enough sample size: at least ~80 closed trade legs (partials count separately, matching existing metrics).
4. Win rate is secondary; ~40–50% is acceptable if expectancy improves.

## Entry rules

Chart: 15-minute XAUUSD.

Indicators:

- Fast EMA length 20 (input)
- Slow EMA length 50 (input)
- Trend EMA length 200 (input)
- ATR length 14

Signals (evaluated on bar close; market order fills next open):

- Long: `crossover(fast_ema, slow_ema)` and `close > trend_ema`
- Short: `crossunder(fast_ema, slow_ema)` and `close < trend_ema`
- Session filter enabled by default: `0300-1600` America/New_York
- New entries only when flat (`position_size == 0`)
- No reverse-on-opposite-crossover

Position sizing: `strategy.percent_of_equity` = 10% (unchanged). Initial capital $10,000.

## Trade management

Levels locked to the signal bar’s close and ATR (not rebased to fill price), matching existing backtest conventions.

1. **Stop:** `ATR × atr_mult` (default 1.5)
2. **TP1:** `tp1_ratio × R` (default 1.0R) — scale out `scale_pct` (default 50%)
3. **Break-even:** after TP1, move stop to entry ± `be_offset` (default 0.30, same units as price / spread)
4. **Runner:** trail remaining size with `ATR × trail_atr_mult` (default 1.0)
5. **TP2 cap:** `tp2_ratio × R` (default 2.0; grid also tries 2.5) hard limit on the runner
6. **Friday flatten:** close open positions at 16:00 chart timezone (UTC)

Same-bar stop/target resolution follows the existing TradingView bar-path helper (`open` → nearer extreme → far extreme → `close`).

## Tunable parameters

Small grid search on the 3y window, ranked by expectancy (fallback: net profit), with `min_trades` ≈ 80:

| Param | Defaults / candidates |
|---|---|
| `atr_mult` | 1.2, 1.5, 2.0 |
| `tp1_ratio` | 0.8, 1.0, 1.2 |
| `tp2_ratio` | 2.0, 2.5 |
| `scale_pct` | 50 |
| `trail_atr_mult` | 0.8, 1.0, 1.2 |
| `use_session` | true (default), false as one ablation |

Fixed unless search shows a clear edge: EMA lengths 20/50/200, `qty_pct` 10, session `0300-1600` NY.

## Deliverables

1. Update `20-50ema-strategy` Pine script to the rules above.
2. Update Python `xauusd-backtest/src/ema_strategy.py` and `config_ema.yaml` to match.
3. Extend / reuse existing exit planning helpers where possible (`plan_exits`, trail/BE patterns from scalper or sweep strategy).
4. Add focused unit tests for TP1 scale-out, BE move, and trail/TP2.
5. Run baseline vs improved backtest on the 3y window; write metrics under `results_ema/`.
6. Optional small optimizer script or one-off grid in the EMA backtest module — keep scope small (no full walk-forward in v1).

## Out of scope

- RSI / volume / HTF dual-EMA filters
- Replacing the crossover signal
- Walk-forward optimization (can follow if v1 clears success criteria)
- Changing broker cost model beyond existing spread defaults

## Assumptions

- Dukascopy bid candles will not match TradingView broker bars trade-for-trade.
- Partial closes are counted as separate closed trades in metrics (existing convention).
- Pine and Python must share the same input names and default values after the update.

# Specification: Merged 1H Sweep + Pro Management Pine Script v6 Strategy

**Date:** 2026-09-28  
**Topic:** TradingView Strategy Script for XAUUSD 1H Sweep with Pro Management  
**Status:** Approved

---

## 1. Executive Summary
Provide a production-ready, Pine Script v6 (`//@version=6`) strategy script for TradingView designed for XAUUSD on the 5-minute timeframe. The strategy incorporates HTF liquidity sweeps, market structure shifts, fair value gap entries, trend and chase filters, partial profit scaling with break-even/trailing stops, a daily circuit breaker, and an optional toggle for intrabar volume profile calculation.

---

## 2. Requirements & Inputs

### 2.1 Default Parameters (Tuned to Optimal Backtest)
- `account_size`: `5000.0` (USD)
- `lot_size`: `0.1` (10 units for Gold with `units_per_lot = 100`)
- `atr_sl_mult`: `2.5`
- `tp1_ratio`: `1.5`
- `tp2_ratio`: `2.5`
- `scale_pct`: `50%`
- `use_trail`: `true` (trails at `1.5 * ATR` after TP1 is hit)
- `max_daily_loss`: `2` (stops trading after 2 consecutive daily losses)
- `close_friday`: `true` (closes all positions at 16:00 Friday)
- `use_session`: `true` (`0300-1200` in `America/New_York`)
- `use_vol`: `true` (`vol_length = 30`)
- `use_bias`: `true` (`bias_tf = "D"`, `bias_length = 20`)
- `max_chase_atr`: `0.75`
- `htf`: `"60"` (1-hour timeframe)
- `swing_len`: `3`
- `fvg_timeout`: `10`
- `enable_vp`: `false` (Toggleable intrabar volume profile, disabled by default for maximum performance)

---

## 3. Order Execution & Strategy Lifecycle

1. **Order Engine**:
   - `strategy("Merged 1H Sweep + Pro Management [Optimized]", overlay=true, initial_capital=5000, margin_long=10, margin_short=10, max_labels_count=500, max_boxes_count=500, max_lines_count=500)`
   - Evaluated on bar close, fills on next bar open (`process_orders_on_close = false`).
2. **Exits**:
   - Long: `strategy.exit("TP1", "Long", qty=qty1, stop=sl_p, limit=tp1_p)`
   - Long Runner: `strategy.exit("TP2", "Long", stop=sl_p, limit=tp2_p)`
   - Trailing: Upon TP1 trigger, `sl_p` advances to `entry_p`, and subsequently trails `high - 1.5 * ATR`.
   - Short mirrors symmetrically.
3. **Safety**:
   - Weekend Flatten: `dayofweek == dayofweek.friday and hour >= 16` -> `strategy.close_all("Weekend Flatten")`.
   - Circuit Breaker: If `daily_losses >= max_daily_loss`, `can_trade := false`.

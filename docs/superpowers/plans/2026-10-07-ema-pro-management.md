# 20/50 EMA + Pro Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the 20/50 EMA 15m XAUUSD strategy with trend/session filters and partial TP + BE + ATR trail management so 3y expectancy/net profit beat the current baseline.

**Architecture:** Keep crossover entries; restore 200 EMA + session gates; replace stop-only/reverse exits with the same two-leg exit model used in `src/strategy.py` (`plan_exits` + TP1 scale, BE, trail, TP2). Pine (`20-50ema-strategy`) and Python (`ema_strategy.py`) share input names/defaults. Small expectancy-ranked grid on the 3y window, then one final backtest write to `results_ema/`.

**Tech Stack:** Python 3.9+, pandas/numpy, existing `xauusd-backtest` helpers, Pine Script v5, pytest.

## Global Constraints

- Chart: XAUUSD `15min`; window `2023-10-01` → `2026-10-01` (exclusive end).
- Broker: Dukascopy bid, `spread = 0.30`, `commission_per_unit = 0.0`.
- Sizing: `account_size = 10000`, `qty_pct = 10` (percent of equity).
- Entries only when flat; no reverse-on-cross.
- Levels locked to signal-bar `close` + ATR (not rebased to fill).
- Rank grid by `expectancy`, `min_trades = 80`.
- Commit messages ≤ 30 words when committing.
- Do not commit unless the user explicitly asks (skip commit steps otherwise).

## File structure

| File | Responsibility |
|---|---|
| `xauusd-backtest/config_ema.yaml` | 15m data + strategy/broker defaults |
| `xauusd-backtest/src/ema_strategy.py` | Param resolve + bar-by-bar EMA pro backtest |
| `xauusd-backtest/src/backtest_ema.py` | CLI runner + optional small grid |
| `xauusd-backtest/tests/test_ema_strategy.py` | Unit tests for params, exits, management |
| `20-50ema-strategy` | Matching Pine strategy |
| `xauusd-backtest/results_ema/` | metrics/trades/equity outputs |

Reuse (do not duplicate): `pine_ema` from `scalper_strategy.py`; `pine_atr`, `plan_exits`, `execution_price`, `session_mask`, `bar_path` from `strategy.py`.

---

### Task 1: Config and param defaults

**Files:**
- Modify: `xauusd-backtest/config_ema.yaml`
- Modify: `xauusd-backtest/src/ema_strategy.py` (`EMA_INPUT_ORDER`, `resolve_ema_params`)
- Modify: `xauusd-backtest/tests/test_ema_strategy.py`

**Interfaces:**
- Produces: `resolve_ema_params(config, overrides=None) -> dict` with keys below
- Consumes: existing `load_config` YAML shape (`strategy` / `broker` / `fixed`)

Required defaults after resolve:

```python
{
    "account_size": 10000.0,
    "qty_pct": 10.0,
    "fast_len": 20,
    "slow_len": 50,
    "trend_len": 200,
    "atr_length": 14,
    "atr_mult": 1.5,
    "tp1_ratio": 1.0,
    "tp2_ratio": 2.0,
    "scale_pct": 50,
    "use_trail": True,
    "trail_atr_mult": 1.0,
    "be_offset": 0.30,
    "use_session": True,
    "trading_hours": "0300-1600",
    "session_tz": "America/New_York",
    "close_friday": True,
    "spread": 0.30,
    "slippage": 0.0,
    "commission_per_unit": 0.0,
    "margin_pct": 10.0,
    "chart_timezone": "UTC",
}
```

- [ ] **Step 1: Write the failing test**

Replace `test_resolve_defaults` in `tests/test_ema_strategy.py` with:

```python
def test_resolve_defaults():
    params = resolve_ema_params({"strategy": {}, "broker": {}, "fixed": {}})
    assert params["fast_len"] == 20
    assert params["slow_len"] == 50
    assert params["trend_len"] == 200
    assert params["tp1_ratio"] == 1.0
    assert params["tp2_ratio"] == 2.0
    assert params["scale_pct"] == 50
    assert params["use_trail"] is True
    assert params["trail_atr_mult"] == 1.0
    assert params["be_offset"] == 0.30
    assert params["use_session"] is True
    assert params["trading_hours"] == "0300-1600"
    assert params["close_friday"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd xauusd-backtest && .venv/bin/pytest tests/test_ema_strategy.py::test_resolve_defaults -v`  
Expected: FAIL on missing keys (`trend_len` / `tp1_ratio` / etc.)

- [ ] **Step 3: Update `config_ema.yaml`**

```yaml
# 20/50 EMA + Pro Management — 15-minute XAUUSD

data:
  instrument: xauusd
  price_type: bid
  date_from: "2015-01-01"
  date_to: now
  download_timeframes: ["m1"]
  chart_timeframe: "15min"
  timezone: "UTC"
  include_flats: true
  raw_dir: data/raw
  processed_dir: data/processed

broker:
  spread: 0.30
  slippage: 0.0
  commission_per_unit: 0.0
  margin_pct: 10.0

strategy:
  account_size: 10000.0
  qty_pct: 10.0
  fast_len: 20
  slow_len: 50
  trend_len: 200
  atr_mult: 1.5
  atr_length: 14
  tp1_ratio: 1.0
  tp2_ratio: 2.0
  scale_pct: 50
  use_trail: true
  trail_atr_mult: 1.0
  be_offset: 0.30
  use_session: true
  trading_hours: "0300-1600"
  session_tz: "America/New_York"
  close_friday: true

fixed:
  atr_length: 14
  chart_timezone: "UTC"

output:
  directory: results_ema
```

- [ ] **Step 4: Update `EMA_INPUT_ORDER` and `resolve_ema_params` defaults**

Set `EMA_INPUT_ORDER` to:

```python
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
```

Add the matching `setdefault` calls in `resolve_ema_params` for every key in the defaults table above.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd xauusd-backtest && .venv/bin/pytest tests/test_ema_strategy.py::test_resolve_defaults -v`  
Expected: PASS

- [ ] **Step 6: Commit (only if user asked)**

```bash
git add xauusd-backtest/config_ema.yaml xauusd-backtest/src/ema_strategy.py xauusd-backtest/tests/test_ema_strategy.py
git commit -m "$(cat <<'EOF'
feat: add EMA pro-management config defaults

EOF
)"
```

---

### Task 2: Pro-management backtest engine

**Files:**
- Modify: `xauusd-backtest/src/ema_strategy.py`
- Modify: `xauusd-backtest/tests/test_ema_strategy.py`

**Interfaces:**
- Consumes: `plan_exits`, `execution_price`, `pine_atr`, `session_mask` from `src.strategy`; `pine_ema` from `src.scalper_strategy`
- Produces: `run_ema_backtest(frame, params, *, collect_trades=True, collect_equity=True, trade_after=None) -> dict` with the same result keys as `run_backtest` / `run_scalper_backtest` (`pnls`, `trades`, `equity`, `daily_equity`, `max_drawdown`, `max_drawdown_pct`, `final_equity`, `initial_capital`, `open_qty`, `open_side`)

Remove `plan_stop_exit` (replaced by `plan_exits`). Keep signals/management logic inside `run_ema_backtest`.

- [ ] **Step 1: Write failing management tests**

Add to `tests/test_ema_strategy.py`:

```python
def _params(**overrides):
    base = resolve_ema_params(
        {
            "strategy": {
                "account_size": 10000.0,
                "qty_pct": 10.0,
                "use_session": False,
                "close_friday": False,
            },
            "broker": {"spread": 0.0, "slippage": 0.0, "commission_per_unit": 0.0},
            "fixed": {},
        }
    )
    base.update(overrides)
    return base


def test_long_scales_at_tp1_then_be_stop():
    """Synthetic: cross up, hit TP1, then stop at BE+offset on runner."""
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 2000.0)
    # Force a clean uptrend cross around bar 210, then spike to TP1, then fade to BE
    close[:210] = np.linspace(1980, 2000, 210)
    close[210:] = 2005.0
    close[220] = 2020.0  # TP1 area after entry
    close[221:] = 2004.0
    high = close + 0.5
    low = close - 0.5
    high[220] = 2030.0
    low[222] = 1990.0  # take runner BE stop
    open_ = close.copy()
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )
    result = run_ema_backtest(
        frame,
        _params(atr_mult=1.0, tp1_ratio=1.0, tp2_ratio=5.0, scale_pct=50, use_trail=True, be_offset=0.3),
    )
    reasons = result["trades"]["reason"].tolist() if result["trades"] is not None else []
    assert "tp1" in reasons


def test_no_entry_when_against_trend_ema():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # Price always below a rising series that will sit under a high EMA200 seed — use falling market
    close = np.linspace(2200, 2000, n)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    result = run_ema_backtest(frame, _params())
    sides = [] if result["trades"] is None else result["trades"]["side"].tolist()
    assert "long" not in sides  # downtrend: longs blocked by trend filter
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd xauusd-backtest && .venv/bin/pytest tests/test_ema_strategy.py::test_long_scales_at_tp1_then_be_stop tests/test_ema_strategy.py::test_no_entry_when_against_trend_ema -v`  
Expected: FAIL (old engine has no `tp1` reason / no trend filter)

- [ ] **Step 3: Rewrite `run_ema_backtest`**

Implementation sketch (full rewrite of the loop body):

```python
from src.strategy import (
    bar_path,  # only if still needed locally; prefer plan_exits
    execution_price,
    pine_atr,
    plan_exits,
    session_mask,
)

# Precompute:
fast_ema = pine_ema(close, fast_len)
slow_ema = pine_ema(close, slow_len)
trend_ema = pine_ema(close, trend_len)
atr = pine_atr(high, low, close, atr_len)
in_session = session_mask(index, trading_hours, session_tz, use_session)

# State:
# side, pos_qty1, pos_qty2, entry_fill, entry_ref, sl, tp1, tp2, tp1_hit
# pending: dict | None
# pending_flatten: bool

# Each bar order:
# 1) Friday flatten if pending_flatten
# 2) Fill pending entry on open ONLY if side == 0
#    qty = (equity * qty_pct/100) / entry_px
#    qty1 = qty * scale_pct/100; qty2 = qty - qty1
#    entry_ref / sl / tp1 / tp2 from pending (signal close based)
# 3) If in position: plan_exits(...); record per-leg exits (reasons tp1/tp2/stop/friday)
# 4) After exits, if still in position: BE + trail update (same as strategy.py):
#      long: on first high>=tp1 -> tp1_hit; sl = entry_ref + be_offset
#            then sl = max(sl, high - atr*trail_atr_mult) if use_trail
#      short: mirror with entry_ref - be_offset
# 5) If flat and pending is None and allowed and in_session:
#      long_cross & close > trend_ema -> pending long
#      short_cross & close < trend_ema -> pending short
#    Do NOT queue if side != 0
# 6) If close_friday and Friday>=16:00 chart TZ -> pending_flatten = True
```

Pending payload fields: `side`, `entry_ref`, `sl`, `tp1`, `tp2`, `signal_i`.

Long levels at signal `i`:

```python
risk = atr[i] * atr_mult
entry_ref = close[i]
sl = entry_ref - risk
tp1 = entry_ref + risk * tp1_ratio
tp2 = entry_ref + risk * tp2_ratio
```

Short: mirror.

Delete reverse-on-cross behavior entirely.

- [ ] **Step 4: Run EMA tests**

Run: `cd xauusd-backtest && .venv/bin/pytest tests/test_ema_strategy.py -v`  
Expected: all PASS (update/remove obsolete stop-only / reverse tests if they conflict)

- [ ] **Step 5: Commit (only if user asked)**

```bash
git add xauusd-backtest/src/ema_strategy.py xauusd-backtest/tests/test_ema_strategy.py
git commit -m "$(cat <<'EOF'
feat: EMA crossover with partial TP BE trail

EOF
)"
```

---

### Task 3: CLI runner + small expectancy grid

**Files:**
- Modify: `xauusd-backtest/src/backtest_ema.py`

**Interfaces:**
- Consumes: `resolve_ema_params`, `run_ema_backtest`, `EMA_INPUT_ORDER`, `compute_metrics`, `load_candles`, `slice_dates`
- Produces: CLI flags `--start`, `--end`, `--refresh`, `--optimize`; writes `results_ema/*`

- [ ] **Step 1: Extend `backtest_ema.py`**

Keep existing single-run path. Add:

```python
GRID = {
    "atr_mult": [1.2, 1.5, 2.0],
    "tp1_ratio": [0.8, 1.0, 1.2],
    "tp2_ratio": [2.0, 2.5],
    "trail_atr_mult": [0.8, 1.0, 1.2],
}
MIN_TRADES = 80


def run_grid(frame, base_params: dict) -> list[dict]:
    from itertools import product
    rows = []
    keys = list(GRID.keys())
    for values in product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        params = dict(base_params)
        params.update(overrides)
        # also try session ablation once at defaults only — handled separately
        result = run_ema_backtest(frame, params, collect_equity=False)
        metrics = compute_metrics(result)
        row = {**overrides, **metrics}
        rows.append(row)
    return rows
```

CLI:

```python
parser.add_argument("--optimize", action="store_true")
```

When `--optimize`:

1. Run grid with `use_session=True`.
2. Run one ablation row: best-so-far knobs with `use_session=False` (or include `use_session` in an extra loop of `[True]` plus one False at defaults).
3. Rank rows with `expectancy` where `trades >= MIN_TRADES`; ignore `-inf`.
4. Write `results_ema/optimization.csv`.
5. Re-run the winner with full equity collection; write trades/equity/metrics/`ema_inputs.txt`.
6. Print top 5 rows + winner metrics.

When not optimizing: current single-run behavior with updated input dump.

- [ ] **Step 2: Smoke-run a short window**

Run:

```bash
cd xauusd-backtest && .venv/bin/python -m src.backtest_ema --start 2024-01-01 --end 2024-04-01
```

Expected: prints metrics, writes `results_ema/metrics.json`, no traceback. Trade reasons include some of `tp1` / `tp2` / `stop` (not `reverse`).

- [ ] **Step 3: Commit (only if user asked)**

```bash
git add xauusd-backtest/src/backtest_ema.py
git commit -m "$(cat <<'EOF'
feat: add EMA expectancy grid optimize flag

EOF
)"
```

---

### Task 4: Matching Pine script

**Files:**
- Modify: `20-50ema-strategy`

**Interfaces:**
- Produces: Pine inputs matching `EMA_INPUT_ORDER` names/defaults from Task 1
- Consumes: none (standalone TradingView script)

- [ ] **Step 1: Rewrite `20-50ema-strategy`**

Replace contents with a v5 strategy that mirrors Python:

```pine
//@version=5
strategy("20/50 EMA + Pro Management", overlay=true,
     initial_capital=10000,
     default_qty_type=strategy.percent_of_equity,
     default_qty_value=10,
     margin_long=10, margin_short=10,
     max_boxes_count=500)

fast_len       = input.int(20, "Fast EMA Length")
slow_len       = input.int(50, "Slow EMA Length")
trend_len      = input.int(200, "Trend Filter EMA")
atr_mult       = input.float(1.5, "ATR Stop Multiplier", step=0.1)
tp1_ratio      = input.float(1.0, "TP1 Risk/Reward Ratio", step=0.1)
tp2_ratio      = input.float(2.0, "TP2 Risk/Reward Ratio", step=0.1)
scale_pct      = input.int(50, "TP1 Scale Out %", minval=10, maxval=90)
use_trail      = input.bool(true, "Trail Stop Loss after TP1 Hit")
trail_atr_mult = input.float(1.0, "Trail ATR Mult", step=0.1)
be_offset      = input.float(0.30, "BE Offset after TP1", step=0.05)
use_session    = input.bool(true, "Use Trading Session Filter")
trading_hours  = input.session("0300-1600", "Trading Hours (NY)", inline="sess")
session_tz     = input.string("America/New_York", "Session Timezone")
close_friday   = input.bool(true, "Close All Trades Friday at 16:00")

fast_ema  = ta.ema(close, fast_len)
slow_ema  = ta.ema(close, slow_len)
trend_ema = ta.ema(close, trend_len)
atr_val   = ta.atr(14)
trail_dist = atr_val * trail_atr_mult

in_session = not use_session or not na(time(timeframe.period, trading_hours, session_tz))
long_trend = close > trend_ema
short_trend = close < trend_ema
long_cond = ta.crossover(fast_ema, slow_ema) and long_trend and in_session
short_cond = ta.crossunder(fast_ema, slow_ema) and short_trend and in_session

var float entry_p = na
var float sl_p = na
var float tp1_p = na
var float tp2_p = na
var bool tp1_hit = false

qty_total = strategy.equity * 0.10 / close
qty1 = qty_total * (scale_pct / 100.0)

if close_friday and dayofweek == dayofweek.friday and hour >= 16 and strategy.position_size != 0
    strategy.close_all("Friday")

if long_cond and strategy.position_size == 0
    entry_p := close
    sl_distance = atr_val * atr_mult
    sl_p := entry_p - sl_distance
    tp1_p := entry_p + sl_distance * tp1_ratio
    tp2_p := entry_p + sl_distance * tp2_ratio
    tp1_hit := false
    strategy.entry("Long", strategy.long)
    strategy.exit("TP1", "Long", qty_percent=scale_pct, stop=sl_p, limit=tp1_p)
    strategy.exit("TP2", "Long", stop=sl_p, limit=tp2_p)

if short_cond and strategy.position_size == 0
    entry_p := close
    sl_distance = atr_val * atr_mult
    sl_p := entry_p + sl_distance
    tp1_p := entry_p - sl_distance * tp1_ratio
    tp2_p := entry_p - sl_distance * tp2_ratio
    tp1_hit := false
    strategy.entry("Short", strategy.short)
    strategy.exit("TP1", "Short", qty_percent=scale_pct, stop=sl_p, limit=tp1_p)
    strategy.exit("TP2", "Short", stop=sl_p, limit=tp2_p)

if strategy.position_size > 0
    if not tp1_hit and high >= tp1_p
        tp1_hit := true
        if use_trail
            sl_p := entry_p + be_offset
    if tp1_hit and use_trail
        new_trail = high - trail_dist
        if new_trail > sl_p
            sl_p := new_trail
    strategy.exit("TP1", "Long", qty_percent=scale_pct, stop=sl_p, limit=tp1_p)
    strategy.exit("TP2", "Long", stop=sl_p, limit=tp2_p)

if strategy.position_size < 0
    if not tp1_hit and low <= tp1_p
        tp1_hit := true
        if use_trail
            sl_p := entry_p - be_offset
    if tp1_hit and use_trail
        new_trail = low + trail_dist
        if new_trail < sl_p
            sl_p := new_trail
    strategy.exit("TP1", "Short", qty_percent=scale_pct, stop=sl_p, limit=tp1_p)
    strategy.exit("TP2", "Short", stop=sl_p, limit=tp2_p)

plot(fast_ema, "20 EMA", color=color.blue, linewidth=2)
plot(slow_ema, "50 EMA", color=color.orange, linewidth=2)
plot(trend_ema, "200 EMA", color=color.white, linewidth=2)
```

Notes for implementer:

- Prefer `qty_percent=scale_pct` on TP1 exit so sizing matches percent-of-equity without manual qty math; if TradingView rejects combining `default_qty_type` with `qty_percent`, fall back to explicit `qty=` like the Merged sweep script.
- Keep boxes optional/minimal; visuals are not required for parity.
- No opposite-crossover `strategy.close` / reverse entries.

- [ ] **Step 2: Sanity-check Pine against Python input names**

Diff input identifiers vs `EMA_INPUT_ORDER`; they must match for the `ema_inputs.txt` paste workflow.

- [ ] **Step 3: Commit (only if user asked)**

```bash
git add 20-50ema-strategy
git commit -m "$(cat <<'EOF'
feat: Pine EMA strategy with pro management

EOF
)"
```

---

### Task 5: Optimize and report 3y results

**Files:**
- Write: `xauusd-backtest/results_ema/*` (generated)
- Optional note in chat only (no new markdown doc unless user asks)

- [ ] **Step 1: Run the grid on the 3y window**

```bash
cd xauusd-backtest && .venv/bin/python -m src.backtest_ema --start 2023-10-01 --end 2026-10-01 --optimize
```

Expected: `results_ema/optimization.csv`, winner metrics printed, `metrics.json` / `trades.csv` / `equity.csv` / `equity_drawdown.png` / `ema_inputs.txt` written for the winner.

- [ ] **Step 2: Compare to baseline**

Record in the final response:

| | Baseline (pre-change) | Winner |
|---|---|---|
| Net profit | from prior run / re-run old tag if needed | |
| Expectancy | | |
| Profit factor | | |
| Win rate | | |
| Trades | | |

Success check vs spec: winner net profit & expectancy > baseline; PF ≥ 1.1 preferred; trades ≥ 80.

If the grid fails success criteria, do **one** bounded follow-up only: widen `atr_mult` to include `1.0` and `2.5`, re-run grid once, report both. Do not add RSI/volume filters (out of scope).

- [ ] **Step 3: Commit results only if user asked**

Prefer leaving `results_ema/` uncommitted unless requested.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Crossover + 200 EMA filter | Task 2, 4 |
| Session `0300-1600` NY | Task 1–2, 4 |
| Flat-only entries / no reverse | Task 2, 4 |
| ATR stop, TP1 scale, BE+offset, trail, TP2 | Task 2, 4 |
| Friday flatten | Task 2, 4 |
| Percent equity 10% / $10k | Task 1, 4 |
| Small expectancy grid | Task 3, 5 |
| Pine + Python parity | Task 1, 4 |
| Unit tests TP1/BE/trend | Task 2 |
| 3y backtest outputs | Task 5 |
| Out of scope: RSI/vol/HTF/walk-forward | Not scheduled |

## Placeholder / consistency review

- Input names aligned: `trend_len`, `tp1_ratio`, `tp2_ratio`, `scale_pct`, `use_trail`, `trail_atr_mult`, `be_offset`, `use_session`, `trading_hours`, `session_tz`, `close_friday`.
- BE uses `entry_ref ± be_offset` (signal close), matching level-lock convention.
- `plan_exits` reason strings: `tp1`, `tp2`, `stop` (plus `friday` for flatten).

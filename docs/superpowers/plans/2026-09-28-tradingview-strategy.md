# Merged 1H Sweep Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, production-ready TradingView Pine Script v6 strategy script implementing the 1H Sweep + Pro Management strategy with optimized default parameters and optional toggleable Volume Profile.

**Architecture:** Pure Pine Script v6 strategy with modular sections: Inputs, Calculations, Circuit Breaker, Session/HTF Sweep, MSS, FVG Entry, Order Execution with two-stage TP/Trailing, optional Intrabar Volume Profile, and Visual Dashboard table.

**Tech Stack:** Pine Script v6 (`//@version=6`), TradingView Strategy Engine.

## Global Constraints
- Must use `//@version=6`.
- Must match the backtested parameter set by default: `atr_sl_mult = 2.5`, `tp1_ratio = 1.5`, `tp2_ratio = 2.5`, `vol_length = 30`, `fvg_timeout = 10`.
- Must support toggleable Volume Profile (`enable_vp = false` default) to prevent lag or historical data limits on TradingView charts.
- Word count on commits must be <= 30 words.

---

### Task 1: Create the TradingView Pine Script v6 Strategy File

**Files:**
- Create: `tradingview/xauusd_1h_sweep_strategy.pine`

- [ ] **Step 1: Write the Pine Script v6 file with complete logic and inputs**
- [ ] **Step 2: Verify syntax conformity with Pine Script v6 rules**
- [ ] **Step 3: Commit the new strategy script**

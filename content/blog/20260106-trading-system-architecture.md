+++
date = '2026-01-06'
title = 'Building a Robust Trading System: From Signal to Execution'
+++

*A comprehensive guide to designing trading infrastructure that scales from weekly rebalancing to high-frequency market making.*

---

## Introduction

After years of working with trading systems across different firms and strategies, I've come to appreciate that the difference between a fragile system and a robust one isn't the sophistication of the alpha—it's the clarity of the architecture.

This post distills the key architectural patterns I've learned into a coherent framework. Whether you're building a simple portfolio rebalancer or a multi-strategy hedge fund platform, these principles apply.

---

## Table of Contents

1. [The Four Levels of Abstraction](#the-four-levels-of-abstraction)
2. [Level 1: Risk Management (Not Allocation!)](#level-1-risk-management-not-allocation)
3. [Level 2: Signal Generation](#level-2-signal-generation)
4. [Level 3: Order Management](#level-3-order-management)
5. [Level 4: Market Access](#level-4-market-access)
6. [The OMS and PMS: Bookkeeping Layers](#the-oms-and-pms-bookkeeping-layers)
7. [Rule-Based vs Statistical Approaches](#rule-based-vs-statistical-approaches)
8. [Backtesting: Matching Fidelity to Purpose](#backtesting-matching-fidelity-to-purpose)
9. [Continuous Signals and Position Sizing](#continuous-signals-and-position-sizing)
10. [Adaptive Risk Systems](#adaptive-risk-systems)
11. [Putting It All Together](#putting-it-all-together)
12. [The Human Analogy: Roles on a Trading Desk](#the-human-analogy-roles-on-a-trading-desk)

---

## The Four Levels of Abstraction

Every trading system, regardless of complexity, can be understood through four levels of abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1: RISK MANAGEMENT                                       │
│  "What can't I do?" — Hard limits, constraints, don't blow up   │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 2: SIGNAL GENERATION                                     │
│  "What should I do?" — Alpha logic, buy/sell decisions          │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 3: ORDER MANAGEMENT                                      │
│  "How should I do it?" — TWAP, VWAP, execution algorithms       │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 4: MARKET ACCESS                                         │
│  "Send the order" — Exchange APIs, connectivity                 │
└─────────────────────────────────────────────────────────────────┘
```

**The key insight:** Each layer has a single responsibility and communicates through well-defined interfaces. This separation allows you to:

- Swap execution algorithms without touching alpha logic
- Add new alpha strategies without changing risk infrastructure
- Test signals without real money
- Scale each layer independently

---

## Level 1: Risk Management (Not Allocation!)

Here's a common mistake I see: conflating **portfolio allocation** with **risk management**.

### The Wrong Way

```python
# L1 as "Portfolio Allocation" — PROBLEMATIC
class Level1:
    target_weights = {"QQQ": 0.70, "XAU": 0.30}  # This is an alpha view!
```

This creates problems:
- The "30% QQQ target" is actually a bet that QQQ exposure is desirable
- L1 ends up blocking L2 signals that disagree
- You can't tell if returns came from allocation alpha or trading alpha

### The Right Way

```python
# L1 as "Pure Risk Management" — CLEAN
class Level1:
    """
    L1 = ONLY risk constraints. No alpha views.
    Just: "Don't do anything that could blow up the portfolio."
    """

    # Position limits
    max_single_position = 0.40      # No single position > 40%
    max_gross_exposure = 1.50       # Total |long| + |short| < 150%
    max_net_exposure = 1.00         # Net exposure < 100%

    # Drawdown limits
    max_drawdown = -0.15            # Stop if down 15%

    # Leverage limits
    max_leverage = 2.0

    # Liquidity
    min_cash_buffer = 0.05          # Always keep 5% cash

    def check(self, signal, portfolio):
        """
        Pure risk check. No opinion on whether trade is "good".
        Just: "Is this safe?"
        """
        projected = portfolio.project_state(signal)

        if projected.single_position > self.max_single_position:
            return Reject("position_limit")
        if projected.gross_exposure > self.max_gross_exposure:
            return Reject("gross_exposure")
        if portfolio.drawdown < self.max_drawdown:
            return Reject("drawdown_limit")

        return Approve()
```

**The principle:** L1 should have NO OPINION on whether a trade is profitable. It only ensures the trade is safe.

Portfolio allocation (like "30% QQQ") should be a Level 2 alpha signal that competes with other alpha signals on equal footing.

---

## Level 2: Signal Generation

Level 2 is where all alpha views live—including strategic allocation.

### Multiple Time Horizons, One Framework

```python
class MultiAlphaL2:
    """
    All alpha sources compete on equal footing.
    Different time horizons, same interface.
    """

    alphas = {
        # Slow: Strategic allocation (monthly/quarterly)
        "strategic": StrategicAllocationAlpha(
            target_weights={"QQQ": 0.30, "XAU": 0.20, "CASH": 0.50},
            rebalance_freq="monthly",
            conviction=0.8
        ),

        # Medium: Tactical allocation (weekly)
        "tactical": TacticalRegimeAlpha(
            signals=["momentum", "value", "volatility"],
            rebalance_freq="weekly",
            conviction=0.5
        ),

        # Fast: Trading alpha (daily)
        "sma_cross": SMACrossoverAlpha(
            fast_period=10,
            slow_period=50,
            conviction=0.3
        ),
    }

    def generate_signals(self, market_data):
        """Aggregate all alpha signals by conviction weight"""

        all_signals = defaultdict(list)

        for name, alpha in self.alphas.items():
            for signal in alpha.generate(market_data):
                all_signals[signal.instrument].append({
                    "source": name,
                    "direction": signal.direction,
                    "conviction": signal.conviction * alpha.conviction
                })

        # Combine signals per instrument
        return self._aggregate(all_signals)
```

### The Signal Interface

A signal should express **intent**, not execution details:

```python
@dataclass
class Signal:
    instrument: str          # What to trade
    direction: str           # "BUY" or "SELL"
    size: float              # How much (in units or notional)

    # Execution hints (not full specs)
    urgency: str             # "immediate" | "high" | "medium" | "low"
    horizon: str             # How long can execution take?

    # Metadata
    source: str              # Which alpha generated this
    conviction: float        # How confident (0-1)
    expected_return: float   # For logging/analysis
```

**The key:** Signals say WHAT to do and roughly HOW URGENTLY. They don't specify TWAP vs VWAP—that's Level 3's job.

---

## Level 3: Order Management

Level 3 translates intent into execution. The signal's `urgency` naturally maps to execution strategy:

```python
class ExecutionRouter:
    """
    Maps signal urgency to execution algorithm
    """

    URGENCY_TO_ALGO = {
        "immediate": ("MARKET", {"timeout": "1s"}),
        "high":      ("TWAP", {"duration": "2min"}),
        "medium":    ("TWAP", {"duration": "10min"}),
        "low":       ("PASSIVE", {"duration": "1hour"})
    }

    def execute(self, signal):
        algo, params = self.URGENCY_TO_ALGO[signal.urgency]

        # Adjust for size (larger orders need more time)
        if signal.size > self.large_order_threshold:
            params["duration"] *= 2

        return self.algo_router.run(algo, signal, params)
```

### Order Types Spectrum

From simple to complex:

| Type | Use Case | Example |
|------|----------|---------|
| **Market** | Immediate fill, any price | Emergency exit |
| **Limit** | Specific price, may not fill | Patient entry |
| **TWAP** | Time-slice large order | Buy 100 BTC over 1 hour |
| **VWAP** | Volume-weighted execution | Match market volume profile |
| **Iceberg** | Hide order size | Large institutional order |
| **Adaptive** | Adjust based on market conditions | Smart order routing |

---

## Level 4: Market Access

The final layer translates internal order representation to exchange-specific API calls:

```python
class MarketAccess:
    """
    Exchange connectivity layer
    """

    def submit_order(self, order):
        exchange = self.get_exchange(order.instrument)

        # Translate to exchange format
        exchange_order = {
            "symbol": self.translate_symbol(order.instrument),
            "side": order.direction.upper(),
            "type": self.translate_order_type(order.type),
            "quantity": str(order.size),
            "price": str(order.price) if order.price else None,
        }

        response = exchange.api.place_order(**exchange_order)
        return self.parse_response(response)
```

**MA should be thin:** It handles connectivity, rate limits, and API translation—nothing more.

---

## The OMS and PMS: Bookkeeping Layers

Two critical systems that don't fit neatly into the four levels: Order Management System (OMS) and Portfolio Management System (PMS).

### OMS: The Transaction Ledger

OMS tracks every order from birth to death:

```
┌─────────────────────────────────────────────────────────────────┐
│  ORDER LIFECYCLE IN OMS                                         │
│                                                                 │
│  NEW → PENDING → PARTIAL_FILL → FILLED                         │
│           │                        │                            │
│           └──→ CANCELLED ←─────────┘                            │
│           └──→ REJECTED                                         │
│           └──→ EXPIRED                                          │
└─────────────────────────────────────────────────────────────────┘

OMS Tracks:
- Order ID, status, timestamps
- Original vs filled quantity
- Average fill price
- Commission, fees
- Parent-child relationships (for algo orders)
```

**Virtual balances:** OMS maintains its own view of balances, updated immediately when orders are placed/filled. This is faster than querying the exchange.

### PMS: The Control Room

PMS monitors portfolio state and risk:

```python
class PMS:
    """
    Portfolio Management System

    NOT in the order path! Runs in parallel.
    Observes, calculates, signals.
    """

    def run_monitoring_loop(self):
        while True:
            # 1. Observe: Get latest positions from OMS
            positions = self.sync_from_oms()

            # 2. Calculate: Risk metrics
            pnl = self.calculate_pnl(positions)
            exposure = self.calculate_exposure(positions)
            drawdown = self.calculate_drawdown(positions)

            # 3. Signal: Set flags if limits breached
            if pnl < self.pnl_limit:
                self.set_flag("pnl_breach", True)
            if exposure > self.exposure_limit:
                self.set_flag("exposure_breach", True)

            time.sleep(self.check_interval)
```

**Critical insight:** PMS does NOT reject orders directly. It sets flags that execution loops check:

```
┌─────────────────────────────────────────────────────────────────┐
│  PMS IS NOT A GATE — IT'S A CONTROL ROOM                        │
│                                                                 │
│  Orders flow:  Strategy → OMS → Exchange                        │
│                              ↑                                  │
│                              │ Check flags                      │
│  PMS monitors: ────────→ Redis Flags ←──── Set flags           │
│                                                                 │
│  Execution loops check flags BEFORE placing orders.             │
│  PMS doesn't block; it turns on warning lights.                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Rule-Based vs Statistical Approaches

Two paradigms for signal generation:

### Rule-Based (Traditional Quant)

```python
class RuleBasedSignal:
    """
    IF indicator crosses threshold THEN trade
    """

    def generate(self, prices):
        sma_fast = prices[-10:].mean()
        sma_slow = prices[-50:].mean()

        if sma_fast > sma_slow:
            return Signal(direction="BUY")  # Discrete output
        return None
```

**Characteristics:**
- Discrete output: BUY/SELL/HOLD
- No probability attached
- Hand-crafted rules
- Easy to understand and debug

### Statistical Expected Value

```python
class StatisticalEVSignal:
    """
    Predict E[return] for each decision, trade if EV > threshold
    """

    def generate(self, features):
        # ML model predicts expected return
        ev_buy = self.model.predict(features)["ev_buy"]
        ev_sell = self.model.predict(features)["ev_sell"]

        # The "rule" is just: EV > threshold
        net_ev = ev_buy - self.fee_rate

        if net_ev > self.min_edge:
            return Signal(
                direction="BUY",
                expected_return=net_ev,
                confidence=self.model.confidence
            )
        return None
```

**Characteristics:**
- Continuous output: expected return
- Probability/confidence attached
- ML model (trainable)
- Harder to interpret

### They Can Coexist

```python
# Wrap statistical model in rule interface
class StatModelWrapper:
    def generate(self, market_data):
        ev = self.model.predict(market_data)

        # Statistical output → Rule-based interface
        if ev - self.fee > self.threshold:
            return Signal(direction="BUY")
        return None
```

**The framework doesn't care what's inside the signal generator.** It just needs a Signal out.

---

## Backtesting: Matching Fidelity to Purpose

Different questions require different backtest fidelity:

### Level 1: Signal-Only (Fast Iteration)

```python
def backtest_l1(signals, prices):
    """
    Assumption: Fill at signal price + flat haircut
    """
    pnl = 0
    slippage = 0.001  # 10 bps

    for signal in signals:
        entry = prices[signal.entry_time] * (1 + slippage)
        exit = prices[signal.exit_time] * (1 - slippage)
        pnl += (exit - entry) * signal.size

    return pnl
```

**Use for:** Idea screening, early research
**Ignores:** Order book, partial fills, market impact

### Level 2: Execution-Aware (Development)

```python
def backtest_l2(signals, prices, volumes):
    """
    Model execution costs based on size and urgency
    """
    pnl = 0

    for signal in signals:
        # Size-dependent slippage (square root model)
        size_fraction = signal.size / volumes[signal.time]
        impact = 0.1 * sqrt(size_fraction)

        # Urgency multiplier
        urgency_cost = {"immediate": 3, "high": 2, "medium": 1, "low": 0.5}

        total_cost = (0.0001 + impact) * urgency_cost[signal.urgency]

        entry = prices[signal.entry_time] * (1 + total_cost)
        # ...
```

**Use for:** Strategy development, parameter tuning
**Models:** Size impact, urgency costs
**Ignores:** Order book dynamics, queue position

### Level 3: Full Simulation (Pre-Production)

```python
def backtest_l3(signals, order_book_history, trades_history):
    """
    Simulate actual order book interaction
    """
    for signal in signals:
        fills = simulate_twap(
            signal,
            order_book_history,
            trades_history,
            duration="10min",
            slices=20
        )
        # Models: Queue position, partial fills, actual algo behavior
```

**Use for:** Pre-production validation, HFT
**Models:** Everything
**Cost:** Slow, requires tick data

### When to Use What

| Strategy Type | Hold Time | Recommended Backtest |
|--------------|-----------|---------------------|
| Weekly rebalance | Weeks | Level 1 |
| Swing trading | Days | Level 1-2 |
| Intraday momentum | Hours | Level 2 |
| Stat arb | Minutes | Level 2-3 |
| Market making | Seconds | Level 3 |

---

## Continuous Signals and Position Sizing

A common question: When SMA says "bullish," do I keep buying every day?

**No.** A continuous signal represents a TARGET POSITION, not repeated trades.

### Event vs State

```python
# EVENT-BASED: Trade on state changes
def generate_signal_event(self, prices):
    current_state = "BULLISH" if sma_fast > sma_slow else "BEARISH"

    if current_state != self.previous_state:
        self.previous_state = current_state
        return Signal(direction="BUY" if current_state == "BULLISH" else "SELL")

    return None  # No signal while state unchanged

# STATE-BASED: Signal = target position
def generate_signal_state(self, prices, current_position):
    if sma_fast > sma_slow:
        target = 0.30  # Want to be 30% long while bullish
    else:
        target = 0.00  # Want to be flat while bearish

    delta = target - current_position

    if abs(delta) > 0.01:
        return Signal(direction="BUY" if delta > 0 else "SELL", size=abs(delta))

    return None
```

### Sizing Methods

**1. Fixed Size**
```python
target = 0.30  # Always 30% when bullish
```

**2. Conviction-Based**
```python
spread = (sma_fast - sma_slow) / sma_slow
conviction = min(abs(spread) / 0.05, 1.0)
target = max_position * conviction
```

**3. Risk-Budget-Based**
```python
risk_budget = 0.05  # 5% vol contribution
asset_vol = get_volatility(instrument)
target = risk_budget / asset_vol  # Inverse vol scaling
```

---

## Adaptive Risk Systems

Different strategies need different risk check intensities. Match the risk infrastructure to the strategy speed.

### The Spectrum

| Strategy | Position Δ/hr | Risk Check | Implementation |
|----------|--------------|------------|----------------|
| HFT/MM | 1000s | Every tick | Sync, pre-order |
| Stat arb | 100s | Every 100ms | Async, fast cache |
| Intraday | 10s | Every 10s | Flag-based |
| Swing | 1-2 | Every minute | Periodic batch |
| Long-term | <1 | Every 5min | Dashboard |

### Pluggable Risk Tiers

```python
class AdaptiveRiskRouter:
    """
    Route signals to appropriate risk checker based on urgency
    """

    def __init__(self):
        self.checkers = {
            "realtime": RealtimeRiskChecker(),  # HFT-grade
            "fast": FastRiskChecker(),          # 100ms cache
            "standard": StandardRiskChecker(),  # Flag-based
            "relaxed": RelaxedRiskChecker(),    # Batch
        }

        self.urgency_to_tier = {
            "immediate": "realtime",
            "high": "fast",
            "medium": "standard",
            "low": "relaxed"
        }

    def check(self, signal):
        tier = self.urgency_to_tier[signal.urgency]
        return self.checkers[tier].check(signal)
```

### Start Simple, Scale Up

```python
class RealtimeRiskChecker:
    """
    STUB: Reserved for future HFT implementation
    Falls back to fast tier until needed
    """

    def __init__(self):
        self.enabled = config.get("realtime_enabled", False)
        if not self.enabled:
            self.fallback = FastRiskChecker()

    def check(self, signal):
        if not self.enabled:
            return self.fallback.check(signal)
        # Future: implement microsecond checks
```

This lets you build infrastructure incrementally—start with flag-based checks, add HFT-grade when needed.

---

## Putting It All Together

Here's how all the pieces fit:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE TRADING SYSTEM                              │
│                                                                              │
│   MARKET DATA                                                                │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LEVEL 2: SIGNAL GENERATION (All Alpha)                             │   │
│   │                                                                      │   │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │   │
│   │  │  Strategic   │ │   Tactical   │ │   Trading    │                │   │
│   │  │  Allocation  │ │    Alpha     │ │    Alpha     │                │   │
│   │  │  (monthly)   │ │   (weekly)   │ │   (daily)    │                │   │
│   │  └──────────────┘ └──────────────┘ └──────────────┘                │   │
│   │           │               │               │                         │   │
│   │           └───────────────┼───────────────┘                         │   │
│   │                           │                                          │   │
│   │                    SIGNAL AGGREGATOR                                 │   │
│   │                  (conviction-weighted)                               │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LEVEL 1: RISK CHECK (Pure Safety)                                  │   │
│   │                                                                      │   │
│   │  Position limits? ✓   Leverage? ✓   Drawdown? ✓   Cash buffer? ✓   │   │
│   │                                                                      │   │
│   │  [Adaptive: tier selected by signal.urgency]                        │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LEVEL 3: ORDER MANAGEMENT                                          │   │
│   │                                                                      │   │
│   │  Urgency → Algo Selection → Order Slicing → Child Orders            │   │
│   │                                                                      │   │
│   │  OMS tracks: status, fills, virtual balances                        │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LEVEL 4: MARKET ACCESS                                             │   │
│   │                                                                      │   │
│   │  Internal format → Exchange API → Submit                            │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│                          EXCHANGES                                           │
│                               │                                              │
│                               │ Fills                                        │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PMS (Parallel Monitoring)                                          │   │
│   │                                                                      │   │
│   │  Observe positions → Calculate risk → Set flags if breach           │   │
│   │                                                                      │   │
│   │  [Feeds back to L1 risk checks]                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Separate concerns clearly.** L1 = safety, L2 = alpha, L3 = execution, L4 = connectivity.

2. **Portfolio allocation is alpha.** Put it in L2, not L1. L1 should only contain hard risk limits.

3. **Signals express intent, not execution.** Use urgency hints; let L3 choose the algo.

4. **Match infrastructure to strategy speed.** Don't build HFT plumbing for a weekly rebalancer.

5. **PMS monitors, OMS tracks.** Neither should be in the critical order path.

6. **Continuous signals = target positions.** Trade the delta, not a new order every tick.

7. **Backtest at the right fidelity.** Level 1 for screening, Level 2 for development, Level 3 for validation.

8. **Build incrementally.** Stub the HFT risk tier; implement when needed.

---

## The Human Analogy: Roles on a Trading Desk

The architecture directly mirrors how trading desks organize human roles:

```
┌───────────────────────┬────────────────────────┬─────────────────────────┐
│  SYSTEM LAYER         │  HUMAN ROLE            │  RESPONSIBILITY         │
├───────────────────────┼────────────────────────┼─────────────────────────┤
│  L1: Risk Management  │  Risk Manager          │  "You can't do that"    │
│                       │                        │  Sets limits, no alpha  │
│                       │                        │  views, just safety     │
├───────────────────────┼────────────────────────┼─────────────────────────┤
│  L2: Signal Generation│  Researcher / Analyst  │  "We should do this"    │
│                       │  Portfolio Manager     │  Investment thesis,     │
│                       │                        │  alpha ideas, sizing    │
├───────────────────────┼────────────────────────┼─────────────────────────┤
│  L3: Order Management │  Trader                │  "I'll work the order"  │
│                       │  Execution Desk        │  HOW to execute, algo   │
│                       │                        │  selection, timing      │
├───────────────────────┼────────────────────────┼─────────────────────────┤
│  L4: Market Access    │  Prime Broker / DMA    │  Connectivity to        │
│                       │  Infrastructure        │  exchanges              │
├───────────────────────┼────────────────────────┼─────────────────────────┤
│  OMS / PMS            │  Middle & Back Office  │  Bookkeeping, P&L,      │
│                       │  Operations            │  reconciliation         │
└───────────────────────┴────────────────────────┴─────────────────────────┘
```

### Why the Parallel Exists

The human org structure evolved to solve the same problem as good software architecture: **separation of concerns and clear accountability**.

| Principle | Human Organization | Software Architecture |
|-----------|-------------------|----------------------|
| **No conflicts of interest** | Risk manager doesn't get paid on P&L | L1 has no alpha views |
| **Specialized expertise** | Traders know market microstructure | L3 handles execution algos |
| **Clear handoffs** | PM says "buy 10k shares", trader decides how | Signal → Order separation |
| **Audit trail** | Operations tracks everything | OMS/PMS logging |

### Real Workflow Example

**Human version:**
1. **Analyst** researches and says: "AAPL is undervalued, we should be 5% long"
2. **PM** agrees, sizes it: "Let's do 3% given current exposure"
3. **Risk Manager** checks: "That's within limits, approved"
4. **Trader** executes: "I'll TWAP it over 2 hours to minimize impact"
5. **Operations** reconciles: "Filled 100k shares @ $182.34 avg"

**System version:**
1. **L2 (Alpha)** generates signal: `{instrument: AAPL, direction: BUY, target: 0.03}`
2. **L1 (Risk)** checks: `projected_position < max_single_position ✓`
3. **L3 (Execution)** routes: `TWAP(duration=2h, slices=24)`
4. **L4 (MA)** sends orders to exchange
5. **OMS/PMS** tracks fills and updates P&L

### Why This Matters

When roles blur in human organizations—PM overrides risk, trader picks stocks—things go wrong. The same applies to software: when L1 has alpha views or L2 specifies execution details, the system becomes fragile.

The cleanest trading operations have the same separation in both people AND code. This architecture isn't just software engineering—it's encoding decades of institutional wisdom about how to run a trading operation without blowing up.

---

## Final Thoughts

The best trading systems I've seen share a common trait: they're boring. The architecture is clean, the responsibilities are clear, and there are no clever hacks.

When you're tempted to add complexity, ask: "Does this belong in L1, L2, L3, or L4?" If the answer isn't immediately obvious, you're probably mixing concerns.

Build the simple version first. Make it work. Then scale.

---

*If you found this useful, feel free to share. Questions and feedback welcome.*

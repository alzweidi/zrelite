# ZAWAD RISK ENGINE (ZRE) v3.9

Multi-Level Take Profit Position Sizing System with Enhanced Kelly Criterion

---

## Overview

ZRE is a mathematically rigorous position sizing engine for leveraged cryptocurrency trading. It applies an Enhanced Kelly Criterion with multi-level take profits, distance-based scenario modelling, and institutional-grade risk constraints. The system is designed for capital preservation, accurate sizing, and probabilistic consistency.

---

## Mathematical Framework

### 1. Enhanced Kelly Criterion

The raw Kelly fraction is calculated as:

$$
k = \frac{p \cdot (R_{w} + 1) - 1}{R_{w}}
$$

Where:

* $k$ = Optimal capital fraction to risk
* $p$ = Win rate (0.01 to 0.99)
* $R_{w}$ = Weighted reward-to-risk ratio:

$$
R_{w} = \sum_{i=1}^{n} a_i \cdot R_i
$$

* $a_i$ = Allocation fraction to TP level $i$
* $R_i = \frac{|TP_i - Entry|}{|Entry - SL|}$ = Individual reward-to-risk ratio

### 2. Safety Constraints (Sequential)

1. **Quarter-Kelly**:

$$
k_{1} = k \cdot 0.25
$$

2. **Multi-TP Adjustment**:

$$
k_{2} = k_{1} \cdot 0.90
$$

3. **Confidence Scaling**:

$$
k_{3} = k_{2} \cdot \frac{Confidence}{100}
$$

4. **Kelly Cap**:

$$
k_{4} = \min(k_{3}, 0.50)
$$

5. **Hard Risk Cap**:

$$
k_{final} = \min(k_{4}, 0.05)
$$

### 3. Leverage Adjustment

Volatility scales with the square root of leverage:

$$
k_{adj} = \frac{k_{final}}{\sqrt{Leverage}}
$$

### 4. Scenario Probability Model

Probability for each TP scenario is based on exponential distance and sequential penalties:

$$
P_{i} = p \cdot 0.5^{d_i / 0.1} \cdot 0.5^{i - 1}
$$

Where:

* $d_i$ = Relative price distance to final TP in scenario $i$
* Probabilities are normalized to sum to 1.

Scenarios are named like TP1\_TP2\_TP3, representing sequential TP hits.

---

## Features

* Multi-level take profit (1–4 levels)
* Allocation-weighted R–R ratio
* Scenario matrix with exponential decay
* CLI with validation and reporting
* High-precision (28-digit) decimal arithmetic

---

## Risk Philosophy

* Capital at risk is capped at 5%
* Position sizing never exceeds 50% Kelly
* Sequential TP progression prevents unrealistic outcomes
* Conservative constraints mitigate volatility and sizing errors

---

## Input Parameters

| Parameter     | Type    | Description                            |
| ------------- | ------- | -------------------------------------- |
| capital       | Decimal | Trading capital in USD                 |
| entry         | Decimal | Entry price                            |
| stop\_loss    | Decimal | Stop loss price                        |
| take\_profits | List    | Up to 4 TP levels with allocations     |
| direction     | str     | 'LONG' or 'SHORT'                      |
| win\_rate     | Decimal | Historical win probability (0.01–0.99) |
| confidence    | Decimal | Trader confidence (1–100)              |
| leverage      | Decimal | Position leverage (1–100)              |

---

## Outputs

### PositionSizingResult:

* Raw and adjusted Kelly fractions
* Final capital at risk
* Base and leveraged position sizes

### RiskMetrics:

* Expected value (EV)
* Risk-adjusted return (Sharpe-style)
* Maximum drawdown impact
* Profit probability and breakeven rates

---

## CLI Usage

Run:

```bash
python3 zre.py
```

Follow prompts for input. You’ll be prompted to select an allocation preset (1–4) after entering TP levels. The system returns detailed sizing and risk analysis.

---

## API Usage

```python
from zre import ZawadRiskEngine
from zre import EnhancedTradeSetup, TakeProfitLevel
from decimal import Decimal

engine = ZawadRiskEngine()

setup = EnhancedTradeSetup(
    capital=Decimal('20000'),
    entry=Decimal('100'),
    stop_loss=Decimal('90'),
    take_profits=[
        TakeProfitLevel(price=Decimal('110'), allocation_percentage=Decimal('50')),
        TakeProfitLevel(price=Decimal('120'), allocation_percentage=Decimal('30')),
        TakeProfitLevel(price=Decimal('130'), allocation_percentage=Decimal('20'))
    ],
    direction='LONG',
    win_rate=Decimal('0.65'),
    confidence=Decimal('80'),
    leverage=Decimal('5')
)

if engine.validate_enhanced_inputs(setup).is_valid:
    result = engine.calculate_enhanced_kelly(setup)
    metrics = engine.calculate_risk_metrics(setup, result)
```

---

## Example

* Capital: 20,000
* Entry: 100
* SL: 90
* TPs: 110 (50%), 120 (30%), 130 (20%)
* Win rate: 0.65
* Confidence: 80
* Leverage: 5x

**Result:**

* Weighted R: 3.4
* Final risk: 2.06%
* Position size: \$2,063 unleveraged, \$10,315 leveraged
* EV: +\$820
* Breakeven win rate: \~22.7%

---

## Installation

* Python 3.7+
* Only uses Python standard library (`decimal`, `math`, `typing`, `sys`)
* Download [`zre.py`](zre.py) and run or import as a module

---

## Author

Abedalaziz Alzweidi
Version: 3.9

---

## Disclaimer

For educational use only. Trading involves substantial risk. Use at your own discretion.

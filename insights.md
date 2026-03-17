# 📊 Trader Behavior Insights — Hyperliquid × Fear & Greed Index

**Dataset:** 211,224 trades | 32 accounts | 246 coins | May 2023 – May 2025  
**Sentiment Source:** Alternative.me Fear & Greed Index (5 classes)

---

## 1. Key Findings

### Finding 1: Extreme Greed Produces the Highest Average PnL
Traders earned an average of **$67.89/trade during Extreme Greed** — nearly double the
**$34.54/trade during Extreme Fear**. This suggests that trending bull markets create
favorable momentum for the trader cohort as a whole.

| Sentiment     | Avg PnL  |
|---------------|----------|
| Extreme Greed | $67.89   |
| Fear          | $54.29   |
| Greed         | $43.58   |
| Extreme Fear  | $34.54   |
| Neutral       | $34.31   |

> **Counterintuitive result:** Fear periods ($54.29) outperformed Greed periods ($43.58).
> This may reflect contrarian traders who exploit market dislocations during fear-driven selloffs.

### Finding 2: Overall Win Rate is Below 50%
The cohort achieves only a **41.1% win rate** across all trades. Despite this, the positive
average PnL ($48.75) indicates that winning trades are significantly larger in magnitude
than losing ones — a classic asymmetric payoff structure.

### Finding 3: Sentiment Alone Has Weak Predictive Power
A 1-day lagged correlation between the Fear/Greed score and the following day's aggregate
PnL yields **r = -0.107**. This weak negative correlation suggests:
- Sentiment is not a reliable standalone signal for daily performance
- Other factors (coin selection, position sizing, entry timing) dominate outcomes
- The slight negative correlation hints at mild contrarian opportunity — high greed scores
  marginally precede lower-PnL days

---

## 2. Behavioral Patterns

### Pattern 1: Trade Volume Concentrates in Fear Phases
Fear and Extreme Fear periods account for a disproportionate share of total trade volume.
This indicates traders are most active when markets are stressed — likely reacting to price
movements rather than acting on pre-planned strategies. Reactive trading under fear = higher
frequency, lower quality decisions.

### Pattern 2: Top Traders Are Highly Concentrated
The top 10 traders by total PnL represent an outsized share of aggregate profits. The best
performer achieved significantly higher win rates and total PnL than the median trader,
suggesting a small group of disciplined participants drives the majority of positive returns.

### Pattern 3: Momentum Chasers Dominate (23 of 32 Traders)
K-Means clustering across win rate, avg PnL, avg trade size, and liquidation frequency
identified three behavioral archetypes:

| Archetype               | Count | Characteristic                              |
|-------------------------|-------|---------------------------------------------|
| Momentum Chaser         | 23    | Average win rate, moderate size, reactive   |
| Disciplined Accumulator | 8     | Higher win rate, consistent positive PnL    |
| Reactive Liquidator     | 1     | Negative PnL, liquidation events present    |

The dominance of Momentum Chasers (72%) suggests most traders in this cohort follow price
rather than lead it — a pattern consistent with retail behavior in crypto markets.

---

## 3. Risk Analysis

### Risk 1: Liquidation Events Are Rare But Concentrated
Only 1 liquidation event was identified in the dataset — a remarkably low number for 32
crypto traders over 2 years. This may reflect the dataset's trader selection (experienced
wallets) rather than the broader Hyperliquid user base.

### Risk 2: High Variance in PnL Distribution
The PnL distribution is heavily right-skewed: median closed PnL is $0 (many small/flat
trades) while the mean is $48.75, driven by a small number of very large winners. The
maximum single trade gain was $135,329 vs a maximum single loss of -$117,990. Traders
need robust risk management to survive the downside tail.

### Risk 3: Neutral Sentiment Is the Worst Environment
Neutral sentiment days produced the lowest average PnL ($34.31) — lower even than Extreme
Fear. This may reflect that directionless markets provide fewer clear momentum opportunities,
forcing traders into low-conviction positions with poor risk/reward.

---

## 4. Recommendations

**For traders:**
1. **Avoid neutral/ranging markets** — the data shows Neutral sentiment correlates with the
   weakest average outcomes. Wait for directional sentiment confirmation.
2. **Don't blindly follow the crowd in Greed** — Fear periods actually outperform Greed
   periods in this cohort. Consider counter-sentiment positions during high greed phases.
3. **Focus on asymmetric setups** — the cohort's 41% win rate with positive expected value
   confirms that *how much* you make on wins matters more than *how often* you win.

**For strategy development:**
1. **Sentiment is a filter, not a signal** — with r = -0.107, sentiment cannot drive
   entry/exit decisions alone. Combine it with on-chain flow, order book imbalance, or
   momentum indicators.
2. **Archetype-aware position sizing** — Disciplined Accumulators outperform by doing less,
   not more. Reducing trade frequency in low-conviction environments is a competitive edge.
3. **Expand the feature set** — trader performance variance is largely unexplained by
   sentiment. Future analysis should incorporate: time-of-day patterns, coin-specific
   sentiment, on-chain wallet age, and cross-exchange flow data.

---

## 5. What This Analysis Cannot Tell Us

- **No intraday sentiment resolution** — the Fear & Greed Index is a single daily value.
  Intraday sentiment shifts (social media, liquidation cascades) are not captured.
- **No order book data** — we cannot distinguish limit vs market orders, measure slippage,
  or identify whether trades were entries or exits of larger positions.
- **Small sample size** — 32 accounts limits statistical power. Findings are directional,
  not definitive.
- **Closed PnL bias** — unrealized losses are excluded. Traders holding underwater positions
  appear better-performing than they actually are.
- **Unknown trader types** — we cannot distinguish algorithmic bots from human traders,
  a distinction that would significantly change behavioral interpretation.

---

*"This analysis aims to bridge market psychology and trader behavior to derive actionable
trading intelligence."*

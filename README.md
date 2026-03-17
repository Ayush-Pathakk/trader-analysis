# 🧠 Hyperliquid Trader Behavior × Market Sentiment

> *"This analysis bridges market psychology and trader behavior to derive actionable trading intelligence."*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightblue)
![scikit-learn](https://img.shields.io/badge/sklearn-KMeans-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Do traders perform better or worse depending on market sentiment? And more importantly —
**do they even know it?**

This project analyzes 211,224 trades from 32 Hyperliquid accounts across 2023–2025,
cross-referenced with the daily Fear & Greed Index, to uncover behavioral patterns, risk
tendencies, and sentiment-driven performance shifts.

---

## 🗂 Repository Structure

```
hyperliquid-trader-analysis/
├── data/
│   ├── fear_greed_index.csv          ← Daily sentiment scores (2018–2025)
│   └── compressed_data_csv.gz        ← 211K Hyperliquid trades
│
├── notebooks/
│   └── analysis.ipynb                ← Full analysis notebook (run top to bottom)
│
├── src/
│   └── utils.py                      ← All functions: load, clean, feature eng, plots
│
├── outputs/
│   └── figures/                      ← 6 charts auto-generated here
│
├── insights.md                       ← Written findings & recommendations
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Ayush-Pathakk/trader-analysis
cd hyperliquid-trader-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (generates all charts + stats)
cd notebooks
jupyter notebook analysis.ipynb
# OR run headlessly:
jupyter nbconvert --to notebook --execute analysis.ipynb
```

---

## 🔍 Approach

### Data
- **211,224 trades** from 32 Hyperliquid accounts (May 2023 – May 2025)
- **Sentiment** from Alternative.me Fear & Greed Index (5 classes, daily granularity)
- Merge strategy: daily date join with forward-fill for unmatched trading days

> *Note: Analysis is at daily granularity due to absence of intraday sentiment resolution.*

### Feature Engineering
Eight behavior-driven features were engineered — focused on decision-making patterns
rather than raw metrics:

| Feature | Description |
|---------|-------------|
| `profit_flag` | Binary: trade closed in profit |
| `trade_size_usd` | Absolute USD exposure per trade |
| `sentiment_binary` | Fear / Neutral / Greed (3-class) |
| `is_liquidation` | Flagged from Direction column |
| `is_open` / `is_close` | Trade direction type |
| Trader profiles | Win rate, total PnL, avg size (per account) |

### Analysis Pipeline
1. **PnL vs Sentiment** — average closed PnL across all 5 sentiment classes
2. **Win Rate vs Sentiment** — does sentiment shift trade quality?
3. **Trade Volume & Size** — behavioral activity patterns by sentiment
4. **Top 10 vs Bottom 10 Traders** — what separates winners from losers?
5. **Sentiment Lag Analysis** — does yesterday's sentiment predict today's PnL? *(r = -0.107)*
6. **Trader Archetype Clustering** — K-Means on behavioral features → 3 archetypes

---

## 💡 Key Insights

| # | Finding |
|---|---------|
| 1 | **Extreme Greed → highest avg PnL ($67.89)**, but Fear ($54.29) beats plain Greed ($43.58) |
| 2 | **Overall win rate is 41.1%** — traders profit via asymmetric payoff, not win frequency |
| 3 | **Sentiment has weak predictive power (lag-1 r = -0.107)** — not sufficient as a standalone signal |
| 4 | **72% of traders are Momentum Chasers** — reactive, not strategic |
| 5 | **Neutral sentiment = worst PnL** — directionless markets should be avoided |

---

## 📊 Visualizations

| Chart | What It Shows |
|-------|---------------|
| `01_pnl_by_sentiment` | Avg closed PnL across 5 sentiment classes |
| `02_winrate_by_sentiment` | Win % per class vs 50% baseline |
| `03_volume_by_sentiment` | Trade count + avg trade size per sentiment |
| `04_top_vs_bottom_traders` | Side-by-side comparison: top 10 vs bottom 10 |
| `05_sentiment_lag_correlation` | Lag-1 sentiment score vs next-day PnL scatter |
| `06_trader_archetypes` | Cluster scatter: win rate vs avg PnL, sized by trade count |

---

## ⚡ Business Impact

This framework can directly inform:
- **Sentiment-gated trading strategies** — avoid neutral markets, tighten risk in extreme greed
- **Trader segmentation** — identify Disciplined Accumulators for signal-following or copy-trading
- **Risk alerts** — flag Momentum Chasers increasing size in high-greed environments

---

## 🔬 Limitations & Future Work

- Intraday sentiment (social, on-chain) not included — would sharpen timing analysis
- 32-account sample limits statistical power; findings are directional
- Future: add order book data, wallet age, cross-exchange flow, time-of-day patterns

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter
```
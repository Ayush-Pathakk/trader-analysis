"""
utils.py — Core utility functions for Hyperliquid Trader Analysis
All data loading, cleaning, feature engineering, and plotting helpers live here.
If something breaks, fix it here — the notebook stays clean.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — change thresholds/paths here only
# ─────────────────────────────────────────────
CONFIG = {
    "trades_path":     "../data/compressed_data.csv",
    "sentiment_path":  "../data/fear_greed_index.csv",
    "figures_path":    "../outputs/figures",
    "high_leverage_threshold": 10,
    "top_n_traders":           10,
    "n_clusters":               3,
    "random_state":            42,
    "sentiment_order": ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
    "sentiment_colors": {
        "Extreme Fear":  "#d62728",
        "Fear":          "#ff7f0e",
        "Neutral":       "#bcbd22",
        "Greed":         "#2ca02c",
        "Extreme Greed": "#1f77b4",
    },
}

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_trades(path: str) -> pd.DataFrame:
    """Load and normalize the Hyperliquid trades CSV / gz file."""
    try:
        df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    except Exception as e:
        raise FileNotFoundError(f"Could not load trades file at '{path}'. Error: {e}")

    # Normalize column names → snake_case
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_", regex=False)
    )

    # Parse date from 'timestamp_ist' (format: DD-MM-YYYY HH:MM)
    df["date"] = pd.to_datetime(
        df["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
    ).dt.date

    # Drop rows where date parse failed
    bad = df["date"].isna().sum()
    if bad > 0:
        print(f"  [load_trades] Warning: {bad} rows had unparseable timestamps — dropped.")
        df = df.dropna(subset=["date"])

    df["date"] = pd.to_datetime(df["date"])

    print(f"  [load_trades] Loaded {len(df):,} trades | {df['account'].nunique()} accounts | "
          f"{df['coin'].nunique()} coins | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_sentiment(path: str) -> pd.DataFrame:
    """Load and normalize the Fear & Greed Index CSV."""
    try:
        fg = pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load sentiment file at '{path}'. Error: {e}")

    fg.columns = fg.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    fg["date"] = pd.to_datetime(fg["date"], errors="coerce")

    bad = fg["date"].isna().sum()
    if bad > 0:
        print(f"  [load_sentiment] Warning: {bad} rows with bad dates dropped.")
        fg = fg.dropna(subset=["date"])

    fg = fg[["date", "value", "classification"]].rename(
        columns={"value": "sentiment_score", "classification": "sentiment"}
    )

    print(f"  [load_sentiment] Loaded {len(fg):,} days | {fg['date'].min().date()} → {fg['date'].max().date()}")
    print(f"  [load_sentiment] Classes: {fg['sentiment'].value_counts().to_dict()}")
    return fg


def merge_datasets(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge trades with daily sentiment on date.
    Analysis is at daily granularity — no intraday sentiment resolution available.
    """
    merged = trades.merge(sentiment, on="date", how="left")

    missing = merged["sentiment"].isna().sum()
    if missing > 0:
        pct = 100 * missing / len(merged)
        print(f"  [merge] {missing:,} trades ({pct:.1f}%) had no sentiment match — forward filled.")
        merged["sentiment"]       = merged["sentiment"].ffill()
        merged["sentiment_score"] = merged["sentiment_score"].ffill()

    # Drop any remaining nulls in sentiment (beginning of dataset)
    merged = merged.dropna(subset=["sentiment"])
    print(f"  [merge] Final merged dataset: {len(merged):,} trades")
    return merged


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavior-driven features to the trades dataframe.
    Focus: decision-making patterns, not just raw metrics.
    """
    df = df.copy()

    # Trade outcome
    df["profit_flag"]     = (df["closed_pnl"] > 0).astype(int)
    df["loss_flag"]       = (df["closed_pnl"] < 0).astype(int)
    df["pnl_abs"]         = df["closed_pnl"].abs()

    # Sentiment binary grouping (for simpler comparisons)
    df["sentiment_binary"] = df["sentiment"].apply(
        lambda x: "Fear" if "Fear" in str(x) else ("Greed" if "Greed" in str(x) else "Neutral")
    )

    # High leverage flag
    # NOTE: Hyperliquid data doesn't have a direct leverage column;
    # we proxy leverage via size_usd / (size_tokens * execution_price) where available,
    # or flag large-size trades as high exposure.
    # Using size_usd as a trade size proxy for risk analysis.
    df["trade_size_usd"] = df["size_usd"].abs()

    # Liquidation flag
    df["is_liquidation"] = df["direction"].str.contains(
        "Liquidat|Auto-Delerag", case=False, na=False
    ).astype(int)

    # Closing trade flag
    df["is_close"] = df["direction"].str.contains(
        "Close|Settlement", case=False, na=False
    ).astype(int)

    # Opening trade flag
    df["is_open"] = df["direction"].str.contains(
        "Open", case=False, na=False
    ).astype(int)

    print(f"  [features] Added 8 features. Shape: {df.shape}")
    return df


def build_trader_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-trader statistics. Returns one row per account.
    This is the foundation for clustering and top/bottom analysis.
    """
    grp = df.groupby("account")

    profiles = pd.DataFrame({
        "total_trades":      grp["closed_pnl"].count(),
        "total_pnl":         grp["closed_pnl"].sum(),
        "avg_pnl":           grp["closed_pnl"].mean(),
        "median_pnl":        grp["closed_pnl"].median(),
        "win_rate":          grp["profit_flag"].mean(),
        "total_wins":        grp["profit_flag"].sum(),
        "total_losses":      grp["loss_flag"].sum(),
        "avg_trade_size":    grp["trade_size_usd"].mean(),
        "max_single_loss":   grp["closed_pnl"].min(),
        "max_single_win":    grp["closed_pnl"].max(),
        "liquidation_count": grp["is_liquidation"].sum(),
        "coins_traded":      grp["coin"].nunique(),
    })

    # Win/loss ratio (avoid div by zero)
    profiles["win_loss_ratio"] = (
        profiles["total_wins"] / profiles["total_losses"].replace(0, np.nan)
    ).fillna(profiles["total_wins"])  # all wins, no losses → just use win count

    # Sentiment breakdown per trader
    for sentiment in ["Fear", "Greed", "Neutral"]:
        sub = df[df["sentiment_binary"] == sentiment].groupby("account")
        profiles[f"pnl_{sentiment.lower()}"]        = sub["closed_pnl"].sum().reindex(profiles.index, fill_value=0)
        profiles[f"trades_{sentiment.lower()}"]     = sub["closed_pnl"].count().reindex(profiles.index, fill_value=0)
        profiles[f"win_rate_{sentiment.lower()}"]   = sub["profit_flag"].mean().reindex(profiles.index, fill_value=0)

    profiles = profiles.reset_index()
    print(f"  [profiles] Built profiles for {len(profiles)} traders.")
    return profiles


# ─────────────────────────────────────────────
# 3. CLUSTERING — TRADER ARCHETYPES
# ─────────────────────────────────────────────

def cluster_traders(profiles: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    K-Means clustering to identify trader archetypes.
    Returns profiles df with 'archetype' column + cluster centers.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ["win_rate", "avg_pnl", "avg_trade_size", "liquidation_count"]
    X = profiles[features].fillna(0)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = km.fit_predict(X_scaled)

    # Auto-label archetypes based on cluster centers
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    centers["cluster"] = range(n_clusters)

    label_map = {}
    for _, row in centers.iterrows():
        c = int(row["cluster"])
        if row["win_rate"] > 0.55 and row["avg_pnl"] > 0:
            label_map[c] = "Disciplined Accumulator"
        elif row["liquidation_count"] > 1 or row["avg_pnl"] < 0:
            label_map[c] = "Reactive Liquidator"
        else:
            label_map[c] = "Momentum Chaser"

    # Ensure unique labels if logic overlaps
    used = set()
    all_labels = ["Disciplined Accumulator", "Momentum Chaser", "Reactive Liquidator"]
    for c in sorted(label_map):
        if label_map[c] in used:
            for lbl in all_labels:
                if lbl not in used:
                    label_map[c] = lbl
                    break
        used.add(label_map[c])

    profiles["archetype"] = profiles["cluster"].map(label_map)
    print(f"  [clustering] Archetypes: {profiles['archetype'].value_counts().to_dict()}")
    return profiles, centers, label_map


# ─────────────────────────────────────────────
# 4. PLOTTING HELPERS
# ─────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
}

def set_style():
    plt.rcParams.update(PLOT_STYLE)

def save_fig(fig, name: str, out_dir: str = "outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [plot] Saved → {path}")
    return path


def plot_pnl_by_sentiment(df: pd.DataFrame, out_dir: str = "outputs/figures"):
    """Bar chart: Average PnL per sentiment class."""
    set_style()
    order  = CONFIG["sentiment_order"]
    colors = [CONFIG["sentiment_colors"][s] for s in order]

    grp = df.groupby("sentiment")["closed_pnl"].mean().reindex(order).dropna()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    bars = ax.bar(grp.index, grp.values, color=[CONFIG["sentiment_colors"][s] for s in grp.index],
                  edgecolor="#30363d", linewidth=0.8, width=0.6)

    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_title("Average Closed PnL by Market Sentiment", pad=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg Closed PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (5 if val >= 0 else -15),
                f"${val:,.1f}", ha="center", va="bottom", fontsize=9, color="#e6edf3")

    plt.tight_layout()
    return save_fig(fig, "01_pnl_by_sentiment", out_dir)


def plot_winrate_by_sentiment(df: pd.DataFrame, out_dir: str = "outputs/figures"):
    """Win rate per sentiment class."""
    set_style()
    order = CONFIG["sentiment_order"]
    grp   = df.groupby("sentiment")["profit_flag"].mean().reindex(order).dropna() * 100

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    bars = ax.bar(grp.index, grp.values,
                  color=[CONFIG["sentiment_colors"][s] for s in grp.index],
                  edgecolor="#30363d", linewidth=0.8, width=0.6)

    ax.axhline(50, color="#8b949e", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.set_title("Win Rate (%) by Market Sentiment", pad=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()

    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=9, color="#e6edf3")

    plt.tight_layout()
    return save_fig(fig, "02_winrate_by_sentiment", out_dir)


def plot_trade_volume_by_sentiment(df: pd.DataFrame, out_dir: str = "outputs/figures"):
    """Trade count + avg trade size per sentiment."""
    set_style()
    order = CONFIG["sentiment_order"]

    count = df.groupby("sentiment")["closed_pnl"].count().reindex(order).dropna()
    size  = df.groupby("sentiment")["trade_size_usd"].mean().reindex(order).dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")

    colors = [CONFIG["sentiment_colors"][s] for s in count.index]
    ax1.bar(count.index, count.values, color=colors, edgecolor="#30363d", width=0.6)
    ax1.set_title("Trade Volume by Sentiment")
    ax1.set_ylabel("Number of Trades")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    colors2 = [CONFIG["sentiment_colors"][s] for s in size.index]
    ax2.bar(size.index, size.values, color=colors2, edgecolor="#30363d", width=0.6)
    ax2.set_title("Avg Trade Size (USD) by Sentiment")
    ax2.set_ylabel("Avg Trade Size (USD)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    return save_fig(fig, "03_volume_by_sentiment", out_dir)


def plot_top_vs_bottom_traders(profiles: pd.DataFrame, n: int = 10, out_dir: str = "outputs/figures"):
    """Compare top N vs bottom N traders on key metrics."""
    set_style()

    top    = profiles.nlargest(n, "total_pnl")
    bottom = profiles.nsmallest(n, "total_pnl")

    metrics = ["win_rate", "avg_trade_size", "liquidation_count"]
    labels  = ["Win Rate", "Avg Trade Size (USD)", "Liquidations"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")

    for ax, metric, label in zip(axes, metrics, labels):
        ax.set_facecolor("#161b22")
        vals_top = top[metric].values
        vals_bot = bottom[metric].values
        x = np.arange(n)
        w = 0.35

        ax.bar(x - w/2, vals_top, w, label="Top 10", color="#2ca02c", alpha=0.85, edgecolor="#30363d")
        ax.bar(x + w/2, vals_bot, w, label="Bottom 10", color="#d62728", alpha=0.85, edgecolor="#30363d")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{i+1}" for i in range(n)], fontsize=7)
        ax.legend(fontsize=8)

    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Top 10 vs Bottom 10 Traders — Key Metrics", fontsize=14, fontweight="bold",
                 color="#e6edf3", y=1.01)
    plt.tight_layout()
    return save_fig(fig, "04_top_vs_bottom_traders", out_dir)


def plot_sentiment_lag_correlation(df: pd.DataFrame, out_dir: str = "outputs/figures"):
    """
    Does yesterday's Fear/Greed score predict today's aggregate PnL?
    1-day lag correlation — shows signal-processing thinking.
    """
    set_style()

    daily = df.groupby("date").agg(
        total_pnl=("closed_pnl", "sum"),
        sentiment_score=("sentiment_score", "first"),
    ).reset_index().sort_values("date")

    daily["lagged_sentiment"] = daily["sentiment_score"].shift(1)
    daily = daily.dropna(subset=["lagged_sentiment"])

    corr = daily["lagged_sentiment"].corr(daily["total_pnl"])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    ax.scatter(daily["lagged_sentiment"], daily["total_pnl"],
               alpha=0.5, color="#58a6ff", s=18, edgecolors="none")

    # Trend line
    z = np.polyfit(daily["lagged_sentiment"], daily["total_pnl"], 1)
    p = np.poly1d(z)
    xs = np.linspace(daily["lagged_sentiment"].min(), daily["lagged_sentiment"].max(), 200)
    ax.plot(xs, p(xs), color="#f78166", linewidth=1.5, linestyle="--", label=f"Trend (r={corr:.3f})")

    ax.set_title("Lagged Sentiment Score vs Next-Day Aggregate PnL", pad=14)
    ax.set_xlabel("Yesterday's Fear/Greed Score (0=Extreme Fear, 100=Extreme Greed)")
    ax.set_ylabel("Next-Day Total PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.axhline(0, color="#8b949e", linewidth=0.6, linestyle=":")
    ax.axvline(50, color="#8b949e", linewidth=0.6, linestyle=":")

    plt.tight_layout()
    print(f"  [lag] Pearson r (lag-1 sentiment → next-day PnL): {corr:.4f}")
    return save_fig(fig, "05_sentiment_lag_correlation", out_dir), corr


def plot_archetypes(profiles: pd.DataFrame, out_dir: str = "outputs/figures"):
    """Scatter plot of trader archetypes by win rate vs avg PnL."""
    set_style()

    archetype_colors = {
        "Disciplined Accumulator": "#2ca02c",
        "Momentum Chaser":         "#1f77b4",
        "Reactive Liquidator":     "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d1117")

    for archetype, grp in profiles.groupby("archetype"):
        ax.scatter(grp["win_rate"] * 100, grp["avg_pnl"],
                   label=archetype,
                   color=archetype_colors.get(archetype, "#bcbd22"),
                   s=grp["total_trades"] / grp["total_trades"].max() * 300 + 30,
                   alpha=0.85, edgecolors="#30363d", linewidth=0.5)

        for _, row in grp.iterrows():
            ax.annotate(row["account"][:6] + "…",
                        (row["win_rate"] * 100, row["avg_pnl"]),
                        fontsize=6.5, color="#8b949e",
                        xytext=(3, 3), textcoords="offset points")

    ax.axhline(0, color="#8b949e", linewidth=0.7, linestyle="--")
    ax.axvline(50, color="#8b949e", linewidth=0.7, linestyle="--")
    ax.set_title("Trader Archetypes: Win Rate vs Avg PnL\n(bubble size = number of trades)", pad=14)
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Avg PnL per Trade (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    return save_fig(fig, "06_trader_archetypes", out_dir)


# ─────────────────────────────────────────────
# 5. INSIGHT GENERATORS
# ─────────────────────────────────────────────

def generate_summary_stats(df: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Return a dict of key numbers used in the written insights."""
    stats = {}

    stats["total_trades"]    = len(df)
    stats["total_accounts"]  = df["account"].nunique()
    stats["date_range"]      = f"{df['date'].min().date()} → {df['date'].max().date()}"
    stats["overall_win_rate"]= df["profit_flag"].mean()
    stats["overall_avg_pnl"] = df["closed_pnl"].mean()

    # PnL by sentiment
    pnl_sent = df.groupby("sentiment")["closed_pnl"].mean()
    stats["pnl_by_sentiment"] = pnl_sent.to_dict()

    # Win rate by sentiment
    wr_sent = df.groupby("sentiment")["profit_flag"].mean()
    stats["winrate_by_sentiment"] = wr_sent.to_dict()

    # Best/worst sentiment for trading
    stats["best_sentiment"]  = pnl_sent.idxmax()
    stats["worst_sentiment"] = pnl_sent.idxmin()

    # Top trader
    top = profiles.nlargest(1, "total_pnl").iloc[0]
    stats["top_trader_pnl"]      = top["total_pnl"]
    stats["top_trader_win_rate"] = top["win_rate"]

    # Liquidations
    stats["total_liquidations"] = df["is_liquidation"].sum()
    liq_by_sent = df.groupby("sentiment_binary")["is_liquidation"].sum()
    stats["liquidations_by_sentiment"] = liq_by_sent.to_dict()

    return stats

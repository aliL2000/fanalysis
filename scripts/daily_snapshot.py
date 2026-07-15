"""
Daily Snapshot Generator

Runs after each ingestion cycle. Does three things that go beyond a plain
"re-pull and re-chart" job:

1. Change detection — identifies stocks that newly entered an anomalous
   state today (flagged today, not flagged yesterday), not just which
   stocks are currently flagged. This is the interesting part: a static
   report tells you *what* is anomalous; this tells you *when it started*.
2. Appends today's findings to a persistent anomaly_history.csv, so the
   repo accumulates a real time series of flag events instead of only
   ever showing "right now."
3. Regenerates the risk-return scatter and Sharpe ranking charts, and
   writes a human-readable SNAPSHOT.md summarizing the day.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import psycopg2
from datetime import datetime

DATABASE_URL = os.environ["DATABASE_URL"]


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def load_latest_two_days_flags(conn):
    """Get anomaly flags for the two most recent trading dates, per symbol."""
    query = """
        WITH ranked AS (
            SELECT symbol, trade_date, is_anomalous,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
            FROM computed_metrics
        )
        SELECT symbol, trade_date, is_anomalous, rn
        FROM ranked
        WHERE rn IN (1, 2)
        ORDER BY symbol, rn;
    """
    return pd.read_sql(query, conn)


def detect_new_anomalies(flags_df):
    """A symbol is 'newly anomalous' if today (rn=1) is flagged and
    yesterday (rn=2) was not."""
    today = flags_df[flags_df["rn"] == 1].set_index("symbol")
    yesterday = flags_df[flags_df["rn"] == 2].set_index("symbol")
    merged = today.join(yesterday, lsuffix="_today", rsuffix="_yesterday", how="left")
    newly_flagged = merged[
        (merged["is_anomalous_today"] == True) &
        (merged["is_anomalous_yesterday"].fillna(False) == False)
    ]
    return newly_flagged.index.tolist(), today["trade_date"].iloc[0] if len(today) else None


def load_daily_movers(conn, trade_date):
    """Today's best and worst performing tickers by daily return."""
    query = """
        SELECT symbol, daily_return
        FROM daily_returns
        WHERE trade_date = %s
        ORDER BY daily_return DESC;
    """
    df = pd.read_sql(query, conn, params=(trade_date,))
    return df


def load_risk_summary(conn):
    query = """
        SELECT
            dr.symbol,
            ROUND(AVG(dr.daily_return) * 252 * 100, 2) AS annualized_return_pct,
            ROUND(STDDEV_SAMP(dr.daily_return) * SQRT(252) * 100, 2) AS annualized_vol_pct,
            ROUND(
                (AVG(dr.daily_return) * 252 - 0.04) / (STDDEV_SAMP(dr.daily_return) * SQRT(252)), 2
            ) AS sharpe,
            ROUND(
                (AVG(dr.daily_return) * 252 - 0.04) /
                (STDDEV_SAMP(dr.daily_return) FILTER (WHERE dr.daily_return < 0) * SQRT(252)), 2
            ) AS sortino,
            ROUND(SUM(CASE WHEN cm.is_anomalous THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_anomalous
        FROM daily_returns dr
        JOIN computed_metrics cm USING (symbol, trade_date)
        GROUP BY dr.symbol
        ORDER BY sharpe DESC;
    """
    return pd.read_sql(query, conn)


def regenerate_charts(risk_df):
    os.makedirs("visuals", exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#2ca02c" if s > 1.5 else "#1f77b4" if s > 0 else "#d62728" for s in risk_df["sharpe"]]
    ax.scatter(risk_df["annualized_vol_pct"], risk_df["annualized_return_pct"],
               s=risk_df["sharpe"].abs() * 120 + 40, c=colors, alpha=0.7,
               edgecolors="black", linewidth=0.8)
    for _, row in risk_df.iterrows():
        ax.annotate(row["symbol"], (row["annualized_vol_pct"], row["annualized_return_pct"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title(f"Risk-Return Profile — Updated {datetime.today().strftime('%Y-%m-%d')}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/risk_return_scatter.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 9))
    df_sorted = risk_df.sort_values("sharpe")
    colors_bar = ["#d62728" if s < 0 else "#2ca02c" if s > 1.5 else "#1f77b4" for s in df_sorted["sharpe"]]
    ax.barh(df_sorted["symbol"], df_sorted["sharpe"], color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Sharpe Ratio (rf=4%)")
    ax.set_title(f"Risk-Adjusted Ranking — Updated {datetime.today().strftime('%Y-%m-%d')}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("visuals/sharpe_ranking_bar.png", dpi=150)
    plt.close()


def append_anomaly_history(newly_flagged, trade_date):
    row_df = pd.DataFrame({
        "trade_date": [trade_date] * len(newly_flagged),
        "symbol": newly_flagged,
        "event": ["newly_flagged"] * len(newly_flagged),
    })
    path = "anomaly_history.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True).drop_duplicates()
    else:
        combined = row_df
    combined.to_csv(path, index=False)


def write_snapshot(trade_date, movers_df, newly_flagged, risk_df):
    top = movers_df.iloc[0]
    bottom = movers_df.iloc[-1]
    top_sharpe = risk_df.iloc[0]

    lines = [
        f"# Daily Risk Snapshot — {trade_date}",
        "",
        f"**Best mover:** {top['symbol']} ({top['daily_return']*100:+.2f}%)",
        f"**Worst mover:** {bottom['symbol']} ({bottom['daily_return']*100:+.2f}%)",
        f"**Top risk-adjusted performer (trailing window):** {top_sharpe['symbol']} "
        f"(Sharpe {top_sharpe['sharpe']})",
        "",
    ]
    if newly_flagged:
        lines.append(f"**⚠ Newly anomalous today:** {', '.join(newly_flagged)} — "
                      f"return moved more than 2 std. deviations from its own 20-day trailing mean.")
    else:
        lines.append("**No new anomaly flags today.**")

    lines.append("")
    lines.append("_Auto-generated by `scripts/daily_snapshot.py` via GitHub Actions._")

    with open("SNAPSHOT.md", "w") as f:
        f.write("\n".join(lines))


def main():
    conn = get_conn()
    flags_df = load_latest_two_days_flags(conn)
    newly_flagged, trade_date = detect_new_anomalies(flags_df)

    if trade_date is None:
        print("No data found — skipping snapshot.")
        return

    movers_df = load_daily_movers(conn, trade_date)
    risk_df = load_risk_summary(conn)

    regenerate_charts(risk_df)
    append_anomaly_history(newly_flagged, trade_date)
    write_snapshot(trade_date, movers_df, newly_flagged, risk_df)
    risk_df.to_csv("data_summary.csv", index=False)

    conn.close()
    print(f"Snapshot complete for {trade_date}. Newly flagged: {newly_flagged}")


if __name__ == "__main__":
    main()

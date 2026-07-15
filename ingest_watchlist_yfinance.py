"""
Portfolio Risk & Anomaly Monitoring Pipeline — Ingestion Script (yfinance + Postgres)
Pulls daily prices for a fixed 20-stock watchlist using yfinance (no API key needed)
and loads into a Postgres database (e.g. a free Neon project).

Setup:
    pip install yfinance pandas psycopg2-binary

Usage:
    export DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"
    python ingest_watchlist_yfinance.py
"""

import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WATCHLIST = [
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "AMD", "WMT",
    "JPM", "V", "JNJ", "XOM", "INTC", "CSCO", "ABBV", "BAC", "COST", "UNH", "GE"
]

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def get_last_pulled_date(conn, symbol: str) -> str:
    """Incremental load: only pull data newer than what's already stored."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(trade_date) FROM daily_prices WHERE symbol = %s", (symbol,))
    result = cur.fetchone()[0]
    cur.close()
    if result is None:
        # New symbol — pull 1 year of history
        return (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    return (result + timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_prices(symbol: str, from_date: str, to_date: str, max_retries: int = 2) -> pd.DataFrame:
    """Pull daily adjusted close prices for one symbol via yfinance, with retry."""
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=from_date, end=to_date, auto_adjust=True)
            if hist.empty:
                return pd.DataFrame()
            df = hist.reset_index()[["Date", "Close", "Volume"]].rename(
                columns={"Date": "trade_date", "Close": "close_price", "Volume": "volume"}
            )
            df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
            df["symbol"] = symbol
            return df[["symbol", "trade_date", "close_price", "volume"]]
        except Exception as e:
            logger.warning(f"{symbol}: attempt {attempt + 1} failed ({e})")
            time.sleep(2 ** attempt)  # exponential backoff
    raise RuntimeError(f"{symbol}: failed after {max_retries + 1} attempts")


def run_pipeline():
    conn = get_connection()
    today = datetime.today().strftime("%Y-%m-%d")
    succeeded, failed = [], []

    for symbol in WATCHLIST:
        from_date = get_last_pulled_date(conn, symbol)
        if from_date >= today:
            logger.info(f"{symbol}: already current, skipping")
            continue
        try:
            df = fetch_prices(symbol, from_date, today)
            if df.empty:
                logger.info(f"{symbol}: no new rows")
                continue
            cur = conn.cursor()
            for _, row in df.iterrows():
                cur.execute(
                    """INSERT INTO daily_prices (symbol, trade_date, close_price, volume)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (symbol, trade_date) DO NOTHING""",
                    (row.symbol, row.trade_date, row.close_price, int(row.volume)),
                )
            conn.commit()
            cur.close()
            succeeded.append(symbol)
            logger.info(f"{symbol}: inserted {len(df)} rows")
        except Exception as e:
            conn.rollback()
            failed.append((symbol, str(e)))
            logger.error(f"{symbol}: FAILED — {e}")
        time.sleep(0.5)  # be polite, yfinance has no official rate limit but don't hammer it

    cur = conn.cursor()
    cur.execute(
        """INSERT INTO ingestion_log
           (symbols_attempted, symbols_succeeded, symbols_failed, error_detail, latest_trade_date_pulled)
           VALUES (%s, %s, %s, %s, %s)""",
        (len(WATCHLIST), len(succeeded), len(failed), str(failed), today),
    )
    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"Pipeline complete: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed symbols: {failed}")


if __name__ == "__main__":
    run_pipeline()

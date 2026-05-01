"""
gnw_scraper.py — Scrape GlobeNewsWire via monthly sitemaps.

Parses sitemap metadata (ticker, title, date, keywords) for each month.

Fields: date, time, datetime, ticker, exchange, tickers, title, url, keywords

Sitemap coverage starts June 2023. Earlier dates are not supported.

Usage:
    python scraper/gnw_scraper.py                              # last 30 days
    python scraper/gnw_scraper.py --days 90
    python scraper/gnw_scraper.py --from 2023-06-01 --to 2024-12-31
    python scraper/gnw_scraper.py --from 2024-01-01 --tickers AAPL,MSFT
"""

import argparse
import csv
import os
import time
import random
from datetime import date, timedelta
from xml.etree import ElementTree as ET

from curl_cffi import requests

OUTPUT_CSV = "data/gnw_news.csv"
SITEMAP_BASE = "https://sitemaps.globenewswire.com/news/en"
SITEMAP_MIN_DATE = date(2023, 6, 1)

SITEMAP_DELAY = 0.5

CSV_FIELDS = ["date", "time", "datetime", "ticker", "exchange", "tickers", "title", "url", "keywords"]

SM_NS  = "http://www.sitemaps.org/schemas/sitemap/0.9"
NEWS_NS = "http://www.google.com/schemas/sitemap-news/0.9"
NS = {"sm": SM_NS, "news": NEWS_NS}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def parse_ticker(raw: str) -> tuple:
    """'NASDAQ:AAPL, NYSE:XYZ' → (primary_ticker, primary_exchange, pipe_joined_tickers)"""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    tickers_all = []
    primary_ticker = primary_exchange = ""
    for p in parts:
        if ":" in p:
            exchange, ticker = p.split(":", 1)
            tickers_all.append(ticker.strip())
            if not primary_ticker:
                primary_ticker = ticker.strip()
                primary_exchange = exchange.strip()
        else:
            tickers_all.append(p)
            if not primary_ticker:
                primary_ticker = p
    return primary_ticker, primary_exchange, "|".join(tickers_all)


def parse_dt(iso: str) -> tuple:
    """'2024-01-31T23:40:30-05:00' → (date_str, time_str, datetime_str)"""
    try:
        dt_part = iso[:19]
        d, t = dt_part.split("T")
        t_short = t[:5]
        return d, t_short, f"{d} {t_short}"
    except Exception:
        return iso[:10], "", iso[:10]


def fetch_sitemap(year: int, month: int, session) -> list:
    url = f"{SITEMAP_BASE}/{year}-{month:02d}.xml"
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
    except Exception as e:
        print(f"  sitemap {year}-{month:02d}: request error — {e}")
        return []

    if resp.status_code != 200:
        print(f"  sitemap {year}-{month:02d}: status={resp.status_code}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  sitemap {year}-{month:02d}: parse error — {e}")
        return []

    articles = []
    for url_el in root.findall(f"{{{SM_NS}}}url"):
        loc = url_el.findtext(f"{{{SM_NS}}}loc") or ""
        if not loc or "/news-release/" not in loc or "/en/" not in loc:
            continue

        news_el = url_el.find(f"{{{NEWS_NS}}}news")
        if news_el is None:
            continue

        pub_date    = news_el.findtext(f"{{{NEWS_NS}}}publication_date") or ""
        title       = news_el.findtext(f"{{{NEWS_NS}}}title") or ""
        raw_tickers = news_el.findtext(f"{{{NEWS_NS}}}stock_tickers") or ""
        keywords    = news_el.findtext(f"{{{NEWS_NS}}}keywords") or ""

        d, t, dt = parse_dt(pub_date)
        ticker, exchange, tickers_all = parse_ticker(raw_tickers) if raw_tickers else ("", "", "")

        articles.append({
            "date": d, "time": t, "datetime": dt,
            "ticker": ticker, "exchange": exchange, "tickers": tickers_all,
            "title": title, "url": loc, "keywords": keywords,
        })

    return articles



def load_existing_urls() -> set:
    if not os.path.exists(OUTPUT_CSV):
        return set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        return {row["url"] for row in csv.DictReader(f)}


def append_rows(rows: list):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def months_in_range(start: date, end: date):
    """Yield (year, month) tuples from newest to oldest."""
    seen = set()
    d = end
    while d >= start:
        ym = (d.year, d.month)
        if ym not in seen:
            seen.add(ym)
            yield ym
        d -= timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--tickers", help="comma-separated ticker filter e.g. AAPL,MSFT")
    args = parser.parse_args()

    today = date.today()
    if args.from_date:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date) if args.to_date else today
    else:
        end = today
        start = end - timedelta(days=args.days - 1)

    if start < SITEMAP_MIN_DATE:
        print(f"Warning: GNW sitemaps start {SITEMAP_MIN_DATE}. Clamping start date.")
        start = SITEMAP_MIN_DATE

    ticker_filter = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None

    print(f"Scraping {start} to {end}")
    if ticker_filter:
        print(f"Ticker filter: {ticker_filter}")
    print(f"Output: {OUTPUT_CSV}\n")

    existing_urls = load_existing_urls()
    print(f"Already have {len(existing_urls)} articles in CSV\n")

    session = requests.Session(impersonate="chrome124")
    total_new = 0

    for year, month in months_in_range(start, end):
        print(f"  {year}-{month:02d} ...", end=" ", flush=True)
        articles = fetch_sitemap(year, month, session)

        # clamp to exact date range (sitemap is monthly, may include out-of-range dates)
        articles = [
            a for a in articles
            if a["date"] and start <= date.fromisoformat(a["date"]) <= end
        ]

        if ticker_filter:
            articles = [
                a for a in articles
                if a["ticker"].upper() in ticker_filter
                or any(t.upper() in ticker_filter for t in a["tickers"].split("|") if t)
            ]

        new = [a for a in articles if a["url"] not in existing_urls]
        print(f"{len(articles)} matched  {len(new)} new")

        if new:
            append_rows(new)
            for a in new:
                existing_urls.add(a["url"])
            total_new += len(new)

        time.sleep(SITEMAP_DELAY + random.uniform(0, 0.3))

    print(f"\nDone. {total_new} new articles added to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

"""
anw_classifier.py — Classify ANW signal articles: ticker, exchange, catalyst.

Input:  data/anw/anw_signal_articles.csv  (output of anw_extract_fields.py)
Output: data/anw/anw_classified.csv       (datetime, url, ticker, exchange, catalyst)

Two-step ticker resolution (cheapest first):
  1. anw_tickers (body-parsed pipe-joined EXCHANGE:SYM) → first symbol in universe
  2. anw_source_company → name-index prefix match (same lookup used by GNW/PRNW)
  Rows with no resolved listed ticker are dropped.

Catalyst is classified from anw_title (full og:title, not the truncated slug).
Append-safe: skips URLs already in output.

Usage:
  python -m sources.anw.anw_classifier
  python -m sources.anw.anw_classifier --input data/anw/anw_signal_articles.csv
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Article bodies can be large; raise the per-field cap.
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit //= 10

from config.paths import ANW_ARTICLES, ANW_CLASSIFIED, TICKER_UNIVERSE, ensure_dirs
from sources.prnw.prnw_classifier import build_ticker_index, lookup_ticker, _LISTED_EXCHANGES
from sources.gnw.gnw_classifier import build_ticker_to_mic
from regex.catalysts import classify_catalyst

_ET = ZoneInfo("America/New_York")


def _parse_anw_dt(s: str) -> str | None:
    """TZ-naive ISO from og:article:published_time → ISO with ET offset."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=_ET).isoformat()
    except Exception:
        return None


def _first_listed_ticker(anw_tickers: str, ticker_to_mic: dict) -> tuple[str, str] | None:
    """Pipe-joined EXCHANGE:SYM → first (ticker, MIC) present in ticker_universe."""
    if not anw_tickers:
        return None
    for part in anw_tickers.split("|"):
        sym = part.strip().split(":", 1)[-1].strip().upper()
        if not sym:
            continue
        mic = ticker_to_mic.get(sym)
        if mic and mic in _LISTED_EXCHANGES:
            return sym, mic
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=ANW_ARTICLES)
    parser.add_argument("--output", default=ANW_CLASSIFIED)
    parser.add_argument("--ticker-universe", default=TICKER_UNIVERSE)
    args = parser.parse_args()

    ensure_dirs()

    done_urls: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            done_urls = {row["url"] for row in csv.DictReader(f)}
        print(f"Resuming — {len(done_urls)} URLs already classified")

    print(f"Building indexes from {args.ticker_universe}...")
    name_index, sorted_keys = build_ticker_index(args.ticker_universe)
    ticker_to_mic = build_ticker_to_mic(args.ticker_universe)
    print(f"  {len(name_index)} names, {len(ticker_to_mic)} tickers loaded")

    fieldnames = ["datetime", "url", "ticker", "exchange", "catalyst"]
    write_header = not os.path.exists(args.output)

    total = no_dt = no_ticker = kept = 0

    with open(args.input, encoding="utf-8") as f_in, \
         open(args.output, "a", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url or url in done_urls:
                continue
            total += 1

            dt = _parse_anw_dt(row.get("anw_published_time") or "")
            if not dt:
                no_dt += 1
                continue

            hit = _first_listed_ticker(row.get("anw_tickers") or "", ticker_to_mic)
            if not hit:
                hit = lookup_ticker(row.get("anw_source_company") or "", name_index, sorted_keys)
                if hit and hit[1] not in _LISTED_EXCHANGES:
                    hit = None
            if not hit:
                no_ticker += 1
                continue

            ticker, exchange = hit
            catalyst = str(classify_catalyst(row.get("anw_title") or ""))
            writer.writerow({"datetime": dt, "url": url, "ticker": ticker,
                             "exchange": exchange, "catalyst": catalyst})
            done_urls.add(url)
            kept += 1

    print(f"\ntotal rows processed : {total:,}")
    print(f"  dropped no datetime : {no_dt:,}")
    print(f"  dropped no ticker   : {no_ticker:,}")
    print(f"  KEPT                : {kept:,}")

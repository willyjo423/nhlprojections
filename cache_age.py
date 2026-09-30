"""How old is the history? Printed every run, and warned about when it matters.

A page built from a frozen cache looks exactly like a page built from a
current one: same layout, same plausible numbers, same timestamp in the
footer saying it was generated this morning. The only thing that differs is
that every rate is weeks out of date, which is invisible unless something
says so out loud.

    python cache_age.py            # prints, warns, exits 0
    python cache_age.py --max-age 4 --fail
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys

import pandas as pd


def newest_game(pattern: str = "data/skaters_*.csv.gz"):
    newest = None
    files = sorted(glob.glob(pattern))
    for f in files:
        try:
            col = pd.read_csv(f, usecols=["date"])["date"]
        except Exception as exc:                               # noqa: BLE001
            print(f"  {f}: unreadable ({type(exc).__name__})")
            continue
        d = pd.to_datetime(col, errors="coerce").max()
        if pd.isna(d):
            continue
        print(f"  {f}: newest game {d.date()}")
        newest = d if newest is None or d > newest else newest
    return newest, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=int, default=4,
                    help="days behind before this is worth shouting about")
    ap.add_argument("--fail", action="store_true",
                    help="exit non-zero when the history is too old")
    ap.add_argument("--today", default=None, help="for testing")
    a = ap.parse_args()

    newest, files = newest_game()
    if not files:
        print("::error::no history cache in data/ at all")
        return 1
    if newest is None:
        print("::error::the cache has no readable dates in it")
        return 1

    today = (dt.date.fromisoformat(a.today) if a.today else dt.date.today())
    age = (today - newest.date()).days
    print(f"newest game in the history: {newest.date()} ({age} days old)")
    if age > a.max_age:
        # A warning rather than a failure by default: yesterday's projections
        # from a slightly stale history are worth more than no page at all.
        print(f"::warning::The history is {age} days behind, so every rate on "
              f"the page is that stale. The nightly 'Fetch history' schedule "
              f"may be failing - check the Actions tab.")
        if a.fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

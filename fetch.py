"""Download the history once and keep it.

MoneyPuck publishes careers, not slates, so building the history means one
HTTP request per player - about a thousand of them, roughly twelve minutes.
That is far too slow to do before every projection, so it is done here and
cached as gzipped CSV in `data/`, and the daily job just reads the cache.

`--limit` is a SMOKE TEST AND WRITES NOTHING. That is deliberate and it is a
bug I have already written once: saving twenty-five players under the real
filename makes the season look cached, and the real fetch then skips it
forever. A smoke test that poisons the thing it is testing is worse than no
smoke test.
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import pandas as pd

import source as SRC

log = logging.getLogger("fetch")
DATA = pathlib.Path(__file__).parent / "data"


def season_file(season: int, side: str) -> pathlib.Path:
    return DATA / f"{side}_{season}.csv.gz"


def fetch_season(season: int, side: str, limit: int = 0,
                 pause: float = 0.05) -> pd.DataFrame:
    directory = SRC.season_summary(season, side)
    ids = directory["player_id"].dropna().unique().tolist()
    if limit:
        ids = ids[:limit]
        log.warning("SMOKE TEST: %d of %d %s, and NOTHING WILL BE SAVED",
                    len(ids), len(directory), side)
    log.info("%s %s: %d players to fetch", season, side, len(ids))
    hist = SRC.game_by_game(ids, side=side, pause=pause)
    # The directory is the only place a current team lives; the game logs
    # carry the team he played for at the time, which is what the model wants
    # for history and exactly wrong for tonight's roster.
    hist = hist.merge(
        directory[["player_id", "name"]].drop_duplicates("player_id"),
        on="player_id", how="left", suffixes=("", "_dir"))
    hist["name"] = hist["name"].fillna(hist.pop("name_dir"))
    return hist


def load(seasons: list[int], side: str, refresh: list[int] | None = None,
         limit: int = 0) -> pd.DataFrame:
    refresh = set(refresh or [])
    frames, missing = [], []
    for s in seasons:
        path = season_file(s, side)
        if path.exists() and s not in refresh and not limit:
            df = pd.read_csv(path, low_memory=False)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            log.info("%s: %d rows from cache", path.name, len(df))
            frames.append(df)
            continue
        try:
            df = fetch_season(s, side, limit=limit)
        except SRC.NotPlayedYet as exc:
            # A season that has not started yet is not a failure. In
            # September the current season's file is empty and that is simply
            # what September looks like.
            log.warning("%s %s skipped: %s", s, side, exc)
            missing.append(s)
            continue
        if not limit:
            DATA.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, compression="gzip")
            log.info("wrote %s (%d rows)", path.name, len(df))
        frames.append(df)

    if not frames:
        raise SRC.DataUnavailable(
            f"no {side} history at all for seasons {seasons} "
            f"(skipped as unplayed: {missing or 'none'})")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023,2024,2025",
                    help="MoneyPuck season start years, comma separated")
    ap.add_argument("--refresh", default="",
                    help="seasons to re-download even if cached")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test only - fetches N players and SAVES NOTHING")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    refresh = [int(s) for s in a.refresh.split(",") if s.strip()]
    for side in ("skaters", "goalies"):
        df = load(seasons, side, refresh=refresh, limit=a.limit)
        print(f"{side}: {len(df)} player-games, "
              f"{df['player_id'].nunique()} players, "
              f"{df['game_id'].nunique()} games")

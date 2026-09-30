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


# Columns whose absence means the cache was written by an older tidier. A
# file missing these is re-fetched rather than concatenated beside a current
# one, because pandas would fill the gap with NaN and the model would then be
# reading empty game-state columns for most of the history.
CACHE_MUST_HAVE = {
    "skaters": ["rest_toi", "pp_toi", "pk_toi",
                "rest_goals", "rest_sog", "rest_assists",
                # The substitute columns too, and their per-state versions.
                # Leaving `xg` off this list is how a cache written before xG
                # existed passed the check, concatenated beside a current
                # file, and put half the rows' xG at NaN.
                "xg", "rest_xg", "pp_xg",
                "attempts", "rest_attempts",
                "onice_xg", "rest_onice_xg"],
    "goalies": ["shots_against", "saves"],
}

# THE KEYS ARE STRINGS, AND A CSV DOES NOT REMEMBER THAT.
#
# A fresh frame carries game_id as str; the same frame read back from the
# cache carries it as int64. '2025020123' never equals 2025020123, so the
# de-duplication below matched nothing at all, dropped zero rows, and the
# distinct-game count went from 21,392 to 38,225 - the same games counted
# twice, once in each dtype. Pinned on read so the round trip is lossless.
KEY_DTYPES = {"player_id": str, "game_id": str,
               # team and opponent for the same reason: a blank team becomes
               # a float NaN on the way back in, and a set holding a NaN
               # beside a string cannot be sorted.
               "team": str, "opponent": str}


def read_cache(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype=KEY_DTYPES)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def fetch_season(season: int, side: str, limit: int = 0,
                 pause: float = 0.15) -> pd.DataFrame:
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
         limit: int = 0, since=None) -> pd.DataFrame:
    refresh = set(refresh or [])
    frames, missing = [], []
    for s in seasons:
        path = season_file(s, side)
        if path.exists() and s not in refresh and not limit:
            df = read_cache(path)
            stale = [c for c in CACHE_MUST_HAVE.get(side, []) if c not in df.columns]
            if stale:
                # A cache written by an older version of the tidier. Left
                # alone it would concat beside a current file and fill the
                # missing columns with NaN for most of the history - which is
                # worse than having no cache at all, because the model would
                # see game-state columns, believe them, and find them empty.
                log.warning("%s predates %s, so it is being RE-FETCHED rather "
                            "than mixed with current files",
                            path.name, ", ".join(stale))
            else:
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
    # Belt and braces: whatever route a frame took to get here, the keys are
    # strings before anything is compared on them.
    for k in KEY_DTYPES:
        if k in out.columns:
            out[k] = out[k].astype(str)

    # MONEYPUCK'S gameByGame FILES ARE WHOLE CAREERS, NOT SEASONS.
    #
    # The path says `careers/gameByGame/regular/skaters/{id}.csv` and it means
    # it: one request returns every game that player has ever played. The
    # season in the URL only chooses WHICH PLAYERS are in that directory, not
    # which of their games come back.
    #
    # So fetching three seasons downloads most players three times and every
    # one of their games appears three times in the concat. Nothing crashes.
    # The rates barely move, because a tripled numerator over a tripled
    # denominator is the same ratio - which is exactly why this survived a
    # clean-looking run. What it breaks is everything that counts: the
    # shrinkage thinks it has three times the evidence and therefore trusts a
    # player's own numbers far more than it should, and the exponential decay
    # counts each game three times in a row, so a twenty-game halflife
    # silently becomes a seven-game one.
    before = len(out)
    out = out.drop_duplicates(["player_id", "game_id"], keep="first")
    if len(out) != before:
        log.warning("dropped %d duplicate %s rows (%d -> %d). This is normal "
                    "when several seasons are requested: each one downloads "
                    "the same career files.", before - len(out), side,
                    before, len(out))

    if since is not None:
        cut = pd.Timestamp(since)
        n = len(out)
        out = out[out["date"] >= cut]
        log.info("%s: kept %d of %d rows on or after %s", side, len(out), n,
                 cut.date())
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025,2026",
                    help="MoneyPuck season start years, comma separated")
    ap.add_argument("--refresh", default="",
                    help="seasons to re-download even if cached")
    ap.add_argument("--refresh-latest", action="store_true",
                    help="re-download the most recent season. THIS IS WHAT "
                         "KEEPS THE MODEL CURRENT: a cached season file is "
                         "never re-read from the source, so without this the "
                         "history freezes on the day it was fetched and the "
                         "page goes on producing confident projections from "
                         "a month-old view of every player.")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test only - fetches N players and SAVES NOTHING")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    refresh = [int(s) for s in a.refresh.split(",") if s.strip()]
    if a.refresh_latest and seasons:
        latest = max(seasons)
        if latest not in refresh:
            refresh.append(latest)
        log.info("refreshing the %s season so last night's games are in the "
                 "history", latest)
    for side in ("skaters", "goalies"):
        df = load(seasons, side, refresh=refresh, limit=a.limit)
        print(f"{side}: {len(df)} player-games, "
              f"{df['player_id'].nunique()} players, "
              f"{df['game_id'].nunique()} games")

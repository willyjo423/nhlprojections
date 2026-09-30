"""MoneyPuck, read carefully.

Every REQUIRED column name in this file is a fact, taken from a probe run
against the live files, not a guess. Where a name is pinned rather than
searched for, the comment says what the wrong match would have been - because
in several places the obvious search finds a real column that means something
else entirely, and each of those produces a model that trains cleanly and is
wrong.

Run `python source.py --probe` to print every column, dtype and sample value
of every source and build nothing.

The traps, all confirmed against the real files
-----------------------------------------------
**`shotsBlockedByPlayer` is not `I_F_blockedShotAttempts`.** Both exist. The
first is blocks HE made. The second is HIS OWN shots that got blocked, which
is close to the opposite thing. A search for "blocked" finds the wrong one on
a coin flip.

**The team column is `playerTeam`, and there is no bare `team`.** A
contains-match on "team" finds `playerTeam` AND `opposingTeam`, and
longest-first tie-breaking picks `opposingTeam` - silently assigning every
player to the club he was playing against.

**`gameDate` is an INTEGER like 20191008.** Handing that to `pd.to_datetime`
without a format reads it as nanoseconds since 1970 and dates every game to
January 1970, which sorts the whole history into one day and destroys every
rolling feature without raising anything.

**Goalies have no saves column.** They have `ongoal` (shots faced) and
`goals` (goals ALLOWED), and saves is the difference. A search for "goals"
finds `goals` and quietly reads goals-allowed as goals-scored.

**Ice time is in SECONDS.** A factor-of-sixty error here would not look like
an error, it would look like every skater being unremarkable, so the unit is
decided by inspecting the median rather than asserted.

**Power-play ice time is a ROW, not a column.** MoneyPuck publishes one row
per player per game per game state, so the `5on4` row's icetime IS power-play
ice time. Reading the file unfiltered quintuples the history.

Season aggregates are not used as features
------------------------------------------
`season_summary` is a whole-season average, so it already contains the outcome
of any game you would use it to predict. It is read for the player directory -
who exists, what they are called, where they play - and nothing else.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

log = logging.getLogger("source")

TIMEOUT = 30
RETRIES = 3

MP = "https://moneypuck.com/moneypuck"
SEASON_SKATERS = f"{MP}/playerData/seasonSummary/{{season}}/regular/skaters.csv"
SEASON_GOALIES = f"{MP}/playerData/seasonSummary/{{season}}/regular/goalies.csv"
GAME_SKATERS = f"{MP}/playerData/careers/gameByGame/regular/skaters/{{pid}}.csv"
GAME_GOALIES = f"{MP}/playerData/careers/gameByGame/regular/goalies/{{pid}}.csv"

SITUATION_ALL = "all"
SITUATION_PP = "5on4"


class DataUnavailable(RuntimeError):
    """A source did not answer. The caller decides whether that is fatal."""


class NotPlayedYet(RuntimeError):
    """A season that has not happened. Normal in September, not an error."""


# --------------------------------------------------------------- fetching
def _get(url: str) -> bytes:
    """One fetch, retried only where retrying can possibly help.

    A 4xx IS NOT RETRIED. A refusal is a decision the far end has already
    made; asking twice more, a second apart, is three offences instead of one
    against a host that has just said no. 5xx and timeouts ARE retried -
    those are the far end having a bad moment, which is what a retry is for.
    429 gets a long wait, because it is the one refusal that means "later".
    """
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={
                "User-Agent": "Mozilla/5.0 (compatible; nhl-projections/1.0)"})
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or 20)
                log.warning("%s asked us to slow down (429); waiting %.0fs",
                            url.split("/")[2], wait)
                time.sleep(min(wait, 60))
                continue
            if 400 <= r.status_code < 500:
                raise DataUnavailable(
                    f"{url} -> HTTP {r.status_code}. A refusal, not a hiccup, "
                    f"so it is not being retried.")
        except DataUnavailable:
            raise
        except Exception as exc:                               # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))
    raise DataUnavailable(f"{url} -> {last}")


def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(url)), low_memory=False)


# --------------------------------------------------------------- helpers
def need(df: pd.DataFrame, col: str, what: str) -> pd.Series:
    """A column this file knows the exact name of, or a full explanation.

    Deliberately NOT a fuzzy search. Every name here was read off the real
    file, and a fuzzy fallback is what puts `opposingTeam` in the team column.
    If a name has moved, the right response is a probe run and a one-line
    edit, not a guess that might land on a column meaning the opposite.
    """
    if col in df.columns:
        return df[col]
    raise DataUnavailable(
        f"the {what} column '{col}' is not in this file - MoneyPuck has "
        f"renamed it.\nEVERY COLUMN PRESENT:\n  "
        + "\n  ".join(map(str, df.columns))
        + "\n\nRun `python source.py --probe` and update the constant.")


def optional(df: pd.DataFrame, names: list[str], what: str) -> pd.Series | None:
    """A stat worth having but not worth failing over.

    A candidate list is safe HERE and nowhere else in this file: for hits and
    penalty minutes there is no near-miss column that means something
    dangerously different, so taking the first name that exists costs nothing.
    When none of them exists the stat is dropped and SAID OUT LOUD, rather
    than silently projected as zero - a column of zeros looks like a player
    who never does the thing, which is a lie a table cannot defend itself
    against.
    """
    for n in names:
        if n in df.columns:
            log.info("%s: using column '%s'", what, n)
            return pd.to_numeric(df[n], errors="coerce")
    log.warning("%s: none of %s is in this file, so the stat is DROPPED "
                "rather than projected as zero", what, names)
    return None


def normalise(name) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s).strip()


def _situation(df: pd.DataFrame, keep: str, where: str) -> pd.DataFrame:
    """One row per player-game, at one game state.

    Without this every player appears five times - all, 5on5, 5on4, 4on5,
    other - and the history silently quintuples. It trains cleanly and it is
    wrong, which is the expensive kind of wrong.
    """
    if "situation" not in df.columns:
        log.info("%s: no situation column; rows are already one per game", where)
        return df
    values = df["situation"].astype(str).str.strip()
    out = df[values == keep].copy()
    if not len(out):
        log.error("%s: no rows at situation '%s'. Present: %s", where, keep,
                  ", ".join(sorted(set(values))[:8]))
    return out


def _minutes(s: pd.Series, where: str = "") -> pd.Series:
    """Ice time in minutes. MoneyPuck publishes SECONDS.

    Decided by the data rather than asserted, because a factor-of-sixty error
    in the single most important column in this model would not look like an
    error - it would look like every skater being unremarkable.
    """
    num = pd.to_numeric(s, errors="coerce")
    med = float(num.dropna().median()) if num.notna().any() else 0.0
    if med > 200:
        log.info("%s: ice time is SECONDS (median %.0f), divided by 60",
                 where, med)
        return num / 60.0
    return num


def _dates(s: pd.Series) -> pd.Series:
    """`gameDate` is an integer like 20191008, not an epoch."""
    text = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


# --------------------------------------------------------------- sources
def season_summary(season: int, side: str = "skaters") -> pd.DataFrame:
    """The player directory. NOT features - every column here is a leak."""
    pat = SEASON_SKATERS if side == "skaters" else SEASON_GOALIES
    url = pat.format(season=season)
    df = _read_csv(url)
    if not len(df):
        raise NotPlayedYet(f"{season} {side}: the file is empty")
    log.info("%s %s: %d rows", season, side, len(df))
    df = _situation(df, SITUATION_ALL, f"{season} {side}")
    out = pd.DataFrame({
        "player_id": need(df, "playerId", "player id").astype(str),
        "name": need(df, "name", "player name").astype(str),
        "team": need(df, "team", "team").astype(str).str.upper(),
        "position": need(df, "position", "position").astype(str).str.upper(),
        "season": season,
    })
    out["norm"] = out["name"].map(normalise)
    return out


def game_by_game(player_ids, side: str = "skaters",
                 pause: float = 0.05) -> pd.DataFrame:
    """One row per player per game - the only thing a model may be fitted on.

    Fetched per player, because MoneyPuck publishes careers rather than
    slates. One player failing is logged and skipped: losing a skater's
    history costs a little accuracy, losing the fetch costs the night.
    """
    pat = GAME_SKATERS if side == "skaters" else GAME_GOALIES
    frames, failed = [], []
    ids = [str(p) for p in player_ids]
    for n, pid in enumerate(ids, start=1):
        try:
            df = _read_csv(pat.format(pid=pid))
        except Exception as exc:                               # noqa: BLE001
            failed.append(pid)
            if len(failed) <= 5:
                log.warning("no game log for %s (%s)", pid, str(exc)[:70])
            continue
        if len(df):
            frames.append(df)
        if n % 100 == 0:
            log.info("  %d of %d %s fetched", n, len(ids), side)
        time.sleep(pause)
    if not frames:
        raise DataUnavailable(
            f"not one {side} game log could be fetched. Run --probe.")
    if failed:
        log.warning("%d of %d %s had no game log and were skipped",
                    len(failed), len(ids), side)
    raw = pd.concat(frames, ignore_index=True)
    return _tidy_skaters(raw) if side == "skaters" else _tidy_goalies(raw)


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": need(df, "playerId", "player id").astype(str),
        "name": need(df, "name", "player name").astype(str),
        "game_id": need(df, "gameId", "game id").astype(str),
        "date": _dates(need(df, "gameDate", "game date")),
        # PINNED. `opposingTeam` also matches a search for "team" and would
        # assign every player to the club he was playing against.
        "team": need(df, "playerTeam", "the player's OWN team")
                .astype(str).str.upper(),
        "opponent": need(df, "opposingTeam", "opponent").astype(str).str.upper(),
        "is_home": (need(df, "home_or_away", "home or away")
                    .astype(str).str.upper().eq("HOME").astype(float)),
        "position": need(df, "position", "position").astype(str).str.upper(),
    })


# Stats this model will report if MoneyPuck carries them, and skip if not.
# Order matters: the first name present wins.
OPTIONAL_SKATER = {
    "hits": ["I_F_hits", "hits"],
    # MoneyPuck's own spelling has historically been "penality". Both are
    # tried; if neither is there the column is dropped, not zeroed.
    "pim": ["penalityMinutes", "penaltyMinutes",
            "I_F_penalityMinutes", "I_F_penaltyMinutes"],
    "takeaways": ["I_F_takeaways", "takeaways"],
    "giveaways": ["I_F_giveaways", "giveaways"],
}


def _tidy_skaters(raw: pd.DataFrame) -> pd.DataFrame:
    rows = _situation(raw, SITUATION_ALL, "skater game logs")
    out = _identity(rows)

    out["toi"] = _minutes(need(rows, "icetime", "ice time"), "skater ice time")
    out["sog"] = pd.to_numeric(
        need(rows, "I_F_shotsOnGoal", "shots on goal"), errors="coerce")
    out["goals"] = pd.to_numeric(
        need(rows, "I_F_goals", "goals"), errors="coerce")

    # Primary and secondary assists are SUMMED. Taking whichever column
    # matched first would drop roughly a third of every playmaker's assists
    # and look entirely plausible on the page.
    a1 = pd.to_numeric(need(rows, "I_F_primaryAssists", "primary assists"),
                       errors="coerce").fillna(0)
    a2 = pd.to_numeric(need(rows, "I_F_secondaryAssists", "secondary assists"),
                       errors="coerce").fillna(0)
    out["assists"] = a1 + a2
    out["points"] = out["goals"].fillna(0) + out["assists"]

    # BLOCKS HE MADE. `I_F_blockedShotAttempts` is his own shots that got
    # blocked and also matches a search for "blocked".
    out["blocks"] = pd.to_numeric(
        need(rows, "shotsBlockedByPlayer", "blocks the player MADE"),
        errors="coerce")

    # Expected goals. The whole reason to read MoneyPuck rather than a box
    # score: xG predicts future scoring materially better than goals do,
    # because a goal is one bounce and xG is the chance that created it.
    out["xg"] = pd.to_numeric(rows.get("I_F_xGoals"), errors="coerce")

    for key, names in OPTIONAL_SKATER.items():
        col = optional(rows, names, key)
        if col is not None:
            out[key] = col

    # POWER-PLAY ICE TIME IS A ROW, NOT A COLUMN.
    pp = _situation(raw, SITUATION_PP, "power-play rows")
    if len(pp):
        pptoi = pd.DataFrame({
            "player_id": need(pp, "playerId", "player id").astype(str),
            "game_id": need(pp, "gameId", "game id").astype(str),
            "pp_toi": _minutes(need(pp, "icetime", "pp ice time"), "pp ice time"),
        }).drop_duplicates(["player_id", "game_id"])
        before = len(out)
        out = out.merge(pptoi, on=["player_id", "game_id"], how="left")
        if len(out) != before:
            raise DataUnavailable(
                f"the power-play join fanned out {before} rows to {len(out)}")
    else:
        out["pp_toi"] = float("nan")
        log.warning("no 5on4 rows - power-play ice time is empty, and it is "
                    "the largest single difference between two otherwise "
                    "identical forwards")

    out["played"] = (out["toi"].fillna(0) > 0).astype(int)
    return _finish(out, "skaters")


def _tidy_goalies(raw: pd.DataFrame) -> pd.DataFrame:
    rows = _situation(raw, SITUATION_ALL, "goalie game logs")
    out = _identity(rows)
    out["toi"] = _minutes(need(rows, "icetime", "ice time"), "goalie ice time")

    # THERE IS NO SAVES COLUMN. `ongoal` is shots faced, `goals` is goals
    # ALLOWED, and saves is the difference. Read the other way round a model
    # rewards a goalie for being beaten.
    shots = pd.to_numeric(need(rows, "ongoal", "shots on goal faced"),
                          errors="coerce")
    against = pd.to_numeric(need(rows, "goals", "goals ALLOWED"),
                            errors="coerce")
    out["shots_against"] = shots
    out["goals_against"] = against
    out["saves"] = (shots - against).clip(lower=0)
    out["xga"] = pd.to_numeric(rows.get("xGoals"), errors="coerce")
    out["position"] = "G"
    out["played"] = (out["toi"].fillna(0) > 0).astype(int)
    return _finish(out, "goalies")


def _finish(out: pd.DataFrame, side: str) -> pd.DataFrame:
    out = out[out["date"].notna()].sort_values(["player_id", "date"])
    dup = int(out.duplicated(["player_id", "game_id"]).sum())
    if dup:
        log.error("%d duplicate player-game rows survived the situation "
                  "filter - every rolling average is now double-counting", dup)
    log.info("%s: %d player-games, %s to %s", side, len(out),
             out["date"].min().date() if len(out) else "-",
             out["date"].max().date() if len(out) else "-")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- probe
def probe(season: int) -> None:
    """Print what the sources actually contain and build nothing.

    This exists because the alternative - guessing a column name, shipping,
    reading the traceback, guessing again - costs a whole evening per guess.
    One read of the real file answers every question at once.
    """
    def show(title, df):
        print("\n" + "=" * 70)
        print(title, f"- {len(df)} rows, {len(df.columns)} columns")
        print("=" * 70)
        for c in df.columns:
            sample = df[c].dropna().head(3).tolist()
            print(f"  {c:38s} {str(df[c].dtype):10s} {sample}")

    sk = _read_csv(SEASON_SKATERS.format(season=season))
    show(f"seasonSummary/{season}/regular/skaters.csv", sk)
    go = _read_csv(SEASON_GOALIES.format(season=season))
    show(f"seasonSummary/{season}/regular/goalies.csv", go)
    if len(sk):
        pid = str(sk["playerId"].iloc[0])
        show(f"gameByGame skater {pid}", _read_csv(GAME_SKATERS.format(pid=pid)))
    if len(go):
        pid = str(go["playerId"].iloc[0])
        show(f"gameByGame goalie {pid}", _read_csv(GAME_GOALIES.format(pid=pid)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--season", type=int, default=2025)
    a = ap.parse_args()
    if a.probe:
        probe(a.season)
    else:
        print("nothing to do; --probe prints every column of every source")
        sys.exit(0)

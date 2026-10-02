"""Build tonight's projections and write them where the page can read them.

    python project.py                      # tonight, from the NHL schedule
    python project.py --date 2026-10-09
    python project.py --games "TOR@MTL,EDM@CGY"     # no network needed

Writes `docs/data/<date>.json` for the page, `docs/data/<date>_skaters.csv`
and `_goalies.csv` for a spreadsheet, and rebuilds `docs/data/index.json` from
every file present so an old slate never disappears because today's run
failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import pathlib

import pandas as pd

import fetch as FETCH
import model as M
import slate as SLATE
import source as SRC

log = logging.getLogger("project")
DOCS = pathlib.Path(__file__).parent / "docs" / "data"

# Two different windows, for two different jobs.
#
# ROSTER_DAYS decides who is even listed. It has to span an off-season,
# because on the second night of a new season a thirty-day window contains
# one game and the page would come up nearly empty - the roster would be
# whoever happened to dress on opening night. A year and a bit means "the
# most recent team he actually played for", which also handles trades
# correctly: his latest appearance is the one that counts.
#
# FRESH_DAYS is the flag, not the filter. Somebody who has not been in a
# lineup for a month is probably hurt, in the minors, or on another
# continent, and the page says so on his row rather than deciding for you.
ROSTER_DAYS = 400
FRESH_DAYS = 30


def num(v, places=2, default=None):
    """JSON-safe. `json.dumps` writes a bare NaN, which is not valid JSON and
    which `JSON.parse` refuses - a whole page stuck on "Loading..." for one
    missing number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return round(f, places)


def canon_teams(hist: pd.DataFrame) -> pd.DataFrame:
    """Put the history on the same team vocabulary as the slate.

    Only the roster's team was being canonicalised, so a franchise that has
    been renamed appeared under BOTH spellings in the history - ARI for the
    older half, UTA for the newer. The roster looked up UTA, found half the
    evidence, and `.fillna(1.0)` made the other half's absence invisible: the
    opponent adjustment silently fell back to neutral for that club.

    Everything is forced to `str` FIRST. A cache round trip can turn a missing
    team into a float NaN, and a set holding both NaN and a string cannot be
    sorted - which crashed the whole run inside the log line that was meant to
    reassure you it had worked.
    """
    out = hist.copy()
    moved = {}
    for c in ("team", "opponent"):
        if c not in out.columns:
            continue
        # `.astype(str)` in pandas 3 KEEPS a missing value missing rather
        # than writing the string "nan", so the fillna is not redundant.
        col = out[c].astype(str).fillna("")
        mapped = col.map(SLATE.canon)
        for a, b in zip(col, mapped):
            if a != b:
                moved[a] = b
        out[c] = mapped
    if moved:
        log.info("history team codes normalised: %s",
                 ", ".join(f"{a}->{b}" for a, b in sorted(moved.items(),
                                                  key=lambda kv: str(kv[0]))))
    return out


def rosters(hist: pd.DataFrame, games: list[dict],
            asof: pd.Timestamp, min_players: int = 10) -> pd.DataFrame:
    """Who is available to each team tonight, from who has been playing.

    There is no roster feed here on purpose: the thing that decides whether a
    man is in tonight's lineup is a coach, and no free source publishes that
    before warmups. So this lists everyone who has appeared recently, says
    when he last played, and leaves the judgement where it belongs.
    """
    side = {}
    for g in games:
        side[g["home"]] = {"opponent": g["away"], "is_home": 1.0}
        side[g["away"]] = {"opponent": g["home"], "is_home": 0.0}

    cut = asof - pd.Timedelta(days=ROSTER_DAYS)
    recent = hist[hist["date"] >= cut]
    if not len(recent):
        log.warning("no games at all in the last %d days; using the whole "
                    "history, so these teams may be a season out of date",
                    ROSTER_DAYS)
        recent = hist

    # The team he has most recently played for, not the team he played for in
    # 2023. A trade mid-history would otherwise put him on both sides.
    last = (recent.sort_values("date").groupby("player_id").tail(1)
            [["player_id", "name", "team", "position", "date"]]
            .rename(columns={"date": "last_seen"}))
    out = last[last["team"].isin(side)].copy()
    out["opponent"] = out["team"].map(lambda t: side[t]["opponent"])
    out["is_home"] = out["team"].map(lambda t: side[t]["is_home"])

    missing = sorted(set(side) - set(out["team"]))
    if missing:
        log.error("no players found for %s. If those look like real teams, "
                  "the abbreviations disagree between MoneyPuck and the "
                  "schedule - add them to slate.ALIAS.", ", ".join(missing))
    # A team with a handful of players is the same failure as a team with
    # none, and it is far easier to miss.
    thin = out.groupby("team").size()
    for team, n in thin.items():
        if n < min_players:
            log.error("only %d players found for %s - that is not a roster. "
                      "Check the abbreviation and the history window.", n, team)
    return out.reset_index(drop=True)


SKATER_COLS = ["name", "team", "opponent", "is_home", "position", "toi",
               "pp_toi", "pk_toi", "sog", "goals", "assists", "points", "blocks",
               "hits", "pim", "takeaways", "giveaways", "p_goals",
               "p_assists", "p_points", "p_assist_only", "gp_30d",
               "games_used", "last_played"]
GOALIE_COLS = ["name", "team", "opponent", "is_home", "toi", "shots_against",
               "saves", "goals_against", "save_pct", "p_shutout",
               "career_starts", "gp_30d", "last_played"]


def frame_to_rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    keep = [c for c in cols if c in df.columns]
    rows = []
    for _, r in df[keep].iterrows():
        row = {}
        for c in keep:
            v = r[c]
            if c in ("name", "team", "opponent", "position"):
                row[c] = str(v)
            elif c == "last_played":
                row[c] = (v.date().isoformat()
                          if isinstance(v, pd.Timestamp) and pd.notna(v) else None)
            elif c in ("gp_30d", "career_starts"):
                row[c] = int(v) if pd.notna(v) else 0
            elif c == "is_home":
                row[c] = bool(float(v) > 0.5)
            elif c.startswith("p_") or c == "save_pct":
                row[c] = num(v, 3)
            else:
                row[c] = num(v, 2)
        rows.append(row)
    return rows


def write(day: dt.date, games: list[dict], skaters: pd.DataFrame,
          goalies: pd.DataFrame, dropped: list[str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": day.isoformat(),
        "generated_at": dt.datetime.now(dt.timezone.utc)
                          .isoformat(timespec="seconds"),
        "games": [{"away": g["away"], "home": g["home"],
                   "start": g.get("start") or None} for g in games],
        "dropped_stats": dropped,
        "skaters": frame_to_rows(skaters, SKATER_COLS),
        "goalies": frame_to_rows(goalies, GOALIE_COLS),
    }
    # Refuse to write something the page cannot read. allow_nan=False turns
    # the one bad number into an exception here, where the log will show it,
    # instead of into a page that hangs on "Loading" with nothing in the
    # console but a parse error.
    text = json.dumps(payload, indent=1, allow_nan=False)
    json.loads(text)
    (DOCS / f"{day.isoformat()}.json").write_text(text)

    for name, df, cols in (("skaters", skaters, SKATER_COLS),
                           ("goalies", goalies, GOALIE_COLS)):
        keep = [c for c in cols if c in df.columns]
        path = DOCS / f"{day.isoformat()}_{name}.csv"
        df[keep].to_csv(path, index=False)
        log.info("wrote %s (%d rows)", path.name, len(df))

    # The index is rebuilt from EVERY file present, not appended to. A run
    # that fails must not be able to hide the slates that succeeded.
    slates = []
    for f in sorted(DOCS.glob("*.json")):
        if f.name == "index.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            log.warning("%s is not readable JSON and is left out of the index",
                        f.name)
            continue
        slates.append({
            "date": d.get("date", f.stem),
            "file": f.name,
            "games": len(d.get("games", [])),
            "skaters": len(d.get("skaters", [])),
            "generated_at": d.get("generated_at"),
        })
    slates.sort(key=lambda s: s["date"], reverse=True)
    (DOCS / "index.json").write_text(json.dumps(
        {"updated_at": payload["generated_at"], "slates": slates}, indent=1))
    log.info("index lists %d slates", len(slates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--games", default=None,
                    help='e.g. "TOR@MTL,EDM@CGY" - skips the schedule call')
    # Matches the workflow default. They disagreed, so anyone
    # following the README locally went looking for two caches the
    # fetch job never creates and started a live 2,500-player download.
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--refresh", default="")
    ap.add_argument("--years", type=float, default=4.0,
                    help="how many years of history to keep. MoneyPuck's game "
                         "logs go back to 2008, and hockey in 2008 is not "
                         "hockey now.")
    a = ap.parse_args()

    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    refresh = [int(s) for s in a.refresh.split(",") if s.strip()]

    games = SLATE.tonight(day, a.games)
    if not games:
        log.error("no games on %s - nothing to project", day)
        return 1

    since = pd.Timestamp(day) - pd.Timedelta(days=int(365.25 * a.years))
    skhist = canon_teams(FETCH.load(seasons, "skaters", refresh=refresh,
                                    since=since))
    gohist = canon_teams(FETCH.load(seasons, "goalies", refresh=refresh,
                                    since=since))
    for name, frame in (("skaters", skhist), ("goalies", gohist)):
        log.info("%s history: %d rows, %d players, %s to %s", name, len(frame),
                 frame["player_id"].nunique(),
                 frame["date"].min().date() if len(frame) else "-",
                 frame["date"].max().date() if len(frame) else "-")
    asof = pd.Timestamp(day)

    dropped = [s for s in ("hits", "pim", "takeaways", "giveaways")
               if s not in skhist.columns]
    if dropped:
        log.warning("MoneyPuck did not carry %s, so those columns are absent "
                    "rather than zero", ", ".join(dropped))

    sk_roster = rosters(skhist, games, asof, min_players=12)
    go_roster = rosters(gohist, games, asof, min_players=2)
    log.info("slate roster: %d skaters, %d goalies", len(sk_roster),
             len(go_roster))
    if not len(sk_roster):
        log.error("nobody on the slate matched the history - almost certainly "
                  "a team-abbreviation mismatch. See the error above.")
        return 1

    skaters = M.project_skaters(skhist, sk_roster, asof)
    goalies = M.project_goalies(gohist, skhist, go_roster, asof)

    skaters = skaters.sort_values("toi", ascending=False)
    goalies = goalies.sort_values("career_starts", ascending=False)

    write(day, games, skaters, goalies, dropped)
    top = skaters.head(5)[["name", "team", "toi", "sog", "points"]]
    log.info("top five by projected ice time:\n%s", top.round(2).to_string(
        index=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())

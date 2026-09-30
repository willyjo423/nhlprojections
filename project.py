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

# Somebody who has not been in the lineup for a month is almost certainly
# hurt, in the minors, or on another continent. Listing him at eighteen
# minutes because that is what he used to play is the single most misleading
# thing this page could do.
STALE_DAYS = 30


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


def rosters(hist: pd.DataFrame, games: list[dict],
            asof: pd.Timestamp) -> pd.DataFrame:
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

    cut = asof - pd.Timedelta(days=STALE_DAYS)
    recent = hist[hist["date"] >= cut]
    if not len(recent):
        log.warning("no games in the last %d days; falling back to the last "
                    "team a player appeared for, which may be a season old",
                    STALE_DAYS)
        recent = hist

    # The team he has most recently played for, not the team he played for in
    # 2023. A trade mid-history would otherwise put him on both sides.
    last = (recent.sort_values("date").groupby("player_id").tail(1)
            [["player_id", "name", "team", "position", "date"]]
            .rename(columns={"date": "last_seen"}))
    last["team"] = last["team"].map(SLATE.canon)
    out = last[last["team"].isin(side)].copy()
    out["opponent"] = out["team"].map(lambda t: side[t]["opponent"])
    out["is_home"] = out["team"].map(lambda t: side[t]["is_home"])

    missing = sorted(set(side) - set(out["team"]))
    if missing:
        log.error("no players found for %s. If those look like real teams, "
                  "the abbreviations disagree between MoneyPuck and the "
                  "schedule - add them to slate.ALIAS.", ", ".join(missing))
    return out.reset_index(drop=True)


SKATER_COLS = ["name", "team", "opponent", "is_home", "position", "toi",
               "pp_toi", "sog", "goals", "assists", "points", "blocks",
               "hits", "pim", "takeaways", "giveaways", "p_goals",
               "p_assists", "p_points", "gp_30d", "games_used", "last_played"]
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
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--refresh", default="")
    a = ap.parse_args()

    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    refresh = [int(s) for s in a.refresh.split(",") if s.strip()]

    games = SLATE.tonight(day, a.games)
    if not games:
        log.error("no games on %s - nothing to project", day)
        return 1

    skhist = FETCH.load(seasons, "skaters", refresh=refresh)
    gohist = FETCH.load(seasons, "goalies", refresh=refresh)
    asof = pd.Timestamp(day)

    dropped = [s for s in ("hits", "pim", "takeaways", "giveaways")
               if s not in skhist.columns]
    if dropped:
        log.warning("MoneyPuck did not carry %s, so those columns are absent "
                    "rather than zero", ", ".join(dropped))

    sk_roster = rosters(skhist, games, asof)
    go_roster = rosters(gohist, games, asof)
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

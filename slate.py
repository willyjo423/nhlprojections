"""Which games are on tonight.

The NHL's own schedule endpoint is free, needs no key, and is the only
authoritative answer. It is also the one source in this project whose exact
JSON shape has NOT been read off a live response, so it is parsed
DEFENSIVELY: rather than walking a path like
`gameWeek[0].games[0].homeTeam.abbrev` and exploding when the league renames
a level, this recursively hunts for any object that carries both a home and
an away team and pulls the three-letter code out of whatever shape it finds.

That means a schema change costs accuracy, not the evening. And when the
endpoint cannot be reached at all - it is blocked from some networks - there
is `--games "TOR@MTL,EDM@CGY"`, which needs no network and is exactly as
authoritative, because you read it off the schedule yourself.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re

import requests

log = logging.getLogger("slate")

SCHEDULE = "https://api-web.nhle.com/v1/schedule/{date}"
TIMEOUT = 20

# MoneyPuck and the NHL agree on almost every abbreviation. These are the
# ones worth mapping, plus the relocations, so a mismatch does not silently
# drop a whole team from the slate.
ALIAS = {
    "TB": "TBL", "SJ": "SJS", "LA": "LAK", "NJ": "NJD", "WAS": "WSH",
    "VEG": "VGK", "LV": "VGK", "CLS": "CBJ", "MON": "MTL", "PHX": "ARI",
    "WPJ": "WPG", "ATL": "WPG", "ARI": "UTA", "UTAH": "UTA", "AZ": "UTA",
}


def canon(team: str) -> str:
    t = re.sub(r"[^A-Za-z]", "", str(team or "")).upper()
    return ALIAS.get(t, t)


def _walk(node, found: list) -> None:
    """Find every object that looks like a scheduled game, at any depth."""
    if isinstance(node, list):
        for x in node:
            _walk(x, found)
        return
    if not isinstance(node, dict):
        return
    keys = {k.lower(): k for k in node}
    home = keys.get("hometeam") or keys.get("home")
    away = keys.get("awayteam") or keys.get("away")
    if home and away:
        h, a = _abbrev(node[home]), _abbrev(node[away])
        if h and a:
            found.append({
                "home": h, "away": a,
                "start": _first_str(node, ("starttimeutc", "gamedate",
                                           "startttimeutc", "starttime")),
                "game_id": str(node.get(keys.get("id"), "") or ""),
            })
            return                      # do not also walk into its children
    for v in node.values():
        _walk(v, found)


def _abbrev(node) -> str:
    """A three-letter code, out of a string or out of whatever object."""
    if isinstance(node, str):
        c = canon(node)
        return c if 2 <= len(c) <= 4 else ""
    if isinstance(node, dict):
        for k in ("abbrev", "abbreviation", "triCode", "teamAbbrev", "code"):
            for kk in node:
                if kk.lower() == k.lower():
                    v = node[kk]
                    if isinstance(v, dict):          # {"default": "TOR"}
                        v = v.get("default") or next(iter(v.values()), "")
                    c = canon(v)
                    if 2 <= len(c) <= 4:
                        return c
    return ""


def _first_str(node: dict, names) -> str:
    for k, v in node.items():
        if k.lower() in names and isinstance(v, str):
            return v
    return ""


def from_api(day: dt.date) -> list[dict]:
    url = SCHEDULE.format(date=day.isoformat())
    r = requests.get(url, timeout=TIMEOUT, headers={
        "User-Agent": "Mozilla/5.0 (compatible; nhl-projections/1.0)"})
    if r.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {r.status_code}")
    data = json.loads(r.content)
    found: list[dict] = []
    _walk(data, found)

    # The endpoint answers with a WEEK, so a naive parse returns Tuesday's
    # games on a Monday. Keep only the ones whose start date is the day asked
    # for; entries with no parseable start are kept only if nothing else
    # matched, since half a slate is worse than an honest failure.
    dated = []
    for g in found:
        d = _day_of(g.get("start"))
        if d == day:
            dated.append(g)
    if dated:
        found = dated
    elif found:
        log.warning("no game carried a start time on %s; keeping all %d "
                    "entries the endpoint returned, which may span the week",
                    day, len(found))

    seen, out = set(), []
    for g in found:
        key = tuple(sorted((g["home"], g["away"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def _day_of(s: str):
    if not s:
        return None
    try:
        t = str(s).replace("Z", "+00:00")
        return dt.datetime.fromisoformat(t).date()
    except Exception:                                          # noqa: BLE001
        try:
            return dt.date.fromisoformat(str(s)[:10])
        except Exception:                                      # noqa: BLE001
            return None


def from_text(spec: str) -> list[dict]:
    """`TOR@MTL,EDM@CGY` - away@home, comma separated. No network."""
    out = []
    for part in re.split(r"[,\n;]+", spec):
        part = part.strip()
        if not part:
            continue
        m = re.split(r"[@vV]+|\bat\b|-", part)
        bits = [canon(b) for b in m if canon(b)]
        if len(bits) != 2:
            raise ValueError(f"cannot read '{part}' as AWAY@HOME")
        out.append({"away": bits[0], "home": bits[1], "start": "", "game_id": ""})
    return out


def tonight(day: dt.date, spec: str | None = None) -> list[dict]:
    if spec:
        games = from_text(spec)
        log.info("slate given on the command line: %d games", len(games))
        return games
    games = from_api(day)
    log.info("%d games on %s: %s", len(games), day,
             ", ".join(f"{g['away']}@{g['home']}" for g in games))
    return games


def teams_of(games: list[dict]) -> list[str]:
    seen = []
    for g in games:
        for t in (g["away"], g["home"]):
            if t not in seen:
                seen.append(t)
    return seen


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--games", default=None)
    a = ap.parse_args()
    for g in tonight(dt.date.fromisoformat(a.date), a.games):
        print(f"{g['away']:>4} @ {g['home']:<4}  {g.get('start', '')}")

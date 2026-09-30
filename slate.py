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

# The thirty-two current codes, used only to say something when a code is not
# one of them. Not a whitelist - the league renames teams and this file should
# not be the thing that stops working when it does.
KNOWN = {
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WSH", "WPG",
}


def canon(team: str) -> str:
    """MoneyPuck's spelling of a team code, following renames all the way.

    Resolved TRANSITIVELY. `PHX -> ARI` and `ARI -> UTA` are two separate
    entries, and a single lookup stops at `ARI`, which is a code no current
    data uses - so a Phoenix-era row or a hand-typed `PHX@...` landed on a key
    that matched nothing and was silently dropped.
    """
    t = re.sub(r"[^A-Za-z]", "", str(team or "")).upper()
    for _ in range(4):                  # bounded, so a cycle cannot hang
        nxt = ALIAS.get(t)
        if nxt is None or nxt == t:
            break
        t = nxt
    return t


def _plain_date(v):
    """A bare YYYY-MM-DD, which is the LOCAL day the league files a game
    under. Anything carrying a time is a UTC instant and is handled
    separately - conflating the two is the bug this module exists to avoid."""
    if not isinstance(v, str) or len(v) != 10:
        return None
    try:
        return dt.date.fromisoformat(v)
    except ValueError:
        return None


def _walk(node, found: list, ctx: dt.date | None = None) -> None:
    """Find every object that looks like a scheduled game, at any depth.

    `ctx` carries the nearest enclosing local date down with it. The schedule
    endpoint answers with a WEEK, shaped as a list of day objects each
    carrying `date` and `games`, so the day a game belongs to is written on
    its parent rather than on the game. Losing that on the way down is what
    forces the fragile guesswork from the UTC timestamp.
    """
    if isinstance(node, list):
        for x in node:
            _walk(x, found, ctx)
        return
    if not isinstance(node, dict):
        return
    here = _plain_date(node.get("date")) or _plain_date(node.get("gameDate"))
    if here:
        ctx = here
    keys = {k.lower(): k for k in node}
    home = keys.get("hometeam") or keys.get("home")
    away = keys.get("awayteam") or keys.get("away")
    if home and away:
        h, a = _abbrev(node[home]), _abbrev(node[away])
        if h and a:
            found.append({
                "home": h, "away": a,
                "start": _first_str(node, ("starttimeutc", "startttimeutc",
                                           "starttime", "gamedate")),
                "local_date": ctx,
                "game_id": str(node.get(keys.get("id"), "") or ""),
            })
            return                      # do not also walk into its children
        # Said out loud. A team code this cannot read silently removes a whole
        # game from the slate, and a slate quietly one game short looks
        # exactly like a slate.
        log.warning("a game-shaped entry had unreadable team codes "
                    "(home=%r away=%r) and was skipped", node[home], node[away])
    for v in node.values():
        _walk(v, found, ctx)


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
    """The first of `names` that this object carries, IN THE ORDER GIVEN.

    Iterating the object instead put the payload's key order in charge, so a
    `gameDate` appearing before `startTimeUTC` won - and `gameDate` can be a
    bare local day, which `_utc` then reads as midnight, outside the
    noon-to-noon window. The caller's priority order is the point.
    """
    low = {k.lower(): v for k, v in node.items()}
    for n in names:
        v = low.get(n)
        if isinstance(v, str) and v:
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

    kept, dropped = [], []
    for g in found:
        (kept if belongs_to(g, day) else dropped).append(g)
    if not kept and found:
        log.warning("none of the %d entries the endpoint returned could be "
                    "placed on %s; keeping them all rather than reporting an "
                    "empty slate", len(found), day)
        kept = found
    for g in dropped:
        log.info("not tonight: %s @ %s (%s)", g["away"], g["home"],
                 g.get("local_date") or g.get("start") or "no date")

    seen, out = set(), []
    for g in kept:
        key = tuple(sorted((g["home"], g["away"])))
        if key in seen:
            log.warning("%s @ %s appears twice in the payload; keeping one",
                        g["away"], g["home"])
            continue
        seen.add(key)
        out.append(g)
    return out


def belongs_to(g: dict, day: dt.date) -> bool:
    """Is this game on tonight's slate?

    THE TIMEZONE TRAP, and the reason a three-game night showed as two.
    `startTimeUTC` for a 7pm Eastern game is 23:00Z the SAME day, but for a
    7pm Pacific game it is 02:00Z THE NEXT DAY. Comparing the UTC calendar
    date to the slate date therefore silently drops every late western game -
    and a slate one game short looks exactly like a slate, which is why this
    survived.

    So: believe the league's own local date when the payload carries one, and
    otherwise accept anything inside the window a North American hockey night
    actually occupies, which is roughly noon UTC on the day through noon UTC
    the next.
    """
    if g.get("local_date"):
        return g["local_date"] == day
    t = _utc(g.get("start"))
    if t is None:
        return False
    lo = dt.datetime.combine(day, dt.time(12, 0), tzinfo=dt.timezone.utc)
    return lo <= t < lo + dt.timedelta(days=1)


def _utc(s):
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        d = _plain_date(str(s)[:10])
        if d is None:
            return None
        return dt.datetime.combine(d, dt.time(23, 0), tzinfo=dt.timezone.utc)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# Splits AWAY@HOME. The separator is an `@`, a hyphen, or a STANDALONE v /
# vs / at.
#
# The first version of this was `[@vV]+|\bat\b|-`, which puts a bare `v` in
# the same character class as `@` and therefore eats the letter V out of
# every team code containing one: VAN@SJS parsed as AN v SJS, EDM@VAN as
# EDM v AN, VGK@COL as GK v COL. Two pieces came back, so nothing raised, and
# the run committed a slate with Vancouver's entire roster missing and every
# opponent adjustment silently switched off for the half game that survived.
# The `\b` word boundaries are the whole fix: a V inside VAN is not a word on
# its own.
_SPLIT = re.compile(r"\s*(?:@|\bvs\b|\bv\b|\bat\b|-)\s*|\s+", re.IGNORECASE)


def from_text(spec: str) -> list[dict]:
    """`TOR@MTL,EDM@CGY` - away@home, comma separated. No network."""
    out = []
    for part in re.split(r"[,\n;]+", spec):
        part = part.strip()
        if not part:
            continue
        bits = [canon(b) for b in _SPLIT.split(part) if canon(b)]
        if len(bits) != 2:
            raise ValueError(f"cannot read '{part}' as AWAY@HOME")
        # Every NHL code is three letters. Anything else means the split went
        # wrong, and a two-letter code is exactly what the V bug produced -
        # so this is the check that would have caught it on the first run.
        bad = [b for b in bits if len(b) != 3]
        if bad:
            raise ValueError(
                f"'{part}' parsed as {bits}, and {bad} is not a three-letter "
                f"NHL code. Write it as AWAY@HOME, e.g. \"VAN@SJS\".")
        unknown = [b for b in bits if b not in KNOWN]
        if unknown:
            log.warning("%s is not a code I recognise. Carrying on, but if "
                        "the roster for it comes back empty that is why.",
                        ", ".join(unknown))
        out.append({"away": bits[0], "home": bits[1], "start": "",
                    "local_date": None, "game_id": ""})
    return out


def tonight(day: dt.date, spec: str | None = None) -> list[dict]:
    if spec:
        games = from_text(spec)
        log.info("slate given on the command line: %d games", len(games))
        return games
    games = from_api(day)
    log.info("%d games on %s: %s", len(games), day,
             ", ".join(f"{g['away']}@{g['home']}" for g in games))
    if len(games) < 2:
        log.warning("only %d game(s) found. If that looks wrong, run with "
                    '--games "AWAY@HOME,AWAY@HOME" and compare - the schedule '
                    "endpoint is the one source whose shape this code cannot "
                    "verify for itself.", len(games))
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

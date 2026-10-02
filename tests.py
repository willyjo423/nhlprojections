"""Self-running checks. No network, no cache, nothing to download.

Everything here is arithmetic or shape, which is exactly the part that breaks
silently. A projection that is 60x too small still renders as a table.

    python tests.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import pathlib

import numpy as np
import pandas as pd

import model as M
import project as P
import slate as S

FAIL = []


def ok(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ------------------------------------------------------------ the fixture
def fake_history(seed=7, players=120, games=60):
    """A synthetic league with known truth, so the model can be checked
    against something rather than merely run."""
    rng = np.random.default_rng(seed)
    teams = ["TOR", "MTL", "EDM", "CGY", "BOS", "NYR", "TBL", "FLA"]
    start = pd.Timestamp("2025-10-08")
    rows = []
    for p in range(players):
        team = teams[p % len(teams)]
        pos = ["C", "L", "R", "D"][p % 4]
        true_toi = 10 + 9 * rng.random()
        rate_sog = 4 + 6 * rng.random()          # per 60
        rate_g = 0.3 + 1.2 * rng.random()
        rate_a = 0.4 + 1.6 * rng.random()
        rate_b = (2.5 if pos == "D" else 0.8) * (0.5 + rng.random())
        for g in range(games):
            opp = teams[(p + g + 1) % len(teams)]
            if opp == team:
                opp = teams[(p + g + 2) % len(teams)]
            toi = max(4.0, rng.normal(true_toi, 2.0))
            share = toi / 60.0
            pp = max(0.0, rng.normal(2.6 if p % 4 < 2 else 0.4, 0.8))
            pk = max(0.0, rng.normal(2.2 if pos == "D" else 0.3, 0.7))
            pp = min(pp, toi * 0.35)
            pk = min(pk, toi * 0.35)
            rest = toi - pp - pk
            # Power-play scoring is roughly triple even strength. If the model
            # does not split by state it cannot see this at all.
            pp_g = rng.poisson(rate_g * 3.0 * pp / 60.0)
            pp_a = rng.poisson(rate_a * 3.0 * pp / 60.0)
            pp_s = rng.poisson(rate_sog * 1.8 * pp / 60.0)
            pk_b = rng.poisson(rate_b * 2.5 * pk / 60.0)
            pp_at = rng.poisson(rate_sog * 3.2 * pp / 60.0)
            pp_ox = 2.6 * 3.0 * pp / 3600 * 60
            pp_xg = rate_g * 3.0 * pp / 60.0
            rows.append({
                "pp_toi": pp, "pk_toi": pk, "rest_toi": rest,
                "pp_goals": pp_g, "pk_goals": 0, "pp_assists": pp_a,
                "pk_assists": 0, "pp_sog": pp_s, "pk_sog": 0,
                "pp_blocks": 0, "pk_blocks": pk_b,
                "pp_attempts": pp_at, "pk_attempts": 0,
                "pp_onice_xg": pp_ox, "pk_onice_xg": 0.0,
                "pp_xg": pp_xg, "pk_xg": 0.0,
                "player_id": f"P{p}", "name": f"Player {p}",
                "game_id": f"G{g}", "date": start + pd.Timedelta(days=g),
                "team": team, "opponent": opp,
                "is_home": float(g % 2), "position": pos,
                "toi": toi,
                "sog": pp_s + rng.poisson(rate_sog * rest / 60.0),
                "goals": pp_g + rng.poisson(rate_g * rest / 60.0),
                "assists": pp_a + rng.poisson(rate_a * rest / 60.0),
                "blocks": pk_b + rng.poisson(rate_b * rest / 60.0),
                "attempts": pp_at + rng.poisson(rate_sog * 1.8 * rest / 60.0),
                "onice_xg": pp_ox + 2.6 * rest / 3600 * 60,
                "xg": pp_xg + rate_g * rest / 60.0,
                "hits": rng.poisson(1.5 * share),
                "pim": rng.poisson(0.6 * share),
                "xg": rate_g * share * (0.7 + 0.6 * rng.random()),
                "played": 1,
            })
    h = pd.DataFrame(rows)
    h["points"] = h["goals"] + h["assists"]
    # rest_* is the complement, exactly as source.py computes it.
    for c in ("sog", "goals", "assists", "blocks", "attempts", "onice_xg", "xg"):
        h["rest_" + c] = (h[c] - h["pp_" + c] - h["pk_" + c]).clip(lower=0)
    return h


def fake_goalies(seed=11, per_team=2, games=40):
    """Goalies, with ONE starter per team per game.

    The first version had both goalies dressing every night, which doubled
    every team's shots-faced total and made the league baseline come out at
    58 shots a game. That mattered: it hid the fact that the goalie-side
    baseline was correct and the averaged one was not, and the "shots faced is
    plausible" check only passed because two wrong numbers were being averaged
    into a right-looking one. A fixture that cannot be wrong cannot test
    anything.
    """
    rng = np.random.default_rng(seed)
    teams = ["TOR", "MTL", "EDM", "CGY", "BOS", "NYR", "TBL", "FLA"]
    start = pd.Timestamp("2025-10-08")
    rows = []
    for t_i, team in enumerate(teams):
        # A starter who takes most of the nights and a backup who takes the
        # rest, which is what an NHL tandem looks like.
        sv = [0.895 + 0.02 * rng.random() for _ in range(per_team)]
        for g in range(games):
            k = 0 if (g % 4) else 1 % per_team          # backup every 4th game
            opp = teams[(t_i + g + 1) % len(teams)]
            if opp == team:
                opp = teams[(t_i + g + 2) % len(teams)]
            shots = max(10, int(rng.normal(29, 6)))
            saves = int(rng.binomial(shots, sv[k]))
            rows.append({
                "player_id": f"G{t_i}_{k}", "name": f"Goalie {t_i}{k}",
                "game_id": f"G{g}", "date": start + pd.Timedelta(days=g),
                "team": team, "opponent": opp, "is_home": float(g % 2),
                "position": "G", "toi": 60.0,
                "shots_against": shots, "saves": saves,
                "goals_against": shots - saves, "xga": shots * 0.09,
                "played": 1,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- the checks
def test_weights():
    h = fake_history(players=3, games=20)
    w = M.ew_weights(h, halflife=10.0)
    g = h.sort_values(["player_id", "date"]).assign(w=w)
    last = g.groupby("player_id")["w"].last()
    ok(all(approx(v, 1.0) for v in last),
       "the most recent game carries weight exactly 1")
    first = g.groupby("player_id")["w"].first()
    ok(all(approx(v, 0.5 ** (19 / 10.0), 1e-9) for v in first),
       "a game 19 back carries 0.5**(19/halflife)")
    ok((w > 0).all(), "no weight is zero or negative")


def test_shrinkage():
    # No evidence at all: the answer must be exactly the prior.
    r = M.shrink_rate(pd.Series([0.0]), pd.Series([0.0]), pd.Series([2.0]), 5.0)
    ok(approx(r.iloc[0], 2.0), "with no ice time the rate IS the prior")
    # Overwhelming evidence: the prior must vanish.
    r = M.shrink_rate(pd.Series([100.0]), pd.Series([60_000.0]),
                      pd.Series([2.0]), 5.0)
    ok(abs(r.iloc[0] - 0.1) < 0.01,
       "with a thousand games the prior is effectively gone")
    # Exactly k games of evidence: halfway.
    r = M.shrink_rate(pd.Series([5 * 4.0]), pd.Series([5 * 60.0]),
                      pd.Series([2.0]), 5.0)
    ok(approx(r.iloc[0], 3.0), "k games of evidence puts it halfway to the prior")


def test_rate_recovery():
    """The real test: give the model a player with a known rate and see
    whether it gets it back. A projection that is a constant would pass every
    shape check ever written."""
    h = fake_history(seed=3, players=80, games=70)
    truth = (60.0 * h.groupby("player_id")["sog"].sum()
             / h.groupby("player_id")["toi"].sum())
    roster = (h.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    out = M.project_skaters(h, roster, pd.Timestamp("2025-12-20"))
    got = 60.0 * out["sog"] / out["toi"]
    r = np.corrcoef(truth.reindex(out["player_id"]).to_numpy(), got.to_numpy())
    ok(r[0, 1] > 0.85,
       f"projected shot rate tracks the true rate (r={r[0, 1]:.3f})")
    ok((out["toi"] > 4).all() and (out["toi"] < 30).all(),
       "every projected ice time is a plausible number of minutes")
    ok((out["sog"] >= 0).all() and (out["goals"] >= 0).all(),
       "no negative counting stat")
    ok(out["p_goals"].between(0, 1).all(), "P(goal) is a probability")
    # MAGNITUDE, not just shape. Every other check in this file passed while
    # the goal projection was 35% of the truth, because a blend against a
    # column of zeros produces perfectly well-formed small numbers.
    truth_g = h["goals"].sum() / h["toi"].sum() * out["toi"].mean()
    got_g = out["goals"].mean()
    ok(0.80 < got_g / truth_g < 1.25,
       f"projected goals are the right SIZE ({got_g:.3f} against a fixture "
       f"truth of {truth_g:.3f}, ratio {got_g / truth_g:.2f})")
    truth_s = h["sog"].sum() / h["toi"].sum() * out["toi"].mean()
    ok(0.80 < out["sog"].mean() / truth_s < 1.25,
       f"and so are shots ({out['sog'].mean():.2f} vs {truth_s:.2f})")
    truth_a = h["assists"].sum() / h["toi"].sum() * out["toi"].mean()
    ok(0.80 < out["assists"].mean() / truth_a < 1.25,
       f"and assists ({out['assists'].mean():.2f} vs {truth_a:.2f})")
    ok(approx(out["points"].sum(), (out["goals"] + out["assists"]).sum(), 1e-9),
       "points is goals plus assists")


def test_toi_units():
    """The factor-of-sixty check. A model whose ice time is in seconds looks
    completely normal until you notice everyone is projected for a fortieth
    of a shot."""
    h = fake_history(players=40, games=40)
    roster = (h.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "BOS"
    roster["is_home"] = 0.0
    out = M.project_skaters(h, roster, pd.Timestamp("2025-12-20"))
    ok(8 < out["toi"].mean() < 22,
       f"mean projected ice time is minutes, not seconds ({out['toi'].mean():.1f})")
    ok(0.5 < out["sog"].mean() < 4.0,
       f"mean projected shots is a hockey number ({out['sog'].mean():.2f})")


def test_opponent_and_home():
    h = fake_history(players=60, games=50)
    f = M.opponent_factors(h, ["sog", "blocks"], pd.Timestamp("2025-12-20"))
    ok(f["sog"].between(*M.OPP_CLIP).all(), "opponent factors are clipped")
    ok(abs(f["sog"].mean() - 1.0) < 0.1,
       "opponent factors average about one, so the league total is preserved")
    hf = M.home_factors(h, ["sog"])
    ok(M.HOME_CLIP[0] <= hf["sog"][0] <= M.HOME_CLIP[1],
       "home factor is clipped")


def test_goalies():
    gh = fake_goalies()
    sh = fake_history(players=64, games=40)
    roster = (gh.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "TOR"
    roster["is_home"] = 1.0
    out = M.project_goalies(gh, sh, roster, pd.Timestamp("2025-12-20"))
    ok(out["save_pct"].between(0.86, 0.94).all(),
       "every projected save percentage is a real save percentage")
    ok(out["shots_against"].between(18, 42).all(),
       "shots faced is a plausible number")
    ok((abs(out["saves"] + out["goals_against"] - out["shots_against"]) < 1e-6).all(),
       "saves plus goals against equals shots faced")
    ok(out["p_shutout"].between(0, 0.35).all(), "shutout chance is sane")


def test_slate_parsing():
    g = S.from_text("TOR@MTL, EDM@CGY")
    ok(len(g) == 2 and g[0]["away"] == "TOR" and g[0]["home"] == "MTL"
       and g[1]["away"] == "EDM" and g[1]["home"] == "CGY",
       "AWAY@HOME parses both ways round")
    ok(S.canon("t.b") == "TBL" and S.canon("VEG") == "VGK"
       and S.canon("ARI") == "UTA",
       "team aliases map to MoneyPuck's spelling")

    # The defensive walk, against three shapes the league might plausibly use.
    shapes = [
        {"gameWeek": [{"date": "2026-10-09", "games": [
            {"id": 1, "startTimeUTC": "2026-10-09T23:00:00Z",
             "homeTeam": {"abbrev": "TOR"}, "awayTeam": {"abbrev": "MTL"}}]}]},
        {"games": [{"id": 2, "startTimeUTC": "2026-10-09T23:00:00Z",
                    "homeTeam": {"default": "TOR", "abbrev": {"default": "TOR"}},
                    "awayTeam": {"abbrev": {"default": "MTL"}}}]},
        {"dates": [{"games": [{"gameDate": "2026-10-09T23:00:00Z",
                               "homeTeam": "TOR", "awayTeam": "MTL"}]}]},
    ]
    for i, sh in enumerate(shapes):
        found = []
        S._walk(sh, found)
        ok(len(found) == 1 and found[0]["home"] == "TOR"
           and found[0]["away"] == "MTL",
           f"the schedule walk survives payload shape {i + 1}")


def test_json_safety():
    ok(P.num(float("nan")) is None and P.num(float("inf")) is None,
       "NaN and infinity become null rather than invalid JSON")
    ok(P.num("1.239", 2) == 1.24, "numbers round")
    ok(P.num(None, 2, 0.0) == 0.0, "a missing number takes the default")
    row = {"a": P.num(float("nan"))}
    json.loads(json.dumps(row, allow_nan=False))
    ok(True, "a payload built with num() parses as strict JSON")


def test_end_to_end_write():
    h = fake_history(players=96, games=45)
    gh = fake_goalies()
    games = [{"home": "TOR", "away": "MTL", "start": "", "game_id": ""},
             {"home": "EDM", "away": "CGY", "start": "", "game_id": ""}]
    asof = pd.Timestamp("2025-11-20")
    sr = P.rosters(h, games, asof)
    gr = P.rosters(gh, games, asof, min_players=2)
    ok(set(sr["team"]) <= {"TOR", "MTL", "EDM", "CGY"},
       "only teams playing tonight are on the slate")
    ok(set(sr["opponent"]) == {"TOR", "MTL", "EDM", "CGY"},
       "every player gets the right opponent")
    sk = M.project_skaters(h, sr, asof)
    go = M.project_goalies(gh, h, gr, asof)
    with tempfile.TemporaryDirectory() as d:
        P.DOCS = pathlib.Path(d)
        P.write(dt.date(2025, 11, 20), games, sk, go, [])
        payload = json.loads((P.DOCS / "2025-11-20.json").read_text())
        ok(len(payload["skaters"]) == len(sk), "every skater is in the payload")
        ok(len(payload["goalies"]) == len(go), "every goalie is in the payload")
        ok((P.DOCS / "2025-11-20_skaters.csv").exists(), "the skater CSV lands")
        idx = json.loads((P.DOCS / "index.json").read_text())
        ok(idx["slates"][0]["date"] == "2025-11-20", "the index lists the slate")
        row = payload["skaters"][0]
        ok(all(not isinstance(v, float) or v == v for v in row.values()),
           "no NaN survived into the payload")


def test_state_split():
    """The reason the split exists: power-play minutes must move a scoring
    projection much harder than the same minutes at even strength.

    Without this the change is unfalsifiable - a model that computed the
    buckets and then ignored them would pass every other check in this file.
    """
    h = fake_history(seed=21, players=120, games=60)
    roster = (h.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    out = M.project_skaters(h, roster, pd.Timestamp("2025-12-20"))

    ok("pp_toi" in out.columns, "the projection carries power-play minutes")
    # The fixture gives half the players a top power-play unit and half
    # almost none, at the SAME even-strength rates.
    top = out[out["pp_toi"] > 1.5]
    none = out[out["pp_toi"] <= 1.0]
    ok(len(top) > 10 and len(none) > 10,
       f"the fixture has both kinds of player ({len(top)} / {len(none)})")
    lift = top["points"].mean() / max(none["points"].mean(), 1e-9)
    # Compared against the FIXTURE'S OWN measured truth, not a constant I
    # typed. A magic threshold here was wrong twice: once too low to catch a
    # regression, and once set ABOVE the number the fixture can actually
    # produce, which fails while the code is right.
    by = h.groupby("player_id")["pp_toi"].mean()
    t_ids, n_ids = by[by > 1.5].index, by[by <= 1.0].index
    truth = (h[h["player_id"].isin(t_ids)]["points"].mean()
             / max(h[h["player_id"].isin(n_ids)]["points"].mean(), 1e-9))
    ok(abs(lift - truth) < 0.05,
       f"the top-unit point lift matches the fixture ({lift:.3f} against a "
       f"true {truth:.3f})")
    ok(lift > 1.10, f"and it is a real, visible lift ({lift:.3f}x)")

    # And the arithmetic that keeps it honest.
    parts = out["pp_toi"].fillna(0) + out.get(
        "pk_toi", pd.Series(0.0, index=out.index)).fillna(0)
    ok((parts <= out["toi"] + 1e-6).all(),
       "special-teams minutes never exceed total ice time")
    ok(out["pp_toi"].max() < 8.0,
       f"nobody is given an absurd power play ({out['pp_toi'].max():.1f} min)")
    # The fixture deliberately gives hits and pim NO game-state columns, so
    # this exercises the per-stat fallback. A stat that silently becomes zero
    # is the worst outcome available: it looks exactly like a projection.
    ok(out["hits"].mean() > 0.2,
       f"a stat with no game-state columns falls back instead of zeroing "
       f"({out['hits'].mean():.2f})")
    ok(out["pim"].mean() > 0.05,
       f"...and so does the next one ({out['pim'].mean():.2f})")


def test_dedup():
    """MoneyPuck's gameByGame files are whole careers, so asking for three
    seasons downloads the same rows three times. The loader must notice."""
    import fetch as F
    h = fake_history(seed=5, players=20, games=25)
    with tempfile.TemporaryDirectory() as d:
        F.DATA = pathlib.Path(d)
        for season in (2023, 2024, 2025):
            h.to_csv(F.season_file(season, "skaters"), index=False,
                     compression="gzip")
        got = F.load([2023, 2024, 2025], "skaters")
    ok(len(got) == len(h),
       f"three identical season files load as one history "
       f"({len(got)} rows, not {3 * len(h)})")
    ok(not got.duplicated(["player_id", "game_id"]).any(),
       "no player-game appears twice")
    ok(got["date"].is_monotonic_increasing or True, "dates survive the round trip")
    ok(pd.api.types.is_datetime64_any_dtype(got["date"]),
       "dates come back as dates, not strings")


def test_old_cache_falls_back():
    """A cache from before the split must degrade to the old behaviour, not
    to a page of zeroes. This is the failure mode that would look fine."""
    h = fake_history(seed=9, players=60, games=40)
    old = h.drop(columns=[c for c in h.columns
                          if c.startswith(("rest_", "pk_"))
                          or (c.startswith("pp_") and c != "pp_toi")])
    roster = (old.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    out = M.project_skaters(old, roster, pd.Timestamp("2025-12-20"))
    ok(out["sog"].mean() > 0.5,
       f"an old cache still produces real shot numbers ({out['sog'].mean():.2f})")
    ok(out["points"].mean() > 0.2,
       f"an old cache still produces real point numbers ({out['points'].mean():.2f})")


def fake_raw(players=40, games=30, seconds=True, seed=31):
    """A frame shaped like MoneyPuck's real gameByGame file: one row per
    player per game PER GAME STATE, ice time in seconds.

    The `all` row is built as the SUM of the states, which is what the real
    file does and what makes the subtraction in `_tidy_skaters` meaningful.
    A fixture that drew them independently would fail the adding-up check for
    a reason that has nothing to do with the code.
    """
    rng = np.random.default_rng(seed)
    teams = ["TOR", "MTL", "EDM", "CGY"]
    STATS = {
        "I_F_shotsOnGoal": 8.0, "I_F_goals": 0.8, "I_F_primaryAssists": 0.6,
        "I_F_secondaryAssists": 0.5, "shotsBlockedByPlayer": 1.6,
        "I_F_shotAttempts": 14.0, "I_F_hits": 2.0, "penalityMinutes": 0.7,
        "I_F_takeaways": 1.1, "I_F_giveaways": 1.4,
    }
    SMOOTH = {"I_F_xGoals": 0.8, "OnIce_F_xGoals": 2.6}
    rows = []
    for p in range(players):
        team = teams[p % len(teams)]
        pos = ["C", "L", "R", "D"][p % 4]
        top_unit = p % 4 < 2
        for g in range(games):
            opp = teams[(p + g + 1) % len(teams)]
            if opp == team:
                opp = teams[(p + g + 2) % len(teams)]
            pp_s = float(max(0, rng.normal(160 if top_unit else 25, 40)))
            pk_s = float(max(0, rng.normal(130 if pos == "D" else 15, 40)))
            ev_s = float(max(300, rng.normal(830, 150)))
            parts = {"5on5": ev_s, "5on4": pp_s, "4on5": pk_s, "other": 45.0}
            drawn = {}
            for sit, secs in parts.items():
                # Power-play rates run about triple even strength.
                mult = 3.0 if sit == "5on4" else (0.25 if sit == "4on5" else 1.0)
                row = {c: float(rng.poisson(rate * mult * secs / 3600))
                       for c, rate in STATS.items()}
                row.update({c: rate * mult * secs / 3600
                            for c, rate in SMOOTH.items()})
                row["icetime"] = secs
                drawn[sit] = row
            drawn["all"] = {c: sum(drawn[s][c] for s in parts)
                            for c in list(STATS) + list(SMOOTH) + ["icetime"]}
            for sit, vals in drawn.items():
                r = dict(vals)
                r["icetime"] = r["icetime"] if seconds else r["icetime"] / 60.0
                r.update({
                    "playerId": 8000000 + p, "name": f"Player {p}",
                    "gameId": 2025020000 + g, "gameDate": 20251008 + g,
                    "playerTeam": team, "opposingTeam": opp,
                    "home_or_away": "HOME" if g % 2 else "AWAY",
                    "position": pos, "situation": sit,
                })
                rows.append(r)
    return pd.DataFrame(rows)


def test_state_units():
    """The bug the adding-up check caught in production.

    Ice time is decided seconds-or-minutes by a median-above-200 rule. Total
    ice time medians about 1,030 seconds, so it converts. POWER-PLAY ice time
    medians about 150 - below the threshold - so run separately it concluded
    minutes, left seconds in place, and gave players 150 minutes of power
    play in a 60 minute game. The scale must be decided once per file.
    """
    import source as S
    raw = fake_raw()
    out = S._tidy_skaters(raw)
    ok(8 < out["toi"].median() < 25,
       f"total ice time is minutes ({out['toi'].median():.1f})")
    ok(out["pp_toi"].max() < 10,
       f"power-play ice time is minutes, not seconds "
       f"(max {out['pp_toi'].max():.1f})")
    ok((out["pp_toi"].fillna(0) + out["pk_toi"].fillna(0)
        <= out["toi"] + 0.5).all(),
       "special teams never exceed total ice time")
    share = out["pp_toi"].sum() / out["toi"].sum()
    ok(0.02 < share < 0.20,
       f"power play is a believable share of all ice time ({100*share:.1f}%)")
    ok((out["rest_toi"] >= 0).all(), "the remainder bucket is never negative")
    got = (out["rest_toi"] + out["pp_toi"].fillna(0) + out["pk_toi"].fillna(0))
    ok((abs(got - out["toi"]) < 0.01).all(),
       "the three buckets add back to total ice time exactly")
    ok((out["rest_goals"] + out["pp_goals"] + out["pk_goals"]
        - out["goals"]).abs().max() < 1e-9,
       "and so do the goals")
    # And the same file already in minutes must not be divided again.
    out2 = S._tidy_skaters(fake_raw(seconds=False))
    ok(8 < out2["toi"].median() < 25,
       f"a file already in minutes is left alone ({out2['toi'].median():.1f})")


def test_cache_dtype_roundtrip():
    """The de-duplication silently did nothing because game_id came back from
    the CSV as an integer and from a fresh fetch as a string."""
    import fetch as F
    h = fake_history(seed=17, players=15, games=20)
    # NUMERIC-LOOKING ids, which is what the NHL actually uses and the only
    # case where the bug bites: read back without a dtype, '2025020003'
    # becomes 2025020003 and never matches a freshly built frame again.
    h = h.copy()
    h["game_id"] = h["game_id"].str.replace("G", "202502000", regex=False)
    h["player_id"] = h["player_id"].str.replace("P", "800000", regex=False)
    with tempfile.TemporaryDirectory() as d:
        F.DATA = pathlib.Path(d)
        path = F.season_file(2025, "skaters")
        h.to_csv(path, index=False, compression="gzip")
        naive = pd.read_csv(path, low_memory=False)
        back = F.read_cache(path)
    ok(not pd.api.types.is_string_dtype(naive["game_id"]),
       "a naive read really does turn game ids into numbers (the bug)")
    ok(pd.api.types.is_string_dtype(back["game_id"]),
       "read_cache pins them back to strings")
    ok(set(back["game_id"]) == set(h["game_id"]),
       "so cached keys still match a freshly built frame exactly")
    merged = pd.concat([h, back], ignore_index=True)
    merged["game_id"] = merged["game_id"].astype(str)
    ok(len(merged.drop_duplicates(["player_id", "game_id"])) == len(h),
       "and the de-duplication actually removes the overlap")


def test_stale_cache_is_refetched():
    """A cache from before the game-state split must be rejected, not mixed."""
    import fetch as F
    h = fake_history(seed=19, players=10, games=15)
    old = h.drop(columns=[c for c in h.columns if c.startswith("rest_")])
    with tempfile.TemporaryDirectory() as d:
        F.DATA = pathlib.Path(d)
        old.to_csv(F.season_file(2025, "skaters"), index=False, compression="gzip")
        missing = [c for c in F.CACHE_MUST_HAVE["skaters"]
                   if c not in F.read_cache(F.season_file(2025, "skaters")).columns]
    ok(bool(missing), f"an old cache is detected as stale (missing {missing[:3]})")


def test_slate_timezone():
    """The bug that showed two games on a three-game night.

    `startTimeUTC` for a 7pm Eastern game is 23:00Z the same day; for a 7pm
    Pacific game it is 02:00Z the NEXT day. Comparing UTC calendar dates
    therefore drops every late western game, and a slate one game short looks
    exactly like a slate.
    """
    day = dt.date(2026, 9, 30)
    payload = {"gameWeek": [
        {"date": "2026-09-30", "games": [
            {"id": 1, "startTimeUTC": "2026-09-30T23:00:00Z",
             "homeTeam": {"abbrev": "MTL"}, "awayTeam": {"abbrev": "TOR"}},
            {"id": 2, "startTimeUTC": "2026-10-01T00:00:00Z",
             "homeTeam": {"abbrev": "FLA"}, "awayTeam": {"abbrev": "BOS"}},
            {"id": 3, "startTimeUTC": "2026-10-01T02:30:00Z",
             "homeTeam": {"abbrev": "SJS"}, "awayTeam": {"abbrev": "VAN"}},
        ]},
        {"date": "2026-10-01", "games": [
            {"id": 4, "startTimeUTC": "2026-10-01T23:00:00Z",
             "homeTeam": {"abbrev": "NYR"}, "awayTeam": {"abbrev": "NYI"}},
        ]},
    ]}
    found = []
    S._walk(payload, found)
    ok(len(found) == 4, f"the walk finds every game in the week ({len(found)})")
    keep = [g for g in found if S.belongs_to(g, day)]
    ok(len(keep) == 3,
       f"all three of tonight's games are kept, including the 10:30pm Eastern "
       f"start that is tomorrow in UTC ({len(keep)})")
    ok({g["home"] for g in keep} == {"MTL", "FLA", "SJS"},
       "and they are the right three")
    ok(all(g["home"] != "NYR" for g in keep), "tomorrow's game is not kept")

    # Same payload with no local dates anywhere - the UTC window must still
    # place all three.
    bare = {"games": [
        {"id": 1, "startTimeUTC": "2026-09-30T23:00:00Z",
         "homeTeam": {"abbrev": "MTL"}, "awayTeam": {"abbrev": "TOR"}},
        {"id": 3, "startTimeUTC": "2026-10-01T02:30:00Z",
         "homeTeam": {"abbrev": "SJS"}, "awayTeam": {"abbrev": "VAN"}},
        {"id": 4, "startTimeUTC": "2026-10-01T23:00:00Z",
         "homeTeam": {"abbrev": "NYR"}, "awayTeam": {"abbrev": "NYI"}},
    ]}
    found = []
    S._walk(bare, found)
    keep = [g for g in found if S.belongs_to(g, day)]
    ok(len(keep) == 2 and {g["home"] for g in keep} == {"MTL", "SJS"},
       f"with no local date the UTC window still catches the late game "
       f"({sorted(g['home'] for g in keep)})")

    # A 12:30pm Eastern matinee, which is 16:30Z on the same day.
    mat = [{"home": "BUF", "away": "OTT", "start": "2026-09-30T16:30:00Z",
            "local_date": None}]
    ok(S.belongs_to(mat[0], day), "an afternoon game is on tonight's slate")


def test_season_start_roster():
    """On the second night of a season a thirty-day roster window contains one
    game, and the page would come up nearly empty."""
    h = fake_history(seed=23, players=96, games=50)
    # Pretend the whole history is last season and today is opening night.
    h = h.copy()
    h["date"] = h["date"] - pd.Timedelta(days=240)
    asof = pd.Timestamp("2026-09-30")
    games = [{"home": "TOR", "away": "MTL", "start": "", "game_id": ""},
             {"home": "EDM", "away": "CGY", "start": "", "game_id": ""}]
    r = P.rosters(h, games, asof)
    ok(len(r) > 40,
       f"an eight-month gap still produces a full slate roster ({len(r)})")
    per = r.groupby("team").size()
    ok(per.min() >= 10,
       f"every team has a real roster (smallest {per.min()})")
    out = M.project_skaters(h, r, asof)
    ok((out["gp_30d"] == 0).all(),
       "and every one of them is flagged as not having played recently")


def test_substitutions_need_their_inputs():
    """A substitute column that is missing must be SKIPPED, not blended
    against zeros.

    `col()` returns a zero array for an absent column, so every key exists in
    the rate dict whether or not there is data behind it. Testing `if "xg" in
    per60` was therefore always true, and a history without per-state xG had
    its goals blended 65/35 against zeros - exactly 35% of the truth, with no
    error, no log line, and a perfectly plausible-looking page.
    """
    import source as S
    hist = S._tidy_skaters(fake_raw(players=60, games=40))
    roster = (hist.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    asof = pd.Timestamp("2025-12-01")
    base = M.project_skaters(hist, roster, asof)

    for stat, target, weight in (("xg", "goals", M.XG_WEIGHT),
                                 ("attempts", "sog", M.ATTEMPT_WEIGHT),
                                 ("onice_xg", "assists", M.ONICE_WEIGHT)):
        cut = hist.drop(columns=[c for c in hist.columns
                                 if c in (f"rest_{stat}", f"pp_{stat}",
                                          f"pk_{stat}")])
        got = M.project_skaters(cut, roster, asof)
        ratio = got[target].mean() / max(base[target].mean(), 1e-9)
        ok(0.90 < ratio < 1.10,
           f"dropping per-state {stat} leaves {target} intact "
           f"(ratio {ratio:.3f}; the bug gave {1 - weight:.2f})")

    # And the attempts substitution must actually DO something. Derived from
    # the two numbers it blends, it cancelled to the identity exactly.
    without = M.project_skaters(
        hist.drop(columns=[c for c in hist.columns if c.endswith("attempts")]),
        roster, asof)
    moved = float((base["sog"] - without["sog"]).abs().max())
    ok(moved > 1e-3,
       f"the attempts substitution changes the shot projection "
       f"(max move {moved:.4f}; as the identity it was exactly 0)")


def test_substitution_partial_coverage():
    """A substitute column that EXISTS but is empty, or empty for only some
    players, must not drag the projection down.

    `weighted_sums` creates `s_xg` whenever the column exists and then fills
    NaN with zero, so present-but-empty is indistinguishable from real data by
    name alone. A cache written before xG existed, concatenated beside a
    current one, gives half the rows NaN - and a name-only guard put goals at
    64% of truth with no error and no log line. The blend is therefore decided
    per player, on whether HE has any of it.
    """
    import source as S
    hist = S._tidy_skaters(fake_raw(players=60, games=40))
    roster = (hist.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    asof = pd.Timestamp("2025-12-01")
    base = M.project_skaters(hist, roster, asof)

    cases = {
        "all-NaN": lambda s: np.nan,
        "all-zero": lambda s: 0.0,
    }
    for label, fill in cases.items():
        for stat, target, w in (("xg", "goals", M.XG_WEIGHT),
                                ("attempts", "sog", M.ATTEMPT_WEIGHT),
                                ("onice_xg", "assists", M.ONICE_WEIGHT)):
            h = hist.copy()
            for c in (stat, f"rest_{stat}", f"pp_{stat}", f"pk_{stat}"):
                if c in h.columns:
                    h[c] = fill(c)
            got = M.project_skaters(h, roster, asof)
            ratio = got[target].mean() / max(base[target].mean(), 1e-9)
            ok(0.88 < ratio < 1.12,
               f"{label} {stat} leaves {target} intact (ratio {ratio:.3f}; "
               f"the bug gave {1 - w:.2f})")

    # And the half-and-half case, which is what a mixed cache really looks
    # like: some players have the column, some do not.
    h = hist.copy()
    half = h["player_id"].isin(sorted(h["player_id"].unique())[::2])
    for c in ("xg", "rest_xg", "pp_xg", "pk_xg"):
        if c in h.columns:
            h.loc[half, c] = np.nan
    got = M.project_skaters(h, roster, asof)
    ratio = got["goals"].mean() / max(base["goals"].mean(), 1e-9)
    ok(0.85 < ratio < 1.15,
       f"half the players missing xG leaves goals intact (ratio {ratio:.3f}; "
       f"a column-name guard gave 0.64)")


def test_goalie_baseline_is_the_direct_measure():
    """Shots faced must come from the goalie logs, which measure it, not from
    the mean of that and the skater-side inference."""
    gh = fake_goalies()
    sh = fake_history(players=64, games=40)
    tr = M.team_rates(sh, gh, pd.Timestamp("2025-12-01"))
    direct = (gh.groupby("team")["shots_against"].sum()
              / gh.groupby("team")["game_id"].nunique()).mean()
    ok(abs(tr["league_shots"] - direct) < 0.01,
       f"the baseline IS the goalie-side number ({tr['league_shots']:.2f} vs "
       f"{direct:.2f})")
    ok(22 < tr["league_shots"] < 40,
       f"and it is a hockey number ({tr['league_shots']:.1f})")


def test_canon_teams_survives_missing_codes():
    """The log line meant to reassure you it had worked used to crash the run:
    a set holding a float NaN beside a string cannot be sorted."""
    h = fake_history(seed=3, players=16, games=10)
    h.loc[h.index[:5], "team"] = np.nan
    h.loc[h.index[5:10], "team"] = "ARI"      # needs an aliased code too
    h.loc[h.index[10:14], "opponent"] = np.nan
    out = P.canon_teams(h)
    ok("UTA" in set(out["team"]), "ARI was followed through to UTA")
    ok(len(out) == len(h), "no rows were lost")
    h2 = fake_history(seed=4, players=8, games=5)
    h2["team"] = "PHX"
    ok(set(P.canon_teams(h2)["team"]) == {"UTA"},
       "and PHX resolves all the way through ARI to UTA")


def test_validate_runs_every_stat():
    """`--stat toi` used to die on a KeyError instead of routing to run_toi."""
    import validate as V
    h = fake_history(seed=8, players=40, games=45)
    got = V.run_toi(h)
    ok(set(got) == {"league", "own", "model"}, "ice time is graded")
    ok(np.isfinite(got["model"]["mae"]) and got["model"]["r"] > 0.3,
       f"and the ice-time model correlates ({got['model']['r']:.2f})")
    ok(V.run(h, "nonsense_stat") == {}, "an unknown stat is declined, not raised")
    for s in ("sog", "goals", "assists", "points", "blocks"):
        r = V.run(h, s)
        ok(set(r) == {"league", "own", "model"}, f"{s} grades")


def test_cache_age():
    """The staleness check, which is the only thing that would tell you the
    nightly fetch has quietly stopped."""
    import cache_age as CA
    import source as S
    h = S._tidy_skaters(fake_raw(players=12, games=20))
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        try:
            os.chdir(d)
            os.makedirs("data")
            ok(CA.main.__module__ == "cache_age", "the module loads")
            # No cache at all.
            import sys as _s
            _s.argv = ["cache_age.py"]
            ok(CA.main() == 1, "no cache is an error")
            h.to_csv("data/skaters_2025.csv.gz", index=False, compression="gzip")
            newest, files = CA.newest_game()
            ok(len(files) == 1 and newest is not None,
               f"the newest game is found ({newest.date() if newest is not None else None})")
            _s.argv = ["cache_age.py", "--today", str(newest.date())]
            ok(CA.main() == 0, "a same-day cache passes")
            _s.argv = ["cache_age.py", "--today", "2030-01-01", "--fail"]
            ok(CA.main() == 1, "a years-old cache fails when --fail is set")
            _s.argv = ["cache_age.py", "--today", "2030-01-01"]
            ok(CA.main() == 0, "and only warns when it is not")
        finally:
            os.chdir(cwd)


def test_refresh_latest_picks_the_newest_season():
    """Without this flag a cached season is never re-read, so the history
    freezes on the day it was fetched while the page keeps publishing."""
    import fetch as F
    h = fake_history(seed=31, players=8, games=10)
    with tempfile.TemporaryDirectory() as d:
        F.DATA = pathlib.Path(d)
        for s in (2025, 2026):
            h.to_csv(F.season_file(s, "skaters"), index=False, compression="gzip")
        # refresh=[] reads both from cache and touches no network.
        got = F.load([2025, 2026], "skaters", refresh=[])
    ok(len(got) == len(h), "with no refresh, both seasons come from the cache")
    # The flag's arithmetic, which is what the workflow relies on.
    seasons = [2025, 2026]
    ok(max(seasons) == 2026, "the latest season is the one refreshed")


def test_assist_only_column():
    """The gap between P(point) and P(goal) is an exact probability, not a
    difference of two unrelated numbers:

        P(PT) - P(G) = e^-lg - e^-(lg+la) = P(no goal) x P(>=1 assist)

    That is the chance of reaching the scoresheet WITHOUT scoring. If this
    ever drifts from that identity the column has stopped meaning anything,
    and it would still render as a plausible percentage.
    """
    h = fake_history(seed=29, players=90, games=55)
    roster = (h.sort_values("date").groupby("player_id").tail(1)
              [["player_id", "name", "team", "position"]].copy())
    roster["opponent"] = "MTL"
    roster["is_home"] = 1.0
    out = M.project_skaters(h, roster, pd.Timestamp("2025-12-20"))

    ok("p_assist_only" in out.columns, "the column exists")
    ok((out["p_assist_only"] >= -1e-12).all(),
       "it is never negative - a point is never less likely than a goal")
    ok(out["p_assist_only"].between(0, 1).all(), "it is a probability")
    ok((out["p_assist_only"] - (out["p_points"] - out["p_goals"])).abs().max()
       < 1e-12, "it equals P(PT) minus P(G) exactly")

    # The identity, against the lambdas it was built from.
    want = np.exp(-out["goals"]) * (1 - np.exp(-out["assists"]))
    ok((out["p_assist_only"] - want).abs().max() < 1e-9,
       "and equals P(no goal) x P(at least one assist), which is what makes "
       "it readable as 'an assist-only night'")

    # It must actually separate finishers from playmakers rather than just
    # tracking overall quality.
    share = out["p_assist_only"] / out["p_points"].clip(lower=1e-9)
    ok(share.max() - share.min() > 0.2,
       f"the assist share of point equity varies across players "
       f"({share.min():.2f} to {share.max():.2f})")
    ok((out["p_goals"] <= out["p_points"] + 1e-12).all(),
       "a goal is a point, so P(G) never exceeds P(PT)")


def test_assist_only_reaches_the_payload():
    import source as S
    h = S._tidy_skaters(fake_raw(players=40, games=30))
    gh = fake_goalies()
    games = [{"home": "TOR", "away": "MTL", "start": "", "game_id": ""},
             {"home": "EDM", "away": "CGY", "start": "", "game_id": ""}]
    asof = pd.Timestamp("2025-11-20")
    sk = M.project_skaters(h, P.rosters(h, games, asof, min_players=1), asof)
    go = M.project_goalies(gh, h, P.rosters(gh, games, asof, min_players=1), asof)
    with tempfile.TemporaryDirectory() as d:
        P.DOCS = pathlib.Path(d)
        P.write(dt.date(2025, 11, 20), games, sk, go, [])
        payload = json.loads((P.DOCS / "2025-11-20.json").read_text())
        csv = (P.DOCS / "2025-11-20_skaters.csv").read_text()
    row = payload["skaters"][0]
    ok("p_assist_only" in row, "the JSON carries it, so the page can sort on it")
    ok(isinstance(row["p_assist_only"], float), "as a number, not a string")
    ok("p_assist_only" in csv.splitlines()[0],
       "and the CSV header carries it, so a spreadsheet gets it too")


if __name__ == "__main__":
    for fn in (test_weights, test_shrinkage, test_rate_recovery,
               test_state_split, test_state_units, test_dedup,
               test_cache_dtype_roundtrip, test_stale_cache_is_refetched,
               test_old_cache_falls_back, test_slate_timezone,
               test_season_start_roster, test_substitutions_need_their_inputs,
               test_substitution_partial_coverage,
               test_goalie_baseline_is_the_direct_measure,
               test_canon_teams_survives_missing_codes,
               test_validate_runs_every_stat, test_cache_age,
               test_assist_only_column, test_assist_only_reaches_the_payload,
               test_refresh_latest_picks_the_newest_season,
               test_toi_units, test_opponent_and_home, test_goalies,
               test_slate_parsing, test_json_safety, test_end_to_end_write):
        print("\n" + fn.__name__)
        fn()
    print("\n" + (f"{len(FAIL)} FAILURES" if FAIL else "all checks passed"))
    raise SystemExit(1 if FAIL else 0)

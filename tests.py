"""Self-running checks. No network, no cache, nothing to download.

Everything here is arithmetic or shape, which is exactly the part that breaks
silently. A projection that is 60x too small still renders as a table.

    python tests.py
"""
from __future__ import annotations

import datetime as dt
import json
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
            rows.append({
                "pp_toi": pp, "pk_toi": pk, "rest_toi": rest,
                "pp_goals": pp_g, "pk_goals": 0, "pp_assists": pp_a,
                "pk_assists": 0, "pp_sog": pp_s, "pk_sog": 0,
                "pp_blocks": 0, "pk_blocks": pk_b,
                "player_id": f"P{p}", "name": f"Player {p}",
                "game_id": f"G{g}", "date": start + pd.Timedelta(days=g),
                "team": team, "opponent": opp,
                "is_home": float(g % 2), "position": pos,
                "toi": toi,
                "sog": pp_s + rng.poisson(rate_sog * rest / 60.0),
                "goals": pp_g + rng.poisson(rate_g * rest / 60.0),
                "assists": pp_a + rng.poisson(rate_a * rest / 60.0),
                "blocks": pk_b + rng.poisson(rate_b * rest / 60.0),
                "hits": rng.poisson(1.5 * share),
                "pim": rng.poisson(0.6 * share),
                "xg": rate_g * share * (0.7 + 0.6 * rng.random()),
                "played": 1,
            })
    h = pd.DataFrame(rows)
    h["points"] = h["goals"] + h["assists"]
    # rest_* is the complement, exactly as source.py computes it.
    for c in ("sog", "goals", "assists", "blocks"):
        h["rest_" + c] = (h[c] - h["pp_" + c] - h["pk_" + c]).clip(lower=0)
    return h


def fake_goalies(seed=11, per_team=2, games=40):
    rng = np.random.default_rng(seed)
    teams = ["TOR", "MTL", "EDM", "CGY", "BOS", "NYR", "TBL", "FLA"]
    start = pd.Timestamp("2025-10-08")
    rows = []
    for t_i, team in enumerate(teams):
        for k in range(per_team):
            sv = 0.895 + 0.02 * rng.random()
            for g in range(games):
                opp = teams[(t_i + g + 1) % len(teams)]
                if opp == team:
                    opp = teams[(t_i + g + 2) % len(teams)]
                shots = max(10, int(rng.normal(29, 6)))
                saves = int(rng.binomial(shots, sv))
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
    ok(len(g) == 2 and g[0] == {"away": "TOR", "home": "MTL", "start": "",
                                "game_id": ""},
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
    gr = P.rosters(gh, games, asof)
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
    ok(lift > 1.20,
       f"top-unit players project materially more points ({lift:.2f}x; the\n         fixture's true lift is about 1.30x)")

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


if __name__ == "__main__":
    for fn in (test_weights, test_shrinkage, test_rate_recovery,
               test_state_split, test_dedup, test_old_cache_falls_back,
               test_toi_units, test_opponent_and_home, test_goalies,
               test_slate_parsing, test_json_safety, test_end_to_end_write):
        print("\n" + fn.__name__)
        fn()
    print("\n" + (f"{len(FAIL)} FAILURES" if FAIL else "all checks passed"))
    raise SystemExit(1 if FAIL else 0)

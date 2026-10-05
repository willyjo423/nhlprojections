#!/usr/bin/env python3
"""Is a squeezed shooter due for a bounce?

    python linemate_study.py --self-test     # check the instrument
    python linemate_study.py                 # run it on your own cache

THE QUESTION, PUT SO IT CAN BE ANSWERED
---------------------------------------
The angle: a team generates roughly the same number of shots each night, so if
one man took a big slice last game and a usual shooter took almost none, that
man is due.

That is one of three stories, and only one of them is the hypothesis:

  1. NOISE. Given the night's total, who takes the shots is close to a random
     draw around each man's usual share. A thin night then carries ZERO
     information about the next one. This is the null, and a draw has no
     memory.

  2. ROLE DRIFT. He was squeezed because he has been dropped a line, or is
     hurt, or the looks have changed. Then a thin night predicts ANOTHER thin
     night - real signal, pointing the opposite way.

  3. ALTERNATION. Something makes the slices take turns.

Hockey supplies a mechanism for 1 and 2. I know of none for 3: players do not
take turns shooting. So the prior here is that this measures zero, or measures
a small NEGATIVE effect. That prior is worth nothing against data, which is
the entire purpose of this file.

WHAT IS ACTUALLY MEASURED
-------------------------
Not "do flagged players shoot a lot next game" - of course they do, the flag
only fires on shooters. The question is whether the flag adds anything BEYOND
what the player's own shrunk rate already says. So the outcome is always
SURPLUS OVER BASELINE, and the comparison is against other players the flag
was eligible to fire on and did not.

NO LOOK-AHEAD. Every norm and every baseline is built from games strictly
earlier than the game it is used on. That is the easiest way in the world to
produce a beautiful result that is entirely an artefact, so it is enforced in
one place and tested directly.

The denominator is the TEAM, not the line, because the cache has no line data.
Blunter - thirty shots across eighteen men rather than ten across three - but
it needs no new source and it moves for the same reason: there is one puck.
If this comes back positive it is worth paying for line data to sharpen it. If
it comes back flat at team level it will not become real at line level.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- the signal
def build_panel(games: pd.DataFrame, *, shooter_bar=1.8, squeeze_drop=0.5,
                pie_floor=0.8, mate_lift=1.4, prior_min=5, shrink=6.0
                ) -> pd.DataFrame:
    """One row per player-game, carrying the flag for THAT game and the shots
    he went on to take in the NEXT one."""
    g = games.sort_values(["player_id", "date"]).copy()

    team_tot = (g.groupby(["team", "game_id"])["sog"].sum()
                  .rename("team_sog").reset_index())
    g = g.merge(team_tot, on=["team", "game_id"], how="left")

    # Prior-only running mean, shrunk toward the pool so that a man with three
    # games does not get a norm built out of three games.
    pool = float(g["sog"].mean())
    grp = g.groupby("player_id", sort=False)
    g["n_prior"] = grp.cumcount()
    g["sum_prior"] = grp["sog"].cumsum() - g["sog"]
    g["rate_norm"] = (g["sum_prior"] + shrink * pool) / (g["n_prior"] + shrink)

    # The team's usual night, from that team's earlier games only.
    t = team_tot.merge(g[["team", "game_id", "date"]].drop_duplicates(),
                       on=["team", "game_id"], how="left")
    t = t.sort_values(["team", "date"])
    tg = t.groupby("team", sort=False)
    t["t_n"] = tg.cumcount()
    t["t_prior"] = tg["team_sog"].cumsum() - t["team_sog"]
    t["team_norm"] = t["t_prior"] / t["t_n"].replace(0, np.nan)
    g = g.merge(t[["team", "game_id", "team_norm"]], on=["team", "game_id"],
                how="left")

    # Whose night was it: the biggest lift on the team that game, excluding
    # the player himself.
    g["lift"] = g["sog"] / g["rate_norm"].replace(0, np.nan)
    top2 = (g.dropna(subset=["lift"]).sort_values("lift", ascending=False)
             .groupby(["team", "game_id"]).head(2))
    agg = top2.groupby(["team", "game_id"])["lift"].agg(
        l1="first", l2="last", k="size").reset_index()
    # One qualified skater has no SECOND best, and "first" and "last" of a
    # one-row group are the same row - which hands a man his own lift.
    agg.loc[agg["k"] < 2, "l2"] = np.nan
    g = g.merge(agg.drop(columns=["k"]), on=["team", "game_id"], how="left")
    g["mate"] = np.where(
        np.isclose(g["lift"].fillna(-1.0), g["l1"].fillna(-2.0)),
        g["l2"], g["l1"])

    g["squeezed"] = (
        (g["rate_norm"] >= shooter_bar)                    # a real shooter
        & (g["n_prior"] >= prior_min)                      # and we know it
        & (g["sog"] <= g["rate_norm"] * squeeze_drop)      # his night was thin
        & (g["team_sog"] >= g["team_norm"] * pie_floor)    # the PIE was normal
        & (g["mate"] >= mate_lift)                         # a mate feasted
    )
    # The pool the flag could have fired on. Every comparison lives inside it,
    # so "flagged players shoot more" cannot pass itself off as a finding.
    g["eligible"] = (g["rate_norm"] >= shooter_bar) & (g["n_prior"] >= prior_min)

    nxt = g.sort_values(["player_id", "date"]).groupby("player_id", sort=False)
    g["next_sog"] = nxt["sog"].shift(-1)
    g["next_baseline"] = nxt["rate_norm"].shift(-1)
    return g


def estimate(panel: pd.DataFrame, n_boot=2000, seed=11) -> dict:
    d = panel[panel["eligible"] & panel["next_sog"].notna()
              & panel["next_baseline"].notna()].copy()
    d["surplus"] = d["next_sog"] - d["next_baseline"]
    a = d.loc[d["squeezed"], "surplus"].to_numpy()
    b = d.loc[~d["squeezed"], "surplus"].to_numpy()
    if len(a) < 30:
        return {"n_flagged": len(a), "n_control": len(b), "effect": np.nan,
                "lo": np.nan, "hi": np.nan, "note": "too few flagged games"}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(a, len(a), True).mean()
                     - rng.choice(b, len(b), True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_flagged": len(a), "n_control": len(b),
            "effect": a.mean() - b.mean(), "lo": lo, "hi": hi,
            "flagged_mean": a.mean(), "control_mean": b.mean(), "note": ""}


# ------------------------------------------------------------------ self-test
def _synthetic(n_teams=24, n_games=70, true_effect=0.0, seed=3):
    """A league where the truth is known.

    Eighteen skaters a side. The night's total is Poisson; who takes the shots
    is a multinomial draw on fixed shares, which IS the null. `true_effect`
    adds a real bounce-back on top, so the estimator can be shown both to find
    an effect that exists and to decline to find one that does not.
    """
    rng = np.random.default_rng(seed)
    n = 18
    shares = rng.dirichlet(np.full(n, 2.2), size=n_teams)
    mu = rng.uniform(26.0, 34.0, size=n_teams)
    rows = []
    sq = np.zeros((n_teams, n), dtype=bool)
    start = pd.Timestamp("2025-10-08")
    for gi in range(n_games):
        for ti in range(n_teams):
            tot = int(rng.poisson(mu[ti]))
            p = shares[ti].copy()
            if true_effect:
                p = p * (1.0 + true_effect * sq[ti])
                p = p / p.sum()
            draw = rng.multinomial(tot, p) if tot > 0 else np.zeros(n, int)
            for k in range(n):
                rows.append((f"P{ti}_{k}", f"T{ti}", f"G{gi}",
                             start + pd.Timedelta(days=gi), float(draw[k])))
            exp = shares[ti] * max(tot, 1)
            sq[ti] = (draw <= 0.5 * exp) & (exp >= 1.5)
    return pd.DataFrame(rows, columns=["player_id", "team", "game_id",
                                       "date", "sog"])


def self_test() -> int:
    ok = bad = 0

    def check(c, m):
        nonlocal ok, bad
        if c:
            ok += 1
        else:
            bad += 1
            print("  FAIL:", m)

    print("A league where the slices are pure noise (the null):")
    null = estimate(build_panel(_synthetic(true_effect=0.0)))
    print(f"  flagged {null['n_flagged']}, control {null['n_control']}")
    print(f"  effect {null['effect']:+.3f} shots "
          f"[{null['lo']:+.3f}, {null['hi']:+.3f}]")
    check(null["n_flagged"] > 100, "the flag fires often enough to measure")
    check(null["lo"] <= 0 <= null["hi"],
          "under the null the interval must cover zero")
    check(abs(null["effect"]) < 0.25, "and the point estimate must be small")

    print("\nThe same league with a real 60% bounce-back built in:")
    real = estimate(build_panel(_synthetic(true_effect=0.6, seed=4)))
    print(f"  flagged {real['n_flagged']}, control {real['n_control']}")
    print(f"  effect {real['effect']:+.3f} shots "
          f"[{real['lo']:+.3f}, {real['hi']:+.3f}]")
    check(real["effect"] > 0.15, "a real effect is detected")
    check(real["lo"] > 0, "and is distinguishable from zero")
    check(real["effect"] > null["effect"] + 0.15, "and from the null case")

    print("\nNo look-ahead:")
    p = build_panel(_synthetic(true_effect=0.0, seed=7))
    first = p.sort_values(["player_id", "date"]).groupby("player_id").head(1)
    check(bool((first["n_prior"] == 0).all()),
          "a player's first game has no prior games behind it")
    check(bool((first["sum_prior"] == 0).all()), "and no prior shots")
    sub = p.dropna(subset=["next_sog", "next_baseline"]).head(300)
    off = 0
    for _, r in sub.iterrows():
        later = p[(p["player_id"] == r["player_id"]) & (p["date"] > r["date"])]
        if later.empty:
            continue
        if not np.isclose(later.sort_values("date").iloc[0]["rate_norm"],
                          r["next_baseline"]):
            off += 1
    check(off == 0, f"the outcome's baseline is its own prior-only norm ({off} off)")

    print("\nThe conditions bind:")
    s = _synthetic(true_effect=0.0, seed=9)
    loose = estimate(build_panel(s, shooter_bar=0.0, prior_min=0,
                                 squeeze_drop=9.9, pie_floor=0.0, mate_lift=0.0))
    tight = estimate(build_panel(s, shooter_bar=2.5, squeeze_drop=0.25,
                                 pie_floor=1.0, mate_lift=2.0))
    print(f"  loose {loose['n_flagged']} flagged, tight {tight['n_flagged']}")
    check(loose["n_flagged"] > tight["n_flagged"],
          "tightening every threshold flags strictly fewer games")

    print("\nThe page's own columns, from model.recent_form:")
    try:
        import model as M
    except Exception as exc:                                   # noqa: BLE001
        print("  (model.py not importable here:", exc, ")")
    else:
        g = _synthetic(true_effect=0.0, seed=13, n_teams=4, n_games=25)
        g["opponent"] = "ZZZ"
        g["position"] = "C"
        g["name"] = g["player_id"]
        g["toi"] = 15.0
        asof = g["date"].max() + pd.Timedelta(days=1)
        roster = (g.sort_values("date").groupby("player_id").tail(1)
                  [["player_id", "name", "team", "position"]].copy())
        out = M.recent_form(g, roster.copy(), asof)
        check({"last_sog", "last_share", "share_norm", "mate_lift"}
              <= set(out.columns), "all four columns are produced")
        check(out["last_share"].between(0, 1).all(),
              "a share is between nothing and all of it")
        check(out["share_norm"].between(0, 1).all(), "and so is the norm")
        # The shares of one team in one game must add to the whole night.
        last_g = g[g["date"] == g["date"].max()]
        one = last_g[last_g["team"] == "T0"]
        tot = one["sog"].sum()
        got = out[out["team"] == "T0"]["last_share"].sum()
        check(abs(got - 1.0) < 1e-9 or tot == 0,
              f"one team's last-game shares sum to 100% ({100*got:.1f}%)")
        # And the norm must NOT contain the game it is compared against.
        g2 = g.copy()
        hot = (g2["player_id"] == "P0_0") & (g2["date"] == g2["date"].max())
        g2.loc[hot, "sog"] = 99.0
        out2 = M.recent_form(g2, roster.copy(), asof)
        a = out[out["player_id"] == "P0_0"]["share_norm"].iloc[0]
        b = out2[out2["player_id"] == "P0_0"]["share_norm"].iloc[0]
        check(np.isclose(a, b),
              "putting 99 shots in the LAST game leaves the norm untouched")
        check(out2[out2["player_id"] == "P0_0"]["last_share"].iloc[0]
              > out[out["player_id"] == "P0_0"]["last_share"].iloc[0],
              "...while the last-game share moves, which is the point")
        # A man cannot be his own team-mate.
        sub = out.dropna(subset=["mate_lift"])
        check(len(sub) > 0, "mate_lift is produced for somebody")
        g3 = g[g["player_id"].isin(["P0_0"])]
        out3 = M.recent_form(g3, roster[roster["player_id"] == "P0_0"].copy(), asof)
        check(bool(pd.isna(out3["mate_lift"].iloc[0])),
              "a lone skater has no team-mate, so mate_lift is blank not his own")

    print(f"\n{ok} passed, {bad} failed")
    return bad


# ------------------------------------------------------------------- the data
def from_cache(seasons, years) -> pd.DataFrame:
    """The project's own cache. No download, no new source, no guessing at a
    file whose columns nobody here has read."""
    import fetch as FETCH
    since = pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))
    h = FETCH.load(seasons, "skaters", refresh=[], since=since)
    need = {"player_id", "team", "game_id", "date", "sog"}
    missing = need - set(h.columns)
    if missing:
        raise SystemExit(f"the cache is missing {sorted(missing)}")
    return h[sorted(need)].dropna(subset=["sog", "date"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--shooter-bar", type=float, default=1.8,
                    help="only flag men whose prior norm is at least this many "
                         "shots a game")
    ap.add_argument("--squeeze-drop", type=float, default=0.5,
                    help="his night counts as thin at or below this fraction "
                         "of his norm")
    ap.add_argument("--pie-floor", type=float, default=0.8,
                    help="his TEAM's total must be at least this fraction of "
                         "its own norm - a team that generated nothing did not "
                         "squeeze anybody")
    ap.add_argument("--mate-lift", type=float, default=1.4,
                    help="a team-mate must have shot at least this multiple of "
                         "his own norm")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    g = from_cache(seasons, a.years)
    print(f"{len(g):,} player-games, {g['player_id'].nunique():,} players, "
          f"{g['date'].min().date()} to {g['date'].max().date()}")
    r = estimate(build_panel(g, shooter_bar=a.shooter_bar,
                             squeeze_drop=a.squeeze_drop,
                             pie_floor=a.pie_floor, mate_lift=a.mate_lift))
    print()
    print(f"  flagged games   {r['n_flagged']:,}")
    print(f"  control games   {r['n_control']:,}")
    if r["note"]:
        print("  " + r["note"])
        return 0
    print(f"  flagged beat their baseline by  {r['flagged_mean']:+.3f} shots")
    print(f"  control beat theirs by          {r['control_mean']:+.3f} shots")
    print(f"  DIFFERENCE                      {r['effect']:+.3f} shots "
          f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print()
    if r["lo"] > 0:
        print("  The interval clears zero. The angle is real at team level, "
              "and line-level data is now worth paying for.")
    elif r["hi"] < 0:
        print("  Negative and clear of zero. A squeezed man shoots LESS next "
              "game - that is role drift, not alternation. Do not build it, "
              "or build it backwards.")
    else:
        print("  The interval covers zero. On your own history the flag adds "
              "nothing beyond the player's shrunk rate. The Last% column is "
              "still worth having as a thing the FIELD overlooks, but it is "
              "not a projection signal and should not be fed into one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

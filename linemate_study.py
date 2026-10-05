#!/usr/bin/env python3
"""
Does a squeezed linemate bounce back?

THE QUESTION, STATED SO IT CAN BE ANSWERED
------------------------------------------
The angle: a line generates roughly the same number of shots each night, so if
one winger took most of them last game and his linemate took none, the
linemate is due.

That is one of three stories, and only one of them is the hypothesis:

  1. MULTINOMIAL NOISE. Given the line's total shots, who takes them is close
     to a random draw around each man's true share. A low share last game then
     carries ZERO information about this game. This is the null.

  2. ROLE DRIFT. He got squeezed because he has been demoted inside the line,
     or is hurt, or the coach has changed the look. Then a low share predicts
     ANOTHER low share - the signal exists but points the opposite way.

  3. ALTERNATION. Something makes the shares take turns.

Hockey supplies a mechanism for 1 and 2. I know of none for 3: players do not
take turns shooting. So my prior is that this measures zero, or measures a
small NEGATIVE effect from role drift. That prior is worth exactly nothing
against data, which is what this script is for.

WHAT IT ACTUALLY MEASURES
-------------------------
Not "do flagged players shoot more next game" - of course they do, the flag
only fires on shooters. The question is whether the flag adds anything BEYOND
what the player's own shrunk rate already says. So every comparison is against
a baseline expectation built from his prior games, and the effect reported is
the average surplus over that baseline, flagged minus unflagged, inside a
matched pool of players the flag was eligible to fire on.

NO LOOK-AHEAD. Every baseline and every norm is computed from games strictly
before the game being predicted. This is the single easiest way to produce a
beautiful result that is entirely an artefact, so it is enforced in one place
and tested.

USAGE
    python3 linemate_study.py --self-test          # verify the estimator
    python3 linemate_study.py --seasons 2022 2023 2024
"""

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

SKATERS_URL = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/skaters/{year}.zip"
LINES_URL = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/lines/{year}.zip"


# --------------------------------------------------------------- loading
def _pick(cols, *candidates, required=True, what=""):
    """Find a column by any of several plausible names.

    MoneyPuck's headers are not documented for the line files and have changed
    before. A missing column must therefore fail LOUDLY, naming what it looked
    for and printing what is actually there - a silent None here would quietly
    turn into a column of zeros and a confident wrong answer.
    """
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if not required:
        return None
    raise SystemExit(
        f"Could not find the {what or candidates[0]} column.\n"
        f"  looked for: {', '.join(candidates)}\n"
        f"  file has:   {', '.join(map(str, cols))}"
    )


def _read_zip(path_or_url, cache_dir):
    """Read the single CSV inside a MoneyPuck zip, caching the download."""
    name = os.path.basename(path_or_url)
    local = os.path.join(cache_dir, name)
    if not os.path.exists(local):
        import urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        sys.stderr.write(f"downloading {path_or_url}\n")
        urllib.request.urlretrieve(path_or_url, local)
    with zipfile.ZipFile(local) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not inner:
            raise SystemExit(f"{local} contains no CSV")
        with z.open(inner[0]) as fh:
            return pd.read_csv(io.BytesIO(fh.read()), low_memory=False)


def load_real(seasons, cache_dir):
    sk, ln = [], []
    for y in seasons:
        s = _read_zip(SKATERS_URL.format(year=y), cache_dir)
        l = _read_zip(LINES_URL.format(year=y), cache_dir)
        s["season"], l["season"] = y, y
        sk.append(s)
        ln.append(l)
    skaters = pd.concat(sk, ignore_index=True)
    lines = pd.concat(ln, ignore_index=True)

    # Skaters: one row per player per game per situation. "all" is the whole
    # game; anything else would answer a different question.
    sit = _pick(skaters.columns, "situation", required=False)
    if sit is not None:
        skaters = skaters[skaters[sit].astype(str) == "all"].copy()

    out = pd.DataFrame({
        "season": skaters["season"],
        "player_id": skaters[_pick(skaters.columns, "playerId", "player_id", "id")],
        "name": skaters[_pick(skaters.columns, "name", "playerName")],
        "team": skaters[_pick(skaters.columns, "team", "teamCode")],
        "game_id": skaters[_pick(skaters.columns, "gameId", "game_id")],
        "toi": pd.to_numeric(
            skaters[_pick(skaters.columns, "icetime", "iceTime", "timeOnIce")],
            errors="coerce"),
        "sog": pd.to_numeric(
            skaters[_pick(skaters.columns, "I_F_shotsOnGoal", "shotsOnGoal",
                          "I_F_shots", "shots")],
            errors="coerce"),
    })
    # MoneyPuck ice time is in seconds.
    if out["toi"].median() > 200:
        out["toi"] = out["toi"] / 60.0

    lid = _pick(lines.columns, "lineId", "line_id", "lineID", what="line id")
    lsit = _pick(lines.columns, "situation", required=False)
    if lsit is not None:
        lines = lines[lines[lsit].astype(str) == "all"].copy()
    members = _pick(lines.columns, "playerIds", "players", "lineNames",
                    "name", what="line membership")
    linemap = pd.DataFrame({
        "season": lines["season"],
        "game_id": lines[_pick(lines.columns, "gameId", "game_id")],
        "line_id": lines[lid],
        "members": lines[members].astype(str),
    })
    return out.dropna(subset=["sog", "toi"]), linemap


# ---------------------------------------------------------- the signal
def build_panel(games, linemap, *, shooter_bar=1.8, squeeze_drop=0.5,
                pie_floor=0.8, mate_lift=1.4, prior_min=5, shrink=6.0):
    """Attach, for every (player, game), the flag computed on his PREVIOUS game.

    Every norm below uses games strictly earlier than the game the flag is
    computed on, and the outcome is the game after that. Two strict
    inequalities, enforced by construction rather than by filtering afterwards.
    """
    g = games.sort_values(["player_id", "season", "game_id"]).copy()

    # Prior-only running means, shrunk toward the pool mean so that a man with
    # three games does not get a norm built out of three games.
    pool_sog = g["sog"].mean()
    grp = g.groupby("player_id", sort=False)
    g["n_prior"] = grp.cumcount()
    g["sum_prior"] = grp["sog"].cumsum() - g["sog"]
    g["rate_norm"] = (g["sum_prior"] + shrink * pool_sog) / (g["n_prior"] + shrink)

    # Line membership for the game in which the flag is evaluated.
    mem = linemap.copy()
    mem["members"] = mem["members"].astype(str)
    rows = []
    for _, r in mem.iterrows():
        for pid in [p for p in r["members"].replace(";", ",").split(",") if p.strip()]:
            rows.append((r["season"], r["game_id"], r["line_id"], pid.strip()))
    long = pd.DataFrame(rows, columns=["season", "game_id", "line_id", "pid"])
    long["pid"] = pd.to_numeric(long["pid"], errors="coerce")
    long = long.dropna(subset=["pid"])
    long["pid"] = long["pid"].astype(np.int64)

    g2 = g.merge(long, left_on=["season", "game_id", "player_id"],
                 right_on=["season", "game_id", "pid"], how="left")
    g2 = g2.drop(columns=["pid"])

    # The line's own total that night, and each man's slice of it.
    line_tot = (g2.dropna(subset=["line_id"])
                  .groupby(["season", "game_id", "line_id"])["sog"].sum()
                  .rename("line_sog").reset_index())
    g2 = g2.merge(line_tot, on=["season", "game_id", "line_id"], how="left")
    g2["share"] = np.where(g2["line_sog"] > 0, g2["sog"] / g2["line_sog"], np.nan)

    # The line's normal total, from that line's earlier games only.
    g2 = g2.sort_values(["line_id", "season", "game_id"])
    lg = g2.dropna(subset=["line_id"]).groupby(
        ["season", "game_id", "line_id"], as_index=False)["line_sog"].first()
    lg = lg.sort_values(["line_id", "game_id"])
    lgg = lg.groupby("line_id", sort=False)
    lg["line_n"] = lgg.cumcount()
    lg["line_prior"] = lgg["line_sog"].cumsum() - lg["line_sog"]
    lg["line_norm"] = lg["line_prior"] / lg["line_n"].replace(0, np.nan)
    g2 = g2.merge(lg[["season", "game_id", "line_id", "line_norm"]],
                  on=["season", "game_id", "line_id"], how="left")

    # Did a LINEMATE have an unusually big night, in the same game?
    mate = g2.dropna(subset=["line_id"])[
        ["season", "game_id", "line_id", "player_id", "sog", "rate_norm"]].copy()
    mate["lift"] = mate["sog"] / mate["rate_norm"].replace(0, np.nan)
    best = (mate.sort_values("lift", ascending=False)
                .groupby(["season", "game_id", "line_id"], as_index=False)
                .head(2))
    # The best lift that is NOT the player himself.
    top = best.groupby(["season", "game_id", "line_id"]).agg(
        l1=("lift", "max")).reset_index()
    second = (best.sort_values("lift", ascending=False)
                  .groupby(["season", "game_id", "line_id"])["lift"]
                  .apply(lambda s: s.iloc[1] if len(s) > 1 else np.nan)
                  .rename("l2").reset_index())
    top = top.merge(second, on=["season", "game_id", "line_id"], how="left")
    g2 = g2.merge(top, on=["season", "game_id", "line_id"], how="left")
    # His own lift, so it can be excluded: the biggest night on the line is
    # often HIS, and a man cannot be squeezed out by himself.
    g2["own_lift"] = g2["sog"] / g2["rate_norm"].replace(0, np.nan)
    g2["mate_lift"] = np.where(
        np.isclose(g2["own_lift"].fillna(-1), g2["l1"].fillna(-2)),
        g2["l2"], g2["l1"])

    # THE FLAG, on this game, about the next one.
    g2["squeezed"] = (
        (g2["rate_norm"] >= shooter_bar)                 # he is a real shooter
        & (g2["n_prior"] >= prior_min)                   # and we know that
        & (g2["sog"] <= g2["rate_norm"] * squeeze_drop)  # his own night was thin
        & (g2["line_sog"] >= g2["line_norm"] * pie_floor)  # the PIE was normal
        & (g2["mate_lift"] >= mate_lift)                 # a mate feasted
    )
    # Eligible = the flag could have fired on him: a known shooter who played.
    # Everything is measured inside this pool, so "flagged players shoot more"
    # cannot pass for a finding.
    g2["eligible"] = (g2["rate_norm"] >= shooter_bar) & (g2["n_prior"] >= prior_min)

    # Line the outcome up: next game for the same player, same season.
    g2 = g2.sort_values(["player_id", "season", "game_id"])
    nxt = g2.groupby(["player_id", "season"], sort=False)
    g2["next_sog"] = nxt["sog"].shift(-1)
    g2["next_baseline"] = nxt["rate_norm"].shift(-1)
    return g2


# ------------------------------------------------------------ estimation
def estimate(panel, n_boot=2000, seed=11):
    """Average surplus over baseline, flagged minus unflagged, within the pool."""
    d = panel[panel["eligible"] & panel["next_sog"].notna()
              & panel["next_baseline"].notna()].copy()
    d["surplus"] = d["next_sog"] - d["next_baseline"]
    a = d.loc[d["squeezed"], "surplus"].to_numpy()
    b = d.loc[~d["squeezed"], "surplus"].to_numpy()
    if len(a) < 30:
        return {"n_flagged": len(a), "n_control": len(b), "effect": np.nan,
                "lo": np.nan, "hi": np.nan, "note": "too few flagged games"}
    eff = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (rng.choice(a, len(a), replace=True).mean()
                   - rng.choice(b, len(b), replace=True).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_flagged": len(a), "n_control": len(b), "effect": eff,
            "lo": lo, "hi": hi,
            "flagged_mean": a.mean(), "control_mean": b.mean(),
            "note": ""}


# ------------------------------------------------------------- self-test
def _synthetic(n_players=240, n_games=70, true_effect=0.0, seed=3):
    """A league where the truth is known, so the estimator can be checked.

    Lines of three. Each line has a nightly total; who takes the shots is a
    multinomial draw on fixed shares - which is the NULL. `true_effect` adds a
    real bounce-back on top, so the estimator can be shown to find an effect
    that is there and not find one that is not.
    """
    rng = np.random.default_rng(seed)
    n_lines = n_players // 3
    shares = rng.dirichlet(np.array([4.0, 3.0, 2.0]), size=n_lines)
    line_mu = rng.uniform(5.0, 11.0, size=n_lines)
    rows, lrows = [], []
    squeezed_prev = np.zeros((n_lines, 3), dtype=bool)
    for gi in range(n_games):
        for li in range(n_lines):
            tot = rng.poisson(line_mu[li])
            p = shares[li].copy()
            if true_effect:
                p = p * (1.0 + true_effect * squeezed_prev[li])
                p = p / p.sum()
            draw = rng.multinomial(tot, p) if tot > 0 else np.zeros(3, int)
            gid = 20000 + gi
            pids = [li * 3 + k for k in range(3)]
            for k, pid in enumerate(pids):
                rows.append((2024, pid, f"P{pid}", f"T{li%32}", gid, 15.0,
                             float(draw[k])))
            lrows.append((2024, gid, f"L{li}", ",".join(map(str, pids))))
            # Who was squeezed THIS game, for the next one.
            with np.errstate(invalid="ignore"):
                exp = shares[li] * max(tot, 1)
            squeezed_prev[li] = (draw <= 0.5 * exp) & (exp >= 1.5)
    games = pd.DataFrame(rows, columns=["season", "player_id", "name", "team",
                                        "game_id", "toi", "sog"])
    linemap = pd.DataFrame(lrows, columns=["season", "game_id", "line_id",
                                           "members"])
    return games, linemap


def self_test():
    ok = fails = 0

    def check(cond, msg):
        nonlocal ok, fails
        if cond:
            ok += 1
        else:
            fails += 1
            print("  FAIL:", msg)

    print("A league where shot shares are pure multinomial noise (the null):")
    games, linemap = _synthetic(true_effect=0.0)
    panel = build_panel(games, linemap)
    res = estimate(panel)
    print(f"  flagged {res['n_flagged']} games, control {res['n_control']}")
    print(f"  effect {res['effect']:+.3f} shots  "
          f"[{res['lo']:+.3f}, {res['hi']:+.3f}]")
    check(res["n_flagged"] > 100, "the flag fires often enough to measure")
    check(res["lo"] <= 0 <= res["hi"],
          "under the null the interval must cover zero")
    check(abs(res["effect"]) < 0.25, "and the point estimate must be small")

    print("\nThe same league with a real 60% bounce-back built in:")
    games, linemap = _synthetic(true_effect=0.6, seed=4)
    panel = build_panel(games, linemap)
    res2 = estimate(panel)
    print(f"  flagged {res2['n_flagged']} games, control {res2['n_control']}")
    print(f"  effect {res2['effect']:+.3f} shots  "
          f"[{res2['lo']:+.3f}, {res2['hi']:+.3f}]")
    check(res2["effect"] > 0.15, "a real effect is detected")
    check(res2["lo"] > 0, "and is distinguishable from zero")
    check(res2["effect"] > res["effect"] + 0.15,
          "and is clearly larger than the null case")

    print("\nNo look-ahead:")
    games, linemap = _synthetic(true_effect=0.0, seed=7)
    panel = build_panel(games, linemap)
    first = panel.sort_values(["player_id", "game_id"]).groupby("player_id").head(1)
    check(bool((first["n_prior"] == 0).all()),
          "a player's first game has no prior games behind it")
    check(bool(first["sum_prior"].eq(0).all()),
          "and no prior shots")
    check(not panel.loc[panel["eligible"], "rate_norm"].isna().any(),
          "every eligible row has a baseline")
    # The baseline for the outcome game must not contain the outcome.
    sub = panel.dropna(subset=["next_sog", "next_baseline"]).head(400)
    bad = 0
    for _, r in sub.iterrows():
        same = panel[(panel["player_id"] == r["player_id"])
                     & (panel["game_id"] > r["game_id"])]
        if same.empty:
            continue
        nxt = same.sort_values("game_id").iloc[0]
        if not np.isclose(nxt["rate_norm"], r["next_baseline"]):
            bad += 1
    check(bad == 0, f"the outcome's baseline is that game's prior-only norm ({bad} off)")

    print("\nThe flag's conditions actually bind:")
    games, linemap = _synthetic(true_effect=0.0, seed=9)
    loose = estimate(build_panel(games, linemap, shooter_bar=0.0, prior_min=0,
                                 squeeze_drop=9.9, pie_floor=0.0, mate_lift=0.0))
    tight = estimate(build_panel(games, linemap, shooter_bar=2.5,
                                 squeeze_drop=0.25, pie_floor=1.0,
                                 mate_lift=2.0))
    print(f"  loose: {loose['n_flagged']} flagged   tight: {tight['n_flagged']} flagged")
    check(loose["n_flagged"] > tight["n_flagged"],
          "tightening every threshold flags strictly fewer games")

    print(f"\n{ok} passed, {fails} failed")
    return fails


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--cache", default=".mp_cache")
    ap.add_argument("--shooter-bar", type=float, default=1.8,
                    help="only flag men whose prior-games norm is at least this "
                         "many shots a game")
    ap.add_argument("--squeeze-drop", type=float, default=0.5,
                    help="his own night counts as thin at or below this "
                         "fraction of his norm")
    ap.add_argument("--pie-floor", type=float, default=0.8,
                    help="the LINE's total must be at least this fraction of "
                         "its own norm - a line that generated nothing did not "
                         "squeeze anybody")
    ap.add_argument("--mate-lift", type=float, default=1.4,
                    help="a linemate must have shot at least this multiple of "
                         "his own norm")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    games, linemap = load_real(a.seasons, a.cache)
    print(f"{len(games):,} player-games, {linemap['line_id'].nunique():,} lines")
    panel = build_panel(games, linemap, shooter_bar=a.shooter_bar,
                        squeeze_drop=a.squeeze_drop, pie_floor=a.pie_floor,
                        mate_lift=a.mate_lift)
    res = estimate(panel)
    print()
    print(f"  flagged games   {res['n_flagged']:,}")
    print(f"  control games   {res['n_control']:,}")
    if res["note"]:
        print("  " + res["note"])
        return
    print(f"  flagged beat their baseline by  {res['flagged_mean']:+.3f} shots")
    print(f"  control beat theirs by          {res['control_mean']:+.3f} shots")
    print(f"  DIFFERENCE                      {res['effect']:+.3f} shots "
          f"[{res['lo']:+.3f}, {res['hi']:+.3f}]")
    print()
    if res["lo"] > 0:
        print("  The interval clears zero. The angle is worth building.")
    elif res["hi"] < 0:
        print("  Negative and clear of zero: a squeezed man shoots LESS next "
              "game, which is role drift, not alternation. Do not build it, "
              "or build it backwards.")
    else:
        print("  The interval covers zero. On this history the flag adds "
              "nothing beyond the player's own shrunk rate. Shipping it as a "
              "projection signal would be shipping noise.")


if __name__ == "__main__":
    main()

"""Does any of this actually predict anything? Measured, not asserted.

Walks forward through the cached history: for every game, builds the rate
from that player's EARLIER games only, projects the stat, and compares it to
what happened. Nothing here is fitted, so there is nothing to overfit - it is
purely a report on the model you are about to trust.

Three baselines, because a number with nothing to beat is not a result:

  league   every player gets his position's average rate
  own      the player's own simple career average rate
  model    the shrunk, exponentially weighted rate, with the xG blend

WHAT THIS DOES NOT GRADE, stated plainly because a benchmark that quietly
measures something else is worse than none: the game-state split, the
opponent and home-ice factors, the shot-attempt and on-ice-xG substitutions,
and the ice-time projection (every stat here is projected onto the ACTUAL ice
time of the game being graded, to isolate the rate from the deployment). So
this is a floor on the shipped model, not a measurement of it.

If `model` cannot beat `own`, the shrinkage is costing more than it saves.
If neither can beat `league`, the stat is not predictable at the player level
and should be read as a position label with extra steps.

    python validate.py
    python validate.py --stat goals --xg-sweep
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import fetch as FETCH
import model as M

log = logging.getLogger("validate")

MIN_PRIOR_GAMES = 10       # do not grade a prediction made from four games


def prior_rates(hist: pd.DataFrame, stat: str, halflife: float) -> pd.DataFrame:
    """For every row: the exponentially weighted rate over EARLIER games only.

    `.shift(1)` inside the group is the whole point. Without it the rate for a
    game includes that game, every correlation comes out beautiful, and the
    model is being graded on its ability to remember the answer.
    """
    h = hist.sort_values(["player_id", "date"]).copy()
    g = h.groupby("player_id")
    num = g[stat].transform(lambda s: s.fillna(0).ewm(halflife=halflife)
                            .mean().shift(1))
    den = g["toi"].transform(lambda s: s.fillna(0).ewm(halflife=halflife)
                             .mean().shift(1))
    cnt = g.cumcount()
    h["_num"] = num
    h["_den"] = den
    h["_n"] = cnt
    h["_own_num"] = g[stat].transform(
        lambda s: s.fillna(0).expanding().mean().shift(1))
    h["_own_den"] = g["toi"].transform(
        lambda s: s.fillna(0).expanding().mean().shift(1))
    return h


def grade(actual, pred) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if len(a) < 50:
        return {"n": len(a), "mae": float("nan"), "r": float("nan"),
                "bias": float("nan")}
    return {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(a - p))),
        "r": float(np.corrcoef(a, p)[0, 1]) if np.std(p) > 0 else float("nan"),
        "bias": float(np.mean(p - a)),
    }


def run(hist: pd.DataFrame, stat: str, xg_weight: float | None = None) -> dict:
    if stat not in hist.columns:
        log.warning("%s is not in the history", stat)
        return {}
    if stat not in M.PRIOR:
        log.warning("%s has no prior strength in model.PRIOR, so it cannot "
                    "be graded the way the model estimates it", stat)
        return {}
    h = prior_rates(hist, stat, M.HALFLIFE_RATE)
    h["grp"] = h["position"].map(M.group_of)
    pri = M.position_priors(hist, [stat])
    prior_rate = h["grp"].map(pri[stat]).fillna(0.0)

    use = h[(h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)].copy()
    if not len(use):
        return {}

    # Everything is projected onto the ACTUAL ice time of the game being
    # graded. That is deliberate: it isolates the rate model from the ice-time
    # model, so a bad number here means the rate is wrong rather than that the
    # player got benched.
    share = use["toi"] / 60.0
    ew_rate = 60.0 * use["_num"] / use["_den"].replace(0, np.nan)
    own_rate = 60.0 * use["_own_num"] / use["_own_den"].replace(0, np.nan)
    # The sample size, in sixty-minute games, matching `weighted_sums`.
    #
    # `_den` is an exponentially weighted MEAN of ice time, and the model's
    # sample size is the weighted SUM, so the mean has to be multiplied by the
    # sum of the weights. That sum SATURATES at 1/(1 - r) - about 29.4 games
    # at a twenty-game halflife - it does not grow linearly with career
    # length. Multiplying by the raw game count instead applied roughly half
    # the shrinkage the model applies, so the sweep was tuning a constant
    # against an estimator that does not ship.
    r = 0.5 ** (1.0 / M.HALFLIFE_RATE)
    wsum = (1.0 - r ** use["_n"].clip(lower=1)) / (1.0 - r)
    games = (use["_den"] * wsum) / 60.0
    shrunk = ((ew_rate.fillna(0) * games + prior_rate[use.index] * M.PRIOR[stat])
              / (games + M.PRIOR[stat]))

    if stat == "goals" and "xg" in hist.columns:
        hx = prior_rates(hist, "xg", M.HALFLIFE_RATE)
        xg_rate = 60.0 * hx["_num"] / hx["_den"].replace(0, np.nan)
        xg_rate = xg_rate.reindex(use.index)
        prix = M.position_priors(hist, ["xg"])
        px = use["grp"].map(prix["xg"]).fillna(0.0)
        xg_shrunk = ((xg_rate.fillna(0) * games + px * M.PRIOR["xg"])
                     / (games + M.PRIOR["xg"]))
        w = M.XG_WEIGHT if xg_weight is None else xg_weight
        shrunk = w * xg_shrunk + (1 - w) * shrunk

    out = {
        "league": grade(use[stat], prior_rate[use.index] * share),
        "own": grade(use[stat], own_rate.fillna(0) * share),
        "model": grade(use[stat], shrunk * share),
    }
    return out


def report(name: str, res: dict) -> None:
    if not res:
        return
    print(f"\n{name}")
    print(f"  {'':8s} {'n':>7s} {'MAE':>8s} {'corr':>7s} {'bias':>8s}")
    for k in ("league", "own", "model"):
        g = res.get(k)
        if not g:
            continue
        print(f"  {k:8s} {g['n']:7d} {g['mae']:8.4f} {g['r']:7.3f} "
              f"{g['bias']:+8.4f}")
    base = res.get("own", {}).get("mae")
    mod = res.get("model", {}).get("mae")
    if base and mod and np.isfinite(base) and np.isfinite(mod):
        print(f"  model beats a player's own average by "
              f"{100 * (base - mod) / base:+.1f}% of MAE")


def run_toi(hist: pd.DataFrame) -> dict:
    """Ice time graded on its own terms.

    It was previously in the stat list and then skipped by a `continue`, so
    the README promised a number nobody printed. It cannot go through `run()`
    because every other stat is graded against ACTUAL ice time - grading ice
    time against itself would score a perfect one.
    """
    h = hist.sort_values(["player_id", "date"]).copy()
    g = h.groupby("player_id")
    h["_ew"] = g["toi"].transform(
        lambda s: s.fillna(0).ewm(halflife=M.HALFLIFE_TOI).mean().shift(1))
    h["_own"] = g["toi"].transform(
        lambda s: s.fillna(0).expanding().mean().shift(1))
    h["_n"] = g.cumcount()
    h["grp"] = h["position"].map(M.group_of)
    lg = h.groupby("grp")["toi"].mean()
    use = h[(h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)].copy()
    # SHRUNK, the way model.py shrinks it. Grading the bare exponential
    # average would flatter an estimator that does not ship - the same
    # mistake this file used to make for every other stat.
    r = 0.5 ** (1.0 / M.HALFLIFE_TOI)
    wsum = (1.0 - r ** use["_n"].clip(lower=1)) / (1.0 - r)
    prior = use["grp"].map(lg).fillna(14.0)
    shrunk = ((use["_ew"].fillna(0) * wsum + M.PRIOR_TOI * prior)
              / (wsum + M.PRIOR_TOI))
    return {
        "league": grade(use["toi"], prior),
        "own": grade(use["toi"], use["_own"]),
        "model": grade(use["toi"], shrunk),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--stat", default=None, help="one stat instead of all")
    ap.add_argument("--xg-sweep", action="store_true",
                    help="grade goals at several xG weights and print the best")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    hist = FETCH.load(seasons, "skaters")

    if a.stat == "toi":
        # Routed rather than crashed. It used to sit in the stat list and be
        # skipped by a `continue`; asking for it by name then died on
        # KeyError inside PRIOR.
        report("time on ice (predicting the NEXT game's minutes)",
               run_toi(hist))
        return 0
    stats = ([a.stat] if a.stat else
             [s for s in ("sog", "goals", "assists", "points", "blocks",
                          "hits", "pim") if s in hist.columns])
    for s in stats:
        report(s, run(hist, s))
    if not a.stat:
        report("time on ice (predicting the NEXT game's minutes)",
               run_toi(hist))

    if a.xg_sweep and "xg" in hist.columns:
        print("\ngoals, by how much weight the projection puts on expected "
              "goals rather than on goals")
        print(f"  {'weight':>7s} {'MAE':>8s} {'corr':>7s}")
        best = None
        for w in [0.0, 0.25, 0.5, 0.65, 0.8, 1.0]:
            g = run(hist, "goals", xg_weight=w).get("model", {})
            if not g or not np.isfinite(g.get("mae", np.nan)):
                continue
            print(f"  {w:7.2f} {g['mae']:8.4f} {g['r']:7.3f}")
            if best is None or g["mae"] < best[1]:
                best = (w, g["mae"])
        if best:
            print(f"  -> lowest error at XG_WEIGHT = {best[0]:.2f} "
                  f"(model.py currently uses {M.XG_WEIGHT:.2f})")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())

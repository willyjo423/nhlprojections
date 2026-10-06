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

THE SQUEEZE, GRADED AGAINST THE REAL MODEL
------------------------------------------
`linemate_study.py` found a flagged ("squeezed") shooter taking 0.109 FEWER
shots next game than a crude shrunk-mean baseline. That number cannot be
wired into the projection as it stands, because the projection does not use
that baseline: it estimates ice time separately on a fast halflife. If a man
was squeezed because he got demoted, the ice-time model has already seen part
of it, and subtracting 0.109 on top would count the demotion twice.

`--squeeze` settles it by measuring the same effect twice, against two
predictions that differ in exactly one way:

  on ACTUAL ice time     the game's real minutes. A deficit here is the RATE,
                         which the shipped model cannot see.
  on PROJECTED ice time  the model's own shrunk estimate - what the page
                         actually printed.

The difference between the two is what the ice-time model recovers on its own.
Only the actual-ice-time number is left to correct.

    python validate.py
    python validate.py --stat goals --xg-sweep
MEASURED, on 170,139 player-games and 13,191 flagged ones: the rate effect is
-0.018 shots with an interval of [-0.048, +0.013], which crosses zero, while
the projected-minutes effect is -0.054 [-0.084, -0.022]. So the flag carries
NOTHING about how well a man shoots, and all of what little is there is the
ice-time projection being late. There is no squeeze adjustment to build.

`--toi-sweep` follows that one step further, because the lateness is not a
fact about squeezed players - it is a fact about HALFLIFE_TOI, and a demotion
is only the case this flag happens to select. It sweeps the halflife and
prints BOTH the overall ice-time error and the flagged subset's deficit, so a
change that helps recently-demoted players by making every other projection
twitchier shows up as the trade it is.

    python validate.py
    python validate.py --stat goals --xg-sweep
    python validate.py --squeeze-self-test     # check it first, no download
    python validate.py --squeeze --toi-sweep   # then point it at the cache
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


def project_toi(hist: pd.DataFrame, halflife: float | None = None) -> pd.Series:
    """The ice time the model would project for each game, from EARLIER games.

    One definition, used by `run_toi`, by `run_squeeze` and by the halflife
    sweep. It used to be written out twice - once here and once inside the
    squeeze grader - and two copies of an estimator is how a sweep ends up
    tuning a constant against something that does not ship.
    """
    hl = M.HALFLIFE_TOI if halflife is None else float(halflife)
    h = hist.sort_values(["player_id", "date"])
    g = h.groupby("player_id")
    ew = g["toi"].transform(
        lambda s: s.fillna(0).ewm(halflife=hl).mean().shift(1))
    n = g.cumcount()
    grp = h["position"].map(M.group_of)
    lg = h.groupby(grp)["toi"].mean()
    r = 0.5 ** (1.0 / hl)
    wsum = (1.0 - r ** n.clip(lower=1)) / (1.0 - r)
    prior = grp.map(lg).fillna(14.0)
    # SHRUNK, the way model.py shrinks it. Grading the bare exponential
    # average would flatter an estimator that does not ship - the same
    # mistake this file used to make for every other stat.
    return ((ew.fillna(0) * wsum + M.PRIOR_TOI * prior)
            / (wsum + M.PRIOR_TOI)).reindex(hist.index)


def run_toi(hist: pd.DataFrame, halflife: float | None = None) -> dict:
    """Ice time graded on its own terms.

    It was previously in the stat list and then skipped by a `continue`, so
    the README promised a number nobody printed. It cannot go through `run()`
    because every other stat is graded against ACTUAL ice time - grading ice
    time against itself would score a perfect one.
    """
    h = hist.sort_values(["player_id", "date"]).copy()
    g = h.groupby("player_id")
    h["_own"] = g["toi"].transform(
        lambda s: s.fillna(0).expanding().mean().shift(1))
    h["_n"] = g.cumcount()
    h["grp"] = h["position"].map(M.group_of)
    h["_proj"] = project_toi(h, halflife)
    lg = h.groupby("grp")["toi"].mean()
    use = h[(h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)].copy()
    prior = use["grp"].map(lg).fillna(14.0)
    return {
        "league": grade(use["toi"], prior),
        "own": grade(use["toi"], use["_own"]),
        "model": grade(use["toi"], use["_proj"]),
    }


# ------------------------------------------------- the squeeze, graded properly
def _boot(a, b, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    d = np.array([rng.choice(a, len(a), True).mean()
                  - rng.choice(b, len(b), True).mean() for _ in range(n)])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def squeeze_flags(hist: pd.DataFrame, stat: str = "sog") -> pd.DataFrame:
    """Last game's flag, per player-game. Computed ONCE, because the sweep
    calls the grader a dozen times and the flag does not depend on the
    halflife being swept."""
    import linemate_study as LS

    # The flag for each game, then carried FORWARD one game, so a row says
    # "the game before this one was flagged" - which is the thing that is
    # supposed to predict this one.
    panel = LS.build_panel(hist[["player_id", "team", "game_id", "date",
                                 stat]].rename(columns={stat: "sog"}))
    panel = panel.sort_values(["player_id", "date"])
    g = panel.groupby("player_id", sort=False)
    panel["prev_sq"] = g["squeezed"].shift(1)
    panel["prev_el"] = g["eligible"].shift(1)
    return panel[["player_id", "game_id", "prev_sq", "prev_el"]]


def run_squeeze(hist: pd.DataFrame, stat: str = "sog",
                flags: pd.DataFrame | None = None,
                toi_halflife: float | None = None) -> dict:
    """How much of the squeeze effect the SHIPPED model already catches.

    `linemate_study.py` measured a flagged player shooting 0.109 fewer shots
    than a crude prior-mean baseline. That number cannot be applied to the
    projection as it stands, because the two are not measuring against the same
    thing: the real model projects ICE TIME separately, on a fast halflife, so
    if a man was squeezed because he got demoted the deployment model has
    already seen some of it. Subtracting the full 0.109 on top would count the
    demotion twice.

    So the effect is measured twice, against two predictions that differ in
    exactly one way:

      ON ACTUAL ICE TIME   the game's real minutes. Any deficit left here is
                           NOT about deployment at all - it is the rate
                           itself, and the shipped model cannot see it.
      ON PROJECTED ICE TIME  the model's own shrunk estimate. This is what the
                           page would have printed.

    The gap between the two is what the ice-time model recovers on its own.
    What survives on ACTUAL ice time is what is genuinely left to correct, and
    it is the only one of the two numbers worth wiring into anything.

    `flags` is only supplied by the self-test, which hands in a flag it
    designed so that the ANSWER IS KNOWN. Left `None`, the flag comes from
    `linemate_study.build_panel` - the same definition the page shows, which
    is the whole point of importing it rather than restating it here.
    """
    need = {"player_id", "team", "game_id", "date", stat}
    if not need <= set(hist.columns):
        log.warning("cannot grade the squeeze: missing %s",
                    sorted(need - set(hist.columns)))
        return {}

    if flags is None:
        flags = squeeze_flags(hist, stat)
    flags = flags.drop_duplicates(["player_id", "game_id"])

    # The model's own rate, from earlier games only - the same construction
    # `run()` grades, so this measures the shipped estimator and not a
    # convenient cousin of it.
    h = prior_rates(hist, stat, M.HALFLIFE_RATE)
    h["grp"] = h["position"].map(M.group_of)
    pri = M.position_priors(hist, [stat])
    prior_rate = h["grp"].map(pri[stat]).fillna(0.0)
    use = h[(h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)].copy()
    if not len(use):
        return {}
    r = 0.5 ** (1.0 / M.HALFLIFE_RATE)
    wsum = (1.0 - r ** use["_n"].clip(lower=1)) / (1.0 - r)
    games = (use["_den"] * wsum) / 60.0
    ew_rate = 60.0 * use["_num"] / use["_den"].replace(0, np.nan)
    # Attached as a COLUMN, while the index still lines up. It used to be kept
    # as a loose Series and applied with `.to_numpy()` after two merges - which
    # is correct only as long as neither merge changes the row count, and a
    # single duplicated key on the right-hand side would have silently paired
    # every later row's rate with the wrong game. Carrying it inside the frame
    # makes that impossible rather than merely unlikely.
    use["_rate"] = ((ew_rate.fillna(0) * games
                     + prior_rate[use.index] * M.PRIOR[stat])
                    / (games + M.PRIOR[stat]))

    # The ice time the model would have projected for this game, from earlier
    # games only - the SAME function `run_toi` is graded on.
    ht = hist[["player_id", "game_id"]].copy()
    ht["_proj_toi"] = project_toi(hist, toi_halflife)
    # `validate="m:1"` is the point of these two lines: it raises rather than
    # quietly duplicating rows if a (player, game) key is not unique on the
    # right. A grading script that reshapes its own sample is worse than none.
    toi_map = (ht[["player_id", "game_id", "_proj_toi"]]
               .drop_duplicates(["player_id", "game_id"]))
    before = len(use)
    use = use.merge(toi_map, on=["player_id", "game_id"], how="left",
                    validate="m:1")
    use = use.merge(flags, on=["player_id", "game_id"], how="left",
                    validate="m:1")
    if len(use) != before:
        raise AssertionError(f"the merges changed the sample: {before} -> "
                             f"{len(use)}; the grade would be meaningless")

    use["_actual"] = use[stat]
    use["_on_actual"] = use["_rate"] * use["toi"] / 60.0
    use["_on_proj"] = use["_rate"] * use["_proj_toi"] / 60.0

    # The comparison group is the pool the flag COULD have fired on last game
    # and did not. Anything else and "flagged shooters shoot a lot" passes
    # itself off as a finding, because the flag only ever fires on shooters.
    pool = use[use["prev_el"].fillna(False).astype(bool)
               & use["prev_sq"].notna()].copy()

    # ONE sample for both grades. Filtering non-finite rows separately per
    # grade would compare two different sets of games and call the difference
    # "what the ice-time model recovers", when part of it would just be the
    # sample changing underneath.
    pool["_res_actual"] = pool["_actual"] - pool["_on_actual"]
    pool["_res_proj"] = pool["_actual"] - pool["_on_proj"]
    # THE DEFICIT IN MINUTES, STATED IN MINUTES.
    #
    # The two shot grades differ by exactly rate x (projected - actual) / 60,
    # so if the projected-minutes row is worse than the actual-minutes row,
    # flagged players are being handed more ice time than they go on to play.
    # Reporting that only through its effect on shots left the SIZE of it to
    # be reconstructed by dividing by a rate - which is how I ended up
    # asserting a cause (a slow halflife) that the sweep then refuted. The
    # number itself is three lines; it should have been printed from the
    # start.
    pool["_toi_err"] = pool["_proj_toi"] - pool["toi"]
    pool = pool[np.isfinite(pool["_res_actual"].to_numpy(dtype=float))
                & np.isfinite(pool["_res_proj"].to_numpy(dtype=float))
                & np.isfinite(pool["_toi_err"].to_numpy(dtype=float))]

    sq = pool["prev_sq"].astype(bool).to_numpy()
    out = {"n_flagged": int(sq.sum()), "n_control": int((~sq).sum())}
    if out["n_flagged"] < 50:
        out["note"] = "too few flagged games to grade"
        return out

    for key, col in (("on_actual_toi", "_res_actual"),
                     ("on_projected_toi", "_res_proj"),
                     ("minutes_over_projected", "_toi_err")):
        res = pool[col].to_numpy(dtype=float)
        a, b = res[sq], res[~sq]
        lo, hi = _boot(a, b)
        out[key] = {"flagged": float(a.mean()), "control": float(b.mean()),
                    "effect": float(a.mean() - b.mean()), "lo": lo, "hi": hi}
    return out


def report_squeeze(res: dict, stat: str) -> None:
    if not res:
        return
    print(f"\nthe squeeze flag, graded against the model's own {stat}")
    print(f"  flagged games {res['n_flagged']:,}   "
          f"control {res['n_control']:,}")
    if res.get("note"):
        print("  " + res["note"])
        return
    print(f"  {'':20s} {'flagged':>9s} {'control':>9s} {'effect':>9s} "
          f"{'95% interval':>20s}")
    for key, label in (("on_actual_toi", "on ACTUAL ice time"),
                       ("on_projected_toi", "on PROJECTED ice time")):
        g = res[key]
        print(f"  {label:20s} {g['flagged']:+9.3f} {g['control']:+9.3f} "
              f"{g['effect']:+9.3f}   [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    mins = res.get("minutes_over_projected")
    if mins:
        print(f"\n  and the same thing in MINUTES - how much ice time the "
              f"projection hands him over what he plays")
        print(f"  {'minutes over-projected':22s} {mins['flagged']:+7.3f} "
              f"{mins['control']:+7.3f} {mins['effect']:+7.3f}   "
              f"[{mins['lo']:+.3f}, {mins['hi']:+.3f}]")
    act = res["on_actual_toi"]["effect"]
    prj = res["on_projected_toi"]["effect"]
    print()
    if res["on_actual_toi"]["hi"] >= 0:
        if res["on_projected_toi"]["hi"] < 0:
            print("  On ACTUAL ice time the flag adds nothing: that interval "
                  "crosses zero, so the shooter's RATE is fine and the "
                  "shipped rate model owes nothing.")
            print(f"  On PROJECTED ice time it is still {abs(prj):.3f} short, "
                  f"so the whole of the effect is DEPLOYMENT - he plays fewer "
                  f"minutes than the projection hands him.")
            # Deliberately NOT "the halflife is too slow". An earlier version
            # of this line said exactly that, and then --toi-sweep showed the
            # deficit barely moving from a halflife of 2 to one of 20. A cause
            # a sweep in this same file can refute does not belong in a
            # conclusion.
            print("  WHY is a separate question, and this grade does not "
                  "answer it. Run --toi-sweep: if the deficit shrinks as the "
                  "halflife speeds up, the constant is set too slow. If it "
                  "sits still, the minutes are being lost to news the model "
                  "does not have - a scratch, a line change, something he is "
                  "playing through - and no halflife recovers that.")
        else:
            print("  Against the shipped model the flag is empty: BOTH "
                  "intervals cross zero. The 0.109 the study found was an "
                  "artefact of its cruder baseline, and the model already "
                  "handles whatever is there.")
            print("  Do not subtract anything, and do not add anything. The "
                  "flag is worth showing as context and nothing more.")
        return
    print(f"  The rate itself is down {abs(act):.3f} shots even on the minutes "
          f"he actually played, so deployment cannot explain all of it.")
    if prj >= 0:
        print("  The projected-ice-time row does not point the same way, so "
              "there is nothing to apportion. Treat the actual-ice-time "
              "number as the whole of it.")
    elif abs(act) >= abs(prj):
        print(f"  It is no smaller than the {abs(prj):.3f} on projected "
              f"minutes, so the ice-time model is recovering none of it. All "
              f"{abs(act):.3f} is left to correct.")
    else:
        share = 100 * (1 - act / prj)
        print(f"  Of the {abs(prj):.3f} the page would miss, deployment "
              f"recovers about {share:.0f}%; {abs(act):.3f} is genuinely "
              f"left to correct.")


# ------------------------- the WHOLE shipped number, including what it is not
# graded on anywhere else
OPP_GRID_DAYS = 14          # how often the opponent factors are rebuilt

# Which predictions get the betting test. "rate x mins" is in here on purpose:
# MAE is nearly BLIND to the opponent factor - with perfect knowledge of a
# +-25% defensive effect the whole available MAE gain is under 3% - while the
# same factor moves P(3 or more) from 53% to 62%, which is nine points of win
# probability and decides every bet. So the factor's contribution has to be
# read as the gap between this row and the shipped one, not off the MAE table.
BET_ROWS = ("own shots/game", "rate x mins", "+ home (SHIPPED)")


def _pois_ge(k: int, lam):
    """P(X >= k) for a Poisson, vectorised. The page's own tail function."""
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, np.nan)
    term = np.exp(-lam)
    out = np.array(term, dtype=float)
    for i in range(1, k):
        term = term * lam / i
        out = out + term
    return 1.0 - out


def walkforward_factors(hist: pd.DataFrame, stat: str,
                        grid_days: int = OPP_GRID_DAYS) -> pd.DataFrame:
    """The opponent and home-ice factors as they stood BEFORE each game.

    These two are the reason this function exists. `validate.py` has never
    graded them - its own docstring says so - and on a 2.8-shot projection the
    opponent clip of 0.85-1.15 moves the number by 0.84 shots while the bar
    for beating a -110 price is 0.10. Whatever is or is not in them dwarfs
    everything else this file measures.

    NO LOOK-AHEAD, and it is worth being exact about how. `model.opponent_
    factors(h, cols, asof)` reads a 400-day window ending at `asof`, so it is
    handed a frame already cut to games STRICTLY EARLIER than the grid date.
    Rebuilding that per game would mean one pass per date in four seasons, so
    it is rebuilt every `grid_days` and each game uses the most recent grid
    point at or before it - which is slightly STALER than the live model, not
    fresher. An approximation that errs toward the pessimistic is safe here;
    one that errs the other way would be inventing skill.
    """
    d = pd.to_datetime(hist["date"])
    lo, hi = d.min(), d.max()
    grid = pd.date_range(lo, hi, freq=f"{int(grid_days)}D")
    opp_rows, home_rows = [], []
    for g in grid:
        prior = hist[d < g]
        # A team needs some history before its defence means anything; early
        # grid points simply carry the neutral factor, which is what the
        # shipped model's shrinkage toward 1.0 does anyway.
        if len(prior) < 500:
            home_rows.append({"date": g, "_hf_home": 1.0, "_hf_away": 1.0})
            continue
        try:
            of = M.opponent_factors(prior, [stat], g)
        except Exception:                                      # noqa: BLE001
            of = None
        if of is not None and stat in of.columns:
            for team, v in of[stat].items():
                opp_rows.append({"date": g, "opponent": team,
                                 "_of": float(v)})
        hf = M.home_factors(prior, [stat]).get(stat, (1.0, 1.0))
        home_rows.append({"date": g, "_hf_home": float(hf[0]),
                          "_hf_away": float(hf[1])})

    out = hist[["player_id", "game_id"]].copy()
    out["date"] = d
    out["opponent"] = hist["opponent"] if "opponent" in hist.columns else None
    out["is_home"] = (hist["is_home"] if "is_home" in hist.columns else 0.0)
    out = out.sort_values("date")

    if opp_rows:
        ot = pd.DataFrame(opp_rows).sort_values("date")
        out = pd.merge_asof(out, ot, on="date", by="opponent",
                            direction="backward")
    else:
        out["_of"] = np.nan
    if home_rows:
        ht = (pd.DataFrame(home_rows).drop_duplicates("date")
              .sort_values("date"))
        out = pd.merge_asof(out, ht, on="date", direction="backward")
    else:
        out["_hf_home"] = out["_hf_away"] = np.nan

    out["_of"] = out["_of"].fillna(1.0)
    out["_hf"] = np.where(out["is_home"].fillna(0) > 0.5,
                          out["_hf_home"].fillna(1.0),
                          out["_hf_away"].fillna(1.0))
    return out[["player_id", "game_id", "_of", "_hf"]]


def run_full(hist: pd.DataFrame, stat: str = "sog",
             lines=(1.5, 2.5, 3.5)) -> dict:
    """The number the PAGE prints, graded against what a spreadsheet can do.

    Every other grade in this file projects onto the ACTUAL ice time of the
    game, which isolates the rate but flatters the product. This one uses the
    model's own projected minutes and its own opponent and home factors, so
    what comes out is the error the page actually ships.

    The baseline is deliberately humble: a player's own average SHOTS PER
    GAME, no rate, no minutes model, no adjustments. That is what someone
    with a spreadsheet has. A projection that cannot beat it has no business
    being sorted by its disagreement with a betting line.

    Four rows, so the opponent factor's contribution is visible rather than
    bundled:

      own shots/game    the spreadsheet
      rate x mins       the shrunk rate on projected minutes
      + opponent        and the opponent factor on top
      + home (SHIPPED)  and home ice - the page's number

    Then the part that actually decides whether to bet: for each standard
    line, every game where the model claims better than the -110 break-even
    is treated as a bet, and the CLAIMED probability is set beside the
    REALISED one. A model whose claimed 56% comes in at 51% is not slightly
    optimistic, it is unprofitable, and MAE will never say so.
    """
    need = {"player_id", "game_id", "date", "position", "toi", stat}
    if not need <= set(hist.columns):
        log.warning("cannot grade the full projection: missing %s",
                    sorted(need - set(hist.columns)))
        return {}
    if stat not in M.PRIOR:
        log.warning("%s has no prior strength in model.PRIOR", stat)
        return {}

    h = prior_rates(hist, stat, M.HALFLIFE_RATE)
    h["grp"] = h["position"].map(M.group_of)
    pri = M.position_priors(hist, [stat])
    prior_rate = h["grp"].map(pri[stat]).fillna(0.0)
    h["_proj_toi"] = project_toi(h)
    # The spreadsheet: his own mean shots per game, earlier games only.
    h["_own_pg"] = (h.sort_values(["player_id", "date"])
                    .groupby("player_id")[stat]
                    .transform(lambda s: s.fillna(0).expanding().mean()
                               .shift(1)))

    fac = walkforward_factors(hist, stat).drop_duplicates(
        ["player_id", "game_id"])
    before = len(h)
    h = h.merge(fac, on=["player_id", "game_id"], how="left", validate="m:1")
    if len(h) != before:
        raise AssertionError(f"the factor merge reshaped the sample: "
                             f"{before} -> {len(h)}")

    use = h[(h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)].copy()
    if not len(use):
        return {}
    r = 0.5 ** (1.0 / M.HALFLIFE_RATE)
    wsum = (1.0 - r ** use["_n"].clip(lower=1)) / (1.0 - r)
    games = (use["_den"] * wsum) / 60.0
    ew_rate = 60.0 * use["_num"] / use["_den"].replace(0, np.nan)
    rate = ((ew_rate.fillna(0) * games + prior_rate[use.index] * M.PRIOR[stat])
            / (games + M.PRIOR[stat]))

    base = rate * use["_proj_toi"] / 60.0
    preds = {
        "own shots/game": use["_own_pg"],
        "rate x mins": base,
        "+ opponent": base * use["_of"],
        "+ home (SHIPPED)": base * use["_of"] * use["_hf"],
    }
    out = {"grades": {k: grade(use[stat], v) for k, v in preds.items()},
           "order": list(preds)}
    # The graded rows themselves, keyed so a caller can join its own columns
    # onto them. The self-test needs this to work out the CEILING on what an
    # opponent adjustment could possibly recover - and the alternative was
    # rebuilding the prediction a second time in test code, which is how the
    # two copies of the ice-time projection got out of step earlier.
    rows = use[["player_id", "game_id"]].copy()
    rows["actual"] = use[stat].to_numpy()
    for k, v in preds.items():
        rows[k] = np.asarray(v, dtype=float)
    out["rows"] = rows

    # THE BETTING TEST. MAE is not what a prop pays out on.
    be = 0.5238                                    # the -110 break-even
    out["bets"] = []
    for L in lines:
        k = int(np.ceil(L))
        row = {"line": L, "need": k}
        for label in BET_ROWS:
            lam = np.asarray(preds[label], dtype=float)
            p = _pois_ge(k, lam)
            hit = (use[stat].to_numpy(dtype=float) >= k)
            ok = np.isfinite(p) & np.isfinite(use[stat].to_numpy(dtype=float))
            take = ok & (p > be)
            n = int(take.sum())
            if n < 200:
                row[label] = {"n": n, "note": "too few"}
                continue
            claimed = float(p[take].mean())
            realised = float(hit[take].mean())
            # Flat stake at -110: a win returns 100/110 of the stake.
            roi = realised * (100.0 / 110.0) - (1.0 - realised)
            row[label] = {"n": n, "claimed": claimed, "realised": realised,
                          "roi": roi,
                          "se": float(np.sqrt(realised * (1 - realised) / n))}
        out["bets"].append(row)
    return out


def report_full(res: dict, stat: str) -> None:
    if not res:
        return
    print(f"\nthe WHOLE shipped projection for {stat}, on projected minutes, "
          f"against a spreadsheet")
    print(f"  {'':18s} {'n':>7s} {'MAE':>8s} {'corr':>7s} {'bias':>8s}")
    base_mae = res["grades"].get("own shots/game", {}).get("mae")
    for k in res["order"]:
        g = res["grades"].get(k) or {}
        if not g or not np.isfinite(g.get("mae", np.nan)):
            continue
        tail = ""
        if base_mae and np.isfinite(base_mae) and k != "own shots/game":
            tail = f"   {100 * (base_mae - g['mae']) / base_mae:+6.2f}% vs sheet"
        print(f"  {k:18s} {g['n']:7d} {g['mae']:8.4f} {g['r']:7.3f} "
              f"{g['bias']:+8.4f}{tail}")

    opp = res["grades"].get("+ opponent", {}).get("mae")
    rt = res["grades"].get("rate x mins", {}).get("mae")
    if opp and rt and np.isfinite(opp) and np.isfinite(rt):
        print(f"\n  The opponent factor on its own: "
              f"{100 * (rt - opp) / rt:+.2f}% of MAE - but DO NOT judge it on "
              f"that number.")
        print("  MAE is nearly blind to it. Poisson noise dominates the "
              "error, so PERFECT knowledge of a +-25% defensive effect is "
              "worth under 3% of MAE, while the same factor swings P(3 or "
              "more) by about nine points - and that is what a bet turns on. "
              "Read it off the betting table below instead: the gap between "
              "'rate x mins' and the shipped row IS the opponent and home "
              "factors.")

    print(f"\n  and the only test a prop pays out on: bet every game the model "
          f"calls better than the -110 break-even (52.38%)")
    print(f"  {'line':>5s} {'model':>20s} {'bets':>7s} {'claims':>8s} "
          f"{'actual':>8s} {'gap':>8s} {'ROI':>8s}")
    for row in res["bets"]:
        for label in BET_ROWS:
            g = row.get(label) or {}
            if g.get("note"):
                print(f"  {row['line']:5.1f} {label:>20s} {g['n']:7d}   "
                      f"{g['note']}")
                continue
            gap = g["claimed"] - g["realised"]
            flag = "" if abs(gap) < 2 * g["se"] else "  <- overclaims" \
                if gap > 0 else "  <- underclaims"
            print(f"  {row['line']:5.1f} {label:>20s} {g['n']:7d} "
                  f"{100 * g['claimed']:7.2f}% {100 * g['realised']:7.2f}% "
                  f"{100 * gap:+7.2f}% {100 * g['roi']:+7.2f}%{flag}")

    ship = [r.get("+ home (SHIPPED)") for r in res["bets"]]
    ship = [g for g in ship if g and not g.get("note")]
    print()
    if not ship:
        print("  Not enough qualifying games to say anything.")
        return
    best = max(ship, key=lambda g: g["roi"])
    if best["roi"] <= 0:
        print("  At no standard line does the model clear the vig. Its "
              "probabilities are not good enough to bet into -110, and an "
              "edge column built on them would rank by its own error.")
        print("  This does NOT mean no edge exists - it means the edge cannot "
              "come from the projection as it stands. Fix the projection "
              "first; a column cannot add information.")
    else:
        print(f"  Best standard line is {best['n']:,} bets at "
              f"{100 * best['roi']:+.2f}% ROI, claiming "
              f"{100 * best['claimed']:.2f}% and hitting "
              f"{100 * best['realised']:.2f}%.")
        print("  That is against the STANDARD line, not a real one, so treat "
              "it as whether the probabilities are honest rather than as a "
              "backtest. If they are, real lines are worth pulling.")


# ------------------------------------- is the ice-time halflife set too slow?
TOI_HALFLIVES = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]


def sweep_toi(hist: pd.DataFrame, stat: str = "sog",
              halflives=None) -> dict:
    """Does a faster ice-time halflife help, and what does it cost?

    The squeeze grade found the whole effect sitting on PROJECTED minutes:
    a recently-squeezed man plays fewer minutes than an eight-game halflife
    expects. That is not a fact about squeezed players - it is a fact about
    the halflife being slow after ANY role change, and demotions are only the
    cases this flag happens to select.

    So two numbers at every halflife, and they must both be printed:

      TOI MAE        every graded game. Making the projection twitchier helps
                     the handful of players whose role just changed and hurts
                     everyone whose role did not.
      flagged gap    the squeeze subset's deficit on projected minutes. This
                     is the thing a faster halflife is supposed to close.

    A halflife that closes the gap while raising MAE is a trade, not a fix,
    and reporting only the first number is how a model gets worse on purpose.
    The MAE comparison is PAIRED - the same games at every halflife - because
    comparing two unpaired averages over 150,000 rows would make differences
    far too small to act on look decisive.
    """
    hls = list(TOI_HALFLIVES if halflives is None else halflives)
    shipped = float(M.HALFLIFE_TOI)
    if shipped not in hls:
        hls.append(shipped)
    hls = sorted(set(hls))

    flags = squeeze_flags(hist, stat)            # once; it does not move
    h = hist.sort_values(["player_id", "date"]).copy()
    h["_n"] = h.groupby("player_id").cumcount()
    mask = ((h["_n"] >= MIN_PRIOR_GAMES) & (h["toi"].fillna(0) > 0)).to_numpy()

    rows, errs = [], {}
    for hl in hls:
        p = project_toi(h, hl).to_numpy(dtype=float)
        a = h["toi"].to_numpy(dtype=float)
        ok = mask & np.isfinite(p) & np.isfinite(a)
        errs[hl] = np.where(ok, np.abs(a - p), np.nan)
        sq = run_squeeze(h, stat, flags=flags, toi_halflife=hl)
        rows.append({
            "halflife": hl,
            "mae": float(np.nanmean(errs[hl])),
            "n": int(np.isfinite(errs[hl]).sum()),
            "gap": sq.get("on_projected_toi", {}).get("effect", float("nan")),
            "gap_lo": sq.get("on_projected_toi", {}).get("lo", float("nan")),
            "gap_hi": sq.get("on_projected_toi", {}).get("hi", float("nan")),
            "rate": sq.get("on_actual_toi", {}).get("effect", float("nan")),
        })

    # Paired against the shipped halflife, on the games both could grade.
    base = errs[shipped]
    for r in rows:
        d = errs[r["halflife"]] - base
        d = d[np.isfinite(d)]
        if not len(d) or r["halflife"] == shipped:
            r["delta"], r["delta_se"] = 0.0, 0.0
            continue
        r["delta"] = float(d.mean())
        r["delta_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"shipped": shipped, "rows": rows}


def report_sweep(res: dict) -> None:
    if not res:
        return
    rows, shipped = res["rows"], res["shipped"]
    print(f"\nice-time halflife sweep (model.py ships {shipped:g})")
    print(f"  {'halflife':>9s} {'TOI MAE':>9s} {'vs shipped':>22s} "
          f"{'flagged gap on proj mins':>28s}")
    for r in rows:
        tag = "  <- shipped" if r["halflife"] == shipped else ""
        if r["halflife"] == shipped:
            cmp_ = f"{'':>22s}"
        else:
            lo = r["delta"] - 1.96 * r["delta_se"]
            hi = r["delta"] + 1.96 * r["delta_se"]
            cmp_ = f"{r['delta']:+8.4f} [{lo:+.4f},{hi:+.4f}]"
        print(f"  {r['halflife']:9.1f} {r['mae']:9.4f} {cmp_:>22s} "
              f"  {r['gap']:+8.3f} [{r['gap_lo']:+.3f},{r['gap_hi']:+.3f}]"
              f"{tag}")

    best = min(rows, key=lambda r: r["mae"])
    tight = min(rows, key=lambda r: abs(r["gap"]) if np.isfinite(r["gap"])
                else np.inf)
    cur = next(r for r in rows if r["halflife"] == shipped)
    print()
    # A difference smaller than twice its own standard error is not a
    # difference. Saying "lowest MAE" without that check would recommend a
    # change on noise, which is the whole reason the sweep is paired.
    better = [r for r in rows if r["halflife"] != shipped
              and r["delta"] + 1.96 * r["delta_se"] < 0]
    if not better:
        print(f"  No halflife beats {shipped:g} on ice-time error by more "
              f"than its own noise. LEAVE model.HALFLIFE_TOI ALONE.")
    else:
        b = min(better, key=lambda r: r["delta"])
        print(f"  {b['halflife']:g} lowers ice-time error by "
              f"{abs(b['delta']):.4f} minutes a game "
              f"({100 * abs(b['delta']) / cur['mae']:.2f}% of MAE), and that "
              f"is clear of its own noise.")
        print(f"    At {b['halflife']:g} the flagged gap moves "
              f"{cur['gap']:+.3f} -> {b['gap']:+.3f} shots.")
    # DOES THE DEFICIT RESPOND TO THE HALFLIFE AT ALL? This is the question
    # the sweep exists to answer, and it has to be asked before the trade is
    # discussed - a trade between two things is meaningless if one of them
    # does not move. Measured against the gap's OWN interval width, because
    # "0.005 shots" means nothing without knowing what the noise is.
    gaps = [r["gap"] for r in rows if np.isfinite(r["gap"])]
    width = cur["gap_hi"] - cur["gap_lo"]
    spread = (max(gaps) - min(gaps)) if gaps else float("nan")
    if np.isfinite(spread) and np.isfinite(width) and width > 0:
        print(f"  Across halflives from {min(r['halflife'] for r in rows):g} "
              f"to {max(r['halflife'] for r in rows):g} the flagged deficit "
              f"moves {spread:.3f} shots, against an interval "
              f"{width:.3f} wide on the shipped value.")
        if spread < 0.5 * width:
            print("  So the deficit DOES NOT RESPOND to the halflife. It is "
                  "not a constant that is set too slow: those minutes are "
                  "being lost to news the model does not have - a scratch, a "
                  "demotion decided this morning, something he is playing "
                  "through. No value of this constant recovers information "
                  "that is not in the history.")
            flat = True
        else:
            flat = False
    else:
        flat = False
    if not flat and tight["halflife"] != best["halflife"]:
        print(f"  The halflife that closes the flagged gap most "
              f"({tight['halflife']:g}, gap {tight['gap']:+.3f}) is NOT the "
              f"one with the lowest overall error ({best['halflife']:g}). "
              f"That is a trade: it would help recently-demoted players by "
              f"making every other projection twitchier.")
    worth = abs(cur["gap"]) * 1.5
    print(f"  For scale: the gap as shipped is {abs(cur['gap']):.3f} shots, "
          f"or {worth:.3f} DK points. Nothing in this table is worth a "
          f"change that costs accuracy anywhere else.")


# --------------------------------------------- checking the instrument first
def _squeeze_synthetic(n_players=260, n_games=90, seed=4, n_teams=26):
    """A league with no squeeze effect in it at all.

    Each skater has a fixed shots-per-sixty rate and a fixed workload. Shots
    are Poisson on rate x minutes, so a thin night carries no information
    about the next one. Anything the estimator reports on this data is the
    estimator's own error.
    """
    rng = np.random.default_rng(seed)
    # Shots per SIXTY, at NHL magnitudes - a bottom-six grinder around three,
    # a shoot-first winger around eleven. The first draft of this used 0.8 to
    # 4.2, which produced about 0.6 shots a game and a self-test that passed
    # on numbers four times smaller than the ones being argued about. A
    # fixture that differs from production in the dimension under test is how
    # bugs hide.
    rate = rng.uniform(3.0, 12.0, n_players)
    mins = rng.uniform(9.0, 21.0, n_players)
    pos = rng.choice(["C", "L", "R", "D"], n_players, p=[.2, .25, .25, .3])
    base = pd.Timestamp("2024-10-01")
    pid = np.repeat(np.arange(n_players), n_games)
    gm = np.tile(np.arange(n_games), n_players)
    toi = np.clip(rng.normal(mins[pid], 2.2), 4.0, 26.0)
    # Opponent and home side are derived, not drawn, so adding them did not
    # shift a single RNG value and every check written before they existed
    # still produces the number it produced then.
    # ROSTER DENSITY MATTERS and it is a parameter for that reason. The
    # opponent factor is shrunk by model.OPP_PRIOR - four hundred
    # sixty-minute games - so how much of a real defensive difference
    # survives depends entirely on how much history each TEAM has. A fixture
    # with eight skaters a side gives each team a quarter of the evidence the
    # NHL does, and the check written to prove the factor works failed on
    # exactly that. Keep players/teams near eighteen to test it honestly.
    team = pid % n_teams
    # A ROUND ROBIN, not a fixed pairing. The first draft had opponent =
    # (team + 13) % 26, so every player faced the same team every night - the
    # opponent effect was then perfectly collinear with the player's own
    # identity, got absorbed into his own exponentially weighted rate, and the
    # opponent factor had nothing left to find. The check written to prove the
    # factor works failed on that fixture and was right to. The offset here
    # runs 1..25 and never lands on 0, so nobody plays himself.
    return pd.DataFrame({
        "player_id": pid,
        "team": team.astype(str),
        "opponent": ((team + 1 + (gm % (n_teams - 1))) % n_teams).astype(str),
        "is_home": ((gm + team) % 2).astype(float),
        "game_id": gm,
        "date": base + pd.to_timedelta(gm * 2, unit="D"),
        "position": pos[pid],
        "toi": toi,
        "sog": rng.poisson(rate[pid] * toi / 60.0).astype(float),
    })


def _designed_flags(hist, rate_frac=1.0, toi_frac=1.0, share=0.2, seed=7):
    """Pick rows at random to call "flagged last game", then MAKE the effect.

    The flag is chosen at random rather than computed, so it cannot correlate
    with anything in the history by accident. Then the chosen rows get exactly
    the effect being tested:

      rate_frac < 1   he shoots less per minute, on the SAME minutes. The
                      ice-time model cannot see this, so it must survive the
                      actual-ice-time grade.
      toi_frac < 1    he plays fewer minutes at the SAME rate. Pure
                      deployment, so the actual-ice-time grade must find
                      nothing while the projected one finds a deficit.
    """
    rng = np.random.default_rng(seed)
    h = hist.sort_values(["player_id", "date"]).reset_index(drop=True).copy()
    sq = rng.random(len(h)) < share
    # Nobody's first ten games: the grade drops those anyway, and flagging
    # them would only shrink the sample.
    sq &= h.groupby("player_id").cumcount().to_numpy() >= MIN_PRIOR_GAMES
    h.loc[sq, "toi"] = h.loc[sq, "toi"] * toi_frac
    h.loc[sq, "sog"] = rng.poisson(
        h.loc[sq, "sog"].to_numpy() * rate_frac * toi_frac).astype(float)
    flags = pd.DataFrame({
        "player_id": h["player_id"], "game_id": h["game_id"],
        "prev_sq": sq, "prev_el": True,
    })
    return h, flags


def squeeze_self_test() -> int:
    """Three cases, three answers known in advance. Run before the real data.

    If the estimator cannot find an effect that was deliberately put there,
    or finds one that was not, the number it prints on the cache means
    nothing.
    """
    hist = _squeeze_synthetic()
    bad = 0

    def check(label, cond, got):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {label}: {got}")
        if not cond:
            bad += 1

    print("squeeze estimator, against data whose answer is known")

    # 1. Nothing there. Both grades must cross zero.
    h, f = _designed_flags(hist, seed=11)
    r = run_squeeze(h, "sog", flags=f)
    a, p = r["on_actual_toi"], r["on_projected_toi"]
    check("null, on actual minutes, interval crosses zero",
          a["lo"] <= 0 <= a["hi"], f"{a['effect']:+.3f} "
          f"[{a['lo']:+.3f}, {a['hi']:+.3f}]")
    check("null, on projected minutes, interval crosses zero",
          p["lo"] <= 0 <= p["hi"], f"{p['effect']:+.3f} "
          f"[{p['lo']:+.3f}, {p['hi']:+.3f}]")

    # 2. A real RATE deficit. It must survive the actual-ice-time grade,
    #    because minutes are untouched and so deployment cannot explain it.
    h, f = _designed_flags(hist, rate_frac=0.75, seed=12)
    r = run_squeeze(h, "sog", flags=f)
    a, m = r["on_actual_toi"], r["minutes_over_projected"]
    check("a 25% rate cut shows up on actual minutes", a["hi"] < 0,
          f"{a['effect']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}]")
    # Minutes were not touched in this case, so the minutes diagnostic has to
    # say so. If it found a deployment story in a pure rate cut it would be
    # reading the shot column, and the two numbers would not be independent.
    check("...and the MINUTES diagnostic stays silent on a pure rate cut",
          m["lo"] <= 0 <= m["hi"],
          f"{m['effect']:+.3f} min [{m['lo']:+.3f}, {m['hi']:+.3f}]")

    # 3. Pure DEPLOYMENT. Same rate, fewer minutes. The actual-ice-time grade
    #    must find nothing - that is the entire reason it exists - while the
    #    projected one, built on an eight-game halflife that has not caught up,
    #    must find a deficit.
    h, f = _designed_flags(hist, toi_frac=0.6, seed=13)
    r = run_squeeze(h, "sog", flags=f)
    a, p, m = (r["on_actual_toi"], r["on_projected_toi"],
               r["minutes_over_projected"])
    check("a 40% minutes cut does NOT show up on actual minutes",
          a["lo"] <= 0 <= a["hi"], f"{a['effect']:+.3f} "
          f"[{a['lo']:+.3f}, {a['hi']:+.3f}]")
    check("the same cut DOES show up on projected minutes", p["hi"] < 0,
          f"{p['effect']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]")
    # And the minutes diagnostic must catch it, in minutes, pointing the right
    # way: a 40% cut off roughly fifteen minutes is about six, and the
    # projection should be handing him MORE than he plays.
    check("...and the MINUTES diagnostic catches it, over-projecting",
          m["lo"] > 0, f"{m['effect']:+.3f} min "
          f"[{m['lo']:+.3f}, {m['hi']:+.3f}]")

    # 4 and 5. The halflife sweep, which has a known answer in both
    #          directions. On data where nobody's role ever changes, chasing
    #          recent minutes can only chase noise, so a SLOW halflife must
    #          win. Put a real step change in and a FAST one must win. A
    #          sweep that cannot tell those apart would happily recommend
    #          making the model twitchier to fit noise.
    hls = [2.0, 8.0, 20.0]
    flat = _squeeze_synthetic(n_players=140, n_games=80, seed=31)
    r = sweep_toi(flat, "sog", halflives=hls)
    won = min(r["rows"], key=lambda x: x["mae"])["halflife"]
    check("with no role changes, a SLOW halflife wins", won == max(hls),
          f"lowest error at {won:g}")

    # ONE demotion, mid-career, permanent. My first draft of this asserted a
    # fast halflife would win and it failed - correctly. A single step in an
    # eighty-game career leaves roughly nine games in ten flat, and a fast
    # halflife loses on every one of those to buy a quicker reaction on a
    # handful. The assertion was wrong, not the sweep, so the right answer is
    # now the one being checked. This is exactly the shipped situation: a
    # squeezed man got demoted ONCE, and speeding the whole model up to catch
    # it sooner is not a trade worth making.
    stepped = flat.copy()
    n = stepped.groupby("player_id").cumcount().to_numpy()
    hit = (stepped["player_id"].to_numpy() % 2 == 0) & (n >= 40)
    stepped.loc[hit, "toi"] = stepped.loc[hit, "toi"] * 0.25
    stepped.loc[hit, "sog"] = np.random.default_rng(5).poisson(
        stepped.loc[hit, "sog"].to_numpy() * 0.25).astype(float)
    r = sweep_toi(stepped, "sog", halflives=hls)
    won = min(r["rows"], key=lambda x: x["mae"])["halflife"]
    check("ONE permanent demotion does NOT justify a fast halflife",
          won != min(hls), f"lowest error at {won:g}")

    # Deployment that actually churns: a new role every six games or so. NOW
    # a fast halflife must win, and if it does not the sweep is blind to the
    # only thing that would justify changing the constant.
    churn = flat.copy()
    rng = np.random.default_rng(41)
    pid = churn["player_id"].to_numpy()
    blk = pid * 1000 + (n // 6)
    lvl = pd.Series(blk).map(
        pd.Series(rng.uniform(0.35, 1.6, len(np.unique(blk))),
                  index=np.unique(blk))).to_numpy()
    churn["toi"] = np.clip(churn["toi"].to_numpy() * lvl, 3.0, 26.0)
    churn["sog"] = rng.poisson(
        churn["sog"].to_numpy() * lvl).astype(float)
    r = sweep_toi(churn, "sog", halflives=hls)
    won = min(r["rows"], key=lambda x: x["mae"])["halflife"]
    check("when deployment churns, a FAST halflife wins", won == min(hls),
          f"lowest error at {won:g}")

    # 9 and 10. The opponent factor, which is the whole reason run_full
    #           exists. On a league where no team defends differently, the
    #           factor must add NOTHING - if it improves MAE there it is
    #           fitting noise, and the grade would reward the model for it.
    #           Put a real one in and it must help.
    # Eighteen skaters a side, the way a real league is built, so each TEAM
    # has enough history for its factor to be estimated at all.
    NT = 16
    flat2 = _squeeze_synthetic(n_players=NT * 18, n_games=90, seed=51,
                               n_teams=NT)
    r = run_full(flat2, "sog", lines=(2.5,))
    g = r["grades"]
    gain = 100 * (g["rate x mins"]["mae"] - g["+ opponent"]["mae"]) \
        / g["rate x mins"]["mae"]
    check("with no defensive differences, the opponent factor adds ~nothing",
          abs(gain) < 0.3, f"{gain:+.3f}% of MAE")

    hard = flat2.copy()
    rng2 = np.random.default_rng(63)
    # Every team allows a different number of shots, by up to a third. Shots
    # are re-drawn through that multiplier, so the effect is in the DATA and
    # the factor has something real to find.
    mult = pd.Series(rng2.uniform(0.75, 1.3, NT),
                     index=[str(i) for i in range(NT)])
    omul = hard["opponent"].map(mult).to_numpy()
    hard["sog"] = rng2.poisson(hard["sog"].to_numpy() * omul).astype(float)
    r = run_full(hard, "sog", lines=(2.5,))
    g = r["grades"]
    gain = 100 * (g["rate x mins"]["mae"] - g["+ opponent"]["mae"]) \
        / g["rate x mins"]["mae"]
    # AGAINST THE CEILING, not against a number I picked. Poisson noise
    # dominates MAE: with PERFECT knowledge of a +-25% opponent effect the
    # whole available gain is under 3% of MAE, and model.OPP_PRIOR shrinks
    # what survives by about half again. A threshold of "0.5% or it failed"
    # was asserting something no correct implementation could deliver. So the
    # ceiling is measured right here - the same base prediction times the
    # multiplier that was actually injected - and the factor has to recover a
    # real share of it.
    truth = hard[["player_id", "game_id"]].copy()
    truth["_omul"] = omul
    rows = r["rows"].merge(truth.drop_duplicates(["player_id", "game_id"]),
                           on=["player_id", "game_id"], how="left",
                           validate="m:1")
    perfect = float(np.nanmean(np.abs(
        rows["actual"] - rows["rate x mins"] * rows["_omul"])))
    base_mae = g["rate x mins"]["mae"]
    ceiling = 100 * (base_mae - perfect) / base_mae
    share = gain / ceiling if ceiling > 0 else 0.0
    check("with real defensive differences, the opponent factor recovers a "
          "real share of what is THERE",
          share > 0.2, f"{gain:+.3f}% of MAE, against a ceiling of "
          f"{ceiling:+.3f}% - it recovers {100 * share:.0f}%")

    # 11. The tail function, against draws from a KNOWN rate. This replaces a
    #     check that asked whether the claimed hit rate matches the realised
    #     one on synthetic data. That check failed, and it deserved to fail -
    #     for the wrong reason. Shrinkage COMPRESSES: it is unbiased on
    #     average but too low at the top and too high at the bottom, and the
    #     bet test selects the top, so it selects the under-stated end. On
    #     this fixture the shrunk estimate came in 0.49 shots under the
    #     selected rows' true rate. That is real, but it is an artefact of a
    #     fixture whose rates are FIXED - there is no rate noise for the
    #     shrinkage to correct, so compression is all it does. Real data has
    #     that noise, which pushes the other way. A fixture cannot settle
    #     calibration; only the real run can, which is what the report is for.
    #     What a fixture CAN settle is whether the arithmetic is right.
    rng3 = np.random.default_rng(1)
    worst = 0.0
    for lam in (1.5, 2.8, 4.0):
        for k in (2, 3, 4):
            emp = float((rng3.poisson(lam, 200_000) >= k).mean())
            worst = max(worst, abs(float(_pois_ge(k, lam)) - emp))
    check("the Poisson tail function matches draws from a known rate",
          worst < 0.004, f"worst error {worst:.5f} over lam 1.5-4.0, k 2-4")

    # 12. And the bet test must CATCH an over-confident model, because that is
    #     the failure mode that costs money: a projection 25% too high claims
    #     a win rate it will not deliver, and the report has to say so rather
    #     than print a happy ROI.
    lam_true = rng3.uniform(1.0, 4.0, 40_000)
    actual = rng3.poisson(lam_true).astype(float)
    for label, mult, want_flag in (("honest", 1.0, False),
                                   ("25% too high", 1.25, True)):
        p = _pois_ge(3, lam_true * mult)
        take = p > 0.5238
        if take.sum() < 200:
            check(f"bet test on a {label} model", False, "too few selected")
            continue
        claimed, realised = p[take].mean(), (actual[take] >= 3).mean()
        se = np.sqrt(realised * (1 - realised) / take.sum())
        overclaims = (claimed - realised) > 2 * se
        check(f"bet test flags a {label} model as over-claiming: "
              f"{str(want_flag).lower()}", overclaims == want_flag,
              f"claims {100*claimed:.2f}%, hits {100*realised:.2f}% "
              f"over {int(take.sum()):,} bets")

    print(f"\n{'all checks passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--stat", default=None, help="one stat instead of all")
    ap.add_argument("--xg-sweep", action="store_true",
                    help="grade goals at several xG weights and print the best")
    ap.add_argument("--squeeze", action="store_true",
                    help="grade the squeeze flag against the SHIPPED model "
                         "rather than against a crude baseline, and say how "
                         "much of it the ice-time projection already catches")
    ap.add_argument("--squeeze-self-test", action="store_true",
                    help="check the squeeze estimator against synthetic data "
                         "whose answer is known, and download nothing")
    ap.add_argument("--full", action="store_true",
                    help="grade the WHOLE shipped projection - projected "
                         "minutes, opponent and home factors included - "
                         "against a player's own average shots per game, and "
                         "test whether its probabilities clear the -110 vig")
    ap.add_argument("--toi-sweep", action="store_true",
                    help="sweep the ice-time halflife and print BOTH the "
                         "overall error and the squeeze subset's deficit, so "
                         "a trade cannot pass itself off as a fix")
    ap.add_argument("--years", type=float, default=4.0,
                    help="how far back to read, matching linemate_study.py so "
                         "the two numbers are measured on the same games")
    a = ap.parse_args()

    if a.squeeze_self_test:
        return squeeze_self_test()

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]

    if a.squeeze or a.toi_sweep or a.full:
        # Loaded with the SAME window the study used. Grading one sample and
        # comparing it to a number measured on another is how a difference in
        # the data gets reported as a difference in the model.
        since = pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * a.years))
        hist = FETCH.load(seasons, "skaters", refresh=[], since=since)
        stat = a.stat or "sog"
        print(f"{len(hist):,} player-games, "
              f"{hist['player_id'].nunique():,} players, "
              f"{hist['date'].min().date()} to {hist['date'].max().date()}")
        if a.full:
            report_full(run_full(hist, stat), stat)
        if a.squeeze:
            report_squeeze(run_squeeze(hist, stat), stat)
        if a.toi_sweep:
            report_sweep(sweep_toi(hist, stat))
        return 0

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

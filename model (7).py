"""The projection.

The shape of it, in one paragraph: every counting stat is projected as a RATE
PER SIXTY MINUTES multiplied by projected ice time, because ice time is the
opportunity and the rate is the skill, and the two move for completely
different reasons. A fourth-liner promoted to the top six does not become a
better player, he becomes a player with more minutes, and a model that
projects "shots per game" directly cannot tell those apart. Each rate is then
shrunk toward the average for his position by how much ice time it was
measured over, adjusted for what the opponent actually allows, and adjusted a
little for home ice.

Three things worth knowing before reading a number off this
-----------------------------------------------------------
**These are expectations, not predictions.** A projection of 0.35 goals does
not mean a third of a goal. It means: usually none, sometimes one. That is
why every scoring line also carries P(≥1), which is the number a human can
actually use.

**Shrinkage is doing most of the work on goals and assists.** Nine games of
shooting tells you almost nothing about a shooting percentage, so the model
mostly does not believe it. A hot player's projection will look lower than
his recent numbers, and that is the model working, not failing.

**Expected goals carry more weight than goals.** A goal is one bounce; xG is
the chance that produced it, and it predicts the next game materially better.
`validate.py` measures this on your own cached history rather than asking you
to take it on faith.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("model")

# How fast the past stops counting, in games. Ten means a game ten starts ago
# is worth half of last night's.
HALFLIFE_RATE = 20.0        # rates are skill; they move slowly
HALFLIFE_TOI = 8.0          # deployment is a coach's decision; it moves fast

# Prior strength, in units of sixty-minute games. Bigger = the model believes
# a player's own history less. These are set by how noisy each stat is, not
# fitted - a fitted value on one season would be its own kind of overfit, and
# `validate.py` is there to show what each one is worth.
PRIOR = {
    "sog":      4.0,
    "goals":    14.0,       # the noisiest thing a skater does
    "xg":       6.0,        # the same event, measured with far less noise
    "assists":  11.0,
    # The projection DERIVES points by adding goals and assists rather than
    # modelling them separately. This entry exists so validate.py can grade
    # points as a rate of its own and see whether the derived number is worse
    # than one fitted directly - if it ever is, that is worth knowing.
    "points":   12.0,
    "blocks":   5.0,
    "hits":     5.0,
    "pim":      9.0,
    "takeaways": 7.0,
    "giveaways": 7.0,
    # Twice the sample of shots on goal, so it needs less help.
    "attempts": 3.0,
    # On-ice expected goals is a team-level number the player is only one
    # fifth of, so it is far less noisy than anything individual.
    "onice_xg": 4.0,
}
PRIOR_TOI = 3.0             # in games

# How much of the goal projection comes from expected goals rather than from
# goals. A judgement, stated out loud, and measurable with validate.py.
XG_WEIGHT = 0.65

# The same trade in two more places: swap a noisy measurement for a quieter
# one of the same quantity. Shots on goal from shot ATTEMPTS (twice the
# sample, and the fraction reaching the net is a stable player trait), and
# assists from the team's expected goals WHILE HE IS ON THE ICE (an assist
# needs a team-mate to score while you are out there; individual assist rate
# is mostly noise). Both are blends rather than replacements, because the
# direct measurement is not worthless - it is just noisy.
ATTEMPT_WEIGHT = 0.5
ONICE_WEIGHT = 0.45
# How many attempts of league-average accuracy to add before believing a
# player's own on-goal fraction. Without a prior of its own this substitution
# is algebraically the identity - see the note in `refine`.
REACH_PRIOR = 60.0
# The league fraction, used only when the history cannot supply one.
LEAGUE_ON_GOAL = 0.52

# Prior strength for the special-teams buckets, relative to all situations.
# Power-play ice time is about two and a half minutes a night, so sixty games
# is roughly ONE sixty-minute game of evidence; at full prior strength every
# top-unit forward would shrink to the same league rate and the whole
# top-unit lift would come from minutes rather than from skill. A judgement,
# not a fitted value, and written here so it can be argued with.
BUCKET_PRIOR_SCALE = {"rest": 1.0, "pp": 0.30, "pk": 0.30}

# Opponent and venue adjustments are real but small, and an unclipped
# multiplicative factor built from a few hundred games will happily claim a
# team allows 40% more blocks than the league. Clipped to what a season of
# hockey can actually support.
OPP_CLIP = (0.85, 1.15)
HOME_CLIP = (0.93, 1.07)
OPP_PRIOR = 400.0           # sixty-minute games before a team is believed
OPP_WINDOW_DAYS = 400       # teams change; ancient defence is not this defence

SV_PRIOR_SHOTS = 1000.0     # a goalie needs a LOT of shots to prove a save%
GOALIE_MINUTES = 60.0       # projections are "if he plays the game"
# How recent "recently" is for the games-played flag on each row. This is a
# LABEL, never a filter - who is listed is decided in project.py, which has
# to span an off-season.
FRESH_DAYS = 30

POSITION_GROUP = {"C": "C", "L": "W", "R": "W", "LW": "W", "RW": "W",
                  "W": "W", "D": "D", "G": "G"}


def group_of(pos) -> str:
    p = str(pos or "").upper().split("/")[0].strip()
    return POSITION_GROUP.get(p, "W")


# ------------------------------------------------------ weighted histories
def ew_weights(hist: pd.DataFrame, halflife: float) -> pd.Series:
    """One weight per row: 0.5 ** (games ago / halflife).

    Counted in GAMES rather than days on purpose. A player who misses three
    weeks has not become less predictable, he has simply not added evidence,
    and decaying by the calendar would quietly punish him for being injured.
    """
    h = hist.sort_values(["player_id", "date"])
    n = h.groupby("player_id")["date"].transform("size")
    rank = h.groupby("player_id").cumcount()
    ago = (n - 1 - rank).astype(float)
    return pd.Series(0.5 ** (ago / float(halflife)), index=h.index)


def weighted_sums(hist: pd.DataFrame, cols: list[str],
                  halflife: float) -> pd.DataFrame:
    """Per player: the exponentially weighted sum of each column, and of TOI.

    Sums rather than means, because the ratio of two weighted sums IS the
    weighted rate, and the weighted TOI sum is exactly the sample size the
    shrinkage needs. Carrying means instead would throw the sample size away
    and the shrinkage would have to guess it.
    """
    h = hist.sort_values(["player_id", "date"]).copy()
    w = ew_weights(h, halflife)
    out = pd.DataFrame(index=sorted(h["player_id"].unique()))
    acc = {}
    for c in cols + ["toi"]:
        if c not in h.columns:
            continue
        acc["s_" + c] = (h[c].fillna(0) * w).groupby(h["player_id"]).sum()
    acc["w_games"] = w.groupby(h["player_id"]).sum()
    for k, v in acc.items():
        out[k] = v
    return out.fillna(0.0)


def shrink_rate(stat_sum: pd.Series, toi_sum: pd.Series,
                prior_rate: pd.Series, k: float) -> pd.Series:
    """Poisson-gamma shrinkage, in sixty-minute units.

    `k` is "how many sixty-minute games of league-average evidence to add".
    With ten games of real ice time and k=14, the model is still believing the
    prior more than the player - which is correct for goals, and would be
    absurd for ice time.
    """
    games = toi_sum / 60.0
    return (stat_sum + k * prior_rate) / (games + k)


# ------------------------------------------------- league and opponent shape
def position_priors(hist: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """League rate per sixty, by position group. The thing everyone is
    shrunk toward, and the reason a defenceman is not expected to shoot like
    a first-line winger."""
    h = hist.copy()
    h["grp"] = h["position"].map(group_of)
    toi = h.groupby("grp")["toi"].sum()
    out = {}
    for c in cols:
        if c in h.columns:
            out[c] = 60.0 * h.groupby("grp")[c].sum() / toi.replace(0, np.nan)
    pri = pd.DataFrame(out).fillna(0.0)
    log.info("position priors per 60:\n%s", pri.round(3).to_string())
    return pri


def bucket_priors(hist: pd.DataFrame, cols: list[str], b: str) -> pd.DataFrame:
    """League rate per sixty AT ONE GAME STATE, by position group.

    A separate prior per state matters more than it looks. Power-play scoring
    rates are roughly triple even-strength ones, so shrinking a player's
    power-play rate toward his all-situations position average would pull
    every top-unit forward down by two thirds.
    """
    h = hist.copy()
    h["grp"] = h["position"].map(group_of)
    toi = h.groupby("grp")[f"{b}_toi"].sum()
    out = {}
    for c in cols:
        col = f"{b}_{c}"
        if col in h.columns:
            out[c] = 60.0 * h.groupby("grp")[col].sum() / toi.replace(0, np.nan)
    return pd.DataFrame(out).fillna(0.0)


def bucket_frame(hist: pd.DataFrame, cols: list[str], b: str) -> pd.DataFrame:
    """One game state's columns, renamed to look like a whole history.

    Built as a new frame rather than by renaming in place: a rename would
    collide with the all-situations column of the same name and pandas would
    hand back two columns called `goals`, one of which is silently the wrong
    one.
    """
    ren = {f"{b}_{c}": c for c in list(cols) + ["toi"]
           if f"{b}_{c}" in hist.columns}
    keep = ["player_id", "date"] + list(ren)
    return hist[keep].rename(columns=ren)


def opponent_factors(hist: pd.DataFrame, cols: list[str],
                     asof: pd.Timestamp) -> pd.DataFrame:
    """How much of each stat a team lets the other side produce, relative to
    the league. Built from what opponents DID against them, which is the only
    direct measurement available."""
    cut = asof - pd.Timedelta(days=OPP_WINDOW_DAYS)
    h = hist[hist["date"] >= cut]
    if not len(h):
        h = hist
    toi = h.groupby("opponent")["toi"].sum()
    out = {}
    for c in cols:
        if c not in h.columns:
            continue
        league = 60.0 * h[c].sum() / max(h["toi"].sum(), 1e-9)
        allowed = 60.0 * h.groupby("opponent")[c].sum() / toi.replace(0, np.nan)
        games = toi / 60.0
        shrunk = (allowed * games + league * OPP_PRIOR) / (games + OPP_PRIOR)
        out[c] = (shrunk / league).clip(*OPP_CLIP)
    return pd.DataFrame(out).fillna(1.0)


def home_factors(hist: pd.DataFrame, cols: list[str]) -> dict:
    """Home ice, measured rather than assumed. It is small for most stats and
    not small for hits, which is a scorer-bias artefact as much as a real
    effect - clipping keeps either interpretation from running away."""
    out = {}
    home = hist[hist["is_home"] > 0.5]
    away = hist[hist["is_home"] <= 0.5]
    for c in cols:
        if c not in hist.columns or not len(home) or not len(away):
            out[c] = (1.0, 1.0)
            continue
        rh = home[c].sum() / max(home["toi"].sum(), 1e-9)
        ra = away[c].sum() / max(away["toi"].sum(), 1e-9)
        mid = (rh + ra) / 2.0
        if mid <= 0:
            out[c] = (1.0, 1.0)
            continue
        out[c] = (float(np.clip(rh / mid, *HOME_CLIP)),
                  float(np.clip(ra / mid, *HOME_CLIP)))
    return out


# ------------------------------------------------------------- the skaters
def project_skaters(hist: pd.DataFrame, roster: pd.DataFrame,
                    asof: pd.Timestamp) -> pd.DataFrame:
    """One row per skater on the slate.

    `roster` carries player_id, name, team, position, opponent, is_home.
    """
    stats = [c for c in ("sog", "goals", "assists", "blocks", "hits", "pim",
                         "takeaways", "giveaways", "xg", "attempts",
                         "onice_xg")
             if c in hist.columns]
    # The three buckets a game divides into. `rest` is everything that is not
    # a power play or a penalty kill, computed by subtraction upstream, so the
    # three add back to the total exactly.
    #
    # `rest_toi` is the signal that this cache was built by the current
    # tidier. An OLDER cache carries `pp_toi` and nothing else, and taking
    # that as a bucket would be catastrophic rather than merely wrong: the
    # projection would be the sum over buckets, the only bucket would be the
    # power play, every power-play stat column would be missing and therefore
    # zero, and the page would fill with plausible-looking near-zeroes. So the
    # split is all-or-nothing, and the fallback is the previous behaviour.
    buckets = []
    if "rest_toi" in hist.columns:
        buckets = [b for b in ("rest", "pp", "pk") if f"{b}_toi" in hist.columns]
        missing = [c for c in ("goals", "assists", "sog")
                   if f"rest_{c}" not in hist.columns]
        if missing:
            log.error("the cache has game-state ice time but not game-state "
                      "%s, so the split is being IGNORED rather than half "
                      "applied", ", ".join(missing))
            buckets = []
    else:
        log.warning("this history predates the game-state split - every rate "
                    "is an all-situations blend, so a promotion to the top "
                    "power-play unit is invisible. Re-run the fetch to fix.")

    rates = weighted_sums(hist, stats, HALFLIFE_RATE)
    bucket_rates = {b: weighted_sums(bucket_frame(hist, stats, b), stats,
                                     HALFLIFE_RATE) for b in buckets}
    toi_w = weighted_sums(hist, [f"{b}_toi" for b in buckets], HALFLIFE_TOI)
    pri = position_priors(hist, stats)
    bucket_pri = {b: bucket_priors(hist, stats, b) for b in buckets}
    opp = opponent_factors(hist, stats, asof)
    hf = home_factors(hist, stats)

    # How many assists a skater picks up per expected goal his team scores
    # while he is on the ice. By position, because a centre and a defenceman
    # standing on the same ice for the same goal do not collect assists at the
    # same rate. Measured rather than assumed.
    # The league's own on-goal fraction, measured rather than assumed.
    league_reach = LEAGUE_ON_GOAL
    if ({"attempts", "sog"} <= set(hist.columns)
            and float(hist["attempts"].sum()) > 0):
        league_reach = float(hist["sog"].sum() / hist["attempts"].sum())
        log.info("league shots-on-goal per attempt: %.3f", league_reach)

    a_per_xg = {}
    if "onice_xg" in stats:
        hh = hist.copy()
        hh["grp"] = hh["position"].map(group_of)
        num = hh.groupby("grp")["assists"].sum()
        den = hh.groupby("grp")["onice_xg"].sum().replace(0, np.nan)
        a_per_xg = (num / den).fillna(0.30).to_dict()
        log.info("assists per on-ice expected goal, by position: %s",
                 {k: round(v, 3) for k, v in a_per_xg.items()})

    h = hist.copy()
    h["grp"] = h["position"].map(group_of)
    toi_prior = h.groupby("grp")["toi"].mean()

    out = roster.copy()
    out["grp"] = out["position"].map(group_of)
    idx = out["player_id"]

    def col(frame, name, default=0.0):
        return frame[name].reindex(idx).fillna(default).to_numpy() \
            if name in frame.columns else np.full(len(out), default)

    games = col(rates, "w_games")
    toi_sum = col(rates, "s_toi")
    games_fast = col(toi_w, "w_games")

    # Projected ice time, per bucket. The prior is the position average and
    # three games of it is deliberately weak: a coach's deployment is the most
    # observable thing about a player and the least in need of a prior. Power
    # play minutes get their own projection because they are the thing that
    # actually moves - a promotion to the top unit is a coach's decision worth
    # more to a scoring projection than anything the player himself does.
    prior_toi = out["grp"].map(toi_prior).fillna(14.0).to_numpy()
    out["toi"] = ((col(toi_w, "s_toi") + PRIOR_TOI * prior_toi)
                  / (games_fast + PRIOR_TOI))
    bt = {}
    for b in buckets:
        pb = h.groupby("grp")[f"{b}_toi"].mean()
        prior_b = out["grp"].map(pb).fillna(0.5).to_numpy()
        bt[b] = ((col(toi_w, f"s_{b}_toi") + PRIOR_TOI * prior_b)
                 / (games_fast + PRIOR_TOI))
    if bt:
        # The buckets are projected independently, so nothing forces them to
        # add to the total. Rescale so they do - otherwise a player can be
        # given nineteen minutes in the TOI column and twenty-one across the
        # three states he could have been on the ice in.
        tot = sum(bt.values())
        scale = np.where(tot > 0, out["toi"].to_numpy() / np.maximum(tot, 1e-9), 0.0)
        for b in bt:
            bt[b] = bt[b] * scale
    # WRITTEN OUT AFTER THE RESCALE, not before. `out["pp_toi"] = bt["pp"]`
    # copies the array into the frame, so rebinding bt["pp"] afterwards does
    # not update it - the page would show one set of minutes while the
    # projection used another. Harmless while the rescale is a no-op, which is
    # exactly why it would go unnoticed until it stopped being one.
    if "pp" in bt:
        out["pp_toi"] = bt["pp"]
    if "pk" in bt:
        out["pk_toi"] = bt["pk"]

    out["games_used"] = np.round(games, 1)
    out["toi_minutes_used"] = np.round(toi_sum, 0)

    def rates_from(frame, priors, k_scale=1.0):
        # WHICH COLUMNS THIS FRAME ACTUALLY HAS, not which ones were asked
        # for. `col()` returns zeros for a column that is not there, so every
        # key below exists whether or not there is any data behind it - and
        # `refine` used to test `if "xg" in per60`, which was therefore always
        # true. A history whose per-state xG columns were missing had its
        # goals blended against a column of zeros and came out at exactly
        # 1 - XG_WEIGHT of the truth: 35%. Nothing raised, nothing logged,
        # and a 35% goal projection looks like a cautious goal projection.
        have = {c for c in stats
                if ("s_" + c) in frame.columns
                and float(np.abs(frame["s_" + c]).sum()) > 0}
        per60 = {}
        for c in stats:
            pr = out["grp"].map(priors[c] if c in priors
                                else pd.Series(dtype=float))
            per60[c] = shrink_rate(pd.Series(col(frame, "s_" + c)),
                                   pd.Series(col(frame, "s_toi")),
                                   pd.Series(pr.fillna(0.0).to_numpy()),
                                   PRIOR[c] * k_scale).to_numpy()
        return refine(per60, have, frame)

    def refine(per60, have, frame):
        """The three substitutions that trade a noisy measurement for a
        quieter one measuring the same thing.

        Blended PER PLAYER, not per column. A name-only guard was not enough:
        `weighted_sums` creates `s_xg` whenever the column exists and then
        fills NaN with zero, so a column that is present but empty is
        indistinguishable from one full of real data - and a cache written
        before xG existed, concatenated beside a current one, gives half the
        rows NaN and the other half data. That reproduced the original defect
        almost exactly (goals at 64% of truth) while passing a column check.
        Asking "does THIS PLAYER have any of it" is the guard that holds.
        """
        def blend(target, sub, weight, scale=None):
            if not {target, sub} <= have:
                return
            evidence = col(frame, "s_" + sub) > 0
            w = np.where(evidence, weight, 0.0)
            sub_rate = per60[sub] if scale is None else per60[sub] * scale
            per60[target] = w * sub_rate + (1.0 - w) * per60[target]

        # Goals from expected goals, mostly. Both are shrunk first and then
        # blended; blending raw and shrinking after would let one hot week of
        # real goals drag the result.
        blend("goals", "xg", XG_WEIGHT)

        # Shots from shot ATTEMPTS times the fraction that reach the net.
        #
        # The fraction must be SHRUNK ON ITS OWN, and the first version was
        # worthless because it was not. Taking reach as the player's shrunk
        # shots over his shrunk attempts and multiplying back by attempts is
        # algebraically `sog` again: the two cancel exactly, the blend
        # collapses to the identity, and the only thing the code did was
        # penalise accurate shooters when the clip bound.
        if {"attempts", "sog"} <= have:
            att = col(frame, "s_attempts")
            sog = col(frame, "s_sog")
            reach = ((sog + REACH_PRIOR * league_reach)
                     / np.maximum(att + REACH_PRIOR, 1e-9))
            blend("sog", "attempts", ATTEMPT_WEIGHT,
                  scale=np.clip(reach, 0.25, 0.90))

        # Assists from the team's expected goals WHILE HE IS ON THE ICE. An
        # assist needs a team-mate to score while you are out there, and this
        # measures that directly - individual assist rate is mostly noise.
        if a_per_xg:
            blend("assists", "onice_xg", ONICE_WEIGHT,
                  scale=out["grp"].map(a_per_xg).fillna(0.30).to_numpy())
        return per60

    per60 = rates_from(rates, pri)
    # A player accumulates about two and a half power-play minutes a night, so
    # sixty games is roughly ONE sixty-minute game of power-play evidence.
    # Against PRIOR["goals"] of 14 that is ~92% prior, which would make every
    # top-unit forward identical and leave the whole top-unit lift coming from
    # minutes rather than from skill. The prior is scaled down for the
    # special-teams buckets so their own state-specific priors can still be
    # moved by real evidence. This number is a judgement, not a fitted value.
    per60_b = {b: rates_from(bucket_rates[b], bucket_pri[b],
                             k_scale=BUCKET_PRIOR_SCALE.get(b, 1.0))
               for b in buckets}

    share = out["toi"].to_numpy() / 60.0
    ohome = out["is_home"].to_numpy() > 0.5
    for c in stats:
        if c in ("xg", "attempts", "onice_xg"):
            continue
        of = out["opponent"].map(opp[c] if c in opp else pd.Series(dtype=float))
        of = of.fillna(1.0).to_numpy()
        hfac = np.where(ohome, hf.get(c, (1.0, 1.0))[0], hf.get(c, (1.0, 1.0))[1])
        # Per stat, not per cache: a history can carry game-state goals and
        # no game-state hits, and a stat whose state columns are missing must
        # fall back to the all-situations rate rather than be summed over
        # buckets that are all zero. That failure is silent and total - the
        # column simply reads 0.00 for every player, which looks like a
        # projection rather than like a bug.
        splittable = bool(bt) and all(
            ("s_" + c) in bucket_rates[b].columns for b in buckets)
        if splittable:
            # The whole point of the split: each state's own rate against that
            # state's own minutes, summed. A winger with three minutes on the
            # top power-play unit is scored at his power-play rate for those
            # three minutes, not at a blend that a fourth-liner also gets.
            base = sum(per60_b[b][c] * bt[b] / 60.0 for b in buckets)
        else:
            if bt:
                log.info("%s has no game-state columns; using the "
                         "all-situations rate for it", c)
            base = per60[c] * share
        out[c] = np.maximum(base * of * hfac, 0.0)

    if "goals" in out and "assists" in out:
        out["points"] = out["goals"] + out["assists"]

    # P(at least one), Poisson. Not exactly right - a hat-trick game is not
    # three independent goals - but close enough at these means to be the most
    # useful number on the row, and far more honest than reporting 0.31 goals
    # and letting someone read it as a third of a goal.
    for c in ("goals", "assists", "points"):
        if c in out:
            out["p_" + c] = 1.0 - np.exp(-out[c].clip(lower=0))

    # THE GAP BETWEEN P(point) AND P(goal), which is not a vague spread but an
    # exact probability of its own:
    #
    #   P(PT) - P(G) = e^-lg - e^-(lg+la)
    #                = e^-lg * (1 - e^-la)
    #                = P(no goal) x P(at least one assist)
    #
    # That is the chance he records a point WITHOUT scoring - an assist-only
    # night. Verified against four million simulated games: 0.2704 predicted,
    # 0.2703 observed.
    #
    # It separates the two ways a skater reaches the scoresheet, which the two
    # columns on their own do not: a winger at P(G) 27% / P(PT) 54% and a
    # playmaking centre at 8% / 50% have similar point equity bought in
    # completely different currencies, and only one of them is a bet on
    # finishing.
    if "p_points" in out and "p_goals" in out:
        out["p_assist_only"] = (out["p_points"] - out["p_goals"]).clip(lower=0)

    last = hist.groupby("player_id")["date"].max()
    out["last_played"] = out["player_id"].map(last)
    recent = hist[hist["date"] >= asof - pd.Timedelta(days=FRESH_DAYS)]
    out["gp_30d"] = out["player_id"].map(
        recent.groupby("player_id")["date"].size()).fillna(0).astype(int)

    out = recent_form(hist, out, asof)
    return out


# ------------------------------------------- last night, and whose night it was
# How many games of league-average slicing to add before believing a player's
# own share of his team's shooting. Five, because a share is a ratio of two
# small counts and one game of it means almost nothing.
SHARE_PRIOR = 5.0


def recent_form(hist: pd.DataFrame, out: pd.DataFrame,
                asof: pd.Timestamp) -> pd.DataFrame:
    """What happened in his last game, and how big a slice of his team's
    shooting it was.

    THESE ARE FACTS, NOT A FORECAST, and the distinction is the whole point of
    putting them in their own columns rather than folding them into the
    projection. The projection already knows what a player's shot rate is and
    already declines to believe one game of it - that is what the shrinkage is
    for. What it cannot tell you is whether last night was quiet because HE was
    quiet or because a team-mate took the night's shooting, and those look
    identical in a rate.

    The share is computed against his TEAM's total rather than his line's,
    because the cache has no line data. A team denominator is blunter - thirty
    shots across eighteen men rather than ten across three - but it is built
    from numbers already on disk and needs no new source, and it moves for the
    same reason: there is one puck.

    No claim is made here that a squeezed man bounces back. That is a testable
    proposition and `linemate_study.py` tests it on this very cache. Until it
    says otherwise these columns are context, not signal.
    """
    for c in ("last_sog", "last_share", "share_norm", "mate_lift"):
        out[c] = np.nan
    if "sog" not in hist.columns or not len(hist):
        return out

    h = hist[hist["date"] < asof].copy()
    if not len(h):
        return out
    h = h.sort_values(["player_id", "date"])

    # His team's whole night, from the same rows. Every skater who dressed is
    # in this cache, so the denominator is the real total rather than a sample
    # of it.
    team_tot = (h.groupby(["team", "game_id"])["sog"].sum()
                  .rename("team_sog").reset_index())
    h = h.merge(team_tot, on=["team", "game_id"], how="left")
    h["share"] = h["sog"] / h["team_sog"].replace(0, np.nan)

    # 0 is his most recent game, and the norms are built from everything
    # BEFORE it. A night inside the average it is being compared against pulls
    # that average toward itself, so the gap collapses exactly when it is most
    # interesting - an eighteen-game sample would hide a sixth of its own
    # signal, and a five-game one a fifth.
    h["ago"] = h.groupby("player_id").cumcount(ascending=False)
    prior = h[h["ago"] > 0]
    if not len(prior):
        return out

    # BOTH CONSTANTS FROM THE PRIOR ROWS ONLY. They are league-wide and a
    # single player barely moves them, but "barely" is not "not at all": built
    # from every row they would include the very game being described, so the
    # norm a night is compared against would contain a trace of that night.
    # A leak small enough to argue about is a leak nobody will ever find.
    league_share = float(prior["sog"].sum()
                         / max(prior["team_sog"].sum(), 1e-9))
    team_night = float(prior["team_sog"].mean()) if len(prior) else 30.0
    sums = prior.groupby("player_id").agg(
        s_sog=("sog", "sum"), s_team=("team_sog", "sum"), n=("sog", "size"))
    share_norm = ((sums["s_sog"] + SHARE_PRIOR * league_share * team_night)
                  / (sums["s_team"] + SHARE_PRIOR * team_night))
    sog_norm = prior.groupby("player_id")["sog"].mean()

    # WHOSE night was it. A team-mate's shots measured against his OWN usual
    # number, so a fourth-liner with three is a bigger event than a first-line
    # winger with four - which is what "somebody else took them" means.
    h["lift"] = h["sog"] / h["player_id"].map(sog_norm).replace(0, np.nan)
    top2 = (h.dropna(subset=["lift"]).sort_values("lift", ascending=False)
             .groupby(["team", "game_id"]).head(2))
    agg = top2.groupby(["team", "game_id"])["lift"].agg(
        l1="first", l2="last", n="size").reset_index()
    # With one qualified skater there is no SECOND best, and "first" and
    # "last" of a one-row group are the same row - which would hand a player
    # his own lift as his team-mate's.
    agg.loc[agg["n"] < 2, "l2"] = np.nan
    # Taken from `h` AFTER the lift column exists, not from a slice made
    # before it - a slice is a copy, and a column added to the parent
    # afterwards is simply not on it.
    last = h[h["ago"] == 0].merge(agg.drop(columns=["n"]),
                                  on=["team", "game_id"], how="left")
    own = last["lift"].to_numpy()
    mate = np.where(np.isclose(np.nan_to_num(own, nan=-1.0),
                               np.nan_to_num(last["l1"].to_numpy(), nan=-2.0)),
                    last["l2"].to_numpy(), last["l1"].to_numpy())
    last = last.assign(mate_lift=mate).set_index("player_id")

    idx = out["player_id"]
    out["last_sog"] = idx.map(last["sog"]).to_numpy()
    out["last_share"] = idx.map(last["share"]).to_numpy()
    out["share_norm"] = idx.map(share_norm).to_numpy()
    out["mate_lift"] = idx.map(last["mate_lift"]).to_numpy()
    return out


# ------------------------------------------------------------- the goalies
def team_rates(skaters: pd.DataFrame, goalies: pd.DataFrame,
               asof: pd.Timestamp) -> dict:
    """Shots generated and shots allowed, per team per game.

    Shots faced by a goalie are two things at once: how much his own side
    gives up, and how much the other side generates. Modelling it from the
    goalie's own history alone confuses the two - a good goalie on a bad team
    looks like a high-volume goalie, and then gets projected as one against
    an opponent who does not shoot.
    """
    cut = asof - pd.Timedelta(days=OPP_WINDOW_DAYS)
    sk = skaters[skaters["date"] >= cut]
    go = goalies[goalies["date"] >= cut]
    if not len(sk):
        sk = skaters
    if not len(go):
        go = goalies

    sf = sk.groupby("team")["sog"].sum()
    gp_for = sk.groupby("team")["game_id"].nunique()
    for_pg = (sf / gp_for.replace(0, np.nan)).dropna()

    sa = go.groupby("team")["shots_against"].sum()
    gp_ag = go.groupby("team")["game_id"].nunique()
    ag_pg = (sa / gp_ag.replace(0, np.nan)).dropna()

    league_for = float(for_pg.mean()) if len(for_pg) else 30.0
    league_ag = float(ag_pg.mean()) if len(ag_pg) else 30.0

    # These are two estimates of ONE number: in a closed league, shots taken
    # and shots faced are the same total counted from opposite ends. They
    # differ only because the two caches cover different player sets - and
    # `game_by_game` logs a failed player fetch and carries on, so a few
    # percent of missing skater logs would drag the skater-side estimate down
    # and quietly move every goalie projection with it.
    #
    # The baseline therefore comes from the GOALIE side, which measures shots
    # faced directly rather than inferring them, and the gap between the two
    # is reported because it is a coverage check nothing else performs.
    gap = abs(league_for - league_ag) / max(league_ag, 1e-9)
    if gap > 0.06:
        log.error("shots taken (%.1f/game, from skater logs) and shots faced "
                  "(%.1f/game, from goalie logs) disagree by %.0f%%. These are "
                  "the same quantity counted twice, so one of the two caches "
                  "is incomplete. Using the goalie side.",
                  league_for, league_ag, 100 * gap)
    else:
        log.info("shots taken %.1f/game vs shots faced %.1f/game (%.1f%% apart)",
                 league_for, league_ag, 100 * gap)

    # An NHL team puts about 30 shots on goal a night and has for decades. A
    # number far from that does not mean hockey changed, it means the history
    # is being counted wrongly - the classic cause being situation rows that
    # were never filtered, which quintuples every total while still producing
    # a table that looks entirely reasonable.
    mid = league_ag
    if not 22.0 <= mid <= 40.0:
        log.error("league shots per game came out at %.1f, which is not a "
                  "hockey number. Every goalie projection below is built on "
                  "it. The usual cause is unfiltered `situation` rows or a "
                  "game_id that is not unique per game.", mid)
    return {
        "for": (for_pg / league_for).clip(*OPP_CLIP).to_dict(),
        "against": (ag_pg / league_ag).clip(*OPP_CLIP).to_dict(),
        # THE GOALIE SIDE, not the mean of the two. Shots faced is what this
        # baseline is for, and the goalie logs measure it directly; the skater
        # side infers it and drifts with any missing player log. An earlier
        # version logged "using the goalie side" while still returning the
        # average, which is the worst of both - wrong number, reassuring log.
        "league_shots": league_ag,
    }


def project_goalies(goalies: pd.DataFrame, skaters: pd.DataFrame,
                    roster: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """One row per goalie on the slate, IF HE PLAYS THE WHOLE GAME.

    There is no attempt here to guess who starts. The NHL publishes no
    pre-game lineup card and no confirmed starter, so a model that guessed
    would be inventing the single largest fact on the page. Both goalies are
    listed, and which one dresses is something you know and this does not.
    """
    tr = team_rates(skaters, goalies, asof)

    w = ew_weights(goalies.sort_values(["player_id", "date"]), HALFLIFE_RATE)
    g = goalies.sort_values(["player_id", "date"])
    sv_shots = (g["shots_against"].fillna(0) * w).groupby(g["player_id"]).sum()
    sv_saves = (g["saves"].fillna(0) * w).groupby(g["player_id"]).sum()
    starts = g.groupby("player_id")["game_id"].nunique()

    league_sv = float(goalies["saves"].sum()
                      / max(goalies["shots_against"].sum(), 1e-9))

    out = roster.copy()
    idx = out["player_id"]
    shots_w = sv_shots.reindex(idx).fillna(0).to_numpy()
    saves_w = sv_saves.reindex(idx).fillna(0).to_numpy()
    out["save_pct"] = ((saves_w + SV_PRIOR_SHOTS * league_sv)
                       / (shots_w + SV_PRIOR_SHOTS))

    own_def = out["team"].map(tr["against"]).fillna(1.0).to_numpy()
    opp_off = out["opponent"].map(tr["for"]).fillna(1.0).to_numpy()
    out["shots_against"] = tr["league_shots"] * own_def * opp_off
    out["saves"] = out["shots_against"] * out["save_pct"]
    out["goals_against"] = out["shots_against"] * (1 - out["save_pct"])
    out["toi"] = GOALIE_MINUTES

    # A shutout is the whole game at this rate. Poisson on goals against.
    out["p_shutout"] = np.exp(-out["goals_against"].clip(lower=0))
    out["career_starts"] = out["player_id"].map(starts).fillna(0).astype(int)
    last = goalies.groupby("player_id")["date"].max()
    out["last_played"] = out["player_id"].map(last)
    recent = goalies[goalies["date"] >= asof - pd.Timedelta(days=FRESH_DAYS)]
    out["gp_30d"] = out["player_id"].map(
        recent.groupby("player_id")["date"].size()).fillna(0).astype(int)
    log.info("league save%% %.4f, league shots/game %.1f",
             league_sv, tr["league_shots"])
    return out

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
}
PRIOR_TOI = 3.0             # in games

# How much of the goal projection comes from expected goals rather than from
# goals. A judgement, stated out loud, and measurable with validate.py.
XG_WEIGHT = 0.65

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
                         "takeaways", "giveaways", "xg")
             if c in hist.columns]

    rates = weighted_sums(hist, stats, HALFLIFE_RATE)
    toi_w = weighted_sums(hist, ["pp_toi"], HALFLIFE_TOI)
    pri = position_priors(hist, stats)
    opp = opponent_factors(hist, stats, asof)
    hf = home_factors(hist, stats)

    # Projected ice time. The prior here is the position average, and three
    # games of prior is deliberately weak: a coach's deployment is the most
    # observable thing about a player and the least in need of a prior.
    h = hist.copy()
    h["grp"] = h["position"].map(group_of)
    toi_prior = h.groupby("grp")["toi"].mean()
    pp_prior = h.groupby("grp")["pp_toi"].mean() if "pp_toi" in h else None

    out = roster.copy()
    out["grp"] = out["position"].map(group_of)
    idx = out["player_id"]

    def col(frame, name, default=0.0):
        return frame[name].reindex(idx).fillna(default).to_numpy() \
            if name in frame.columns else np.full(len(out), default)

    games = col(rates, "w_games")
    toi_sum = col(rates, "s_toi")
    toi_sum_fast = col(toi_w, "s_toi")
    games_fast = col(toi_w, "w_games")
    pp_sum = col(toi_w, "s_pp_toi")

    prior_toi = out["grp"].map(toi_prior).fillna(14.0).to_numpy()
    out["toi"] = (toi_sum_fast + PRIOR_TOI * prior_toi) / (games_fast + PRIOR_TOI)
    if pp_prior is not None:
        prior_pp = out["grp"].map(pp_prior).fillna(0.8).to_numpy()
        out["pp_toi"] = (pp_sum + PRIOR_TOI * prior_pp) / (games_fast + PRIOR_TOI)

    out["games_used"] = np.round(games, 1)
    out["toi_minutes_used"] = np.round(toi_sum, 0)

    per60 = {}
    for c in stats:
        prior_c = out["grp"].map(pri[c] if c in pri else pd.Series(dtype=float))
        prior_c = prior_c.fillna(0.0).to_numpy()
        per60[c] = shrink_rate(pd.Series(col(rates, "s_" + c)),
                               pd.Series(toi_sum),
                               pd.Series(prior_c), PRIOR[c]).to_numpy()

    # Goals from expected goals, mostly. Both rates are shrunk first, then
    # blended - blending raw and then shrinking would let one hot week of real
    # goals drag the whole thing.
    if "xg" in per60:
        per60["goals"] = (XG_WEIGHT * per60["xg"]
                          + (1 - XG_WEIGHT) * per60["goals"])

    share = out["toi"].to_numpy() / 60.0
    ohome = out["is_home"].to_numpy() > 0.5
    for c in stats:
        if c == "xg":
            continue
        of = out["opponent"].map(opp[c] if c in opp else pd.Series(dtype=float))
        of = of.fillna(1.0).to_numpy()
        hfac = np.where(ohome, hf.get(c, (1.0, 1.0))[0], hf.get(c, (1.0, 1.0))[1])
        out[c] = np.maximum(per60[c] * share * of * hfac, 0.0)

    if "goals" in out and "assists" in out:
        out["points"] = out["goals"] + out["assists"]

    # P(at least one), Poisson. Not exactly right - a hat-trick game is not
    # three independent goals - but close enough at these means to be the most
    # useful number on the row, and far more honest than reporting 0.31 goals
    # and letting someone read it as a third of a goal.
    for c in ("goals", "assists", "points"):
        if c in out:
            out["p_" + c] = 1.0 - np.exp(-out[c].clip(lower=0))

    last = hist.groupby("player_id")["date"].max()
    out["last_played"] = out["player_id"].map(last)
    recent = hist[hist["date"] >= asof - pd.Timedelta(days=30)]
    out["gp_30d"] = out["player_id"].map(
        recent.groupby("player_id")["date"].size()).fillna(0).astype(int)

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

    # An NHL team puts about 30 shots on goal a night and has for decades. A
    # number far from that does not mean hockey changed, it means the history
    # is being counted wrongly - the classic cause being situation rows that
    # were never filtered, which quintuples every total while still producing
    # a table that looks entirely reasonable.
    mid = (league_for + league_ag) / 2.0
    if not 22.0 <= mid <= 40.0:
        log.error("league shots per game came out at %.1f, which is not a "
                  "hockey number. Every goalie projection below is built on "
                  "it. The usual cause is unfiltered `situation` rows or a "
                  "game_id that is not unique per game.", mid)
    return {
        "for": (for_pg / league_for).clip(*OPP_CLIP).to_dict(),
        "against": (ag_pg / league_ag).clip(*OPP_CLIP).to_dict(),
        "league_shots": (league_for + league_ag) / 2.0,
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
    recent = goalies[goalies["date"] >= asof - pd.Timedelta(days=30)]
    out["gp_30d"] = out["player_id"].map(
        recent.groupby("player_id")["date"].size()).fillna(0).astype(int)
    log.info("league save%% %.4f, league shots/game %.1f",
             league_sv, tr["league_shots"])
    return out

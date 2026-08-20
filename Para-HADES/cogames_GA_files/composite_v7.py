"""composite_v7 miner-only mine-and-deposit fitness scoring (self-contained).

Self-contained module with no imports from earlier composite_v* modules.

Curriculum:

  1. Pick the miner role -- and *only* the miner role. The mission's
     hub places 4 gear-stations (c:miner / c:aligner / c:scrambler /
     c:scout); using one grants a same-named inventory token in the
     agent's `gear` bucket (e.g. c:miner -> {miner: 1}). Switching gear
     stations clears the gear bucket first, so the role tokens are
     mutually exclusive. Holding `miner` is rewarded (score_role) and
     gates the mining/deposit terms. Holding any of the other three
     role tokens is *penalised* (score_wrong_role) so the GA selects
     for miner-specifically, not just any role.

  2. Mine and haul. Once gated in (holding `miner`), filling the `cargo`
     inventory bucket (carbon/oxygen/germanium/silicon, capacity 100) is
     rewarded -- but credited by *completed haul round-trips*, not by the
     raw per-step inventory change. A round-trip is detected with
     hysteresis on the cargo fill fraction: cargo must rise past
     HIGH_FILL_FRAC (a real load) and then fall below LOW_FILL_FRAC (a
     real dump). The reward saturates at TARGET_ROUNDTRIPS hauls.

     This replaces the old gross-delta accounting, which summed *every*
     positive per-step inventory change as "mining" and every negative
     change as a "deposit". That was trivially farmable: an agent parked
     on the cooldown-0 carbon extractor (or toggling a chest's
     deposit/withdraw vibes) oscillates its inventory in place and racks
     up tens of thousands of fake extract+deposit units without ever
     completing a haul. Hysteresis ignores in-place oscillation: a peak
     that never clears HIGH_FILL_FRAC, or a trough that never clears
     LOW_FILL_FRAC, scores zero.

  3. Return and deposit. The deposit term counts completed round-trips
     (cargo loaded then dumped). The fuller the cargo, the more a holding
     penalty accrues, pressuring the agent to actually dump rather than
     hoard. (There is no explicit chest-proximity test -- the env has no
     "deposit" action, and a chest can both take and give resources, so a
     proximity gate would not stop in-place farming. The fill->dump
     hysteresis is what separates a genuine haul from oscillation.)

Cold-start bootstrap
--------------------
A fully-gated score (mining multiplied by miner_fraction) leaves the
fitness surface flat at 0 until some candidate happens to hold a miner
token -- and a random initial population may never stumble onto it, so
the GA gets no gradient. To break that flat landscape, two terms are
*ungated*:

  * score_explore -- continuous map-coverage reward. Every moving
    candidate scores something, and broad coverage is exactly what makes
    an agent pass over a gear-station.
  * score_touch   -- a one-time bonus the first step the miner token is
    ever held. A discrete rung that rewards the discovery itself
    without needing sustained holding.

The wrong-role penalty is *also* ungated -- picking up the wrong role
is bad regardless of what else the agent does.

Partial gate (gate_floor)
-------------------------
Hard-gating mine/deposit behind miner_fraction proved to be a dead end:
agents mine prolifically (1000+ units/episode) WITHOUT ever holding the
miner token, but with miner_fraction ~= 0 the entire mine/deposit reward
is multiplied to ~0, so the composite collapses to the saturated
exploration ceiling and the GA random-searches a flat plateau (verified
on the jun2 founders). The gate is therefore softened to a *floor*:

  effective_gate = gate_floor + (1 - gate_floor) * miner_fraction

A non-miner now earns gate_floor of the mining reward (continuous
gradient from step one), while a full-time miner still earns the full
reward -- so holding the role gives up to a 1/gate_floor boost and the
"grab the gear first" curriculum is preserved. gate_floor=0.0 recovers
the old fully-gated behaviour (the module default, for old-founder
replay); the live training YAMLs set gate_floor>0.

Per-agent score:

  miner_steps        = steps holding `miner`
  wrong_role_steps   = steps holding any of {scout, aligner, scrambler}
  miner_fraction     = miner_steps     / steps_seen   # 0..1 gate + role signal
  wrong_role_frac    = wrong_role_steps / steps_seen
  touched_miner      = 1.0 if miner_steps > 0 else 0.0
  completed_cycles   = full fill->dump haul round-trips (hysteresis)
  total_extracted    = cargo hauled (dumped loads + current load peak)
  total_deposited    = cargo dumped over completed round-trips
  mine_progress      = min(total_extracted / (cap*TARGET_ROUNDTRIPS), 1)
  deposit_progress   = min(completed_cycles / TARGET_ROUNDTRIPS,      1)
  participated       = 1 if (mined anything or explored) else 0
  carry_fraction     = min(cargo_peak_ever / cap, 1)  # best load this episode
  score_role         =  w_role    * miner_fraction
  score_wrong_role   = -w_wrong   * wrong_role_frac   # UNGATED penalty
  score_touch        =  w_touch   * touched_miner     # UNGATED bootstrap rung
  score_explore      =  w_explore * min(exploration_capped, explore_cap) # UNGATED
  score_carry        =  w_carry   * carry_fraction    # UNGATED carry ramp (NEW)
  score_idle         = -w_idle    * (1 - participated) # UNGATED idle penalty
  score_mine         =  w_mine    * mine_progress      # bounded [0, w_mine]
  score_deposit      =  w_deposit * deposit_progress   # deposit_progress on a
                                                       #   (cycles/target)**deposit_pow
                                                       #   curve (sqrt when pow<1)
  score_hold         = -w_hold    * mean_fill_fraction
  bootstrap          = score_explore + score_touch + score_idle + score_carry
  gated              = score_mine + score_deposit + score_hold
  score              = bootstrap + score_role + score_wrong_role
                       + effective_gate * gated

Cross-agent reduction defaults to `bottom2_mean` (the mean of the two
*lowest*-scoring agents) so a single hero agent can no longer carry the
team: lifting the aggregate now requires the weakest agents to mine too.
The old `mean`/`max`/`trimmed_mean` reductions let one agent maximise the
objective while the rest idled. Combined with score_idle, this is the
participation pressure (curriculum goal: every agent should mine, not
just one).

Public API mirrors prior versions with a `_v7` suffix so the
evaluate_*_cogames.py drivers can swap modules with a single rename.
"""

import json
import os
import tempfile
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 4 base resources sharing the single `cargo` inventory bucket.
RESOURCE_KEYS = ('carbon', 'oxygen', 'germanium', 'silicon')

# Cargo "full load" scale, used as the denominator for the fill fraction that
# drives the haul-roundtrip hysteresis (HIGH/LOW_FILL_FRAC) and mine_progress.
# = a miner's intended full cargo: base 4 (CargoLimitVariant.limit) + miner role
# modifier 40 (roles/miner.py cargo_modifier) = 44. NOTE tutorial.scout does NOT
# install CargoLimitVariant, so resources are actually bounded only by the agent
# default_limit (65535) — there is no enforced cap — but 44 is the natural
# full-load scale: a loaded role-miner reaches fill≈1.0 (peaks measured ~50),
# while a non-miner (measured ~13-26) stays below the 0.8 load threshold, so only
# genuine miner-level hauls register a roundtrip. WAS 100 (a guess), which left
# fill≤~0.5 so the hysteresis never fired and mine/deposit_progress read ~0.
# `energy` is a separate bucket and is deliberately excluded from cargo-fill.
DEFAULT_CARGO_CAPACITY = 44

# Miner-only scoring: holding `miner` is the reward signal; the other three
# role tokens (granted by the c:scout / c:aligner / c:scrambler gear-stations)
# are penalised. Tokens are mutually exclusive in the gear bucket because
# every gear-station use clears the bucket first.
MINER_ROLE_KEY = 'miner'
WRONG_ROLE_KEYS = ('scout', 'aligner', 'scrambler')

# Haul round-trip detection (hysteresis on cargo fill fraction). A load is
# "ready" once fill clears HIGH_FILL_FRAC; the round-trip completes once fill
# falls back below LOW_FILL_FRAC. In-place inventory oscillation that never
# spans both thresholds counts for nothing. Mine/deposit reward saturate at
# TARGET_ROUNDTRIPS completed hauls (a strong episode), which also bounds the
# score so a fast farmer cannot run it to infinity.
HIGH_FILL_FRAC = 0.8
LOW_FILL_FRAC = 0.2
TARGET_ROUNDTRIPS = 5

# An agent "participated" if it mined anything OR explored at least this many
# distinct cells (read from per-agent episode stats). Truly idle agents (no
# mining, ~no movement) take the ungated score_idle penalty.
MIN_PARTICIPATION_CELLS = 5

DEFAULT_WEIGHTS_V7 = {
    'w_role':    5.0,   # heavily reward holding the miner role
    'w_wrong':   5.0,   # symmetric penalty for holding scout/aligner/scrambler
    'w_touch':   1.0,   # ungated one-time bonus for first miner pickup
    'w_explore': 2.0,   # ungated efficient-exploration bootstrap
    'w_approach': 0.0,  # ungated approach-to-miner-station shaping (module default
                        #   0.0 = off for old-founder replay; live YAMLs set >0)
    'w_idle':    1.0,   # ungated penalty for an agent that neither mines nor explores
    'w_mine':    4.0,   # reward mine-and-haul (round-trips, gated by miner_fraction)
    'w_deposit': 5.0,   # reward completed deposits (> mine so dump-and-repeat pays)
    'w_hold':    1.0,   # fill-pressure penalty strength
    'w_carry':   0.0,   # 2026-06-12 NEW — UNGATED continuous carry ramp: rewards the
                        #   best cargo load reached this episode (partial credit from
                        #   unit 1, role or not). Module default 0.0 = off (old replay);
                        #   live YAMLs set >0. This is the missing smooth gradient that
                        #   leads from "wander" -> "stand on extractor and fill" -> haul.

    # Explore saturation cap (was a hardcoded 0.8 inside compute_v3_behaviour).
    # score_explore = w_explore * min(exploration_capped, explore_cap). Default 0.8
    # = old behaviour (compute_v3_behaviour already caps at 0.8, so min is a no-op);
    # live YAMLs lower it (e.g. 0.4) so the explore plateau can't substitute for real
    # mining/hauling work.
    'explore_cap': 0.8,

    # Deposit-progress curve exponent: deposit_progress = (completed_cycles/target)
    # ** deposit_pow. Default 1.0 = linear (old behaviour). Live YAMLs set 0.5 (sqrt)
    # so the FIRST completed haul is a large beacon (0.2 -> 0.447 of the term) that
    # dwarfs the free bootstrap ceiling and pulls the GA onto real depositing.
    'deposit_pow': 1.0,

    # Partial gate floor in [0, 1]: effective_gate = gate_floor + (1-gate_floor)
    # * miner_fraction. 0.0 = old fully-gated behaviour (module default, kept so
    # old-founder replay is unchanged); the live YAMLs set gate_floor>0 so mining
    # done without the role still earns a continuous gradient. See the
    # "Partial gate" section of the module docstring.
    'gate_floor': 0.0,

    # 2026-06-17 carry-gating fraction in [0, 1]. The score_carry ramp was the
    # single largest term on the jun14 founders (~2.7 pts) and ENTIRELY ungated,
    # so a wander-and-load agent banked it without ever holding the miner role --
    # making fitness indifferent between "hold the role, don't mine" (gru) and
    # "mine, don't hold the role" (heb). gate_carry routes this fraction of
    # score_carry through effective_gate so holding the role pays up to
    # 1/gate_floor more on the carry ramp too, COUPLING role -> carrying -> haul.
    # 0.0 = old fully-ungated carry (module default, kept so old-founder replay is
    # unchanged); live YAMLs set 1.0. gate_floor still gives a role-less cold-start
    # agent a continuous fraction of the ramp.
    'gate_carry': 0.0,

    # 2026-06-17 completed-haul target. mine_progress saturates at
    # cap*target_roundtrips extracted and deposit_progress at target_roundtrips
    # dumps. The jun14 founders extracted ~0.7 of ONE full haul against a target
    # of 5, so the deposit beacon was unreachable and gave no gradient. Lowering
    # the target makes the first completed haul a large, reachable beacon. Default
    # = TARGET_ROUNDTRIPS (old behaviour for replay); live YAMLs lower it.
    'target_roundtrips': TARGET_ROUNDTRIPS,

    # 2026-06-24 haul-detection hysteresis thresholds (cargo fill fraction),
    # consumed by PerEpisodeAccumulatorV7.__init__ (NOT episode_score). A load
    # is "ready" once fill clears high_fill_frac; the round-trip completes once
    # fill falls back below low_fill_frac. Defaults = the old module constants
    # (HIGH_FILL_FRAC / LOW_FILL_FRAC) so old-founder replay is unchanged. The
    # jun22 founders peaked at fill ~0.55 of cap and so NEVER cleared the 0.8
    # default -- completed_cycles ~= 0 and the deposit beacon gave no gradient.
    # Live YAMLs lower high_fill_frac (e.g. 0.45) so genuine miner-level hauls
    # register a round-trip while small non-miner loads still do not. low keeps
    # the dump requiring travel to a chest, so in-place oscillation still scores
    # zero. Evaluators must forward these into the accumulator constructor.
    'high_fill_frac': HIGH_FILL_FRAC,
    'low_fill_frac':  LOW_FILL_FRAC,

    # Reduction across agents:
    #   'max' | 'mean' | 'min' | 'trimmed_mean' | 'top2_mean' | 'bottom2_mean'.
    # Default 'bottom2_mean' (mean of the two lowest agents) so one hero agent
    # cannot carry an idle team -- raising the aggregate requires the weakest
    # agents to mine too.
    'reduction': 'bottom2_mean',
}


# ---------------------------------------------------------------------------
# Obs parsing
# ---------------------------------------------------------------------------

def parse_inventory_at_center(obs, obs_hr=6, obs_wr=6):
    """Return dict {resource_name: int} for inv:* tokens at the obs center.

    The observing agent sits at the center of its HxW egocentric window, and its
    OWN inventory tokens (the role token + cargo) are emitted at that center
    cell. The default (6,6) is the center of the 13x13 window the cogames 0.24.3
    wheel emits for tutorial.scout. THIS WAS (5,5) — the center of an 11x11
    window — which read a neighbouring cell on the 13x13 wheel and so saw the
    agent's own role/cargo almost never (miner_fraction and the haul metrics
    collapsed to ~0). Callers that know the true window size pass it explicitly
    (PerEpisodeAccumulatorV7 stores the env-derived center).
    """
    inv = {}
    for tok in obs.tokens:
        if tok.location == (obs_hr, obs_wr):
            name = tok.feature.name
            if name.startswith('inv:'):
                inv[name[4:]] = int(tok.value)
    return inv


def obs_center_from_env(env_cfg, default=(6, 6)):
    """Return (row, col) of the egocentric obs-window center for this env.

    The observing agent sits at the center of its HxW observation window, so the
    center is (H//2, W//2). Read from ``env_cfg.game.obs.{height,width}``; falls
    back to ``default`` (the 13x13 wheel's center) if the config can't be read.
    Load-bearing: composite_v7 parses the agent's own inventory (role token +
    cargo) at this cell, so an off-by-one here silently zeroes miner_fraction and
    the haul metrics.
    """
    try:
        obs = env_cfg.game.obs
        return (int(obs.height) // 2, int(obs.width) // 2)
    except Exception:
        return default


def min_chebyshev_to_tag(obs, tag_value, obs_hr, obs_wr):
    """Min Chebyshev distance from the obs center to any `tag` token whose value
    equals ``tag_value``, or None if no such token is in view.

    Used to locate a specific gear-station (identified by its distinguishing tag
    id -- e.g. the c:miner station's tag) within the egocentric observation
    window, so the approach-shaping term can reward getting closer to it. Cheap:
    a single pass over the obs tokens (which update() already iterates).
    """
    best = None
    for tok in obs.tokens:
        feat = getattr(tok, 'feature', None)
        if feat is None or feat.name != 'tag':
            continue
        if int(tok.value) != int(tag_value):
            continue
        loc = getattr(tok, 'location', None)
        if loc is None:
            continue
        d = max(abs(int(loc[0]) - obs_hr), abs(int(loc[1]) - obs_wr))
        if best is None or d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# v3 behaviour (re-used as the always-on exploration term)
# ---------------------------------------------------------------------------

def compute_v3_behaviour(agent_stats_list, max_steps):
    """Return (behaviour_capped_at_0p8, behaviour_raw_mean, comp_log).

    comp_log is a list of per-agent (unique, dist, n_moves, fail_ratio,
    activity, efficiency, behaviour_raw) tuples for diagnostic printing.
    """
    agent_scores = []
    comp_log = []
    for s in agent_stats_list:
        unique     = float(s.get('cell.unique_visited', 0.0))
        dist       = float(s.get('cell.max_distance_from_spawn', 0.0))
        n_success  = float(s.get('action.move.success', 0.0))
        n_failed   = float(s.get('action.move.failed', 0.0))
        n_moves    = n_success + n_failed
        fail_ratio = n_failed / max(n_moves, 1.0)
        activity   = min(n_moves / max_steps, 1.0)
        efficiency = max(unique - 1.0, 0.0) / max(n_moves, 1.0)
        behaviour_raw = (
            1.0 * (unique / max_steps)
            + 0.5 * (dist / (max_steps ** 0.5))
            + 0.3 * activity
            + 0.2 * efficiency * activity
            - 0.2 * fail_ratio
        )
        agent_scores.append(behaviour_raw)
        comp_log.append((unique, dist, n_moves, fail_ratio, activity,
                         efficiency, behaviour_raw))
    behaviour_raw_mean = float(np.mean(agent_scores)) if agent_scores else 0.0
    behaviour_capped = min(behaviour_raw_mean, 0.8)
    return behaviour_capped, behaviour_raw_mean, comp_log


# ---------------------------------------------------------------------------
# Per-episode accumulator
# ---------------------------------------------------------------------------

class PerEpisodeAccumulatorV7:
    """One per agent. Tracks miner-role holding, wrong-role holding,
    mining/deposit deltas, and cargo-fill pressure."""

    def __init__(self, max_steps, quadrant_to_role=None,
                 cargo_capacity=DEFAULT_CARGO_CAPACITY, obs_center=None,
                 high_fill_frac=HIGH_FILL_FRAC, low_fill_frac=LOW_FILL_FRAC):
        # `quadrant_to_role` is accepted for back-compat with v6 callers but
        # no longer used in scoring: this v7 variant is miner-only regardless
        # of spawn quadrant.
        del quadrant_to_role
        self.max_steps = max(int(max_steps), 1)
        self.cargo_capacity = max(int(cargo_capacity), 1)

        # Haul round-trip hysteresis thresholds (fill fraction). Tunable per
        # run so the live YAMLs can lower the load threshold when agents
        # plateau below the default 0.8 (jun22: peak loads ~0.55 of cap never
        # cleared 0.8, so completed_cycles ~= 0 and the deposit term gave no
        # gradient). Defaults = the old module constants so old-founder replay
        # is unchanged. Clamped to a sane order (low < high, both in (0,1]).
        hi = min(max(float(high_fill_frac), 1e-3), 1.0)
        lo = min(max(float(low_fill_frac), 0.0), hi - 1e-3)
        self.high_fill_frac = hi
        self.low_fill_frac = lo

        # Egocentric obs-window center = the agent's own cell, where its role
        # token and cargo are read. Defaults to (6,6) = center of the 13x13
        # window; evaluators pass the env-derived center via obs_center.
        if obs_center is None:
            self._obs_hr, self._obs_wr = 6, 6
        else:
            self._obs_hr, self._obs_wr = int(obs_center[0]), int(obs_center[1])

        self.steps_seen = 0

        # Steps holding the `miner` role token (the reward signal + gate).
        self.miner_steps = 0
        # Steps holding any of the wrong role tokens (scout/aligner/scrambler).
        self.wrong_role_steps = 0

        # Mine-and-deposit tracked as completed *haul round-trips* via
        # hysteresis on the cargo fill fraction (NOT gross per-step inventory
        # deltas, which an agent farms by oscillating its inventory in place
        # at an extractor/chest -- see module docstring). A round-trip = cargo
        # rises past HIGH_FILL_FRAC (loaded) then falls below LOW_FILL_FRAC
        # (dumped).
        self.completed_cycles = 0          # full fill->dump round-trips
        self.deposited_units = 0.0         # cargo dumped over completed cycles
        self.cargo_peak_since_dump = 0.0   # peak cargo of the in-progress haul
        self.cargo_peak_ever = 0.0         # best cargo load reached this episode (for
                                           #   the ungated score_carry ramp; never reset)
        self._phase = 'fill'               # 'fill' (loading) or 'dump' (returning)

        # Deposit pressure: running sum of per-step cargo-fill fraction.
        self.sum_fill_fraction = 0.0

        self._spawn_pos = None
        self._map_extents = None
        self._quadrant = 'NA'

        # Approach-to-miner-station shaping. The miner gear-station is located in
        # the egocentric obs by its distinguishing `tag` id (set per episode from
        # sim.grid_objects); we track the closest it ever came to the obs center
        # (Chebyshev). None until the tag is set / the station is first seen.
        self._miner_tag = None
        self._min_station_dist = None

    # ---- setup -----------------------------------------------------------

    def set_spawn(self, spawn_pos, map_extents):
        # Spawn / quadrant are diagnostic only -- they no longer influence
        # scoring (miner-only target for every agent).
        if spawn_pos is None or map_extents is None:
            return
        self._spawn_pos = (int(spawn_pos[0]), int(spawn_pos[1]))
        R, C = map_extents
        self._map_extents = (int(R), int(C))
        r0, c0 = self._spawn_pos
        ns = 'N' if r0 < self._map_extents[0] / 2 else 'S'
        we = 'W' if c0 < self._map_extents[1] / 2 else 'E'
        self._quadrant = ns + we

    def set_miner_tag(self, tag_id):
        """Record the miner gear-station's distinguishing obs `tag` id so the
        per-step approach shaping can find the station in the egocentric obs.
        Pass None to disable the term (mission with no miner station / unknown
        tag) -- score_approach then stays 0."""
        self._miner_tag = None if tag_id is None else int(tag_id)

    # ---- per-step --------------------------------------------------------

    def update(self, obs, obs_hr=None, obs_wr=None):
        hr = self._obs_hr if obs_hr is None else obs_hr
        wr = self._obs_wr if obs_wr is None else obs_wr
        inv = parse_inventory_at_center(obs, obs_hr=hr, obs_wr=wr)
        self.steps_seen += 1

        # Miner role signal: holding the `miner` gear token.
        if inv.get(MINER_ROLE_KEY, 0) > 0:
            self.miner_steps += 1
        # Wrong-role signal: holding any of the other three role tokens.
        # The gear bucket is cleared on every gear-station use, so these
        # are mutually exclusive with `miner` -- no double-counting risk.
        if any(inv.get(k, 0) > 0 for k in WRONG_ROLE_KEYS):
            self.wrong_role_steps += 1

        # Cargo-fill fraction (0..1, capped) -- deposit pressure + hysteresis.
        cargo = float(sum(inv.get(k, 0) for k in RESOURCE_KEYS))
        fill = min(cargo / self.cargo_capacity, 1.0)
        self.sum_fill_fraction += fill

        # Haul round-trip hysteresis. Only a genuine fill->dump swing counts;
        # in-place oscillation (peak never clears HIGH_FILL_FRAC, or trough
        # never clears LOW_FILL_FRAC) yields nothing -- this is what defeats
        # the parked-on-extractor / chest-toggle farming exploit.
        self.cargo_peak_since_dump = max(self.cargo_peak_since_dump, cargo)
        # Episode-wide best load, for the ungated score_carry ramp (never reset).
        self.cargo_peak_ever = max(self.cargo_peak_ever, cargo)
        if self._phase == 'fill':
            if fill >= self.high_fill_frac:
                self._phase = 'dump'       # loaded; now expect a dump
        else:  # 'dump'
            if fill <= self.low_fill_frac:  # dumped -> one round-trip done
                self.completed_cycles += 1
                self.deposited_units += min(self.cargo_peak_since_dump,
                                            float(self.cargo_capacity))
                self.cargo_peak_since_dump = cargo
                self._phase = 'fill'

        # Approach-to-station shaping: track the closest the miner gear-station
        # ever came to the obs center (only updates while the station is in
        # view). Monotone min -> the final reward is the single closest approach,
        # so it can't be farmed by lingering near the station.
        if self._miner_tag is not None:
            d = min_chebyshev_to_tag(obs, self._miner_tag, hr, wr)
            if d is not None:
                self._min_station_dist = (d if self._min_station_dist is None
                                          else min(self._min_station_dist, d))

    # ---- finalize --------------------------------------------------------

    def episode_score(self, ep_reward_total, exploration_capped,
                      agent_stats=None, weights=None):
        """Combine accumulator state into the v7 score dict.

        `ep_reward_total` is kept for signature compatibility but is not
        part of the formula (hearts dropped from the objective).
        """
        w = {**DEFAULT_WEIGHTS_V7, **(weights or {})}
        steps = max(self.steps_seen, 1)
        cap = float(self.cargo_capacity)
        target = max(float(w.get('target_roundtrips',
                                 DEFAULT_WEIGHTS_V7['target_roundtrips'])), 1e-6)

        miner_fraction       = self.miner_steps      / steps
        wrong_role_fraction  = self.wrong_role_steps / steps
        touched_miner        = 1.0 if self.miner_steps > 0 else 0.0
        mean_fill_fraction   = self.sum_fill_fraction / steps

        # Hauling: total_extracted = cargo dumped over completed round-trips
        # plus the peak of the haul currently in progress (so partial progress
        # before the first dump still gives gradient). total_deposited counts
        # only fully completed dumps. Both reward terms saturate at
        # TARGET_ROUNDTRIPS hauls, so a fast in-place farmer cannot run the
        # score up without bound.
        total_extracted  = self.deposited_units + min(self.cargo_peak_since_dump, cap)
        total_deposited  = self.deposited_units
        mine_progress    = min(total_extracted / (cap * target), 1.0)
        # Deposit progress, optionally on a concave (sqrt) curve so the FIRST
        # completed haul is a large beacon rather than a 1/target step. deposit_pow
        # defaults to 1.0 (linear, old behaviour); live YAMLs set 0.5.
        deposit_pow      = max(float(w.get('deposit_pow',
                                           DEFAULT_WEIGHTS_V7['deposit_pow'])), 1e-6)
        deposit_progress = min((self.completed_cycles / target) ** deposit_pow, 1.0)
        # Best cargo load reached this episode -> ungated carry ramp (partial credit).
        carry_fraction   = min(self.cargo_peak_ever / cap, 1.0)

        # Per-agent participation -> idle penalty. Uses per-agent episode stats
        # when supplied; an agent that neither mined nor explored is idle.
        stats = agent_stats or {}
        unique_cells = float(stats.get('cell.unique_visited', 0.0))
        participated = 1.0 if (total_extracted > 0.0
                               or unique_cells >= MIN_PARTICIPATION_CELLS) else 0.0
        w_idle = float(w.get('w_idle', DEFAULT_WEIGHTS_V7['w_idle']))

        # Explore term, tightened by an optional saturation cap (default 0.8 = no-op;
        # live YAMLs lower it so wandering can't substitute for real work).
        explore_cap         = float(w.get('explore_cap', DEFAULT_WEIGHTS_V7['explore_cap']))
        exploration_capped  = min(float(exploration_capped), explore_cap)

        score_role       =  float(w['w_role'])    * miner_fraction
        score_wrong_role = -float(w['w_wrong'])   * wrong_role_fraction
        score_touch      =  float(w['w_touch'])   * touched_miner
        score_explore    =  float(w['w_explore']) * exploration_capped
        score_idle       = -w_idle                * (1.0 - participated)
        score_mine       =  float(w['w_mine'])    * mine_progress
        score_deposit    =  float(w['w_deposit']) * deposit_progress
        score_hold       = -float(w['w_hold'])    * mean_fill_fraction
        # Ungated carry ramp: continuous partial credit for loading cargo (role or
        # not), the smooth gradient that leads from wandering to a full haul.
        score_carry      =  float(w.get('w_carry', DEFAULT_WEIGHTS_V7['w_carry'])) * carry_fraction

        # Approach-to-station shaping (ungated bootstrap rung). station_proximity
        # in [0,1]: 1.0 = the agent stood on the miner gear-station cell, 0.0 =
        # the station never came within view. Linear in how close the closest
        # approach got to the obs center, so it gives a continuous gradient that
        # *leads to* the station -- turning miner-role acquisition from a sparse
        # discrete event into something the GA can climb. Monotone (best-ever,
        # not per-step) so it isn't farmable by lingering, and keyed on the miner
        # station tag specifically (not chests), so it pulls toward `miner` over
        # the adjacent scout/aligner/scrambler stations.
        obs_radius = max(self._obs_hr, self._obs_wr, 1)
        if self._min_station_dist is None:
            station_proximity = 0.0
        else:
            station_proximity = max(0.0, (obs_radius - self._min_station_dist)
                                    / float(obs_radius))
        w_approach     = float(w.get('w_approach', DEFAULT_WEIGHTS_V7['w_approach']))
        score_approach = w_approach * station_proximity

        # Ungated rungs keep the surface non-flat and pressure every agent to
        # engage: explore + first-touch + approach reward discovery; the
        # wrong-role and idle penalties bite regardless of role. Mine/deposit are
        # gated by a *floored* miner_fraction: a non-miner still earns gate_floor
        # of the mining reward (continuous gradient), a full-time miner earns all
        # of it. gate_floor=0 recovers the old hard gate. See module docstring.
        gate_floor    = float(w.get('gate_floor', DEFAULT_WEIGHTS_V7['gate_floor']))
        gate_floor    = min(max(gate_floor, 0.0), 1.0)
        effective_gate = gate_floor + (1.0 - gate_floor) * miner_fraction
        # Carry-gating: route gate_carry of the carry ramp through effective_gate
        # so the (otherwise role-free) ramp pays the role-holder more. gate_carry=0
        # leaves carry fully ungated (old replay); =1 puts it fully under the gate.
        gate_carry    = float(w.get('gate_carry', DEFAULT_WEIGHTS_V7['gate_carry']))
        gate_carry    = min(max(gate_carry, 0.0), 1.0)
        carry_ungated = (1.0 - gate_carry) * score_carry
        carry_gated   = gate_carry * score_carry
        bootstrap = (score_explore + score_touch + score_idle + score_approach
                     + carry_ungated)
        gated     = score_mine + score_deposit + score_hold + carry_gated
        score     = (bootstrap + score_role + score_wrong_role
                     + effective_gate * gated)

        return {
            'score':                float(score),
            'effective_gate':       float(effective_gate),
            'score_role':           float(score_role),
            'score_wrong_role':     float(score_wrong_role),
            'score_touch':          float(score_touch),
            'score_explore':        float(score_explore),
            'score_approach':       float(score_approach),
            'station_proximity':    float(station_proximity),
            'score_idle':           float(score_idle),
            'score_carry':          float(score_carry),
            'carry_fraction':       float(carry_fraction),
            'score_mine':           float(score_mine),
            'score_deposit':        float(score_deposit),
            'score_hold':           float(score_hold),
            'miner_fraction':       float(miner_fraction),
            'wrong_role_fraction':  float(wrong_role_fraction),
            'touched_miner':        float(touched_miner),
            'participated':         float(participated),
            'completed_cycles':     float(self.completed_cycles),
            'mine_progress':        float(mine_progress),
            'deposit_progress':     float(deposit_progress),
            'total_extracted':      float(total_extracted),
            'total_deposited':      float(total_deposited),
            'mean_fill_fraction':   float(mean_fill_fraction),
            'exploration_capped':   float(exploration_capped),
            'ep_reward':            float(ep_reward_total),
            'spawn_pos':            self._spawn_pos,
            'quadrant':             self._quadrant,
        }


# ---------------------------------------------------------------------------
# Component fields that are always mean-reduced for diagnostic readability.
# ---------------------------------------------------------------------------

COMPONENT_KEYS = (
    'score_role', 'score_wrong_role', 'score_touch', 'score_explore',
    'score_approach', 'station_proximity',
    'score_idle', 'score_carry', 'carry_fraction',
    'score_mine', 'score_deposit', 'score_hold',
    'miner_fraction', 'effective_gate', 'wrong_role_fraction', 'touched_miner',
    'participated', 'completed_cycles', 'mine_progress', 'deposit_progress',
    'total_extracted', 'total_deposited',
    'mean_fill_fraction', 'exploration_capped', 'ep_reward',
)


# ---------------------------------------------------------------------------
# Cross-agent aggregation
# ---------------------------------------------------------------------------

def aggregate_agent_results_v7(per_agent_results, reduction='trimmed_mean'):
    """Reduce a list of per-agent v7 dicts into one aggregate dict.

    Reduction options:
      'max'           best-of-N (high run-to-run variance).
      'mean'          average of all agents.
      'min'           weakest-link (strongest cooperation pressure).
      'trimmed_mean'  drop the single lowest-scoring agent, mean the rest
                      (removes single-agent luck variance).
      'top2_mean'     average of the top-2 agents.
      'bottom2_mean'  default: average of the two *lowest* agents -- one
                      hero agent can't carry an idle team; lifting the
                      aggregate requires the weakest agents to mine too.

    Component fields are always mean-reduced. `score_mean` / `score_min` /
    `score_max` / `score_std` and `best_agent_idx` are always reported.
    """
    if not per_agent_results:
        return {}

    score_vals = [r['score'] for r in per_agent_results if 'score' in r]
    out = {}
    if score_vals:
        arr = np.asarray(score_vals, dtype=float)
        n = arr.shape[0]
        if reduction == 'max':
            out['score'] = float(arr.max())
        elif reduction == 'mean':
            out['score'] = float(arr.mean())
        elif reduction == 'min':
            out['score'] = float(arr.min())
        elif reduction == 'trimmed_mean':
            if n >= 3:
                out['score'] = float(np.sort(arr)[1:].mean())
            elif n == 2:
                out['score'] = float(arr.max())
            else:
                out['score'] = float(arr[0])
        elif reduction == 'top2_mean':
            if n >= 2:
                out['score'] = float(np.sort(arr)[-2:].mean())
            else:
                out['score'] = float(arr[0])
        elif reduction == 'bottom2_mean':
            if n >= 2:
                out['score'] = float(np.sort(arr)[:2].mean())
            else:
                out['score'] = float(arr[0])
        else:
            out['score'] = float(arr.max())
        out['score_mean']     = float(arr.mean())
        out['score_min']      = float(arr.min())
        out['score_max']      = float(arr.max())
        out['score_std']      = float(arr.std())
        out['best_agent_idx'] = int(np.argmax(arr))
    else:
        out['score'] = 0.0
        out['score_mean'] = 0.0
        out['score_min']  = 0.0
        out['score_max']  = 0.0
        out['score_std']  = 0.0
        out['best_agent_idx'] = -1
    out['reduction'] = str(reduction)

    for k in COMPONENT_KEYS:
        vals = [r[k] for r in per_agent_results if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0

    best_idx = out['best_agent_idx']
    best = per_agent_results[best_idx] if score_vals and best_idx >= 0 else per_agent_results[0]
    out['spawn_pos'] = best.get('spawn_pos')
    out['quadrant']  = best.get('quadrant')
    return out


# ---------------------------------------------------------------------------
# Aggregation across episodes
# ---------------------------------------------------------------------------

def aggregate_over_episodes_v7(per_episode_aggregates):
    """Reduce a list of per-episode aggregates into one over-all-episodes
    aggregate. `score` is the mean across episodes; component fields are
    mean-reduced.
    """
    if not per_episode_aggregates:
        return {}
    out = {}
    score_vals = [a.get('score', 0.0) for a in per_episode_aggregates]
    if score_vals:
        arr = np.asarray(score_vals, dtype=float)
        out['score']      = float(arr.mean())
        out['score_mean'] = float(arr.mean())
        out['score_min']  = float(arr.min())
        out['score_max']  = float(arr.max())
        out['score_std']  = float(arr.std())
    else:
        out['score'] = 0.0
        out['score_mean'] = 0.0
        out['score_min']  = 0.0
        out['score_max']  = 0.0
        out['score_std']  = 0.0
    out['n_episodes'] = len(per_episode_aggregates)
    out['reduction']  = per_episode_aggregates[0].get('reduction', 'trimmed_mean')

    for k in COMPONENT_KEYS:
        vals = [a[k] for a in per_episode_aggregates if k in a]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# Diagnostic stdout (regex fallback for evaluators not using sidecar)
# ---------------------------------------------------------------------------

def format_diagnostic_lines_v7(agg, ep_reward_total):
    """Multi-line v7 diagnostic block. Lines begin with 'v7:' so the regex
    parser in task drivers picks up ep_reward without changes.
    """
    spawn = agg.get('spawn_pos')
    spawn_str = f'({spawn[0]},{spawn[1]})' if spawn else 'NA'
    line1 = (
        f'  v7: spawn={spawn_str} quadrant={agg.get("quadrant", "NA")} '
        f'best_agent={agg.get("best_agent_idx", -1)} '
        f'reduction={agg.get("reduction", "trimmed_mean")}'
    )
    line2 = (
        f'  v7: miner_frac={agg.get("miner_fraction", 0):.3f} '
        f'wrong_frac={agg.get("wrong_role_fraction", 0):.3f} '
        f'touched={agg.get("touched_miner", 0):.3f} '
        f'approach={agg.get("station_proximity", 0):.3f} '
        f'participated={agg.get("participated", 0):.3f} '
        f'cycles={agg.get("completed_cycles", 0):.2f} '
        f'extracted={agg.get("total_extracted", 0):.1f} '
        f'deposited={agg.get("total_deposited", 0):.1f} '
        f'fill={agg.get("mean_fill_fraction", 0):.3f} '
        f'explore={agg.get("exploration_capped", 0):.3f}'
    )
    line3 = (
        f'  v7: score_role={agg.get("score_role", 0):.4f} '
        f'score_wrong={agg.get("score_wrong_role", 0):.4f} '
        f'score_touch={agg.get("score_touch", 0):.4f} '
        f'score_explore={agg.get("score_explore", 0):.4f} '
        f'score_approach={agg.get("score_approach", 0):.4f} '
        f'score_carry={agg.get("score_carry", 0):.4f} '
        f'score_idle={agg.get("score_idle", 0):.4f} '
        f'score_mine={agg.get("score_mine", 0):.4f} '
        f'score_deposit={agg.get("score_deposit", 0):.4f} '
        f'score_hold={agg.get("score_hold", 0):.4f}'
    )
    line4 = (
        f'  v7: ep_reward={float(ep_reward_total):.4f} '
        f'score_mean={agg.get("score_mean", 0):.4f} '
        f'-> score={agg.get("score", 0):.4f}'
    )
    return '\n'.join([line1, line2, line3, line4])


# ---------------------------------------------------------------------------
# Sidecar JSON write (atomic)
# ---------------------------------------------------------------------------

def write_component_sidecar(path, per_episode_components, aggregate,
                             fitness_stat='composite_v7_zstat'):
    """Write per-episode + aggregate component dicts to a JSON sidecar.

    Atomic via temp + os.replace. No-op if `path` is empty / None.
    """
    if not path:
        return
    payload = {
        'version':      'v7.3',
        'fitness_stat': fitness_stat,
        'n_episodes':   len(per_episode_components),
        'aggregate':    aggregate,
        'per_episode':  per_episode_components,
    }
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.components_v7_', suffix='.json.tmp', dir=parent)
    try:
        with os.fdopen(fd, 'wt') as f:
            json.dump(payload, f, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, Counter):
        return dict(o)
    return str(o)


# ---------------------------------------------------------------------------
# CLI weight parsing
# ---------------------------------------------------------------------------

def parse_weights_str(s):
    """Parse 'k1=v1,k2=v2,...' from a CLI flag into a dict.

    Float values are stored as float; the special key 'reduction' is
    preserved as a string ('max' | 'mean' | 'min' | 'trimmed_mean' |
    'top2_mean').
    """
    out = {}
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        k = k.strip()
        v = v.strip()
        if k == 'reduction':
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Spawn-position lookup helper for evaluators
# ---------------------------------------------------------------------------

def get_agent_spawn_positions(sim, num_agents):
    """Return [(row, col)] for each agent_id 0..num_agents-1, or None if not found.

    Reads from sim.grid_objects(); locations are reported as (x, y) so we
    swap to (row, col).
    """
    positions = [None] * num_agents
    try:
        objs = sim.grid_objects(ignore_types=['wall'])
    except TypeError:
        objs = sim.grid_objects()
    for obj in objs.values():
        agent_id = obj.get('agent_id')
        if agent_id is None:
            continue
        if not (0 <= agent_id < num_agents):
            continue
        loc = obj.get('location')
        if loc is None:
            continue
        positions[agent_id] = (int(loc[1]), int(loc[0]))
    return positions


def get_map_extents(sim):
    """Return (R, C) for the current sim, or (1, 1) on failure."""
    try:
        return int(sim.map_height), int(sim.map_width)
    except Exception:
        return 1, 1


def get_miner_station_tag(sim, station_type='miner', shared_tag=1):
    """Return the miner gear-station's distinguishing obs `tag` id, or None.

    The c:miner gear-station is a single grid object of type ``station_type``
    carrying tag_ids like ``[shared_tag, X]`` (e.g. [1, 8] in tutorial.scout);
    that same X appears as a `tag` token at the station's cell in every agent's
    egocentric obs. We return X (the non-shared id) so an accumulator can locate
    the station in-obs for approach shaping. Read per episode (the procedural map
    re-rolls the layout, but the tag id is stable by object type). Falls back to
    the object's scalar ``tag`` field if tag_ids is unavailable.
    """
    try:
        objs = sim.grid_objects()
    except Exception:
        return None
    for o in objs.values():
        if o.get('type_name') != station_type:
            continue
        for t in (o.get('tag_ids') or []):
            if int(t) != shared_tag:
                return int(t)
        tag = o.get('tag')
        if tag is not None and int(tag) != shared_tag:
            return int(tag)
    return None

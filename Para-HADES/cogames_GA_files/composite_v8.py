"""composite_v8 staged multi-role curriculum fitness (self-contained).

Self-contained module with no imports from earlier composite_v* modules.
Public API mirrors composite_v7 with a `_v8` suffix so the evaluate_*_cogames.py
drivers can add a `composite_v8_zstat` branch that parallels the v7 branch.

Why v8 exists
-------------
composite_v7 is *miner-only*: it rewards holding the `miner` role and the
mine -> haul -> deposit loop, and it PENALISES every other role
(scout/aligner/scrambler). That trained good miners (the jun28 Hebbian
founders), but the real tournament objective is territory control
(`aligned_junction_held`), which requires the *whole* team loop:

    mine elements -> deposit at hub -> craft a heart at the hub ->
    pick the aligner (or scrambler) role -> align (or scramble) a junction
    near the hub/net -> hold the aligned junction (= territory).

v8 opens the fitness up to reward ALL roles except scout, but *staged* so the
GA climbs a ladder instead of being handed a flat multi-objective surface:

  Stage A (early game) -- reward choosing the MINER role, which gates the
    miner-specific rewards (mine / haul / deposit).  Scout is the only
    penalised role.  Holding aligner/scrambler is neither rewarded (yet) nor
    penalised -- it just costs miner-gate, so early role2 is self-limiting.

  Stage B (after returning to the hub and depositing) -- a completed haul
    round-trip (deposit) UNLOCKS a reward for holding the other two roles
    (aligner or scrambler), and simultaneously DECAYS the reward for merely
    holding the miner role ("reward the miner role less").  So once an agent
    can deposit, the gradient pushes it to move on to a capturer role.

  Stage C (after choosing aligner or scrambler) -- holding a capturer role
    UNLOCKS the big prize: getting hearts and capturing territory with those
    hearts (aligning / scrambling junctions).  This is the true objective and
    is weighted to dominate once it is reachable.

Because every founder is evaluated as a team of 4 *identical* policies (same
genome, 4 instances), the natural target is that each agent individually walks
the whole ladder, so the staging is applied per-agent and gated by that
agent's own progress.  Team division of labour is not designed in -- it can
only emerge from spawn/stochastic asymmetry.

Signals (verified against the running cogames wheel, tutorial.miner)
--------------------------------------------------------------------
Per-step, read at the egocentric obs centre (the agent's own cell):
  * role gear tokens: `miner`, `aligner`, `scrambler`, `scout`
    (mutually exclusive in the `gear` bucket; every gear-station use clears it)
  * `heart` token (heart bucket, cap 10)
  * cargo resources: carbon / oxygen / germanium / silicon (single bucket)

Per-episode agent stats (episode_stats['agent'][i], read at finalize; every
lookup uses .get(k, 0.0) because a stat key is ABSENT until its event fires):
  * `junction.aligned_by_agent`   -- junctions this agent aligned (capture)
  * `junction.scrambled_by_agent` -- junctions this agent scrambled (contest)
  * `heart.gained`                -- hearts acquired this episode
  * `cell.unique_visited`         -- exploration / participation

Gates keep the ladder monotone but SOFT (a floor gives a continuous gradient
below each rung so the GA never sits on a flat plateau):

  effective_gate = gate_floor + (1-gate_floor) * miner_fraction     # stage A
  gateB          = b_floor    + (1-b_floor)    * deposit_progress    # stage B
  gateC          = c_floor    + (1-c_floor)    * capturer_commit     # stage C

Per-agent score:

  miner_fraction     = miner_steps / steps
  scout_fraction     = scout_steps / steps                 # only penalised role
  capturer_frac      = (aligner_steps + scrambler_steps) / steps
  capturer_commit    = min(capturer_frac / role2_target, 1)
  completed_cycles   = full fill->dump haul round-trips (hysteresis; deposits)
  deposit_progress   = min((completed_cycles/target)**deposit_pow, 1)
  heart_progress     = min(hearts_gained / heart_target, 1)
  territory_events   = w_align_share*aligned + w_scramble_share*scrambled
  territory_progress = min(territory_events / territory_target, 1)

  # ungated bootstrap (keeps the surface non-flat; scout is the only penalty)
  score_explore  =  w_explore  * min(exploration, explore_cap)
  score_touch    =  w_touch    * touched_miner
  score_approach =  w_approach * miner_station_proximity
  score_idle     = -w_idle     * (1 - participated)
  score_scout    = -w_scout    * scout_fraction
  score_carry    =  w_carry    * carry_fraction   (gate_carry routed into stage A)

  # stage A -- miner role + gated mine/haul/deposit
  miner_decay      = 1 - w_miner_decay * deposit_progress     # >= 1-w_miner_decay
  score_role_miner =  w_role_miner * miner_fraction * miner_decay
  score_mine       =  w_mine    * mine_progress
  score_deposit    =  w_deposit * deposit_progress
  score_hold       = -w_hold    * mean_fill_fraction

  # stage B -- aligner/scrambler role holding, unlocked by depositing
  score_role2      =  gateB * w_role2 * capturer_frac

  # stage C -- hearts + territory, unlocked by holding a capturer role
  score_heart      =  gateB * gateC * w_heart     * heart_progress
  score_territory  =  gateB * gateC * w_territory * territory_progress

  score = (score_explore + score_touch + score_approach + score_idle
           + score_scout + carry_ungated)                       # bootstrap
        + score_role_miner
        + effective_gate * (score_mine + score_deposit + score_hold + carry_gated)
        + score_role2
        + score_heart + score_territory

Cross-agent reduction defaults to `mean` (restore `bottom2_mean` once a
competent full-loop agent reliably emerges -- that is the coordination rung).
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

# Cargo "full load" scale (denominator for the fill fraction driving the haul
# hysteresis and mine_progress). See composite_v7 for the derivation: base
# CargoLimit 4 + miner role modifier 40 = 44, the natural full-load scale.
DEFAULT_CARGO_CAPACITY = 44

# Role gear tokens (granted by the c:miner/c:aligner/c:scrambler/c:scout gear
# stations). v8 rewards all except `scout`; `scout` is the only penalised role.
MINER_ROLE_KEY = 'miner'
CAPTURER_ROLE_KEYS = ('aligner', 'scrambler')   # rewarded in stages B/C
PENALISED_ROLE_KEYS = ('scout',)                # discouraged throughout

# Heart resource token (crafted at the hub from deposited elements; the shared
# currency consumed to align/scramble junctions).
HEART_KEY = 'heart'

# Per-agent territory-capture stat keys (episode_stats['agent'][i]). Absent
# until the corresponding event first fires -> always read with .get(k, 0.0).
STAT_ALIGNED = 'junction.aligned_by_agent'
STAT_SCRAMBLED = 'junction.scrambled_by_agent'
STAT_HEART_GAINED = 'heart.gained'

# Haul round-trip detection (hysteresis on cargo fill fraction). Same mechanism
# as composite_v7: a load is "ready" once fill clears HIGH_FILL_FRAC and the
# round-trip completes once fill falls back below LOW_FILL_FRAC. In-place
# oscillation that never spans both thresholds counts for nothing.
HIGH_FILL_FRAC = 0.45
LOW_FILL_FRAC = 0.2
TARGET_ROUNDTRIPS = 3

# An agent "participated" if it mined anything OR explored at least this many
# distinct cells.
MIN_PARTICIPATION_CELLS = 5

DEFAULT_WEIGHTS_V8 = {
    # ---- ungated bootstrap ------------------------------------------------
    'w_explore':   2.0,    # efficient-exploration cold-start rung
    'explore_cap': 0.4,    # saturation cap so wandering can't replace real work
    'w_touch':     0.5,    # one-time bonus the first step `miner` is ever held
    'w_approach':  0.5,    # continuous gradient toward the c:miner station
    'w_idle':      1.0,    # penalty for an agent that neither mines nor explores
    'w_scout':     2.0,    # penalty for holding the scout role (only penalty)

    # ---- stage A: miner role + gated mine/haul/deposit --------------------
    'w_role_miner':  0.5,  # reward holding miner (a KEY that unlocks the gate)
    'w_miner_decay': 0.5,  # miner-role reward shrinks by up to this fraction as
                           #   deposit_progress rises ("reward the miner less")
    'w_mine':        8.0,  # mine-and-haul progress (round-trips)
    'w_deposit':    20.0,  # completed deposits -- a completed haul must pay big
    'deposit_pow':   0.5,  # sqrt curve: the FIRST completed haul is a big beacon
    'w_hold':        0.5,  # fill-pressure penalty: holding full cargo bleeds
    'w_carry':       1.5,  # continuous carry ramp (best load reached this episode)
    'gate_carry':    1.0,  # route this fraction of carry through effective_gate
    'gate_floor':    0.15, # stage-A soft gate floor (0 = hard miner gate)

    # ---- stage B: aligner/scrambler role, unlocked by depositing ----------
    'w_role2':       3.0,  # reward holding aligner OR scrambler (post-deposit)
    'role2_target':  0.30, # capturer_frac (fraction of episode) counted as full
                           #   commitment to a capturer role
    'b_floor':       0.10, # stage-B soft gate floor on deposit_progress

    # ---- stage C: hearts + territory, unlocked by a capturer role ---------
    'w_heart':          4.0,   # reward getting hearts (the align/scramble fuel)
    'heart_target':     3.0,   # hearts_gained counted as full heart progress
    'w_territory':     30.0,   # THE prize: capture/contest territory with hearts
    'territory_target': 3.0,   # blended align+scramble events counted as full
    'w_align_share':    1.0,   # aligning (neutral->team) is the true capture
    'w_scramble_share': 0.5,   # scrambling (contest enemy) counts, weighted less
    'c_floor':          0.10,  # stage-C soft gate floor on capturer commitment

    # ---- haul-detection hysteresis (consumed by the accumulator ctor) -----
    'high_fill_frac': HIGH_FILL_FRAC,
    'low_fill_frac':  LOW_FILL_FRAC,
    'target_roundtrips': TARGET_ROUNDTRIPS,

    # Reduction across agents: 'max'|'mean'|'min'|'trimmed_mean'|'top2_mean'|
    # 'bottom2_mean'. Default 'mean' while the full-loop behaviour is still
    # emerging; switch to 'bottom2_mean' once it reliably does to add the
    # participation/coordination rung.
    'reduction': 'mean',
}


# ---------------------------------------------------------------------------
# Obs parsing (self-contained copies of the v7 helpers)
# ---------------------------------------------------------------------------

def parse_inventory_at_center(obs, obs_hr=6, obs_wr=6):
    """Return dict {resource_name: int} for inv:* tokens at the obs center.

    The observing agent sits at the center of its HxW egocentric window; its own
    inventory tokens (role gear + cargo + hearts) are emitted at that center
    cell. Default (6,6) = center of the 13x13 window the cogames wheel emits.
    Callers pass the env-derived center explicitly.
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

    center = (H//2, W//2) read from env_cfg.game.obs.{height,width}; falls back
    to `default` (the 13x13 wheel's center). Load-bearing: an off-by-one here
    silently zeroes every per-step signal (role/cargo/heart holding).
    """
    try:
        obs = env_cfg.game.obs
        return (int(obs.height) // 2, int(obs.width) // 2)
    except Exception:
        return default


def min_chebyshev_to_tag(obs, tag_value, obs_hr, obs_wr):
    """Min Chebyshev distance from the obs center to any `tag` token whose value
    equals tag_value, or None if no such token is in view. Used to locate the
    c:miner gear-station for the approach-shaping bootstrap."""
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

class PerEpisodeAccumulatorV8:
    """One per agent. Tracks role holding (miner / aligner / scrambler /
    scout), heart holding, the mine/haul/deposit hysteresis, and cargo-fill
    pressure. Ctor signature mirrors PerEpisodeAccumulatorV7 so the evaluator
    setup code is identical modulo the module name."""

    def __init__(self, max_steps, quadrant_to_role=None,
                 cargo_capacity=DEFAULT_CARGO_CAPACITY, obs_center=None,
                 high_fill_frac=HIGH_FILL_FRAC, low_fill_frac=LOW_FILL_FRAC):
        del quadrant_to_role  # accepted for back-compat; not used (staged per-agent)
        self.max_steps = max(int(max_steps), 1)
        self.cargo_capacity = max(int(cargo_capacity), 1)

        hi = min(max(float(high_fill_frac), 1e-3), 1.0)
        lo = min(max(float(low_fill_frac), 0.0), hi - 1e-3)
        self.high_fill_frac = hi
        self.low_fill_frac = lo

        if obs_center is None:
            self._obs_hr, self._obs_wr = 6, 6
        else:
            self._obs_hr, self._obs_wr = int(obs_center[0]), int(obs_center[1])

        self.steps_seen = 0

        # Role holding (steps holding each gear token).
        self.miner_steps = 0
        self.aligner_steps = 0
        self.scrambler_steps = 0
        self.scout_steps = 0
        # Steps holding a capturer role (aligner/scrambler) AFTER the agent has
        # completed at least one deposit -- diagnostic proxy for "chose the
        # capturer role after returning and depositing" (the stage-B intent).
        self.capturer_steps_after_deposit = 0

        # Heart holding (per-step; the finalize also reads the heart.gained stat).
        self.heart_steps = 0
        self.heart_peak = 0

        # First-touch discovery flags.
        self.touched_miner = False

        # Mine/haul/deposit round-trips via cargo-fill hysteresis.
        self.completed_cycles = 0
        self.deposited_units = 0.0
        self.cargo_peak_since_dump = 0.0
        self.cargo_peak_ever = 0.0
        self._phase = 'fill'

        # Deposit pressure: running sum of per-step cargo-fill fraction.
        self.sum_fill_fraction = 0.0

        self._spawn_pos = None
        self._map_extents = None
        self._quadrant = 'NA'

        # Approach-to-miner-station shaping (stage-A bootstrap).
        self._miner_tag = None
        self._min_station_dist = None

    # ---- setup -----------------------------------------------------------

    def set_spawn(self, spawn_pos, map_extents):
        # Spawn / quadrant are diagnostic only.
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
        """Record the c:miner gear-station's distinguishing obs `tag` id so the
        per-step approach shaping can find it. Pass None to disable the term."""
        self._miner_tag = None if tag_id is None else int(tag_id)

    # ---- per-step --------------------------------------------------------

    def update(self, obs, obs_hr=None, obs_wr=None):
        hr = self._obs_hr if obs_hr is None else obs_hr
        wr = self._obs_wr if obs_wr is None else obs_wr
        inv = parse_inventory_at_center(obs, obs_hr=hr, obs_wr=wr)
        self.steps_seen += 1

        # Role holding. Gear tokens are mutually exclusive (each gear-station use
        # clears the gear bucket first), so at most one of these increments.
        holding_miner = inv.get(MINER_ROLE_KEY, 0) > 0
        if holding_miner:
            self.miner_steps += 1
            self.touched_miner = True
        if inv.get('aligner', 0) > 0:
            self.aligner_steps += 1
        if inv.get('scrambler', 0) > 0:
            self.scrambler_steps += 1
        if any(inv.get(k, 0) > 0 for k in PENALISED_ROLE_KEYS):
            self.scout_steps += 1
        holding_capturer = (inv.get('aligner', 0) > 0
                            or inv.get('scrambler', 0) > 0)
        if holding_capturer and self.completed_cycles >= 1:
            self.capturer_steps_after_deposit += 1

        # Heart holding.
        hearts = int(inv.get(HEART_KEY, 0))
        if hearts > 0:
            self.heart_steps += 1
        self.heart_peak = max(self.heart_peak, hearts)

        # Cargo-fill fraction -> deposit pressure + haul hysteresis.
        cargo = float(sum(inv.get(k, 0) for k in RESOURCE_KEYS))
        fill = min(cargo / self.cargo_capacity, 1.0)
        self.sum_fill_fraction += fill

        self.cargo_peak_since_dump = max(self.cargo_peak_since_dump, cargo)
        self.cargo_peak_ever = max(self.cargo_peak_ever, cargo)
        if self._phase == 'fill':
            if fill >= self.high_fill_frac:
                self._phase = 'dump'
        else:  # 'dump'
            if fill <= self.low_fill_frac:
                self.completed_cycles += 1
                self.deposited_units += min(self.cargo_peak_since_dump,
                                            float(self.cargo_capacity))
                self.cargo_peak_since_dump = cargo
                self._phase = 'fill'

        # Approach-to-station shaping (monotone best-ever, not farmable).
        if self._miner_tag is not None:
            d = min_chebyshev_to_tag(obs, self._miner_tag, hr, wr)
            if d is not None:
                self._min_station_dist = (d if self._min_station_dist is None
                                          else min(self._min_station_dist, d))

    # ---- finalize --------------------------------------------------------

    def episode_score(self, ep_reward_total, exploration_capped,
                      agent_stats=None, weights=None):
        """Combine accumulator state + end-of-episode agent stats into the v8
        staged score dict. `ep_reward_total` is kept for signature
        compatibility but is not part of the formula."""
        w = {**DEFAULT_WEIGHTS_V8, **(weights or {})}
        steps = max(self.steps_seen, 1)
        cap = float(self.cargo_capacity)
        stats = agent_stats or {}

        def _wf(key):
            return float(w.get(key, DEFAULT_WEIGHTS_V8[key]))

        # ---- fractions & progress -------------------------------------
        miner_fraction   = self.miner_steps   / steps
        scout_fraction   = self.scout_steps   / steps
        aligner_fraction = self.aligner_steps / steps
        scramble_frac    = self.scrambler_steps / steps
        capturer_frac    = (self.aligner_steps + self.scrambler_steps) / steps
        heart_hold_frac  = self.heart_steps / steps
        mean_fill        = self.sum_fill_fraction / steps
        touched_miner    = 1.0 if self.touched_miner else 0.0

        target = max(_wf('target_roundtrips'), 1e-6)
        total_extracted = self.deposited_units + min(self.cargo_peak_since_dump, cap)
        total_deposited = self.deposited_units
        mine_progress   = min(total_extracted / (cap * target), 1.0)
        deposit_pow     = max(_wf('deposit_pow'), 1e-6)
        deposit_progress = min((self.completed_cycles / target) ** deposit_pow, 1.0)
        carry_fraction  = min(self.cargo_peak_ever / cap, 1.0)

        # Territory + hearts from end-of-episode agent stats (.get => 0 if absent).
        aligned   = float(stats.get(STAT_ALIGNED, 0.0))
        scrambled = float(stats.get(STAT_SCRAMBLED, 0.0))
        hearts_gained = float(stats.get(STAT_HEART_GAINED, 0.0))
        territory_events = (_wf('w_align_share') * aligned
                            + _wf('w_scramble_share') * scrambled)
        territory_target = max(_wf('territory_target'), 1e-6)
        territory_progress = min(territory_events / territory_target, 1.0)
        heart_target = max(_wf('heart_target'), 1e-6)
        heart_progress = min(hearts_gained / heart_target, 1.0)

        role2_target  = max(_wf('role2_target'), 1e-6)
        capturer_commit = min(capturer_frac / role2_target, 1.0)

        # Participation -> idle penalty.
        unique_cells = float(stats.get('cell.unique_visited', 0.0))
        participated = 1.0 if (total_extracted > 0.0
                               or unique_cells >= MIN_PARTICIPATION_CELLS) else 0.0

        # Explore term, tightened by a saturation cap.
        explore_cap = _wf('explore_cap')
        exploration_capped = min(float(exploration_capped), explore_cap)

        # ---- soft staged gates ----------------------------------------
        gate_floor = min(max(_wf('gate_floor'), 0.0), 1.0)
        effective_gate = gate_floor + (1.0 - gate_floor) * miner_fraction
        b_floor = min(max(_wf('b_floor'), 0.0), 1.0)
        gateB = b_floor + (1.0 - b_floor) * deposit_progress
        c_floor = min(max(_wf('c_floor'), 0.0), 1.0)
        gateC = c_floor + (1.0 - c_floor) * capturer_commit

        # ---- bootstrap (ungated) --------------------------------------
        score_explore  = _wf('w_explore')  * exploration_capped
        score_touch    = _wf('w_touch')    * touched_miner
        score_idle     = -_wf('w_idle')    * (1.0 - participated)
        score_scout    = -_wf('w_scout')   * scout_fraction

        obs_radius = max(self._obs_hr, self._obs_wr, 1)
        if self._min_station_dist is None:
            station_proximity = 0.0
        else:
            station_proximity = max(0.0, (obs_radius - self._min_station_dist)
                                    / float(obs_radius))
        score_approach = _wf('w_approach') * station_proximity

        score_carry = _wf('w_carry') * carry_fraction
        gate_carry = min(max(_wf('gate_carry'), 0.0), 1.0)
        carry_ungated = (1.0 - gate_carry) * score_carry
        carry_gated   = gate_carry * score_carry

        # ---- stage A: miner role + gated mine/haul/deposit ------------
        miner_decay = 1.0 - min(max(_wf('w_miner_decay'), 0.0), 1.0) * deposit_progress
        score_role_miner = _wf('w_role_miner') * miner_fraction * miner_decay
        score_mine    = _wf('w_mine')    * mine_progress
        score_deposit = _wf('w_deposit') * deposit_progress
        score_hold    = -_wf('w_hold')   * mean_fill
        gated_mine = score_mine + score_deposit + score_hold + carry_gated

        # ---- stage B: capturer role, unlocked by depositing -----------
        score_role2 = gateB * _wf('w_role2') * capturer_frac

        # ---- stage C: hearts + territory, unlocked by capturer role ---
        score_heart     = gateB * gateC * _wf('w_heart')     * heart_progress
        score_territory = gateB * gateC * _wf('w_territory') * territory_progress

        bootstrap = (score_explore + score_touch + score_approach + score_idle
                     + score_scout + carry_ungated)
        score = (bootstrap
                 + score_role_miner
                 + effective_gate * gated_mine
                 + score_role2
                 + score_heart + score_territory)

        return {
            'score':                float(score),
            'effective_gate':       float(effective_gate),
            'gateB':                float(gateB),
            'gateC':                float(gateC),
            # bootstrap
            'score_explore':        float(score_explore),
            'score_touch':          float(score_touch),
            'score_approach':       float(score_approach),
            'station_proximity':    float(station_proximity),
            'score_idle':           float(score_idle),
            'score_scout':          float(score_scout),
            'score_carry':          float(score_carry),
            'carry_fraction':       float(carry_fraction),
            # stage A
            'score_role_miner':     float(score_role_miner),
            'score_mine':           float(score_mine),
            'score_deposit':        float(score_deposit),
            'score_hold':           float(score_hold),
            'mine_progress':        float(mine_progress),
            'deposit_progress':     float(deposit_progress),
            'completed_cycles':     float(self.completed_cycles),
            'total_extracted':      float(total_extracted),
            'total_deposited':      float(total_deposited),
            'mean_fill_fraction':   float(mean_fill),
            # stage B
            'score_role2':          float(score_role2),
            'capturer_fraction':    float(capturer_frac),
            'capturer_commit':      float(capturer_commit),
            'capturer_after_deposit_frac': float(self.capturer_steps_after_deposit / steps),
            # stage C
            'score_heart':          float(score_heart),
            'score_territory':      float(score_territory),
            'heart_progress':       float(heart_progress),
            'hearts_gained':        float(hearts_gained),
            'heart_hold_frac':      float(heart_hold_frac),
            'territory_progress':   float(territory_progress),
            'junctions_aligned':    float(aligned),
            'junctions_scrambled':  float(scrambled),
            # role fractions
            'miner_fraction':       float(miner_fraction),
            'aligner_fraction':     float(aligner_fraction),
            'scrambler_fraction':   float(scramble_frac),
            'scout_fraction':       float(scout_fraction),
            'touched_miner':        float(touched_miner),
            'participated':         float(participated),
            'exploration_capped':   float(exploration_capped),
            'ep_reward':            float(ep_reward_total),
            'spawn_pos':            self._spawn_pos,
            'quadrant':             self._quadrant,
        }


# ---------------------------------------------------------------------------
# Component fields that are always mean-reduced for diagnostic readability.
# ---------------------------------------------------------------------------

COMPONENT_KEYS = (
    'effective_gate', 'gateB', 'gateC',
    'score_explore', 'score_touch', 'score_approach', 'station_proximity',
    'score_idle', 'score_scout', 'score_carry', 'carry_fraction',
    'score_role_miner', 'score_mine', 'score_deposit', 'score_hold',
    'mine_progress', 'deposit_progress', 'completed_cycles',
    'total_extracted', 'total_deposited', 'mean_fill_fraction',
    'score_role2', 'capturer_fraction', 'capturer_commit',
    'capturer_after_deposit_frac',
    'score_heart', 'score_territory', 'heart_progress', 'hearts_gained',
    'heart_hold_frac', 'territory_progress', 'junctions_aligned',
    'junctions_scrambled',
    'miner_fraction', 'aligner_fraction', 'scrambler_fraction',
    'scout_fraction', 'touched_miner', 'participated',
    'exploration_capped', 'ep_reward',
)


# ---------------------------------------------------------------------------
# Cross-agent aggregation
# ---------------------------------------------------------------------------

def aggregate_agent_results_v8(per_agent_results, reduction='mean'):
    """Reduce a list of per-agent v8 dicts into one aggregate dict.

    `score` uses the configured reduction:
      'max' | 'mean' | 'min' | 'trimmed_mean' | 'top2_mean' | 'bottom2_mean'.
    Component fields are always mean-reduced. score_mean/min/max/std and
    best_agent_idx are always reported.
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

def aggregate_over_episodes_v8(per_episode_aggregates):
    """Reduce a list of per-episode aggregates into one over-all-episodes
    aggregate. `score` is the mean across episodes; component fields are
    mean-reduced."""
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
    out['reduction']  = per_episode_aggregates[0].get('reduction', 'mean')

    for k in COMPONENT_KEYS:
        vals = [a[k] for a in per_episode_aggregates if k in a]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# Diagnostic stdout (regex fallback for evaluators not using sidecar)
# ---------------------------------------------------------------------------

def format_diagnostic_lines_v8(agg, ep_reward_total):
    """Multi-line v8 diagnostic block. Lines begin with 'v8:' so the task
    drivers' regex parser can pick up ep_reward."""
    spawn = agg.get('spawn_pos')
    spawn_str = f'({spawn[0]},{spawn[1]})' if spawn else 'NA'
    line1 = (
        f'  v8: spawn={spawn_str} quadrant={agg.get("quadrant", "NA")} '
        f'best_agent={agg.get("best_agent_idx", -1)} '
        f'reduction={agg.get("reduction", "mean")}'
    )
    line2 = (
        f'  v8: miner_frac={agg.get("miner_fraction", 0):.3f} '
        f'align_frac={agg.get("aligner_fraction", 0):.3f} '
        f'scram_frac={agg.get("scrambler_fraction", 0):.3f} '
        f'scout_frac={agg.get("scout_fraction", 0):.3f} '
        f'cycles={agg.get("completed_cycles", 0):.2f} '
        f'deposited={agg.get("total_deposited", 0):.1f} '
        f'hearts={agg.get("hearts_gained", 0):.2f} '
        f'aligned={agg.get("junctions_aligned", 0):.2f} '
        f'scrambled={agg.get("junctions_scrambled", 0):.2f} '
        f'explore={agg.get("exploration_capped", 0):.3f}'
    )
    line3 = (
        f'  v8: gate={agg.get("effective_gate", 0):.3f} '
        f'gateB={agg.get("gateB", 0):.3f} gateC={agg.get("gateC", 0):.3f} '
        f'role_miner={agg.get("score_role_miner", 0):.3f} '
        f'mine={agg.get("score_mine", 0):.3f} '
        f'deposit={agg.get("score_deposit", 0):.3f} '
        f'role2={agg.get("score_role2", 0):.3f} '
        f'heart={agg.get("score_heart", 0):.3f} '
        f'terr={agg.get("score_territory", 0):.3f} '
        f'scout={agg.get("score_scout", 0):.3f}'
    )
    line4 = (
        f'  v8: ep_reward={float(ep_reward_total):.4f} '
        f'score_mean={agg.get("score_mean", 0):.4f} '
        f'-> score={agg.get("score", 0):.4f}'
    )
    return '\n'.join([line1, line2, line3, line4])


# ---------------------------------------------------------------------------
# Sidecar JSON write (atomic)
# ---------------------------------------------------------------------------

def write_component_sidecar(path, per_episode_components, aggregate,
                            fitness_stat='composite_v8_zstat'):
    """Write per-episode + aggregate component dicts to a JSON sidecar.
    Atomic via temp + os.replace. No-op if `path` is empty / None."""
    if not path:
        return
    payload = {
        'version':      'v8.0',
        'fitness_stat': fitness_stat,
        'n_episodes':   len(per_episode_components),
        'aggregate':    aggregate,
        'per_episode':  per_episode_components,
    }
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.components_v8_', suffix='.json.tmp', dir=parent)
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
    """Parse 'k1=v1,k2=v2,...' from a CLI flag into a dict. Float values are
    stored as float; the special key 'reduction' is preserved as a string."""
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
# Spawn-position / map / station helpers for evaluators
# ---------------------------------------------------------------------------

def get_agent_spawn_positions(sim, num_agents):
    """Return [(row, col)] for each agent_id 0..num_agents-1, or None if absent."""
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
    """Return the c:miner gear-station's distinguishing obs `tag` id, or None."""
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

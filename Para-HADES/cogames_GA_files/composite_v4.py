"""composite_v4_role_gated fitness scoring (shared across evaluators).

Per-step gear-gated tracking of role-specific progress + the v3 exploration
behaviour term + a spawn-quadrant role-match diversity term + a target-
agnostic role-commitment term. Episode score across agents is reduced by
`max` (best-of-N agents), so a genome's fitness reflects its top agent.

Pipeline:
  1. evaluator constructs PerEpisodeAccumulator(max_steps) per agent
  2. evaluator calls acc.set_spawn((row, col), (R, C)) right after rollout init
  3. evaluator wraps each policy.step so it calls acc.update(obs) before
     forwarding to the underlying network
  4. after rollout.run_until_done(), evaluator calls
     acc.episode_score(ep_reward_total, exploration_capped, weights)

Per-role progress is built from inventory deltas (observable from obs alone):
  - miner    : resources accumulated while wearing decoder gear
  - aligner  : hearts spent / hearts gained while wearing modulator gear
  - scrambler: hearts spent + scaled distance while wearing scrambler gear
  - scout    : scaled unique-cell coverage + scaled max-distance while
               wearing resonator gear

Distance/coverage signals are scaled by the fraction of steps the agent spent
in that role (gear-fraction approximation), since cell.unique_visited and
cell.max_distance_from_spawn are only available as end-of-episode aggregates.
"""

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    'w_M':      1.0,
    'w_A':      1.0,
    'w_S':      1.0,
    'w_Sc':     1.0,
    'w_exp':    0.4,
    'w_init':   1.0,
    'w_commit': 1.5,
    'w_ep':     2.0,
    # Gear-independent milestone tier (early-gen gradient)
    'w_ms_res':    0.5,
    'w_ms_gear':   0.5,
    'w_ms_heart':  0.5,
    'w_ms_spent':  0.5,
    'w_engage':    0.4,
    'w_chain':     0.5,
    'alpha_M_res':  0.05,
    'alpha_M_dep':  1.0,
    'alpha_A_with': 1.0,
    'alpha_A_assm': 0.5,
    'alpha_S_with': 1.0,
    'alpha_S_far':  0.5,
    'alpha_Sc_unq': 1.0,
    'alpha_Sc_far': 0.5,
}

DEFAULT_QUADRANT_TO_ROLE = {
    ('N', 'W'): 'miner',
    ('N', 'E'): 'aligner',
    ('S', 'W'): 'scrambler',
    ('S', 'E'): 'scout',
}

ROLE_TO_GEAR_KEY = {
    'miner':     'decoder',
    'aligner':   'modulator',
    'scrambler': 'scrambler',
    'scout':     'resonator',
}

ROLES = ('miner', 'aligner', 'scrambler', 'scout')


# ---------------------------------------------------------------------------
# Obs parsing
# ---------------------------------------------------------------------------

def parse_inventory_at_center(obs, obs_hr=5, obs_wr=5):
    """Return dict {resource_name: int} for inv:* tokens at obs center."""
    inv = {}
    for tok in obs.tokens:
        if tok.location == (obs_hr, obs_wr):
            name = tok.feature.name
            if name.startswith('inv:'):
                inv[name[4:]] = int(tok.value)
    return inv


def determine_gear(inv):
    """Return one of 'miner'|'aligner'|'scrambler'|'scout'|None.

    Priority if multiple gears equipped (env shouldn't allow it but guard):
    scrambler > resonator > decoder > modulator.
    """
    if inv.get('scrambler', 0) > 0:
        return 'scrambler'
    if inv.get('resonator', 0) > 0:
        return 'scout'
    if inv.get('decoder', 0) > 0:
        return 'miner'
    if inv.get('modulator', 0) > 0:
        return 'aligner'
    return None


# ---------------------------------------------------------------------------
# v3 behaviour (re-used as the always-on exploration term in v4)
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

class PerEpisodeAccumulator:
    """One per agent. Updated each step with the agent's observation."""

    def __init__(self, max_steps, quadrant_to_role=None):
        self.max_steps = max(int(max_steps), 1)
        self.quadrant_to_role = quadrant_to_role or DEFAULT_QUADRANT_TO_ROLE

        self.steps_seen = 0
        self.gear_step_count = Counter()

        # Inventory-delta-driven role progress sums
        self.miner_res    = 0.0   # sum of positive Δ(carbon+oxy+germ+sili) while miner
        self.miner_dep    = 0.0   # sum of |negative Δresources| while miner (deposit proxy)
        self.aligner_with = 0.0   # sum of |negative Δhearts| while aligner
        self.aligner_assm = 0.0   # sum of positive Δhearts while aligner
        self.scrambler_with = 0.0 # sum of |negative Δhearts| while scrambler

        # Gear-independent milestone counters (dense early-gen gradient)
        self.any_res_gain = 0.0           # Σ positive Δresources, gear-agnostic
        self.any_heart_gain = 0.0         # Σ positive Δhearts, gear-agnostic
        self.any_heart_loss = 0.0         # Σ |negative Δhearts|, gear-agnostic
        self.any_gear_equipped_ever = False  # any role gear ever in inventory

        self._prev_inv = None
        self._spawn_pos = None
        self._map_extents = None
        self._target_role = None
        self._quadrant = 'NA'

    # ---- setup -----------------------------------------------------------

    def set_spawn(self, spawn_pos, map_extents):
        """Record (row, col) spawn and (R, C) map extents.

        Computes the spawn quadrant immediately; target_role is the role the
        agent is incentivised to equip on this episode.
        """
        if spawn_pos is None or map_extents is None:
            return
        self._spawn_pos = (int(spawn_pos[0]), int(spawn_pos[1]))
        R, C = map_extents
        self._map_extents = (int(R), int(C))
        r0, c0 = self._spawn_pos
        ns = 'N' if r0 < self._map_extents[0] / 2 else 'S'
        we = 'W' if c0 < self._map_extents[1] / 2 else 'E'
        self._quadrant = ns + we
        self._target_role = self.quadrant_to_role.get((ns, we))

    # ---- per-step --------------------------------------------------------

    def update(self, obs, obs_hr=5, obs_wr=5):
        """Read inventory from obs, update gear histogram and Δ-driven sums."""
        inv = parse_inventory_at_center(obs, obs_hr=obs_hr, obs_wr=obs_wr)
        gear = determine_gear(inv)
        self.gear_step_count[gear if gear is not None else 'none'] += 1
        self.steps_seen += 1

        if gear is not None:
            self.any_gear_equipped_ever = True

        if self._prev_inv is not None:
            res_now = (inv.get('carbon', 0) + inv.get('oxygen', 0)
                       + inv.get('germanium', 0) + inv.get('silicon', 0))
            res_prev = (self._prev_inv.get('carbon', 0) + self._prev_inv.get('oxygen', 0)
                        + self._prev_inv.get('germanium', 0) + self._prev_inv.get('silicon', 0))
            d_res = res_now - res_prev

            hearts_now = inv.get('heart', 0)
            hearts_prev = self._prev_inv.get('heart', 0)
            d_hearts = hearts_now - hearts_prev

            # Gear-independent milestone accumulation (early-gen gradient).
            if d_res > 0:
                self.any_res_gain += d_res
            if d_hearts > 0:
                self.any_heart_gain += d_hearts
            elif d_hearts < 0:
                self.any_heart_loss += -d_hearts

            if gear == 'miner':
                if d_res > 0:
                    self.miner_res += d_res
                elif d_res < 0:
                    # Resources leaving inventory while in miner gear is a
                    # deposit-like event (chest deposit or assembler input).
                    self.miner_dep += -d_res
            elif gear == 'aligner':
                if d_hearts < 0:
                    self.aligner_with += -d_hearts
                elif d_hearts > 0:
                    self.aligner_assm += d_hearts
            elif gear == 'scrambler':
                if d_hearts < 0:
                    self.scrambler_with += -d_hearts

        self._prev_inv = inv

    # ---- finalize --------------------------------------------------------

    def episode_score(self, ep_reward_total, exploration_capped,
                      agent_stats=None, weights=None):
        """Combine accumulator state + final stats into the v4 score.

        agent_stats: this agent's entry from sim.episode_stats['agent'] (dict).
                     Used to read end-of-episode cell.unique_visited and
                     cell.max_distance_from_spawn for the scout/scrambler
                     distance/coverage components, scaled by the fraction of
                     steps the agent spent in the relevant role.
        """
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        ms = float(self.max_steps)
        steps = max(self.steps_seen, 1)

        # Gear fractions (proportion of steps in each role)
        frac = {role: self.gear_step_count.get(role, 0) / steps
                for role in ROLES}

        # End-of-episode exploration aggregates (used for scout/scrambler
        # distance & scout coverage, gated by gear fraction).
        unique_total = 0.0
        dist_total   = 0.0
        if agent_stats is not None:
            unique_total = float(agent_stats.get('cell.unique_visited', 0.0))
            dist_total   = float(agent_stats.get('cell.max_distance_from_spawn', 0.0))

        # ---- per-role normalised scores (capped at 1.0 each) -------------
        score_miner = (
            w['alpha_M_res'] * (self.miner_res / ms)
            + w['alpha_M_dep'] * (self.miner_dep / ms)
        )
        score_aligner = (
            w['alpha_A_with'] * (self.aligner_with / ms)
            + w['alpha_A_assm'] * (self.aligner_assm / ms)
        )
        score_scrambler = (
            w['alpha_S_with'] * (self.scrambler_with / ms)
            + w['alpha_S_far']  * frac['scrambler'] * (dist_total / (ms ** 0.5))
        )
        score_scout = (
            w['alpha_Sc_unq'] * frac['scout'] * (unique_total / ms)
            + w['alpha_Sc_far'] * frac['scout'] * (dist_total / (ms ** 0.5))
        )

        score_miner     = min(max(score_miner, 0.0), 1.0)
        score_aligner   = min(max(score_aligner, 0.0), 1.0)
        score_scrambler = min(max(score_scrambler, 0.0), 1.0)
        score_scout     = min(max(score_scout, 0.0), 1.0)

        # ---- spawn-quadrant role-match (initial_location) ----------------
        if self._target_role is not None:
            init_loc = self.gear_step_count.get(self._target_role, 0) / ms
        else:
            init_loc = 0.0
        init_loc = min(max(init_loc, 0.0), 1.0)

        # ---- role commitment (any single role, target-agnostic) ----------
        # Fraction of steps spent in whichever role the agent committed to
        # most. Rewards "picked a role and stuck with it" independent of
        # whether it matched the spawn-quadrant target.
        role_commit = max((frac[r] for r in ROLES), default=0.0)
        role_commit = min(max(role_commit, 0.0), 1.0)

        # ---- gear-independent milestone tier ----------------------------
        # Dense early-gen gradient: each rung of the natural progression
        # chain (resource → gear → heart-in-inv → heart-spent) gives credit
        # without conditioning on what gear the agent currently wears.
        ms_resource_any = min(self.any_res_gain / 4.0, 1.0)
        ms_gear_any     = 1.0 if self.any_gear_equipped_ever else 0.0
        ms_heart_inv    = min(self.any_heart_gain / 2.0, 1.0)
        ms_heart_spent  = min(self.any_heart_loss / 2.0, 1.0)
        engagement_any = (
            0.25 * float(self.any_res_gain > 0)
            + 0.25 * ms_gear_any
            + 0.25 * float(self.any_heart_gain > 0)
            + 0.25 * float(self.any_heart_loss > 0)
        )
        chain_complete = 1.0 if (
            self.any_res_gain > 0
            and self.any_gear_equipped_ever
            and self.any_heart_gain > 0
            and self.any_heart_loss > 0
        ) else 0.0

        # ---- final combine -----------------------------------------------
        ep_reward_clipped = max(float(ep_reward_total), -1.0)
        score = (
            w['w_M']      * score_miner
            + w['w_A']      * score_aligner
            + w['w_S']      * score_scrambler
            + w['w_Sc']     * score_scout
            + w['w_exp']    * float(exploration_capped) * role_commit
            + w['w_init']   * init_loc
            + w['w_commit'] * role_commit
            + w['w_ep']     * ep_reward_clipped
            + w['w_ms_res']    * ms_resource_any
            + w['w_ms_gear']   * ms_gear_any
            + w['w_ms_heart']  * ms_heart_inv
            + w['w_ms_spent']  * ms_heart_spent
            + w['w_engage']    * engagement_any
            + w['w_chain']     * chain_complete
        )

        return {
            'score': float(score),
            'score_miner': float(score_miner),
            'score_aligner': float(score_aligner),
            'score_scrambler': float(score_scrambler),
            'score_scout': float(score_scout),
            'exploration_capped': float(exploration_capped),
            'init_loc': float(init_loc),
            'role_commit': float(role_commit),
            'ep_reward_clipped': float(ep_reward_clipped),
            'ms_resource_any': float(ms_resource_any),
            'ms_gear_any':     float(ms_gear_any),
            'ms_heart_inv':    float(ms_heart_inv),
            'ms_heart_spent':  float(ms_heart_spent),
            'engagement_any':  float(engagement_any),
            'chain_complete':  float(chain_complete),
            'gear_share': {k: self.gear_step_count.get(k, 0) / steps
                           for k in ('miner', 'aligner', 'scrambler', 'scout', 'none')},
            'spawn_pos': self._spawn_pos,
            'quadrant': self._quadrant,
            'target_role': self._target_role,
            'role_match_frac': float(init_loc),
        }


# ---------------------------------------------------------------------------
# Episode-level aggregation across agents and diagnostic formatting
# ---------------------------------------------------------------------------

def aggregate_agent_results(per_agent_results):
    """Reduce a list of per-agent result dicts into one dict.

    The single value the GA consumes is `score`; we reduce that with `max`
    so the genome is selected by its best agent. Component scores are
    mean-reduced for diagnostic readability and also surface `score_mean`
    + `best_agent_idx` for monitoring.
    """
    if not per_agent_results:
        return {}
    score_vals = [r['score'] for r in per_agent_results if 'score' in r]
    keys_numeric_mean = (
        'score_miner', 'score_aligner', 'score_scrambler', 'score_scout',
        'exploration_capped', 'init_loc', 'role_commit',
        'ep_reward_clipped', 'role_match_frac',
        'ms_resource_any', 'ms_gear_any', 'ms_heart_inv', 'ms_heart_spent',
        'engagement_any', 'chain_complete',
    )
    out = {}
    if score_vals:
        out['score'] = float(np.max(score_vals))
        out['score_mean'] = float(np.mean(score_vals))
        out['best_agent_idx'] = int(np.argmax(score_vals))
    else:
        out['score'] = 0.0
        out['score_mean'] = 0.0
        out['best_agent_idx'] = -1
    for k in keys_numeric_mean:
        vals = [r[k] for r in per_agent_results if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0
    # gear_share: mean across agents
    gs_keys = ('miner', 'aligner', 'scrambler', 'scout', 'none')
    out['gear_share'] = {
        k: float(np.mean([r['gear_share'].get(k, 0.0) for r in per_agent_results]))
        for k in gs_keys
    }
    # Best agent's spawn / quadrant / target_role (for the diagnostic line —
    # the agent that earned the genome's score).
    best = per_agent_results[out['best_agent_idx']] if score_vals else per_agent_results[0]
    out['spawn_pos']   = best.get('spawn_pos')
    out['quadrant']    = best.get('quadrant')
    out['target_role'] = best.get('target_role')
    return out


def format_diagnostic_lines(agg, ep_reward_total):
    """Return the multi-line v4 diagnostic block (printed by each evaluator).

    The first line begins with 'v4:' so the regex parser in task .py picks up
    ep_reward, mirroring the v2/v3 contract.
    """
    gs = agg.get('gear_share', {})
    spawn = agg.get('spawn_pos')
    spawn_str = f'({spawn[0]},{spawn[1]})' if spawn else 'NA'
    line1 = (
        f'  v4: gear_share=miner:{gs.get("miner", 0):.2f},'
        f'aligner:{gs.get("aligner", 0):.2f},'
        f'scrambler:{gs.get("scrambler", 0):.2f},'
        f'scout:{gs.get("scout", 0):.2f},'
        f'none:{gs.get("none", 0):.2f}'
    )
    line2 = (
        f'  v4: spawn={spawn_str} quadrant={agg.get("quadrant", "NA")} '
        f'target_role={agg.get("target_role")} '
        f'role_match_frac={agg.get("role_match_frac", 0):.3f} '
        f'role_commit={agg.get("role_commit", 0):.3f} '
        f'best_agent={agg.get("best_agent_idx", -1)}'
    )
    line3 = (
        f'  v4: score_miner={agg.get("score_miner", 0):.4f} '
        f'score_aligner={agg.get("score_aligner", 0):.4f} '
        f'score_scrambler={agg.get("score_scrambler", 0):.4f} '
        f'score_scout={agg.get("score_scout", 0):.4f}'
    )
    line4 = (
        f'  v4: exploration={agg.get("exploration_capped", 0):.4f} '
        f'init_loc={agg.get("init_loc", 0):.4f} '
        f'ep_reward={float(ep_reward_total):.4f} '
        f'ep_reward_clipped={agg.get("ep_reward_clipped", 0):.4f} '
        f'score_mean={agg.get("score_mean", 0):.4f} '
        f'-> score={agg.get("score", 0):.4f}'
    )
    line5 = (
        f'  v4: ms_res={agg.get("ms_resource_any", 0):.3f} '
        f'ms_gear={agg.get("ms_gear_any", 0):.3f} '
        f'ms_heart={agg.get("ms_heart_inv", 0):.3f} '
        f'ms_spent={agg.get("ms_heart_spent", 0):.3f} '
        f'engage={agg.get("engagement_any", 0):.3f} '
        f'chain={agg.get("chain_complete", 0):.3f}'
    )
    return '\n'.join([line1, line2, line3, line4, line5])


# ---------------------------------------------------------------------------
# CLI weight parsing
# ---------------------------------------------------------------------------

def parse_weights_str(s):
    """Parse 'k1=v1,k2=v2,...' from a CLI flag into a dict of floats."""
    out = {}
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        k = k.strip()
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

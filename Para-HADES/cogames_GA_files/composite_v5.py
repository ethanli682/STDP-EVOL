"""composite_v5_role_event fitness scoring.

Successor to composite_v4_role_gated. Lives beside v4 (do not edit v4 in place
because running CMA-ES checkpoints reference its scoring). Switch consumption
via YAML `fitness_stat: composite_v5_role_event`.

Differences from v4:

  - F + B decomposition (MATE-style).
      score_F = sparse delivery   (hearts produced, chest deposit, env reward)
      score_B = dense bounty      (achievements, role events, exploration)
      score   = score_F + score_B (B has a hard cap so it cannot drown F)

  - Achievement flags. One-shot booleans per agent per episode:
      ach_first_ore, ach_first_gear, ach_first_role_event_{role},
      ach_first_heart, ach_first_heart_spent.

  - Role-distinctive events. role_event_count_{role} = positive deltas in the
    matching gear item's inventory. Per stations.py, these can ONLY come from
    the assembler firing its gear-crafting protocol (vibes=["gear","X_a"],
    input={X:1}, output={gear_item:1}) — a true env-side role credit, not a
    position+gear-fraction heuristic. Replaces the gameable v4 score_miner /
    score_aligner / score_scrambler / score_scout.

  - Dropped: ms_gear_any (one-shot saturation), role_commit (sit-in-one-role
    farming), engagement_any (partial-flag farming).

  - Per-component persistence via JSON sidecar (write_component_sidecar). The
    optimizer driver reads this directly instead of regex-parsing stdout, so
    every component gets logged per genome.

  - Reduction across agents is configurable: 'max' (v4 default, retained for
    parity), 'mean', or 'min'. init_loc gets weight 0 under 'max' (one agent's
    score wins, spawn-quadrant routing doesn't drive selection there).

Pipeline mirrors v4:
  1. evaluator constructs PerEpisodeAccumulatorV5(max_steps) per agent.
  2. evaluator calls acc.set_spawn(...) post rollout init.
  3. evaluator wraps policy.step so it calls acc.update(obs) before forwarding.
  4. after rollout.run_until_done(), evaluator calls
     acc.episode_score(ep_reward_total, exploration_capped, ...).
  5. evaluator aggregates across agents via aggregate_agent_results_v5 and
     (optionally) writes the per-episode + aggregate components to a sidecar
     via write_component_sidecar.
"""

import json
import os
import tempfile
from collections import Counter

import numpy as np

from composite_v4 import (
    DEFAULT_QUADRANT_TO_ROLE,
    ROLE_TO_GEAR_KEY,
    ROLES,
    compute_v3_behaviour,
    determine_gear,
    get_agent_spawn_positions,
    get_map_extents,
    parse_inventory_at_center,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS_V5 = {
    # F: sparse delivery
    'w_F_heart':       5.0,   # per heart produced (cumulative)
    'w_F_chest':       5.0,   # per chest-deposited heart (game-level / num_agents share)
    'w_F_ep':          3.0,   # env reward (clipped at -1)

    # B: dense bounty (capped at score_B_cap)
    'w_B_ach_ore':     0.3,   # one-shot
    'w_B_ach_gear':    0.5,   # one-shot
    'w_B_role_events': 0.2,   # per role event, summed across roles, clamped to 5
    'w_B_explore':     0.4,
    'w_B_init':        0.0,   # default 0 because we ship reduction='max'; flip to 0.5 with 'mean'
    'score_B_cap':     3.0,

    # Reduction across agents: 'max' | 'mean' | 'min'
    'reduction':       'max',
}

ROLE_EVENT_COUNT_KEYS = tuple(f'role_event_count_{r}' for r in ROLES)
ACHIEVEMENT_KEYS = (
    'ach_first_ore',
    'ach_first_gear',
    'ach_first_heart',
    'ach_first_heart_spent',
    'ach_first_role_event_miner',
    'ach_first_role_event_aligner',
    'ach_first_role_event_scrambler',
    'ach_first_role_event_scout',
)


# ---------------------------------------------------------------------------
# Per-episode accumulator
# ---------------------------------------------------------------------------

class PerEpisodeAccumulatorV5:
    """One per agent. Updated each step with the agent's observation."""

    def __init__(self, max_steps, quadrant_to_role=None):
        self.max_steps = max(int(max_steps), 1)
        self.quadrant_to_role = quadrant_to_role or DEFAULT_QUADRANT_TO_ROLE

        self.steps_seen = 0
        self.gear_step_count = Counter()

        # Role-distinctive event counters (positive gear-inventory deltas).
        self.role_event_count = {role: 0 for role in ROLES}
        self.first_role_event = {role: False for role in ROLES}
        self.first_role_event_step = {role: -1 for role in ROLES}

        # Resource / heart event counters.
        self.heart_gain_count = 0
        self.heart_spend_count = 0
        self.first_ore = False
        self.first_gear = False
        self.first_heart = False
        self.first_heart_spent = False
        self.any_gear_equipped_ever = False

        self._prev_inv = None
        self._spawn_pos = None
        self._map_extents = None
        self._target_role = None
        self._quadrant = 'NA'

    # ---- setup -----------------------------------------------------------

    def set_spawn(self, spawn_pos, map_extents):
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
        inv = parse_inventory_at_center(obs, obs_hr=obs_hr, obs_wr=obs_wr)
        gear = determine_gear(inv)
        self.gear_step_count[gear if gear is not None else 'none'] += 1
        self.steps_seen += 1

        if gear is not None:
            self.any_gear_equipped_ever = True

        if self._prev_inv is not None:
            # Ore deltas (any of 4 raw resources).
            d_ore = sum(
                max(inv.get(r, 0) - self._prev_inv.get(r, 0), 0)
                for r in ('carbon', 'oxygen', 'germanium', 'silicon')
            )
            if d_ore > 0 and not self.first_ore:
                self.first_ore = True

            # Gear-item deltas → role events. Positive delta in `decoder` /
            # `modulator` / `scrambler` / `resonator` inventory can ONLY come
            # from the assembler's gear-crafting protocol firing while THIS
            # agent supplied the input resource (stations.py:231-235).
            for role, gear_key in ROLE_TO_GEAR_KEY.items():
                d_gear = inv.get(gear_key, 0) - self._prev_inv.get(gear_key, 0)
                if d_gear > 0:
                    self.role_event_count[role] += int(d_gear)
                    if not self.first_role_event[role]:
                        self.first_role_event[role] = True
                        self.first_role_event_step[role] = self.steps_seen
                    if not self.first_gear:
                        self.first_gear = True

            # Heart deltas.
            d_heart = inv.get('heart', 0) - self._prev_inv.get('heart', 0)
            if d_heart > 0:
                self.heart_gain_count += int(d_heart)
                if not self.first_heart:
                    self.first_heart = True
            elif d_heart < 0:
                self.heart_spend_count += int(-d_heart)
                if not self.first_heart_spent:
                    self.first_heart_spent = True

        self._prev_inv = inv

    # ---- finalize --------------------------------------------------------

    def episode_score(self, ep_reward_total, exploration_capped,
                      agent_stats=None, weights=None):
        """Combine accumulator state into the v5 score dict.

        `chest_heart_deposited_share` was removed: chest deposits are not
        tracked as a separate game-level stat in cogames. The per-agent
        `heart_spend_count` (negative heart inventory delta) is the cleanest
        chest-deposit proxy — in arena the only routine way to lose a heart
        is the chest's heart_b transfer protocol.
        """
        w = {**DEFAULT_WEIGHTS_V5, **(weights or {})}
        ms = float(self.max_steps)
        steps = max(self.steps_seen, 1)

        frac = {role: self.gear_step_count.get(role, 0) / steps for role in ROLES}

        ach_first_ore = 1.0 if self.first_ore else 0.0
        ach_first_gear = 1.0 if self.first_gear else 0.0
        ach_first_heart = 1.0 if self.first_heart else 0.0
        ach_first_heart_spent = 1.0 if self.first_heart_spent else 0.0
        ach_first_role_event = {
            role: (1.0 if self.first_role_event[role] else 0.0)
            for role in ROLES
        }

        # Soft role gating: spawn-quadrant target gear-fraction.
        if self._target_role is not None:
            init_loc = self.gear_step_count.get(self._target_role, 0) / ms
        else:
            init_loc = 0.0
        init_loc = min(max(init_loc, 0.0), 1.0)

        ep_reward_clipped = max(float(ep_reward_total), -1.0)

        score_F = (
            w['w_F_heart'] * float(self.heart_gain_count)
            + w['w_F_chest'] * float(self.heart_spend_count)
            + w['w_F_ep'] * ep_reward_clipped
        )

        role_event_total = sum(self.role_event_count.values())
        role_event_total_clipped = min(role_event_total, 5)
        score_B_raw = (
            w['w_B_ach_ore'] * ach_first_ore
            + w['w_B_ach_gear'] * ach_first_gear
            + w['w_B_role_events'] * float(role_event_total_clipped)
            + w['w_B_explore'] * float(exploration_capped)
            + w['w_B_init'] * init_loc
        )
        score_B = min(max(score_B_raw, 0.0), float(w['score_B_cap']))

        score = score_F + score_B

        return {
            'score': float(score),
            'score_F': float(score_F),
            'score_B': float(score_B),
            'F_over_total': float(score_F / max(score_F + score_B, 1e-9)),
            'ach_first_ore': ach_first_ore,
            'ach_first_gear': ach_first_gear,
            'ach_first_heart': ach_first_heart,
            'ach_first_heart_spent': ach_first_heart_spent,
            'ach_first_role_event_miner':     ach_first_role_event['miner'],
            'ach_first_role_event_aligner':   ach_first_role_event['aligner'],
            'ach_first_role_event_scrambler': ach_first_role_event['scrambler'],
            'ach_first_role_event_scout':     ach_first_role_event['scout'],
            'role_event_count_miner':     int(self.role_event_count['miner']),
            'role_event_count_aligner':   int(self.role_event_count['aligner']),
            'role_event_count_scrambler': int(self.role_event_count['scrambler']),
            'role_event_count_scout':     int(self.role_event_count['scout']),
            'heart_gain_count':  int(self.heart_gain_count),
            'heart_spend_count': int(self.heart_spend_count),
            'ep_reward_clipped': float(ep_reward_clipped),
            'exploration_capped': float(exploration_capped),
            'init_loc':          float(init_loc),
            'role_event_total':  int(role_event_total),
            'gear_share': {k: self.gear_step_count.get(k, 0) / steps
                           for k in ('miner', 'aligner', 'scrambler', 'scout', 'none')},
            'spawn_pos':   self._spawn_pos,
            'quadrant':    self._quadrant,
            'target_role': self._target_role,
        }


# ---------------------------------------------------------------------------
# Cross-agent aggregation
# ---------------------------------------------------------------------------

def aggregate_agent_results_v5(per_agent_results, reduction='max'):
    """Reduce a list of per-agent v5 dicts into one aggregate dict.

    `reduction` selects how the genome's scalar `score` is computed across
    agents:
      'max'  — best-of-N (matches v4; agnostic to weak teammates)
      'mean' — average team competence (selects for cooperation indirectly)
      'min'  — weakest-link (strong cooperation pressure, high variance)

    Component fields are mean-reduced for diagnostic readability regardless of
    `reduction`. `score_mean` and `best_agent_idx` are always reported.
    """
    if not per_agent_results:
        return {}

    score_vals = [r['score'] for r in per_agent_results if 'score' in r]
    out = {}
    if score_vals:
        if reduction == 'max':
            out['score'] = float(np.max(score_vals))
        elif reduction == 'mean':
            out['score'] = float(np.mean(score_vals))
        elif reduction == 'min':
            out['score'] = float(np.min(score_vals))
        else:
            out['score'] = float(np.max(score_vals))
        out['score_mean'] = float(np.mean(score_vals))
        out['best_agent_idx'] = int(np.argmax(score_vals))
    else:
        out['score'] = 0.0
        out['score_mean'] = 0.0
        out['best_agent_idx'] = -1
    out['reduction'] = str(reduction)

    keys_numeric_mean = (
        'score_F', 'score_B', 'F_over_total',
        'ach_first_ore', 'ach_first_gear',
        'ach_first_heart', 'ach_first_heart_spent',
        'ach_first_role_event_miner', 'ach_first_role_event_aligner',
        'ach_first_role_event_scrambler', 'ach_first_role_event_scout',
        'role_event_count_miner', 'role_event_count_aligner',
        'role_event_count_scrambler', 'role_event_count_scout',
        'role_event_total',
        'heart_gain_count', 'heart_spend_count',
        'ep_reward_clipped', 'exploration_capped', 'init_loc',
    )
    for k in keys_numeric_mean:
        vals = [r[k] for r in per_agent_results if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0

    gs_keys = ('miner', 'aligner', 'scrambler', 'scout', 'none')
    out['gear_share'] = {
        k: float(np.mean([r['gear_share'].get(k, 0.0) for r in per_agent_results]))
        for k in gs_keys
    }

    best = per_agent_results[out['best_agent_idx']] if score_vals else per_agent_results[0]
    out['spawn_pos']   = best.get('spawn_pos')
    out['quadrant']    = best.get('quadrant')
    out['target_role'] = best.get('target_role')
    return out


# ---------------------------------------------------------------------------
# Diagnostic stdout (regex fallback for evaluators not using sidecar)
# ---------------------------------------------------------------------------

def format_diagnostic_lines_v5(agg, ep_reward_total):
    """Multi-line v5 diagnostic block, printed by each evaluator.

    Lines begin with 'v5:' so the regex parser in task drivers picks up
    ep_reward without changes (mirrors the v2/v3/v4 contract).
    """
    gs = agg.get('gear_share', {})
    spawn = agg.get('spawn_pos')
    spawn_str = f'({spawn[0]},{spawn[1]})' if spawn else 'NA'
    line1 = (
        f'  v5: gear_share=miner:{gs.get("miner", 0):.2f},'
        f'aligner:{gs.get("aligner", 0):.2f},'
        f'scrambler:{gs.get("scrambler", 0):.2f},'
        f'scout:{gs.get("scout", 0):.2f},'
        f'none:{gs.get("none", 0):.2f}'
    )
    line2 = (
        f'  v5: spawn={spawn_str} quadrant={agg.get("quadrant", "NA")} '
        f'target_role={agg.get("target_role")} '
        f'best_agent={agg.get("best_agent_idx", -1)} '
        f'reduction={agg.get("reduction", "max")}'
    )
    line3 = (
        f'  v5: score_F={agg.get("score_F", 0):.4f} '
        f'score_B={agg.get("score_B", 0):.4f} '
        f'F_over_total={agg.get("F_over_total", 0):.3f} '
        f'init_loc={agg.get("init_loc", 0):.3f} '
        f'explore={agg.get("exploration_capped", 0):.3f}'
    )
    line4 = (
        f'  v5: hearts_gained={agg.get("heart_gain_count", 0):.2f} '
        f'hearts_spent={agg.get("heart_spend_count", 0):.2f} '
        f'role_events_total={agg.get("role_event_total", 0):.2f} '
        f'ep_reward={float(ep_reward_total):.4f} '
        f'ep_reward={agg.get("ep_reward_clipped", 0):.4f} '
        f'score_mean={agg.get("score_mean", 0):.4f} '
        f'-> score={agg.get("score", 0):.4f}'
    )
    line5 = (
        f'  v5: ach_ore={agg.get("ach_first_ore", 0):.2f} '
        f'ach_gear={agg.get("ach_first_gear", 0):.2f} '
        f'ach_heart={agg.get("ach_first_heart", 0):.2f} '
        f'ach_spent={agg.get("ach_first_heart_spent", 0):.2f} '
        f'ach_role_M={agg.get("ach_first_role_event_miner", 0):.2f} '
        f'ach_role_A={agg.get("ach_first_role_event_aligner", 0):.2f} '
        f'ach_role_S={agg.get("ach_first_role_event_scrambler", 0):.2f} '
        f'ach_role_Sc={agg.get("ach_first_role_event_scout", 0):.2f}'
    )
    return '\n'.join([line1, line2, line3, line4, line5])


# ---------------------------------------------------------------------------
# Sidecar JSON write (atomic)
# ---------------------------------------------------------------------------

def write_component_sidecar(path, per_episode_components, aggregate,
                             fitness_stat='composite_v5_role_event'):
    """Write per-episode + aggregate component dicts to a JSON sidecar.

    Atomic via temp + os.replace. No-op if `path` is empty / None.

    `per_episode_components`: list of per-episode dicts. Each dict should look
    like {'episode_idx': k, 'seed': s, 'aggregate': {...},
          'per_agent': [{...}, ...]}.

    `aggregate`: the over-all-episodes aggregate dict (typically
    aggregate_over_episodes_v5(...) output).
    """
    if not path:
        return
    payload = {
        'version':      'v5.0',
        'fitness_stat': fitness_stat,
        'n_episodes':   len(per_episode_components),
        'aggregate':    aggregate,
        'per_episode':  per_episode_components,
    }
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.components_v5_', suffix='.json.tmp', dir=parent)
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
    return str(o)


# ---------------------------------------------------------------------------
# Aggregation across episodes (used by evaluators before sidecar write)
# ---------------------------------------------------------------------------

def aggregate_over_episodes_v5(per_episode_aggregates):
    """Reduce a list of per-episode aggregate dicts (from
    aggregate_agent_results_v5) into one over-all-episodes aggregate.

    `score` is mean-reduced across episodes (within an episode it's already
    the chosen cross-agent reduction). Other fields are mean-reduced.
    """
    if not per_episode_aggregates:
        return {}
    out = {}
    score_vals = [a.get('score', 0.0) for a in per_episode_aggregates]
    out['score'] = float(np.mean(score_vals)) if score_vals else 0.0
    out['score_mean'] = float(np.mean(score_vals)) if score_vals else 0.0
    out['n_episodes'] = len(per_episode_aggregates)
    out['reduction'] = per_episode_aggregates[0].get('reduction', 'max')

    keys_numeric_mean = (
        'score_F', 'score_B', 'F_over_total',
        'ach_first_ore', 'ach_first_gear',
        'ach_first_heart', 'ach_first_heart_spent',
        'ach_first_role_event_miner', 'ach_first_role_event_aligner',
        'ach_first_role_event_scrambler', 'ach_first_role_event_scout',
        'role_event_count_miner', 'role_event_count_aligner',
        'role_event_count_scrambler', 'role_event_count_scout',
        'role_event_total',
        'heart_gain_count', 'heart_spend_count',
        'ep_reward_clipped', 'exploration_capped', 'init_loc',
    )
    for k in keys_numeric_mean:
        vals = [a[k] for a in per_episode_aggregates if k in a]
        out[k] = float(np.mean(vals)) if vals else 0.0

    gs_keys = ('miner', 'aligner', 'scrambler', 'scout', 'none')
    out['gear_share'] = {
        k: float(np.mean([a.get('gear_share', {}).get(k, 0.0)
                          for a in per_episode_aggregates]))
        for k in gs_keys
    }
    return out


# ---------------------------------------------------------------------------
# CLI weight parsing (mirrors composite_v4.parse_weights_str)
# ---------------------------------------------------------------------------

def parse_weights_str(s):
    """Parse 'k1=v1,k2=v2,...' from a CLI flag into a dict.

    Float values are stored as float; the special key 'reduction' is preserved
    as a string ('max' | 'mean' | 'min').
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

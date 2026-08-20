"""composite_v6 miner-gated mine-and-deposit fitness scoring.

Successor to the v6 AlphaStar-style z-trajectory scorer. Simplified and
re-targeted: hearts and gear crafting are dropped from the objective.
The fitness now teaches a single curriculum loop:

  1. Pick the miner role. The mission places 4 role objects right below
     spawn; interacting with the miner one grants a `decoder` item, so
     holding `decoder` is the detectable miner signal. Choosing miner is
     heavily rewarded (score_role) and gates the rest of the score: an
     agent that never picks miner scores ~0.

  2. Mine any material. Once gated in, extracting any of the 4 base
     resources (carbon/oxygen/germanium/silicon) is heavily rewarded.

  3. Return and deposit. The `cargo` inventory bucket holds the 4
     resources up to 100 units total. The fuller the cargo, the more a
     holding penalty accrues, pressuring the agent to deposit. Any drop
     in held cargo is credited as a deposit (cargo-fill pressure only --
     no chest-proximity detection).

Cold-start bootstrap
--------------------
A fully-gated score (everything multiplied by miner_fraction) leaves the
CMA-ES fitness surface flat at 0 until some candidate happens to hold a
decoder -- and a random initial population may never stumble onto it, so
the GA gets no gradient and never learns to pick miner. To break that
flat landscape, two terms are *ungated*:

  * score_explore -- continuous map-coverage reward. Every moving
    candidate scores something, and broad coverage is exactly what makes
    an agent pass over the miner role object placed right below spawn.
  * score_touch   -- a one-time bonus the first step a decoder is ever
    held. A discrete rung that rewards the discovery itself without
    needing sustained holding.

Mining and depositing stay gated behind miner_fraction: a non-miner is
not paid to mine. The result is a curriculum ladder -- explore -> touch
the role object -> hold the role -> mine/deposit -- each rung giving a
gradient toward the next.

Per-agent score:

  miner_fraction  = decoder_steps / steps_seen         # 0..1 gate + role signal
  touched_decoder = 1.0 if decoder_steps > 0 else 0.0  # ungated first-touch flag
  score_role      = w_role    * miner_fraction
  score_touch     = w_touch   * touched_decoder        # UNGATED bootstrap rung
  score_explore   = w_explore * exploration_capped     # UNGATED bootstrap rung
  score_mine      = w_mine    * total_extracted / max_steps
  score_deposit   = w_deposit * total_deposited / max_steps
  score_hold      = -w_hold   * mean_fill_fraction
  bootstrap       = score_explore + score_touch
  gated           = score_mine + score_deposit + score_hold
  score           = bootstrap + score_role + miner_fraction * gated

miner_fraction is a smooth gate so the CMA-ES fitness surface stays
continuous; picking miner early maximises both score_role and the gate.

Public API (class/function names and signatures) mirrors the previous
v6 so the evaluate_*_cogames.py drivers need no changes.
"""

import json
import os
import tempfile
from collections import Counter

import numpy as np

from composite_v4 import (
    DEFAULT_QUADRANT_TO_ROLE,
    compute_v3_behaviour,
    get_agent_spawn_positions,
    get_map_extents,
    parse_inventory_at_center,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 4 base resources sharing the single `cargo` inventory bucket.
RESOURCE_KEYS = ('carbon', 'oxygen', 'germanium', 'silicon')

# Total `cargo` bucket capacity (src/cogames/cogs_vs_clips/mission.py).
# `energy` is a separate bucket and is deliberately excluded from cargo-fill.
DEFAULT_CARGO_CAPACITY = 100


DEFAULT_WEIGHTS_V6 = {
    'w_role':    5.0,   # heavily reward choosing the miner role
    'w_touch':   1.0,   # ungated one-time bonus for first decoder pickup
    'w_explore': 2.0,   # ungated efficient-exploration bootstrap
    'w_mine':    4.0,   # heavily reward mining any material
    'w_deposit': 5.0,   # > per-unit mine reward so dump-and-repeat pays
    'w_hold':    1.0,   # fill-pressure penalty strength

    # Reduction across agents:
    #   'max' | 'mean' | 'min' | 'trimmed_mean' | 'top2_mean'.
    'reduction': 'trimmed_mean',
}


# ---------------------------------------------------------------------------
# Per-episode accumulator
# ---------------------------------------------------------------------------

class PerEpisodeAccumulatorV6:
    """One per agent. Public API mirrors the previous PerEpisodeAccumulatorV6."""

    def __init__(self, max_steps, quadrant_to_role=None,
                 cargo_capacity=DEFAULT_CARGO_CAPACITY):
        self.max_steps = max(int(max_steps), 1)
        self.quadrant_to_role = quadrant_to_role or DEFAULT_QUADRANT_TO_ROLE
        self.cargo_capacity = max(int(cargo_capacity), 1)

        self.steps_seen = 0

        # Miner role: steps spent holding a decoder item.
        self.decoder_steps = 0

        # Mining / depositing: cumulative positive / negative resource deltas.
        self.total_extracted = 0.0
        self.total_deposited = 0.0

        # Deposit pressure: running sum of per-step cargo-fill fraction.
        self.sum_fill_fraction = 0.0

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
        self.steps_seen += 1

        # Miner role signal: holding a decoder item.
        if inv.get('decoder', 0) > 0:
            self.decoder_steps += 1

        # Cargo-fill fraction (deposit pressure).
        cargo_fill = sum(inv.get(k, 0) for k in RESOURCE_KEYS)
        self.sum_fill_fraction += min(cargo_fill / self.cargo_capacity, 1.0)

        # Mining (positive deltas) and depositing (negative deltas).
        if self._prev_inv is not None:
            for k in RESOURCE_KEYS:
                d = inv.get(k, 0) - self._prev_inv.get(k, 0)
                if d > 0:
                    self.total_extracted += float(d)
                elif d < 0:
                    self.total_deposited += float(-d)

        self._prev_inv = inv

    # ---- finalize --------------------------------------------------------

    def episode_score(self, ep_reward_total, exploration_capped,
                      agent_stats=None, weights=None):
        """Combine accumulator state into the v6 score dict.

        `ep_reward_total` is kept for signature compatibility but is no
        longer part of the formula (hearts dropped from the objective).
        """
        w = {**DEFAULT_WEIGHTS_V6, **(weights or {})}
        steps = max(self.steps_seen, 1)
        ms = float(self.max_steps)

        miner_fraction = self.decoder_steps / steps
        touched_decoder = 1.0 if self.decoder_steps > 0 else 0.0
        mean_fill_fraction = self.sum_fill_fraction / steps

        score_role    = float(w['w_role'])    * miner_fraction
        score_touch   = float(w['w_touch'])   * touched_decoder
        score_explore = float(w['w_explore']) * float(exploration_capped)
        score_mine    = float(w['w_mine'])    * (self.total_extracted / ms)
        score_deposit = float(w['w_deposit']) * (self.total_deposited / ms)
        score_hold    = -float(w['w_hold'])   * mean_fill_fraction

        # Ungated bootstrap rungs (explore + first-touch) keep the fitness
        # surface non-flat before any candidate has learned to hold a
        # decoder. Mining/depositing stay gated behind miner_fraction.
        bootstrap = score_explore + score_touch
        gated     = score_mine + score_deposit + score_hold
        score     = bootstrap + score_role + miner_fraction * gated

        return {
            'score':              float(score),
            'score_role':         float(score_role),
            'score_touch':        float(score_touch),
            'score_explore':      float(score_explore),
            'score_mine':         float(score_mine),
            'score_deposit':      float(score_deposit),
            'score_hold':         float(score_hold),
            'miner_fraction':     float(miner_fraction),
            'touched_decoder':    float(touched_decoder),
            'total_extracted':    float(self.total_extracted),
            'total_deposited':    float(self.total_deposited),
            'mean_fill_fraction': float(mean_fill_fraction),
            'exploration_capped': float(exploration_capped),
            'ep_reward':          float(ep_reward_total),
            'spawn_pos':          self._spawn_pos,
            'quadrant':           self._quadrant,
            'target_role':        self._target_role,
        }


# ---------------------------------------------------------------------------
# Component fields that are always mean-reduced for diagnostic readability.
# ---------------------------------------------------------------------------

COMPONENT_KEYS = (
    'score_role', 'score_touch', 'score_explore', 'score_mine',
    'score_deposit', 'score_hold',
    'miner_fraction', 'touched_decoder', 'total_extracted', 'total_deposited',
    'mean_fill_fraction', 'exploration_capped', 'ep_reward',
)


# ---------------------------------------------------------------------------
# Cross-agent aggregation
# ---------------------------------------------------------------------------

def aggregate_agent_results_v6(per_agent_results, reduction='trimmed_mean'):
    """Reduce a list of per-agent v6 dicts into one aggregate dict.

    Reduction options:
      'max'           best-of-N (high run-to-run variance).
      'mean'          average of all agents.
      'min'           weakest-link (strongest cooperation pressure).
      'trimmed_mean'  default: drop the single lowest-scoring agent, mean
                      the rest (removes single-agent luck variance).
      'top2_mean'     average of the top-2 agents.

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
    out['spawn_pos']   = best.get('spawn_pos')
    out['quadrant']    = best.get('quadrant')
    out['target_role'] = best.get('target_role')
    return out


# ---------------------------------------------------------------------------
# Aggregation across episodes
# ---------------------------------------------------------------------------

def aggregate_over_episodes_v6(per_episode_aggregates):
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

def format_diagnostic_lines_v6(agg, ep_reward_total):
    """Multi-line v6 diagnostic block. Lines begin with 'v6:' so the regex
    parser in task drivers picks up ep_reward without changes.
    """
    spawn = agg.get('spawn_pos')
    spawn_str = f'({spawn[0]},{spawn[1]})' if spawn else 'NA'
    line1 = (
        f'  v6: spawn={spawn_str} quadrant={agg.get("quadrant", "NA")} '
        f'target_role={agg.get("target_role")} '
        f'best_agent={agg.get("best_agent_idx", -1)} '
        f'reduction={agg.get("reduction", "trimmed_mean")}'
    )
    line2 = (
        f'  v6: miner_fraction={agg.get("miner_fraction", 0):.3f} '
        f'touched={agg.get("touched_decoder", 0):.3f} '
        f'extracted={agg.get("total_extracted", 0):.1f} '
        f'deposited={agg.get("total_deposited", 0):.1f} '
        f'fill={agg.get("mean_fill_fraction", 0):.3f} '
        f'explore={agg.get("exploration_capped", 0):.3f}'
    )
    line3 = (
        f'  v6: score_role={agg.get("score_role", 0):.4f} '
        f'score_touch={agg.get("score_touch", 0):.4f} '
        f'score_explore={agg.get("score_explore", 0):.4f} '
        f'score_mine={agg.get("score_mine", 0):.4f} '
        f'score_deposit={agg.get("score_deposit", 0):.4f} '
        f'score_hold={agg.get("score_hold", 0):.4f}'
    )
    line4 = (
        f'  v6: ep_reward={float(ep_reward_total):.4f} '
        f'score_mean={agg.get("score_mean", 0):.4f} '
        f'-> score={agg.get("score", 0):.4f}'
    )
    return '\n'.join([line1, line2, line3, line4])


# ---------------------------------------------------------------------------
# Sidecar JSON write (atomic)
# ---------------------------------------------------------------------------

def write_component_sidecar(path, per_episode_components, aggregate,
                             fitness_stat='composite_v6_zstat'):
    """Write per-episode + aggregate component dicts to a JSON sidecar.

    Atomic via temp + os.replace. No-op if `path` is empty / None.
    """
    if not path:
        return
    payload = {
        'version':      'v6.1',
        'fitness_stat': fitness_stat,
        'n_episodes':   len(per_episode_components),
        'aggregate':    aggregate,
        'per_episode':  per_episode_components,
    }
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.components_v6_', suffix='.json.tmp', dir=parent)
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

"""composite_v5_diff_reward — COMA-style counterfactual difference reward.

Prototype for measuring per-agent causal contribution: D_i = score_F minus
score_F when agent i's actions are replaced by noop. Positive D_i means
agent i's actions causally contributed to delivery.

STATUS: BLOCKED on env determinism (2026-05-08).

The plan flagged a pre-flight gate: counterfactual replay requires the env
to be deterministic given (seed, action sequences) — i.e. running the SAME
recorded actions through Rollout(env_cfg, ..., seed=S) twice should yield
identical sim.episode_stats.

Result of the pre-flight test in this module's `verify_determinism_arena`
(at module load, run via `python composite_v5_diff_reward.py --preflight`):
  agent[1].action.move.failed:  88 vs 54
  agent[2].cell.max_distance_from_spawn: 18 vs 29
  agent[2].oxygen.amount: 0 vs 40
  ... per-agent stats diverge despite identical actions and seed

This means cogames has internal RNG not tied to the rollout `seed` argument
(likely procedural clip-team spawn timing or assembler protocol RNG). Strict
"replay agent i's actions while noop'ing one agent" is therefore NOT a valid
counterfactual: differences between base and counterfactual would conflate
agent i's contribution with re-rolled env randomness.

Until the env exposes deterministic seeding for ALL random draws (or we
serialize/deserialize the full sim state), do NOT use this prototype to
score genomes. Use composite_v5_role_event for selection.

What CAN still be useful (research mode only, not selection):
  - Run the diff-reward analysis with multiple counterfactual rollouts per
    agent and average D_i over those rolls. The variance from env RNG
    becomes a noise floor; if D_i / std(D_i) > some threshold, the agent's
    contribution is statistically detectable.
  - Refactor the env to seed all internal RNG from the rollout `seed`. Then
    re-run `verify_determinism_arena` and remove the gate.

This module exposes:
  - verify_determinism_arena(num_agents=4, max_steps=200) -> (bool, dict)
        Re-runs the pre-flight. Returns (is_deterministic, mismatches_dict).
  - DiffRewardEvaluator (DRAFT) — the scaffold for counterfactual rollouts.
        Refuses to run unless verify_determinism_arena passes.
"""

import numpy as np

from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import Action


# ---------------------------------------------------------------------------
# Pre-flight: env-determinism gate
# ---------------------------------------------------------------------------

class _RecordingPolicy(AgentPolicy):
    """Wraps a base policy; records every action it returns."""

    def __init__(self, env_info, base):
        super().__init__(env_info)
        self.base = base
        self.recorded = []

    def step(self, obs):
        a = self.base.step(obs)
        self.recorded.append(a)
        return a


class _ScriptedPolicy(AgentPolicy):
    """Replays a recorded per-step action list. Returns noop if exhausted."""

    def __init__(self, env_info, action_seq):
        super().__init__(env_info)
        self._actions = list(action_seq)
        self._t = 0

    def step(self, obs):
        if self._t >= len(self._actions):
            a = Action(name='noop')
        else:
            a = self._actions[self._t]
        self._t += 1
        return a


class _NoopPolicy(AgentPolicy):
    def step(self, obs):
        return Action(name='noop')


class _SeededFixedPolicy(AgentPolicy):
    """Stateless action sampler driven by a numpy Generator.

    Used only by the determinism preflight to generate a deterministic but
    non-trivial action sequence to replay.
    """

    def __init__(self, env_info, seed):
        super().__init__(env_info)
        self._rng = np.random.default_rng(seed)
        self.action_names = list(env_info.action_names)

    def step(self, obs):
        idx = int(self._rng.integers(0, len(self.action_names)))
        return Action(name=self.action_names[idx])


def verify_determinism_arena(num_agents=4, mission='arena', seed=99,
                             gen_seed=12345):
    """Returns (is_deterministic, mismatches_dict).

    Two passes through the env:
      1. RecordingPolicy wraps SeededFixedPolicy(gen_seed+i) — produces a
         deterministic action sequence per agent.
      2. ScriptedPolicy replays the recorded actions.
    Then compares per-agent episode_stats. If any value mismatches, the env
    has internal RNG not tied to the rollout `seed` — counterfactual replay
    is invalid in this env.
    """
    from cogames.cli.mission import get_mission

    _, env_cfg, _ = get_mission(mission)
    env_cfg.game.num_agents = num_agents
    try:
        env_cfg.game.map_builder.instance.spawn_count = num_agents
    except AttributeError:
        pass
    info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    bases = [_SeededFixedPolicy(info, seed=gen_seed + i) for i in range(num_agents)]
    recorders = [_RecordingPolicy(info, b) for b in bases]
    r1 = Rollout(env_cfg, recorders, render_mode='none', seed=seed)
    r1.run_until_done()
    stats1 = r1._sim.episode_stats

    replayers = [_ScriptedPolicy(info, recorders[i].recorded) for i in range(num_agents)]
    r2 = Rollout(env_cfg, replayers, render_mode='none', seed=seed)
    r2.run_until_done()
    stats2 = r2._sim.episode_stats

    mismatches = {}
    for i in range(num_agents):
        a1 = dict(stats1['agent'][i])
        a2 = dict(stats2['agent'][i])
        diffs = {}
        all_keys = set(a1.keys()) | set(a2.keys())
        for k in all_keys:
            v1 = a1.get(k, '<missing>')
            v2 = a2.get(k, '<missing>')
            if v1 != v2:
                diffs[k] = (v1, v2)
        if diffs:
            mismatches[f'agent[{i}]'] = diffs
    return (len(mismatches) == 0), mismatches


# ---------------------------------------------------------------------------
# Diff-reward evaluator (DRAFT — gated)
# ---------------------------------------------------------------------------

class DiffRewardEvaluator:
    """Counterfactual rollout helper.

    Given a base rollout that produced (action_seqs, score_F), runs N
    additional rollouts — one per agent — with that agent's actions replaced
    by noop and the others' actions replayed. Returns per-agent D_i.

    REFUSES to run unless `verify_determinism_arena` passes — see module
    docstring. To force-run for research (averaging over re-rolls), pass
    `force=True` and `n_repeats > 1`.
    """

    def __init__(self, env_cfg, base_action_seqs, *, env_info=None,
                 num_agents=None, force=False):
        if num_agents is None:
            num_agents = len(base_action_seqs)
        self.env_cfg = env_cfg
        self.num_agents = int(num_agents)
        self.base_action_seqs = base_action_seqs
        self.env_info = env_info or PolicyEnvInterface.from_mg_cfg(env_cfg)

        if not force:
            ok, mism = verify_determinism_arena(num_agents=self.num_agents)
            if not ok:
                raise RuntimeError(
                    'Env is not deterministic under action replay — '
                    'counterfactual diff-reward is invalid in this env. '
                    'See composite_v5_diff_reward module docstring. '
                    'Pass force=True to run anyway (research mode only, '
                    'not for selection).\n\n'
                    f'First mismatch: {next(iter(mism.items())) if mism else None}'
                )

    def counterfactual_score_F(self, score_F_callable, drop_agent_idx, seed,
                               *, render_mode='none'):
        """Run one counterfactual rollout with agent[drop_agent_idx]'s actions
        replaced by noop, others replayed from `self.base_action_seqs`.

        `score_F_callable(rollout)` extracts the F component (caller-defined,
        e.g. from composite_v5).
        """
        policies = []
        for i in range(self.num_agents):
            if i == drop_agent_idx:
                policies.append(_NoopPolicy(self.env_info))
            else:
                policies.append(_ScriptedPolicy(self.env_info,
                                                self.base_action_seqs[i]))
        rollout = Rollout(self.env_cfg, policies, render_mode=render_mode, seed=seed)
        rollout.run_until_done()
        return float(score_F_callable(rollout))


# ---------------------------------------------------------------------------
# CLI: `python composite_v5_diff_reward.py --preflight`
# ---------------------------------------------------------------------------

def _cli_preflight():
    print('[diff-reward preflight] verifying env determinism under action replay...',
          flush=True)
    ok, mism = verify_determinism_arena()
    if ok:
        print('[diff-reward preflight] PASS — env is deterministic under action '
              'replay; counterfactual diff-reward is valid.')
        return 0
    print('[diff-reward preflight] FAIL — env is NOT deterministic under action '
          'replay. counterfactual diff-reward is INVALID for selection.')
    print(f'[diff-reward preflight] {len(mism)} agents have mismatched stats. '
          'First few mismatches:')
    n = 0
    for ag, diffs in mism.items():
        for k, (v1, v2) in diffs.items():
            print(f'  {ag}.{k}: {v1} != {v2}')
            n += 1
            if n >= 8:
                print('  ...')
                return 1
    return 1


if __name__ == '__main__':
    import sys
    if '--preflight' in sys.argv:
        sys.exit(_cli_preflight())
    print('Usage: python composite_v5_diff_reward.py --preflight')
    sys.exit(2)

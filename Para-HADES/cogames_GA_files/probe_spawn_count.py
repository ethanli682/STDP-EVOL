"""One-shot probe: does --num-agents 1 actually produce a single agent?

We suspect the evaluator's override pattern

    try:
        env_cfg.game.map_builder.instance.spawn_count = num_agents
    except AttributeError:
        pass

silently fails under newer cogames versions because the attribute path
drifted. This probe reports:
  (a) env_cfg.game.num_agents before and after the override
  (b) every '*spawn*' attribute reachable under env_cfg.game.map_builder
  (c) len(sim.episode_stats['agent']) after one very short rollout — the
      ground-truth count of agents the env actually instantiated
  (d) cell.visited per agent, to see whether 7000/8000 matches N*max_steps

If (c) is != 1, the override isn't being honoured, Finding 3 is confirmed,
and the correct attribute path is whichever one shows up in (b).
"""
import sys
import numpy as np
import torch

from cogames.cli.mission import get_mission
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

# Minimal no-op policy so we don't need weights.
from mettagrid.policy.policy import AgentPolicy
from mettagrid.simulator import Action


class NoopPolicy(AgentPolicy):
    def __init__(self, info):
        super().__init__(info)
        self._name = info.action_names[0]  # whatever idx 0 is

    def step(self, obs):
        return Action(name=self._name)


def dump_attrs(obj, path='env_cfg.game.map_builder', depth=0, max_depth=4, seen=None):
    """Recursively list attribute paths whose name contains 'spawn' or 'agent'."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen or depth > max_depth:
        return
    seen.add(oid)
    for name in dir(obj):
        if name.startswith('_'):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        full = f'{path}.{name}'
        lname = name.lower()
        if 'spawn' in lname or 'agent' in lname or 'num_' in lname:
            print(f'  {full} = {val!r}')
        # Recurse into structured config nodes but not into primitives/containers.
        if hasattr(val, '__dict__') or (hasattr(val, '__class__') and val.__class__.__module__ not in ('builtins',)):
            if not isinstance(val, (str, bytes, int, float, bool, list, tuple, dict, set, type(None), np.ndarray, torch.Tensor)):
                dump_attrs(val, full, depth + 1, max_depth, seen)


def main():
    mission = 'tutorial.scout'
    target_num_agents = 1

    _, env_cfg, _ = get_mission(mission)
    print(f'=== mission={mission}, requesting num_agents={target_num_agents} ===')
    print()

    print(f'[before override] env_cfg.game.num_agents = {env_cfg.game.num_agents}')
    print(f'[before override] env_cfg.game.max_steps  = {env_cfg.game.max_steps}')

    print()
    print('[attribute scan: env_cfg.game.map_builder — names containing spawn/agent/num_]')
    try:
        dump_attrs(env_cfg.game.map_builder)
    except Exception as e:
        print(f'  scan failed: {e!r}')

    # Apply the exact pattern the evaluators use.
    env_cfg.game.num_agents = target_num_agents
    override_ok = False
    override_err = None
    try:
        env_cfg.game.map_builder.instance.spawn_count = target_num_agents
        override_ok = True
    except AttributeError as e:
        override_err = e

    print()
    print(f'[override attempt] env_cfg.game.map_builder.instance.spawn_count = {target_num_agents}')
    print(f'[override attempt] succeeded? {override_ok}  (err={override_err})')

    # Peek: post-override
    print()
    print('[post-override scan]')
    try:
        dump_attrs(env_cfg.game.map_builder)
    except Exception as e:
        print(f'  scan failed: {e!r}')

    # Ground truth: run a *very* short rollout and count agents.
    print()
    print('=== ground-truth agent count from a 1-step rollout ===')
    # Shorten the episode to avoid a full 1000-step rollout.
    try:
        env_cfg.game.max_steps = 5
    except Exception:
        pass

    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    N = env_cfg.game.num_agents
    policies = [NoopPolicy(policy_env_info) for _ in range(N)]
    rollout = Rollout(env_cfg, policies, render_mode='none', seed=12345)
    rollout.run_until_done()
    sim = rollout._sim

    agent_stats_list = sim.episode_stats['agent']
    print(f'env_cfg.game.num_agents       = {env_cfg.game.num_agents}')
    print(f'len(sim.episode_stats[agent]) = {len(agent_stats_list)}')
    print(f'max_steps used                = {env_cfg.game.max_steps}')
    for i, s in enumerate(agent_stats_list):
        print(f'  agent {i}: cell.visited={s.get("cell.visited", None)} '
              f'cell.unique_visited={s.get("cell.unique_visited", None)}')


if __name__ == '__main__':
    sys.exit(main() or 0)

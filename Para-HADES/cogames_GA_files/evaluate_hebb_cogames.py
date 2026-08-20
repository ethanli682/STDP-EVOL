import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

from hebbian_weights_update import (
    hebbian_update_A,
    hebbian_update_AD,
    hebbian_update_AD_lr,
    hebbian_update_ABC,
    hebbian_update_ABC_lr,
    hebbian_update_ABCD,
    hebbian_update_ABCD_lr_D_in,
    hebbian_update_ABCD_lr_D_out,
    hebbian_update_ABCD_lr_D_in_and_out,
)

from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation
from mettagrid.simulator.rollout import Rollout

import composite_v7
import composite_v8
try:
    import composite_v4  # type: ignore
except ImportError:
    composite_v4 = None  # removed in v7-only cutover; v4 fitness_stat path is dead
try:
    import composite_v5  # type: ignore
except ImportError:
    composite_v5 = None  # removed in v7-only cutover; v5 fitness_stat path is dead
import obs_encoder as _obs_encoder


# ---------------------------------------------------------------------------
# MLP architecture (mirrors policies.py:MLP_heb but with cogames dims)
# input=900 (300 tokens × 3 dims), hidden=[64,32], output=num_actions
# ---------------------------------------------------------------------------

class MLP_heb_cogames(nn.Module):
    """MLP with no bias, same structure as MLP_heb in policies.py."""

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, action_dim, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob[0]).float().detach()
        x1 = torch.tanh(self.fc1(state))
        x2 = torch.tanh(self.fc2(x1))
        o = self.fc3(x2)
        return state, x1, x2, o


def _weights_init(m, init_weights='uni'):
    """Initialise a single Linear module in place."""
    if isinstance(m, nn.Linear):
        if init_weights == 'xa_uni':
            nn.init.xavier_uniform_(m.weight.data, 0.3)
        elif init_weights == 'sparse':
            nn.init.sparse_(m.weight.data, 0.8)
        elif init_weights == 'uni':
            nn.init.uniform_(m.weight.data, -0.1, 0.1)
        elif init_weights == 'normal':
            nn.init.normal_(m.weight.data, 0, 0.024)
        elif init_weights == 'ka_uni':
            nn.init.kaiming_uniform_(m.weight.data, 3)
        elif init_weights == 'uni_big':
            nn.init.uniform_(m.weight.data, -1, 1)
        elif init_weights == 'xa_uni_big':
            nn.init.xavier_uniform_(m.weight.data)
        elif init_weights in ('default', None):
            pass


# ---------------------------------------------------------------------------
# CoGames agent policy: wraps MLP + Hebbian update, used by Rollout
# ---------------------------------------------------------------------------

class CoGamesAgentPolicy(AgentPolicy):
    """Per-agent policy that runs a Hebbian MLP and updates weights each step."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        network: MLP_heb_cogames,
        hebb_coeffs: np.ndarray,
        hebb_rule: str = 'ABCD_lr',
        encoder=None,
    ):
        super().__init__(policy_env_info)
        self._p = network.float()
        self._hebb_coeffs = hebb_coeffs
        self._hebb_rule = hebb_rule
        self._encoder = encoder
        self._action_names = policy_env_info.action_names
        self._obs_shape = policy_env_info.observation_space.shape  # (num_tokens, token_dim)

        # Extract numpy views sharing memory with torch tensors (CPU only)
        params = list(self._p.parameters())
        self._w1 = params[0].detach().numpy()   # shape (64, input_dim)
        self._w2 = params[1].detach().numpy()   # shape (32, 64)
        self._w3 = params[2].detach().numpy()   # shape (num_actions, 32)

    def _obs_to_flat(self, obs: AgentObservation) -> np.ndarray:
        """Convert token observation to flat float32 array normalised to [0,1]."""
        num_tokens, token_dim = self._obs_shape
        obs_array = np.zeros((num_tokens, token_dim), dtype=np.uint8)
        for idx, token in enumerate(obs.tokens):
            if idx >= num_tokens:
                break
            raw = token.raw_token
            obs_array[idx, : len(raw)] = raw
        return obs_array.flatten().astype(np.float32) / 255.0

    def step(self, obs: AgentObservation) -> Action:
        obs_flat = self._obs_to_flat(obs)
        if self._encoder is not None:
            obs_flat = self._encoder(obs_flat)

        # Forward pass
        with torch.no_grad():
            o0, o1, o2, o3 = self._p([obs_flat])
            o0_np = o0.numpy()
            o1_np = o1.numpy()
            o2_np = o2.numpy()
            o3_np = o3.numpy()

        # Action selection (argmax over raw logits)
        action_idx = int(np.argmax(o3_np))

        # Hebbian weight update (in-place via shared numpy/torch memory)
        if self._hebb_rule == 'A':
            self._w1, self._w2, self._w3 = hebbian_update_A(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'AD':
            self._w1, self._w2, self._w3 = hebbian_update_AD(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'AD_lr':
            self._w1, self._w2, self._w3 = hebbian_update_AD_lr(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABC':
            self._w1, self._w2, self._w3 = hebbian_update_ABC(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABC_lr':
            self._w1, self._w2, self._w3 = hebbian_update_ABC_lr(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABCD':
            self._w1, self._w2, self._w3 = hebbian_update_ABCD(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABCD_lr':
            self._w1, self._w2, self._w3 = hebbian_update_ABCD_lr_D_in(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABCD_lr_D_out':
            self._w1, self._w2, self._w3 = hebbian_update_ABCD_lr_D_out(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        elif self._hebb_rule == 'ABCD_lr_D_in_and_out':
            self._w1, self._w2, self._w3 = hebbian_update_ABCD_lr_D_in_and_out(
                self._hebb_coeffs, self._w1, self._w2, self._w3, o0_np, o1_np, o2_np, o3_np)
        else:
            raise ValueError(f'Unknown Hebbian rule: {self._hebb_rule}')

        # Soft weight clamp to prevent runaway Hebbian growth without collapsing
        # the action distribution. Per-step max-abs rescaling (the ATARI version)
        # drove weights to saturate at ±1 within a few steps, producing a frozen
        # policy and ~1-5% map coverage regardless of optimizer. Clamp keeps
        # magnitudes bounded while preserving relative structure.
        for param in self._p.parameters():
            param.data.clamp_(-3.0, 3.0)

        return Action(name=self._action_names[action_idx])


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_hebb_cogames(
    hebb_rule: str,
    mission: str,
    init_weights: str = 'uni',
    render: bool = False,
    eval_episodes: int = 1,
    seed_base: int = 12345,
    hebb_coeffs: np.ndarray = None,
    fitness_stat: str = 'episode_reward',
    num_agents: int = None,
    obs_encoder_kind: str = 'none',
    encoder_dim: int = 64,
    encoder_seed: int = 1337,
    fitness_v4_weights: dict = None,
    fitness_v5_weights: dict = None,
    fitness_v7_weights: dict = None,
    fitness_v8_weights: dict = None,
    components_sidecar: str = '',
    max_steps: int = None,
) -> None:
    """
    Run Hebbian MLP policy on a CoGames mission for eval_episodes episodes.

    Prints per-episode and mean cumulative reward in the same format as
    evaluate_hebb.py so that task_cogames_hebb.py can parse the output.
    """
    assert hebb_coeffs is not None, 'hebb_coeffs must be provided'
    v4_weights = fitness_v4_weights or {}
    v5_weights = fitness_v5_weights or {}
    v7_weights = fitness_v7_weights or {}
    v8_weights = fitness_v8_weights or {}
    per_episode_components_v5 = []
    per_episode_components_v7 = []
    per_episode_components_v8 = []

    with torch.no_grad():
        eval_episodes = max(1, int(eval_episodes))
        episode_rewards = []

        # Build env config once (same across episodes; seed controls randomness)
        # Bridge the split game registry only if needed. Historically CvCGame
        # lived in a separate `cogsguard` package that registered into its own
        # registry (cogsguard.core._GAMES), so we had to import and re-register
        # it into the one cogames.get_game()/get_mission() reads
        # (cogames.game._GAMES). As of mettagrid 0.2.0.58 CvCGame ships inside
        # cogames itself (registered as "cogs_vs_clips"), so the bridge is a
        # no-op there; guard the legacy import so its absence isn't fatal.
        import cogames.game as _cg
        if 'cogs_vs_clips' not in _cg._GAMES and 'cogsguard' not in _cg._GAMES:
            try:
                from cogsguard.game.game import CvCGame as _CvCGame
                _cg.register_game(_CvCGame())
            except ModuleNotFoundError:
                pass
        from cogames.cli.mission import get_mission as _get_mission
        _, env_cfg, _ = _get_mission(mission)

        # Override num_agents and keep the map builder's spawn_count in sync,
        # otherwise the procedural MachinaArena still generates 4 spawns and
        # the rollout aggregates stats across agents we don't want to score.
        if num_agents is not None:
            env_cfg.game.num_agents = num_agents
            try:
                env_cfg.game.map_builder.instance.spawn_count = num_agents
            except AttributeError:
                pass

        # Optional episode-length override (v8 needs a long horizon: the full
        # mine -> deposit -> craft heart -> pick capturer role -> align junction
        # loop does not fit in the mission default of 1000 steps).
        if max_steps is not None:
            env_cfg.game.max_steps = int(max_steps)

        policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
        obs_shape = policy_env_info.observation_space.shape   # (num_tokens, token_dim)
        raw_input_dim = int(np.prod(obs_shape))               # 300*3=900
        encoder, input_dim = _obs_encoder.maybe_make_encoder(
            obs_encoder_kind, raw_input_dim, encoder_dim, encoder_seed
        )
        num_actions = len(policy_env_info.action_names)
        num_agents = env_cfg.game.num_agents

        # Verify hebb_coeffs shape matches this architecture
        expected_synapses = 64 * input_dim + 32 * 64 + num_actions * 32
        if hebb_coeffs.shape[0] != expected_synapses:
            raise ValueError(
                f'hebb_coeffs row count {hebb_coeffs.shape[0]} does not match '
                f'expected {expected_synapses} synapses for dims '
                f'{input_dim}\u219264\u219232\u2192{num_actions}'
            )

        print(
            f'CoGames eval: mission={mission}  obs_shape={obs_shape}  '
            f'raw_input_dim={raw_input_dim}  obs_encoder={obs_encoder_kind}  '
            f'encoder_dim={encoder_dim}  encoder_seed={encoder_seed}  '
            f'input_dim={input_dim}  num_actions={num_actions}  '
            f'num_agents={num_agents}  episodes={eval_episodes}',
            flush=True,
        )

        for ep_idx in range(eval_episodes):
            ep_seed = int(seed_base) + ep_idx
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)

            # Create one fresh network per agent per episode
            agent_policies = []
            for _agent_id in range(num_agents):
                p = MLP_heb_cogames(input_dim, num_actions)
                p.apply(lambda m: _weights_init(m, init_weights))
                policy_obj = CoGamesAgentPolicy(
                    policy_env_info, p, hebb_coeffs, hebb_rule,
                    encoder=encoder,
                )
                agent_policies.append(policy_obj)

            # v4/v5 setup: per-agent accumulator + per-step hook around policy.step
            accumulators = None
            if fitness_stat == 'composite_v4_role_gated':
                v4_max_steps = max(env_cfg.game.max_steps or 1000, 100)
                accumulators = [
                    composite_v4.PerEpisodeAccumulator(v4_max_steps)
                    for _ in range(num_agents)
                ]
                for _i, _p in enumerate(agent_policies):
                    _orig_step = _p.step
                    _acc = accumulators[_i]

                    def _wrapped_step(obs, _orig=_orig_step, _acc=_acc):
                        _acc.update(obs)
                        return _orig(obs)

                    _p.step = _wrapped_step
            elif fitness_stat == 'composite_v5_role_event':
                v5_max_steps = max(env_cfg.game.max_steps or 1000, 100)
                accumulators = [
                    composite_v5.PerEpisodeAccumulatorV5(v5_max_steps)
                    for _ in range(num_agents)
                ]
                for _i, _p in enumerate(agent_policies):
                    _orig_step = _p.step
                    _acc = accumulators[_i]

                    def _wrapped_step(obs, _orig=_orig_step, _acc=_acc):
                        _acc.update(obs)
                        return _orig(obs)

                    _p.step = _wrapped_step
            elif fitness_stat == 'composite_v7_zstat':
                v7_max_steps = max(env_cfg.game.max_steps or 1000, 100)
                accumulators = [
                    composite_v7.PerEpisodeAccumulatorV7(
                        v7_max_steps,
                        obs_center=composite_v7.obs_center_from_env(env_cfg),
                        high_fill_frac=v7_weights.get(
                            'high_fill_frac', composite_v7.HIGH_FILL_FRAC),
                        low_fill_frac=v7_weights.get(
                            'low_fill_frac', composite_v7.LOW_FILL_FRAC))
                    for _ in range(num_agents)
                ]
                for _i, _p in enumerate(agent_policies):
                    _orig_step = _p.step
                    _acc = accumulators[_i]

                    def _wrapped_step(obs, _orig=_orig_step, _acc=_acc):
                        _acc.update(obs)
                        return _orig(obs)

                    _p.step = _wrapped_step
            elif fitness_stat == 'composite_v8_zstat':
                v8_max_steps = max(env_cfg.game.max_steps or 1000, 100)
                accumulators = [
                    composite_v8.PerEpisodeAccumulatorV8(
                        v8_max_steps,
                        obs_center=composite_v8.obs_center_from_env(env_cfg),
                        high_fill_frac=v8_weights.get(
                            'high_fill_frac', composite_v8.HIGH_FILL_FRAC),
                        low_fill_frac=v8_weights.get(
                            'low_fill_frac', composite_v8.LOW_FILL_FRAC))
                    for _ in range(num_agents)
                ]
                for _i, _p in enumerate(agent_policies):
                    _orig_step = _p.step
                    _acc = accumulators[_i]

                    def _wrapped_step(obs, _orig=_orig_step, _acc=_acc):
                        _acc.update(obs)
                        return _orig(obs)

                    _p.step = _wrapped_step

            render_mode = 'gui' if render else 'none'
            rollout = Rollout(
                env_cfg,
                agent_policies,
                render_mode=render_mode,
                seed=ep_seed,
            )

            if accumulators is not None:
                _spawns = composite_v7.get_agent_spawn_positions(rollout._sim, num_agents)
                _extents = composite_v7.get_map_extents(rollout._sim)
                for _i, _acc in enumerate(accumulators):
                    _acc.set_spawn(_spawns[_i], _extents)
                _miner_tag = composite_v7.get_miner_station_tag(rollout._sim)
                for _acc in accumulators:
                    _acc.set_miner_tag(_miner_tag)

            rollout.run_until_done()

            if fitness_stat == 'episode_reward':
                ep_score = float(sum(rollout._sim.episode_rewards))
            elif fitness_stat == 'map_coverage':
                # Fraction of walkable map visited, averaged across agents.
                # Walkable = total grid cells minus wall cells (computed per
                # episode because the procedural map differs by seed).
                sim = rollout._sim
                grid_objs = sim.grid_objects()
                wall_cells = {
                    o['location'] for o in grid_objs.values()
                    if o['type_name'] == 'wall'
                }
                walkable = max(sim.map_width * sim.map_height - len(wall_cells), 1)
                agent_stats_list = sim.episode_stats['agent']
                coverages = [
                    min(float(s.get('cell.unique_visited', 0.0)) / walkable, 1.0)
                    for s in agent_stats_list
                ]
                ep_score = float(np.mean(coverages))
                print(
                    f'  walkable={walkable}  '
                    f'unique_per_agent={[int(s.get("cell.unique_visited",0)) for s in agent_stats_list]}  '
                    f'coverage={ep_score:.4f}',
                    flush=True,
                )
            elif fitness_stat == 'composite_exploration':
                # Composite fitness that rewards exploration and penalises
                # repetitive movement and wall hits:
                #   + unique cells visited     (normalised by max_steps)
                #   + exploration efficiency   (unique / total visited — low for any repetitive pattern)
                #   + max distance from spawn  (normalised by sqrt(max_steps))
                #   - failed moves ratio²      (convex penalty: lenient on a few wall hits,
                #                               increasingly harsh as wall-hitting dominates)
                agent_stats_list = rollout._sim.episode_stats['agent']
                # Guard: env max_steps=None or 0 would make unique/max_steps blow up.
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                dist_norm = max_steps ** 0.5  # proxy for map scale
                agent_scores = []
                for s in agent_stats_list:
                    unique    = float(s.get('cell.unique_visited', 0.0))
                    visited   = float(s.get('cell.visited', 0.0))
                    dist      = float(s.get('cell.max_distance_from_spawn', 0.0))
                    n_moves   = float(s.get('action.move.success', 0.0)) + float(s.get('action.move.failed', 0.0))
                    fail_ratio = float(s.get('action.move.failed', 0.0)) / max(n_moves, 1.0)
                    efficiency = unique / max(visited, 1.0)  # 1.0 = every step is new; low = repetitive
                    agent_scores.append(
                        unique / max_steps          # absolute exploration coverage
                        + 0.4 * efficiency          # penalise repetitive movement patterns
                        + 0.3 * dist / dist_norm    # reward reaching far from spawn
                        - 0.5 * fail_ratio ** 2     # convex wall-hit penalty
                    )
                ep_score = float(np.mean(agent_scores))
            elif fitness_stat == 'composite_exploration_v2':
                # Revised fitness. v1 rewarded "stand still" (score ≈ 0) over any
                # active-but-imperfect policy (score often negative due to convex wall
                # penalty), causing the GA to converge on do-nothing agents. v2:
                #   - gates efficiency by activity so inactivity scores 0, not 0.4
                #   - linear wall penalty (not convex) so moving is not a cliff
                #   - adds episode_reward so the actual scout-mission signal counts
                #   - adds an activity floor so moving at all is rewarded
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                dist_norm = max_steps ** 0.5
                ep_reward_total = float(sum(sim.episode_rewards))
                agent_scores = []
                comp_log = []
                for s in agent_stats_list:
                    unique     = float(s.get('cell.unique_visited', 0.0))
                    visited    = float(s.get('cell.visited', 0.0))
                    dist       = float(s.get('cell.max_distance_from_spawn', 0.0))
                    n_success  = float(s.get('action.move.success', 0.0))
                    n_failed   = float(s.get('action.move.failed', 0.0))
                    n_moves    = n_success + n_failed
                    fail_ratio = n_failed / max(n_moves, 1.0)
                    activity   = min(n_moves / max_steps, 1.0)
                    efficiency = unique / max(visited, 1.0)
                    agent_scores.append(
                        1.0 * (unique / max_steps)
                        + 0.5 * (dist / dist_norm)
                        + 0.3 * activity
                        + 0.2 * efficiency * activity
                        - 0.2 * fail_ratio
                    )
                    comp_log.append((unique, visited, dist, n_moves, fail_ratio, activity, efficiency))
                behaviour = float(np.mean(agent_scores))
                ep_score = behaviour + 0.5 * ep_reward_total
                u, v, d, nm, fr, act, eff = (float(np.mean(x)) for x in zip(*comp_log))
                print(
                    f'  v2: unique={u:.1f} visited={v:.1f} dist={d:.1f} n_moves={nm:.1f} '
                    f'fail_ratio={fr:.3f} activity={act:.3f} efficiency={eff:.3f} '
                    f'ep_reward={ep_reward_total:.4f} behaviour={behaviour:.4f} -> score={ep_score:.4f}',
                    flush=True,
                )
            elif fitness_stat == 'composite_v4_role_gated':
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                episode_rewards = list(sim.episode_rewards)
                ep_reward_team = float(sum(episode_rewards))
                behaviour_capped, _bh_raw, _ = composite_v4.compute_v3_behaviour(
                    agent_stats_list, max_steps
                )
                per_agent_results = []
                for _i, _acc in enumerate(accumulators or []):
                    _stats_i = agent_stats_list[_i] if _i < len(agent_stats_list) else {}
                    _ep_reward_agent = float(episode_rewards[_i]) if _i < len(episode_rewards) else 0.0
                    per_agent_results.append(
                        _acc.episode_score(
                            ep_reward_total=_ep_reward_agent,
                            exploration_capped=behaviour_capped,
                            agent_stats=_stats_i,
                            weights=v4_weights,
                        )
                    )
                _agg = composite_v4.aggregate_agent_results(per_agent_results)
                print(composite_v4.format_diagnostic_lines(_agg, ep_reward_team),
                      flush=True)
                ep_score = _agg['score']
            elif fitness_stat == 'composite_v5_role_event':
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                episode_rewards = list(sim.episode_rewards)
                ep_reward_team = float(sum(episode_rewards))
                behaviour_capped, _bh_raw, _ = composite_v5.compute_v3_behaviour(
                    agent_stats_list, max_steps
                )
                per_agent_results = []
                for _i, _acc in enumerate(accumulators or []):
                    _stats_i = agent_stats_list[_i] if _i < len(agent_stats_list) else {}
                    _ep_reward_agent = float(episode_rewards[_i]) if _i < len(episode_rewards) else 0.0
                    per_agent_results.append(
                        _acc.episode_score(
                            ep_reward_total=_ep_reward_agent,
                            exploration_capped=behaviour_capped,
                            agent_stats=_stats_i,
                            weights=v5_weights,
                        )
                    )
                _reduction = (v5_weights or {}).get(
                    'reduction', composite_v5.DEFAULT_WEIGHTS_V5['reduction'])
                _agg5 = composite_v5.aggregate_agent_results_v5(
                    per_agent_results, reduction=_reduction)
                print(composite_v5.format_diagnostic_lines_v5(_agg5, ep_reward_team),
                      flush=True)
                ep_score = _agg5['score']
                per_episode_components_v5.append({
                    'aggregate':   _agg5,
                    'per_agent':   per_agent_results,
                    'episode_idx': ep_idx,
                    'seed':        ep_seed,
                })
            elif fitness_stat == 'composite_v7_zstat':
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                # NOTE: do NOT reuse the name `episode_rewards` here — that is
                # the outer 40-episode accumulator (declared above the loop).
                # Reassigning it clobbered the accumulator with this episode's 4
                # per-agent env rewards, so the GA's Hebbian selection scalar
                # became mean([last episode's 4 env rewards + its composite]) —
                # a single-episode, env-reward-dominated number, NOT the 40-ep
                # composite mean. The static evaluators never hit this because
                # their equivalent line is function-local in _score_episode_v7.
                team_env_rewards = list(sim.episode_rewards)
                ep_reward_team = float(sum(team_env_rewards))
                behaviour_capped, _bh_raw, _ = composite_v7.compute_v3_behaviour(
                    agent_stats_list, max_steps
                )
                per_agent_results = []
                for _i, _acc in enumerate(accumulators or []):
                    _stats_i = agent_stats_list[_i] if _i < len(agent_stats_list) else {}
                    _ep_reward_agent = float(team_env_rewards[_i]) if _i < len(team_env_rewards) else 0.0
                    per_agent_results.append(
                        _acc.episode_score(
                            ep_reward_total=_ep_reward_agent,
                            exploration_capped=behaviour_capped,
                            agent_stats=_stats_i,
                            weights=v7_weights,
                        )
                    )
                _reduction = (v7_weights or {}).get(
                    'reduction', composite_v7.DEFAULT_WEIGHTS_V7['reduction'])
                _agg7 = composite_v7.aggregate_agent_results_v7(
                    per_agent_results, reduction=_reduction)
                print(composite_v7.format_diagnostic_lines_v7(_agg7, ep_reward_team),
                      flush=True)
                ep_score = _agg7['score']
                per_episode_components_v7.append({
                    'aggregate':   _agg7,
                    'per_agent':   per_agent_results,
                    'episode_idx': ep_idx,
                    'seed':        ep_seed,
                })
            elif fitness_stat == 'composite_v8_zstat':
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                # NOTE: do NOT reuse the name `episode_rewards` here (see the v7
                # branch above) -- that is the outer eval-episode accumulator.
                team_env_rewards = list(sim.episode_rewards)
                ep_reward_team = float(sum(team_env_rewards))
                behaviour_capped, _bh_raw, _ = composite_v8.compute_v3_behaviour(
                    agent_stats_list, max_steps
                )
                per_agent_results = []
                for _i, _acc in enumerate(accumulators or []):
                    _stats_i = agent_stats_list[_i] if _i < len(agent_stats_list) else {}
                    _ep_reward_agent = float(team_env_rewards[_i]) if _i < len(team_env_rewards) else 0.0
                    per_agent_results.append(
                        _acc.episode_score(
                            ep_reward_total=_ep_reward_agent,
                            exploration_capped=behaviour_capped,
                            agent_stats=_stats_i,
                            weights=v8_weights,
                        )
                    )
                _reduction = (v8_weights or {}).get(
                    'reduction', composite_v8.DEFAULT_WEIGHTS_V8['reduction'])
                _agg8 = composite_v8.aggregate_agent_results_v8(
                    per_agent_results, reduction=_reduction)
                print(composite_v8.format_diagnostic_lines_v8(_agg8, ep_reward_team),
                      flush=True)
                ep_score = _agg8['score']
                per_episode_components_v8.append({
                    'aggregate':   _agg8,
                    'per_agent':   per_agent_results,
                    'episode_idx': ep_idx,
                    'seed':        ep_seed,
                })
            elif fitness_stat == 'composite_v3_ep_dominant':
                # v3: ep_reward dominates; behaviour only prevents the do-nothing basin.
                #   - behaviour capped at 0.8 (no reward for excess wandering)
                #   - ep_reward weight 1.0 (was 0.5 in v2)
                #   - ep_reward floor -1.0 (wall-trap episodes can't dominate)
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                ep_reward_total = float(sum(sim.episode_rewards))
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
                    comp_log.append((unique, dist, n_moves, fail_ratio, activity, efficiency, behaviour_raw))
                behaviour = float(np.mean(agent_scores))
                behaviour_capped = min(behaviour, 0.8)
                ep_reward_clipped = max(ep_reward_total, -1.0)
                ep_score = behaviour_capped + 1.0 * ep_reward_clipped
                u, d, nm, fr, act, eff, bh_raw = (float(np.mean(x)) for x in zip(*comp_log))
                print(
                    f'  v3: unique={u:.1f} dist={d:.1f} n_moves={nm:.1f} '
                    f'fail_ratio={fr:.3f} activity={act:.3f} efficiency={eff:.3f} '
                    f'ep_reward={ep_reward_total:.4f} ep_reward_clipped={ep_reward_clipped:.4f} '
                    f'behaviour={behaviour:.4f} behaviour_capped={behaviour_capped:.4f} -> score={ep_score:.4f}',
                    flush=True,
                )
            else:
                agent_stats_list = rollout._sim.episode_stats['agent']
                vals = [float(s.get(fitness_stat, 0.0)) for s in agent_stats_list]
                raw = float(np.mean(vals))
                max_steps = env_cfg.game.max_steps
                ep_score = raw / max_steps if max_steps > 0 else raw
            episode_rewards.append(ep_score)
            print(f' Episode {ep_idx} cumulative rewards  {ep_score:.4f}', flush=True)

        mean_reward = float(np.mean(episode_rewards))
        print(f'\n Episode cumulative rewards  {mean_reward:.8f}', flush=True)

        if fitness_stat == 'composite_v5_role_event' and components_sidecar:
            ep_aggs = [ec['aggregate'] for ec in per_episode_components_v5]
            over_eps = composite_v5.aggregate_over_episodes_v5(ep_aggs)
            try:
                composite_v5.write_component_sidecar(
                    components_sidecar,
                    per_episode_components_v5,
                    over_eps,
                )
                print(f'  v5: components sidecar written: {components_sidecar}', flush=True)
            except Exception as _sc_exc:
                print(f'  v5: WARN failed to write sidecar ({_sc_exc})', flush=True)

        if fitness_stat == 'composite_v7_zstat' and components_sidecar:
            ep_aggs = [ec['aggregate'] for ec in per_episode_components_v7]
            over_eps = composite_v7.aggregate_over_episodes_v7(ep_aggs)
            try:
                composite_v7.write_component_sidecar(
                    components_sidecar,
                    per_episode_components_v7,
                    over_eps,
                )
                print(f'  v7: components sidecar written: {components_sidecar}', flush=True)
            except Exception as _sc_exc:
                print(f'  v7: WARN failed to write sidecar ({_sc_exc})', flush=True)

        if fitness_stat == 'composite_v8_zstat' and components_sidecar:
            ep_aggs = [ec['aggregate'] for ec in per_episode_components_v8]
            over_eps = composite_v8.aggregate_over_episodes_v8(ep_aggs)
            try:
                composite_v8.write_component_sidecar(
                    components_sidecar,
                    per_episode_components_v8,
                    over_eps,
                )
                print(f'  v8: components sidecar written: {components_sidecar}', flush=True)
            except Exception as _sc_exc:
                print(f'  v8: WARN failed to write sidecar ({_sc_exc})', flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(
        description='Evaluate Hebbian MLP policy on CoGames environment'
    )
    parser.add_argument(
        '--mission', type=str, default='tutorial.scout',
        help='CoGames mission name (e.g. tutorial.scout)',
    )
    parser.add_argument(
        '--hebb_rule', type=str, default='ABCD_lr',
        help='Hebbian rule: A, AD, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr, ABCD_lr_D_out, ABCD_lr_D_in_and_out',
    )
    parser.add_argument(
        '--init_weights', type=str, default='uni',
        help='Weight init distribution: uni, normal, default, xa_uni, sparse, ka_uni',
    )
    parser.add_argument(
        '--hebb_coeffs_path', type=str, default=None,
        help='Path to evolved Hebbian coefficients (.pt file)',
    )
    parser.add_argument(
        '--render', type=str, default='False',
        help='Render environment: True/False',
    )
    parser.add_argument(
        '--eval_episodes', type=int, default=1,
        help='Number of episodes to average per fitness evaluation',
    )
    parser.add_argument(
        '--seed_base', type=int, default=12345,
        help='Base seed for deterministic per-episode seeds',
    )
    parser.add_argument(
        '--fitness_stat', type=str, default='episode_reward',
        help=(
            'Stat to use as fitness score. '
            '"episode_reward" uses the env reward signal (e.g. chest deposit). '
            'Any agent stat key (e.g. "cell.unique_visited") is read from '
            'episode_stats and normalised by max_steps.'
        ),
    )
    parser.add_argument(
        '--num_agents', type=int, default=None,
        help='Override num_agents (e.g. 4 when map has only 4 spawn points)',
    )
    parser.add_argument(
        '--obs_encoder', type=str, default='none',
        choices=list(_obs_encoder.VALID_KINDS),
        help='Frozen pre-network observation encoder. random_proj collapses '
             'the 900-dim raw obs to --encoder_dim before the Hebbian MLP.',
    )
    parser.add_argument('--encoder_dim', type=int, default=64,
                        help='Output dim of the frozen obs encoder.')
    parser.add_argument('--encoder_seed', type=int, default=1337,
                        help='Seed for the frozen obs encoder matrix; share '
                             'across architectures => identical projection.')
    parser.add_argument(
        '--fitness_v4_weights', type=str, default='',
        help='Comma-separated key=val list of v4 weights/coefficients '
             '(see composite_v4.DEFAULT_WEIGHTS for keys).',
    )
    parser.add_argument(
        '--fitness_v5_weights', type=str, default='',
        help='Comma-separated key=val list of v5 weights/coefficients '
             '(see composite_v5.DEFAULT_WEIGHTS_V5 for keys; the special '
             'key reduction=max|mean|min is also accepted).',
    )
    parser.add_argument(
        '--fitness_v7_weights', type=str, default='',
        help='Comma-separated key=val list of v7 weights/coefficients '
             '(see composite_v7.DEFAULT_WEIGHTS_V7 for keys; the special '
             'key reduction accepts max|mean|min|trimmed_mean|top2_mean).',
    )
    parser.add_argument(
        '--fitness_v8_weights', type=str, default='',
        help='Comma-separated key=val list of v8 weights/coefficients '
             '(see composite_v8.DEFAULT_WEIGHTS_V8 for keys; the special '
             'key reduction accepts max|mean|min|trimmed_mean|top2_mean|'
             'bottom2_mean).',
    )
    parser.add_argument(
        '--max_steps', type=int, default=None,
        help='Override the mission episode length (env.game.max_steps). v8 '
             'needs a long horizon for the full mine->deposit->heart->align loop.',
    )
    parser.add_argument(
        '--components_sidecar', type=str, default='',
        help='If set and fitness_stat is composite_v5_role_event or '
             'composite_v7_zstat, dump per-episode + aggregate component '
             'dicts to this JSON path before exit (atomic write).',
    )

    args = parser.parse_args(argv[1:])

    if args.hebb_coeffs_path is None:
        print('ERROR: --hebb_coeffs_path is required', flush=True)
        sys.exit(1)

    hebb_coeffs = torch.load(args.hebb_coeffs_path, weights_only=False)
    if not isinstance(hebb_coeffs, np.ndarray):
        hebb_coeffs = hebb_coeffs.numpy() if hasattr(hebb_coeffs, 'numpy') else np.asarray(hebb_coeffs)
    hebb_coeffs = np.asarray(hebb_coeffs, dtype=np.float32)

    render = str(args.render).lower() == 'true'

    evaluate_hebb_cogames(
        hebb_rule=args.hebb_rule,
        mission=args.mission,
        init_weights=args.init_weights,
        render=render,
        eval_episodes=args.eval_episodes,
        seed_base=args.seed_base,
        hebb_coeffs=hebb_coeffs,
        fitness_stat=args.fitness_stat,
        num_agents=args.num_agents,
        obs_encoder_kind=args.obs_encoder,
        encoder_dim=args.encoder_dim,
        encoder_seed=args.encoder_seed,
        fitness_v4_weights=composite_v7.parse_weights_str(args.fitness_v4_weights),
        fitness_v5_weights=composite_v7.parse_weights_str(args.fitness_v5_weights),
        fitness_v7_weights=composite_v7.parse_weights_str(args.fitness_v7_weights),
        fitness_v8_weights=composite_v8.parse_weights_str(args.fitness_v8_weights),
        components_sidecar=args.components_sidecar,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main(sys.argv)

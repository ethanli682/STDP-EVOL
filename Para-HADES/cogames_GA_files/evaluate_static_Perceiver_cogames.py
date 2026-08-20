import os
import sys
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Perceiver-lite cross-attention policy for CoGames.
#
# Input side uses the same stacked_grid encoding as evaluate_static_GRU_cogames.py:
# each raw observation is decoded into a [11x11x3 spatial grid | 70-dim global]
# vector of length 433. The last FRAME_STACK frames are kept; instead of flattening
# them into one vector, we treat the four frames as four 433-dim *tokens*.
#
# A small bank of N persistent latent vectors is carried across timesteps within
# an episode (reset to an evolved L_init at episode start). At every step the
# latents cross-attend to the frame tokens and are refreshed via a residual
# update. The latents provide memory that extends beyond the frame_stack window
# (analogous to GRU hidden state, but shaped like attention).
#
# Layout:
#   tokens  : (frame_stack, per_frame_dim=433)
#   tokens' : tanh(W_proj @ tokens)              (frame_stack, d_kv)
#   K = W_k @ tokens'                            (frame_stack, d_qk)
#   V = W_v @ tokens'                            (frame_stack, d_v)
#   Q = W_q @ latents                            (N, d_qk)
#   A = softmax(Q K^T / sqrt(d_qk)) @ V          (N, d_v)
#   latents <- tanh(latents + W_o @ A)           (N, d_latent)
#   logits  = W_out @ mean(latents)              (action_dim,)
#
# We set d_latent == d_qk == d_v == d_kv == `d_model` to keep the gene compact.
# ---------------------------------------------------------------------------

GRID_H, GRID_W   = 11, 11
NUM_CHANNELS     = 3
GLOBAL_STATE_DIM = 70
FRAME_STACK      = 2   # 2026-05-22: dropped 5 -> 2 (see evaluate_static_cogames.py).

SPATIAL_DIM   = GRID_H * GRID_W * NUM_CHANNELS           # 363
PER_FRAME_DIM = SPATIAL_DIM + GLOBAL_STATE_DIM           # 433

EMPTY_TOKEN_LOC  = 0xFF
GLOBAL_TOKEN_LOC = 0xFE


def obs_to_frame(tokens, num_expected_tokens: int) -> np.ndarray:
    grid = np.zeros((GRID_H, GRID_W, NUM_CHANNELS), dtype=np.float32)
    global_vec = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)

    for idx, tok in enumerate(tokens):
        if idx >= num_expected_tokens:
            break
        raw = tok.raw_token
        loc = int(raw[0])
        fid = int(raw[1]) if len(raw) > 1 else 0
        val = int(raw[2]) if len(raw) > 2 else 0

        if loc == EMPTY_TOKEN_LOC:
            continue
        if loc == GLOBAL_TOKEN_LOC:
            if 0 <= fid < GLOBAL_STATE_DIM:
                global_vec[fid] = val / 255.0
            continue

        r = (loc >> 4) & 0xF
        c = loc & 0xF
        if r >= GRID_H or c >= GRID_W:
            continue
        grid[r, c, 0] = 1.0
        grid[r, c, 1] = fid / 255.0
        grid[r, c, 2] = val / 255.0

    return np.concatenate([grid.reshape(-1), global_vec])


# ---------------------------------------------------------------------------
# Perceiver-lite module
# ---------------------------------------------------------------------------

class Perceiver_static_cogames(nn.Module):
    def __init__(
        self,
        per_frame_dim: int,
        num_tokens: int,
        action_dim: int,
        d_model: int = 32,
        num_latents: int = 8,
    ):
        super().__init__()
        self.per_frame_dim = per_frame_dim
        self.num_tokens = num_tokens
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_latents = num_latents

        # No biases anywhere; every tensor below is a flat-array gene.
        self.W_proj = nn.Parameter(torch.zeros(d_model, per_frame_dim), requires_grad=False)
        self.W_q    = nn.Parameter(torch.zeros(d_model, d_model),       requires_grad=False)
        self.W_k    = nn.Parameter(torch.zeros(d_model, d_model),       requires_grad=False)
        self.W_v    = nn.Parameter(torch.zeros(d_model, d_model),       requires_grad=False)
        self.W_o    = nn.Parameter(torch.zeros(d_model, d_model),       requires_grad=False)
        self.L_init = nn.Parameter(torch.zeros(num_latents, d_model),   requires_grad=False)
        self.W_out  = nn.Parameter(torch.zeros(action_dim, d_model),    requires_grad=False)

        self._scale = 1.0 / float(d_model) ** 0.5

    def init_latents(self) -> torch.Tensor:
        return self.L_init.detach().clone()

    def forward(self, tokens: torch.Tensor, latents: torch.Tensor):
        # tokens:  (num_tokens, per_frame_dim)
        # latents: (num_latents, d_model)
        t_proj = torch.tanh(tokens @ self.W_proj.T)        # (T, d_model)
        K = t_proj @ self.W_k.T                             # (T, d_model)
        V = t_proj @ self.W_v.T                             # (T, d_model)
        Q = latents @ self.W_q.T                            # (N, d_model)

        attn = F.softmax((Q @ K.T) * self._scale, dim=-1)   # (N, T)
        A = attn @ V                                        # (N, d_model)

        latents_new = torch.tanh(latents + A @ self.W_o.T)  # (N, d_model)

        pooled = latents_new.mean(dim=0)                    # (d_model,)
        logits = pooled @ self.W_out.T                      # (action_dim,)
        return logits, latents_new


class CoGamesPerceiverAgentPolicy(AgentPolicy):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        network: Perceiver_static_cogames,
        frame_stack: int = FRAME_STACK,
        encoder=None,
        encoded_dim: int = 0,
    ):
        super().__init__(policy_env_info)
        self._p = network.float().eval()
        self._latents = self._p.init_latents()
        self._action_names = policy_env_info.action_names
        self._obs_shape = policy_env_info.observation_space.shape
        self._frame_stack = frame_stack
        self._encoder = encoder
        per_frame = int(encoded_dim) if encoded_dim else PER_FRAME_DIM
        self._history: deque = deque(
            [np.zeros(per_frame, dtype=np.float32) for _ in range(frame_stack)],
            maxlen=frame_stack,
        )

    def _build_tokens(self, obs: AgentObservation) -> torch.Tensor:
        frame = obs_to_frame(obs.tokens, self._obs_shape[0])
        if self._encoder is not None:
            frame = self._encoder(frame)
        self._history.append(frame)
        tokens_np = np.stack(list(self._history), axis=0)   # (frame_stack, per_frame_dim)
        return torch.as_tensor(tokens_np, dtype=torch.float32)

    def step(self, obs: AgentObservation) -> Action:
        tokens = self._build_tokens(obs)
        with torch.no_grad():
            logits, self._latents = self._p(tokens, self._latents)
        action_idx = int(np.argmax(logits.numpy()))
        return Action(name=self._action_names[action_idx])


# ---------------------------------------------------------------------------
# Scoring helper  (identical to evaluate_static_GRU_cogames.py)
# ---------------------------------------------------------------------------

def _score_episode(rollout, env_cfg, fitness_stat: str) -> float:
    sim = rollout._sim
    if fitness_stat == 'episode_reward':
        return float(sum(sim.episode_rewards))

    if fitness_stat == 'map_coverage':
        grid_objs = sim.grid_objects()
        wall_cells = {o['location'] for o in grid_objs.values() if o['type_name'] == 'wall'}
        walkable = max(sim.map_width * sim.map_height - len(wall_cells), 1)
        agent_stats_list = sim.episode_stats['agent']
        coverages = [
            min(float(s.get('cell.unique_visited', 0.0)) / walkable, 1.0)
            for s in agent_stats_list
        ]
        score = float(np.mean(coverages))
        print(
            f'  walkable={walkable}  '
            f'unique_per_agent={[int(s.get("cell.unique_visited",0)) for s in agent_stats_list]}  '
            f'coverage={score:.4f}',
            flush=True,
        )
        return score

    if fitness_stat == 'composite_exploration':
        agent_stats_list = sim.episode_stats['agent']
        max_steps = max(env_cfg.game.max_steps or 1000, 100)
        dist_norm = max_steps ** 0.5
        agent_scores = []
        for s in agent_stats_list:
            unique  = float(s.get('cell.unique_visited', 0.0))
            visited = float(s.get('cell.visited', 0.0))
            dist    = float(s.get('cell.max_distance_from_spawn', 0.0))
            n_moves = float(s.get('action.move.success', 0.0)) + float(s.get('action.move.failed', 0.0))
            fail_ratio = float(s.get('action.move.failed', 0.0)) / max(n_moves, 1.0)
            efficiency = unique / max(visited, 1.0)
            agent_scores.append(
                unique / max_steps
                + 0.4 * efficiency
                + 0.3 * dist / dist_norm
                - 0.5 * fail_ratio ** 2
            )
        return float(np.mean(agent_scores))

    if fitness_stat == 'composite_exploration_v2':
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
            efficiency = max(unique - 1.0, 0.0) / max(n_moves, 1.0)
            agent_scores.append(
                1.0 * (unique / max_steps)
                + 0.5 * (dist / dist_norm)
                + 0.3 * activity
                + 0.2 * efficiency * activity
                - 0.2 * fail_ratio
            )
            comp_log.append((unique, visited, dist, n_moves, fail_ratio, activity, efficiency))
        behaviour = float(np.mean(agent_scores))
        score = behaviour + 0.5 * ep_reward_total
        u, v, d, nm, fr, act, eff = (float(np.mean(x)) for x in zip(*comp_log))
        print(
            f'  v2: unique={u:.1f} visited={v:.1f} dist={d:.1f} n_moves={nm:.1f} '
            f'fail_ratio={fr:.3f} activity={act:.3f} efficiency={eff:.3f} '
            f'ep_reward={ep_reward_total:.4f} behaviour={behaviour:.4f} -> score={score:.4f}',
            flush=True,
        )
        return score

    if fitness_stat == 'composite_v4_role_gated':
        raise RuntimeError(
            'composite_v4_role_gated must go through _score_episode_v4 — '
            'check the evaluation loop wiring.'
        )

    if fitness_stat == 'composite_v3_ep_dominant':
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
        score = behaviour_capped + 1.0 * ep_reward_clipped
        u, d, nm, fr, act, eff, bh_raw = (float(np.mean(x)) for x in zip(*comp_log))
        print(
            f'  v3: unique={u:.1f} dist={d:.1f} n_moves={nm:.1f} '
            f'fail_ratio={fr:.3f} activity={act:.3f} efficiency={eff:.3f} '
            f'ep_reward={ep_reward_total:.4f} ep_reward_clipped={ep_reward_clipped:.4f} '
            f'behaviour={behaviour:.4f} behaviour_capped={behaviour_capped:.4f} -> score={score:.4f}',
            flush=True,
        )
        return score

    agent_stats_list = sim.episode_stats['agent']
    vals = [float(s.get(fitness_stat, 0.0)) for s in agent_stats_list]
    raw = float(np.mean(vals))
    max_steps = env_cfg.game.max_steps
    return raw / max_steps if max_steps else raw


def _score_episode_v4(rollout, env_cfg, accumulators, weights):
    """composite_v4_role_gated scoring (see evaluate_static_cogames._score_episode_v4)."""
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _ = composite_v4.compute_v3_behaviour(
        agent_stats_list, max_steps
    )
    per_agent_results = []
    for i, acc in enumerate(accumulators):
        stats_i = agent_stats_list[i] if i < len(agent_stats_list) else {}
        ep_reward_agent = float(episode_rewards[i]) if i < len(episode_rewards) else 0.0
        per_agent_results.append(
            acc.episode_score(
                ep_reward_total=ep_reward_agent,
                exploration_capped=behaviour_capped,
                agent_stats=stats_i,
                weights=weights,
            )
        )
    agg = composite_v4.aggregate_agent_results(per_agent_results)
    print(composite_v4.format_diagnostic_lines(agg, ep_reward_team), flush=True)
    return agg['score']


def _score_episode_v5(rollout, env_cfg, accumulators, weights):
    """composite_v5_role_event scoring (see evaluate_static_cogames._score_episode_v5)."""
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _ = composite_v5.compute_v3_behaviour(
        agent_stats_list, max_steps
    )
    per_agent_results = []
    for i, acc in enumerate(accumulators):
        stats_i = agent_stats_list[i] if i < len(agent_stats_list) else {}
        ep_reward_agent = float(episode_rewards[i]) if i < len(episode_rewards) else 0.0
        per_agent_results.append(
            acc.episode_score(
                ep_reward_total=ep_reward_agent,
                exploration_capped=behaviour_capped,
                agent_stats=stats_i,
                weights=weights,
            )
        )
    reduction = (weights or {}).get('reduction',
                                    composite_v5.DEFAULT_WEIGHTS_V5['reduction'])
    agg = composite_v5.aggregate_agent_results_v5(per_agent_results,
                                                   reduction=reduction)
    print(composite_v5.format_diagnostic_lines_v5(agg, ep_reward_team), flush=True)
    episode_components = {'aggregate': agg, 'per_agent': per_agent_results}
    return float(agg['score']), episode_components


def _score_episode_v7(rollout, env_cfg, accumulators, weights):
    """composite_v7_zstat scoring (see evaluate_static_cogames._score_episode_v7)."""
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _ = composite_v7.compute_v3_behaviour(
        agent_stats_list, max_steps
    )
    per_agent_results = []
    for i, acc in enumerate(accumulators):
        stats_i = agent_stats_list[i] if i < len(agent_stats_list) else {}
        ep_reward_agent = float(episode_rewards[i]) if i < len(episode_rewards) else 0.0
        per_agent_results.append(
            acc.episode_score(
                ep_reward_total=ep_reward_agent,
                exploration_capped=behaviour_capped,
                agent_stats=stats_i,
                weights=weights,
            )
        )
    reduction = (weights or {}).get('reduction',
                                    composite_v7.DEFAULT_WEIGHTS_V7['reduction'])
    agg = composite_v7.aggregate_agent_results_v7(per_agent_results,
                                                   reduction=reduction)
    print(composite_v7.format_diagnostic_lines_v7(agg, ep_reward_team), flush=True)
    episode_components = {'aggregate': agg, 'per_agent': per_agent_results}
    return float(agg['score']), episode_components


def _score_episode_v8(rollout, env_cfg, accumulators, weights):
    """composite_v8_zstat scoring (see evaluate_static_cogames._score_episode_v8)."""
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _ = composite_v8.compute_v3_behaviour(
        agent_stats_list, max_steps
    )
    per_agent_results = []
    for i, acc in enumerate(accumulators):
        stats_i = agent_stats_list[i] if i < len(agent_stats_list) else {}
        ep_reward_agent = float(episode_rewards[i]) if i < len(episode_rewards) else 0.0
        per_agent_results.append(
            acc.episode_score(
                ep_reward_total=ep_reward_agent,
                exploration_capped=behaviour_capped,
                agent_stats=stats_i,
                weights=weights,
            )
        )
    reduction = (weights or {}).get('reduction',
                                    composite_v8.DEFAULT_WEIGHTS_V8['reduction'])
    agg = composite_v8.aggregate_agent_results_v8(per_agent_results,
                                                   reduction=reduction)
    print(composite_v8.format_diagnostic_lines_v8(agg, ep_reward_team), flush=True)
    episode_components = {'aggregate': agg, 'per_agent': per_agent_results}
    return float(agg['score']), episode_components


# ---------------------------------------------------------------------------
# Weight loading: flat arrays -> Perceiver_static_cogames
# ---------------------------------------------------------------------------

def _load_weights_into_model(
    model: Perceiver_static_cogames,
    weights_path: str,
    per_frame_dim: int,
    d_model: int,
    num_latents: int,
    num_actions: int,
) -> None:
    blob = torch.load(weights_path, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(
            'Perceiver weights file must contain a dict with keys '
            'W_proj, W_q, W_k, W_v, W_o, L_init, W_out.'
        )

    shapes = {
        'W_proj': (d_model, per_frame_dim),
        'W_q':    (d_model, d_model),
        'W_k':    (d_model, d_model),
        'W_v':    (d_model, d_model),
        'W_o':    (d_model, d_model),
        'L_init': (num_latents, d_model),
        'W_out':  (num_actions, d_model),
    }
    targets = {
        'W_proj': model.W_proj,
        'W_q':    model.W_q,
        'W_k':    model.W_k,
        'W_v':    model.W_v,
        'W_o':    model.W_o,
        'L_init': model.L_init,
        'W_out':  model.W_out,
    }

    for name, (rows, cols) in shapes.items():
        if name not in blob:
            raise ValueError(f"Missing weight '{name}' in {weights_path}.")
        arr = np.asarray(blob[name], dtype=np.float32).ravel()
        expected = rows * cols
        if arr.size != expected:
            raise ValueError(
                f"Weight '{name}' has {arr.size} values; expected {expected} "
                f"for shape ({rows}, {cols})."
            )
        targets[name].data = torch.from_numpy(arr.reshape(rows, cols))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_static_Perceiver_cogames(
    mission: str,
    weights_path: str,
    d_model: int = 32,
    num_latents: int = 8,
    render: bool = False,
    eval_episodes: int = 1,
    seed_base: int = 12345,
    fitness_stat: str = 'episode_reward',
    num_agents: int = None,
    frame_stack: int = FRAME_STACK,
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
    v4_weights = fitness_v4_weights or {}
    v5_weights = fitness_v5_weights or {}
    v7_weights = fitness_v7_weights or {}
    v8_weights = fitness_v8_weights or {}
    per_episode_components_v8 = []
    per_episode_components_v5 = []
    per_episode_components_v7 = []
    with torch.no_grad():
        eval_episodes = max(1, int(eval_episodes))
        episode_rewards = []

        from cogames.cli.mission import get_mission as _get_mission
        _, env_cfg, _ = _get_mission(mission)

        if num_agents is not None:
            env_cfg.game.num_agents = num_agents
            try:
                env_cfg.game.map_builder.instance.spawn_count = num_agents
            except AttributeError:
                pass

        # Optional episode-length override (v8 needs a long horizon for the full
        # mine -> deposit -> craft heart -> pick capturer role -> align loop).
        if max_steps is not None:
            env_cfg.game.max_steps = int(max_steps)

        policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
        obs_shape = policy_env_info.observation_space.shape
        num_actions = len(policy_env_info.action_names)
        num_agents = env_cfg.game.num_agents

        raw_per_frame = PER_FRAME_DIM
        encoder, per_frame_dim = _obs_encoder.maybe_make_encoder(
            obs_encoder_kind, raw_per_frame, encoder_dim, encoder_seed
        )
        num_tokens    = frame_stack

        print(
            f'CoGames Perceiver eval: mission={mission}  obs_shape={obs_shape}  '
            f'frame_stack={frame_stack}  raw_per_frame={raw_per_frame}  '
            f'obs_encoder={obs_encoder_kind}  encoder_dim={encoder_dim}  '
            f'encoder_seed={encoder_seed}  '
            f'per_frame_dim={per_frame_dim}  '
            f'num_tokens={num_tokens}  d_model={d_model}  num_latents={num_latents}  '
            f'num_actions={num_actions}  num_agents={num_agents}  episodes={eval_episodes}',
            flush=True,
        )

        for ep_idx in range(eval_episodes):
            ep_seed = int(seed_base) + ep_idx
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)

            agent_policies = []
            for _agent_id in range(num_agents):
                net = Perceiver_static_cogames(
                    per_frame_dim=per_frame_dim,
                    num_tokens=num_tokens,
                    action_dim=num_actions,
                    d_model=d_model,
                    num_latents=num_latents,
                )
                _load_weights_into_model(
                    net, weights_path, per_frame_dim, d_model, num_latents, num_actions
                )
                agent_policies.append(
                    CoGamesPerceiverAgentPolicy(
                        policy_env_info, net, frame_stack=frame_stack,
                        encoder=encoder, encoded_dim=per_frame_dim,
                    )
                )

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
            rollout = Rollout(env_cfg, agent_policies, render_mode=render_mode, seed=ep_seed)

            if accumulators is not None:
                _spawns = composite_v7.get_agent_spawn_positions(rollout._sim, num_agents)
                _extents = composite_v7.get_map_extents(rollout._sim)
                for _i, _acc in enumerate(accumulators):
                    _acc.set_spawn(_spawns[_i], _extents)
                _miner_tag = composite_v7.get_miner_station_tag(rollout._sim)
                for _acc in accumulators:
                    _acc.set_miner_tag(_miner_tag)

            rollout.run_until_done()

            if fitness_stat == 'composite_v4_role_gated':
                ep_score = _score_episode_v4(rollout, env_cfg, accumulators, v4_weights)
            elif fitness_stat == 'composite_v5_role_event':
                ep_score, ep_components = _score_episode_v5(
                    rollout, env_cfg, accumulators, v5_weights
                )
                ep_components['episode_idx'] = ep_idx
                ep_components['seed'] = ep_seed
                per_episode_components_v5.append(ep_components)
            elif fitness_stat == 'composite_v7_zstat':
                ep_score, ep_components = _score_episode_v7(
                    rollout, env_cfg, accumulators, v7_weights
                )
                ep_components['episode_idx'] = ep_idx
                ep_components['seed'] = ep_seed
                per_episode_components_v7.append(ep_components)
            elif fitness_stat == 'composite_v8_zstat':
                ep_score, ep_components = _score_episode_v8(
                    rollout, env_cfg, accumulators, v8_weights
                )
                ep_components['episode_idx'] = ep_idx
                ep_components['seed'] = ep_seed
                per_episode_components_v8.append(ep_components)
            else:
                ep_score = _score_episode(rollout, env_cfg, fitness_stat)
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


def main(argv):
    parser = argparse.ArgumentParser(
        description='Evaluate a static Perceiver-lite cross-attention policy on CoGames.'
    )
    parser.add_argument('--mission', type=str, default='tutorial.scout')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_latents', type=int, default=8)
    parser.add_argument('--render', type=str, default='False')
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--seed_base', type=int, default=12345)
    parser.add_argument('--fitness_stat', type=str, default='episode_reward')
    parser.add_argument('--num_agents', type=int, default=None)
    parser.add_argument('--frame_stack', type=int, default=FRAME_STACK)
    parser.add_argument('--obs_encoder', type=str, default='none',
                        choices=list(_obs_encoder.VALID_KINDS),
                        help='Frozen pre-network observation encoder. '
                             "'random_proj' projects each raw frame to "
                             '--encoder_dim with a fixed Gaussian matrix '
                             '(shrinks Perceiver W_proj from d_model*433 to '
                             'd_model*encoder_dim).')
    parser.add_argument('--encoder_dim', type=int, default=64,
                        help='Output dim of the frozen obs encoder (per frame).')
    parser.add_argument('--encoder_seed', type=int, default=1337,
                        help='Seed for the frozen obs encoder matrix; share '
                             'across architectures => identical projection.')
    parser.add_argument('--fitness_v4_weights', type=str, default='',
                        help='Comma-separated key=val list of v4 weights/coefficients '
                             '(see composite_v4.DEFAULT_WEIGHTS for keys).')
    parser.add_argument('--fitness_v5_weights', type=str, default='',
                        help='Comma-separated key=val list of v5 weights/coefficients '
                             '(see composite_v5.DEFAULT_WEIGHTS_V5 for keys; the '
                             'special key reduction=max|mean|min is also accepted).')
    parser.add_argument('--fitness_v7_weights', type=str, default='',
                        help='Comma-separated key=val list of v7 weights/coefficients '
                             '(see composite_v7.DEFAULT_WEIGHTS_V7 for keys; the '
                             'special key reduction accepts '
                             'max|mean|min|trimmed_mean|top2_mean).')
    parser.add_argument('--fitness_v8_weights', type=str, default='',
                        help='Comma-separated key=val list of v8 weights/coefficients '
                             '(see composite_v8.DEFAULT_WEIGHTS_V8 for keys; the '
                             'special key reduction accepts '
                             'max|mean|min|trimmed_mean|top2_mean|bottom2_mean).')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override the mission episode length '
                             '(env.game.max_steps). v8 needs a long horizon for '
                             'the full mine->deposit->heart->align loop.')
    parser.add_argument('--components_sidecar', type=str, default='',
                        help='If set and fitness_stat is composite_v5_role_event '
                             'or composite_v7_zstat, dump per-episode + aggregate '
                             'component dicts to this JSON path before exit '
                             '(atomic write).')

    args = parser.parse_args(argv[1:])

    evaluate_static_Perceiver_cogames(
        mission=args.mission,
        weights_path=args.weights_path,
        d_model=args.d_model,
        num_latents=args.num_latents,
        render=str(args.render).lower() == 'true',
        eval_episodes=args.eval_episodes,
        seed_base=args.seed_base,
        fitness_stat=args.fitness_stat,
        num_agents=args.num_agents,
        frame_stack=args.frame_stack,
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

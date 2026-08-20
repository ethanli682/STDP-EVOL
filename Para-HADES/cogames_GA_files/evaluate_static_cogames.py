import os
import sys
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn

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
# Observation encoding
# ---------------------------------------------------------------------------
# Two raw obs modes:
#
#   legacy_flat  : flatten the raw (num_tokens, token_dim) uint8 observation
#                  array / 255 -> 900 floats for tutorial.scout. No stacking.
#
#   stacked_grid : decode tokens into a 11x11x3 spatial grid + length-70
#                  fid-indexed global-state vector per frame (433 dims/frame),
#                  then frame-stack the last FRAME_STACK frames oldest->newest.
#
# On top of either mode, an optional frozen random projection (see
# `obs_encoder.py`) collapses each raw frame to `encoder_dim` floats before it
# enters the evolved network. This is the lever that keeps W1's parameter
# count manageable for CMA-ES.
# ---------------------------------------------------------------------------

GRID_H, GRID_W   = 11, 11
NUM_CHANNELS     = 3
GLOBAL_STATE_DIM = 70
FRAME_STACK      = 2   # 2026-05-22: dropped from 5 to 2 — frame_stack > 2 mostly
                       # added redundant temporal channels and bloated W1 without
                       # shifting the inter-generation fitness signal.

SPATIAL_DIM   = GRID_H * GRID_W * NUM_CHANNELS           # 363
PER_FRAME_DIM = SPATIAL_DIM + GLOBAL_STATE_DIM           # 433

EMPTY_TOKEN_LOC  = 0xFF
GLOBAL_TOKEN_LOC = 0xFE


def obs_to_frame(tokens, num_expected_tokens: int) -> np.ndarray:
    """Decode one raw token observation into a 433-dim [spatial | global] frame."""
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


def legacy_obs_to_flat(tokens, obs_shape) -> np.ndarray:
    num_tokens, token_dim = obs_shape
    arr = np.zeros((num_tokens, token_dim), dtype=np.uint8)
    for idx, tok in enumerate(tokens):
        if idx >= num_tokens:
            break
        raw = tok.raw_token
        arr[idx, : len(raw)] = raw
    return arr.flatten().astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Static MLP — same topology as MLP_heb (900 -> 64 -> 32 -> 5), no bias,
# no plasticity. Weights are supplied by the GA as the gene.
# ---------------------------------------------------------------------------

class MLP_static_cogames(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, action_dim, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob[0]).float().detach()
        x1 = torch.tanh(self.fc1(state))
        x2 = torch.tanh(self.fc2(x1))
        return self.fc3(x2)


class CoGamesStaticAgentPolicy(AgentPolicy):
    """Per-agent policy running a fixed-weight MLP. No weight updates."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        network: MLP_static_cogames,
        obs_mode: str = "legacy_flat",
        frame_stack: int = FRAME_STACK,
        encoder=None,
        encoded_dim: int = 0,
    ):
        super().__init__(policy_env_info)
        self._p = network.float().eval()
        self._action_names = policy_env_info.action_names
        self._obs_shape = policy_env_info.observation_space.shape
        self._obs_mode = obs_mode
        self._frame_stack = frame_stack
        self._encoder = encoder
        # `encoded_dim` is the per-frame width AFTER the encoder (== encoder
        # output_dim if encoder is set, else raw per-frame size). We need it
        # to size the zero-padded history deque consistently with what the
        # network's W1 expects.
        if obs_mode == "stacked_grid":
            per_frame = int(encoded_dim) if encoded_dim else PER_FRAME_DIM
            self._history: deque = deque(
                [np.zeros(per_frame, dtype=np.float32) for _ in range(frame_stack)],
                maxlen=frame_stack,
            )
        else:
            self._history = None

    def _build_input(self, obs: AgentObservation) -> np.ndarray:
        if self._obs_mode == "stacked_grid":
            frame = obs_to_frame(obs.tokens, self._obs_shape[0])
            if self._encoder is not None:
                frame = self._encoder(frame)
            self._history.append(frame)
            return np.concatenate(list(self._history))  # oldest -> newest
        flat = legacy_obs_to_flat(obs.tokens, self._obs_shape)
        if self._encoder is not None:
            flat = self._encoder(flat)
        return flat

    def step(self, obs: AgentObservation) -> Action:
        net_in = self._build_input(obs)
        with torch.no_grad():
            logits = self._p([net_in]).numpy()
        action_idx = int(np.argmax(logits))
        return Action(name=self._action_names[action_idx])


# ---------------------------------------------------------------------------
# Episode fitness scoring — matches evaluate_hebb_cogames to keep fitness
# definitions comparable across Hebbian and static variants.
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
        # Guard: env max_steps=None or 0 would make unique/max_steps blow up.
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
                - 0.5 * fail_ratio ** 2     # convex wall-hit penalty
            )
        return float(np.mean(agent_scores))

    if fitness_stat == 'composite_exploration_v2':
        # Fixes three pathologies of composite_exploration that caused premature
        # convergence on april17 runs:
        #   (1) standing still → fail_ratio=0, activity=0 → score ≈ 0, while any
        #       random mover with wall hits → score < 0. GA discovers "don't move".
        #   (2) efficiency = unique/visited rewards 10-cell loops almost as much
        #       as 500-cell exploration.
        #   (3) the env's own episode reward (scout_gained etc.) never factored in.
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
            activity   = min(n_moves / max_steps, 1.0)        # floor against stand-still
            # Efficiency = fraction of moves that reached a cell BEYOND spawn.
            # Subtracting 1 from `unique` accounts for the spawn cell being
            # counted in `unique` but not corresponding to any move; without
            # it an idle agent scores 1.0 and a 1-move agent scores 2.0. The
            # original v2 used `unique / visited`, but `cell.visited` in
            # mettagrid is an 8x-per-step cell-ticks counter, not per-step
            # visits, so the ratio was ~0 for every policy.
            efficiency = max(unique - 1.0, 0.0) / max(n_moves, 1.0)

            coverage_term = unique / max_steps
            distance_term = dist / dist_norm
            eff_gated     = efficiency * activity              # only if agent actually moves
            wall_term     = fail_ratio                          # linear, not convex

            agent_scores.append(
                1.0 * coverage_term
                + 0.5 * distance_term
                + 0.3 * activity
                + 0.2 * eff_gated
                - 0.2 * wall_term
            )
            comp_log.append((unique, visited, dist, n_moves, fail_ratio, activity, efficiency))
        behaviour = float(np.mean(agent_scores))
        # Include the real task signal (scout_gained, cell.visited shaping, etc.) at small weight.
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
        # v4 is computed per-step in the episode loop (this helper is called
        # from there with rollout=ROLLOUT_AFTER_RUN). The actual score is
        # produced by _score_episode_v4 below.
        raise RuntimeError(
            'composite_v4_role_gated must go through _score_episode_v4 — '
            'check the evaluation loop wiring.'
        )

    if fitness_stat == 'composite_v3_ep_dominant':
        # v3: ep_reward dominates; behaviour only prevents the do-nothing basin.
        #   - behaviour capped at 0.8 -> no reward for excess wandering
        #   - ep_reward weight 1.0 (was 0.5 in v2); becomes sole ranking signal
        #     once the agent is active enough
        #   - ep_reward floor -1.0 -> a single wall-trap episode can't dominate
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
            coverage_term = unique / max_steps
            distance_term = dist / (max_steps ** 0.5)
            eff_gated     = efficiency * activity
            wall_term     = fail_ratio
            behaviour_raw = (
                1.0 * coverage_term
                + 0.5 * distance_term
                + 0.3 * activity
                + 0.2 * eff_gated
                - 0.2 * wall_term
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

    # Fallback: any agent stat key, normalised by max_steps.
    agent_stats_list = sim.episode_stats['agent']
    vals = [float(s.get(fitness_stat, 0.0)) for s in agent_stats_list]
    raw = float(np.mean(vals))
    max_steps = env_cfg.game.max_steps
    return raw / max_steps if max_steps else raw


def _score_episode_v4(rollout, env_cfg, accumulators, weights):
    """composite_v4_role_gated scoring.

    Per-step gear-gated role progress comes from `accumulators`, populated by
    the wrapped policy.step calls. Exploration is computed from the v3
    behaviour term using sim.episode_stats. Per-agent results are aggregated
    by mean.
    """
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, behaviour_raw_mean, _comp_log = composite_v4.compute_v3_behaviour(
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
    """composite_v5_role_event scoring.

    Mirrors `_score_episode_v4` in structure but returns
        (score, episode_components_dict)
    where episode_components_dict has 'aggregate' and 'per_agent' keys ready
    for sidecar persistence.
    """
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _comp_log = composite_v5.compute_v3_behaviour(
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

    episode_components = {
        'aggregate':  agg,
        'per_agent':  per_agent_results,
    }
    return float(agg['score']), episode_components


def _score_episode_v7(rollout, env_cfg, accumulators, weights):
    """composite_v7_zstat scoring.

    Mirrors `_score_episode_v5` but uses the v7 accumulator/aggregator which
    produces a smooth z-trajectory score (AlphaStar-style) and a Hamming
    cumulative-stats term. Returns
        (score, episode_components_dict)
    with the same sidecar-ready shape as v5.
    """
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _comp_log = composite_v7.compute_v3_behaviour(
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

    episode_components = {
        'aggregate':  agg,
        'per_agent':  per_agent_results,
    }
    return float(agg['score']), episode_components


def _score_episode_v8(rollout, env_cfg, accumulators, weights):
    """composite_v8_zstat scoring.

    Mirrors `_score_episode_v5` but uses the v7 accumulator/aggregator which
    produces a smooth z-trajectory score (AlphaStar-style) and a Hamming
    cumulative-stats term. Returns
        (score, episode_components_dict)
    with the same sidecar-ready shape as v5.
    """
    sim = rollout._sim
    agent_stats_list = sim.episode_stats['agent']
    max_steps = max(env_cfg.game.max_steps or 1000, 100)
    episode_rewards = list(sim.episode_rewards)
    ep_reward_team = float(sum(episode_rewards))

    behaviour_capped, _bh_raw, _comp_log = composite_v8.compute_v3_behaviour(
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

    episode_components = {
        'aggregate':  agg,
        'per_agent':  per_agent_results,
    }
    return float(agg['score']), episode_components


# ---------------------------------------------------------------------------
# Weight loading: flat float32 → Linear.weight.data
# ---------------------------------------------------------------------------

def _load_weights_into_model(model: MLP_static_cogames, weights_path: str,
                             input_dim: int, hidden, num_actions: int) -> None:
    blob = torch.load(weights_path, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError('Static weights file must contain a dict with keys W1, W2, W3.')

    h1, h2 = hidden
    shapes = {'W1': (h1, input_dim), 'W2': (h2, h1), 'W3': (num_actions, h2)}
    layers = {'W1': model.fc1, 'W2': model.fc2, 'W3': model.fc3}

    for name, (out_dim, in_dim) in shapes.items():
        if name not in blob:
            raise ValueError(f"Missing weight '{name}' in {weights_path}.")
        arr = np.asarray(blob[name], dtype=np.float32).ravel()
        expected = out_dim * in_dim
        if arr.size != expected:
            raise ValueError(
                f"Weight '{name}' has {arr.size} values; expected {expected} "
                f"for shape ({out_dim}, {in_dim})."
            )
        layers[name].weight.data = torch.from_numpy(arr.reshape(out_dim, in_dim))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def _raw_per_frame_dim(obs_shape, obs_mode: str) -> int:
    """Per-frame raw vector size before the optional encoder is applied."""
    if obs_mode == "stacked_grid":
        return PER_FRAME_DIM
    return int(np.prod(obs_shape))


def _resolve_input_dim(obs_shape, obs_mode: str, frame_stack: int,
                       obs_encoder_kind: str, encoder_dim: int) -> int:
    """Network input dim (== effective W1 cols)."""
    raw_per_frame = _raw_per_frame_dim(obs_shape, obs_mode)
    eff_per_frame = _obs_encoder.effective_dim(
        obs_encoder_kind, raw_per_frame, encoder_dim
    )
    return eff_per_frame * (frame_stack if obs_mode == "stacked_grid" else 1)


def evaluate_static_cogames(
    mission: str,
    weights_path: str,
    hidden=(64, 32),
    render: bool = False,
    eval_episodes: int = 1,
    seed_base: int = 12345,
    fitness_stat: str = 'episode_reward',
    num_agents: int = None,
    obs_mode: str = "legacy_flat",
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
    per_episode_components_v5 = []  # only populated under composite_v5_role_event
    per_episode_components_v7 = []  # only populated under composite_v7_zstat
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
        raw_per_frame = _raw_per_frame_dim(obs_shape, obs_mode)
        encoder, eff_per_frame = _obs_encoder.maybe_make_encoder(
            obs_encoder_kind, raw_per_frame, encoder_dim, encoder_seed
        )
        input_dim = _resolve_input_dim(
            obs_shape, obs_mode, frame_stack, obs_encoder_kind, encoder_dim
        )
        num_actions = len(policy_env_info.action_names)
        num_agents = env_cfg.game.num_agents

        print(
            f'CoGames static eval: mission={mission}  obs_shape={obs_shape}  '
            f'obs_mode={obs_mode}  frame_stack={frame_stack}  '
            f'obs_encoder={obs_encoder_kind}  encoder_dim={encoder_dim}  '
            f'encoder_seed={encoder_seed}  '
            f'raw_per_frame={raw_per_frame}  eff_per_frame={eff_per_frame}  '
            f'input_dim={input_dim}  '
            f'num_actions={num_actions}  num_agents={num_agents}  episodes={eval_episodes}  '
            f'hidden={hidden}',
            flush=True,
        )

        for ep_idx in range(eval_episodes):
            ep_seed = int(seed_base) + ep_idx
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)

            # One model per agent so Rollout can own them independently;
            # all share the same evolved weights (loaded from disk once per model).
            agent_policies = []
            for _agent_id in range(num_agents):
                net = MLP_static_cogames(input_dim, num_actions, hidden=hidden)
                _load_weights_into_model(net, weights_path, input_dim, hidden, num_actions)
                agent_policies.append(
                    CoGamesStaticAgentPolicy(
                        policy_env_info, net,
                        obs_mode=obs_mode, frame_stack=frame_stack,
                        encoder=encoder, encoded_dim=eff_per_frame,
                    )
                )

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
        description='Evaluate a static (non-plastic) MLP policy on CoGames.'
    )
    parser.add_argument('--mission', type=str, default='tutorial.scout')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to a torch.save dict with keys W1, W2, W3.')
    parser.add_argument('--hidden_1', type=int, default=64)
    parser.add_argument('--hidden_2', type=int, default=32)
    parser.add_argument('--render', type=str, default='False')
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--seed_base', type=int, default=12345)
    parser.add_argument('--fitness_stat', type=str, default='episode_reward')
    parser.add_argument('--num_agents', type=int, default=None)
    parser.add_argument('--obs_mode', type=str, default='legacy_flat',
                        choices=['legacy_flat', 'stacked_grid'])
    parser.add_argument('--frame_stack', type=int, default=FRAME_STACK)
    parser.add_argument('--obs_encoder', type=str, default='none',
                        choices=list(_obs_encoder.VALID_KINDS),
                        help='Frozen pre-network observation encoder. '
                             "'random_proj' projects each raw frame to "
                             '--encoder_dim with a fixed Gaussian matrix.')
    parser.add_argument('--encoder_dim', type=int, default=64,
                        help='Output dim of the frozen obs encoder (per frame).')
    parser.add_argument('--encoder_seed', type=int, default=1337,
                        help='Seed for the frozen obs encoder matrix; same value '
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

    evaluate_static_cogames(
        mission=args.mission,
        weights_path=args.weights_path,
        hidden=(args.hidden_1, args.hidden_2),
        render=str(args.render).lower() == 'true',
        eval_episodes=args.eval_episodes,
        seed_base=args.seed_base,
        fitness_stat=args.fitness_stat,
        num_agents=args.num_agents,
        obs_mode=args.obs_mode,
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

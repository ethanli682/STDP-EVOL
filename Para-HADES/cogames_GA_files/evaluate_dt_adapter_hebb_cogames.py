"""Evaluate a frozen DT + Hebbian Adapter policy on CoGames.

Used by task_cogames_dt_adapter_hebb.py as a subprocess worker.

Architecture
------------
  TokenEncoder  ->  ContextMemory (body)  ->  HebbianAdapter  ->  Heads
  [frozen]           [frozen]               [plastic weights]   [frozen]

The DT base (encoder, body, heads, adapter.norm) is frozen.
The Adapter's fc1 and fc2 weight matrices are:
  - re-sampled U[-0.1, 0.1] at the start of EVERY episode (Stage B §3.1)
  - updated in-place per-step by the ABCD+η Hebbian rule

Hebbian rule (ABCD_lr_D_in, col layout: [A, B, C, lr, D]):
  Δw_ij = lr_ij * (A_ij * o_i * o_j + B_ij * o_i + C_ij * o_j + D_ij)
  o_i = pre-synaptic activation, o_j = post-synaptic activation

Gene layout (hebb_coeffs shape: (N_SYN, 5)):
  rows 0 .. DIM*DIM-1        : fc1 synapses, synapse (out_j, in_i) at row j*DIM+i
  rows DIM*DIM .. 2*DIM*DIM-1: fc2 synapses, synapse (out_j, in_i) at row DIM*DIM+j*DIM+i

For default DIM=256: N_SYN = 131072, gene total = 655360 floats (5 arrays × 131072).

RTG is held constant at target_rtg throughout each episode (Stage B decision #6):
the adapter is supposed to push fitness above the conditioned ceiling without
access to the reward signal during the episode.

Fitness metric: composite_v7_zstat (miner-gated mine-and-deposit, same as
evaluate_hebb_cogames.py so DT+Adapter and standalone-Hebbian scores are
directly comparable).

Obs bridge: tutorial.scout gives (num_tokens, 3) tokens via AgentObservation;
the DT TokenEncoder expects (N, 500, 3) uint8 where empty slots use loc=0xFF.
We zero-pad from num_tokens -> 500 with the empty-loc marker.
"""

import os
import sys
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation
from mettagrid.simulator.rollout import Rollout

import composite_v7
try:
    import composite_v4
except ImportError:
    composite_v4 = None
try:
    import composite_v5
except ImportError:
    composite_v5 = None

# DT model + action utilities from CoGameAgent
_DT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'CoGameAgent'
)
if os.path.isdir(_DT_MODEL_DIR):
    sys.path.insert(0, os.path.abspath(_DT_MODEL_DIR))

from scripts.chunk_3.model import (  # noqa: E402
    build_model,
    NO_CHANGE_VIBE,
    factored_to_action12,
)

DT_OBS_TOKENS = 500   # DT TokenEncoder input width
DT_EMPTY_LOC  = 0xFF  # mettagrid empty-token marker (model.EMPTY_LOC)


# ---------------------------------------------------------------------------
# Numpy GELU (tanh approximation, matches torch.nn.functional.gelu)
# ---------------------------------------------------------------------------

def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# DTAdapterHebbPolicy — per-agent policy
# ---------------------------------------------------------------------------

class DTAdapterHebbPolicy(AgentPolicy):
    """Frozen DT body + Hebbian-plastic Adapter, wrapped as a mettagrid AgentPolicy.

    One instance per agent per episode. All agents in an episode share the
    same frozen `model` and the same `hebb_coeffs_np` (the evolved gene), but
    each agent has its own plastic w1/w2 tensors (re-randomized per episode).
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model,                    # DTModel, frozen, shared across agents
        hebb_coeffs_np: np.ndarray,  # (N_SYN, 5) float32
        K: int,
        target_rtg: float,
        weight_clamp: float = 3.0,
        adapter_dim: int = 256,
    ):
        super().__init__(policy_env_info)
        self.model       = model
        self.K           = K
        self.target_rtg  = target_rtg
        self.weight_clamp = weight_clamp
        self._action_names = policy_env_info.action_names
        self._obs_shape    = policy_env_info.observation_space.shape  # (num_tokens, 3)

        DIM = adapter_dim
        FC1_SYN = DIM * DIM   # 65536 for DIM=256
        FC2_SYN = DIM * DIM

        assert hebb_coeffs_np.shape == (FC1_SYN + FC2_SYN, 5), (
            f"hebb_coeffs shape mismatch: got {hebb_coeffs_np.shape}, "
            f"expected ({FC1_SYN + FC2_SYN}, 5) for adapter_dim={DIM}"
        )

        # Pre-reshape each coefficient matrix for vectorized outer-product update.
        # shape: (DIM_out, DIM_in) = (DIM, DIM) for both layers.
        def _split(arr):
            return arr.reshape(DIM, DIM).astype(np.float32)

        c1 = hebb_coeffs_np[:FC1_SYN]           # fc1 coefficients
        c2 = hebb_coeffs_np[FC1_SYN:]            # fc2 coefficients

        self._fc1_A  = _split(c1[:, 0])
        self._fc1_B  = _split(c1[:, 1])
        self._fc1_C  = _split(c1[:, 2])
        self._fc1_lr = _split(c1[:, 3])
        self._fc1_D  = _split(c1[:, 4])

        self._fc2_A  = _split(c2[:, 0])
        self._fc2_B  = _split(c2[:, 1])
        self._fc2_C  = _split(c2[:, 2])
        self._fc2_lr = _split(c2[:, 3])
        self._fc2_D  = _split(c2[:, 4])

        self._DIM = DIM

        # Per-episode plastic weights (U[-0.1, 0.1], Stage B §3.1)
        self._w1 = np.random.uniform(-0.1, 0.1, (DIM, DIM)).astype(np.float32)
        self._w2 = np.random.uniform(-0.1, 0.1, (DIM, DIM)).astype(np.float32)

        # Per-agent context buffers (filled with neutral/zero values)
        self._rtg_buf     = np.full((1, K), target_rtg, dtype=np.float32)
        self._s_buf       = np.zeros((1, K, DIM), dtype=np.float32)
        self._primary_buf = np.zeros((1, K), dtype=np.int64)
        self._vibe_buf    = np.full((1, K), NO_CHANGE_VIBE, dtype=np.int64)
        self._t = 0

    # ------------------------------------------------------------------
    def _obs_to_uint8(self, obs: AgentObservation) -> torch.Tensor:
        """Convert AgentObservation tokens -> (1, 500, 3) uint8 for DT encoder.

        Real tokens are written verbatim; remaining slots up to 500 are padded
        with loc=0xFF (DT EMPTY_LOC) so the masked mean pool ignores them.
        """
        arr = np.full((DT_OBS_TOKENS, 3), 0, dtype=np.uint8)
        arr[:, 0] = DT_EMPTY_LOC          # mark all as empty initially
        for idx, token in enumerate(obs.tokens):
            if idx >= DT_OBS_TOKENS:
                break
            raw = token.raw_token
            n = min(len(raw), 3)
            arr[idx, :n] = raw[:n]        # overwrites 0xFF with real loc
        return torch.as_tensor(arr, dtype=torch.uint8).unsqueeze(0)  # (1, 500, 3)

    # ------------------------------------------------------------------
    def step(self, obs: AgentObservation) -> Action:
        DIM = self._DIM
        K   = self.K
        t   = self._t
        slot = min(t, K - 1)

        with torch.no_grad():
            # --- 1. Encode new obs (only new obs; frozen encoder)
            obs_t = self._obs_to_uint8(obs)                  # (1, 500, 3)
            s_new = self.model.token_encoder(obs_t)          # (1, DIM)
            s_new_np = s_new[0].numpy()                      # (DIM,)

            # --- 2. Shift sliding context window when full
            if t >= K:
                self._rtg_buf[:, :-1]     = self._rtg_buf[:, 1:]
                self._s_buf[:, :-1]       = self._s_buf[:, 1:]
                self._primary_buf[:, :-1] = self._primary_buf[:, 1:]
                self._vibe_buf[:, :-1]    = self._vibe_buf[:, 1:]

            self._s_buf[0, slot]   = s_new_np
            self._rtg_buf[0, slot] = self.target_rtg  # RTG fixed (Stage B §6)

            # --- 3. Run frozen DT body to get state representation at slot
            rtg_t  = torch.as_tensor(self._rtg_buf,     dtype=torch.float32)
            s_t    = torch.as_tensor(self._s_buf,        dtype=torch.float32)
            prim_t = torch.as_tensor(self._primary_buf,  dtype=torch.long)
            vibe_t = torch.as_tensor(self._vibe_buf,     dtype=torch.long)

            r = self.model.rtg_embed(rtg_t)               # (1, K, DIM)
            a = self.model.action_embed(prim_t, vibe_t)   # (1, K, DIM)
            pos_ids = torch.arange(K)
            p = self.model.pos_emb(pos_ids)               # (K, DIM)

            seq = torch.stack([r + p, s_t + p, a + p], dim=2).reshape(1, 3 * K, DIM)
            h_body = self.model.body(seq, self.model.causal_mask)  # (1, 3K, DIM)

            # State token position in the interleaved (R, S, A) sequence
            s_out_np = h_body[0, slot * 3 + 1].numpy()   # (DIM,)

        # --- 4. Hebbian Adapter forward (numpy; outside no_grad scope)
        # fc1: pre-synaptic = s_out, post-synaptic = GELU(fc1(s_out))
        pre1 = s_out_np @ self._w1.T          # (DIM,)
        h1   = _gelu_np(pre1)                 # (DIM,)  post-synaptic for fc1

        # fc2: pre-synaptic = h1, post-synaptic = fc2(h1)
        pre2 = h1 @ self._w2.T               # (DIM,)  post-synaptic for fc2

        # Adapter residual + frozen LayerNorm
        with torch.no_grad():
            res     = torch.as_tensor(s_out_np + pre2, dtype=torch.float32).unsqueeze(0)
            adapted = self.model.adapter.norm(res)                # (1, DIM)
            plog    = self.model.primary_head(adapted)            # (1, 5)
            vlog    = self.model.vibe_head(adapted)               # (1, 8)

        primary_idx = int(plog[0].argmax().item())
        vibe_idx    = int(vlog[0].argmax().item())

        # --- 5. Hebbian weight updates (vectorized outer products)
        # fc1: Δw1[j,i] = lr[j,i] * (A*s_out[i]*h1[j] + B*s_out[i] + C*h1[j] + D)
        delta1 = self._fc1_lr * (
            self._fc1_A * s_out_np[np.newaxis, :] * h1[:, np.newaxis]
            + self._fc1_B * s_out_np[np.newaxis, :]
            + self._fc1_C * h1[:, np.newaxis]
            + self._fc1_D
        )
        self._w1 += delta1
        np.clip(self._w1, -self.weight_clamp, self.weight_clamp, out=self._w1)

        # fc2: Δw2[j,i] = lr[j,i] * (A*h1[i]*pre2[j] + B*h1[i] + C*pre2[j] + D)
        delta2 = self._fc2_lr * (
            self._fc2_A * h1[np.newaxis, :] * pre2[:, np.newaxis]
            + self._fc2_B * h1[np.newaxis, :]
            + self._fc2_C * pre2[:, np.newaxis]
            + self._fc2_D
        )
        self._w2 += delta2
        np.clip(self._w2, -self.weight_clamp, self.weight_clamp, out=self._w2)

        # --- 6. Update context buffers with chosen action
        self._primary_buf[0, slot] = primary_idx
        self._vibe_buf[0, slot]    = vibe_idx
        self._t += 1

        action12 = factored_to_action12(primary_idx, vibe_idx)
        return Action(name=self._action_names[action12])


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_dt_adapter_hebb_cogames(
    hebb_coeffs_path: str,
    dt_checkpoint_path: str,
    mission: str = 'tutorial.scout',
    render: bool = False,
    eval_episodes: int = 10,
    seed_base: int = 12345,
    fitness_stat: str = 'composite_v7_zstat',
    num_agents: int = None,
    target_rtg: float = 1.0,
    weight_clamp: float = 3.0,
    adapter_dim: int = 256,
    fitness_v7_weights: dict = None,
    components_sidecar: str = '',
) -> None:
    """Run DT+Hebbian-Adapter policy on a CoGames mission for eval_episodes episodes.

    Prints per-episode score and mean score in the same format as
    evaluate_hebb_cogames.py so that task_cogames_dt_adapter_hebb.py
    can parse the output with the same regex patterns.
    """
    v7_weights = fitness_v7_weights or {}
    per_episode_components_v7 = []

    # Load frozen DT checkpoint once
    ckpt = torch.load(dt_checkpoint_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']
    model = build_model(
        K=cfg['K'], state_dim=cfg['state_dim'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        ffn_dim=cfg['ffn_dim'], encoder=cfg['encoder'],
    )
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    K = cfg['K']
    print(
        f'DT+Adapter eval: checkpoint={dt_checkpoint_path}  '
        f'K={K}  state_dim={cfg["state_dim"]}  adapter_dim={adapter_dim}  '
        f'mission={mission}  episodes={eval_episodes}  '
        f'target_rtg={target_rtg}  weight_clamp={weight_clamp}  '
        f'fitness_stat={fitness_stat}',
        flush=True,
    )

    # Load Hebbian coefficients
    hebb_coeffs_np = torch.load(hebb_coeffs_path, map_location='cpu').numpy()
    if not isinstance(hebb_coeffs_np, np.ndarray):
        hebb_coeffs_np = np.asarray(hebb_coeffs_np, dtype=np.float32)

    N_SYN_EXPECTED = 2 * adapter_dim * adapter_dim
    if hebb_coeffs_np.shape != (N_SYN_EXPECTED, 5):
        raise ValueError(
            f'hebb_coeffs shape {hebb_coeffs_np.shape} does not match '
            f'expected ({N_SYN_EXPECTED}, 5) for adapter_dim={adapter_dim}'
        )
    print(
        f'Hebb coeffs loaded: shape={hebb_coeffs_np.shape}  '
        f'A∈[{hebb_coeffs_np[:,0].min():.3f},{hebb_coeffs_np[:,0].max():.3f}]  '
        f'B∈[{hebb_coeffs_np[:,1].min():.3f},{hebb_coeffs_np[:,1].max():.3f}]  '
        f'C∈[{hebb_coeffs_np[:,2].min():.3f},{hebb_coeffs_np[:,2].max():.3f}]  '
        f'lr∈[{hebb_coeffs_np[:,3].min():.5f},{hebb_coeffs_np[:,3].max():.5f}]  '
        f'D∈[{hebb_coeffs_np[:,4].min():.3f},{hebb_coeffs_np[:,4].max():.3f}]',
        flush=True,
    )

    # Build cogames env config (tutorial.scout gives composite_v7 mechanics)
    from cogames.cli.mission import get_mission as _get_mission
    _, env_cfg, _ = _get_mission(mission)

    if num_agents is not None:
        env_cfg.game.num_agents = num_agents
        try:
            env_cfg.game.map_builder.instance.spawn_count = num_agents
        except AttributeError:
            pass

    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    num_agents_resolved = env_cfg.game.num_agents
    n_action_names = len(policy_env_info.action_names)

    print(
        f'Env: num_agents={num_agents_resolved}  '
        f'obs_shape={policy_env_info.observation_space.shape}  '
        f'n_actions={n_action_names}',
        flush=True,
    )

    episode_scores = []

    with torch.no_grad():
        for ep_idx in range(eval_episodes):
            ep_seed = int(seed_base) + ep_idx
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)

            # Fresh policy per agent per episode (plastic weights re-randomized)
            agent_policies = [
                DTAdapterHebbPolicy(
                    policy_env_info, model, hebb_coeffs_np,
                    K=K, target_rtg=target_rtg,
                    weight_clamp=weight_clamp, adapter_dim=adapter_dim,
                )
                for _ in range(num_agents_resolved)
            ]

            # Set up composite_v7 accumulators
            accumulators = None
            if fitness_stat == 'composite_v7_zstat':
                v7_max_steps = max(env_cfg.game.max_steps or 1000, 100)
                accumulators = [
                    composite_v7.PerEpisodeAccumulatorV7(v7_max_steps)
                    for _ in range(num_agents_resolved)
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
                env_cfg, agent_policies, render_mode=render_mode, seed=ep_seed
            )

            if accumulators is not None:
                _spawns = composite_v7.get_agent_spawn_positions(
                    rollout._sim, num_agents_resolved
                )
                _extents = composite_v7.get_map_extents(rollout._sim)
                for _i, _acc in enumerate(accumulators):
                    _acc.set_spawn(_spawns[_i], _extents)

            rollout.run_until_done()

            if fitness_stat == 'composite_v7_zstat':
                sim = rollout._sim
                agent_stats_list = sim.episode_stats['agent']
                max_steps = max(env_cfg.game.max_steps or 1000, 100)
                episode_rewards = list(sim.episode_rewards)
                ep_reward_team = float(sum(episode_rewards))

                behaviour_capped, _bh_raw, _ = composite_v7.compute_v3_behaviour(
                    agent_stats_list, max_steps
                )
                per_agent_results = []
                for _i, _acc in enumerate(accumulators or []):
                    _stats_i = agent_stats_list[_i] if _i < len(agent_stats_list) else {}
                    _ep_r    = float(episode_rewards[_i]) if _i < len(episode_rewards) else 0.0
                    per_agent_results.append(
                        _acc.episode_score(
                            ep_reward_total=_ep_r,
                            exploration_capped=behaviour_capped,
                            agent_stats=_stats_i,
                            weights=v7_weights,
                        )
                    )
                _reduction = (v7_weights or {}).get(
                    'reduction', composite_v7.DEFAULT_WEIGHTS_V7['reduction']
                )
                _agg7 = composite_v7.aggregate_agent_results_v7(
                    per_agent_results, reduction=_reduction
                )
                print(composite_v7.format_diagnostic_lines_v7(_agg7, ep_reward_team),
                      flush=True)
                ep_score = float(_agg7['score'])
                per_episode_components_v7.append({
                    'aggregate':   _agg7,
                    'per_agent':   per_agent_results,
                    'episode_idx': ep_idx,
                    'seed':        ep_seed,
                })

            elif fitness_stat == 'episode_reward':
                ep_score = float(sum(rollout._sim.episode_rewards))

            else:
                raise ValueError(f'Unsupported fitness_stat for DT+Adapter eval: {fitness_stat}')

            episode_scores.append(ep_score)
            print(f' Episode {ep_idx} cumulative rewards  {ep_score:.4f}', flush=True)

    mean_score = float(np.mean(episode_scores))
    print(f'\n Episode cumulative rewards  {mean_score}', flush=True)

    if fitness_stat == 'composite_v7_zstat' and components_sidecar:
        ep_aggs   = [ec['aggregate'] for ec in per_episode_components_v7]
        over_eps  = composite_v7.aggregate_over_episodes_v7(ep_aggs)
        try:
            composite_v7.write_component_sidecar(
                components_sidecar, per_episode_components_v7, over_eps
            )
            print(f'  v7: components sidecar written: {components_sidecar}', flush=True)
        except Exception as _sc_exc:
            print(f'  v7: WARN failed to write sidecar ({_sc_exc})', flush=True)


# ---------------------------------------------------------------------------
# CLI entry point (called as subprocess by task_cogames_dt_adapter_hebb.py)
# ---------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(
        description='Evaluate DT + Hebbian Adapter policy on CoGames.'
    )
    parser.add_argument('--hebb_coeffs_path',    type=str, required=True)
    parser.add_argument('--dt_checkpoint_path',  type=str, required=True)
    parser.add_argument('--mission',             type=str, default='tutorial.scout')
    parser.add_argument('--render',              type=str, default='False')
    parser.add_argument('--eval_episodes',       type=int, default=10)
    parser.add_argument('--seed_base',           type=int, default=12345)
    parser.add_argument('--fitness_stat',        type=str, default='composite_v7_zstat')
    parser.add_argument('--num_agents',          type=int, default=None)
    parser.add_argument('--target_rtg',          type=float, default=1.0)
    parser.add_argument('--weight_clamp',        type=float, default=3.0)
    parser.add_argument('--adapter_dim',         type=int,   default=256)
    parser.add_argument('--fitness_v7_weights',  type=str,   default='')
    parser.add_argument('--components_sidecar',  type=str,   default='')

    args = parser.parse_args(argv[1:])

    evaluate_dt_adapter_hebb_cogames(
        hebb_coeffs_path=args.hebb_coeffs_path,
        dt_checkpoint_path=args.dt_checkpoint_path,
        mission=args.mission,
        render=str(args.render).lower() == 'true',
        eval_episodes=args.eval_episodes,
        seed_base=args.seed_base,
        fitness_stat=args.fitness_stat,
        num_agents=args.num_agents,
        target_rtg=args.target_rtg,
        weight_clamp=args.weight_clamp,
        adapter_dim=args.adapter_dim,
        fitness_v7_weights=composite_v7.parse_weights_str(args.fitness_v7_weights),
        components_sidecar=args.components_sidecar,
    )


if __name__ == '__main__':
    main(sys.argv)

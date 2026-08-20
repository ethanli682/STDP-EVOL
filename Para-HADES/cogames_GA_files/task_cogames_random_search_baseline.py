"""Random-search baseline driver for cogames composite_v5 fitness.

Goal: validate (or refute) that CMA-ES outperforms uniform random sampling
under the same fitness function and matched wallclock. If CMA-ES doesn't beat
this baseline on aggregate score OR on any decomposed component, the
optimizer is ranking on noise — actionable diagnostic information.

Self-contained:
  - No dependency on slurmHPCHelper / GA_utils_misc_func (the outer harness).
  - Reuses the existing evaluate_static_cogames.py subprocess + composite_v5
    sidecar machinery — zero changes to that path.

Per epoch:
  1. For each candidate, sample W1, W2, W3 ~ Uniform[-3, 3].
  2. Save weights blob to disk.
  3. Run evaluate_static_cogames.py with --components_sidecar set.
  4. Read sidecar JSON, append result row to results_epoch_<n>.jsonl.

Seeds: same CRN scheme as task_cogames_static.py — within an epoch, every
candidate sees the same seed_base; epoch increments rotate maps.

Usage:
    python task_cogames_random_search_baseline.py \\
        --path /tmp/cogames_v5_random \\
        --population_size 10 \\
        --max_epochs 5 \\
        --paramFile task_cogames_random_search_baseline.yaml
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='CMA-ES-free random-search baseline (composite_v5).'
    )
    parser.add_argument('--path', type=str, required=True,
                        help='Output directory for jsonl results and ephemeral '
                             'weights/sidecar files.')
    parser.add_argument('--paramFile', type=str,
                        default='task_cogames_random_search_baseline.yaml',
                        help='YAML config (StaticEval block; same schema as '
                             'task_cogames_static.yaml).')
    parser.add_argument('--population_size', type=int, default=10)
    parser.add_argument('--max_epochs',      type=int, default=5)
    parser.add_argument('--gene_seed_base',  type=int, default=12345,
                        help='Base seed for sampling random gene weights '
                             '(distinct from env seed_base).')
    parser.add_argument('--w_min', type=float, default=-3.0)
    parser.add_argument('--w_max', type=float, default=3.0)
    parser.add_argument('--evaluator', type=str, default='evaluate_static_cogames.py')
    args = parser.parse_args(argv)

    # --- YAML config ---------------------------------------------------------
    cfg = {}
    if os.path.exists(args.paramFile):
        with open(args.paramFile, 'rt') as f:
            cfg = yaml.safe_load(f) or {}
    eval_cfg = ((cfg.get('main', {}) or {}).get('StaticEval', {}) or {})

    mission         = str(eval_cfg.get('mission', 'arena'))
    num_agents      = int(eval_cfg.get('num_agents', 4))
    fitness_stat    = str(eval_cfg.get('fitness_stat', 'composite_v5_role_event'))
    eval_episodes   = int(eval_cfg.get('eval_episodes', 40))
    seed_base_yaml  = int(eval_cfg.get('seed_base', 12345))
    hidden          = tuple(int(x) for x in eval_cfg.get('hidden', [64, 32]))
    obs_mode        = str(eval_cfg.get('obs_mode', 'legacy_flat'))
    frame_stack     = int(eval_cfg.get('frame_stack', 5))
    grid_h          = int(eval_cfg.get('grid_h', 11))
    grid_w          = int(eval_cfg.get('grid_w', 11))
    num_channels    = int(eval_cfg.get('num_channels', 3))
    global_state_dim = int(eval_cfg.get('global_state_dim', 70))

    fv5 = eval_cfg.get('fitness_v5', None)
    fitness_v5_weights_str = ''
    if isinstance(fv5, dict):
        parts = []
        for k, v in fv5.items():
            if v is None:
                continue
            if k == 'reduction':
                parts.append(f'{k}={str(v)}')
            else:
                parts.append(f'{k}={float(v)}')
        fitness_v5_weights_str = ','.join(parts)

    h1, h2 = hidden
    if obs_mode == 'stacked_grid':
        input_dim = (grid_h * grid_w * num_channels + global_state_dim) * frame_stack
    else:
        input_dim = 900
    action_dim = 5
    expected = {
        'W1': h1 * input_dim,
        'W2': h2 * h1,
        'W3': action_dim * h2,
    }

    os.makedirs(args.path, exist_ok=True)
    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.evaluator)

    print(
        f'[random-search] mission={mission} num_agents={num_agents} '
        f'fitness_stat={fitness_stat} eval_episodes={eval_episodes} '
        f'population_size={args.population_size} max_epochs={args.max_epochs} '
        f'input_dim={input_dim} hidden={hidden}',
        flush=True,
    )

    for epoch in range(args.max_epochs):
        # CRN: same env seed_base across all candidates in this epoch; rotates per epoch.
        env_seed_base = seed_base_yaml + epoch * 999983
        epoch_jsonl = os.path.join(args.path, f'results_epoch_{epoch}.jsonl')
        with open(epoch_jsonl, 'wt') as jl:
            for cand_idx in range(args.population_size):
                gene_rng = np.random.default_rng(args.gene_seed_base + epoch * 1009 + cand_idx)
                W1 = gene_rng.uniform(args.w_min, args.w_max,
                                      size=expected['W1']).astype(np.float32)
                W2 = gene_rng.uniform(args.w_min, args.w_max,
                                      size=expected['W2']).astype(np.float32)
                W3 = gene_rng.uniform(args.w_min, args.w_max,
                                      size=expected['W3']).astype(np.float32)

                unique_suffix = f'{os.getpid()}_{time.time_ns()}'
                weights_path = os.path.join(
                    args.path, f'rs_weights_{epoch}_{cand_idx}_{unique_suffix}.dat'
                )
                sidecar_path = os.path.join(
                    args.path, f'rs_components_{epoch}_{cand_idx}_{unique_suffix}.json'
                )
                torch.save({'W1': W1, 'W2': W2, 'W3': W3}, weights_path)

                cmd = [
                    sys.executable, eval_script,
                    '--mission', mission,
                    '--weights_path', weights_path,
                    '--hidden_1', str(h1),
                    '--hidden_2', str(h2),
                    '--render', 'False',
                    '--eval_episodes', str(eval_episodes),
                    '--seed_base', str(env_seed_base),
                    '--fitness_stat', fitness_stat,
                    '--num_agents', str(num_agents),
                    '--obs_mode', obs_mode,
                    '--frame_stack', str(frame_stack),
                ]
                if fitness_v5_weights_str:
                    cmd += ['--fitness_v5_weights', fitness_v5_weights_str]
                if fitness_stat == 'composite_v5_role_event':
                    cmd += ['--components_sidecar', sidecar_path]

                t0 = time.time()
                row = {
                    'epoch':          epoch,
                    'cand_idx':       cand_idx,
                    'env_seed_base':  env_seed_base,
                    'gene_seed':      args.gene_seed_base + epoch * 1009 + cand_idx,
                    'wallclock_sec':  None,
                    'score':          None,
                    'failure':        None,
                }

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=900)
                    row['wallclock_sec'] = float(time.time() - t0)
                    if result.returncode != 0:
                        row['failure'] = f'returncode-{result.returncode}'
                    else:
                        # Score from sidecar if v5; else parse stdout regex.
                        score = None
                        if (fitness_stat == 'composite_v5_role_event'
                                and os.path.exists(sidecar_path)):
                            try:
                                with open(sidecar_path, 'rt') as scf:
                                    sidecar = json.load(scf)
                                agg = sidecar.get('aggregate', {})
                                row['score']   = float(agg.get('score', 0.0))
                                row['score_F'] = float(agg.get('score_F', 0.0))
                                row['score_B'] = float(agg.get('score_B', 0.0))
                                row['ach_first_ore']        = float(agg.get('ach_first_ore', 0.0))
                                row['ach_first_gear']       = float(agg.get('ach_first_gear', 0.0))
                                row['ach_first_heart']      = float(agg.get('ach_first_heart', 0.0))
                                row['ach_first_heart_spent'] = float(agg.get('ach_first_heart_spent', 0.0))
                                row['heart_gain_count']  = float(agg.get('heart_gain_count', 0.0))
                                row['heart_spend_count'] = float(agg.get('heart_spend_count', 0.0))
                                row['role_event_total']  = float(agg.get('role_event_total', 0.0))
                                row['exploration_capped'] = float(agg.get('exploration_capped', 0.0))
                                row['ep_reward_clipped'] = float(agg.get('ep_reward_clipped', 0.0))
                                score = row['score']
                            except Exception as sc_exc:
                                row['failure'] = f'sidecar-load-{sc_exc}'
                        if score is None:
                            # Fallback: parse stdout for "Episode cumulative rewards <num>"
                            m = re.search(
                                r'[Ee]pisode\s+cumulative\s+rewards?\s+(-?[\d.]+)',
                                result.stdout)
                            if m:
                                try:
                                    score = float(m.group(1))
                                except ValueError:
                                    pass
                        if score is None:
                            row['failure'] = row.get('failure') or 'score-parse-failed'
                        else:
                            if row.get('score') is None:
                                row['score'] = float(score)
                except subprocess.TimeoutExpired:
                    row['wallclock_sec'] = float(time.time() - t0)
                    row['failure'] = 'timeout'
                except Exception as run_exc:
                    row['wallclock_sec'] = float(time.time() - t0)
                    row['failure'] = f'exception-{run_exc}'
                finally:
                    for p in (weights_path, sidecar_path):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

                jl.write(json.dumps(row) + '\n')
                jl.flush()
                print(
                    f'[random-search] epoch={epoch} cand={cand_idx} '
                    f"score={row.get('score')} F={row.get('score_F')} "
                    f"B={row.get('score_B')} time={row.get('wallclock_sec'):.1f}s "
                    f"failure={row.get('failure')}",
                    flush=True,
                )

        # Per-epoch summary
        scores = []
        with open(epoch_jsonl, 'rt') as jl:
            for ln in jl:
                try:
                    r = json.loads(ln)
                    if r.get('score') is not None and math.isfinite(float(r['score'])):
                        scores.append(float(r['score']))
                except Exception:
                    pass
        if scores:
            print(
                f'[random-search] epoch={epoch} summary: '
                f'n={len(scores)} best={max(scores):.4f} '
                f'mean={sum(scores)/len(scores):.4f} min={min(scores):.4f}',
                flush=True,
            )

    print('[random-search] done.', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])

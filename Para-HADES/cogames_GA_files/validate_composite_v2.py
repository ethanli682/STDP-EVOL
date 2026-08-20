"""Offline validator for composite_exploration_v2.

Addresses the five pushback points raised against the april17 rewrite:

  1. ordering   — rank {do-nothing, random-small, random-big, half, strong}
                  under v1 and v2, and confirm v2 is monotone where v1 was
                  not. A fitness function that scores unordered checkpoints
                  consistently is necessary-but-not-sufficient; ordering is
                  the real property we care about.

  2. baseline   — measure the v2 random-weights baseline (replacement for
                  the "~0.46" number that the old yaml comment carried for
                  v1). Without this number nobody can tell whether a mid-run
                  score of 0.4 means "barely moving" or "slightly better
                  than random".

  3. smoke      — exercise each of the three evaluators (static, hebb,
                  static_GRU) under v2, on a cheap synthetic checkpoint,
                  to verify the v2 code path doesn't blow up anywhere.

  4. rollout    — score two checkpoints (e.g. iter-65 and iter-160) under
                  both v1 and v2. Needed to answer "was the 0.53 → 0.09
                  degrade real behaviour regression, or a v1 artifact?".

  5. cutover    — print a short banner reminding the operator that v1 and
                  v2 are NOT on the same numeric scale, so any fitness
                  trajectory spanning the cutover date is meaningless.

This script shells out to the existing evaluator binaries (same contract the
task runners use). It does not import the evaluators directly, so a broken
evaluator CLI surface would be caught here too.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_EVAL   = os.path.join(HERE, 'evaluate_static_cogames.py')
HEBB_EVAL     = os.path.join(HERE, 'evaluate_hebb_cogames.py')
GRU_EVAL      = os.path.join(HERE, 'evaluate_static_GRU_cogames.py')

# Architecture constants for the static MLP (keep in sync with task yamls).
STATIC_H1, STATIC_H2 = 64, 32
STATIC_INPUT_DIM_LEGACY = 900          # 300 tokens * 3 for tutorial.scout
STATIC_NUM_ACTIONS_FALLBACK = 5        # matches the yaml default; env-derived in practice

# Seed-base for every run in this script. We fix this across v1 vs v2 so
# the ordering test is on matched maps (same common-random-seeds idea
# the GA now uses between candidates in one generation).
DEFAULT_SEED = 12345
DEFAULT_EPISODES = 7


# ---------------------------------------------------------------------------
# Score parsing — reuse the exact regex ladder used by task_cogames_static.py
# so a score this script reports is the score the GA would have seen.
# ---------------------------------------------------------------------------

_SCORE_PATTERNS = [
    r'[Ee]pisode\s+cumulative\s+rewards?\s+(-?[\d.]+)',
    r'[Ff]itness[:\s]+(-?[\d.]+)',
    r'[Ss]core[:\s]+(-?[\d.]+)',
    r'[Rr]eward[:\s]+(-?[\d.]+)',
    r'^(-?[\d.]+)\s*$',
]


def parse_score(stdout: str) -> Optional[float]:
    for line in reversed(stdout.strip().splitlines()):
        for pat in _SCORE_PATTERNS:
            m = re.search(pat, line.strip())
            if m:
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                if math.isfinite(v):
                    return v
    return None


# ---------------------------------------------------------------------------
# Synthetic checkpoints — "do-nothing" and "random" endpoints of the scale.
# ---------------------------------------------------------------------------

def _static_shapes(input_dim: int, num_actions: int):
    return {
        'W1': (STATIC_H1, input_dim),
        'W2': (STATIC_H2, STATIC_H1),
        'W3': (num_actions, STATIC_H2),
    }


def make_static_checkpoint(mode: str, input_dim: int, num_actions: int,
                           path: str, seed: int = 0) -> str:
    """Write a synthetic static-MLP checkpoint.

    mode:
      zero     — all zeros. argmax(zero_logits) = 0. Whether that is a true
                 "do-nothing" depends on env action ordering; we document
                 this caveat rather than paper over it.
      small    — uniform[-0.1, 0.1], matches the task-side 'uni' init used
                 as the historical random baseline (~0.46 under v1).
      big      — uniform[-1, 1], more active policy.
    """
    rng = np.random.default_rng(seed)
    shapes = _static_shapes(input_dim, num_actions)
    blob = {}
    for name, shape in shapes.items():
        if mode == 'zero':
            arr = np.zeros(shape, dtype=np.float32)
        elif mode == 'small':
            arr = rng.uniform(-0.1, 0.1, size=shape).astype(np.float32)
        elif mode == 'big':
            arr = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        else:
            raise ValueError(f'unknown mode {mode!r}')
        blob[name] = torch.from_numpy(arr)
    torch.save(blob, path)
    return path


def make_gru_checkpoint(path: str, input_dim: int, proj: int,
                        hidden: int, num_actions: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    blob = {
        'W_in':  torch.from_numpy(rng.uniform(-0.1, 0.1, size=(proj, input_dim)).astype(np.float32)),
        'W_ih':  torch.from_numpy(rng.uniform(-0.1, 0.1, size=(3 * hidden, proj)).astype(np.float32)),
        'W_hh':  torch.from_numpy(rng.uniform(-0.1, 0.1, size=(3 * hidden, hidden)).astype(np.float32)),
        'W_out': torch.from_numpy(rng.uniform(-0.1, 0.1, size=(num_actions, hidden)).astype(np.float32)),
    }
    torch.save(blob, path)
    return path


def make_hebb_coeffs(path: str, input_dim: int, num_actions: int,
                     seed: int = 0, rule_width: int = 5) -> str:
    """Hebb coeffs shape: (n_synapses, rule_width). Small random values."""
    rng = np.random.default_rng(seed)
    n_syn = STATIC_H1 * input_dim + STATIC_H2 * STATIC_H1 + num_actions * STATIC_H2
    arr = rng.uniform(-0.1, 0.1, size=(n_syn, rule_width)).astype(np.float32)
    torch.save(arr, path)
    return path


# ---------------------------------------------------------------------------
# Evaluator invocation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    score: Optional[float]
    returncode: int
    stdout_tail: str
    v2_lines: list          # per-episode 'v2: ...' diagnostic lines, if any


def extract_v2_lines(stdout: str) -> list:
    """Pull every per-episode v2 diagnostic line out of evaluator stdout.

    Each evaluator emits exactly one line per episode in the shape:
      '  v2: unique=... ep_reward=... behaviour=... -> score=...'
    These lines are what let us audit whether ep_reward dominates behaviour.
    """
    return [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith('v2:')]


def run_static_eval(weights_path: str, fitness_stat: str, mission: str,
                    episodes: int, seed_base: int, num_agents: int = 1,
                    timeout: int = 600) -> EvalResult:
    cmd = [
        sys.executable, STATIC_EVAL,
        '--mission', mission,
        '--weights_path', weights_path,
        '--hidden_1', str(STATIC_H1),
        '--hidden_2', str(STATIC_H2),
        '--render', 'False',
        '--eval_episodes', str(episodes),
        '--seed_base', str(seed_base),
        '--fitness_stat', fitness_stat,
        '--num_agents', str(num_agents),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = '\n'.join(r.stdout.strip().splitlines()[-20:])
    return EvalResult(parse_score(r.stdout), r.returncode, tail, extract_v2_lines(r.stdout))


def run_hebb_eval(coeffs_path: str, fitness_stat: str, mission: str,
                  episodes: int, seed_base: int, num_agents: int = 1,
                  timeout: int = 600) -> EvalResult:
    cmd = [
        sys.executable, HEBB_EVAL,
        '--mission', mission,
        '--hebb_rule', 'ABCD_lr',
        '--hebb_coeffs_path', coeffs_path,
        '--render', 'False',
        '--eval_episodes', str(episodes),
        '--seed_base', str(seed_base),
        '--fitness_stat', fitness_stat,
        '--num_agents', str(num_agents),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = '\n'.join(r.stdout.strip().splitlines()[-20:])
    return EvalResult(parse_score(r.stdout), r.returncode, tail, extract_v2_lines(r.stdout))


def run_gru_eval(weights_path: str, fitness_stat: str, mission: str,
                 episodes: int, seed_base: int, num_agents: int = 1,
                 timeout: int = 600) -> EvalResult:
    cmd = [
        sys.executable, GRU_EVAL,
        '--mission', mission,
        '--weights_path', weights_path,
        '--render', 'False',
        '--eval_episodes', str(episodes),
        '--seed_base', str(seed_base),
        '--fitness_stat', fitness_stat,
        '--num_agents', str(num_agents),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = '\n'.join(r.stdout.strip().splitlines()[-20:])
    return EvalResult(parse_score(r.stdout), r.returncode, tail, extract_v2_lines(r.stdout))


# ---------------------------------------------------------------------------
# Cutover banner — printed once at the top of every mode.
# ---------------------------------------------------------------------------

CUTOVER_BANNER = """\
================================================================================
 CUTOVER NOTICE — composite_exploration (v1)  vs  composite_exploration_v2

 The two functions are NOT on the same numeric scale. A v1 score of 0.5 and a
 v2 score of 0.5 are unrelated quantities. Do not plot, compare, or select
 checkpoints across the v1→v2 cutover without first re-scoring both sides
 under the SAME fitness_stat.

 If a run started under v1 and continued under v2, its fitness trajectory is
 not interpretable. Prefer to restart, or re-score every surviving candidate
 under v2 before comparing.
================================================================================
"""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _tmp(prefix: str, suffix: str = '.pt') -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


def cmd_ordering(args) -> int:
    """Score a ladder of checkpoints under v1 and v2, print the ordering."""
    print(CUTOVER_BANNER)
    print('Mode: ordering test (static MLP evaluator)')
    print(f'  mission={args.mission}  episodes={args.episodes}  seed_base={args.seed_base}')
    print()

    input_dim = args.input_dim
    num_actions = args.num_actions
    ladder: list[tuple[str, str]] = []        # (label, weights_path)

    zero_path  = make_static_checkpoint('zero',  input_dim, num_actions, _tmp('val_zero_'),  seed=1)
    small_path = make_static_checkpoint('small', input_dim, num_actions, _tmp('val_small_'), seed=2)
    big_path   = make_static_checkpoint('big',   input_dim, num_actions, _tmp('val_big_'),   seed=3)

    ladder.append(('do-nothing (zero W)',      zero_path))
    ladder.append(('random small (uni[-0.1])', small_path))
    ladder.append(('random big   (uni[-1])',   big_path))
    if args.half_trained:
        ladder.append(('half-trained',  args.half_trained))
    if args.strong:
        ladder.append(('strong',        args.strong))

    rows = []
    for label, path in ladder:
        v1 = run_static_eval(path, 'composite_exploration',    args.mission,
                             args.episodes, args.seed_base, args.num_agents, args.timeout)
        v2 = run_static_eval(path, 'composite_exploration_v2', args.mission,
                             args.episodes, args.seed_base, args.num_agents, args.timeout)
        rows.append((label, v1, v2))

    # Clean up synthetic checkpoints.
    for p in (zero_path, small_path, big_path):
        try: os.remove(p)
        except OSError: pass

    # Print table.
    print()
    print(f'{"checkpoint":<32} {"v1 score":>12} {"v2 score":>12}   ok?')
    print('-' * 72)
    for label, v1, v2 in rows:
        v1s = 'FAIL' if v1.score is None else f'{v1.score:>12.4f}'
        v2s = 'FAIL' if v2.score is None else f'{v2.score:>12.4f}'
        ok  = 'ok' if (v1.returncode == 0 and v2.returncode == 0) else 'ERR'
        print(f'{label:<32} {v1s:>12} {v2s:>12}   {ok}')
    print()

    # Monotonicity checks — verdict, not just data.
    def monotone(seq):
        seq = [s for s in seq if s is not None]
        return all(seq[i] <= seq[i+1] for i in range(len(seq)-1))

    v1_seq = [r[1].score for r in rows]
    v2_seq = [r[2].score for r in rows]
    print(f'v1 monotone non-decreasing across ladder?  {monotone(v1_seq)}')
    print(f'v2 monotone non-decreasing across ladder?  {monotone(v2_seq)}')
    print()
    print('Expected: v2 monotone non-decreasing, with do-nothing < random < (half) < strong.')
    print('If v2 is not monotone on the synthetic rungs alone, the fitness design is wrong.')
    print('If v2 is monotone on synthetics but not including user checkpoints, look at those rollouts.')
    return 0


def _make_random_ckpt_for(evaluator: str, args, seed: int, path: str) -> str:
    """Dispatch: build a random checkpoint in the format the named evaluator expects."""
    if evaluator == 'static':
        return make_static_checkpoint('small', args.input_dim, args.num_actions, path, seed=seed)
    if evaluator == 'GRU':
        return make_gru_checkpoint(path, args.input_dim,
                                   proj=64, hidden=32,
                                   num_actions=args.num_actions, seed=seed)
    if evaluator == 'hebb':
        return make_hebb_coeffs(path, args.input_dim, args.num_actions, seed=seed)
    raise ValueError(f'unknown evaluator {evaluator!r}')


def _run_eval_for(evaluator: str, ckpt: str, fitness_stat: str, args, seed_base: int) -> EvalResult:
    """Dispatch: invoke the right evaluator subprocess with consistent kwargs."""
    if evaluator == 'static':
        return run_static_eval(ckpt, fitness_stat, args.mission,
                               args.episodes, seed_base, args.num_agents, args.timeout)
    if evaluator == 'GRU':
        return run_gru_eval(ckpt, fitness_stat, args.mission,
                            args.episodes, seed_base, args.num_agents, args.timeout)
    if evaluator == 'hebb':
        return run_hebb_eval(ckpt, fitness_stat, args.mission,
                             args.episodes, seed_base, args.num_agents, args.timeout)
    raise ValueError(f'unknown evaluator {evaluator!r}')


# Per-evaluator yaml targets, so the pasted baseline line can be tied to the
# files it applies to (hebb and GRU baselines are NOT transferable to static).
_EVAL_YAML_MAP = {
    'static': ['task_cogames_static.yaml', 'task_cogames_static_LoRA.yaml'],
    'hebb':   ['task_cogames_hebb.yaml',   'task_cogames_hebb_LoRA.yaml'],
    'GRU':    ['task_cogames_static_GRU.yaml', 'task_cogames_static_GRU_LoRA.yaml'],
}


def cmd_baseline(args) -> int:
    """Measure the v2 random-weights baseline for ONE evaluator, N trials.

    Each evaluator has its own architecture and its own random baseline. A
    static-MLP baseline is NOT a valid reference for a Hebb or GRU run.
    """
    print(CUTOVER_BANNER)
    print(f'Mode: random-baseline measurement ({args.evaluator} evaluator, {args.trials} trials)')
    print(f'  mission={args.mission}  episodes={args.episodes}  seed_base={args.seed_base}')
    print()

    v1_scores, v2_scores = [], []
    for t in range(args.trials):
        ckpt = _make_random_ckpt_for(args.evaluator, args, seed=1000 + t,
                                     path=_tmp(f'val_bl_{args.evaluator}_{t}_'))
        try:
            v1 = _run_eval_for(args.evaluator, ckpt, 'composite_exploration',
                               args, args.seed_base + t)
            v2 = _run_eval_for(args.evaluator, ckpt, 'composite_exploration_v2',
                               args, args.seed_base + t)
        finally:
            try: os.remove(ckpt)
            except OSError: pass
        if v1.score is not None: v1_scores.append(v1.score)
        if v2.score is not None: v2_scores.append(v2.score)
        print(f'  trial {t:>3}  v1={v1.score}  v2={v2.score}')

    def _stats(xs):
        if not xs: return (float('nan'), float('nan'), 0)
        return (float(np.mean(xs)), float(np.std(xs)), len(xs))

    v1m, v1s, v1n = _stats(v1_scores)
    v2m, v2s, v2n = _stats(v2_scores)
    print()
    print(f'v1 random baseline ({args.evaluator}):  mean={v1m:.4f}  std={v1s:.4f}  n={v1n}')
    print(f'v2 random baseline ({args.evaluator}):  mean={v2m:.4f}  std={v2s:.4f}  n={v2n}')

    targets = _EVAL_YAML_MAP.get(args.evaluator, [])
    print()
    print(f'Paste this into the {args.evaluator} task yamls '
          f'({", ".join(targets)}) above the fitness_stat: line:')
    print(f'  # random baseline: v2 {args.evaluator} mean={v2m:.3f} \u00b1 {v2s:.3f} '
          f'(n={v2n}, mission={args.mission}, ep={args.episodes})')

    if args.evaluator == 'GRU' and v2s < 1e-4:
        print()
        print('  note: GRU random baseline has ~zero variance because small random '
              'GRU weights produce a frozen policy (no moves). This is expected '
              'and means GRU runs need non-trivial initialisation to leave the '
              'do-nothing basin without the GA, not a fitness-function problem.')
    return 0


_V2_EP_REWARD_RE   = re.compile(r'ep_reward=(-?[\d.]+)')
_V2_BEHAVIOUR_RE   = re.compile(r'behaviour=(-?[\d.]+)')


def _summarize_v2_lines(v2_lines: list):
    """Return (mean_behaviour, mean_0p5_ep_reward, dominance_ratio) or None."""
    bs, eps = [], []
    for ln in v2_lines:
        mb = _V2_BEHAVIOUR_RE.search(ln)
        me = _V2_EP_REWARD_RE.search(ln)
        if mb: bs.append(float(mb.group(1)))
        if me: eps.append(float(me.group(1)))
    if not bs or not eps:
        return None
    b   = float(np.mean(bs))
    ep  = 0.5 * float(np.mean(eps))
    denom = abs(b) + abs(ep)
    dom = (abs(ep) / denom) if denom > 0 else float('nan')
    return b, ep, dom


def cmd_smoke(args) -> int:
    """Run each of the three evaluators on a cheap random checkpoint.

    Uses --episodes (default 3) so the per-component v2 log has enough
    samples to judge whether ep_reward dominates behaviour in practice.
    """
    print(CUTOVER_BANNER)
    print(f'Mode: smoke (static / hebb / GRU) under composite_exploration_v2  '
          f'(episodes={args.episodes})')
    print()

    results = []

    # static
    p = make_static_checkpoint('small', args.input_dim, args.num_actions,
                               _tmp('smoke_static_'), seed=42)
    try:
        r = run_static_eval(p, 'composite_exploration_v2', args.mission,
                            episodes=args.episodes, seed_base=args.seed_base,
                            num_agents=args.num_agents, timeout=args.timeout)
    finally:
        try: os.remove(p)
        except OSError: pass
    results.append(('static', r))

    # hebb
    p = make_hebb_coeffs(_tmp('smoke_hebb_'), args.input_dim, args.num_actions, seed=43)
    try:
        r = run_hebb_eval(p, 'composite_exploration_v2', args.mission,
                          episodes=args.episodes, seed_base=args.seed_base,
                          num_agents=args.num_agents, timeout=args.timeout)
    finally:
        try: os.remove(p)
        except OSError: pass
    results.append(('hebb', r))

    # GRU
    p = make_gru_checkpoint(_tmp('smoke_gru_'), args.input_dim,
                            proj=64, hidden=32, num_actions=args.num_actions, seed=44)
    try:
        r = run_gru_eval(p, 'composite_exploration_v2', args.mission,
                         episodes=args.episodes, seed_base=args.seed_base,
                         num_agents=args.num_agents, timeout=args.timeout)
    finally:
        try: os.remove(p)
        except OSError: pass
    results.append(('GRU', r))

    failed = 0
    for name, r in results:
        ok = (r.returncode == 0 and r.score is not None)
        print(f'--- {name} ---')
        print(f'  rc={r.returncode}  final_score={r.score}  ok={ok}')
        if r.v2_lines:
            for ln in r.v2_lines:
                print(f'    {ln}')
            summ = _summarize_v2_lines(r.v2_lines)
            if summ is not None:
                b, ep_weighted, dom = summ
                print(f'  summary: mean behaviour={b:.4f}  mean 0.5*ep_reward={ep_weighted:.4f}  '
                      f'ep_reward share={dom*100:.1f}%')
                # A high share only means ep_reward dominates SELECTION if the
                # behaviour term is also meaningful in absolute magnitude. On
                # near-idle policies both terms are near-zero; the share is
                # mechanically inflated but ep_reward isn't really "winning".
                if dom > 0.7 and abs(b) > 0.1:
                    print(f'  WARN: behaviour={b:.3f} is non-trivial but ep_reward still'
                          f' drives >70% of |score|. Consider lowering the 0.5*ep_reward'
                          f' coefficient.')
                elif dom > 0.7:
                    print(f'  note: high ep_reward share, but |behaviour|={abs(b):.3f} is'
                          f' too small for this to indicate a design problem — the policy'
                          f' is effectively idle in this evaluator.')
        if not ok:
            failed += 1
            print(f'  --- tail stdout ---\n{r.stdout_tail}\n  -------------------')
        print()

    print(f'{len(results) - failed}/{len(results)} evaluators passed smoke under v2.')
    return 0 if failed == 0 else 1


def cmd_rollout_diff(args) -> int:
    """Score two checkpoints side-by-side under v1 and v2 on matched seeds."""
    print(CUTOVER_BANNER)
    print(f'Mode: rollout-diff  a={args.ckpt_a}  b={args.ckpt_b}')
    print()
    rows = []
    for label, path in [('A', args.ckpt_a), ('B', args.ckpt_b)]:
        v1 = run_static_eval(path, 'composite_exploration',    args.mission,
                             args.episodes, args.seed_base, args.num_agents, args.timeout)
        v2 = run_static_eval(path, 'composite_exploration_v2', args.mission,
                             args.episodes, args.seed_base, args.num_agents, args.timeout)
        rows.append((label, path, v1, v2))

    print(f'{"ckpt":<6} {"v1":>10} {"v2":>10}   path')
    for label, path, v1, v2 in rows:
        v1s = 'FAIL' if v1.score is None else f'{v1.score:>10.4f}'
        v2s = 'FAIL' if v2.score is None else f'{v2.score:>10.4f}'
        print(f'{label:<6} {v1s:>10} {v2s:>10}   {path}')
    print()
    a_v1, b_v1 = rows[0][2].score, rows[1][2].score
    a_v2, b_v2 = rows[0][3].score, rows[1][3].score
    if None not in (a_v1, b_v1, a_v2, b_v2):
        print(f'  v1 ranks: A {"<" if a_v1 < b_v1 else ">=" } B')
        print(f'  v2 ranks: A {"<" if a_v2 < b_v2 else ">=" } B')
        if (a_v1 < b_v1) != (a_v2 < b_v2):
            print()
            print('  DISAGREEMENT: v1 and v2 disagree on which checkpoint is better.')
            print('  This is the signal to go look at the rollouts visually and decide')
            print('  which ordering is correct behaviourally.')
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Validate composite_exploration_v2 against its pushback points.')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--mission', default='tutorial.scout')
        sp.add_argument('--episodes', type=int, default=DEFAULT_EPISODES)
        sp.add_argument('--seed-base', dest='seed_base', type=int, default=DEFAULT_SEED)
        sp.add_argument('--num-agents', dest='num_agents', type=int, default=1)
        sp.add_argument('--input-dim', dest='input_dim', type=int, default=STATIC_INPUT_DIM_LEGACY,
                        help='Static-MLP input dim (legacy_flat obs on tutorial.scout = 900).')
        sp.add_argument('--num-actions', dest='num_actions', type=int, default=STATIC_NUM_ACTIONS_FALLBACK,
                        help='Must match len(policy_env_info.action_names). Override if the env changes.')
        sp.add_argument('--timeout', type=int, default=600)

    sp_ord = sub.add_parser('ordering', help='rank do-nothing/random/half/strong under v1 and v2')
    common(sp_ord)
    sp_ord.add_argument('--half-trained', default=None, help='optional checkpoint path')
    sp_ord.add_argument('--strong',       default=None, help='optional checkpoint path')
    sp_ord.set_defaults(func=cmd_ordering)

    sp_bl = sub.add_parser('baseline', help='measure v2 random baseline for one evaluator')
    common(sp_bl)
    sp_bl.add_argument('--trials', type=int, default=20)
    sp_bl.add_argument('--evaluator', choices=['static', 'hebb', 'GRU'], default='static',
                       help='Which evaluator to baseline. Each architecture has its '
                            'own random baseline; do not mix numbers across evaluators.')
    sp_bl.set_defaults(func=cmd_baseline)

    sp_sm = sub.add_parser('smoke', help='run each of the 3 evaluators once under v2')
    common(sp_sm)
    sp_sm.set_defaults(func=cmd_smoke)

    sp_rd = sub.add_parser('rollout-diff', help='score two checkpoints under v1 and v2')
    common(sp_rd)
    sp_rd.add_argument('ckpt_a')
    sp_rd.add_argument('ckpt_b')
    sp_rd.set_defaults(func=cmd_rollout_diff)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

# v5 smoke protocol — diagnostic first run

**Goal**: validate the v5 wiring on the SLURM cluster and produce decomposed
per-component fitness data for two architectures (static MLP, hebb) under
two cross-agent reductions (`max`, `mean`) — plus one matched random-search
baseline so we can answer the synthesis-#2 question (does CMA-ES beat
random-search on this fitness?).

**Scope**: this is a SMOKE run. Parameters are deliberately small so the
whole protocol fits in a few cluster-hours. Production-scale follows once
smoke shows non-zero F somewhere and the per-component sidecar JSON is
landing in `.resoult.json`.

---

## Run inventory

5 jobs total:

| Job | Driver | Config | Reduction | Purpose |
|---|---|---|---|---|
| **A** | `task_cogames_static.py`           | `smoke_configs/task_cogames_static_smoke.yaml`        | max  | reference v5 baseline |
| **B** | `task_cogames_static.py`           | `smoke_configs/task_cogames_static_smoke_mean.yaml`   | mean | tests team-coordination signal |
| **C** | `task_cogames_hebb.py`             | `smoke_configs/task_cogames_hebb_smoke.yaml`          | max  | hebb v5 baseline |
| **D** | `task_cogames_hebb.py`             | `smoke_configs/task_cogames_hebb_smoke_mean.yaml`     | mean | hebb team-coordination signal |
| **R** | `task_cogames_random_search_baseline.py` | `smoke_configs/task_cogames_random_search_smoke.yaml` | max | matched-wallclock baseline (vs A) |

Jobs A–D run independently on SLURM via your existing `slurmHPCHelper`
launch path. Job R is a single Python process (no SLURM coordination
needed) but should run on a comparable node so wallclock is matched
against A.

---

## Search-shape parameters (smoke)

These are the values you should plug into the CLI args / wrapper for jobs A–D:

| Parameter | Value | Where it comes from | Why this value for smoke |
|---|---|---|---|
| `eval_episodes`               | **5**   | YAML (`StaticEval.eval_episodes`) | Was 40 in production — drop 8× for fast iteration. |
| `population_size`             | **16**  | CLI / wrapper       | Small but enough for CMA-ES rank statistics to be meaningful (>2× problem dim is overkill at smoke; CMA-ES default µ=λ/2 still fits). |
| `max_epochs` (generations)    | **10**  | CLI / wrapper       | Enough generations for any signal to start moving. Plot per-component trajectories; if `score_F` is flat at 0 by gen 10, scale that channel up before production. |
| `total_test_with_same_param`  | **1**   | CLI (`--total_test_with_same_param`) | Each gene is tested ONCE per generation (no re-sampling). CRN seeds remove map-luck variance across candidates within a generation, so re-tests aren't load-bearing at smoke. |
| `total_testsGene_with_same_agent` | **1** | CLI (`--total_testsGene_with_same_agent`) | One gene per worker per generation. Workers are stateless — fan-out via SLURM is the parallelism. |
| `evolutionTarget`             | **1**   | CLI (`--evolutionTarget`) | Maximize. v5 score is a positive-is-better aggregate. |
| `seed_base` (env)             | 12345   | YAML (`StaticEval.seed_base`) | Combined with `+ epoch * 999983` per [task_cogames_static.py:521-526](task_cogames_static.py) for CRN. |

**In-game agents per episode**: 4 (fixed by `num_agents: 4` in YAML).
**Total evaluations per CMA-ES smoke run**: 16 candidates × 10 generations × 1 retest × 5 episodes = **800 episodes**.

---

## Wallclock estimates

Per-eval cost (single candidate, 5 episodes × ~800 steps × 4 agents on CPU):
~25–60 seconds depending on policy depth and cluster CPU. Using 60 s/eval:

| Job | Total evals | Sequential | With 16 parallel workers |
|---|---:|---:|---:|
| A (static-max)  | 160 | ~2.7 h | **~10 min/gen × 10 gens ≈ 100 min** |
| B (static-mean) | 160 | ~2.7 h | ~100 min |
| C (hebb-max)    | 160 | ~2.7 h | ~100 min |
| D (hebb-mean)   | 160 | ~2.7 h | ~100 min |
| R (random)      | 160 | ~2.7 h | sequential (single process) |

A,B,C,D are SLURM-parallelized via the worker pool — wallclock is dominated
by the slowest gen, not total work. R is sequential by construction, so
budget ~3 hours.

---

## Phase 0 — pre-flight on the cluster (single-eval test)

Before launching the four CMA-ES runs, confirm the v5 sidecar wiring works
inside your SLURM container. From a head node or interactive job, run ONE
evaluator subprocess matching the smoke config:

```bash
# Pick a temp dir on shared cluster storage
RUN_ROOT=/scratch/$USER/cogames_v5_preflight
mkdir -p $RUN_ROOT && cd $RUN_ROOT

# Synthesise a fake gene (random uniform [-0.3, 0.3])
python3 - <<'PY'
import torch, numpy as np
rng = np.random.default_rng(42)
torch.save({
    'W1': rng.uniform(-0.3, 0.3, size=64*900).astype(np.float32),
    'W2': rng.uniform(-0.3, 0.3, size=32*64).astype(np.float32),
    'W3': rng.uniform(-0.3, 0.3, size=5*32).astype(np.float32),
}, 'preflight_weights.dat')
PY

# Run evaluator with sidecar
python3 /path/to/cogames/evaluate_static_cogames.py \
    --mission arena \
    --weights_path preflight_weights.dat \
    --hidden_1 64 --hidden_2 32 \
    --eval_episodes 2 \
    --seed_base 12345 \
    --fitness_stat composite_v5_role_event \
    --num_agents 4 \
    --components_sidecar preflight_components.json \
    --fitness_v5_weights "w_F_heart=5.0,w_F_chest=5.0,w_F_ep=3.0,w_B_ach_ore=0.3,w_B_ach_gear=0.5,w_B_role_events=0.2,w_B_explore=0.4,w_B_init=0.0,score_B_cap=3.0,reduction=max"

# Confirm sidecar shape
python3 -c "
import json
d = json.load(open('preflight_components.json'))
agg = d['aggregate']
print('OK' if all(k in agg for k in ('score','score_F','score_B','ach_first_ore','heart_gain_count')) else 'FAIL')
print({k: agg[k] for k in ('score','score_F','score_B','heart_gain_count','heart_spend_count')})
"
```

**Pass criterion**: prints `OK` and a dict with all listed keys present.
Expected `score` ~0.1–0.5 with random weights (almost all from `score_B`).
If `score_F > 0` you got lucky and an agent crafted/spent a heart — also
fine.

---

## Phase 1 — CMA-ES smoke runs A, B, C, D

For each of A–D, launch via your existing slurmHPCHelper wrapper. The
parameters above translate to the `slurmHPCHelper.run(...)` config you've
been using for v4. Each run gets its OWN output directory:

| Job | Driver | Config | Recommended `--path` |
|---|---|---|---|
| A | `task_cogames_static.py`        | `smoke_configs/task_cogames_static_smoke.yaml`         | `/scratch/$USER/v5_smoke/static_max`  |
| B | `task_cogames_static.py`        | `smoke_configs/task_cogames_static_smoke_mean.yaml`    | `/scratch/$USER/v5_smoke/static_mean` |
| C | `task_cogames_hebb.py`          | `smoke_configs/task_cogames_hebb_smoke.yaml`           | `/scratch/$USER/v5_smoke/hebb_max`    |
| D | `task_cogames_hebb.py`          | `smoke_configs/task_cogames_hebb_smoke_mean.yaml`      | `/scratch/$USER/v5_smoke/hebb_mean`   |

The CLI shape your wrapper feeds to each per-worker call (use the same args
you've been passing v4 — only the `--paramFile` and `--path` change):

```text
python3 task_cogames_static.py \
    --path                         <RUN_PATH> \
    --paramFile                    <SMOKE_YAML> \
    --agent_idx                    <0..population_size-1> \
    --agent_test_num               <epoch * total_testsGene_with_same_agent + gene> \
    --agent_counter                <epoch.float> \
    --total_test_with_same_param   1 \
    --total_testsGene_with_same_agent  1 \
    --evolutionTarget              1 \
    --slurm                        True \
    --parallel                     False \
    --gpu                          False
```

Set the CMA-ES outer loop to `population_size=16, max_epochs=10` in your
slurmHPCHelper invocation. Hebb's job (C, D) calls `task_cogames_hebb.py`
instead of `task_cogames_static.py` — driver only differs.

---

## Phase 2 — random-search baseline R

Run after A is set up so you can match its wallclock. `task_cogames_random_search_baseline.py`
is self-contained — no SLURM coordination needed. Submit as a single SLURM
job (or run on a head/interactive node):

```bash
# Match A's total eval count: population_size=16 × max_epochs=10 = 160 evals
RUN_ROOT=/scratch/$USER/v5_smoke/random_search_static
mkdir -p $RUN_ROOT

python3 task_cogames_random_search_baseline.py \
    --path /scratch/$USER/v5_smoke/random_search_static \
    --paramFile smoke_configs/task_cogames_random_search_smoke.yaml \
    --population_size 16 \
    --max_epochs 10 \
    --gene_seed_base 99999     # distinct from CMA-ES seed_base so no overlap
```

Output: `results_epoch_{0..9}.jsonl` files under `$RUN_ROOT`. Each line
is one evaluation (`epoch, cand_idx, score, score_F, score_B, ach_*, ...`).

**This run uses the same evaluator subprocess and the same CRN seed base
as A**, so the per-eval wallclock will be roughly identical. Total
sequential cost: ~160 evals × 60 s ≈ 2.7 h.

---

## Phase 3 — analysis

After all 5 jobs finish, the data lives in:

- A–D: `/scratch/$USER/v5_smoke/{static_max,static_mean,hebb_max,hebb_mean}/running/<agent_idx>/<test_num>.resoult.json` — each `.resoult.json` is a list of gene dicts; for v5 each dict has `components_v5_per_test` containing the sidecar JSON (per-episode + per-agent).
- R: `/scratch/$USER/v5_smoke/random_search_static/results_epoch_{0..9}.jsonl` — flat jsonl, one row per eval.

### What to check

1. **Sidecar landed**. For at least one `.resoult.json` from each of A–D:
   ```python
   import json
   d = json.load(open('.../0.resoult.json'))
   gene = d[0]
   assert 'components_v5_per_test' in gene
   assert 'aggregate' in gene['components_v5_per_test'][0]
   print(gene['components_v5_per_test'][0]['aggregate']['score_F'])
   ```
   If `components_v5_per_test` is missing, the task driver isn't merging the sidecar — check `taskPath/components_v5_*.json` for orphan files (they should have been deleted after merge).

2. **Per-component trajectories** — does CMA-ES move ANY component over generations? Plot per genome per generation: `aggregate.score`, `aggregate.score_F`, `aggregate.score_B`, each `ach_first_*`, `heart_gain_count`, `role_event_total`. If the entire trajectory is flat noise, the optimizer isn't selecting on anything — escalate before production.

3. **A vs R (CMA-ES vs random-search)**. For each of `score`, `score_F`, `score_B`, and each achievement flag, compare:
   - **Best-so-far at each generation** (CMA-ES tail vs random-search cumulative best)
   - **Mean per generation** (CMA-ES population mean vs random-search per-epoch mean)
   If CMA-ES doesn't strictly beat random-search by gen 10 on aggregate score AND on at least one component, the synthesis-#2 conclusion is "the optimizer is ranking on noise, fix the signal before adding compute".

4. **A vs B (max vs mean reduction)**. Under `max`, the genome's score equals the best agent's score; under `mean`, every agent contributes. Look at `gear_share` and `ach_first_role_event_*` per agent: under `mean` we expect more even role distribution across agents (cooperation); under `max` we expect a specialist pattern (one agent dominates).

5. **C vs D (hebb max vs hebb mean)** — same comparison as #4 for the Hebbian variant. Hebb agents have plasticity, so the trajectory shape may differ from static; that's expected.

### Pass / fail for the smoke run

- **Pass**: any of A–D shows non-zero `score_F` in at least one genome by gen 10, AND the per-component sidecar lands in `.resoult.json`. Then production-scale runs are warranted.
- **Inconclusive**: all `score_F` is 0 across A–D, but `score_B` shows trajectory-like motion. Means dense shaping is doing all the work — scale `w_F_*` weights up by 2× or extend to 30 generations before declaring "v5 doesn't work".
- **Fail**: sidecar is missing, OR random-search beats CMA-ES on aggregate score AND every component. Diagnose before scaling up.

---

## Production-scale follow-up (after smoke passes)

Once smoke confirms wiring + non-trivial signal, scale up:
- `eval_episodes`: 5 → 40 (matches v4 production).
- `population_size`: 16 → 32 or 50 (typical CMA-ES production).
- `max_epochs`: 10 → 50 (or until plateau).
- `total_test_with_same_param`: 1 → 3 (for variance reduction; CRN already cuts most of this).

Then run all 7 unified variants under `max` reduction (and a parallel `mean`
A/B if compute permits).

---

## Output dir layout (cheat sheet)

```
/scratch/$USER/v5_smoke/
├── static_max/
│   └── running/
│       └── <agent_idx>/
│           └── <test_num>.resoult.json   ← contains components_v5_per_test
├── static_mean/
│   └── (same)
├── hebb_max/
├── hebb_mean/
└── random_search_static/
    ├── results_epoch_0.jsonl
    ├── results_epoch_1.jsonl
    └── ...
```

---

## Quick-reference: parameter values

```text
in-game agents per episode:           4              (env-fixed)
eval_episodes per gene per re-test:   5              (smoke; production uses 40)
total_test_with_same_param:           1              (smoke; production uses 3)
total_testsGene_with_same_agent:      1
population_size (CMA-ES candidates):  16
max_epochs (generations):             10
evolutionTarget:                      1              (maximize)
seed_base (env CRN):                  12345          (rotates by epoch * 999983)
gene_seed_base (random-search only): 99999

CMA-ES (already in YAML):
    initial_sigma:                    0.1
    sigma_max_restart:                1.5
    sigma_boost_factor:               1.0
    diag_mode:                        auto
```

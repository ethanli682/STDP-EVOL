# Para-HADES ↔ CoGameAgent integration (spatialmem genome)

Wires Para-HADES' OpenAI-ES to evolve the weights of a CoGameAgent policy net
("the gene") to play `machina_1`, warm-started from a behaviour-cloning net.
The two repos stay **decoupled**: Para-HADES runs a CoGameAgent **scorer
subprocess** and never imports game code. Built from the work order
`CoGameAgent/research/plans/parahades_integration_workorder_20260716.md`.

## Files created (this repo)

| File | Role |
|---|---|
| `task_spatialmem_cogames.py` | Thin Para-HADES task module. `main(args)` reads the candidate gene from `running/<agent>.newMember.json\|pkl`, writes it to a temp `gene.npy`, subprocesses the §1 scorer, reads `fitnessScore` back, writes `<test>.resoult.json\|pkl`. Modeled 1:1 on `task_ANN.py` (same master/worker + preemption/requeue). |
| `task_spatialmem_cogames.yaml` | Run config: one `array` genome of length **928764**, symmetric bounds ±B, decoupled `scorer:` block, per-worker resources (1 core, `mettagrid.sif`), and the OpenAI-ES `ML:` block (sigma 0.02, lr auto, rank, mirroring). |
| `compute_bounds.py` | Stand-in for the scorer's absent `--dump-weight-stats`: prints `max_abs`, `std`, and `B = max(3, 1.5·max_abs)` from a `net.pt` (pure torch, no game code). |
| `slurmHPCHelper.py` (edited) | `SlurmScript` gains an optional `extra_pythonpath` arg, appended to the container `PYTHONPATH`. Backward-compatible: other tasks (e.g. `task_ANN`) are untouched. |

## Full-run launch command (cluster, OLD Pax — run when scorer + archB net exist)

```bash
ssh login-prod-03.pax.tufts.edu
cd /cluster/tufts/levinlab/hhazan01/git_repo/Para-HADES
python3 GA.py \
  --slurm True --parallel False --gpu False \
  --evolutionType OpenAI_ES --evolutionTarget 1 --geneMin -1.0 \
  --populationSize 128 --numOFgenerations 20 \
  --num_of_agents 32 --total_testsGene_with_same_agent 4 --total_test_with_same_param 1 \
  --taskFilename task_spatialmem_cogames.py \
  --paramFile   task_spatialmem_cogames.yaml \
  --path /cluster/tufts/levinlab/hhazan01/Para-HADES_Tasks/spatialmem_machina1_YYYYMMDD
```

- 128 genes/generation = 32 agents × 4 genes (each agent = 2 mirrored pairs); each
  gene = 1 scorer call = 5 seeds × 10k steps ≈ minutes on 1 core.
- `--evolutionTarget 1` = **maximize** the team score (higher fitness = better).
- **§7.4 single-generation smoke:** same command with `--numOFgenerations 1
  --num_of_agents 32`.
- **Before launch:** set the real bounds — `python3 compute_bounds.py <archB
  net.pt>` and paste `min`/`max` into the YAML (see Caveats).

## Local synchronous smoke (what §7.3 ran; uses a MOCK scorer)

```bash
export PYTHONPATH=/mnt/exDisk0/git_repo/Para-HADES OMP_NUM_THREADS=1
python3 GA.py --parallel False --slurm False --gpu False \
  --evolutionType OpenAI_ES --evolutionTarget 1 --geneMin -1.0 \
  --populationSize 8 --numOFgenerations 2 \
  --num_of_agents 2 --total_testsGene_with_same_agent 4 --all_in_first_run 1 \
  --taskFilename task_spatialmem_cogames.py \
  --paramFile <mock plumbing yaml> --path <run dir>
```

## Smoke results (2026-07-15)

1. **Codec round-trip (§7.1)** — real Para-HADES `paramTOgene`→`geneTOparam`:
   max abs err **2.861e-6** (flat BC net @448652, B=17.99; and representative
   gene @928764, B=22). 99.98% of weights round-trip < 1e-6; the 2.86e-6 floor
   is float32 precision of the `[-1,1]` affine map at large outlier-driven B
   (float64 → 1.8e-15). **Does not meet the literal <1e-6**, but is ~10^5×
   smaller than one ES step — practically negligible. See Caveats.
2. **Warm-start sanity (§7.2)** — **BLOCKED**: needs the (absent) scorer and the
   (absent, untrained) SpatialMemNet. Not run; not fabricated.
3. **2-generation micro-run (§7.3)** — real GA.py loop, pop 8, serial, **mock**
   sphere scorer: ran **crash-free**, 104 members scored, OpenAI-ES engaged in
   gen 2, best fitness **-50.72 (gen 0 random) → -3.86 (gen 1 ES)**, wall **38.3 s**.
   The task-IO contract and the ES-engine climb were also unit-verified. *Mock
   fitness is a sphere function — it proves loop/engine mechanics only, NOT game
   skill.*
4. **One cluster generation (§7.4)** — **BLOCKED**: needs the scorer on the
   cluster and an on-cluster session. Command above is ready.

## Caveats / flags for the PI

- **§1 scorer `scripts/stage_b/parahades_score.py` is not yet delivered** on the
  CoGameAgent side; all game-eval acceptance (§7.2/§7.4, real §7.3) is blocked on it.
- **No SpatialMemNet (928764) net.pt exists** (`archB-spatialmem/` has only
  `prereg.json`); the only nets on disk are the FLAT 448652 arch. The YAML bound
  **B=22 is a placeholder** — recompute with `compute_bounds.py` on the real net.
- **Bounds↔sigma interaction:** `max|weight|` (~12–14.6) is a lone outlier; 99.7%
  of weights are within ±0.9. The §4 formula therefore inflates B to ~18–22, and
  §5's `sigma 0.02` becomes `0.02·B ≈ 0.36–0.44` in weight space — **larger than
  the weight std (0.30)**. Consider a smaller sigma (~0.002–0.005) or clipping the
  outlier so B tracks the bulk scale.
- **Warm-start seeding:** GA.py Part 1 bootstraps with *random* genes
  (`generateRandomGene` → weights ~N(0, B/3) ≈ N(0,7.3), i.e. garbage). To
  actually start the ES mean at the BC gene (§4), seed ≥`populationSize` scored
  BC-perturbation genes into the run history first (the ES inits `theta` from the
  population mean). This step needs the scorer and must be finalized with the
  CoGameAgent side.
- **Payload/disk:** one 928764 candidate ≈ 9.3 MB zstd; a 128-pop history over 20
  generations grows to ~24 GB. Prune old epochs if disk is tight.
- **`random_injection_rate: 0.0` still injects a floor of 1 random candidate** per
  agent-batch (observed in the smoke) — a few wasted evals per generation.

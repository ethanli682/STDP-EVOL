# Copilot instructions for Para-HADES

## Big picture architecture
- `GA.py` orchestrates the full loop: bootstrap, spawn task workers, collect reports, append logs, generate next candidates.
- `GA_class.py` (`difEvoInit`) owns gene↔param conversion, gene format choice, and built-in param extraction.
- Evolution is plugin-dispatch by `--evolutionType` in `GA.py` (GA/CMA_ES/SNES/RevFF/HADES/DiffEvo/MultiHead/CondVAE modules).
- `GA_utils_history_loader.py` is shared infrastructure for chunked history streaming and Top-K selection; many plugins depend on its API.

## Runtime data flow (important)
- Two-phase run in `GA.py`: `part=1` (init/spawn), `part=2` (collect reports/evolve).
- File contract (must stay stable):
  - producer: `running/<agent>.newMember.json|pkl`
  - task output: `running/<agent>/<test>.resoult.json|pkl`
  - history: `<agent>.log.csv` (+ `<agent>.log.pkl` when pickle mode)

## Project-specific conventions
- `args.agent_counter` is serialized as `"iteration.epoch"` across process boundaries.
- `geneFormat` is dynamic (`json` vs `pickle`+`zstandard`) based on gene size in `GA_class.py`.
- Task/runtime resource knobs come from task YAML (`main.Task_resource`, `main.Main_resource`, `ML.*`, `param.*`).
- Preserve existing fault-tolerant style (`try/except` with continue) for long-running jobs.
- Keep protocol spellings intact (`.resoult.*`), unless all readers/writers are migrated together.

## Run and debug workflows
- Serial baseline:
  - `python3 GA.py --taskFilename task_FindVector.py --paramFile task_FindVector.yaml --path <run_dir> --parallel False --slurm False`
- Parallel local: set `--parallel True` (task scripts clamp BLAS/OMP threads).
- SLURM: set `--slurm True`; job scripts are emitted to `<run_dir>/scripts/` via `slurmHPCHelper.py` and requeued on `SIGTERM`.
- Quick checks: `ls <run_dir>/running`, `tail -n 5 <run_dir>/*.log.csv`, `ls <run_dir>/scripts`.

## Integration points and dependencies
- Core deps are task-dependent: `numpy`, `torch`, `zstandard`, `yaml`, `sklearn`, `tqdm`, `matplotlib`, `scipy`, `torchvision`.
- MNIST tasks (`task_ANN.py`, LoRa ANN variants) rely on `./MNIST_data` or torchvision download.
- SLURM helper assumes Singularity + `/cluster` bind paths in `slurmHPCHelper.py`.

## Editing guidance for AI agents
- If I/O changes, patch both ends (`GA.py` + active `task_*.py`) and keep callback/requeue behavior intact.
- New task pattern: read `newMember.*` → compute `fitnessScore` per gene → write `*.resoult.*` in task `main(args)`.
- New evolution plugin pattern: add `args.evolutionType` dispatch branch in `GA.py`; keep creator signature aligned with existing modules.
- Plugin outputs must be serializable genes compatible with `ea.geneTOparam(...)` and `newParam[-1]['gene']` writing.

## High-risk pitfalls
- Do not rename `.resoult` files to `.result` unless all producers/consumers are migrated together.
- Avoid changing `agent_counter` encoding format; many scripts parse it as `"iteration.epoch"`.
- Avoid clearing or rewriting history formats casually; `GA_utils_history_loader.py` is shared by most evolution modules.
- `testing.py` references `run_simulation` that is not present in current `GA.py`; use direct `GA.py` CLI runs for verification.

## Troubleshooting matrix (fast triage)
- No child processes: check `run(...)` assembly and `--parallel/--slurm` handling in `GA.py`.
- No `*.resoult.*`: check active `task_*.py` `main(args)` load/write/callback paths.
- `*.log.csv` not growing: check GA part-2 report collection branch in `GA.py`.
- Evolution crash after history edits: check `GA_utils_history_loader.py` and all `GA_class_*`/`condevo/*` callers.
- SLURM requeue loops: check `slurmHPCHelper.py` trap + `nextTask` chain and task signal handlers.

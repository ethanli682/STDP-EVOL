# Dot-tracing analysis

`analyze_dottracing.py` produces PNG/SVG figures and the numeric data behind them.
Run it in the same Python environment used for the dot-tracing task. Plotting old
logs needs NumPy, PyYAML, Matplotlib, scikit-learn (automatic ranking only), and
zstandard (framed histories); new evaluations also need this BindsNET checkout.

## Where Slurm saves evolution histories

The directory passed to `GA.py --path` holds `0.log.csv`, `1.log.csv`, etc.
These are **JSON Lines**, despite their `.csv` extension. Pickle-mode runs also
have `.log.framed.bin` and `.log.framed.idx` companions containing decoded genes.
The analysis automatically reads those companions when present. Read only your
own/trusted history files because the framed format uses pickle.

The supplied job log points to this cluster path:

```
/cluster/home/eli08/bindsnet/Para-HADES_Tasks/dot_tracing10
```

From the cluster's `bindsnet/Para-HADES` checkout, locate histories with:

```sh
rg --files --hidden --no-ignore ../Para-HADES_Tasks -g '*.log.csv'
```

The local sibling `../Para-HADES_Tasks` currently contains only `test_directory`,
without completed histories. A Slurm `.out` file records evaluations, but does
not provide reliable generation boundaries. The analyzer deliberately rejects
it as a source for a generation curve.

## Best and mean individual per generation, one line per starting seed

Use one `--run LABEL=PATH` per independently initialized evolution run. For
example, after replacing these example folder names with the actual folders:

```sh
python analyze_dottracing.py evolution \
  --run seed0=../Para-HADES_Tasks/dot_tracing_seed0 \
  --run seed1=../Para-HADES_Tasks/dot_tracing_seed1 \
  --run seed2=../Para-HADES_Tasks/dot_tracing_seed2 \
  --output out/dottracing_analysis/evolution
```

Outputs: `evolution.png`, `evolution.svg`, and `generation_summary.csv`.
The two panels show the maximum and arithmetic mean of evaluated individuals
within each `(epoch, iteration)` pair. All supplied agents within a run are
pooled; repeated fitness tests are first averaged **per individual**. Best is
per-generation, not best-so-far. Para-HADES runs asynchronously: these are logged
update cohorts, not necessarily a synchronized population of fixed size.
Incomplete cohorts are included with their sample count in the CSV. The x-axis
orders observed `(epoch, iteration)` pairs; missing cohorts are not fabricated.
Original epoch/iteration coordinates are retained in the CSV.

Compare runs with the same architecture, scoring rule, settings, and budget.
Different job numbers or different model versions are not automatically
independent starting seeds. A simulation seed controls network/environment
randomness; it does not seed the Para-HADES optimizer's initial population.

## Choose two parameters using a screen

First sample the full YAML parameter space with a reproducible Latin hypercube.
Each candidate is evaluated through the real `dot_tracingtask.do_task` path.
Defaults preserve the current architecture, 100 decisions, and three training
plus three test episodes. Rendering is disabled. Replicates use common network
and environment seeds across candidate genes.

```sh
python analyze_dottracing.py screen --samples 128 --seeds 0 1 2 \
  --workers 4 --threads 1 --output out/dottracing_analysis/screen
```

`parameter_importance.csv` ranks genes by held-out prediction error increase
when that gene is shuffled. The fit is nonlinear; five train/test splits give
a rough uncertainty estimate. Replicate seeds are averaged before splitting
so the same candidate cannot appear in both training and validation.
`ranking_validation.json` reports held-out R². A weak validation score or
large overlapping uncertainties means the top two are **tentative**, not an
established causal ranking. Increase coverage and/or use more seeds before
making strong conclusions. Screening on several baseline regions is useful
when the system switches sharply between silent and saturated states.

`selected_axes.json` contains the two selected names. Array components are
addressed as `sparsity_mc_opp[0]`, etc. Fitness-based importance answers the
requested fitness question; it does not by itself establish causal importance
for firing activity. Use `--rank-metric ac_active_fraction` on `screen` or
`observed` to rank predictive importance for activity instead. The heatmap
still shows returned fitness and activity side by side.

## Controlled fitness heatmap

Use the selected two parameter names in `--axes`, for example:

```sh
python analyze_dottracing.py sweep \
  --axes scaler_gc_pc1 scaler_inhib_ac \
  --grid-size 11 --seeds 0 1 2 --workers 4 --threads 1 \
  --output out/dottracing_analysis/sweep
```

The axis names above are examples, not a claim that they are the two most
important genes. Every other gene stays at the YAML midpoint, or at a complete
baseline provided with `--baseline baseline.json` (either `{"param": {...}}`
or the decoded parameter dictionary). Axes cover their full YAML bounds.
The same seeds are used in every cell. This is a conditional two-dimensional
slice; it cannot describe all interactions in the full parameter space.

The four heatmaps show mean returned fitness, mean AC activity, the fraction
of evaluations whose **every test episode** meets the activity band, and sample
count. Gray cells have no data. A mean activity of 45% alone does not prove
stability: averaging silence and saturation can produce the same number.
`evaluations.jsonl` also stores per-episode activity variability and the fraction
of individual decision windows within the band.

`experiment.json` records settings, bounds, baseline, seeds, and simulator/config
hashes. `evaluations.jsonl` checkpoints every completed evaluation. Rerunning
the identical command resumes missing evaluations; incompatible settings fail
instead of mixing experiments. `--workers` can change during resume. Each
worker uses `--threads` CPU threads; keep their product within the allocation.
Run only one writer against an output folder at a time.

## Quick pilot and existing worker logs

A shortened screen can validate the pipeline, but must be labeled as a pilot:

```sh
python analyze_dottracing.py screen --samples 64 --seeds 0 \
  --steps 25 --episodes 1 --workers 4 \
  --output out/dottracing_analysis/pilot_screen

python analyze_dottracing.py observed T_dot_tracing10.4999053.out \
  --output out/dottracing_analysis/latest
```

`observed` bins actual samples while other genes vary. It does not interpolate
or pretend to be a controlled slice. Worker stdout has rounded gene values and
activity; prefer full histories. Analyze one consistent model/scoring version
at a time. Earlier dot-tracing logs in this checkout have different fitness
rules and should not be pooled with the latest run.

## Findings in the code relevant to interpretation

- Active means at least four spikes per decision window, averaged over the
  episode. The current default window is 100 ms; this is not a 25–65 Hz target.
- If any test episode is outside 25–65%, fitness is `-10 * ac_penalty` times
  the mean distance to the band **among failing episodes only**. Otherwise it
  is mean task performance. Training fitness is returned separately.
- Healthy candidates can have negative task rewards, so fitness sign alone
  does not classify firing health. A nearly healthy failing candidate can also
  outrank a healthy candidate with negative task reward under this objective.
- `sparsity_rec` currently has no effect: the recurrent mask is commented out.
  `scaler_inhib_ac` scales signed, dense recurrence, not exclusively inhibition.
  The analysis preserves the current model, including these properties.
- `GA.py` initially samples randomly until enough history exists: at least
  `max(int(populationSize * 0.1), 100)` records in its Part 2 gate. With default
  `populationSize=5000`, that threshold is 500. Early fitness curves can
  therefore describe random exploration before optimizer updates begin.
- Previously, the environment received no seed and used wall-clock randomness.
  The analyzer supplies `env_seed=simulation_seed+1` (zero means wall-clock to
  DotSimulator). Ordinary task calls keep their existing behavior unless
  `--env_seed` is explicitly set in the simulator configuration.

## Validation

```sh
python -m unittest test_analyze_dottracing -v
python -m py_compile analyze_dottracing.py dot_tracingtask.py
```

Tests cover per-individual replicate averaging, per-generation rather than
cumulative best, separate run labels, missing generation metadata, truncated
stdout, negative-but-healthy fitness, non-finite results, and array genes.

# dotTracing — PR2.py

`PR2.py` is the dot-tracing SNN restructured along the DeltaQ paper's GC → AC → MC
architecture (Earl et al., bioRxiv 2026). GC → AC is frozen; AC → MC is the only
plastic pathway (MSTDPET).

Debugging history — a silent AC layer, a diagnostic that scored silence as perfect,
and a saturated motor readout — is in `NOTES.md`.

## Default changes vs the original script

| Flag | Was | Now | Why |
|---|---|---|---|
| `--ac_gain` | 25.0 | **45.0** | ~10% AC sparsity, coverage 1.00 |
| `--ac_lbound` | *(none)* | **−80.0** | new; voltage floor, prevents permanent hyperpolarization |
| `--rec_exc` | *(none)* | **15.0** | new; total per-neuron excitatory budget, mV |
| `--rec_inh` | 1.2 (per-synapse) | **25.0** (total budget) | **semantics changed** — see warning below |
| `--ac_target_sparsity` | *(none)* | **0.10** | new; drives the diagnostic verdict band |
| `--ac_mc_init` | *(hardcoded 6.0)* | **0.3** | new flag; de-saturates MC |
| `--w_max` | 12.0 | **1.0** | keeps MC off its firing ceiling |
| `--nu` | 5e-2 | **4e-3** | must scale with `w_max`; 100 updates per decision |
| `--diag_stride` | *(hardcoded 3)* | **4** | new flag; 2-D lattice, 56 probe positions |
| `--diag_thresh` | *(hardcoded 4)* | **4** | new flag |

⚠️ **`--rec_inh` means something different now.** It was a per-synapse magnitude; it
is now a *total per-neuron budget in mV* — the drive a cell receives if every
inhibitory partner fires on one timestep — applied by column normalization so it
does not scale with `--ac`. Old command lines and notes using `1.2` are not
comparable. Same for `--rec_exc`.

## Debugging order

Run `--diagnose True --trn_eps 0 --tst_eps 0` first — seconds, not hours. GC → AC is
frozen, so the encoding diagnostic measures the *architecture* and gives the same
answer before and after training. Read it in this order:

1. **AC coverage / sparsity** — if AC is silent or saturated, nothing downstream can
   learn and the AC → MC weight plot is uninformative.
2. **AC `nn_ratio`** (want well under 0.6) — spatial distance to each position's
   nearest neighbour in code space, over random-pairing distance. ~0 = smoothly
   position-selective, ~1 = no position information. Alive-but-unselective AC gives
   MSTDPET nothing to bind an action to.
3. **MC readout margin / ceiling fraction** — a ceiling-pinned MC makes WTA random
   even with a perfect AC code.
4. Only then look at `|dW_ac_mc|`.

Reason about the feedforward/recurrent balance before touching gains: **if the
recurrent budget swamps the feedforward drive, no feedforward gain can fix it**, and
the failure presents as an unresponsive knob rather than an error. That was the
original bug.

Each episode prints AC/MC activity, `|dW_ac_mc|`, and the fraction of live synapses
at a bound, so a frozen-weight run says *which* stage failed. A warning fires when
>90% of live synapses sit at a bound.

## BindsNET gotchas confirmed here

- `LIFNodes` does **not** floor membrane voltage unless `lbound` is passed.
  Subtractive inhibition can drive `v` arbitrarily negative and silence a layer for
  a whole episode — `net.reset_state_variables()` runs once per *episode*, not once
  per decision.
- MSTDPET `update()` runs **once per simulation timestep**, i.e. `granularity` (100)
  times per decision. Set learning rates with that multiplier in mind.
- MSTDPET does not respect zeros in a sparse weight matrix; the mask must be
  re-applied after updates (`PR2.py` does this every step) or a 40%-sparse
  projection quietly becomes dense.
- `Weight(range=None)` gives `[-inf, +inf]`, so the learning rule does not clamp;
  bounds are enforced manually in the training loop.

## PR5.py — place-cell location map (`--pc_map True`)

Answers the NOTES.md TODO *"I should be able to tell where the target+agent are
just by looking at the place cell spiking"*. It sweeps a dot over every square of
the 28x28 board, twice, and prints one table row per square: the PC neurons that
fired for it, how repeatable that set is, and how far a held-out presentation
decodes from the true square.

```
python PR5.py --pc_map True            # 784 squares x 2 reps, ~2 min on CPU, then exits
python PR5.py --pc_map True --pc_map_stride 4    # 49 squares, ~10 s, for knob sweeps
```

| Flag | Default | Meaning |
|---|---|---|
| `--pc_map` | False | run the map and **exit before training** |
| `--pc_map_stride` | 1 | 1 = every board square |
| `--pc_map_reps` | 2 | presentations per square; `>=2` is what enables decode + stability |
| `--pc_map_thresh` | 4 | spikes / 100 ms for a cell to count as "fired" |
| `--pc_map_top` | 10 | neuron indices per console row (the CSV always has all) |
| `--pc_map_fields` | 25 | place fields drawn in the montage |
| `--pc_map_layers` | PC_A PC_T | both place layers, measured in one sweep |

Writes `pc_map_<layer>.csv` (square -> every fired index + spike count),
`pc_fields_<layer>.csv` (cell -> field size, centroid, spread),
`pc_map_<layer>.png` (four board maps: cells firing, set stability, decode error,
distance to nearest other code) and `pc_fields_<layer>.png`.

**The second presentation is the point.** GC drive is Poisson, so a code that
looks position-specific on one trial says nothing; decoding trial 2 against
trial 1's templates measures position information the way AC would have to read
it. Numbers to read in order — a later one is meaningless if an earlier one is
broken:

1. **coverage** — squares where any cell clears the threshold. A silent square
   cannot be represented at all.
2. **code size** — cells per square. Above ~30% of the layer it is a population
   rate code, not place cells, and neighbours share most of their set.
3. **set stability** — Jaccard between two presentations of the same square.
   Below ~0.4, set membership is decided by Poisson noise.
4. **decode error** — px between the true square and the nearest template, for
   `rate` (cosine on counts, what AC actually sees) and `set` (Jaccard on the
   fired indices, what the table shows). Both are printed against the chance
   value, ~14.6 px on a 28x28 board.
5. **compactness** — 0 = point-like field, 1.0 = scattered over the board. A code
   can decode perfectly and still not look like place cells.

Two traps this diagnostic is built to avoid, both of which it hit while being
written: a decoder scored on a trial with nothing above threshold picks square 0
by tie-break, which reads as a huge error rather than as no data (each decoder
now reports "trials had nothing to decode" instead); and silent squares are
mutually *indistinguishable*, so they are excluded from the unique-set count
rather than counted as unique. The decoder is validated against synthetic codes —
ideal codes decode to 0 px, position-free codes and deliberately mispaired trials
land at chance, all-silent input is refused.

Only the two GC->PC connections are live during the sweep (restored afterwards),
which is what keeps 1568 network runs to ~2 minutes; the layers, weights and
LIF parameters are the live ones, so this is the same pathway training runs.

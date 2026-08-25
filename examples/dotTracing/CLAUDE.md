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

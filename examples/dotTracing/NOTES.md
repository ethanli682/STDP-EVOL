# PR2.py debugging history

Record of the session that fixed a silent AC layer, the diagnostic that was hiding
it, and a saturated motor readout. Kept out of `CLAUDE.md` so the always-loaded
file stays short — see `CLAUDE.md` for the current defaults and debugging order.

## Symptom

AC → MC weight plots were flat across every run, and the encoding diagnostic always
printed the same thing no matter how high `--ac_gain` was pushed:

```
[Encoding quality]  (paper: GC 9.2 -> AC 16.6)
  GC: active 78.6, overlap 11.1, score 7.1
  AC: active 0.0, overlap 0.0, score inf
  !! AC layer nearly silent -- raise --ac_gain
```

## Root cause 1 — unnormalized recurrent inhibition (the actual bug)

`W_rec = excite - rec_inh * (1 - excite)` was never normalized by fan-in. With
`--rec_radius 4` over a 28-wide view, `excite ≈ 0` for ~84% of pairs, so **every
AC's inhibitory row summed to ≈ −785 mV at `--ac 1000`**, against a feedforward
drive of ~2.5 mV/ms. That is negative feedback ~300× stiffer than the signal, so
`--ac_gain` had no purchase at all. Measured on an isolated probe:

```
W_rec: mean -0.786  frac<0 0.836  row-sum mean -785.5  pos row-sum 95.4

NO-REC gain=1.0  : AC spikes 0       frac>=4 0.000
NO-REC gain=5.0  : AC spikes 4103    frac>=4 0.738
NO-REC gain=25.0 : AC spikes 19216   frac>=4 1.000
REC    gain=25.0 : AC spikes 15806   final v mean -2205 mV, min -4202 mV
REC    gain=100  : AC spikes 16476   final v mean -1950 mV, min -4048 mV
```

Quadrupling the gain moved AC activity by 4%. Without recurrence the layer fires
happily at gain 5.

Compounding it: BindsNET's `LIFNodes` has **no membrane-voltage floor by default**,
so a single synchronous burst drove `v` to ≈ −4000 mV. Because
`net.reset_state_variables()` runs **once per episode, not once per decision**, the
layer then stayed dead for the entire episode. AC never spiked → MSTDPET
eligibility stayed zero → AC → MC frozen. The flat weight plot was a symptom, not
the disease.

Fixes:

- `--rec_exc` / `--rec_inh` are now **total per-neuron budgets in mV**, applied by
  column normalization, so they do not scale with `--ac`. Both `local` and `random`
  modes.
- Added `--ac_lbound` (−80 mV) on the AC `LIFNodes`.
- Fixed a latent crash: `--rec_mode none` hit `W_rec.fill_diagonal_()` on `None`,
  then added the connection unconditionally anyway.

Verification after the fix — the same probe, now stable across 10 consecutive
decisions instead of dying after the first:

```
NORM gain=25 npix=1 steps=1  : AC frac>=4 0.865   final v mean -59.1  min -71.6
NORM gain=25 npix=1 steps=10 : AC frac>=4 0.876   final v mean -59.4  min -71.4
NORM gain=10 npix=1 steps=10 : AC frac>=4 0.060   final v mean -56.4  min -71.0
```

## Root cause 2 — the diagnostic reported silence as perfection

`encodingQuality` was actively misleading. Three separate problems:

- **`score = active / overlap` → `0 / 0 = inf`.** A silent layer scored as *perfect*
  separability. This is why raising `--ac_gain` looked like it did nothing: the
  metric could not distinguish "great code" from "no code."
- **Probe was one pixel of value 1.0.** A real `DotSimulator` view (`decay=4`)
  carries ~2.5 units of mass over 3–4 pixels — measured directly:

  ```
  step0: obs nonzero 4  vals [0. 1.]                  | view nonzero 2  sum 2.00
  step2: obs nonzero 7  vals [0. 0.25 0.5 0.75 1.]    | view nonzero 4  sum 2.50
  ```

  So the probe under-drove GC by ~2.5× and would have condemned working configs.
- **`i % stride` over the flattened, centre-excised index** walks a diagonal at
  `side=29`, sampling position space unevenly.

Rewritten as `probeStimulus` / `codeStats` / `encodingQuality` /
`reportEncodingQuality`:

- Probe stimulus is a prey pixel **plus its decay trail**, matching real view mass.
  Only cells where the full trail fits are probed — edge truncation was previously
  showing up as a fake coverage failure (it capped coverage at 0.88 at an otherwise
  healthy operating point, which made the verdict recommend raising a gain that was
  already correct).
- Probe positions sampled on a proper 2-D lattice (`--diag_stride`).
- `sep` (the paper's `active/overlap`) is suppressed as `--` unless the layer
  actually fires, and verdict order is **liveness → sparsity → coverage →
  selectivity**, so silence can never read as healthy.
- Added threshold-free, bounded metrics that cannot degenerate:
  - `cos` — mean pairwise cosine similarity between position codes (0 distinct,
    1 identical).
  - **`nn_ratio`** — spatial distance from each position to its nearest neighbour
    *in code space*, over the distance under random pairing. ~0 = smoothly
    position-selective, ~1 = no position information. This is the number the
    AC → MC readout actually depends on.
- `net.learning` is forced off during the sweep and restored afterwards, so the
  probe can never touch the weights.

The diagnostic is now responsive to the knob it names:

```
gain=10  AC 0.0%   coverage 0.00   silent
gain=20  AC 0.0%   coverage 0.00   nn 0.90
gain=40  AC 3.8%   coverage 1.00   sep 7.7   nn 0.42
gain=45  AC 9.6%   coverage 1.00   sep 4.3   nn 0.45   <- default (GC sep 3.0)
gain=50  AC 18.2%  coverage 1.00   sep 2.9   nn 0.40
gain=80  AC 67.2%  coverage 1.00   sep 1.4   nn 0.31   saturated
```

The transition is sharp — 40 → 80 goes 3.8% → 67% — so this needs the fine sweep,
not coarse doubling.

## Root cause 3 — saturated motor readout (found while verifying the above)

Once AC was alive, `MC active 100.0%` at 27.8 spikes/cell. WTA compares the argmax
over subpopulation spike counts, so at ceiling all actions tie and selection is
random regardless of what AC → MC learned.

My first estimate of the right weight scale was ~5× too high because it counted
only the *above-threshold* AC cells; the sub-threshold cells firing 1–3 spikes each
dominate the total drive (AC mean rate 1.96 spikes/cell × 1000 cells = ~1960 AC
spikes per window reaching MC through a 40% mask).

Separately, `nu=5e-2` fires MSTDPET **`--granularity` (100) times per decision**,
which pins every weight at `w_max` within one episode — indistinguishable from
"weights not changing" on a weight plot.

Fixes: `--ac_mc_init` replaced a hardcoded `6.0`; retuned `ac_mc_init 6.0 → 0.3`,
`w_max 12 → 1.0`, `nu 5e-2 → 4e-3`. The diagnostic now reports MC **WTA margin**,
tie fraction, distinct winners, and fraction of the firing ceiling — the
code-overlap metrics are the wrong lens for MC, since what matters is whether
subpopulation counts separate.

Note the first version of the MC verdict was itself wrong: it flagged
`frac_firing > 0.95` as "saturated," which fires at 2.4 spikes/cell where the
layer is in fact perfectly graded. Replaced with the ceiling fraction and WTA
margin.

## Result

```
Episode 0: reward -9.22 | AC 5.5%,  MC 64.0%  | |dW| 6.66e-03 max 1.01e-01, 1% pinned
Episode 1: reward  2.80 | AC 10.1%, MC 99.9%  | |dW| 6.73e-03 max 1.11e-01, 1% pinned
Episode 2: reward  7.05 | AC 13.1%, MC 100.0% | |dW| 8.08e-03 max 1.50e-01, 1% pinned
```

Verification was only 3 episodes × 40 steps. Reward trending −9.2 → 2.8 → 7.1 is
**not** evidence of learning, only evidence the network is no longer frozen.

## Ablation: is the recurrence doing anything? (2026-08-25)

Compared at **matched sparsity** (~9% AC active), so the comparison is not
confounded by gain — `none` needs a lower gain to reach the same sparsity, which is
itself the check that inhibition is live rather than vestigial:

| `--rec_mode` | gain | AC sparsity | sep | cos ↓ | nn_ratio ↓ |
|---|---|---|---|---|---|
| `none`   | 38 | 5.2% | 5.1 | 0.879 | 0.633 |
| `local`  | 42 | 5.8% | 5.9 | 0.810 | **0.410** |
| `none`   | 40 | 9.1% | 3.9 | 0.898 | 0.539 |
| `random` | 45 | 8.1% | 4.0 | 0.850 | 0.505 |
| `local`  | 45 | 9.6% | 4.3 | 0.837 | **0.445** |
| `none`   | 42 | 14.6% | 3.0 | 0.912 | 0.521 |

Two matched pairs, and **the recurrence's benefit grows as the code gets sparser**:

| sparsity | `none` | `local` | relative |
|---|---|---|---|
| ~5–6% | 0.633 | 0.410 | **35% better** |
| ~9%   | 0.539 | 0.445 | 17% better |

Supported: the recurrence is functional, the budgets sit in a live regime (`none`
needs a lower gain for the same sparsity), and `local < random < none` on `nn_ratio`
is the desired ordering. `local` also wins on `sep` and `cos` at both operating
points, so this is not a single-metric artifact.

The sparsity dependence is *consistent with* the docstring's continuous-attractor
claim — a bump needs sparse competition to form, so the effect should grow as the
code thins, which is what happens. Still not proof: one seed, 56 probe positions, no
error bars, and `local` beats `random` by only 0.505 → 0.445 at ~9%.

Discriminating test, not yet run: sweep `--rec_radius 2/4/8` **at the sparse
operating point (gain ~42), not at 45** — the ~9% point compresses the effect. A
genuine attractor should degrade sharply as the radius moves away from the AC
place-field spacing; flat `nn_ratio` means generic sparsification instead.

## Open issues (not fixed — deliberate)

- **AC separability 4.3 vs GC 3.0** — right direction, far from the paper's
  9.2 → 16.6. Their absolute numbers depend on `n_ac=2000` and their own probe
  stimulus, so the figures are not directly comparable.
- **MC drifts 64% → 100% active over three episodes** while AC climbs 5.5% → 13.1%.
  Growing weights push MC toward ceiling even at `w_max=1.0`. If the post-training
  `margin` stays near 0.007 while `|dW|` keeps growing, the readout is saturating
  rather than differentiating — the fix is **MC lateral inhibition or
  per-subpopulation weight normalization**, not another `w_max` reduction. That is a
  design change and was left alone.
- AC activity in the real loop (5.5–13.1%) runs below the trail-probe's 9.6%
  prediction early in an episode, so the probe is a slightly optimistic proxy for
  live views.

## Things checked and found NOT to be bugs

Recorded so nobody re-investigates:

- `Monitor(time=N)` is a fixed-size FIFO, so repeated `net.run(time=N)` without a
  reset correctly yields the last `N` steps. The WTA readout is not accumulating
  whole-episode history.
- `Weight(range=None)` resolves to `[-inf, +inf]`, so `MCC_LearningRule.update` does
  not silently clamp weights to `[-1, 1]`. Bounds are enforced manually in the
  training loop instead.
- `net.learning = False` does propagate to `MulticompartmentConnection.update`, via
  `Network.run(learning=self.learning)`.
- `MulticompartmentConnection` uses `source.n` / `target.n` generically, so the 5-D
  `Input(shape=[1,1,1,1,n_gc])` vs 2-D `LIFNodes` shape mismatch is harmless.

## Reproducing the isolation probes

The scratch scripts that produced the numbers above (`probe.py` building GC → AC
with and without recurrence, `obsprobe.py` measuring real view mass) were session
temporaries and are not checked in. Both are ~60 lines and re-derive the GC field
construction from `PR2.main()`; the fastest route to rebuilding one is to copy the
`gc_fields` loop out of `main()` and drive `gcRates` directly.

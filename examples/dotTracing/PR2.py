"""
Dot-tracing SNN restructured along the DeltaQ paper's GC -> AC -> MC architecture
(Earl et al., bioRxiv 2026)

Mapping from the paper to this task
-----------------------------------
  Grid Cells (560)        -> GC layer: multi-scale periodic Gaussian basis applied
                             as a FIXED projection over the egocentric pixel view.
                             Prime scales, 7 rotations, 16 phase offsets = 560.
  Association Cells (2000)-> AC layer: LIF, sparse (12%) fixed random projection
                             from GC, coincidence-detection tuning -> sparse,
                             place-field-like responses over relative prey position.
  Motor Cells (400)       -> MC layer: one subpopulation per action, WTA on
                             aggregate subpopulation spike count.
  AC->MC plastic only     -> MSTDPET on AC->MC only. GC->AC frozen. 

Rate convention: the paper's "8 Hz" over a 1000 ms decision window means 8 spikes
per decision. This script matches SPIKES PER DECISION, not Hz, so gc_max_rate
defaults to 80 Hz over the 100 ms granularity window.

Tuning the AC layer
-------------------
Only two knobs set AC activity, and they fight each other:

  --ac_gain   scales the frozen GC->AC feedforward weights (mV per presyn spike,
              after 1/sqrt(fan-in) normalisation).
  --rec_exc / --rec_inh
              TOTAL per-neuron recurrent budgets in mV -- the drive a cell would
              get if every excitatory (resp. inhibitory) partner fired on one
              timestep. Column-normalised, so they do NOT scale with --ac. This
              matters: with the raw Mexican hat, every AC's inhibitory row summed
              to about -790 mV at --ac 1000, roughly 300x the feedforward drive.
              That is a negative-feedback loop stiff enough that --ac_gain has no
              measurable effect -- exactly the failure mode of "keep raising
              ac_gain and nothing happens".

The AC LIF also has --ac_lbound. Recurrent inhibition here is subtractive and
BindsNET does not floor the membrane voltage by default, so without it one
synchronous burst drives v to -1000s of mV; since reset_state_variables() runs
once per episode rather than once per decision, the layer then stays dead for the
whole episode and the plastic AC->MC weights never move.

Target operating point: ~10% of AC active per decision, coverage 1.00, nn_ratio
well below 0.6. --diagnose prints all of these before training starts.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import itertools
import os
import time

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bindsnet.analysis.plotting import plot_spikes
from bindsnet.encoding import poisson
from bindsnet.environment.dot_simulator import DotSimulator
from bindsnet.learning.MCC_learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
# task / env
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--trn_eps", type=int, default=2000)
parser.add_argument("--tst_eps", type=int, default=1000)
parser.add_argument("--decay", type=int, default=4)
parser.add_argument("--herrs", type=int, default=0)
parser.add_argument("--diag", type=bool, default=False)
parser.add_argument("--randr", type=float, default=0.15)
parser.add_argument("--boundh", type=str, default="bounce")
parser.add_argument("--fit_func", type=str, default="dir")
parser.add_argument("--allow_stay", type=bool, default=False)
parser.add_argument("--pandas", type=bool, default=False)
parser.add_argument("--mute", type=bool, default=False)
parser.add_argument("--write", type=bool, default=False)
parser.add_argument("--fcycle", type=int, default=100)
parser.add_argument("--gpu", type=bool, default=True)
parser.add_argument("--view_r", type=int, default=14)

# rendering (now actually gated -- the old script captured every frame of every episode)
parser.add_argument("--render_replays", type=bool, default=True)
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--plot_every", type=int, default=50)

# --- GC layer (paper: 5 scales x 7 rotations x 16 offsets = 560) ---
parser.add_argument("--gc_scales", type=int, nargs="+", default=[5, 7, 11, 13, 17])
parser.add_argument("--gc_rotations", type=int, default=7)
parser.add_argument("--gc_offsets", type=int, default=16)   # 4x4 phase grid
parser.add_argument("--gc_global_scale", type=float, default=1.5)  # paper's g
parser.add_argument("--gc_sharpness", type=float, default=1.0)     # paper's k
parser.add_argument("--gc_max_rate", type=float, default=80.0)     # 8 spikes / 100 ms

# --- AC layer (paper: 2000 neurons, 12% GC->AC sparsity) ---
parser.add_argument("--ac", type=int, default=1000)   # paper uses 2000; 1000 is faster
parser.add_argument("--gc_ac_sparsity", type=float, default=0.12)
parser.add_argument("--ac_gain", type=float, default=45.0)   # TUNE ME (see notes)
parser.add_argument("--ac_lbound", type=float, default=-80.0)  # v floor; see notes
parser.add_argument("--rec_mode", type=str, default="local",
                    choices=["none", "random", "local"])
# rec_exc / rec_inh are TOTAL per-neuron budgets in mV: the drive a neuron would
# receive if every one of its excitatory (resp. inhibitory) presynaptic partners
# fired on the same timestep. Column-normalised, so they do not scale with --ac.
parser.add_argument("--rec_exc", type=float, default=15.0)
parser.add_argument("--rec_inh", type=float, default=25.0)
parser.add_argument("--rec_radius", type=float, default=4.0)
parser.add_argument("--ac_target_sparsity", type=float, default=0.10)

# --- MC layer (paper: 4 x 100 = 400) ---
parser.add_argument("--mc_pop", type=int, default=80)
parser.add_argument("--ac_mc_sparsity", type=float, default=0.40)
parser.add_argument("--ac_mc_init", type=float, default=0.3)  # U(0, init) per synapse
parser.add_argument("--w_max", type=float, default=1.0)
# nu must be scaled with --w_max: MSTDPET updates once per SIMULATION step, i.e.
# --granularity times per decision, so a large nu pins every weight at the cap
# within one episode -- which looks exactly like "weights not changing".
parser.add_argument("--nu", type=float, default=4e-3)

# --- action selection (paper: WTA + epsilon-greedy) ---
parser.add_argument("--select", type=str, default="wta", choices=["wta", "softmax"])
parser.add_argument("--eps_start", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.99)   # per step, as in paper
parser.add_argument("--eps_min", type=float, default=0.05)

parser.add_argument("--diagnose", type=bool, default=True)
parser.add_argument("--diag_stride", type=int, default=4)   # probe positions
parser.add_argument("--diag_thresh", type=int, default=4)   # spikes -> "active"

args = parser.parse_args()

# fixed movement choice count
moveChoices = 9 if args.diag else 5 if args.allow_stay else 4
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
OUT_FILE_PATH = "PR2_RUN/"

LAYER_GC, LAYER_AC, LAYER_MC = "GC", "AC", "MC"

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def getLocalView(obs, row, col, view_r):
    """Flattened (2r+1)^2 - 1 egocentric patch, agent's own cell excluded."""
    padded = np.pad(obs, view_r, mode="constant", constant_values=0)
    patch = padded[row: row + 2 * view_r + 1, col: col + 2 * view_r + 1]
    flat = patch.flatten()
    center = view_r * (2 * view_r + 1) + view_r
    return np.concatenate([flat[:center], flat[center + 1:]])


def gcRates(view_vec, W_grid, peak, max_rate):
    """view (inpt_n,) -> grid-cell firing rates (n_gc,). Paper: f = p/p_max * f_max."""
    drive = view_vec @ W_grid                            # (n_gc,)
    drive = (drive / peak.squeeze(0)).clamp(0.0, 1.0)
    drive[drive < 0.01] = 0.0
    return drive * max_rate

def probeStimulus(inpt_n, side, centre_flat, cell, trail=(1.0, 0.75, 0.5, 0.25)):
    """
    One probe view: a prey pixel at `cell` plus the decay trail behind it.

    The old diagnostic used a single pixel of value 1.0. A real DotSimulator view
    (decay=4) carries ~2.5 units of mass over 3-4 pixels, so the single pixel
    under-drove GC by roughly 2.5x and the test declared the AC layer dead at
    gains where it is in fact fine. This reproduces the real stimulus instead.
    """
    grid = np.zeros((side, side), dtype=np.float32)
    r0, c0 = divmod(cell, side)
    for k, amp in enumerate(trail):
        r, c = r0, c0 - k                      # trail runs left; direction is arbitrary
        if 0 <= r < side and 0 <= c < side and (r * side + c) != centre_flat:
            grid[r, c] = amp
    flat = grid.flatten()
    return np.concatenate([flat[:centre_flat], flat[centre_flat + 1:]])


def codeStats(R, pos, thresh):
    """
    Summarise a population code R (n_positions, n_cells) of spike counts.

    Three families of number, because no single one is trustworthy alone:

      active / coverage   -- is the layer alive, and at what sparsity?
      overlap / sep       -- the paper's binary set measure (9.2 GC -> 16.6 AC).
                             `sep` is active/overlap, which is why the old code
                             printed `inf` for a SILENT layer: 0/0. Silence is the
                             worst possible code, not the best, so sep is reported
                             as None when the layer does not fire.
      cos / nn_ratio      -- graded, threshold-free, bounded, so they never
                             degenerate. cos is mean pairwise cosine similarity
                             between positions (0 distinct, 1 identical).
                             nn_ratio is the mean spatial distance from a position
                             to its nearest neighbour IN CODE SPACE, over the
                             distance under random pairing: ~0 means the code is
                             smoothly position-selective, ~1 means it carries no
                             position information. This is the number to watch --
                             it is what the AC->MC readout actually needs.
    """
    P, n = R.shape
    eye = torch.eye(P, dtype=torch.bool, device=R.device)

    M = (R >= thresh).float()
    active = M.sum(1)
    coverage = (active > 0).float().mean().item()
    ov = (M @ M.T)[~eye]
    a, o = active.mean().item(), ov.mean().item()

    live = R.sum(1) > 0
    if live.sum() >= 2:
        Rn = R[live] / R[live].norm(dim=1, keepdim=True).clamp(min=1e-9)
        S = Rn @ Rn.T
        Pl = int(live.sum())
        eyel = torch.eye(Pl, dtype=torch.bool, device=R.device)
        cos = S[~eyel].mean().item()
        D = torch.cdist(pos[live], pos[live])
        nn = S.masked_fill(eyel, -2.0).argmax(1)
        nn_d = D[torch.arange(Pl, device=R.device), nn].mean().item()
        rand_d = D[~eyel].mean().item()
        nn_ratio = nn_d / rand_d if rand_d > 0 else float("nan")
    else:
        cos, nn_ratio = float("nan"), float("nan")

    return dict(
        avg_active=a, sparsity=a / n, coverage=coverage,
        avg_overlap=o, max_off=ov.max().item() if ov.numel() else 0.0,
        sep=(a / o if (o > 0 and coverage > 0.5) else None),
        cos=cos, nn_ratio=nn_ratio,
        mean_rate=R.mean().item(), max_rate=R.max().item(),
        frac_firing=(R > 0).float().mean().item(),
    )


@torch.no_grad()
def encodingQuality(net, W_grid, peak, view_r, gran, dt, max_rate,
                    spike_thresh=None, stride=None):
    """
    Sweep a realistic prey stimulus over the egocentric view and measure how well
    GC and AC separate positions. This probes the ARCHITECTURE only -- GC->AC is
    frozen, so the answer is the same before and after training. Run it before
    burning hours on a training job: if AC cannot separate positions, the plastic
    AC->MC weights have nothing to learn from and will sit still.
    """
    spike_thresh = args.diag_thresh if spike_thresh is None else spike_thresh
    stride = args.diag_stride if stride is None else stride

    side = 2 * view_r + 1
    inpt_n = side * side - 1
    centre_flat = view_r * side + view_r

    # Sample cells on a coarse 2-D lattice. The old code used `i % stride` over
    # the *flattened, centre-excised* index, which with side=29 walks a diagonal
    # and samples position space unevenly.
    trail_len = 4
    cells, positions = [], []
    for r in range(0, side, max(1, stride)):
        # start far enough in that the whole trail fits: a truncated trail is a
        # weaker stimulus, and it used to show up as a spurious coverage failure
        for c in range(trail_len - 1, side, max(1, stride)):
            cell = r * side + c
            if cell == centre_flat:
                continue
            cells.append(cell)
            positions.append((r - view_r, c - view_r))
    pos = torch.tensor(positions, dtype=torch.float32, device=DEVICE)

    was_learning = net.learning
    net.learning = False                 # never let the probe touch the weights

    gc_R, ac_R, mc_R = [], [], []
    for cell in cells:
        v = torch.from_numpy(
            probeStimulus(inpt_n, side, centre_flat, cell)
        ).to(DEVICE)
        rates = gcRates(v, W_grid, peak, max_rate)
        inputs = {LAYER_GC: poisson(rates.unsqueeze(0), gran, dt, device=DEVICE)}
        net.reset_state_variables()
        net.run(inputs=inputs, time=gran, reward=0.0)
        gc_R.append(net.monitors[LAYER_GC].get("s").squeeze().sum(0).float())
        ac_R.append(net.monitors[LAYER_AC].get("s").squeeze().sum(0).float())
        mc_R.append(net.monitors[LAYER_MC].get("s").squeeze().sum(0).float())

    net.reset_state_variables()
    net.learning = was_learning

    # MC is included because a dead or ceiling-pinned motor layer is just as fatal
    # as a silent association layer, and the numbers that matter for MC are not
    # the code-overlap ones: WTA reads the ARGMAX over subpopulation spike counts,
    # so what decides whether action selection works is the MARGIN between
    # subpopulations and whether that argmax actually varies with position.
    MC = torch.stack(mc_R)                                # (P, n_mc)
    sub = MC.view(MC.shape[0], moveChoices, -1).sum(2)    # (P, n_actions)
    top2 = sub.topk(2, dim=1).values
    denom = sub.mean(1).clamp(min=1e-9)
    mc_extra = dict(
        wta_margin=((top2[:, 0] - top2[:, 1]) / denom).mean().item(),
        tie_frac=(top2[:, 0] == top2[:, 1]).float().mean().item(),
        dead_frac=(sub.sum(1) == 0).float().mean().item(),
        n_winners=len(torch.unique(sub.argmax(1))),
        ceiling=MC.max().item() / gran,
    )

    return {"GC": codeStats(torch.stack(gc_R), pos, spike_thresh),
            "AC": codeStats(torch.stack(ac_R), pos, spike_thresh),
            "MC": {**codeStats(MC, pos, spike_thresh), **mc_extra},
            "_n_probe": len(cells)}


def reportEncodingQuality(q, title="Encoding quality"):
    """Print the diagnostic and, critically, name the right knob when it fails."""
    print(f"\n[{title}]  {q['_n_probe']} probe positions, "
          f"active = >={args.diag_thresh} spikes / {args.granularity} ms",
          flush=True)
    print("  layer   active  sparsity  coverage  overlap    sep     cos  nn_ratio"
          "   spikes/cell", flush=True)
    for k in ("GC", "AC", "MC"):
        v = q[k]
        sep = "   --  " if v["sep"] is None else f"{v['sep']:7.1f}"
        print(f"  {k:5s}  {v['avg_active']:7.1f}  {v['sparsity']:8.3f}  "
              f"{v['coverage']:8.2f}  {v['avg_overlap']:7.1f}  {sep}  "
              f"{v['cos']:6.3f}  {v['nn_ratio']:8.3f}  {v['mean_rate']:11.2f}",
              flush=True)
    print("  (paper's sep: GC 9.2 -> AC 16.6.  nn_ratio: lower is better, "
          "~1.0 means no position info.)", flush=True)

    ac = q["AC"]
    lo, hi = 0.25 * args.ac_target_sparsity, 3.0 * args.ac_target_sparsity
    if ac["frac_firing"] < 0.01:
        print("  !! AC silent. Raise --ac_gain, or lower --rec_inh -- and check "
              "that the recurrent budget is not swamping the feedforward drive.",
              flush=True)
    elif ac["sparsity"] > hi:
        print(f"  !! AC saturated ({ac['sparsity']:.0%} active vs target "
              f"{args.ac_target_sparsity:.0%}). Lower --ac_gain or raise "
              f"--rec_inh. Dense codes overlap, so AC->MC cannot bind an action "
              f"to a position.", flush=True)
    elif ac["sparsity"] < lo:
        print(f"  !! AC too sparse ({ac['sparsity']:.1%} vs target "
              f"{args.ac_target_sparsity:.0%}). Raise --ac_gain.", flush=True)
    elif ac["coverage"] < 0.9:
        print(f"  !! AC is at the right sparsity overall, but only "
              f"{ac['coverage']:.0%} of positions have ANY cell clearing "
              f"{args.diag_thresh} spikes -- some part of the view is a blind "
              f"spot. Raise --ac_gain slightly, or lower --diag_thresh if "
              f"{args.granularity} ms is simply too short a window.", flush=True)
    elif not (ac["nn_ratio"] < 0.6):
        print(f"  !! AC alive at the right sparsity but nn_ratio="
              f"{ac['nn_ratio']:.2f}: the code is not position-selective. Try a "
              f"smaller --rec_radius, or --rec_mode none to check whether the "
              f"recurrence is what is smearing it.", flush=True)
    else:
        print(f"  OK: AC {ac['sparsity']:.1%} active, coverage "
              f"{ac['coverage']:.0%}, nn_ratio {ac['nn_ratio']:.2f}.", flush=True)

    if q["AC"]["sep"] is not None and q["GC"]["sep"] is not None \
            and q["AC"]["sep"] < q["GC"]["sep"]:
        print("  note: AC separability is BELOW GC -- the association layer is "
              "losing information the grid code already had.", flush=True)

    mc = q["MC"]
    print(f"  MC readout: {mc['mean_rate']:.2f} spikes/cell "
          f"({mc['ceiling']:.0%} of the {args.granularity}-step ceiling at peak), "
          f"WTA margin {mc['wta_margin']:.3f}, ties {mc['tie_frac']:.0%}, "
          f"{mc['n_winners']}/{moveChoices} distinct winners over "
          f"{q['_n_probe']} positions", flush=True)
    if mc["dead_frac"] > 0.1:
        print(f"  !! MC silent at {mc['dead_frac']:.0%} of positions -- WTA falls "
              f"back to a random action there. Raise --ac_mc_init.", flush=True)
    elif mc["ceiling"] > 0.5:
        print(f"  !! MC pinned near its firing ceiling. Subpopulation counts "
              f"cannot separate, so WTA is close to random. Lower --ac_mc_init "
              f"AND --w_max -- w_max alone matters because MSTDPET will otherwise "
              f"push every weight back up to the cap, flattening the readout "
              f"again.", flush=True)
    elif mc["tie_frac"] > 0.5 or mc["wta_margin"] < 0.05:
        print(f"  !! MC subpopulations barely differ (margin "
              f"{mc['wta_margin']:.3f}, ties {mc['tie_frac']:.0%}). The readout "
              f"has no resolution yet; expected before training if AC->MC starts "
              f"uniform, but it must grow as |dW_ac_mc| grows.", flush=True)
    elif mc["n_winners"] < 2:
        print(f"  !! MC always picks the same action regardless of position -- "
              f"check the AC->MC mask is not concentrating on one subpopulation.",
              flush=True)


# --------------------------------------------------------------------------- #
# Plotting / replay helpers (unchanged in spirit, now properly gated)
# --------------------------------------------------------------------------- #
def plotWeightGraphs(weight_features, ep, out_path=OUT_FILE_PATH):
    os.makedirs(out_path, exist_ok=True)
    names = list(weight_features.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        data = weight_features[name].value.detach().cpu().numpy()
        vabs = max(abs(data.min()), abs(data.max())) or 1.0
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("target neuron")
        ax.set_ylabel("source neuron")
        div = make_axes_locatable(ax)
        plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
    fig.suptitle(f"Synaptic Weights — Episode {ep}")
    fig.tight_layout()
    fname = os.path.join(out_path, f"weights_ep{ep:05d}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def plotMotorWeightPolar(w_ac_mc, active_mask, ep, out_path=OUT_FILE_PATH):
    """
    The paper's Figure 9: fraction of total AC->MC weight allocated to each motor
    population, split by active vs inactive ACs at the current state.
    """
    os.makedirs(out_path, exist_ok=True)
    W = w_ac_mc.detach().cpu().numpy()
    pops = np.array_split(np.arange(W.shape[1]), moveChoices)
    act = active_mask.detach().cpu().numpy().astype(bool)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ang = np.linspace(0, 2 * np.pi, moveChoices, endpoint=False)
    for label, sel, colour in (("active", act, "r"), ("inactive", ~act, "b")):
        if sel.sum() == 0:
            continue
        tot = W[sel].sum()
        vals = np.array([W[sel][:, p].sum() for p in pops]) / (tot or 1.0)
        ax.plot(np.append(ang, ang[0]), np.append(vals, vals[0]), colour, label=label)
    ax.set_xticks(ang)
    ax.set_xticklabels([f"a{i}" for i in range(moveChoices)])
    ax.legend(loc="upper right")
    ax.set_title(f"AC→MC weight allocation — ep {ep}")
    fname = os.path.join(out_path, f"motorpolar_ep{ep:05d}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def captureFrame(fig=None):
    fig = fig or plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


def saveReplayGIF(frames, fname, fps=20):
    if not frames:
        return None
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(fname, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return fname


def genFileName(ftype, suffix=""):
    t = time.time()
    t = int(1e10 * (t - 1e6 * (t // 1e6)))
    return OUT_FILE_PATH + ftype + "_s" + str(t) + "_" + suffix + ".csv"


# --------------------------------------------------------------------------- #
# Action selection -- paper's WTA over motor subpopulations
# --------------------------------------------------------------------------- #
def wtaAction(mc_spikes, n_actions, eps, rng):
    """mc_spikes: (time, n_mc) -> action index. Aggregate count per subpopulation."""
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    counts = mc_spikes.sum(0)
    pops = torch.chunk(counts, n_actions)
    agg = torch.stack([p.sum() for p in pops])
    if agg.max() == 0:
        return int(rng.integers(0, n_actions))
    return int(torch.argmax(agg).item())


def softmaxAction(mc_spikes, n_actions):
    counts = mc_spikes.sum(0)
    agg = torch.stack([p.sum() for p in torch.chunk(counts, n_actions)])
    return int(torch.multinomial(torch.softmax(agg, dim=0), 1).item())


# --------------------------------------------------------------------------- #
# Training / evaluation loop
# --------------------------------------------------------------------------- #
def runSimulator(net, env, spikes, episodes, W_grid, peak, ac_mc_mask,
                 feat_ac_mc, gran=100, rfname="", pfname="",
                 weight_features=None, render_replays=False, render_every=25,
                 replay_fps=20, replay_prefix="replay", view_r=14,
                 learning=True, eps_start=1.0):
    dt = net.dt
    rng = np.random.default_rng(args.seed)
    spike_ims = spike_axes = None
    eps = eps_start if learning else 0.0

    for ep in range(episodes):
        total_reward, intercepts, step = 0.0, 0, 0
        rewards = np.zeros(env.timesteps)
        net.reset_state_variables()
        env.reset()
        done = False

        capturing = render_replays and (ep % render_every == 0)
        replay_frames = []
        if capturing:
            env.render()
            replay_frames.append(captureFrame())

        action = int(rng.integers(0, env.action_space.n))
        last_active_ac = torch.zeros(net.layers[LAYER_AC].n, device=DEVICE)
        clock = time.time()
        # Per-episode health counters: if ac_frac is ~0 the plastic AC->MC weights
        # cannot move, so w_delta being ~0 is a symptom, not the disease.
        w0 = feat_ac_mc.value.detach().clone()
        ac_frac_sum, mc_frac_sum = 0.0, 0.0

        while not done:
            step += 1

            # TODO: make reward more sparse

            # get prev row+col and calc dist
            pr, pc = env.netDot.row[0], env.netDot.col[0]
            prev_dist = np.hypot(pr - env.dots[0].row[0], pc - env.dots[0].col[0])

            # take a step
            obs, _, done, intercept = env.step(action)


            net_obs = obs.copy()
            net_obs[env.netDot.row, env.netDot.col] = 0.0

            # get the agent's centered view
            view = torch.Tensor(
                getLocalView(net_obs, env.netDot.row[0], env.netDot.col[0], view_r)
            ).to(DEVICE)

            # agent row+col
            cr, cc = env.netDot.row[0], env.netDot.col[0]
            # target row+col
            tr, tc = env.dots[0].row[0], env.dots[0].col[0]

            # calc change in dist
            curr_dist = np.hypot(cr - tr, cc - tc)
            delta = prev_dist - curr_dist

            # calculate alignment based on dot product
            dr_p, dc_p = tr - pr, tc - pc
            dr_m, dc_m = cr - pr, cc - pc
            dmag, mmag = np.hypot(dr_p, dc_p), np.hypot(dr_m, dc_m)

            alignment = 0.0 if (mmag < 1e-8 or dmag < 1e-8) else \
                float(np.clip((dr_m * dr_p + dc_m * dc_p) / (dmag * mmag), -1, 1))
            scale = 0.5 + 0.5 * alignment
            r = delta * ((0.5 + scale) if delta >= 0 else (1.5 - scale))
            r += 0.2 * alignment

            if intercept:
                r += 20.0
            reward = torch.tensor(r, dtype=torch.float32, device=DEVICE)

    #         def gcRates(view_vec, W_grid, peak, max_rate):
    # """view (inpt_n,) -> grid-cell firing rates (n_gc,). Paper: f = p/p_max * f_max."""
    # drive = view_vec @ W_grid                            # (n_gc,)
    # drive = (drive / peak.squeeze(0)).clamp(0.0, 1.0)
    # drive[drive < 0.01] = 0.0
    # return drive * max_rate

            # ---- GC encoding -> network --------------------------------------
            rates = gcRates(view, W_grid, peak, args.gc_max_rate)
            inputs = {LAYER_GC: poisson(rates.unsqueeze(0), gran, dt, device=DEVICE)}

            # run network with gridcell firing as input to get a move choice
            net.run(inputs=inputs, time=gran, reward=reward)

            # TODO: ADD MASK OBJ
            if learning:
                with torch.no_grad():
                    feat_ac_mc.value.mul_(ac_mc_mask).clamp_(0.0, args.w_max)

            mc = spikes[LAYER_MC].get("s").squeeze()          # (time, n_mc)
            ac = spikes[LAYER_AC].get("s").squeeze()
            last_active_ac = (ac.sum(0) >= 4).float()
            ac_frac_sum += last_active_ac.mean().item()
            mc_frac_sum += (mc.sum(0) > 0).float().mean().item()

            action = (wtaAction(mc, env.action_space.n, eps, rng)
                      if args.select == "wta"
                      else softmaxAction(mc, env.action_space.n))
            if learning:
                eps = max(args.eps_min, eps * args.eps_decay)

            rewards[step - 1] = r
            total_reward += r
            intercepts += int(bool(intercept))

            if capturing:
                env.render()
                replay_frames.append(captureFrame())

            if args.plot_every and ep % args.plot_every == 0 and step == 1:
                spikes_ = {l: spikes[l].get("s").view(gran, -1) for l in spikes}
                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims,
                                                    axes=spike_axes)

            if step % 25 == 0:
                print(f"  step {step} ({time.time() - clock:.2f}s) "
                      f"r={r:+.3f} eps={eps:.3f}")
                clock = time.time()

        if net.reward_fn is not None:
            net.reward_fn.update(accumulated_reward=total_reward, steps=step)

        w_now = feat_ac_mc.value.detach()
        dw = (w_now - w0).abs()
        live = ac_mc_mask.bool()
        pinned = ((w_now[live] >= args.w_max - 1e-6) |
                  (w_now[live] <= 1e-6)).float().mean().item()
        print(f"Episode {ep}: total reward {total_reward:.2f}, "
              f"intercepts {intercepts}, eps {eps:.3f} | "
              f"AC active {ac_frac_sum / max(1, step):.1%}, "
              f"MC active {mc_frac_sum / max(1, step):.1%} | "
              f"|dW_ac_mc| mean {dw.mean():.2e} max {dw.max():.2e}, "
              f"{pinned:.0%} of live synapses at a bound")
        if pinned > 0.9:
            print("  !! AC->MC is pinned at its bounds: lower --nu or raise "
                  "--w_max. A saturated weight matrix cannot express a policy "
                  "and will look frozen on the weight plots.")
        os.makedirs(OUT_FILE_PATH, exist_ok=True)

        if capturing and replay_frames:
            saveReplayGIF(replay_frames,
                          os.path.join(OUT_FILE_PATH,
                                       f"{replay_prefix}_ep{ep:05d}.gif"),
                          fps=replay_fps)

        if weight_features is not None and ep % render_every == 0:
            plotWeightGraphs(weight_features, ep)
            plotMotorWeightPolar(feat_ac_mc.value, last_active_ac, ep)

        if args.write and rfname:
            with open(rfname, "ab") as f:
                np.savetxt(f, rewards, delimiter=",", fmt="%.6f")
        if pfname:
            with open(pfname, "a+") as f:
                f.write(("," if ep else "") + str(intercepts))
        if ep % args.fcycle == 0:
            env.cycleOutFiles()

    return eps

# --------------------------------------------------------------------------- #
def main():
    # egocentric view setup
    view_r = args.view_r
    side_v = 2 * view_r + 1                      # window is side_v x side_v
    inpt_n = side_v * side_v - 1                 # minus the excluded centre
    centre_flat = view_r * side_v + view_r       # index of the agent's own cell (middle idx)

    # remove the agent pixel and store relative locations of each neuron to agent
    view_offsets = []
    for k in range(inpt_n):
        # re-insert the gap left by the excised centre cell, then un-flatten
        cell = k if k < centre_flat else k + 1
        pr, pc = divmod(cell, side_v)
        view_offsets.append((pr - view_r, pc - view_r))
    view_offsets = np.array(view_offsets, dtype=np.float32)   # (inpt_n, 2)
 

    # grid cell layer
    extent = float(view_r)      # lattice must cover the window
    phase_side = int(round(np.sqrt(args.gc_offsets)))   # 16 phases -> 4x4 grid
 
    gc_fields = []        # one (inpt_n,) receptive field per grid cell
    gc_meta = []          # (scale, rotation, phase_a, phase_b) bookkeeping
 
    for s in args.gc_scales:
        # s * 1.0
        spacing = s * args.gc_global_scale   # distance between lattice peaks
        d = (1.0 / args.gc_sharpness) * (spacing / 2.0)
        sigma = d / 3.0
 
        # Two basis vectors 60 degrees apart generate a hexagonal lattice.
        b1 = spacing * np.array([1.0, 0.0])
        b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])
 
        # Integer combinations i*b1 + j*b2 give every lattice vertex. Generate
        # enough of them to cover the window even after rotation and shifting.
        n_tile = int(np.ceil(2.0 * extent / spacing)) + 2
        ij = np.array(list(itertools.product(range(-n_tile, n_tile + 1), repeat=2)),
                      dtype=np.float32)
        lattice_base = ij[:, :1] * b1 + ij[:, 1:] * b2       # (n_pts, 2)
 
        for r_i in range(args.gc_rotations):
            theta = np.pi * r_i / max(1, args.gc_rotations)
            c, sn = np.cos(theta), np.sin(theta)
            R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
            lattice_rot = lattice_base @ R.T     # rotate 
 
            for a in range(phase_side):
                for b in range(phase_side):
                    # shift by phase
                    phase = (a / phase_side) * b1 + (b / phase_side) * b2
                    pts = lattice_rot + phase
 
                    # drop vertices far outside the window -- they can never be
                    # the nearest peak to any view offset, so they only cost time
                    keep = (np.abs(pts) <= extent + spacing).all(axis=1)
                    pts = pts[keep]
 
                    # create gaussians for each field
                    d2 = ((view_offsets[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
                    field = np.exp(-d2.min(axis=1) / (2.0 * sigma ** 2))
                    field[field < 0.01] = 0.0        # paper zeroes sub-0.01 tails
 
                    gc_fields.append(field.astype(np.float32))
                    gc_meta.append((s, theta, a, b))
 
    # Stack into the fixed projection matrix: column j = grid cell j's field.
    W_grid_np = np.stack(gc_fields, axis=1)              # (inpt_n, n_gc)
    # p_max per cell: the strongest response any single pixel can evoke. Dividing
    # by it normalises each cell to [0, 1] so one shared --gc_max_rate is
    # meaningful across scales (large-scale cells have broader, taller fields).
    peak_np = W_grid_np.max(axis=0, keepdims=True)
    peak_np[peak_np == 0] = 1.0
 
    W_grid = torch.from_numpy(W_grid_np).to(DEVICE)
    peak = torch.from_numpy(peak_np).to(DEVICE)
    n_gc = W_grid.shape[1]


    # create network
    net = Network(dt=args.dt)
    gc = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)
    # lbound matters: AC->AC inhibition is subtractive and unbounded below, so
    # without a floor one synchronous burst drives v to -1000s of mV and the
    # layer is dead for the rest of the episode (reset_state_variables only runs
    # once per episode, not once per decision).
    ac = LIFNodes(n=args.ac, traces=True,
                  rest=-64.0, reset=-70.0, thresh=-45.0,
                  refrac=1, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.ac_lbound)
    n_mc = moveChoices * args.mc_pop
    mc = LIFNodes(n=n_mc, traces=True,
                  rest=-64.0, reset=-64.0, thresh=-49.0,
                  refrac=0, tc_decay=20.0, tc_trace=20.0)
    net.add_layer(gc, name=LAYER_GC)
    net.add_layer(ac, name=LAYER_AC)
    net.add_layer(mc, name=LAYER_MC)

    # Each AC samples ~12% of the grid cells at random and keeps that sample for
    # the entire run 
    mask_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    gc_ac_mask = (torch.rand(n_gc, args.ac, generator=mask_gen)
                  < args.gc_ac_sparsity).float().to(DEVICE)
 
    # Random positive weights on the surviving synapses (GC input is excitatory).
    W_gc_ac = gc_ac_mask * torch.rand(n_gc, args.ac, device=DEVICE)
 
    # Normalise by sqrt(mean fan-in)
    fan_in = gc_ac_mask.sum(0).mean().clamp(min=1.0)
    W_gc_ac = (W_gc_ac / fan_in.sqrt()) * args.ac_gain * 2.0
 
    feat_gc_ac = Weight(name="w_gc_ac", value=W_gc_ac)
    net.add_connection(
        MulticompartmentConnection(source=gc, target=ac,
                                   pipeline=[feat_gc_ac], device=DEVICE),
        source=LAYER_GC, target=LAYER_AC)
 
    # "local" mode gives that recurrence spatial structure. First we ask where
    # each AC's place field actually sits: its effective receptive field over
    # the view is W_grid @ mask[:, j] (its sampled grid fields summed), and the
    # argmax of that is where several of them coincide -- i.e. its preferred
    # offset. ACs with nearby preferences then excite each other, distant ones
    # inhibit, giving a soft continuous-attractor bump rather than the
    # unstructured noise of a random reservoir.
    effective_rf = W_grid @ gc_ac_mask                   # (inpt_n, n_ac)
    centre_idx = effective_rf.argmax(dim=0).cpu().numpy()
    ac_centres = torch.from_numpy(view_offsets[centre_idx]).float()   # (n_ac, 2)
 
    def normColumns(M, budget):
        """Scale each column so its entries sum to `budget` (a per-neuron mV
        total). Without this the recurrent drive scales with --ac: at ac=1000 the
        old un-normalised Mexican hat gave every AC a row sum of about -790 mV,
        which is ~300x the feedforward drive. That is a negative-feedback loop so
        stiff that raising --ac_gain cannot move AC activity at all."""
        return M / M.sum(0, keepdim=True).clamp(min=1e-6) * budget

    feat_rec = None
    W_rec = None
    if args.rec_mode == "none":
        pass                                             # paper-faithful ablation
    elif args.rec_mode == "random":
        # Unstructured reservoir: ~45% excitatory, ~55% inhibitory, uniform mags.
        rec_gen = torch.Generator(device="cpu").manual_seed(args.seed)
        mag = torch.rand(args.ac, args.ac, generator=rec_gen)
        pos_m = (torch.rand(args.ac, args.ac, generator=rec_gen) < 0.45).float()
        E, I = mag * pos_m, mag * (1.0 - pos_m)
        E.fill_diagonal_(0.0); I.fill_diagonal_(0.0)
        W_rec = normColumns(E, args.rec_exc) - normColumns(I, args.rec_inh)
    else:
        # Local: Gaussian excitation between ACs with nearby place centres,
        # flat inhibition everywhere else (centre-surround / Mexican hat).
        dist = torch.cdist(ac_centres, ac_centres) # calc dist
        excite = torch.exp(-(dist ** 2) / (2 * args.rec_radius ** 2))
        inhib = 1.0 - excite
        excite.fill_diagonal_(0.0); inhib.fill_diagonal_(0.0)
        W_rec = normColumns(excite, args.rec_exc) - normColumns(inhib, args.rec_inh)

    if W_rec is not None:
        W_rec.fill_diagonal_(0.0)                    # no self-connections
        feat_rec = Weight(name="w_rec", value=W_rec.to(DEVICE))
        net.add_connection(
            MulticompartmentConnection(source=ac, target=ac,
                                        pipeline=[feat_rec], device=DEVICE),
            source=LAYER_AC, target=LAYER_AC)
        print(f"[AC->AC] mode='{args.rec_mode}'  per-neuron budgets: "
              f"exc +{args.rec_exc:.1f} mV / inh -{args.rec_inh:.1f} mV  "
              f"(net row sum {W_rec.sum(1).mean():+.2f})")
    else:
        print("[AC->AC] mode='none' -- no recurrent connection")

    # ======================================================================= #
    # STEP 5 -- AC -> MC : the ONLY plastic pathway
    # ======================================================================= #
    # Everything upstream is frozen, so this single projection carries all of
    # the learning -- exactly as in the paper, where only AC->MC is plastic.
    # The paper modulates Hebbian updates by delta-Q from an external Q-table;
    # here the task gives dense per-step reward, so MSTDPET (reward-modulated
    # STDP with eligibility traces) plays the same third-factor role directly.
    ac_mc_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    ac_mc_mask = (torch.rand(args.ac, n_mc, generator=ac_mc_gen)
                  < args.ac_mc_sparsity).float().to(DEVICE)
 
    # Start near-uniform: before learning, no AC prefers any particular action,
    # which is the flat radar plot in the paper's Fig. 9 "pre-training" panels.
    W_ac_mc = ac_mc_mask * torch.rand(args.ac, n_mc, device=DEVICE) * args.ac_mc_init
 
    feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc,
                        learning_rule=MSTDPET, nu=[args.nu, args.nu])
    net.add_connection(
        MulticompartmentConnection(source=ac, target=mc,
                                   pipeline=[feat_ac_mc], device=DEVICE),
        source=LAYER_AC, target=LAYER_MC)
 
    # ac_mc_mask is handed to runSimulator because MSTDPET does NOT respect the
    # zeros: left alone it would grow weights on synapses that are supposed not
    # to exist, quietly turning a 40%-sparse projection dense. The mask is
    # re-applied (and weights clamped to [0, w_max]) after every update.

 
    net.to(DEVICE)
 
    weight_features = {"gc_ac": feat_gc_ac, "ac_mc": feat_ac_mc}
    if feat_rec is not None:
        weight_features["recurrent"] = feat_rec

    # One monitor per layer, sized to the decision window. These are what the
    # WTA readout and all the diagnostics read from.
    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
                                time=int(args.granularity / args.dt), device=DEVICE)
        net.add_monitor(spikes[layer], name=layer)
 
    # ======================================================================= #
    # STEP 7 -- SEPARABILITY CHECK (before any training)
    # ======================================================================= #
    # Nothing downstream can work if the AC layer is silent or saturated, so
    # check the representation first. This measures a property of the ARCHITECTURE
    # -- it would give the same answer before and after training, since the code
    # is frozen. Run this before burning hours on a training job.
    if args.diagnose:
        reportEncodingQuality(
            encodingQuality(net, W_grid, peak, view_r,
                            args.granularity, args.dt, args.gc_max_rate),
            "Encoding quality (pre-training)")
 
    environment = DotSimulator(
        args.steps, decay=args.decay, herrs=args.herrs, diag=args.diag,
        randr=args.randr, write=args.write, mute=args.mute,
        bound_hand=args.boundh, fit_func=args.fit_func,
        allow_stay=args.allow_stay, pandas=args.pandas, fpath=OUT_FILE_PATH,
    )
    environment.reset()
 
    print("Training:")
    environment.addFileSuffix("train")
    runSimulator(
        net, environment, spikes, args.trn_eps, W_grid, peak, ac_mc_mask,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_train", view_r=view_r,
        learning=True, eps_start=args.eps_start,
    )
 
    # Freeze plasticity; epsilon also goes to 0 so evaluation is pure policy.
    net.learning = False
 
    if args.diagnose:
        reportEncodingQuality(
            encodingQuality(net, W_grid, peak, view_r,
                            args.granularity, args.dt, args.gc_max_rate),
            "Encoding quality (post-training)")
        print()
 
    print("Testing:")
    environment.changeFileSuffix("train", "test")
    runSimulator(
        net, environment, spikes, args.tst_eps, W_grid, peak, ac_mc_mask,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "test"), pfname=genFileName("perf", "test"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_test", view_r=view_r,
        learning=False, eps_start=0.0,
    )

if __name__ == "__main__":
    main()
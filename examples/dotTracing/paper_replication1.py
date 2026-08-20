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
parser.add_argument("--render_every", type=int, default=25)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--plot_every", type=int, default=50)

# --- GC layer (paper: 5 scales x 7 rotations x 16 offsets = 560) ---
parser.add_argument("--gc_scales", type=int, nargs="+", default=[3, 5, 7, 11, 13])
parser.add_argument("--gc_rotations", type=int, default=7)
parser.add_argument("--gc_offsets", type=int, default=16)   # 4x4 phase grid
parser.add_argument("--gc_global_scale", type=float, default=1.0)  # paper's g
parser.add_argument("--gc_sharpness", type=float, default=1.0)     # paper's k
parser.add_argument("--gc_max_rate", type=float, default=80.0)     # 8 spikes / 100 ms

# --- AC layer (paper: 2000 neurons, 12% GC->AC sparsity) ---
parser.add_argument("--ac", type=int, default=1000)   # paper uses 2000; 1000 is faster
parser.add_argument("--gc_ac_sparsity", type=float, default=0.12)
parser.add_argument("--ac_gain", type=float, default=14.0)   # TUNE ME (see notes)
parser.add_argument("--rec_mode", type=str, default="local",
                    choices=["none", "random", "local"])
parser.add_argument("--rec_inh", type=float, default=1.2)
parser.add_argument("--rec_radius", type=float, default=4.0)

# --- MC layer (paper: 4 x 100 = 400) ---
parser.add_argument("--mc_pop", type=int, default=80)
parser.add_argument("--ac_mc_sparsity", type=float, default=0.40)
parser.add_argument("--w_max", type=float, default=12.0)
parser.add_argument("--nu", type=float, default=1e-2)

# --- action selection (paper: WTA + epsilon-greedy) ---
parser.add_argument("--select", type=str, default="wta", choices=["wta", "softmax"])
parser.add_argument("--eps_start", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.99)   # per step, as in paper
parser.add_argument("--eps_min", type=float, default=0.05)

parser.add_argument("--diagnose", type=bool, default=True)

args = parser.parse_args()

moveChoices = 9 if args.diag else 5
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
OUT_FILE_PATH = "PAPER1_RUN/"

LAYER_GC, LAYER_AC, LAYER_MC = "GC", "AC", "MC"

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# --------------------------------------------------------------------------- #
# View geometry
# --------------------------------------------------------------------------- #
def getLocalView(obs, row, col, view_r):
    """Flattened (2r+1)^2 - 1 egocentric patch, agent's own cell excluded."""
    padded = np.pad(obs, view_r, mode="constant", constant_values=0)
    patch = padded[row: row + 2 * view_r + 1, col: col + 2 * view_r + 1]
    flat = patch.flatten()
    center = view_r * (2 * view_r + 1) + view_r
    return np.concatenate([flat[:center], flat[center + 1:]])


# def viewIndexToOffset(view_r):
#     """(drow, dcol) of every input index, matching getLocalView's layout exactly."""
#     side = 2 * view_r + 1
#     center = view_r * side + view_r
#     offs = []
#     for k in range(side * side - 1):
#         cell = k if k < center else k + 1
#         pr, pc = divmod(cell, side)
#         offs.append((pr - view_r, pc - view_r))
#     return np.array(offs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# GRID CELL LAYER
# --------------------------------------------------------------------------- #
# def _hexLattice(spacing, rotation, phase, extent):
#     """Hexagonal lattice vertices covering [-extent, extent]^2."""
#     b1 = spacing * np.array([1.0, 0.0])
#     b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])
#     n = int(np.ceil(2.0 * extent / spacing)) + 2
#     ij = np.array(list(itertools.product(range(-n, n + 1), repeat=2)), dtype=np.float32)
#     pts = ij[:, :1] * b1 + ij[:, 1:] * b2
#     c, s = np.cos(rotation), np.sin(rotation)
#     R = np.array([[c, -s], [s, c]], dtype=np.float32)
#     pts = pts @ R.T + phase
#     keep = (np.abs(pts) <= extent + spacing).all(axis=1)
#     return pts[keep]


# def buildGridCellProjection(view_r, scales, n_rot, n_off, g, k, device="cpu"):
#     """
#     Fixed (inpt_n, n_gc) projection. Column j is grid cell j's receptive field over
#     the egocentric view: a hexagonal lattice of 2D Gaussians (the paper's Fig. 3).

#     Applying this to the pixel view -- rather than evaluating a grid code at a
#     known prey coordinate -- keeps the network pixel-driven. The paper hands its
#     grid cells the agent's (x,y) directly; doing that here would leak the extracted
#     prey position and hollow out the task.
#     """
#     offs = viewIndexToOffset(view_r)                     # (inpt_n, 2)
#     inpt_n = offs.shape[0]
#     extent = float(view_r)

#     rotations = [np.pi * i / max(1, n_rot) for i in range(n_rot)]
#     side = int(round(np.sqrt(n_off)))                    # 16 -> 4x4 phase grid

#     cols = []
#     meta = []
#     for s in scales:
#         spacing = s * g
#         d = (1.0 / k) * (spacing / 2.0)                  # paper: d = (1/k)(s*g/2)
#         sigma = d / 3.0                                  # paper: sigma = d/3
#         b1 = spacing * np.array([1.0, 0.0])
#         b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])
#         for theta in rotations:
#             for a in range(side):
#                 for b in range(side):
#                     phase = (a / side) * b1 + (b / side) * b2
#                     pts = _hexLattice(spacing, theta, phase, extent)
#                     # distance from every view offset to its NEAREST lattice vertex
#                     d2 = ((offs[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
#                     field = np.exp(-d2.min(axis=1) / (2.0 * sigma ** 2))
#                     field[field < 0.01] = 0.0            # paper zeroes sub-0.01
#                     cols.append(field.astype(np.float32))
#                     meta.append((s, theta, a, b))

#     W = np.stack(cols, axis=1)                           # (inpt_n, n_gc)
#     peak = W.max(axis=0, keepdims=True)                  # p_max, per cell
#     peak[peak == 0] = 1.0
#     return torch.from_numpy(W).to(device), torch.from_numpy(peak).to(device), meta


def gcRates(view_vec, W_grid, peak, max_rate):
    """view (inpt_n,) -> grid-cell firing rates (n_gc,). Paper: f = p/p_max * f_max."""
    drive = view_vec @ W_grid                            # (n_gc,)
    drive = (drive / peak.squeeze(0)).clamp(0.0, 1.0)
    drive[drive < 0.01] = 0.0
    return drive * max_rate


# --------------------------------------------------------------------------- #
# AC connectivity + place-field centres (for structured local inhibition)
# --------------------------------------------------------------------------- #
# def buildSparseMask(n_pre, n_post, sparsity, seed, device):
#     g = torch.Generator(device="cpu").manual_seed(seed)
#     return (torch.rand(n_pre, n_post, generator=g) < sparsity).float().to(device)

# def acPlaceCentres(W_grid, gc_ac_mask, view_r):
#     """
#     Effective receptive field of AC j over the view = W_grid @ mask[:, j]. Its argmax
#     is where several sampled grid fields coincide -- i.e. the AC's place-field centre.
#     Used only to give the recurrent inhibition a spatial structure.
#     """
#     eff = W_grid @ gc_ac_mask                            # (inpt_n, n_ac)
#     idx = eff.argmax(dim=0).cpu().numpy()
#     offs = viewIndexToOffset(view_r)
#     return torch.from_numpy(offs[idx]).float()           # (n_ac, 2)

# def buildRecurrent(mode, centres, n_ac, inh, radius, device, seed=0):
#     if mode == "none":
#         return None
#     if mode == "random":
#         g = torch.Generator(device="cpu").manual_seed(seed)
#         signs = (torch.rand(n_ac, n_ac, generator=g) < 0.45).float() * 2 - 1
#         W = torch.rand(n_ac, n_ac, generator=g) * signs
#     else:  # "local": short-range excitation, broad surround inhibition
#         d = torch.cdist(centres, centres)
#         exc = torch.exp(-(d ** 2) / (2 * radius ** 2))
#         W = exc - inh * (1.0 - exc)
#     W.fill_diagonal_(0.0)
#     return W.to(device)


# --------------------------------------------------------------------------- #
# Diagnostics -- the paper's Table 4 (Encoding Quality Score)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encodingQuality(net, W_grid, peak, view_r, gran, dt, max_rate,
                    spike_thresh=4, stride=3):
    """
    Sweep a single prey pixel over the view and measure, for GC and AC:
      - mean number of active cells per position
      - mean pairwise overlap between positions
      - quality score = active / overlap  (paper: 9.2 GC -> 16.6 AC)
    This is the paper's separability test, adapted to relative prey position.
    """
    inpt_n = (2 * view_r + 1) ** 2 - 1
    keep = [i for i in range(inpt_n) if (i % stride == 0)]
 
    gc_sets, ac_sets = [], []
    for i in keep:
        v = torch.zeros(inpt_n, device=DEVICE)
        v[i] = 1.0
        rates = gcRates(v, W_grid, peak, max_rate)
        inputs = {LAYER_GC: poisson(rates.unsqueeze(0), gran, dt, device=DEVICE)}
        net.reset_state_variables()
        net.run(inputs=inputs, time=gran, reward=0.0)
        gc = net.monitors[LAYER_GC].get("s").squeeze().sum(0)
        ac = net.monitors[LAYER_AC].get("s").squeeze().sum(0)
        gc_sets.append((gc >= spike_thresh).float())
        ac_sets.append((ac >= spike_thresh).float())
 
    out = {}
    for name, sets in (("GC", gc_sets), ("AC", ac_sets)):
        M = torch.stack(sets)                            # (P, n)
        active = M.sum(1)
        ov = M @ M.T
        n = ov.shape[0]
        off = ov[~torch.eye(n, dtype=bool, device=ov.device)]
        a, o = active.mean().item(), off.mean().item()
        out[name] = dict(avg_active=a, avg_overlap=o,
                         min_diag=active.min().item(), max_off=off.max().item(),
                         quality=a / o if o > 0 else float("inf"))
    net.reset_state_variables()
    return out


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

        while not done:
            step += 1

            # ---- reward: sign from outcome, magnitude from alignment ----------

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

        print(f"Episode {ep}: total reward {total_reward:.2f}, "
              f"intercepts {intercepts}, eps {eps:.3f}")
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
    ac = LIFNodes(n=args.ac, traces=True,
                  rest=-64.0, reset=-70.0, thresh=-45.0,
                  refrac=1, tc_decay=20.0, tc_trace=20.0)
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
    W_gc_ac = W_gc_ac / fan_in.sqrt() * args.ac_gain
 
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
 
    feat_rec = None
    if args.rec_mode == "none":
        W_rec = None                                     # paper-faithful ablation
    elif args.rec_mode == "random":
        # Unstructured reservoir: ~45% excitatory, ~55% inhibitory, uniform mags.
        rec_gen = torch.Generator(device="cpu").manual_seed(args.seed)
        signs = (torch.rand(args.ac, args.ac, generator=rec_gen) < 0.45).float() * 2 - 1
        W_rec = torch.rand(args.ac, args.ac, generator=rec_gen) * signs
    else:
        # Local: Gaussian excitation between ACs with nearby place centres,
        # flat inhibition everywhere else (centre-surround / Mexican hat).
        dist = torch.cdist(ac_centres, ac_centres)       # (n_ac, n_ac)
        excite = torch.exp(-(dist ** 2) / (2 * args.rec_radius ** 2))
        W_rec = excite - args.rec_inh * (1.0 - excite)
 
    W_rec.fill_diagonal_(0.0)                        # no self-connections
    feat_rec = Weight(name="w_rec", value=W_rec.to(DEVICE))
    net.add_connection(
        MulticompartmentConnection(source=ac, target=ac,
                                    pipeline=[feat_rec], device=DEVICE),
        source=LAYER_AC, target=LAYER_AC)
    print(f"[AC->AC] recurrent mode='{args.rec_mode}'")

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
    W_ac_mc = ac_mc_mask * torch.rand(args.ac, n_mc, device=DEVICE) * 6.0
 
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
        q = encodingQuality(net, W_grid, peak, view_r,
                            args.granularity, args.dt, args.gc_max_rate)
        print("\n[Encoding quality]  (paper: GC 9.2 -> AC 16.6)")
        for k, v in q.items():
            print(f"  {k}: active {v['avg_active']:.1f}, "
                  f"overlap {v['avg_overlap']:.1f}, score {v['quality']:.1f}")
        if q["AC"]["avg_active"] < 5:
            print("  !! AC layer nearly silent -- raise --ac_gain")
        if q["AC"]["avg_active"] > 0.20 * args.ac:
            print("  !! AC layer saturated -- lower --ac_gain or raise --rec_inh")
        print()
 
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
        q = encodingQuality(net, W_grid, peak, view_r,
                            args.granularity, args.dt, args.gc_max_rate)
        print("\n[Encoding quality after training]")
        for k, v in q.items():
            print(f"  {k}: active {v['avg_active']:.1f}, "
                  f"overlap {v['avg_overlap']:.1f}, score {v['quality']:.1f}")
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

# # --------------------------------------------------------------------------- #
# def main():
#     net = Network(dt=args.dt)
#     inpt_n = (2 * args.view_r + 1) ** 2 - 1

#     # ---------------- GC: fixed multi-scale periodic basis ------------------ #
#     W_grid, peak, gc_meta = buildGridCellProjection(
#         args.view_r, args.gc_scales, args.gc_rotations, args.gc_offsets,
#         args.gc_global_scale, args.gc_sharpness, device=DEVICE,
#     )
#     n_gc = W_grid.shape[1]
#     print(f"[GC] {n_gc} grid cells "
#           f"({len(args.gc_scales)} scales x {args.gc_rotations} rotations "
#           f"x {args.gc_offsets} offsets) over {inpt_n} view cells")

#     gc = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)

#     # ---------------- AC: coincidence-detecting LIF (paper Table 3) --------- #
#     ac = LIFNodes(n=args.ac, traces=True,
#                   rest=-64.0, reset=-70.0, thresh=-45.0,
#                   refrac=1, tc_decay=20.0, tc_trace=20.0)

#     # ---------------- MC: one subpopulation per action ---------------------- #
#     n_mc = moveChoices * args.mc_pop
#     mc = LIFNodes(n=n_mc, traces=True,
#                   rest=-64.0, reset=-64.0, thresh=-49.0,
#                   refrac=0, tc_decay=20.0, tc_trace=20.0)

#     net.add_layer(gc, name=LAYER_GC)
#     net.add_layer(ac, name=LAYER_AC)
#     net.add_layer(mc, name=LAYER_MC)

#     # GC -> AC : sparse, fixed, excitatory (paper: 12%, never modified)
#     gc_ac_mask = buildSparseMask(n_gc, args.ac, args.gc_ac_sparsity,
#                                  args.seed, DEVICE)
#     W_gc_ac = gc_ac_mask * torch.rand(n_gc, args.ac, device=DEVICE)
#     fan_in = gc_ac_mask.sum(0).mean().clamp(min=1.0)
#     W_gc_ac = W_gc_ac / fan_in.sqrt() * args.ac_gain
#     feat_gc_ac = Weight(name="w_gc_ac", value=W_gc_ac)
#     net.add_connection(
#         MulticompartmentConnection(source=gc, target=ac,
#                                    pipeline=[feat_gc_ac], device=DEVICE),
#         source=LAYER_GC, target=LAYER_AC)

    
#     centres = acPlaceCentres(W_grid, gc_ac_mask, args.view_r)
#     W_rec = buildRecurrent(args.rec_mode, centres, args.ac,
#                            args.rec_inh, args.rec_radius, DEVICE, args.seed)
#     feat_rec = None
#     if W_rec is not None:
#         feat_rec = Weight(name="w_rec", value=W_rec)
#         net.add_connection(
#             MulticompartmentConnection(source=ac, target=ac,
#                                        pipeline=[feat_rec], device=DEVICE),
#             source=LAYER_AC, target=LAYER_AC)

#     # AC -> MC : the ONLY plastic pathway (paper), 40% sparse
#     ac_mc_mask = buildSparseMask(args.ac, n_mc, args.ac_mc_sparsity,
#                                  args.seed + 1, DEVICE)
#     W_ac_mc = ac_mc_mask * torch.rand(args.ac, n_mc, device=DEVICE) * 6.0
#     feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc,
#                         learning_rule=MSTDPET, nu=[args.nu, args.nu])
#     net.add_connection(
#         MulticompartmentConnection(source=ac, target=mc,
#                                    pipeline=[feat_ac_mc], device=DEVICE),
#         source=LAYER_AC, target=LAYER_MC)

#     net.to(DEVICE)

#     weight_features = {"gc_ac": feat_gc_ac, "ac_mc": feat_ac_mc}
#     if feat_rec is not None:
#         weight_features["recurrent"] = feat_rec

#     spikes = {}
#     for layer in net.layers:
#         spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
#                                 time=int(args.granularity / args.dt), device=DEVICE)
#         net.add_monitor(spikes[layer], name=layer)

#     # ---------------- separability check before training -------------------- #
#     if args.diagnose:
#         q = encodingQuality(net, W_grid, peak, args.view_r,
#                             args.granularity, args.dt, args.gc_max_rate)
#         print("\n[Encoding quality]  (paper: GC 9.2 -> AC 16.6)")
#         for k, v in q.items():
#             print(f"  {k}: active {v['avg_active']:.1f}, "
#                   f"overlap {v['avg_overlap']:.1f}, score {v['quality']:.1f}")
#         if q["AC"]["avg_active"] < 5:
#             print("  !! AC layer nearly silent -- raise --ac_gain")
#         if q["AC"]["avg_active"] > 0.20 * args.ac:
#             print("  !! AC layer saturated -- lower --ac_gain or raise --rec_inh")
#         print()

#     environment = DotSimulator(
#         args.steps, decay=args.decay, herrs=args.herrs, diag=args.diag,
#         randr=args.randr, write=args.write, mute=args.mute,
#         bound_hand=args.boundh, fit_func=args.fit_func,
#         allow_stay=args.allow_stay, pandas=args.pandas, fpath=OUT_FILE_PATH,
#     )
#     environment.reset()

#     print("Training:")
#     environment.addFileSuffix("train")
#     final_eps = runSimulator(
#         net, environment, spikes, args.trn_eps, W_grid, peak, ac_mc_mask,
#         feat_ac_mc, gran=args.granularity,
#         rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
#         weight_features=weight_features, render_replays=args.render_replays,
#         render_every=args.render_every, replay_fps=args.replay_fps,
#         replay_prefix="replay_train", view_r=args.view_r,
#         learning=True, eps_start=args.eps_start,
#     )

#     net.learning = False

#     if args.diagnose:
#         q = encodingQuality(net, W_grid, peak, args.view_r,
#                             args.granularity, args.dt, args.gc_max_rate)
#         print("\n[Encoding quality after training]")
#         for k, v in q.items():
#             print(f"  {k}: active {v['avg_active']:.1f}, "
#                   f"overlap {v['avg_overlap']:.1f}, score {v['quality']:.1f}")
#         print()

#     print("Testing:")
#     environment.changeFileSuffix("train", "test")
#     runSimulator(
#         net, environment, spikes, args.tst_eps, W_grid, peak, ac_mc_mask,
#         feat_ac_mc, gran=args.granularity,
#         rfname=genFileName("rew", "test"), pfname=genFileName("perf", "test"),
#         weight_features=weight_features, render_replays=args.render_replays,
#         render_every=args.render_every, replay_fps=args.replay_fps,
#         replay_prefix="replay_test", view_r=args.view_r,
#         learning=False, eps_start=0.0,
#     )


if __name__ == "__main__":
    main()
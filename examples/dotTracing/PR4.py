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

from ast import arg

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
from bindsnet.network.topology_features import Weight, Mask

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
parser.add_argument("--make_sparse", type=bool, default=False)

# rendering 
parser.add_argument("--render_replays", type=bool, default=True)
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--plot_every", type=int, default=50)
# Raster replays: for every episode that gets an env GIF, also emit a spike-raster
# GIF with one frame per decision, plus a whole-episode raster timeline PNG.
parser.add_argument("--raster_replays", type=bool, default=True)
parser.add_argument("--raster_timeline", type=bool, default=True)
parser.add_argument("--raster_layers", type=str, nargs="+",
                    default=["PC_A", "PC_T", "AC", "MC"])
# 0 (the default) plots EVERY neuron in the layer. A positive value draws an
# evenly spaced subsample instead. Spike counts, not neuron counts, drive the
# drawing cost, so plotting all of them is cheap.
parser.add_argument("--raster_max_neurons", type=int, default=0)

# GC layer 
parser.add_argument("--gc_scales", type=int, nargs="+", default=[5, 7, 11, 13, 17])
parser.add_argument("--gc_rotations", type=int, default=7)
parser.add_argument("--gc_offsets", type=int, default=16)   # 4x4 phase grid
parser.add_argument("--gc_global_scale", type=float, default=1.0)  # paper's g
parser.add_argument("--gc_sharpness", type=float, default=1.0)     # paper's k
parser.add_argument("--gc_max_rate", type=float, default=80.0)     # 8 spikes / 100 ms

# PLACE CELL layers: one for the agent, one for the target
parser.add_argument("--n_pc", type=int, default=500)          # per place layer
parser.add_argument("--gc_pc_sparsity", type=float, default=0.05) 
parser.add_argument("--gc_pc_gain", type=float, default=25.0) 
parser.add_argument("--pc_lbound", type=float, default=-80.0)

# ASSOCIATION layer 
parser.add_argument("--ac", type=int, default=1000) 
parser.add_argument("--gc_ac_sparsity", type=float, default=0.12)
parser.add_argument("--pc_ac_sparsity", type=float, default=0.1)
parser.add_argument("--pc_ac_gain", type=float, default=25.0) 
parser.add_argument("--ac_gain", type=float, default=12.0)  
parser.add_argument("--ac_lbound", type=float, default=-80.0) 
parser.add_argument("--rec_mode", type=str, default="random",
                    choices=["random", "local"])

# RECURRENT connection
parser.add_argument("--rec_scaler", type=float, default=5.0)
parser.add_argument("--rec_radius", type=float, default=4.0)
parser.add_argument("--ac_target_sparsity", type=float, default=0.10)

# MOTOR layer 100 neurons per move (5 moves)
parser.add_argument("--mc_pop", type=int, default=100)
parser.add_argument("--ac_mc_sparsity", type=float, default=0.40)
parser.add_argument("--ac_mc_init", type=float, default=0.3)  
parser.add_argument("--w_max", type=float, default=1.0)
parser.add_argument("--nu", type=float, default=4e-3)

parser.add_argument("--diagnose", type=bool, default=True)
parser.add_argument("--diag_stride", type=int, default=4)   # probe positions
parser.add_argument("--diag_thresh", type=int, default=4)   # spikes -> "active"

args = parser.parse_args()

moveChoices = 9 if args.diag else 5
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
OUT_FILE_PATH = "PR4_RUN_TEST3/"

LAYER_GCA, LAYER_GCT = "GC_A", "GC_T"      # grid code at agent / at target
LAYER_PCA, LAYER_PCT = "PC_A", "PC_T"      # place cells for agent / target
LAYER_AC, LAYER_MC = "AC", "MC"

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# select action based on the largest spiking population
def wtaAction(mc_spikes, n_actions, rng):
    """mc_spikes: (time, n_mc) -> action index. Aggregate count per subpopulation."""
    counts = mc_spikes.sum(0)
    pops = torch.chunk(counts, n_actions)
    agg = torch.stack([p.sum() for p in pops])
    if agg.max() == 0:
        return int(rng.integers(0, n_actions))
    return int(torch.argmax(agg).item())

# TRAINING LOOP
# todo: reward is currently more sparse but the reward still needs
# to serve as an indicator
def runSimulator(net, env, spikes, episodes, W_grid, peak,
                 feat_ac_mc, gran=100, rfname="", pfname="",
                 weight_features=None, render_replays=False, render_every=25,
                 replay_fps=20, replay_prefix="replay", view_r=14,
                 dim=28):
    dt = net.dt
    peak_v = peak.squeeze(0)                       # (n_gc,) hoisted out of the loop
    zero_gc = torch.zeros(W_grid.shape[1], device=DEVICE)
    rng = np.random.default_rng(args.seed)
    spike_ims = spike_axes = None

    for ep in range(episodes):
        total_reward, intercepts, step = 0.0, 0, 0
        visible_steps = 0                          # how often the target was in view
        rewards = np.zeros(env.timesteps)
        net.reset_state_variables()
        env.reset()
        done = False

        capturing = render_replays and (ep % render_every == 0)
        replay_frames = []
        raster_frames, raster_windows = [], []
        raster_ims = raster_axes = raster_fig = None
        raster_sizes = {l: (net.layers[l].n if args.raster_max_neurons <= 0
                            else min(net.layers[l].n, args.raster_max_neurons))
                        for l in args.raster_layers}
        if capturing:
            env.render()          # pins plt.gcf() as the env figure on ep 0
            replay_frames.append(captureFrame())
            if args.raster_replays:
                blank = {l: torch.zeros(gran, raster_sizes[l])
                         for l in args.raster_layers}
                raster_ims, raster_axes = plot_spikes(
                    blank, ims=raster_ims, axes=raster_axes, figsize=(8,12))
                raster_fig = plt.gcf()
                styleRasterAxes(raster_axes, args.raster_layers, gran,
                                raster_sizes, first_call=True)
                raster_fig.suptitle(f"ep {ep}  initial state -- no decision yet",
                                    fontsize=9)
                raster_frames.append(captureFrame(raster_fig))

        action = int(rng.integers(0, env.action_space.n))
        last_active_ac = torch.zeros(net.layers[LAYER_AC].n, device=DEVICE)
        clock = time.time()
        ac_frac_sum, mc_frac_sum = 0.0, 0.0

        while not done:
            step += 1

            # get prev row+col and calc dist
            pr, pc = env.netDot.row[0], env.netDot.col[0]
            prev_dist = np.hypot(pr - env.dots[0].row[0], pc - env.dots[0].col[0])

            # take a step
            obs, _, done, intercept = env.step(action)

            # agent row+col after the step
            cr, cc = env.netDot.row[0], env.netDot.col[0]
            # target row+col after the step
            tr, tc = env.dots[0].row[0], env.dots[0].col[0]

            # calc change in dist
            curr_dist = np.hypot(cr - tr, cc - tc)
            r = prev_dist - curr_dist

            # calculate alignment based on dot product
            # dr_p, dc_p = tr - pr, tc - pc
            # dr_m, dc_m = cr - pr, cc - pc
            # dmag, mmag = np.hypot(dr_p, dc_p), np.hypot(dr_m, dc_m)

            # alignment = 0.0 if (mmag < 1e-8 or dmag < 1e-8) else \
            #     float(np.clip((dr_m * dr_p + dc_m * dc_p) / (dmag * mmag), -1, 1))
            # scale = 0.5 + 0.5 * alignment
            # r = delta * ((0.5 + scale) if delta >= 0 else (1.5 - scale))
            # r += 0.2 * alignment

            if intercept:
                r += 10.0
            # decrease reward over time
            if args.make_sparse: 
                r *= (ep/episodes+0.5)
            reward = torch.tensor(r, dtype=torch.float32, device=DEVICE)

            # W_grid is (n_world, n_gc): row k is the grid-cell population code
            # for board square k. Because the lattice is anchored to the WORLD,
            # reading a position is a row lookup, not a matrix product.
            a_drive = (W_grid[int(cr) * dim + int(cc)] / peak_v).clamp(0.0, 1.0)
            a_drive = torch.where(a_drive < 0.01, zero_gc, a_drive) # remove gaussian tail
            rates_a = a_drive * args.gc_max_rate

            # check if target is outside of the viewing radius of the agent
            visible = (abs(int(tr) - int(cr)) <= view_r and
                       abs(int(tc) - int(cc)) <= view_r)
            if visible:
                t_drive = (W_grid[int(tr) * dim + int(tc)] / peak_v).clamp(0.0, 1.0)
                t_drive = torch.where(t_drive < 0.01, zero_gc, t_drive)
                rates_t = t_drive * args.gc_max_rate
            else:
                rates_t = zero_gc                     # silent target stream
            visible_steps += int(visible)

            inputs = {
                LAYER_GCA: poisson(rates_a.unsqueeze(0), gran, dt, device=DEVICE),
                LAYER_GCT: poisson(rates_t.unsqueeze(0), gran, dt, device=DEVICE),
            }

            # run network with gridcell firing as input to get a move choice
            net.run(inputs=inputs, time=gran, reward=reward)

            mc = spikes[LAYER_MC].get("s").squeeze()          # (time, n_mc)
            ac = spikes[LAYER_AC].get("s").squeeze()
            last_active_ac = (ac.sum(0) >= 4).float()
            ac_frac_sum += last_active_ac.mean().item()
            mc_frac_sum += (mc.sum(0) > 0).float().mean().item()

            # get the action from the most active motor cell population
            action = wtaAction(mc, env.action_space.n, rng)

            rewards[step - 1] = r
            total_reward += r
            intercepts += int(bool(intercept))

            if capturing:
                if args.raster_replays or args.raster_timeline:
                    window = {
                        l: rasterSubsample(
                            spikes[l].get("s").squeeze().view(gran, -1),
                            args.raster_max_neurons).detach().cpu()
                        for l in args.raster_layers
                    }
                    if args.raster_timeline:
                        raster_windows.append(window)
                    if args.raster_replays:
                        raster_ims, raster_axes = plot_spikes(
                            window, ims=raster_ims, axes=raster_axes,
                            figsize=(8,12))
                        if raster_fig is None:
                            raster_fig = plt.gcf()
                            styleRasterAxes(raster_axes, args.raster_layers, gran,
                                            raster_sizes, first_call=True)
                        else:
                            styleRasterAxes(raster_axes, args.raster_layers, gran,
                                            raster_sizes)
                        raster_fig.suptitle(f"ep {ep}  decision {step}  "
                                            f"action {action}  r={r:+.2f}",
                                            fontsize=9)
                        raster_frames.append(captureFrame(raster_fig))
                # env.render() must come last: it makes its own figure current,
                # which is what the un-argumented captureFrame() below reads.
                env.render()
                replay_frames.append(captureFrame())

            if args.plot_every and ep % args.plot_every == 0 and step == 1:
                spikes_ = {l: spikes[l].get("s").view(gran, -1) for l in spikes}
                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims,
                                                    axes=spike_axes)

            if step % 25 == 0:
                print(f"  step {step} ({time.time() - clock:.2f}s) "
                      f"r={r:+.3f}")
                clock = time.time()

        if net.reward_fn is not None:
            net.reward_fn.update(accumulated_reward=total_reward, steps=step)

        # plotting + diagnostics code
        print(f"Episode {ep}: total reward {total_reward:.2f}"
              f"intercepts {intercepts} "
              f"target visible {visible_steps / max(1, step):.0%} | "
              f"AC active {ac_frac_sum / max(1, step):.1%}, "
              f"MC active {mc_frac_sum / max(1, step):.1%} | ")
        os.makedirs(OUT_FILE_PATH, exist_ok=True)
        if capturing and replay_frames:
            saveReplayGIF(replay_frames,
                          os.path.join(OUT_FILE_PATH,
                                       f"{replay_prefix}_ep{ep:05d}.gif"),
                          fps=replay_fps)
        if capturing and raster_frames:
            # Frame count matches the env GIF (both open on the initial state), so
            # frame i is the same instant in both and they play in lockstep.
            saveReplayGIF(raster_frames,
                          os.path.join(OUT_FILE_PATH,
                                       f"{replay_prefix}_raster_ep{ep:05d}.gif"),
                          fps=replay_fps)
        if capturing and raster_windows:
            plotRasterTimeline(
                raster_windows, args.raster_layers, gran, ep,
                os.path.join(OUT_FILE_PATH,
                             f"{replay_prefix}_rastertl_ep{ep:05d}.png"))
        if capturing and raster_fig is not None:
            plt.close(raster_fig)
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


# --------------------------------------------------------------------------- #
def main():
    dim = args.dim
    # store all coordinate pairs
    world_xy = np.array([(r, c) for r in range(dim) for c in range(dim)],
                        dtype=np.float32)               
    n_world = world_xy.shape[0]
    # centre the lattice on the board so rotations pivot about the middle
    board_centre = np.array([(dim - 1) / 2.0, (dim - 1) / 2.0], dtype=np.float32)
    world_rel = world_xy - board_centre                       # (n_world, 2)
    extent = dim / 2.0

    phase_side = int(round(np.sqrt(args.gc_offsets)))
    gc_fields, gc_meta = [], []

    for s in args.gc_scales:
        spacing = s * args.gc_global_scale
        d = (1.0 / args.gc_sharpness) * (spacing / 2.0)
        sigma = d / 3.0

        b1 = spacing * np.array([1.0, 0.0])
        b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])

        n_tile = int(np.ceil(2.0 * extent / spacing)) + 2
        ij = np.array(list(itertools.product(range(-n_tile, n_tile + 1), repeat=2)),
                      dtype=np.float32)
        lattice_base = ij[:, :1] * b1 + ij[:, 1:] * b2

        for r_i in range(7): # 7 rotations
            theta = np.pi * r_i / 7
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            lattice_rot = lattice_base @ R.T

            # 4 phase shifts for up and down
            for a in range(4):
                for b in range(4):
                    phase = (a / 4) * b1 + (b / 4) * b2
                    pts = lattice_rot + phase
                    # find out which points to keep (within the 28x28 grid)
                    keep = (np.abs(pts) <= extent + spacing).all(axis=1)
                    pts = pts[keep]

                    # create gaussians
                    d2 = ((world_rel[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
                    field = np.exp(-d2.min(axis=1) / (2.0 * sigma ** 2))
                    field[field < 0.01] = 0.0 # cut off the gaussian tail
                    gc_fields.append(field.astype(np.float32))
                    gc_meta.append((s, theta, a, b))


    W_grid_np = np.stack(gc_fields, axis=1)       
    peak_np = W_grid_np.max(axis=0, keepdims=True)
    peak_np[peak_np == 0] = 1.0
    # this weights grid is the weigths of the grid cells, which is 
    # passed into the simulator
    # grid cell weights are from the gaussians
    W_grid = torch.from_numpy(W_grid_np).to(DEVICE)
    peak = torch.from_numpy(peak_np).to(DEVICE)
    n_gc = W_grid.shape[1]

    net = Network(dt=args.dt)

    gc_a = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)
    gc_t = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)

    # agent place cells
    pc_a = LIFNodes(n=args.n_pc, traces=True,
                    rest=-64.0, reset=-70.0, thresh=-25.0,
                    refrac=1, tc_decay=20.0, tc_trace=20.0,
                    lbound=args.pc_lbound)
    # target place cells
    pc_t = LIFNodes(n=args.n_pc, traces=True,
                    rest=-64.0, reset=-70.0, thresh=-25.0,
                    refrac=1, tc_decay=20.0, tc_trace=20.0,
                    lbound=args.pc_lbound)

    ac = LIFNodes(n=args.ac, traces=True,
                  rest=-64.0, reset=-70.0, thresh=-45.0,
                  refrac=1, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.ac_lbound)

    n_mc = moveChoices * args.mc_pop
    mc = LIFNodes(n=n_mc, traces=True,
                  rest=-64.0, reset=-64.0, thresh=-49.0,
                  refrac=0, tc_decay=20.0, tc_trace=20.0)

    net.add_layer(gc_a, name=LAYER_GCA)
    net.add_layer(gc_t, name=LAYER_GCT)
    net.add_layer(pc_a, name=LAYER_PCA)
    net.add_layer(pc_t, name=LAYER_PCT)
    net.add_layer(ac,   name=LAYER_AC)
    net.add_layer(mc,   name=LAYER_MC)

    # grid cell to place cell connections
    # 5% sparsity
    gen_a = torch.Generator(device="cpu").manual_seed(args.seed + 10)
    gc_pc_mask_a = (torch.rand(n_gc, args.n_pc, generator=gen_a)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_a = gc_pc_mask_a * torch.rand(n_gc, args.n_pc, device=DEVICE)
    W_gc_pc_a = (W_gc_pc_a / gc_pc_mask_a.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * args.gc_pc_gain

    gen_t = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    gc_pc_mask_t = (torch.rand(n_gc, args.n_pc, generator=gen_t)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_t = gc_pc_mask_t * torch.rand(n_gc, args.n_pc, device=DEVICE)
    W_gc_pc_t = (W_gc_pc_t / gc_pc_mask_t.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * args.gc_pc_gain

    feat_gc_pc_a = Weight(name="w_gc_pc_a", value=W_gc_pc_a)
    net.add_connection(
        MulticompartmentConnection(source=gc_a, target=pc_a,
                                   pipeline=[feat_gc_pc_a], device=DEVICE),
        source=LAYER_GCA, target=LAYER_PCA)

    feat_gc_pc_t = Weight(name="w_gc_pc_t", value=W_gc_pc_t)
    net.add_connection(
        MulticompartmentConnection(source=gc_t, target=pc_t,
                                   pipeline=[feat_gc_pc_t], device=DEVICE),
        source=LAYER_GCT, target=LAYER_PCT)

    # PLACE to ASSOCIATION connection 10% sparsity
    dense_fan = float(2 * args.n_pc)
    pc_ac_a_mask = (torch.rand(args.n_pc, args.ac, generator=gen_a) 
                    <= args.pc_ac_sparsity).float().to(DEVICE)
    W_pc_ac_a = (pc_ac_a_mask * (torch.rand(args.n_pc, args.ac, device=DEVICE))
                 / np.sqrt(dense_fan)) * args.pc_ac_gain
    pc_ac_t_mask = (torch.rand(args.n_pc, args.ac, generator=gen_t) 
                        <= args.pc_ac_sparsity).float().to(DEVICE)
    W_pc_ac_t = (pc_ac_t_mask * (torch.rand(args.n_pc, args.ac, device=DEVICE))
                 / np.sqrt(dense_fan)) * args.pc_ac_gain

    feat_pc_ac_a = Weight(name="w_pc_ac_a", value=W_pc_ac_a)
    net.add_connection(
        MulticompartmentConnection(source=pc_a, target=ac,
                                   pipeline=[feat_pc_ac_a], device=DEVICE),
        source=LAYER_PCA, target=LAYER_AC)

    feat_pc_ac_t = Weight(name="w_pc_ac_t", value=W_pc_ac_t)
    net.add_connection(
        MulticompartmentConnection(source=pc_t, target=ac,
                                   pipeline=[feat_pc_ac_t], device=DEVICE),
        source=LAYER_PCT, target=LAYER_AC)

    # Recurrent association layer connection reservoir, no learning done
    rec_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    mag = torch.rand(args.ac, args.ac, generator=rec_gen)
    pos_m = (torch.rand(args.ac, args.ac, generator=rec_gen) < 0.45).float()
    W_rec = mag * pos_m
    # normalize each by sum of their columns and mult by scale factor
    W_rec = W_rec / W_rec.sum(0, keepdim=True).clamp(min=1e-6) * args.rec_scaler
    W_rec.fill_diagonal_(0.0)
    feat_rec = Weight(name="w_rec", value=W_rec.to(DEVICE))
    net.add_connection(
        MulticompartmentConnection(source=ac, target=ac,
                                   pipeline=[feat_rec], device=DEVICE),
        source=LAYER_AC, target=LAYER_AC)

    # association to motor layer, the only learning layer
    ac_mc_gen = torch.Generator(device="cpu").manual_seed(args.seed + 4)
    ac_mc_mask_gen = (torch.rand(args.ac, n_mc, generator=ac_mc_gen)
                  < 0.40).to(DEVICE) # 40% sparsity for ac->mc

    ac_mc_mask = Mask(name='ac_mc_mask',value=ac_mc_mask_gen)
 
    W_ac_mc = ac_mc_mask_gen * torch.rand(args.ac, n_mc, device=DEVICE) * args.ac_mc_init
 
    feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc,
                        learning_rule=MSTDPET, nu=[args.nu, args.nu])
    net.add_connection(
        MulticompartmentConnection(source=ac, target=mc,
                                   pipeline=[feat_ac_mc,ac_mc_mask], device=DEVICE),
        source=LAYER_AC, target=LAYER_MC)

    net.to(DEVICE)
    weight_features = {"gc_pc_a": feat_gc_pc_a, "gc_pc_t": feat_gc_pc_t,
                       "pc_ac_a": feat_pc_ac_a, "pc_ac_t": feat_pc_ac_t,
                       "recurrent": feat_rec, "ac_mc": feat_ac_mc}

    print(f"[arch] GC {n_gc} x2  ->  PC {args.n_pc} x2 (sparse "
          f"{args.gc_pc_sparsity:.0%})  ->  AC {args.ac} (dense)  ->  MC {n_mc}")
    print(f"[gate] view_r={args.view_r} on a {dim}x{dim} board: target is visible "
          f"when Chebyshev distance <= {args.view_r}")

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
                                time=int(args.granularity / args.dt), device=DEVICE)
        net.add_monitor(spikes[layer], name=layer)
 
    # check spike counts
    if args.diagnose:
        reportEncodingQuality(
            encodingQuality(net, W_grid, peak, args.view_r,
                            args.granularity, args.dt, args.gc_max_rate),
            "Encoding quality (pre-training)")
 
    environment = DotSimulator(
        args.steps, decay=args.decay, herrs=args.herrs, diag=args.diag,
        randr=args.randr, write=args.write, mute=args.mute,
        bound_hand=args.boundh, fit_func=args.fit_func, 
        allow_stay=args.allow_stay, pandas=args.pandas, fpath=OUT_FILE_PATH,
    )
    environment.reset()

    # DotSimulator.render() is a no-op when mute=True, so the env GIF would be
    # frames of whatever stale figure happened to be current. Raster GIFs do not
    # depend on env.render() and stay valid either way.

    print("Training:")
    environment.addFileSuffix("train")
    runSimulator(
        net, environment, spikes, args.trn_eps, W_grid, peak,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_train", view_r=args.view_r, dim=dim,
    )
    net.learning = False
 
    # if args.diagnose:
    #     reportEncodingQuality(
    #         encodingQuality(net, W_grid, peak,
    #                         args.granularity, args.dt, args.gc_max_rate),
    #         "Encoding quality (post-training)")
    #     print()
 
    print("Testing:")
    environment.changeFileSuffix("train", "test")
    runSimulator(
        net, environment, spikes, args.tst_eps, W_grid, peak,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "test"), pfname=genFileName("perf", "test"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_test", view_r=args.view_r, dim=dim,
    )

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

    # The state space is no longer "prey offset in an egocentric patch" -- it is
    # "target square, given the agent is somewhere". So pin the agent at the
    # centre of the board and sweep the TARGET over board squares, skipping any
    # that fall outside the agent's egocentric window (PC_T is silent there by
    # design, and scoring those positions would just measure the gate).
    dim = args.dim
    peak_v = peak.squeeze(0)
    zero_gc = torch.zeros(W_grid.shape[1], device=DEVICE)

    ar, ac_col = dim // 2, dim // 2                    # agent pinned at centre
    a_drive = (W_grid[ar * dim + ac_col] / peak_v).clamp(0.0, 1.0)
    a_drive = torch.where(a_drive < 0.01, zero_gc, a_drive)
    rates_a = a_drive * max_rate

    targets, positions = [], []
    for r in range(0, dim, max(1, stride)):
        for c in range(0, dim, max(1, stride)):
            if r == ar and c == ac_col:
                continue
            if abs(r - ar) > view_r or abs(c - ac_col) > view_r:
                continue                              # outside the window
            targets.append((r, c))
            positions.append((r - ar, c - ac_col))    # relative, for nn_ratio
    pos = torch.tensor(positions, dtype=torch.float32, device=DEVICE)

    was_learning = net.learning
    net.learning = False                 # never let the probe touch the weights

    pca_R, pct_R, ac_R, mc_R = [], [], [], []
    for (tr_, tc_) in targets:
        t_drive = (W_grid[tr_ * dim + tc_] / peak_v).clamp(0.0, 1.0)
        t_drive = torch.where(t_drive < 0.01, zero_gc, t_drive)
        rates_t = t_drive * max_rate
        inputs = {
            LAYER_GCA: poisson(rates_a.unsqueeze(0), gran, dt, device=DEVICE),
            LAYER_GCT: poisson(rates_t.unsqueeze(0), gran, dt, device=DEVICE),
        }
        net.reset_state_variables()
        net.run(inputs=inputs, time=gran, reward=0.0)
        pca_R.append(net.monitors[LAYER_PCA].get("s").squeeze().sum(0).float())
        pct_R.append(net.monitors[LAYER_PCT].get("s").squeeze().sum(0).float())
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

    return {"PC_A": codeStats(torch.stack(pca_R), pos, spike_thresh),
            "PC_T": codeStats(torch.stack(pct_R), pos, spike_thresh),
            "AC": codeStats(torch.stack(ac_R), pos, spike_thresh),
            "MC": {**codeStats(MC, pos, spike_thresh), **mc_extra},
            "_n_probe": len(targets)}

def reportEncodingQuality(q, title="Encoding quality"):
    """Print the diagnostic and, critically, name the right knob when it fails."""
    print(f"\n[{title}]  {q['_n_probe']} probe positions, "
          f"active = >={args.diag_thresh} spikes / {args.granularity} ms",
          flush=True)
    print("  layer   active  sparsity  coverage  overlap    sep     cos  nn_ratio"
          "   spikes/cell", flush=True)
    for k in ("PC_A", "PC_T", "AC", "MC"):
        v = q[k]
        sep = "   --  " if v["sep"] is None else f"{v['sep']:7.1f}"
        print(f"  {k:5s} {v['avg_active']:7.1f}  {v['sparsity']:8.3f}  "
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

    if q["AC"]["sep"] is not None and q["PC_T"]["sep"] is not None \
            and q["AC"]["sep"] < q["PC_T"]["sep"]:
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

def rasterSubsample(spk, max_n):
    """(time, n) -> (time, <=max_n). Evenly spaced neuron subset, chosen so it does
    not bias toward low indices. max_n <= 0 means keep every neuron."""
    n = spk.shape[1]
    if max_n <= 0 or n <= max_n:
        return spk
    idx = torch.linspace(0, n - 1, max_n, device=spk.device).long()
    return spk[:, idx]

def styleRasterAxes(axes, layers, gran, sizes, first_call=False):
    """
    Pin the raster axes every frame. plot_spikes() only calls set_offsets() when
    reusing axes and never rescales, so matplotlib keeps whatever limits the FIRST
    frame autoscaled to -- and since our opening frame is deliberately empty, that
    is a degenerate range that squashes every later frame into a sliver. Setting
    the limits explicitly also gives a real time axis, which plot_spikes strips.
    """
    for ax, layer in zip(axes, layers):
        ax.set_xlim(0, gran)
        ax.set_ylim(-1, sizes[layer])
        ax.set_xticks(np.linspace(0, gran, 5))
        ax.set_xlabel("")
    axes[-1].set_xlabel(f"time within decision window (ms of {gran})")
    if first_call:
        axes[0].figure.subplots_adjust(top=0.86, bottom=0.13, hspace=0.6)

def plotRasterTimeline(windows, layers, gran, ep, fname, max_points=400000):
    """
    Whole-episode spike raster: every decision window concatenated along x, with a
    dashed line at each decision boundary. This is the "timeline" view -- the GIF
    shows one window at a time, this shows the whole episode at once.

    `windows` is a list (one per decision) of {layer: (gran, n) bool tensors}.
    """
    if not windows:
        return None
    fig, axes = plt.subplots(len(layers), 1, sharex=True,
                             figsize=(max(8.0, 0.09 * len(windows) * 1.0), 2.2 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        full = torch.cat([w[layer] for w in windows], dim=0).cpu().numpy()
        t, nidx = full.nonzero()
        # Thin very dense rasters -- a scatter with millions of points is slow to
        # draw and reads as a solid block anyway.
        if t.size > max_points:
            keep = np.linspace(0, t.size - 1, max_points).astype(int)
            t, nidx = t[keep], nidx[keep]
            ax.set_ylabel(f"{layer}\n(thinned)")
        else:
            ax.set_ylabel(layer)
        ax.scatter(t, nidx, s=0.4, linewidths=0, color="k")
        for b in range(gran, len(windows) * gran, gran):
            ax.axvline(b, color="tab:red", lw=0.3, ls="--", alpha=0.5)
        ax.set_xlim(0, len(windows) * gran)
        ax.set_ylim(-1, full.shape[1])
    axes[-1].set_xlabel(f"simulation time (ms) -- dashed = decision boundary "
                        f"({gran} ms each, {len(windows)} decisions)")
    fig.suptitle(f"Spike raster timeline -- episode {ep}")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=110)
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

if __name__ == "__main__":
    main()
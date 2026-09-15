# the goal of this is to improve upon the rewarding mechanism

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")


parser = argparse.ArgumentParser()
# task / env
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--trn_eps", type=int, default=100)
parser.add_argument("--tst_eps", type=int, default=50)
parser.add_argument("--decay", type=int, default=4)
parser.add_argument("--herrs", type=int, default=0)
parser.add_argument("--diag", type=str2bool, default=False)
parser.add_argument("--randr", type=float, default=0.15)
parser.add_argument("--boundh", type=str, default="bounce")
parser.add_argument("--fit_func", type=str, default="dir")
parser.add_argument("--allow_stay", type=str2bool, default=False)
parser.add_argument("--pandas", type=str2bool, default=False)
parser.add_argument("--mute", type=str2bool, default=False)
parser.add_argument("--write", type=str2bool, default=False)
parser.add_argument("--fcycle", type=int, default=100)
parser.add_argument("--gpu", type=str2bool, default=True)
parser.add_argument("--view_r", type=int, default=14)
parser.add_argument("--make_sparse", type=str2bool, default=False)

# rendering 
parser.add_argument("--render_replays", type=str2bool, default=True)
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--plot_every", type=int, default=50)
# Raster replays: for every episode that gets an env GIF, also emit a spike-raster
# GIF with one frame per decision, plus a whole-episode raster timeline PNG.
parser.add_argument("--raster_replays", type=str2bool, default=True)
parser.add_argument("--raster_timeline", type=str2bool, default=True)
parser.add_argument("--raster_layers", type=str, nargs="+",
                    default=["PC_A", "PC_T", "AC", "MC"])
# 0 (the default) plots EVERY neuron in the layer. A positive value draws an
# evenly spaced subsample instead. Spike counts, not neuron counts, drive the
# drawing cost, so plotting all of them is cheap.
parser.add_argument("--raster_max_neurons", type=int, default=0)

# GC layer 
parser.add_argument("--gc_scales", type=int, nargs="+", default=[3, 5, 7, 11, 13])
parser.add_argument("--gc_rotations", type=int, default=7)
parser.add_argument("--gc_offsets", type=int, default=16)   # 4x4 phase grid
parser.add_argument("--gc_global_scale", type=float, default=1.0)  # paper's g
parser.add_argument("--gc_sharpness", type=float, default=1.0)     # paper's k
parser.add_argument("--gc_max_rate", type=float, default=80.0)     # 8 spikes / 100 ms

# PLACE CELL layers: one for the agent, one for the target
parser.add_argument("--n_pc", type=int, default=500)          # per place layer
parser.add_argument("--gc_pc_sparsity", type=float, default=0.05) 
parser.add_argument("--pc_lbound", type=float, default=-80.0)

# ASSOCIATION layer 
parser.add_argument("--ac", type=int, default=1000) 
parser.add_argument("--gc_ac_sparsity", type=float, default=0.12)
parser.add_argument("--pc_ac_sparsity", type=float, default=0.1)
parser.add_argument("--ac_lbound", type=float, default=-80.0) 

# MOTOR layer 100 neurons per move (5 moves)
parser.add_argument("--mc_pop", type=int, default=100)
parser.add_argument("--ac_mc_sparsity", type=float, default=0.20)
parser.add_argument("--ac_mc_init", type=float, default=0.3)
parser.add_argument("--mc_lbound", type=float, default=-80.0)
parser.add_argument("--w_max", type=float, default=1.0)
parser.add_argument("--nu", type=float, default=4e-3)

parser.add_argument("--diagnose", type=str2bool, default=True)
parser.add_argument("--diag_stride", type=int, default=4)   # probe positions
parser.add_argument("--diag_thresh", type=int, default=4)   # spikes -> "active"

# PLACE-CELL LOCATION MAP (`--pc_map True` runs it and exits before training)
parser.add_argument("--pc_map", type=str2bool, default=True)
parser.add_argument("--pc_map_stride", type=int, default=1)   # 1 = every square
parser.add_argument("--pc_map_reps", type=int, default=2)     # >=2 enables decode
parser.add_argument("--pc_map_thresh", type=int, default=4)   # spikes -> "fired"
parser.add_argument("--pc_map_top", type=int, default=10)     # idx shown per row
parser.add_argument("--pc_map_fields", type=int, default=25)  # fields to plot
parser.add_argument("--pc_map_layers", type=str, nargs="+",
                    default=["PC_A", "PC_T"])

# Parse sys.argv ONLY when this file is the entry point. When Para-HADES imports
# it, sys.argv belongs to task_dotTracing.py (--path, --agent_idx, ...) and a real
# parse would abort the worker with SystemExit(2) at import time. Passing [] gives
# the defaults above, which are the single source of truth for the architecture --
# task_dotTracing.yaml no longer carries a main.DotTracing block.
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

moveChoices = 9 if args.diag else 5
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

LAYER_GCA, LAYER_GCT = "GC_A", "GC_T"      # grid code at agent / at target
LAYER_PCA, LAYER_PCT = "PC_A", "PC_T"      # place cells for agent / target
LAYER_AC, LAYER_MC = "AC", "MC"


OPPONENT_PAIRS = [(1, 3), (2, 4)]          # up<->down, right<->left


# --------------------------------------------------------------------------- #
# Gene access. Para-HADES hands every `param:` entry back as a flat list, even
# the array: [1] scalars, so unwrap before use.
# --------------------------------------------------------------------------- #
def geneScalar(param, key):
    try:
        v = param["param"][key]
    except (TypeError, KeyError):
        raise KeyError(
            f"gene '{key}' missing from param['param']. task_dotTracing.yaml must "
            f"declare it under `param:` with array: [1]."
        )
    return float(np.asarray(v, dtype=np.float64).ravel()[0])


def geneArray(param, key, n):
    try:
        v = param["param"][key]
    except (TypeError, KeyError):
        raise KeyError(
            f"gene '{key}' missing from param['param']. task_dotTracing.yaml must "
            f"declare it under `param:` with array: [{n}]."
        )
    a = np.asarray(v, dtype=np.float32).ravel()
    if a.size != n:
        raise ValueError(
            f"gene '{key}' has {a.size} values but this config needs {n}. "
            f"Fix `array:` in task_dotTracing.yaml."
        )
    return a


def geneMatrix(param, key, rows, cols):
    """A weight-matrix gene: the DENSE (rows, cols) matrix flattened row-major.

    Every synaptic pathway in this network draws its weights from a gene of this
    shape, so EVOL searches the whole connectome rather than a handful of gains.
    The connectivity mask is applied by the caller AFTER the reshape, so the gene
    length stays a constant the YAML can declare instead of depending on how many
    synapses a particular mask seed happened to open.
    """
    a = geneArray(param, key, rows * cols)
    return torch.from_numpy(a).to(DEVICE).view(rows, cols)

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

    past_performances = []

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

        avg_reward = 0

        while not done:
            step += 1

            # get prev row+col and calc dist
            pr, pc = env.netDot.row[0], env.netDot.col[0]
            # get target row+col before the step
            tr, tc = env.dots[0].row[0], env.dots[0].col[0]
            prev_dist = np.hypot(pr - env.dots[0].row[0], pc - env.dots[0].col[0])

            # take a step
            obs, _, done, intercept = env.step(action)

            # agent row+col after the step
            cr, cc = env.netDot.row[0], env.netDot.col[0]

            # calc change in dist
            curr_dist = np.hypot(cr - tr, cc - tc)
            r = prev_dist - curr_dist

            if intercept:
                r += 10.0

            avg_reward += r
                
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

            # save raster frames and gif replay
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

        # create a sliding history window of past rewards
        if len(past_performances) < 25:
            past_performances.insert(0,avg_reward/100)
        else:
            past_performances.insert(0,avg_reward/100)
            past_performances.pop()
            avg_perf = sum(past_performances)/len(past_performances)
            print(f"AVERAGE REWARD PAST 25: {avg_perf}", flush=True)

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

    # use the average performance of final 25 as the fitness score
    if not past_performances:
        return 0.0
    return sum(past_performances)/len(past_performances)


def run_task(param, task_args=None):
    """
    param      -- one Para-HADES candidate. param['param'] holds the decoded
                  genes: w_ac_mc plus the six/seven scalers.
    task_args  -- the Para-HADES slurm namespace (path, agent_idx, ...). It is
                  NOT the config namespace; deliberately not named `args` so it
                  cannot shadow the module-level parser defaults, which are what
                  every architecture knob is read from.
    Returns {'fitnessScore': float, ...}.
    """
    global OUT_FILE_PATH

    # Per-candidate output folder: parallel agents must not share one.
    if task_args is not None and getattr(task_args, "path", None):
        OUT_FILE_PATH = os.path.join(
            task_args.path, "dotTracing_out",
            f"agent{getattr(task_args, 'agent_idx', 0)}"
            f"_test{getattr(task_args, 'agent_test_num', 0)}") + os.sep
    os.makedirs(OUT_FILE_PATH, exist_ok=True)

    # Seed here, not at import: the module is imported once but run_task is called
    # once per candidate, so import-time seeding never re-seeds between candidates.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- decode the scalar genes ------------------------------------------ #
    s_gc_pc1   = geneScalar(param, "scaler_gc_pc1")
    s_gc_pc2   = geneScalar(param, "scaler_gc_pc2")
    s_pc_ac1   = geneScalar(param, "scaler_pc_ac1")
    s_pc_ac2   = geneScalar(param, "scaler_pc_ac2")
    s_ac_mc    = geneScalar(param, "scaler_ac_mc")
    s_inhib_ac = geneScalar(param, "scaler_inhib_ac")
    s_inhib_mc = geneScalar(param, "scaler_inhib_mc")
    print(f"[gene] gc_pc={s_gc_pc1:.3f}/{s_gc_pc2:.3f}  "
          f"pc_ac={s_pc_ac1:.3f}/{s_pc_ac2:.3f}  ac_mc={s_ac_mc:.3f}  "
          f"inhib_ac={s_inhib_ac:.3f}  inhib_mc={s_inhib_mc:.3f}", flush=True)

    dim = args.dim
    # store all coordinate pairs
    world_xy = np.array([(r, c) for r in range(dim) for c in range(dim)],
                        dtype=np.float32)               
    # centre the lattice on the board so rotations pivot about the middle
    board_centre = np.array([(dim - 1) / 2.0, (dim - 1) / 2.0], dtype=np.float32)
    world_rel = world_xy - board_centre                 
    extent = dim / 2.0

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
                  rest=-64.0, reset=-70.0, thresh=-50.0,
                  refrac=1, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.ac_lbound)

    # neurons 0-99: pause, 100-199: up, 200-299: right, 300-399: down, 400-499: left
    n_mc = moveChoices * args.mc_pop
    mc = LIFNodes(n=n_mc, traces=True,
                  rest=-64.0, reset=-64.0, thresh=-55.0,
                  refrac=0, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.mc_lbound)

    net.add_layer(gc_a, name=LAYER_GCA)
    net.add_layer(gc_t, name=LAYER_GCT)
    net.add_layer(pc_a, name=LAYER_PCA)
    net.add_layer(pc_t, name=LAYER_PCT)
    net.add_layer(ac,   name=LAYER_AC)
    net.add_layer(mc,   name=LAYER_MC)

    # grid cell to place cell connections
    # 5% sparsity
    # The mask (which synapses exist) is fixed architecture drawn from args.seed;
    # the weights themselves are the w_gc_pc_* genes.
    gen_a = torch.Generator(device="cpu").manual_seed(args.seed + 10)
    gc_pc_mask_a = (torch.rand(n_gc, args.n_pc, generator=gen_a)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_a = gc_pc_mask_a * geneMatrix(param, "w_gc_pc_a", n_gc, args.n_pc)
    W_gc_pc_a = (W_gc_pc_a / gc_pc_mask_a.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * s_gc_pc1

    gen_t = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    gc_pc_mask_t = (torch.rand(n_gc, args.n_pc, generator=gen_t)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_t = gc_pc_mask_t * geneMatrix(param, "w_gc_pc_t", n_gc, args.n_pc)
    W_gc_pc_t = (W_gc_pc_t / gc_pc_mask_t.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * s_gc_pc2

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
    # decrease agent place cell influence slightly
    W_pc_ac_a = (pc_ac_a_mask * geneMatrix(param, "w_pc_ac_a", args.n_pc, args.ac)
                 / np.sqrt(dense_fan)) * s_pc_ac1
    pc_ac_t_mask = (torch.rand(args.n_pc, args.ac, generator=gen_t) 
                        <= args.pc_ac_sparsity).float().to(DEVICE)
    W_pc_ac_t = (pc_ac_t_mask * geneMatrix(param, "w_pc_ac_t", args.n_pc, args.ac)
                 / np.sqrt(dense_fan)) * s_pc_ac2

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
    # The reservoir is dense, so there is no mask: the gene IS the matrix. It is
    # declared signed in the YAML (min -1 / max 1) because it replaces a randn()
    # draw and the layer needs both excitatory and inhibitory recurrence.
    W_rec = geneMatrix(param, "w_rec", args.ac, args.ac)
    # normalize each by sum of their columns and mult by scale factor
    # W_rec = W_rec / W_rec.abs().sum(0, keepdim=True).clamp(min=1e-6) 
    # YAML allows scaler_inhib_ac down to 0.0, and this is a DIVISOR, so clamp:
    # at 0 the recurrent weights blow up to inf and the whole episode goes NaN.
    W_rec = W_rec / max(s_inhib_ac, 1e-3)
    W_rec.fill_diagonal_(0.0)
    feat_rec = Weight(name="w_rec", value=W_rec)
    net.add_connection(
        MulticompartmentConnection(source=ac, target=ac,
                                   pipeline=[feat_rec], device=DEVICE),
        source=LAYER_AC, target=LAYER_AC)

    # association to motor layer, the only learning layer
    ac_mc_gen = torch.Generator(device="cpu").manual_seed(args.seed + 4)
    ac_mc_mask_gen = (torch.rand(args.ac, n_mc, generator=ac_mc_gen)
                  < args.ac_mc_sparsity).to(DEVICE) # 20% sparsity for ac->mc

    ac_mc_mask = Mask(name='ac_mc_mask', value=ac_mc_mask_gen)

    # The w_ac_mc gene is one value per (AC neuron, ACTION) -- args.ac * moveChoices
    # -- not per (AC neuron, MC neuron). Each value is broadcast across the mc_pop
    # neurons of its action subpopulation, so EVOL searches args.ac * moveChoices
    # dimensions (5000 here) while MSTDPET still refines all args.ac * n_mc synapses
    # individually over the lifetime. A full per-synapse gene is also accepted.
    n_gene_action = args.ac * moveChoices
    n_gene_dense = args.ac * n_mc
    raw = np.asarray(param["param"]["w_ac_mc"], dtype=np.float32).ravel()
    if raw.size == n_gene_action:
        g = torch.from_numpy(
            geneArray(param, "w_ac_mc", n_gene_action)).to(DEVICE)
        W_ac_mc = g.view(args.ac, moveChoices).repeat_interleave(args.mc_pop, dim=1)
    elif raw.size == n_gene_dense:
        g = torch.from_numpy(
            geneArray(param, "w_ac_mc", n_gene_dense)).to(DEVICE)
        W_ac_mc = g.view(args.ac, n_mc)
    else:
        raise ValueError(
            f"gene 'w_ac_mc' has {raw.size} values. With ac={args.ac}, "
            f"mc_pop={args.mc_pop}, moveChoices={moveChoices} it must be "
            f"array: [{n_gene_action}] (per action, broadcast) or "
            f"array: [{n_gene_dense}] (per synapse). Fix task_dotTracing.yaml."
        )
    # apply the evolved scaling number to the acmc weight
    W_ac_mc = ac_mc_mask_gen * W_ac_mc * s_ac_mc
 
    feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc, range=[0,5],
                        learning_rule=MSTDPET, nu=[args.nu, args.nu])
    net.add_connection(
        MulticompartmentConnection(source=ac, target=mc,
                                   pipeline=[feat_ac_mc,ac_mc_mask], device=DEVICE),
        source=LAYER_AC, target=LAYER_MC)

    # MOTOR opponent inhibition (MC -> MC, frozen)
    feat_mc_opp = None
    W_mc_opp = torch.zeros(n_mc, n_mc, device=DEVICE)
    # scaler_inhib_mc is declared in the YAML as min -5.0 / max 0.0, i.e. the gene
    # is ALREADY negative. Negating it here (as the original line did) would make
    # opponent pairs excite each other, so use it as-is.
    # w_mc_opp holds one MAGNITUDE per directed opponent block, in the order
    # (a->b, b->a) for each entry of OPPONENT_PAIRS, so the two directions of a
    # pair can evolve asymmetrically. s_inhib_mc supplies the (negative) sign.
    g_opp = geneArray(param, "w_mc_opp", 2 * len(OPPONENT_PAIRS))
    for i, (a, b) in enumerate(OPPONENT_PAIRS): # two pairs - up down and right left
        sa = slice(a * args.mc_pop, (a + 1) * args.mc_pop)
        sb = slice(b * args.mc_pop, (b + 1) * args.mc_pop)
        W_mc_opp[sa, sb] = float(g_opp[2 * i]) * s_inhib_mc      # a inhibits b
        W_mc_opp[sb, sa] = float(g_opp[2 * i + 1]) * s_inhib_mc  # b inhibits a

    feat_mc_opp = Weight(name="w_mc_opp", value=W_mc_opp)
    net.add_connection(
        MulticompartmentConnection(source=mc, target=mc,
                                    pipeline=[feat_mc_opp], device=DEVICE),
        source=LAYER_MC, target=LAYER_MC)

    net.to(DEVICE)
    weight_features = {"gc_pc_a": feat_gc_pc_a, "gc_pc_t": feat_gc_pc_t,
                       "pc_ac_a": feat_pc_ac_a, "pc_ac_t": feat_pc_ac_t,
                       "recurrent": feat_rec, "ac_mc": feat_ac_mc}
    if feat_mc_opp is not None:
        weight_features["mc_opp"] = feat_mc_opp

    print(f"[arch] GC {n_gc} x2  ->  PC {args.n_pc} x2 (sparse "
          f"{args.gc_pc_sparsity:.0%})  ->  AC {args.ac} (dense)  ->  MC {n_mc}")
    print(f"[gate] view_r={args.view_r} on a {dim}x{dim} board: target is visible "
          f"when Chebyshev distance <= {args.view_r}")

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
                                time=int(args.granularity / args.dt), device=DEVICE)
        net.add_monitor(spikes[layer], name=layer)
 
    environment = DotSimulator(
        args.steps, decay=args.decay, herrs=args.herrs, diag=args.diag,
        randr=args.randr, write=args.write, mute=args.mute,
        bound_hand=args.boundh, fit_func=args.fit_func, 
        allow_stay=args.allow_stay, pandas=args.pandas, fpath=OUT_FILE_PATH,
    )
    environment.reset()

    print("Training:")
    environment.addFileSuffix("train")
    train_fitness = runSimulator(
        net, environment, spikes, args.trn_eps, W_grid, peak,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_train", view_r=args.view_r, dim=dim,
    )
    net.learning = False
    print("Testing:")
    environment.changeFileSuffix("train", "test")
    fitness = runSimulator(
        net, environment, spikes, args.tst_eps, W_grid, peak,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "test"), pfname=genFileName("perf", "test"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_test", view_r=args.view_r, dim=dim,
    )

    # task_dotTracing.main() indexes outP['fitnessScore'], so return a dict.
    return {
        "fitnessScore": float(fitness),
        "train_fitness": float(train_fitness),
        "out_path": OUT_FILE_PATH,
    }


def plotWeightGraphs(weight_features, ep, out_path=None):
    out_path = out_path or OUT_FILE_PATH
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

def plotMotorWeightPolar(w_ac_mc, active_mask, ep, out_path=None):
    """
    The paper's Figure 9: fraction of total AC->MC weight allocated to each motor
    population, split by active vs inactive ACs at the current state.
    """
    out_path = out_path or OUT_FILE_PATH
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
    os.makedirs(OUT_FILE_PATH, exist_ok=True)
    return os.path.join(OUT_FILE_PATH,
                        ftype + "_s" + str(t) + "_" + suffix + ".csv")


# --------------------------------------------------------------------------- #
# Standalone entry point. Para-HADES normally supplies `param`; when this file is
# run directly there is no GA, so synthesise a candidate from the `param:` block
# of task_dotTracing.yaml (midpoint of each declared range).
# --------------------------------------------------------------------------- #
def standaloneParam(yaml_path=None):
    import yaml
    yaml_path = yaml_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "task_dotTracing.yaml")
    with open(yaml_path) as f:
        spec = (yaml.safe_load(f) or {}).get("param", {}) or {}
    genes = {}
    rng = np.random.default_rng(args.seed)
    for key, d in spec.items():
        n = int((d.get("array") or [1])[0])
        lo, hi = float(d.get("min", 0.0)), float(d.get("max", 1.0))
        if n == 1:
            # A scaler: the midpoint of its range is the neutral standalone value.
            genes[key] = [0.5 * (lo + hi)]
        else:
            # A weight matrix: a constant midpoint would make every neuron in the
            # layer see the identical input, so draw uniformly over the declared
            # range instead -- the same distribution the pre-gene code used.
            genes[key] = rng.uniform(lo, hi, size=n).astype(np.float32).tolist()
    return {"param": genes}


# task_dotTracing.py (the Para-HADES worker) calls dot_tracingtask.do_task(param).
do_task = run_task


if __name__ == "__main__":
    out = run_task(standaloneParam())
    print("\n[standalone] result")
    for k, v in out.items():
        print(f"  {k}: {v}")

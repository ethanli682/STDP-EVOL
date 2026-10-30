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
parser.add_argument("--env_seed", type=int, default=None)  # optional reproducible environment (0 means wall-clock in DotSimulator)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--trn_eps", type=int, default=5)
parser.add_argument("--tst_eps", type=int, default=5)
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
parser.add_argument("--render_every", type=int, default=3)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--plot_every", type=int, default=3)
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
parser.add_argument("--gc_scales", type=int, nargs="+", default=[5, 7, 11, 13]) # dec by one
parser.add_argument("--gc_rotations", type=int, default=6) # dec by one
parser.add_argument("--gc_offsets", type=int, default=16)   # 4x4 phase grid
parser.add_argument("--gc_global_scale", type=float, default=1.0)  # paper's g
parser.add_argument("--gc_sharpness", type=float, default=1.0)     # paper's k
parser.add_argument("--gc_max_rate", type=float, default=80.0)     # 8 spikes / 100 ms

# PLACE CELL layers: one for the agent, one for the target
parser.add_argument("--n_pc", type=int, default=500)          # per place layer
parser.add_argument("--pc_lbound", type=float, default=-80.0)

# ASSOCIATION layer 
parser.add_argument("--ac", type=int, default=1000) 
parser.add_argument("--gc_ac_sparsity", type=float, default=0.12)
parser.add_argument("--ac_lbound", type=float, default=-80.0)
parser.add_argument("--ac_penalty", type=float, default=1.0)   # scales the punishment for distance from the 25%-65% AC activity band

# MOTOR layer 100 neurons per move (5 moves)
parser.add_argument("--mc_pop", type=int, default=100)
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

# Where every CSV / PNG / GIF this run produces is written. run_task() overrides it
# per candidate when Para-HADES supplies a task namespace; this module-level default
# is what a standalone run -- and any HADES call that does not pass one -- uses.
OUT_FILE_PATH = "dotTracing_out" + os.sep


OPPONENT_PAIRS = [(1, 3), (2, 4)]          # up<->down, right<->left
AC_LO, AC_HI = 0.40, 0.60                  # healthy AC firing-rate band


# --------------------------------------------------------------------------- #
# Gene access. Para-HADES hands every `param:` entry back as a flat list, even
# the array: [1] scalars, so unwrap before use.
# --------------------------------------------------------------------------- #
def geneScalar(param, key, default=None):
    try:
        v = param["param"][key]
    except (TypeError, KeyError):
        if default is not None:
            return float(default)
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
                 dim=28, diagnostics=None, reward_type=2, target_spikes=8.0):
    if not np.isfinite(target_spikes) or target_spikes <= 0:
        raise ValueError("target_spikes must be finite and positive")
    dt = net.dt
    peak_v = peak.squeeze(0)                       # (n_gc,) hoisted out of the loop
    zero_gc = torch.zeros(W_grid.shape[1], device=DEVICE)
    rng = np.random.default_rng(args.seed)
    spike_ims = spike_axes = None

    past_performances = []

    spike_mean_sums = {name: 0.0 for name in (
        LAYER_GCA, LAYER_GCT, LAYER_PCA, LAYER_PCT, LAYER_AC, LAYER_MC)}
    active_fraction_sums = dict.fromkeys(spike_mean_sums, 0.0)
    active_spike_sums = dict.fromkeys(spike_mean_sums, 0.0)
    active_neuron_windows = dict.fromkeys(spike_mean_sums, 0)
    layer_fitness_sums = dict.fromkeys(spike_mean_sums, 0.0)
    total_windows = 0

    # Accumulate activity across all decision windows in this call.
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
        clock = time.time()

        sum_reward = 0

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

            sum_reward += r
                
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

            gca = spikes[LAYER_GCA].get("s").squeeze()         
            gct = spikes[LAYER_GCT].get("s").squeeze()
            pca = spikes[LAYER_PCA].get("s").squeeze()          
            pct = spikes[LAYER_PCT].get("s").squeeze()
            mc = spikes[LAYER_MC].get("s").squeeze()          # (time, n_mc)
            ac = spikes[LAYER_AC].get("s").squeeze()

            # Score each window separately: 50% participation, with active
            # neurons individually close to target_spikes. Deviations cannot
            # cancel across neurons or across windows.
            total_windows += 1
            for name, layer_spikes in (
                (LAYER_GCA, gca), (LAYER_GCT, gct),
                (LAYER_PCA, pca), (LAYER_PCT, pct),
                (LAYER_AC, ac), (LAYER_MC, mc),
            ):
                counts = layer_spikes.float().reshape(-1, net.layers[name].n).sum(0)
                active = counts > 0
                active_fraction = active.float().mean().item()
                participation_score = 1.0 - 2.0 * abs(active_fraction - 0.5)
                active_counts = counts[active]
                count_score = 0.0
                if active_counts.numel():
                    count_score = (
                        1.0 - ((active_counts - target_spikes) / target_spikes).square()
                    ).mean().item()
                # Four times the original combined score: ideal +2, silence -2.
                layer_fitness_sums[name] += 2.0 * (
                    participation_score + count_score - 1.0
                )
                spike_mean_sums[name] += counts.mean().item()
                active_fraction_sums[name] += active_fraction
                active_spike_sums[name] += active_counts.sum().item()
                active_neuron_windows[name] += active_counts.numel()

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

       
        # when AC activity is out of range, the punishment scales with how far
        # the firing rate is from the band, instead of task performance
        # ac_penalty = 0.0
        # if ac_out_of_range:
        #     ac_penalty = args.ac_penalty * max(AC_LO - ac_frac, ac_frac - AC_HI)
        #     ac_penalty_sum += ac_penalty
        #     out_of_range_cnter += 1
        # perf = sum_reward/100
        # if diagnostics is not None:
        #     diagnostics.append({
        #         "episode": ep, "steps": step, "ac_active_fraction": ac_frac,
        #         "ac_in_band": not ac_out_of_range,
        #         # "ac_decision_std": float(np.std(ac_decision_fractions)),
        #         # "ac_decisions_in_band_fraction": float(np.mean([
        #         #     AC_LO <= v <= AC_HI for v in ac_decision_fractions])),
        #         "mc_active_fraction": mc_frac_sum / max(1, step),
        #         "total_reward": float(total_reward), "task_performance": float(perf),
        #         "ac_penalty": float(ac_penalty), "intercepts": intercepts,
        #     })

        # # create a sliding history window of past rewards
        # if len(past_performances) < args.trn_eps:
        #     past_performances.insert(0,perf)
        # else:
        #     past_performances.insert(0,perf)
        #     past_performances.pop()
        #     avg_perf = sum(past_performances)/len(past_performances)
        #     print(f"AVERAGE REWARD PAST 5: {avg_perf}", flush=True)

        if net.reward_fn is not None:
            net.reward_fn.update(accumulated_reward=total_reward, steps=step)

        # plotting + diagnostics code
        print(f"Episode {ep}: total reward {total_reward:.2f}"
              f"intercepts {intercepts} "
              f"target visible {visible_steps / max(1, step):.0%} | ")
            #   f"AC active {ac_frac:.1%}, "
            #   f"MC active {mc_frac_sum / max(1, step):.1%} | "
            #   f"AC penalty {ac_penalty:.3f}")
        # os.makedirs(OUT_FILE_PATH, exist_ok=True)
        # if capturing and replay_frames:
        #     saveReplayGIF(replay_frames,
        #                   os.path.join(OUT_FILE_PATH,
        #                                f"{replay_prefix}_ep{ep:05d}.gif"),
        #                   fps=replay_fps)
        # if capturing and raster_frames:
        #     # Frame count matches the env GIF (both open on the initial state), so
        #     # frame i is the same instant in both and they play in lockstep.
        #     saveReplayGIF(raster_frames,
        #                   os.path.join(OUT_FILE_PATH,
        #                                f"{replay_prefix}_raster_ep{ep:05d}.gif"),
        #                   fps=replay_fps)
        # if capturing and raster_windows:
        #     plotRasterTimeline(
        #         raster_windows, args.raster_layers, gran, ep,
        #         os.path.join(OUT_FILE_PATH,
        #                      f"{replay_prefix}_rastertl_ep{ep:05d}.png"))
        if capturing and raster_fig is not None:
            plt.close(raster_fig)
        # if weight_features is not None and args.plot_every and ep % render_every == 0:
            # plotWeightGraphs(weight_features, ep)
            # plotMotorWeightPolar(feat_ac_mc.value, last_active_ac, ep)
        if args.write and rfname:
            with open(rfname, "ab") as f:
                np.savetxt(f, rewards, delimiter=",", fmt="%.6f")
        if pfname:
            with open(pfname, "a+") as f:
                f.write(("," if ep else "") + str(intercepts))
        if ep % args.fcycle == 0:
            env.cycleOutFiles()

    # Average the already-computed window scores, not the counts before scoring.
    selected_layers = {
        1: (LAYER_GCA, LAYER_GCT),
        2: (LAYER_PCA, LAYER_PCT),
        3: (LAYER_AC,),
        4: (LAYER_MC,),
    }.get(reward_type)
    if selected_layers is not None:
        if total_windows == 0:
            raise ValueError("Cannot calculate spike-count fitness without decision windows")
        fitness = 0.0
        summaries = []
        for name in selected_layers:
            mean_spikes = spike_mean_sums[name] / total_windows
            active_fraction = active_fraction_sums[name] / total_windows
            active_mean = active_spike_sums[name] / max(1, active_neuron_windows[name])
            layer_fitness = layer_fitness_sums[name] / total_windows
            fitness += layer_fitness
            summaries.append(
                f"{name} active={active_fraction:.2%}, "
                f"active_mean={active_mean:.4f} spikes/active-neuron/window, "
                f"all_neuron_mean={mean_spikes:.4f} "
                f"(fitness={layer_fitness:+.4f})"
            )
        print(
            f"[Activity fitness | {replay_prefix}] target_active=50%, "
            f"target_spikes={target_spikes:g}/active-neuron/window | " + " | ".join(summaries)
            + f" | Combined fitness={fitness:+.4f} | windows={total_windows}",
            flush=True,
        )
        return float(fitness)
    return sum(past_performances)/len(past_performances)


def do_task(param, task_args=None):
    """
    param      -- one Para-HADES candidate. param['param'] holds the decoded
                  genes: one sparsity_* per pathway plus the scalers.
    task_args  -- the Para-HADES slurm namespace (path, agent_idx, ...). It is
                  NOT the config namespace; deliberately not named `args` so it
                  cannot shadow the module-level parser defaults, which are what
                  every architecture knob is read from.
    Returns {'fitnessScore': float, ...}.
    """
    # global OUT_FILE_PATH

    # # Per-candidate output folder: parallel agents must not share one.
    # if task_args is not None and getattr(task_args, "path", None):
    #     OUT_FILE_PATH = os.path.join(
    #         task_args.path, "dotTracing_out",
    #         f"agent{getattr(task_args, 'agent_idx', 0)}"
    #         f"_test{getattr(task_args, 'agent_test_num', 0)}") + os.sep
    # os.makedirs(OUT_FILE_PATH, exist_ok=True)

    # Seed here, not at import: the module is imported once but run_task is called
    # once per candidate, so import-time seeding never re-seeds between candidates.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- decode the scalar genes ------------------------------------------ #
    s_gc_pc1   = geneScalar(param, "scaler_gc_pc1")
    s_gc_pc2   = geneScalar(param, "scaler_gc_pc2")
    # Fixed baselines while downstream genes are omitted from the YAML.
    # Restoring a gene under param makes its candidate value take precedence.
    s_pc_ac1   = geneScalar(param, "scaler_pc_ac1", default=1.0)
    s_pc_ac2   = geneScalar(param, "scaler_pc_ac2", default=1.0)
    s_ac_mc    = geneScalar(param, "scaler_ac_mc", default=1.0)
    s_inhib_ac = geneScalar(param, "scaler_inhib_ac", default=1.0)
    s_inhib_mc = geneScalar(param, "scaler_inhib_mc", default=1.0)

    pc_tc_trace = geneScalar(param, "tc_trace", default=20.0)
    pc_tc_decay = geneScalar(param, "tc_decay", default=20.0)


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

        for r_i in range(6): # 6 rotations
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

    print(f"USED TC_DECAY: {pc_tc_decay:.3f} USED TC_TRACE: {pc_tc_trace:.3f}",flush=True)

    # agent place cells
    pc_a = LIFNodes(n=args.n_pc, traces=True,
                    rest=-64.0, reset=-70.0, thresh=-61.0,
                    refrac=1, tc_decay=pc_tc_decay, tc_trace=pc_tc_trace,
                    lbound=args.pc_lbound)
    # target place cells
    pc_t = LIFNodes(n=args.n_pc, traces=True,
                    rest=-64.0, reset=-70.0, thresh=-61.0,
                    refrac=1, tc_decay=pc_tc_decay, tc_trace=pc_tc_trace,
                    lbound=args.pc_lbound)

    ac = LIFNodes(n=args.ac, traces=True,
                  rest=-64.0, reset=-70.0, thresh=-62.0,
                  refrac=1, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.ac_lbound)

    # neurons 0-99: pause, 100-199: up, 200-299: right, 300-399: down, 400-499: left
    n_mc = moveChoices * args.mc_pop
    mc = LIFNodes(n=n_mc, traces=True,
                  rest=-64.0, reset=-64.0, thresh=-62.0,
                  refrac=0, tc_decay=20.0, tc_trace=20.0,
                  lbound=args.mc_lbound)

    net.add_layer(gc_a, name=LAYER_GCA)
    net.add_layer(gc_t, name=LAYER_GCT)
    net.add_layer(pc_a, name=LAYER_PCA)
    net.add_layer(pc_t, name=LAYER_PCT)
    net.add_layer(ac,   name=LAYER_AC)
    net.add_layer(mc,   name=LAYER_MC)

    # weight values: separate generator so the mask streams are unaffected
    w_gen = torch.Generator(device="cpu").manual_seed(29)

    # grid cell to place cell connections
    gen_a = torch.Generator(device="cpu").manual_seed(20)
    gc_pc_mask_a = (torch.rand(n_gc, args.n_pc, generator=gen_a)
                    < 0.4).float().to(DEVICE)
    W_gc_pc_a = gc_pc_mask_a * torch.rand(n_gc, args.n_pc, generator=w_gen).to(DEVICE)
    W_gc_pc_a = (W_gc_pc_a / gc_pc_mask_a.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * s_gc_pc1

    gen_t = torch.Generator(device="cpu").manual_seed(23)
    gc_pc_mask_t = (torch.rand(n_gc, args.n_pc, generator=gen_t)
                    < 0.4).float().to(DEVICE)
    W_gc_pc_t = gc_pc_mask_t * torch.rand(n_gc, args.n_pc, generator=w_gen).to(DEVICE)
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

    # PLACE to ASSOCIATION connection
    dense_fan = float(2 * args.n_pc)
    pc_ac_a_mask = (torch.rand(args.n_pc, args.ac, generator=gen_a)
                    < 0.6).float().to(DEVICE)
    W_pc_ac_a = (pc_ac_a_mask
                 * torch.rand(args.n_pc, args.ac, generator=w_gen).to(DEVICE)
                 / np.sqrt(dense_fan)) * s_pc_ac1
    pc_ac_t_mask = (torch.rand(args.n_pc, args.ac, generator=gen_t)
                    < 0.6).float().to(DEVICE)
    W_pc_ac_t = (pc_ac_t_mask
                 * torch.rand(args.n_pc, args.ac, generator=w_gen).to(DEVICE)
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

    W_rec = torch.randn(args.ac, args.ac, generator=w_gen).to(DEVICE)
    W_rec = W_rec * s_inhib_ac
    W_rec.fill_diagonal_(0.0)
    feat_rec = Weight(name="w_rec", value=W_rec)
    net.add_connection(
        MulticompartmentConnection(source=ac, target=ac,
                                   pipeline=[feat_rec], device=DEVICE),
        source=LAYER_AC, target=LAYER_AC)

    # association to motor layer, the only learning layer
    ac_mc_gen = torch.Generator(device="cpu").manual_seed(args.seed + 4)
    ac_mc_mask_gen = (torch.rand(args.ac, n_mc, generator=ac_mc_gen)
                      < 0.7).to(DEVICE)

    ac_mc_mask = Mask(name='ac_mc_mask', value=ac_mc_mask_gen)

    # every existing synapse starts at ac_mc_init; MSTDPET shapes them from there
    W_ac_mc = ac_mc_mask_gen * args.ac_mc_init * s_ac_mc
 
    feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc, range=[0,5],
                        learning_rule=MSTDPET, nu=[args.nu, args.nu])
    net.add_connection(
        MulticompartmentConnection(source=ac, target=mc,
                                   pipeline=[feat_ac_mc,ac_mc_mask], device=DEVICE),
        source=LAYER_AC, target=LAYER_MC)

    # MOTOR opponent inhibition (MC -> MC, frozen)
    feat_mc_opp = None
    W_mc_opp = torch.zeros(n_mc, n_mc, device=DEVICE)
    # scaler_inhib_mc is declared in the YAML as min -3.0 / max 0.0, i.e. the gene
    # is ALREADY negative, so it is used as-is for every existing opponent synapse.
    # sparsity_mc_opp holds one connection probability per directed opponent
    # block, in the order (a->b, b->a) for each entry of OPPONENT_PAIRS, so the
    # two directions of a pair can evolve asymmetrically.
    opp_gen = torch.Generator(device="cpu").manual_seed(13)
    for i, (a, b) in enumerate(OPPONENT_PAIRS): # two pairs - up down and right left
        sa = slice(a * args.mc_pop, (a + 1) * args.mc_pop)
        sb = slice(b * args.mc_pop, (b + 1) * args.mc_pop)
        m_ab = torch.rand(args.mc_pop, args.mc_pop, generator=opp_gen)
        m_ba = torch.rand(args.mc_pop, args.mc_pop, generator=opp_gen)
        W_mc_opp[sa, sb] = m_ab.float().to(DEVICE) * s_inhib_mc  # a inhibits b
        W_mc_opp[sb, sa] = m_ba.float().to(DEVICE) * s_inhib_mc  # b inhibits a

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

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
                                time=int(args.granularity / args.dt), device=DEVICE)
        net.add_monitor(spikes[layer], name=layer)
 
    env_options = {} if args.env_seed is None else {"seed": args.env_seed}
    environment = DotSimulator(
        args.steps, decay=args.decay, herrs=args.herrs, diag=args.diag,
        randr=args.randr, write=args.write, mute=args.mute,
        bound_hand=args.boundh, fit_func=args.fit_func, 
        allow_stay=args.allow_stay, pandas=args.pandas, fpath=OUT_FILE_PATH,
        **env_options,
    )
    environment.reset()

    print("Training:")
    train_diagnostics, test_diagnostics = [], []
    environment.addFileSuffix("train")
    train_fitness = runSimulator(
        net, environment, spikes, args.trn_eps, W_grid, peak,
        feat_ac_mc, gran=args.granularity,
        rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
        weight_features=weight_features, render_replays=args.render_replays,
        render_every=args.render_every, replay_fps=args.replay_fps,
        replay_prefix="replay_train", view_r=args.view_r, dim=dim,
        diagnostics=train_diagnostics, reward_type=2
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
        diagnostics=test_diagnostics, reward_type=2
    )

    print(f"RETURNED FITNESS: {fitness}", flush=True)

    # task_dotTracing.main() indexes outP['fitnessScore'], so return a dict.
    return {
        "fitnessScore": float(fitness),
        "train_fitness": float(train_fitness),
        "train_episodes": train_diagnostics,
        "test_episodes": test_diagnostics,
        "ac_active_fraction": float(np.mean([e["ac_active_fraction"] for e in test_diagnostics])) if test_diagnostics else None,
        "ac_all_episodes_in_band": all(e["ac_in_band"] for e in test_diagnostics) if test_diagnostics else None,
        # "out_path": OUT_FILE_PATH,
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



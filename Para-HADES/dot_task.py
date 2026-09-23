# the goal of this is to improve upon the rewarding mechanism

import argparse
import itertools
import os
import time

import numpy as np
import torch

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
#todo test gpu vs cpu without hades
parser.add_argument("--trn_eps", type=int, default=25)
parser.add_argument("--tst_eps", type=int, default=10)
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

parser.add_argument("--gain_gc_pc", type=float, default=10.0)
parser.add_argument("--gain_pc_ac", type=float, default=5.0)
parser.add_argument("--gain_ac_mc", type=float, default=0.2)

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

OUT_FILE_PATH = "dotTracing_out11" + os.sep

# Fallback evaluation counter. One task process evaluates several candidates
# (task_dotTracing.main loops genes x repeats), and every one of them used to
# write the same filenames into the same folder, so only the LAST candidate's
# reward/perf logs survived. When the caller does not tell us which candidate
# this is, this monotonic counter keeps them apart instead.
_EVAL_SEQ = itertools.count()


def _outDirFor(task_args):
    """Per-candidate output directory.

    Layout: <path>/dotTracing_out/agent<idx>/epoch<counter>_gene<g>_test<t>/

    Separates the three axes that actually vary: agents run as independent
    processes and must never touch each other's files; epochs are successive
    generations; genes/repeats are the candidates within one dispatch. Re-running
    the same epoch reuses the same names, which is the one overwrite we want.
    """
    if task_args is None or not getattr(task_args, "path", None):
        return OUT_FILE_PATH

    def clean(v, default):
        v = default if v is None else v
        # agent_counter arrives as the string "<counter>.<epoch>"; keep it
        # readable but never let a separator escape into the path.
        return str(v).replace(os.sep, "-").replace("/", "-").strip() or str(default)

    agent = clean(getattr(task_args, "agent_idx", None), 0)
    epoch = clean(getattr(task_args, "agent_counter", None), 0)
    gene = getattr(task_args, "gene_num", None)
    test = getattr(task_args, "test_rep", None)
    if gene is None and test is None:
        # Caller did not identify the candidate -- fall back to a running index
        # so candidates still land in distinct folders.
        leaf = f"epoch{epoch}_eval{next(_EVAL_SEQ)}"
    else:
        leaf = f"epoch{epoch}_gene{clean(gene, 0)}_test{clean(test, 0)}"
    return os.path.join(task_args.path, "dotTracing_out",
                        f"agent{agent}", leaf) + os.sep


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
                 gran=100, rfname="", pfname="", view_r=14,
                 dim=28):
    dt = net.dt
    peak_v = peak.squeeze(0)                       # (n_gc,) hoisted out of the loop
    zero_gc = torch.zeros(W_grid.shape[1], device=DEVICE)
    rng = np.random.default_rng(args.seed)

    past_performances = []

    for ep in range(episodes):
        total_reward, intercepts, step = 0.0, 0, 0
        visible_steps = 0                          # how often the target was in view
        rewards = np.zeros(env.timesteps)
        net.reset_state_variables()
        env.reset()
        done = False

        action = int(rng.integers(0, env.action_space.n))
        last_active_ac = torch.zeros(net.layers[LAYER_AC].n, device=DEVICE)
        clock = time.time()
        ac_frac_sum, mc_frac_sum = 0.0, 0.0
        rate_sum = {l: 0.0 for l in (LAYER_PCA, LAYER_PCT, LAYER_AC, LAYER_MC)}

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

            # Mean firing rate per layer, in Hz. A stage that reads 0.00 here has
            # broken the sensor -> motor chain: everything downstream is silent
            # and wtaAction below is returning uniform noise, so the episode's
            # fitness says nothing about the genes. Reported once per episode.
            for l in rate_sum:
                rate_sum[l] += (spikes[l].get("s").float().mean().item()
                                * 1000.0 / dt)

            # get the action from the most active motor cell population
            action = wtaAction(mc, env.action_space.n, rng)

            rewards[step - 1] = r
            total_reward += r
            intercepts += int(bool(intercept))

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
        rates = {l: v / max(1, step) for l, v in rate_sum.items()}
        print(f"Episode {ep}: total reward {total_reward:.2f} "
              f"intercepts {intercepts} "
              f"target visible {visible_steps / max(1, step):.0%} | "
              f"AC active {ac_frac_sum / max(1, step):.1%}, "
              f"MC active {mc_frac_sum / max(1, step):.1%} | rates "
              + " ".join(f"{l} {r:.1f}Hz" for l, r in rates.items()))
        dead = [l for l, r in rates.items()
                if r <= 0.0 and not (l == LAYER_PCT and visible_steps == 0)]
        if dead:
            print(f"  [DEAD] no spikes in {', '.join(dead)} for the whole "
                  f"episode -- actions were uniform random, so this candidate's "
                  f"fitness carries no signal about its genes. Check the "
                  f"--gain_* calibration and the layer thresholds.", flush=True)
        os.makedirs(OUT_FILE_PATH, exist_ok=True)
        # One file per episode. Both of these used to be opened in append mode,
        # so every episode of a phase accumulated into the single name
        # genFileName() produced before the loop -- fine for a post-hoc average,
        # useless for inspecting one episode.
        if args.write and rfname:
            with open(episodeFileName(rfname, ep), "wb") as f:
                np.savetxt(f, rewards, delimiter=",", fmt="%.6f")
        if pfname:
            with open(episodeFileName(pfname, ep), "wt") as f:
                f.write(str(intercepts))
        if ep % args.fcycle == 0:
            env.cycleOutFiles()

    # use the average performance of final 25 as the fitness score
    if not past_performances:
        return 0.0
    return sum(past_performances)/len(past_performances)


def run_dot_task(param, task_args=None):
    """
    param      -- one Para-HADES candidate. param['param'] holds the decoded
                  genes: w_ac_mc plus the seven scalers.
    task_args  -- arguments including path and agent_idx. 
    Returns {'fitnessScore': float, ...}.
    """
    global OUT_FILE_PATH

    # Per-candidate output folder. agent_test_num used to be the only thing
    # separating runs, but GA.py never assigns it (it stays at its -1 default),
    # so every candidate of every epoch landed in one "agent<i>_test-1" folder
    # and clobbered the previous one's plots.
    OUT_FILE_PATH = _outDirFor(task_args)
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
    # tcdecay, refrac, tresh
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
    gen_a = torch.Generator(device="cpu").manual_seed(args.seed + 10)
    gc_pc_mask_a = (torch.rand(n_gc, args.n_pc, generator=gen_a)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_a = gc_pc_mask_a * torch.rand(n_gc, args.n_pc, device=DEVICE)
    W_gc_pc_a = (W_gc_pc_a / gc_pc_mask_a.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * s_gc_pc1 * args.gain_gc_pc

    gen_t = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    gc_pc_mask_t = (torch.rand(n_gc, args.n_pc, generator=gen_t)
                    < args.gc_pc_sparsity).float().to(DEVICE)
    W_gc_pc_t = gc_pc_mask_t * torch.rand(n_gc, args.n_pc, device=DEVICE)
    W_gc_pc_t = (W_gc_pc_t / gc_pc_mask_t.sum(0).mean().clamp(min=1.0).sqrt()
                 ) * s_gc_pc2 * args.gain_gc_pc

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
    W_pc_ac_a = (pc_ac_a_mask * (torch.rand(args.n_pc, args.ac, device=DEVICE))
                 / np.sqrt(dense_fan)) * s_pc_ac1 * args.gain_pc_ac
    pc_ac_t_mask = (torch.rand(args.n_pc, args.ac, generator=gen_t) 
                        <= args.pc_ac_sparsity).float().to(DEVICE)
    W_pc_ac_t = (pc_ac_t_mask * (torch.rand(args.n_pc, args.ac, device=DEVICE))
                 / np.sqrt(dense_fan)) * s_pc_ac2 * args.gain_pc_ac

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
    W_rec = torch.randn(args.ac, args.ac, generator=rec_gen)
    # normalize each by sum of their columns and mult by scale factor
    # W_rec = W_rec / W_rec.abs().sum(0, keepdim=True).clamp(min=1e-6) 
    # YAML allows scaler_inhib_ac down to 0.0, and this is a DIVISOR, so clamp:
    # at 0 the recurrent weights blow up to inf and the whole episode goes NaN.
    W_rec = W_rec / max(s_inhib_ac, 1e-3)
    W_rec.fill_diagonal_(0.0)
    feat_rec = Weight(name="w_rec", value=W_rec.to(DEVICE))
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
    W_ac_mc = ac_mc_mask_gen * W_ac_mc * s_ac_mc * args.gain_ac_mc
 
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
    w_opp = s_inhib_mc
    for a, b in OPPONENT_PAIRS: # two pairs - up down and right left
        sa = slice(a * args.mc_pop, (a + 1) * args.mc_pop)
        sb = slice(b * args.mc_pop, (b + 1) * args.mc_pop)
        W_mc_opp[sa, sb] = w_opp      # a inhibits b
        W_mc_opp[sb, sa] = w_opp      # b inhibits a

    feat_mc_opp = Weight(name="w_mc_opp", value=W_mc_opp)
    net.add_connection(
        MulticompartmentConnection(source=mc, target=mc,
                                    pipeline=[feat_mc_opp], device=DEVICE),
        source=LAYER_MC, target=LAYER_MC)

    net.to(DEVICE)

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
        gran=args.granularity,
        rfname=genFileName("rew", "train"), pfname=genFileName("perf", "train"),
        view_r=args.view_r, dim=dim,
    )
    net.learning = False
    print("Testing:")
    environment.changeFileSuffix("train", "test")
    fitness = runSimulator(
        net, environment, spikes, args.tst_eps, W_grid, peak,
        gran=args.granularity,
        rfname=genFileName("rew", "test"), pfname=genFileName("perf", "test"),
        view_r=args.view_r, dim=dim,
    )

    # task_dotTracing.main() indexes outP['fitnessScore'], so return a dict.
    return {
        "fitnessScore": float(fitness),
        "train_fitness": float(train_fitness),
        "out_path": OUT_FILE_PATH,
    }


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
    for key, d in spec.items():
        n = int((d.get("array") or [1])[0])
        lo, hi = float(d.get("min", 0.0)), float(d.get("max", 1.0))
        genes[key] = [0.5 * (lo + hi)] * n
    return {"param": genes}


if __name__ == "__main__":
    out = run_dot_task(standaloneParam())
    print("\n[standalone] result")
    for k, v in out.items():
        print(f"  {k}: {v}")

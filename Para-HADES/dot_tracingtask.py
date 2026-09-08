import matplotlib
matplotlib.use("Agg")

import argparse
import itertools
import os
import sys
import time
import copy

import numpy as np

# torch and bindsnet are optional at import time so that --gene_lengths and
# YAML validation run on a machine without the full SNN stack installed.
try:
    import torch
except ImportError:  
    torch = None

# bindsnet imports are deferred into _lazy_imports() so that --gene_lengths and
# YAML validation work on a machine without bindsnet installed.
Network = Monitor = Input = LIFNodes = None
MulticompartmentConnection = Weight = MSTDPET = poisson = DotSimulator = None


# imports for SNN
def _lazy_imports():
    global Network, Monitor, Input, LIFNodes
    global MulticompartmentConnection, Weight, MSTDPET, poisson, DotSimulator
    if Network is not None:
        return
    if torch is None:
        raise ImportError(
            "PyTorch is required to run the network but is not installed. "
            "(`--gene_lengths` works without it; `--selftest`, `--calibrate` "
            "and do_task do not.)"
        )
    from bindsnet.encoding import poisson as _poisson
    from bindsnet.environment.dot_simulator import DotSimulator as _DotSimulator
    from bindsnet.learning.MCC_learning import MSTDPET as _MSTDPET
    from bindsnet.network import Network as _Network
    from bindsnet.network.monitors import Monitor as _Monitor
    from bindsnet.network.nodes import Input as _Input, LIFNodes as _LIFNodes
    from bindsnet.network.topology import MulticompartmentConnection as _MCC
    from bindsnet.network.topology_features import Weight as _Weight
    Network, Monitor, Input, LIFNodes = _Network, _Monitor, _Input, _LIFNodes
    MulticompartmentConnection, Weight, MSTDPET = _MCC, _Weight, _MSTDPET
    poisson, DotSimulator = _poisson, _DotSimulator


LAYER_GC, LAYER_AC, LAYER_MC = "GC", "AC", "MC"

# --------------------------------------------------------------------------- #
# Defaults params. Every key here can be overridden by param["DotTracing"] in the YAML.
# Anything the YAML sets that is NOT in here raises a warning (typo guard).
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(
    # --- task / environment ---
    steps=100,
    dim=28,
    granularity=100,
    dt=1.0,
    decay=4,
    herrs=0,
    diag=False,
    randr=0.15,
    bound_hand="bounce",
    fit_func="dir",
    allow_stay=False,
    view_r=14,

    # --- GC layer (fixed, never plastic, never evolved) ---
    gc_scales=[3, 5, 7, 11, 13],
    gc_rotations=7,
    gc_offsets=16,
    gc_global_scale=1.0,
    gc_sharpness=1.0,
    gc_max_rate=80.0,

    # --- AC layer ---
    ac=400,
    gc_ac_convergence=25,
    ac_gain=14.0,
    rec_mode="local",
    rec_inh=1.2,
    rec_radius=4.0,

    # --- MC layer ---
    mc_pop=40,
    ac_mc_convergence=20,

    # --- topology seed: fixed for the whole population ---
    model_seed=0,

    # --- what EVOL optimizes ---
    plastic=["ac_mc"],
    inheritance="darwinian",

    # --- weight box. MUST match param: min/max in the YAML. ---
    w_min=0.0,
    w_max=1.0,

    # --- STDP-RL: the lifetime learning ---
    stdp_enabled=True,
    nu=0.01,
    eps_start=0.0,
    eps_min=0.0,
    eps_decay=1.0,
    select="wta",

    # --- critic     NOT USED FOR NOW ---
    critic_mode="loss",       # 'loss' = paper form | 'env' = alignment reward
    max_reward=1.0,
    eta_positivity=5.0,
    eta_closing=1.0,
    critic_gain=1.0,
    success_eps=0.01,
    critic_dist_scale="auto",   # 'auto' = sqrt(2)*dim (grid diagonal)

    # --- weight normalization (mandatory when STDP is live) ---
    norm_reception=True,
    reception_every=25,
    norm_transmission=True,
    transmission_min=0.1,
    transmission_max=2.0,
    norm_homeostasis=True,
    homeostasis_every=75,
    rate_window=500,
    homeostasis_scale=0.0001,
    rate_target_ac=5.5,
    rate_target_mc=6.0,

    # --- lifetime / fitness ---
    episodes=20,
    fitness_metric="intercepts",   # 'intercepts' | 'reward'
    fitness_last_k=5,
    frozen_episodes=5,
    episode_seed_base=12345,
    seed_mode="common",            # 'common' | 'random'

    # --- misc ---
    gpu=False,
    cache_gc=True,
    verbose=True,
)



# CHECK- potential issue 1: param[param] is not in cfg
# sets cfg to store the param file + default vals
def buildConfig(param):
    """Merge main.DotTracing from the GA candidate over DEFAULT_CFG."""
    cfg = copy.deepcopy(DEFAULT_CFG)
    user = {}
    if isinstance(param, dict):
        user = (param.get("main", {}) or {}).get("DotTracing", {}) or {}
    unknown = [k for k in user if k not in cfg]
    if unknown:
        print(f"[cfg] WARNING unknown DotTracing keys ignored: {unknown}", flush=True)
    for k, v in user.items():
        if k in cfg:
            cfg[k] = v
    cfg["moveChoices"] = 9 if cfg["diag"] else 5
    cfg["n_mc"] = cfg["moveChoices"] * cfg["mc_pop"]
    return cfg


def geneLengths(cfg):
    """Gene vector length per plastic pathway. These are the YAML array sizes."""
    return {
        "gc_ac": cfg["ac"] * cfg["gc_ac_convergence"],
        "ac_mc": cfg["n_mc"] * cfg["ac_mc_convergence"],
    }



def getLocalView(obs, row, col, view_r):
    """returns a 1-d egocentric patch around the agent, agent's own cell excluded."""
    padded = np.pad(obs, view_r, mode="constant", constant_values=0)
    patch = padded[row: row + 2 * view_r + 1, col: col + 2 * view_r + 1]
    flat = patch.flatten()
    center = view_r * (2 * view_r + 1) + view_r
    return np.concatenate([flat[:center], flat[center + 1:]])


def viewIndexToOffset(view_r):
    """(drow, dcol) of every input index (coordinate relative to agent),
        matching getLocalView's layout exactly."""
    side = 2 * view_r + 1
    center = view_r * side + view_r
    offs = []
    for k in range(side * side - 1):
        cell = k if k < center else k + 1
        pr, pc = divmod(cell, side)
        offs.append((pr - view_r, pc - view_r))
    return np.array(offs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# GRID CELL LAYER (fixed projection -- never evolved, never plastic)
# --------------------------------------------------------------------------- #
def _hexLattice(spacing, rotation, phase, extent):
    b1 = spacing * np.array([1.0, 0.0])
    b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])
    n = int(np.ceil(2.0 * extent / spacing)) + 2
    ij = np.array(list(itertools.product(range(-n, n + 1), repeat=2)), dtype=np.float32)
    pts = ij[:, :1] * b1 + ij[:, 1:] * b2
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T + phase
    keep = (np.abs(pts) <= extent + spacing).all(axis=1)
    return pts[keep]


_GC_CACHE = {}

def buildGridCellProjection(view_r, scales, n_rot, n_off, g, k, device="cpu",
                            cache=True):
    """Create grid cell projection inpt->grid: hexagonal lattices of 2D Gaussians."""
    key = (view_r, tuple(scales), n_rot, n_off, g, k, str(device))

    if cache and key in _GC_CACHE:
        return _GC_CACHE[key]

    # viewing radius is 14
    offs = viewIndexToOffset(view_r)
    extent = float(view_r)

    # n_rot is 7
    rotations = [np.pi * i / max(1, n_rot) for i in range(n_rot)]

    # 16 offsets
    side = int(round(np.sqrt(n_off)))

    cols, meta = [], []
    # scales is [3, 5, 7, 11, 13]
    for s in scales:
        # vector setup for the hex lattice
        spacing = s * g
        d = (1.0 / k) * (spacing / 2.0)
        sigma = d / 3.0
        b1 = spacing * np.array([1.0, 0.0])
        b2 = spacing * np.array([0.5, np.sqrt(3.0) / 2.0])
        for theta in rotations:
            for a in range(side):
                for b in range(side):
                    phase = (a / side) * b1 + (b / side) * b2
                    pts = _hexLattice(spacing, theta, phase, extent)
                    d2 = ((offs[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
                    field = np.exp(-d2.min(axis=1) / (2.0 * sigma ** 2))
                    field[field < 0.01] = 0.0
                    cols.append(field.astype(np.float32))
                    meta.append((s, theta, a, b))

    W = np.stack(cols, axis=1)
    peak = W.max(axis=0, keepdims=True)
    peak[peak == 0] = 1.0
    out = (torch.from_numpy(W).to(device), torch.from_numpy(peak).to(device), meta)
    if cache:
        _GC_CACHE[key] = out
    return out


def gcRates(view_vec, W_grid, peak, max_rate):
    drive = view_vec @ W_grid
    drive = (drive / peak.squeeze(0)).clamp(0.0, 1.0)
    drive[drive < 0.01] = 0.0
    return drive * max_rate


# --------------------------------------------------------------------------- #
# Connectivity. Convergence-based, NOT probabilistic -- the gene length has to
# be deterministic or EVOL and the network disagree about what the vector means.
# --------------------------------------------------------------------------- #

# connect each ac neuron to 25 randomly selected grid cell neuron
def buildConvergenceMask(n_pre, n_post, convergence, seed, device):
    """
    Each postsynaptic neuron receives from exactly K distinct presynaptic
    neurons. Returns (mask, src_idx) where src_idx is (K, n_post).

    Gene ordering is src_idx-major: gene.view(K, n_post) indexes exactly the
    live synapses, so applyGene and extractGene are exact inverses.
    """
    K = int(min(convergence, n_pre))
    g = torch.Generator().manual_seed(int(seed))
    src = torch.stack(
        [torch.randperm(n_pre, generator=g)[:K] for _ in range(n_post)], dim=1
    )                                                    # (K, n_post)
    mask = torch.zeros(n_pre, n_post)
    cols = torch.arange(n_post).unsqueeze(0).expand(K, n_post)
    mask[src, cols] = 1.0
    return mask.to(device), src.to(device)


def applyGeneToWeights(weight_tensor, src_idx, gene_vec, w_min, w_max):
    """Write a flat gene vector into the live synapses of a weight matrix."""
    K, n_post = src_idx.shape
    expected = K * n_post
    v = torch.as_tensor(np.asarray(gene_vec, dtype=np.float32),
                        device=weight_tensor.device).flatten()
    if v.numel() != expected:
        raise ValueError(
            f"gene length {v.numel()} != expected {expected} "
            f"(K={K}, n_post={n_post}). Regenerate the YAML array size with "
            f"`python dot_tracingtask.py --gene_lengths`."
        )
    cols = torch.arange(n_post, device=weight_tensor.device).unsqueeze(0).expand(K, n_post)
    with torch.no_grad():
        weight_tensor.zero_()
        weight_tensor[src_idx, cols] = v.view(K, n_post).clamp(w_min, w_max)
    return weight_tensor

#not used 
def extractGeneFromWeights(weight_tensor, src_idx):
    """Inverse of applyGeneToWeights -- used for lamarckian inheritance."""
    K, n_post = src_idx.shape
    cols = torch.arange(n_post, device=weight_tensor.device).unsqueeze(0).expand(K, n_post)
    with torch.no_grad():
        return weight_tensor[src_idx, cols].reshape(-1).detach().cpu().numpy()


def acPlaceCentres(W_grid, gc_ac_mask, view_r):
    eff = W_grid @ gc_ac_mask
    idx = eff.argmax(dim=0).cpu().numpy()
    offs = viewIndexToOffset(view_r)
    return torch.from_numpy(offs[idx]).float()


def buildRecurrent(mode, centres, n_ac, inh, radius, device, seed=0):
    if mode == "none":
        return None
    if mode == "random":
        g = torch.Generator().manual_seed(int(seed))
        signs = (torch.rand(n_ac, n_ac, generator=g) < 0.45).float() * 2 - 1
        W = torch.rand(n_ac, n_ac, generator=g) * signs
    else:  # local: short-range excitation, broad surround inhibition
        d = torch.cdist(centres, centres)
        exc = torch.exp(-(d ** 2) / (2 * radius ** 2))
        W = exc - inh * (1.0 - exc)
    W.fill_diagonal_(0.0)
    return W.to(device)


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
class Critic:
    """
    critic_mode='loss' -- the paper's form, transposed to pursuit:
        loss(t)  = sqrt(distance^2 + eta_closing * closing_rate^2)
        reward   = loss(t-1) - loss(t)
        -- positivity bias 

    critic_mode='env' -- the alignment-scaled reward from the original script.
    """

    def __init__(self, cfg):
        self.mode = cfg["critic_mode"]
        self.max_reward = float(cfg["max_reward"])
        self.eta_pos = float(cfg["eta_positivity"])
        self.eta_closing = float(cfg["eta_closing"])
        self.gain = float(cfg["critic_gain"])
        self.success_eps = float(cfg["success_eps"])

        # Distance normalization. WITHOUT THIS THE CRITIC SATURATES: raw pixel
        # distances on a 28x28 grid give per-step loss deltas of order 1-3,
        # which eta_positivity multiplies to 5-15, so every single step clips
        # to +/-max_reward and the critic degenerates into a sign function.
        # MSTDPET modulates by reward MAGNITUDE, so that throws the graded part
        # of the learning signal away. The paper avoids this only because
        # CartPole's state variables are radians (~0.26 max).
        # Dividing by the grid diagonal puts loss in [0,1] and deltas in the
        # range the positivity bias and the clip were designed for.
        scale = cfg.get("critic_dist_scale", "auto")
        if scale in (None, "auto"):
            scale = float(np.sqrt(2.0) * cfg["dim"])
        self.dist_scale = max(float(scale), 1e-8)

        self.prev_loss = None

    def reset(self):
        self.prev_loss = None

    def _loss(self, dist, closing):
        d = dist / self.dist_scale
        c = closing / self.dist_scale
        return float(np.sqrt(d ** 2 + self.eta_closing * c ** 2))

    def __call__(self, *, dist, closing, moved, intercept, alignment):
        if self.mode == "env":
            scale = 0.5 + 0.5 * alignment
            r = closing * ((0.5 + scale) if closing >= 0 else (1.5 - scale))
            r += 0.2 * alignment
            if intercept:
                r += 20.0
            return float(r)

        # ---- paper form ----
        loss = self._loss(dist, closing)
        if self.prev_loss is None:
            self.prev_loss = loss
            return 0.0

        if not moved:
            raw = -self.max_reward / self.eta_pos
        elif loss < self.success_eps or intercept:
            raw = self.max_reward / self.eta_pos
        elif self.prev_loss < self.success_eps:
            raw = 0.0
        else:
            raw = self.prev_loss - loss

        self.prev_loss = loss
        biased = raw * self.eta_pos if raw > 0 else raw
        return float(np.clip(biased * self.gain, -self.max_reward, self.max_reward))


# --------------------------------------------------------------------------- #
# Weight normalization. MANDATORY when STDP is live -- otherwise weights grow
# without bound and the network goes epileptic.
# --------------------------------------------------------------------------- #
class WeightNormalizer:
    """
    CRITICAL: reception balancing renormalizes to the sum of the OFFSPRING's
    starting weights (the EVOL-issued gene), NOT to a global w_0. If it snapped
    back to a global constant it would silently undo the EVOL perturbation and
    the whole method would degenerate to plain STDP-RL.
    """

    def __init__(self, cfg, weight_tensor, mask):
        self.cfg = cfg
        self.mask = mask
        self.w_min, self.w_max = float(cfg["w_min"]), float(cfg["w_max"])
        with torch.no_grad():
            self.target_reception = weight_tensor.sum(0).clone()      # (n_post,)
            self.start_transmission = weight_tensor.sum(1).clone()    # (n_pre,)
            self.start_transmission.clamp_(min=1e-8)
        self.step_count = 0
        self.rate_hist_ac, self.rate_hist_mc = [], []

    def clamp(self, w):
        with torch.no_grad():
            w.mul_(self.mask).clamp_(self.w_min, self.w_max)

    def scale_delta(self, w_before, w_after):
        """
        Transmission-side normalization: scale each synapse's weight CHANGE by
        that presynaptic neuron's total outgoing weight relative to its start.
        A synapse whose source already transmits a lot gets its potentiation
        damped and its depression amplified.
        """
        if not self.cfg["norm_transmission"]:
            return
        with torch.no_grad():
            cur = w_after.sum(1).clamp(min=1e-8)
            ratio = (self.start_transmission / cur).clamp(
                self.cfg["transmission_min"], self.cfg["transmission_max"])
            delta = (w_after - w_before) * ratio.unsqueeze(1)
            w_after.copy_(w_before + delta)

    def step(self, w, ac_rate=None, mc_rate=None):
        self.step_count += 1
        cfg = self.cfg

        if ac_rate is not None:
            self.rate_hist_ac.append(float(ac_rate))
            self.rate_hist_mc.append(float(mc_rate))
            if len(self.rate_hist_ac) > cfg["rate_window"]:
                self.rate_hist_ac.pop(0)
                self.rate_hist_mc.pop(0)

        # homeostatic gain control: nudge the reception TARGET, not the weights
        if (cfg["norm_homeostasis"] and self.rate_hist_mc
                and self.step_count % cfg["homeostasis_every"] == 0):
            obs = float(np.mean(self.rate_hist_mc))
            tgt = float(cfg["rate_target_mc"])
            if obs > tgt:
                self.target_reception.mul_(1.0 - cfg["homeostasis_scale"])
            elif obs < tgt:
                self.target_reception.mul_(1.0 + cfg["homeostasis_scale"])

        # reception balancing: restore each postsynaptic neuron's summed input
        if cfg["norm_reception"] and self.step_count % cfg["reception_every"] == 0:
            with torch.no_grad():
                cur = w.sum(0).clamp(min=1e-8)
                w.mul_((self.target_reception / cur).unsqueeze(0))
        self.clamp(w)


# --------------------------------------------------------------------------- #
# Action selection
# --------------------------------------------------------------------------- #
def wtaAction(mc_spikes, n_actions, eps, rng):
    """Returns (action, moved) -- 'moved' is False when the tie forced a coin flip."""
    if eps > 0 and rng.random() < eps:
        return int(rng.integers(0, n_actions)), False
    counts = mc_spikes.sum(0)
    agg = torch.stack([p.sum() for p in torch.chunk(counts, n_actions)])
    if agg.max() == 0:
        return int(rng.integers(0, n_actions)), False
    top = (agg == agg.max()).nonzero().flatten()
    if top.numel() > 1:                      # paper: random move on a tie
        return int(top[rng.integers(0, top.numel())].item()), False
    return int(agg.argmax().item()), True


def softmaxAction(mc_spikes, n_actions, rng):
    counts = mc_spikes.sum(0)
    agg = torch.stack([p.sum() for p in torch.chunk(counts, n_actions)])
    return int(torch.multinomial(torch.softmax(agg.float(), dim=0), 1).item()), True


# run a single episode
def runEpisode(net, env, spikes, cfg, W_grid, peak, feat_ac_mc, mask,
               normalizer, critic, rng, device, learning=True, eps=0.0,
               episode_seed=None):
    gran, dt, view_r = cfg["granularity"], cfg["dt"], cfg["view_r"]
    n_actions = env.action_space.n

    net.reset_state_variables()
    if episode_seed is not None:
        # DotSimulator.reset() takes no seed argument and reseeds the stdlib
        # `random` module from self.seed, so the seed must be set on the
        # instance. Seeding np.random here does nothing -- the env never uses it.
        env.seed = int(episode_seed)
    env.reset()
    critic.reset()

    total_reward, intercepts, step = 0.0, 0, 0
    action = int(rng.integers(0, n_actions))
    done = False

    while not done:
        step += 1
        # calculate the distance between the agent and the target dot before taking a step
        pr, pc = env.netDot.row[0], env.netDot.col[0]
        tr0, tc0 = env.dots[0].row[0], env.dots[0].col[0]
        prev_dist = float(np.hypot(pr - tr0, pc - tc0))

        obs, _, done, intercept = env.step(action)

        net_obs = obs.copy()
        net_obs[env.netDot.row, env.netDot.col] = 0.0
        view = torch.as_tensor(
            getLocalView(net_obs, env.netDot.row[0], env.netDot.col[0], view_r),
            dtype=torch.float32, device=device)

        cr, cc = env.netDot.row[0], env.netDot.col[0]
        tr, tc = env.dots[0].row[0], env.dots[0].col[0]
        curr_dist = float(np.hypot(cr - tr, cc - tc))
        closing = prev_dist - curr_dist

        dr_p, dc_p = tr - pr, tc - pc
        dr_m, dc_m = cr - pr, cc - pc
        dmag, mmag = np.hypot(dr_p, dc_p), np.hypot(dr_m, dc_m)
        alignment = 0.0 if (mmag < 1e-8 or dmag < 1e-8) else float(
            np.clip((dr_m * dr_p + dc_m * dc_p) / (dmag * mmag), -1, 1))

        r = critic(dist=curr_dist, closing=closing, moved=True,
                    intercept=bool(intercept), alignment=alignment)
        reward = torch.tensor(r, dtype=torch.float32, device=device)

        rates = gcRates(view, W_grid, peak, cfg["gc_max_rate"])
        inputs = {LAYER_GC: poisson(rates.unsqueeze(0), gran, dt, device=device)}

        w_before = feat_ac_mc.value.detach().clone() if learning else None
        net.run(inputs=inputs, time=gran, reward=reward)

        mc_s = spikes[LAYER_MC].get("s").squeeze()
        ac_s = spikes[LAYER_AC].get("s").squeeze()
        ac_rate = float(ac_s.float().mean().item()) * 1000.0 / dt
        mc_rate = float(mc_s.float().mean().item()) * 1000.0 / dt

        if learning:
            normalizer.scale_delta(w_before, feat_ac_mc.value)
            normalizer.step(feat_ac_mc.value, ac_rate=ac_rate, mc_rate=mc_rate)

        if cfg["select"] == "wta":
            action, _ = wtaAction(mc_s, n_actions, eps, rng)
        else:
            action, _ = softmaxAction(mc_s, n_actions, rng)

        total_reward += r
        intercepts += int(bool(intercept))

    return dict(total_reward=total_reward, intercepts=intercepts, steps=step)


# build network based on the cfg set by defaults + YAML file
def buildNetwork(cfg, device):
    _lazy_imports()
    seed = int(cfg["model_seed"])
    torch.manual_seed(seed)

    net = Network(dt=cfg["dt"])
    inpt_n = (2 * cfg["view_r"] + 1) ** 2 - 1

    W_grid, peak, _ = buildGridCellProjection(
        cfg["view_r"], cfg["gc_scales"], cfg["gc_rotations"], cfg["gc_offsets"],
        cfg["gc_global_scale"], cfg["gc_sharpness"], device=device,
        cache=cfg["cache_gc"])
    n_gc = W_grid.shape[1]

    gc = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)
    ac = LIFNodes(n=cfg["ac"], traces=True, rest=-64.0, reset=-70.0,
                  thresh=-45.0, refrac=1, tc_decay=20.0, tc_trace=20.0)
    mc = LIFNodes(n=cfg["n_mc"], traces=True, rest=-64.0, reset=-64.0,
                  thresh=-49.0, refrac=0, tc_decay=20.0, tc_trace=20.0)

    net.add_layer(gc, name=LAYER_GC)
    net.add_layer(ac, name=LAYER_AC)
    net.add_layer(mc, name=LAYER_MC)

    # --- GC -> AC : fixed. Evolved only if 'gc_ac' is in cfg['plastic']. ---
    gc_ac_mask, gc_ac_src = buildConvergenceMask(
        n_gc, cfg["ac"], cfg["gc_ac_convergence"], seed, device)
    
    g = torch.Generator().manual_seed(seed + 7)
    W_gc_ac = (gc_ac_mask.cpu() * torch.rand(n_gc, cfg["ac"], generator=g)).to(device)
    fan_in = gc_ac_mask.sum(0).mean().clamp(min=1.0)
    W_gc_ac = W_gc_ac / fan_in.sqrt() * cfg["ac_gain"]
    feat_gc_ac = Weight(name="w_gc_ac", value=W_gc_ac)
    net.add_connection(
        MulticompartmentConnection(source=gc, target=ac,
                                   pipeline=[feat_gc_ac], device=device),
        source=LAYER_GC, target=LAYER_AC)

    # --- AC -> AC : structured local inhibition, fixed ---
    feat_rec = None
    if cfg["rec_mode"] != "none":
        centres = acPlaceCentres(W_grid, gc_ac_mask, cfg["view_r"])
        W_rec = buildRecurrent(cfg["rec_mode"], centres, cfg["ac"],
                               cfg["rec_inh"], cfg["rec_radius"], device, seed)
        if W_rec is not None:
            feat_rec = Weight(name="w_rec", value=W_rec)
            net.add_connection(
                MulticompartmentConnection(source=ac, target=ac,
                                           pipeline=[feat_rec], device=device),
                source=LAYER_AC, target=LAYER_AC)

    # --- AC -> MC : the plastic + evolved pathway ---
    ac_mc_mask, ac_mc_src = buildConvergenceMask(
        cfg["ac"], cfg["n_mc"], cfg["ac_mc_convergence"], seed + 1, device)
    mid = 0.5 * (cfg["w_min"] + cfg["w_max"])
    W_ac_mc = ac_mc_mask * mid
    if cfg["stdp_enabled"]:
        feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc,
                            learning_rule=MSTDPET, nu=[cfg["nu"], cfg["nu"]],
                            range=[cfg["w_min"], cfg["w_max"]])
    else:
        feat_ac_mc = Weight(name="w_ac_mc", value=W_ac_mc)
    net.add_connection(
        MulticompartmentConnection(source=ac, target=mc,
                                   pipeline=[feat_ac_mc], device=device),
        source=LAYER_AC, target=LAYER_MC)

    net.to(device)

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"],
                                time=int(cfg["granularity"] / cfg["dt"]),
                                device=device)
        net.add_monitor(spikes[layer], name=layer)

    handles = dict(
        net=net, spikes=spikes, W_grid=W_grid, peak=peak,
        feat_gc_ac=feat_gc_ac, gc_ac_mask=gc_ac_mask, gc_ac_src=gc_ac_src,
        feat_ac_mc=feat_ac_mc, ac_mc_mask=ac_mc_mask, ac_mc_src=ac_mc_src,
        feat_rec=feat_rec, n_gc=n_gc, inpt_n=inpt_n,
    )
    return handles


# main function called by ga task file
def do_task(param):
    """
    param     -- one GA candidate dict. param['param']['w_ac_mc'] is the gene.
                 param['main']['DotTracing'] is the config block.
    Returns a dict with a scalar 'fitnessScore' (higher = better).
    """
    #import libraries
    _lazy_imports()
    t0 = time.time()
    # cfg stores param configuration
    cfg = buildConfig(param)
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg["gpu"])
                          else "cpu")

    H = buildNetwork(cfg, device)
    net, spikes = H["net"], H["spikes"]

    # ---------- load the EVOL gene into the plastic pathways ---------------- #
    gene_block = (param or {}).get("param", {}) or {}
    lengths = geneLengths(cfg)
    loaded = []

    # overwrite the default vals for EVOL weights
    if "ac_mc" in cfg["plastic"]:
        if "w_ac_mc" not in gene_block:
            raise KeyError(
                "cfg['plastic'] contains 'ac_mc' but the candidate has no "
                "param['param']['w_ac_mc']. Add w_ac_mc to the YAML `param:` "
                f"block with array: [{lengths['ac_mc']}]."
            )
        applyGeneToWeights(H["feat_ac_mc"].value, H["ac_mc_src"],
                           gene_block["w_ac_mc"], cfg["w_min"], cfg["w_max"])
        loaded.append("ac_mc")

    if "gc_ac" in cfg["plastic"]:
        if "w_gc_ac" not in gene_block:
            raise KeyError(
                "cfg['plastic'] contains 'gc_ac' but the candidate has no "
                "param['param']['w_gc_ac']. Either add it to the YAML `param:` "
                f"block with array: [{lengths['gc_ac']}], or remove 'gc_ac' "
                "from `plastic:`."
            )
        applyGeneToWeights(H["feat_gc_ac"].value, H["gc_ac_src"],
                           gene_block["w_gc_ac"], cfg["w_min"], cfg["w_max"])
        loaded.append("gc_ac")

    w_start = H["feat_ac_mc"].value.detach().clone()

    # ---------- lifetime ---------------------------------------------------- #
    normalizer = WeightNormalizer(cfg, H["feat_ac_mc"].value, H["ac_mc_mask"])
    critic = Critic(cfg)
    rng = np.random.default_rng(int(cfg["model_seed"]) + 1000)

    # build dot simu environment
    env = DotSimulator(
        cfg["steps"], decay=cfg["decay"], herrs=cfg["herrs"], diag=cfg["diag"],
        randr=cfg["randr"], write=False, mute=not cfg["verbose"],
        bound_hand=cfg["bound_hand"], fit_func=cfg["fit_func"],
        allow_stay=cfg["allow_stay"], pandas=False, fpath="./",
    )
    env.reset()

    net.learning = bool(cfg["stdp_enabled"])
    eps = float(cfg["eps_start"])
    per_ep = []

    
    for ep in range(int(cfg["episodes"])):
        # if we want the episodes to be the same. by default, episodes are the same
        if cfg["seed_mode"] == "common":
            ep_seed = int(cfg["episode_seed_base"]) + ep
        else:
            # 'random': still varies per episode, but not shared across the
            # population -- fitness comparisons pick up the extra variance.
            ep_seed = int(rng.integers(0, 2 ** 31 - 1))
        res = runEpisode(net, env, spikes, cfg, H["W_grid"], H["peak"],
                         H["feat_ac_mc"], H["ac_mc_mask"], normalizer, critic,
                         rng, device, learning=bool(cfg["stdp_enabled"]),
                         eps=eps, episode_seed=ep_seed)
        per_ep.append(res)
        eps = max(cfg["eps_min"], eps * cfg["eps_decay"])
        if cfg["verbose"]:
            print(f"  [life] ep {ep:3d}  reward {res['total_reward']:+8.2f}  "
                  f"intercepts {res['intercepts']:3d}", flush=True)

    # ---------- frozen held-out evaluation (fitness variant iii) ------------ #
    frozen = []
    # testing, no stdp
    if int(cfg["frozen_episodes"]) > 0:
        net.learning = False
        for ep in range(int(cfg["frozen_episodes"])):
            ep_seed = int(cfg["episode_seed_base"]) + 90000 + ep
            frozen.append(runEpisode(
                net, env, spikes, cfg, H["W_grid"], H["peak"],
                H["feat_ac_mc"], H["ac_mc_mask"], normalizer, critic, rng,
                device, learning=False, eps=0.0, episode_seed=ep_seed))

    # ---------- fitness ----------------------------------------------------- #
    key = "intercepts" if cfg["fitness_metric"] == "intercepts" else "total_reward"
    all_vals = [float(r[key]) for r in per_ep]
    k = int(cfg["fitness_last_k"])
    fitness = float(np.mean(all_vals)) if all_vals else 0.0

    # ---------- STDP contribution diagnostic -------------------------------- #
    w_end = H["feat_ac_mc"].value.detach()
    dw = float(torch.linalg.vector_norm(w_end - w_start).item())
    w0n = float(torch.linalg.vector_norm(w_start).item()) or 1.0
    dw_rel = dw / w0n

    # note: only the firnessScore key is used, other stuff might be useful though
    outP = {
        "fitnessScore": fitness,
        "fit_mean_all": fitness,
        "fit_last_k": float(np.mean(all_vals[-k:])) if all_vals else 0.0,
        "fit_frozen": float(np.mean([r[key] for r in frozen])) if frozen else None,
        "mean_reward": float(np.mean([r["total_reward"] for r in per_ep])) if per_ep else 0.0,
        "mean_intercepts": float(np.mean([r["intercepts"] for r in per_ep])) if per_ep else 0.0,
        "stdp_dw_norm": dw,
        "stdp_dw_relative": dw_rel,
        "loaded_pathways": loaded,
        "wallclock_sec": time.time() - t0,
    }

    if cfg["inheritance"] == "lamarckian" and "ac_mc" in cfg["plastic"]:
        outP["gene_out_w_ac_mc"] = extractGeneFromWeights(
            H["feat_ac_mc"].value, H["ac_mc_src"]).tolist()

    if cfg["verbose"]:
        print(f"[do_task] fitness={fitness:.4f}  dw_rel={dw_rel:.4%}  "
              f"({outP['wallclock_sec']:.1f}s)", flush=True)
        if dw_rel < 0.01:
            print("[do_task] WARNING STDP moved the weights <1%. The hybrid is "
                  "effectively plain EVOL. Raise `episodes` or `nu`.", flush=True)

    try:
        env.close()
    except Exception:
        pass
    return outP




# ignore below
def _load_yaml_cfg(path):
    import yaml
    with open(path, "rt") as f:
        y = yaml.safe_load(f)
    return buildConfig({"main": y.get("main", {})}), y


def _cli():
    p = argparse.ArgumentParser(description="dot-tracing SNN utilities")
    p.add_argument("--paramFile", type=str, default=None)
    p.add_argument("--gene_lengths", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--episodes", type=int, default=None)
    a = p.parse_args()

    yaml_path = a.paramFile or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "task_dotTracing.yaml")
    if os.path.exists(yaml_path):
        cfg, raw = _load_yaml_cfg(yaml_path)
    else:
        cfg, raw = buildConfig({}), {}
        print(f"[cli] no YAML at {yaml_path}, using DEFAULT_CFG", flush=True)

    if a.gene_lengths:
        L = geneLengths(cfg)
        print("\nRequired YAML `param:` array sizes for this config")
        print(f"  ac={cfg['ac']}  mc_pop={cfg['mc_pop']}  "
              f"moveChoices={cfg['moveChoices']}  n_mc={cfg['n_mc']}")
        print(f"  gc_ac_convergence={cfg['gc_ac_convergence']}  "
              f"ac_mc_convergence={cfg['ac_mc_convergence']}\n")
        for name, n in L.items():
            marker = "  <-- in plastic:" if name in cfg["plastic"] else ""
            print(f"  w_{name}:  array: [{n}]{marker}")
        print()
        declared = list((raw.get("param", {}) or {}).keys())
        for name in cfg["plastic"]:
            key = f"w_{name}"
            if key not in declared:
                print(f"  ERROR '{name}' is in plastic: but `param:` has no {key}")
        for key in declared:
            want = L.get(key[2:])
            got = ((raw["param"][key] or {}).get("array") or [None])[0]
            if want is not None and got is not None and int(got) != want:
                print(f"  ERROR {key}: YAML says array:[{got}], config needs [{want}]")
        return

    if a.selftest:
        if a.episodes:
            cfg["episodes"] = a.episodes
        L = geneLengths(cfg)
        rng = np.random.default_rng(0)
        mid = 0.5 * (cfg["w_min"] + cfg["w_max"])
        sd = (cfg["w_max"] - cfg["w_min"]) / 6.0
        fake = {"main": {"DotTracing": cfg}, "param": {}}
        for name in cfg["plastic"]:
            fake["param"][f"w_{name}"] = np.clip(
                rng.normal(mid, sd, L[name]), cfg["w_min"], cfg["w_max"]).tolist()
        print(f"[selftest] running {cfg['episodes']} episodes ...", flush=True)
        out = do_task(fake, None)
        print("\n[selftest] result")
        for k, v in out.items():
            if k != "gene_out_w_ac_mc":
                print(f"  {k}: {v}")
        print("\n[selftest] STDP contribution check")
        print(f"  ||w_final - w_start|| / ||w_start|| = {out['stdp_dw_relative']:.4%}")
        if out["stdp_dw_relative"] < 0.01:
            print("  VERDICT: STDP is inert at this lifetime length. The hybrid "
                  "reduces to plain EVOL. Raise `episodes` or `nu`.")
        else:
            print("  VERDICT: STDP is moving the weights. Hybrid is live.")
        return

    if a.calibrate:
        print("[calibrate] sweeping w_max for target MC rate "
              f"{cfg['rate_target_mc']} Hz ...", flush=True)
        L = geneLengths(cfg)
        rng = np.random.default_rng(0)
        for wmax in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            c = copy.deepcopy(cfg)
            c["w_max"], c["episodes"], c["frozen_episodes"] = wmax, 1, 0
            c["stdp_enabled"], c["verbose"] = False, False
            mid = 0.5 * (c["w_min"] + wmax)
            sd = (wmax - c["w_min"]) / 6.0
            fake = {"main": {"DotTracing": c}, "param": {
                f"w_{n}": np.clip(rng.normal(mid, sd, L[n]), c["w_min"], wmax).tolist()
                for n in c["plastic"]}}
            try:
                o = do_task(fake, None)
                print(f"  w_max={wmax:5.2f} -> reward {o['mean_reward']:+8.2f}  "
                      f"intercepts {o['mean_intercepts']:.2f}")
            except Exception as e:
                print(f"  w_max={wmax:5.2f} -> FAILED: {e}")
        return

    p.print_help()


if __name__ == "__main__":
    _cli()
"""
Para-HADES task: HYBRID EVOL + STDP-RL for a BindsNET spiking network on the
dot-tracing task.

Base method: Haşegan et al. (2022), Front. Comput. Neurosci. 16:1017284,
"Training spiking neuronal networks to perform motor control using
reinforcement and evolutionary learning". The paper runs EVOL and STDP-RL as
SEPARATE, mutually exclusive strategies -- during EVOL, STDP is fully
deactivated and weights are frozen inside every episode. This file implements
the HYBRID the paper did not: EVOL sets each offspring's STARTING weights, then
STDP-RL runs live during that offspring's episodes so its weights adapt within
its own "lifetime". Fitness is measured over the lifetime and returned to
Para-HADES, which recombines across the population.

    w_i --perturb--> P offspring --each lives & learns via STDP--> fitness
        --recombine--> w_{i+1} --perturb--> ...

DIVISION OF LABOUR
------------------
Para-HADES owns the OUTER loop (perturbation, fitness normalization,
recombination). This file owns ONE offspring evaluation: gene -> fitness.
That means the EVOL update rule is whatever `evolutionType` you select in
GA.py; `OpenAI_ES` is the closest relative of the paper's EVOL (both are
Salimans et al. style ES with mean-centred, fitness-weighted recombination and
no truncation selection).

Two deliberate divergences from the paper's EVOL, both consequences of using
Para-HADES as the engine:

  * The paper perturbs MULTIPLICATIVELY, w_j = w_i * (1 + sigma * e_j), purely
    so that SNN weights stay positive. Para-HADES instead evolves a NORMALIZED
    gene in [-1, 1] and maps it onto [min, max] from the YAML. With min = 0.0
    that keeps weights non-negative by construction, which is the property the
    paper's multiplicative form was buying. The mapping is affine, so a
    normal perturbation in gene space is a normal perturbation in weight space.
  * Para-HADES OpenAI-ES adds Adam, rank scaling, mirrored sampling and sigma
    decay on top of the plain ES update. Strictly more machinery than the
    paper's `w * (1 + alpha * sigma * (e . N) / P)`. Set
    `ML.OpenAI_ES.{learning_rate, sigma}` explicitly (not `auto`) if you want
    to pin it closer to the paper's alpha = 1.0, sigma = 0.1.

DESIGN (A) DARWINIAN vs (B) LAMARCKIAN  --  `main.DotSNN.inheritance`
---------------------------------------------------------------------
After an offspring runs, its weights have moved twice: once by the EVOL
perturbation, once by STDP during its lifetime. What gets inherited?

  (A) 'darwinian'  [default]  Baldwin effect. Only the gene EVOL generated is
      recombined; STDP-learned changes are discarded at the end of the
      lifetime. Learning influences fitness, fitness shapes evolution, but
      acquired traits are not inherited. Implemented by returning fitness and
      nothing else -- Para-HADES recombines the gene it issued.
  (B) 'lamarckian'  The offspring's FINAL post-STDP weights are written back
      into the record as `gene`/`param`, so the next generation is built from
      what the offspring actually ended up with. Usually converges faster but
      can collapse population diversity.

Comparing (A) and (B) IS the interesting experiment, so the write-back is a
single swappable step (`_apply_inheritance`) rather than being threaded through
the evaluation.

MAPPING TO DOT TRACING
----------------------
The spec this implements was written against the paper's CartPole model. Those
parts are CartPole-specific and do not transfer, so they are re-derived here:

  paper                          this file
  -----                          ---------
  ES (80, 4 subpops of 20,       GC (560): fixed multi-scale periodic Gaussian
    Gaussian-percentile          basis over the egocentric pixel view. Ported
    receptive fields)            from examples/dotTracing/paper_replication1.py.
  EA                             AC: LIF, sparse fixed projection from GC,
                                 optional recurrence (the prey MOVES, so
                                 temporal integration is load-bearing).
  EM-L / EM-R, WTA on spike      MC: one subpopulation per action (5 or 9),
    counts over 50 ms            WTA on aggregate subpopulation spike count
                                 over the `granularity` ms decision window.
  critic on angle + angular      critic on distance-to-prey + closing rate,
    velocity                     same functional form (see `critic`).
  fitness = mean episode length  fitness = mean intercepts per episode
    (500-step ceiling)

The paper's plastic configuration (b) -- ES->EA AND EA->EM -- is the default
here as GC->AC and AC->MC. Configuration (a) (AC->MC only, which plateaued in
the paper) is `plastic: [ac_mc]`.

A NOTE ON THE "TIMESTEP"
------------------------
The paper's timestep is the 50 ms agent/environment sync interval, and its
normalization intervals ("every 25 timesteps", "over 500 timesteps") are in
those units. The dot-tracing analogue is one DECISION step -- `granularity` ms
of simulation -- not one BindsNET `dt`. All normalization counters below are in
decision steps.

STANDALONE MODES (no SLURM needed)
----------------------------------
    python task_dotSNN.py --gene_lengths
        Print the exact YAML `array:` lengths implied by the current
        main.DotSNN config. Run this after changing network size.

    python task_dotSNN.py --selftest
        Build the network, synthesize one perturbed offspring, run a short
        lifetime, and report the MANDATORY STDP-contribution diagnostic:
        ||w_final - w_start|| relative to the EVOL perturbation. If STDP's
        within-lifetime displacement is negligible next to the perturbation,
        the hybrid has silently degenerated to plain EVOL and the result will
        be a null. See RISK in the spec.
"""

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import yaml
import zstandard

from GA_utils_misc_func import pickle_loads_compat
from slurmHPCHelper import *

global args
global taskPath
global params_config

# Returned when an offspring cannot be evaluated at all. None (not 0.0) so that
# GA.py's report ingestion skips the record instead of treating a crash as a
# legitimately terrible offspring.
FAILURE_FITNESS_SCORE = None

# BindsNET lives one directory up from Para-HADES in the expected layout
# (<repo>/bindsnet and <repo>/Para-HADES). Fall back to whatever is installed.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if os.path.isdir(os.path.join(_REPO_ROOT, 'bindsnet')) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =========================================================================== #
# Configuration
# =========================================================================== #
DEFAULT_CFG = {
    # --- task / environment ---
    'steps': 100,              # decision steps per episode
    'dim': 28,
    'granularity': 100,        # ms of simulation per decision step
    'dt': 1.0,
    'decay': 4,
    'herrs': 0,
    'diag': False,             # 9 actions if True, else 5
    'randr': 0.15,
    'bound_hand': 'bounce',
    'fit_func': 'dir',
    'allow_stay': False,
    'view_r': 14,

    # --- GC layer (fixed, never plastic) ---
    'gc_scales': [3, 5, 7, 11, 13],
    'gc_rotations': 7,
    'gc_offsets': 16,
    'gc_global_scale': 1.0,
    'gc_sharpness': 1.0,
    'gc_max_rate': 80.0,

    # --- AC layer ---
    'ac': 400,
    # Convergence = number of presynaptic inputs per postsynaptic neuron.
    # These are the paper's Table 2 values (ES->EA 25, EA->EM 20). Convergence
    # is structural and unit-free, so unlike Table 2's conductance MAGNITUDES it
    # transfers directly from NEURON to BindsNET. See `--calibrate` for how the
    # magnitudes are set instead.
    'gc_ac_convergence': 25,
    'rec_mode': 'local',       # none | random | local
    'rec_inh': 1.2,
    'rec_radius': 4.0,

    # --- MC layer ---
    'mc_pop': 40,
    'ac_mc_convergence': 20,

    # --- model seed: fixes connectivity/topology, shared by ALL offspring ---
    'model_seed': 0,

    # --- what EVOL optimizes ---
    # Paper configuration (b) = both pathways. Configuration (a) = ['ac_mc'].
    'plastic': ['gc_ac', 'ac_mc'],
    'inheritance': 'darwinian',   # darwinian (A) | lamarckian (B)

    # --- STDP-RL (lifetime learning) ---
    'stdp_enabled': True,
    'nu': 1e-2,
    'eps_start': 0.0,          # WTA + random-on-tie; Poisson input is the
    'eps_min': 0.0,            # exploration source, as in the paper
    'eps_decay': 1.0,

    # --- critic ---
    'critic_mode': 'loss',     # loss (paper's form) | env (alignment reward)
    'max_reward': 1.0,
    'eta_positivity': 5.0,     # REQUIRED: without it the critic is
                               # punishment-dominated and weights collapse
    'eta_closing': 1.0,        # analogue of the paper's eta_angvel
    'critic_gain': 1.0,
    'success_eps': 1e-2,

    # --- weight normalization: MANDATORY with STDP live ---
    'norm_reception': True,
    'reception_every': 25,     # decision steps
    'norm_transmission': True,
    'transmission_min': 0.1,
    'transmission_max': 2.0,
    'transmission_invert': False,
    'norm_homeostasis': True,
    'homeostasis_every': 75,   # decision steps
    'rate_window': 500,        # decision steps
    'homeostasis_scale': 1e-4, # 0.01%
    'rate_target_ac': 5.5,     # Hz
    'rate_target_mc': 6.0,     # Hz

    # --- lifetime / fitness ---
    'episodes': 20,            # X. See RISK: 5 is very likely too short.
    'fitness_last_k': 5,       # for logged variant (ii)
    'frozen_episodes': 5,      # for logged variant (iii); 0 disables
    'fitness_metric': 'intercepts',   # intercepts | reward
    'episode_seed_base': 12345,
    # 'common' -> every offspring in a generation faces the SAME episodes
    # (common random numbers; sharply reduces ES fitness noise). 'random' ->
    # fresh episodes per offspring, closer to the spec's literal wording.
    'seed_mode': 'common',

    # --- misc ---
    'gpu': False,
    'cache_gc': True,
    'verbose': True,
}

_PATHWAY_TARGET_POP = {'gc_ac': 'AC', 'ac_mc': 'MC'}

LAYER_GC, LAYER_AC, LAYER_MC = 'GC', 'AC', 'MC'


def load_cfg(param_file):
    """DEFAULT_CFG overlaid with main.DotSNN from the YAML."""
    cfg = dict(DEFAULT_CFG)
    if isinstance(param_file, str) and os.path.exists(param_file):
        with open(param_file, 'rt') as f:
            y = yaml.safe_load(f) or {}
        user = ((y.get('main', {}) or {}).get('DotSNN', {}) or {})
        unknown = [k for k in user if k not in DEFAULT_CFG]
        if unknown:
            print(f'WARNING: unknown main.DotSNN keys ignored: {unknown}', flush=True)
        cfg.update({k: v for k, v in user.items() if k in DEFAULT_CFG})
    return cfg


def load_param_bounds(param_file):
    """{gene_key: (min, max)} from the YAML `param:` block, in declaration order.

    These bounds are the single source of truth: they define both the EVOL
    search box and the STDP clamp box, so the two mechanisms cannot disagree
    about what a legal weight is.
    """
    bounds = {}
    if isinstance(param_file, str) and os.path.exists(param_file):
        with open(param_file, 'rt') as f:
            y = yaml.safe_load(f) or {}
        for key, spec in (y.get('param', {}) or {}).items():
            bounds[key] = (float(spec['min']), float(spec['max']))
    return bounds


# =========================================================================== #
# View geometry  (ported from examples/dotTracing/paper_replication1.py)
# =========================================================================== #
def getLocalView(obs, row, col, view_r):
    """Flattened (2r+1)^2 - 1 egocentric patch, agent's own cell excluded."""
    padded = np.pad(obs, view_r, mode='constant', constant_values=0)
    patch = padded[row: row + 2 * view_r + 1, col: col + 2 * view_r + 1]
    flat = patch.flatten()
    center = view_r * (2 * view_r + 1) + view_r
    return np.concatenate([flat[:center], flat[center + 1:]])


def viewIndexToOffset(view_r):
    """(drow, dcol) of every input index, matching getLocalView's layout."""
    side = 2 * view_r + 1
    center = view_r * side + view_r
    offs = []
    for k in range(side * side - 1):
        cell = k if k < center else k + 1
        pr, pc = divmod(cell, side)
        offs.append((pr - view_r, pc - view_r))
    return np.array(offs, dtype=np.float32)


# =========================================================================== #
# GC layer -- fixed multi-scale periodic basis
# =========================================================================== #
def _hexLattice(spacing, rotation, phase, extent):
    import itertools
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


def buildGridCellProjection(cfg, cache_dir=None):
    """Fixed (inpt_n, n_gc) projection; column j is grid cell j's receptive
    field over the egocentric view. Cached to disk -- it is identical for every
    offspring and costs seconds to rebuild."""
    import torch

    view_r = cfg['view_r']
    key = json.dumps({k: cfg[k] for k in
                      ('view_r', 'gc_scales', 'gc_rotations', 'gc_offsets',
                       'gc_global_scale', 'gc_sharpness')}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    cache_path = (os.path.join(cache_dir, f'gc_proj_{digest}.npz')
                  if (cache_dir and cfg['cache_gc']) else None)

    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            return torch.from_numpy(z['W']), torch.from_numpy(z['peak'])
        except Exception as exc:
            print(f'WARNING: GC cache unreadable ({exc}); rebuilding.', flush=True)

    offs = viewIndexToOffset(view_r)
    extent = float(view_r)
    n_rot = cfg['gc_rotations']
    rotations = [np.pi * i / max(1, n_rot) for i in range(n_rot)]
    side = int(round(np.sqrt(cfg['gc_offsets'])))
    g, k = cfg['gc_global_scale'], cfg['gc_sharpness']

    cols = []
    for s in cfg['gc_scales']:
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

    W = np.stack(cols, axis=1)
    peak = W.max(axis=0, keepdims=True)
    peak[peak == 0] = 1.0

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, W=W, peak=peak)
        except Exception as exc:
            print(f'WARNING: could not write GC cache ({exc}).', flush=True)

    return torch.from_numpy(W), torch.from_numpy(peak)


def gcRates(view_vec, W_grid, peak, max_rate):
    """view (inpt_n,) -> grid-cell rates (n_gc,). Paper: f = p/p_max * f_max."""
    drive = view_vec @ W_grid
    drive = (drive / peak.squeeze(0)).clamp(0.0, 1.0)
    drive[drive < 0.01] = 0.0
    return drive * max_rate


# =========================================================================== #
# Connectivity
# =========================================================================== #
def buildSparseMaskExact(n_pre, n_post, convergence, seed, device):
    """Sparse mask with EXACTLY `convergence` presynaptic sources per target.

    Diverges from paper_replication1.py's `rand < sparsity`, which gives a
    stochastic synapse count. Two reasons to pin it instead:
      * a fixed-length gene needs an exact, reproducible synapse count --
        gene length is then n_post * convergence exactly;
      * convergence is the quantity the paper actually specifies (Table 2),
        and it is what sets each neuron's total input drive.
    """
    import torch

    k = max(1, min(int(convergence), n_pre))
    g = torch.Generator().manual_seed(int(seed))
    mask = torch.zeros(n_pre, n_post)
    for j in range(n_post):
        idx = torch.randperm(n_pre, generator=g)[:k]
        mask[idx, j] = 1.0
    return mask.to(device), k


def acPlaceCentres(W_grid, gc_ac_mask, view_r):
    """Argmax of each AC's effective receptive field -- used only to give the
    recurrent inhibition a spatial structure."""
    import torch

    eff = W_grid @ gc_ac_mask
    idx = eff.argmax(dim=0).cpu().numpy()
    offs = viewIndexToOffset(view_r)
    return torch.from_numpy(offs[idx]).float()


def buildRecurrent(cfg, centres, device):
    import torch

    mode, n_ac = cfg['rec_mode'], cfg['ac']
    if mode == 'none':
        return None
    if mode == 'random':
        g = torch.Generator().manual_seed(int(cfg['model_seed']))
        signs = (torch.rand(n_ac, n_ac, generator=g) < 0.45).float() * 2 - 1
        W = torch.rand(n_ac, n_ac, generator=g) * signs
    else:  # local: short-range excitation, broad surround inhibition
        d = torch.cdist(centres, centres)
        exc = torch.exp(-(d ** 2) / (2 * cfg['rec_radius'] ** 2))
        W = exc - cfg['rec_inh'] * (1.0 - exc)
    W.fill_diagonal_(0.0)
    return W.to(device)


# =========================================================================== #
# Plastic parameter vector
# =========================================================================== #
class PlasticParam:
    """One plastic pathway, exposed as a flat slice of the parameter vector.

    Deliberately an abstraction over "the thing EVOL and STDP both optimize"
    rather than hardcoded weights: swapping in synaptic delays (which the paper
    explicitly did NOT optimize, and for which this repo now has a `Delay`
    feature) means changing only `tensor()`, not the algorithm.
    """

    def __init__(self, name, feature, mask, wmin, wmax):
        self.name = name
        self.feature = feature
        self.mask = mask
        self.wmin = wmin
        self.wmax = wmax
        # Row-major flat indices of the live synapses. torch.nonzero is sorted,
        # so this order is deterministic across processes and runs.
        self.idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        self.n_pre, self.n_post = mask.shape

    def tensor(self):
        return self.feature.value

    def get(self):
        return self.tensor().reshape(-1)[self.idx].detach().clone()

    def set(self, vec):
        flat = self.tensor().reshape(-1)
        flat[self.idx] = vec.to(flat.dtype).to(flat.device)

    def enforce(self):
        """Re-apply sparsity and bounds. Called after every STDP update.

        Note the floor is 0.0, not a positive epsilon: EVOL is expected to
        drive some weights low enough to effectively silence neurons, and the
        paper treats that pruning as a real finding. It is not clipped away.
        """
        import torch

        with torch.no_grad():
            self.tensor().mul_(self.mask).clamp_(self.wmin, self.wmax)

    def col_sums(self):
        return (self.tensor() * self.mask).sum(0)

    def row_sums(self):
        return (self.tensor() * self.mask).sum(1)


def plastic_lengths(cfg):
    """{gene_key: length} for the pathways in cfg['plastic'], without building
    a network. Used by --gene_lengths to generate YAML `array:` values."""
    n_gc = len(cfg['gc_scales']) * cfg['gc_rotations'] * cfg['gc_offsets']
    n_mc = (9 if cfg['diag'] else 5) * cfg['mc_pop']
    k_gc_ac = max(1, min(int(cfg['gc_ac_convergence']), n_gc))
    k_ac_mc = max(1, min(int(cfg['ac_mc_convergence']), cfg['ac']))
    all_lengths = {
        'w_gc_ac': cfg['ac'] * k_gc_ac,
        'w_ac_mc': n_mc * k_ac_mc,
    }
    return {f'w_{p}': all_lengths[f'w_{p}'] for p in cfg['plastic']}


def gene_key(pathway):
    return f'w_{pathway}'


def get_plastic_vector(registry):
    """Flatten every plastic pathway into ONE vector, in registry order."""
    import torch

    return torch.cat([p.get() for p in registry])


def set_plastic_vector(registry, vec):
    """Inverse of get_plastic_vector. Order must match exactly."""
    off = 0
    for p in registry:
        n = p.idx.numel()
        p.set(vec[off:off + n])
        p.enforce()
        off += n
    if off != vec.numel():
        raise ValueError(f'plastic vector length {vec.numel()} != expected {off}')


# =========================================================================== #
# Network
# =========================================================================== #
class DotSNN:
    """The GC -> AC -> MC model plus everything an offspring evaluation needs.

    Built ONCE per task process and reused across genes: topology, the GC
    projection and the sparsity masks are fixed by `model_seed` and are shared
    by the whole population, exactly as the paper requires. Only the plastic
    vector changes between offspring.
    """

    def __init__(self, cfg, bounds, cache_dir=None):
        import torch
        from bindsnet.learning.MCC_learning import MSTDPET
        from bindsnet.network import Network
        from bindsnet.network.monitors import Monitor
        from bindsnet.network.nodes import Input, LIFNodes
        from bindsnet.network.topology import MulticompartmentConnection
        from bindsnet.network.topology_features import Weight

        self.cfg = cfg
        self.device = torch.device('cuda' if (cfg['gpu'] and torch.cuda.is_available())
                                   else 'cpu')
        self.n_actions = 9 if cfg['diag'] else 5
        self.gran = int(cfg['granularity'])

        torch.manual_seed(int(cfg['model_seed']))
        np.random.seed(int(cfg['model_seed']))

        # --- GC projection ---
        W_grid, peak = buildGridCellProjection(cfg, cache_dir)
        self.W_grid = W_grid.to(self.device)
        self.peak = peak.to(self.device)
        n_gc = self.W_grid.shape[1]
        self.n_gc = n_gc

        net = Network(dt=cfg['dt'])
        gc = Input(n=n_gc, shape=[1, 1, 1, 1, n_gc], traces=True)
        ac = LIFNodes(n=cfg['ac'], traces=True, rest=-64.0, reset=-70.0,
                      thresh=-45.0, refrac=1, tc_decay=20.0, tc_trace=20.0)
        n_mc = self.n_actions * cfg['mc_pop']
        mc = LIFNodes(n=n_mc, traces=True, rest=-64.0, reset=-64.0,
                      thresh=-49.0, refrac=0, tc_decay=20.0, tc_trace=20.0)
        self.n_mc = n_mc
        net.add_layer(gc, name=LAYER_GC)
        net.add_layer(ac, name=LAYER_AC)
        net.add_layer(mc, name=LAYER_MC)

        plastic_set = set(cfg['plastic'])
        for p in plastic_set:
            if p not in _PATHWAY_TARGET_POP:
                raise ValueError(f"unknown plastic pathway '{p}'; "
                                 f"expected any of {sorted(_PATHWAY_TARGET_POP)}")

        def rule_kwargs(pathway, wmin, wmax):
            """MSTDPET only on pathways EVOL and STDP share; `range` gives the
            MCC learning rule its clamp box (see MCC_learning.update)."""
            if pathway in plastic_set and cfg['stdp_enabled']:
                return dict(learning_rule=MSTDPET, nu=[cfg['nu'], cfg['nu']],
                            range=[wmin, wmax])
            return dict(range=[wmin, wmax])

        # --- GC -> AC ---
        # w_0 is the MIDPOINT of the YAML box, uniformly. Two reasons:
        #   * The spec says to initialize from the paper's Table 2 and NOT to
        #     re-randomize. Table 2's conductances are NEURON units and do not
        #     transfer, but the box is calibrated so its midpoint reproduces the
        #     paper's target firing rates (see `--calibrate`), which is the
        #     unit-free version of the same statement.
        #   * Para-HADES generates the first population as N(midpoint, range/6)
        #     per synapse, so w_0 = midpoint is exactly the mean of generation 0.
        #     Symmetry is broken by the gene, not by w_0.
        gc_ac_mask, k_gc_ac = buildSparseMaskExact(
            n_gc, cfg['ac'], cfg['gc_ac_convergence'], cfg['model_seed'], self.device)
        w_lo, w_hi = bounds.get(gene_key('gc_ac'), (0.0, 20.0))
        W_gc_ac = gc_ac_mask * (0.5 * (w_lo + w_hi))
        feat_gc_ac = Weight(name='w_gc_ac', value=W_gc_ac,
                            **rule_kwargs('gc_ac', w_lo, w_hi))
        net.add_connection(
            MulticompartmentConnection(source=gc, target=ac,
                                       pipeline=[feat_gc_ac], device=self.device),
            source=LAYER_GC, target=LAYER_AC)

        # --- AC -> AC (never plastic, never evolved: topology is fixed) ---
        centres = acPlaceCentres(self.W_grid, gc_ac_mask, cfg['view_r'])
        W_rec = buildRecurrent(cfg, centres, self.device)
        if W_rec is not None:
            feat_rec = Weight(name='w_rec', value=W_rec,
                              range=[float(W_rec.min()), float(W_rec.max())])
            net.add_connection(
                MulticompartmentConnection(source=ac, target=ac,
                                           pipeline=[feat_rec], device=self.device),
                source=LAYER_AC, target=LAYER_AC)

        # --- AC -> MC ---
        ac_mc_mask, k_ac_mc = buildSparseMaskExact(
            cfg['ac'], n_mc, cfg['ac_mc_convergence'], cfg['model_seed'] + 1,
            self.device)
        m_lo, m_hi = bounds.get(gene_key('ac_mc'), (0.0, 13.0))
        W_ac_mc = ac_mc_mask * (0.5 * (m_lo + m_hi))
        feat_ac_mc = Weight(name='w_ac_mc', value=W_ac_mc,
                            **rule_kwargs('ac_mc', m_lo, m_hi))
        net.add_connection(
            MulticompartmentConnection(source=ac, target=mc,
                                       pipeline=[feat_ac_mc], device=self.device),
            source=LAYER_AC, target=LAYER_MC)

        net.to(self.device)
        self.net = net

        # --- plastic registry, in the SAME order as cfg['plastic'] ---
        avail = {
            'gc_ac': (feat_gc_ac, gc_ac_mask, w_lo, w_hi),
            'ac_mc': (feat_ac_mc, ac_mc_mask, m_lo, m_hi),
        }
        self.registry = [PlasticParam(name, *avail[name][:2], *avail[name][2:])
                         for name in cfg['plastic']]
        self.all_params = {name: PlasticParam(name, *avail[name][:2], *avail[name][2:])
                           for name in avail}

        # w_0: the pristine construction-time weights. Kept ONLY as the
        # reference the reception-balancing guard checks against -- it must
        # never become a renormalization target (see WeightNormalizer).
        self.w0 = get_plastic_vector(self.registry)
        self.w0_col_sums = {p.name: p.col_sums().clone() for p in self.registry}

        self.monitors = {}
        for layer in (LAYER_AC, LAYER_MC):
            m = Monitor(net.layers[layer], state_vars=['s'],
                        time=int(self.gran / cfg['dt']), device=self.device)
            net.add_monitor(m, name=layer)
            self.monitors[layer] = m

        self.gene_lengths = {gene_key(p.name): p.idx.numel() for p in self.registry}
        self.convergence = {'gc_ac': k_gc_ac, 'ac_mc': k_ac_mc}
        self.bounds_used = {'gc_ac': (w_lo, w_hi), 'ac_mc': (m_lo, m_hi)}

    def encode(self, view):
        from bindsnet.encoding import poisson

        rates = gcRates(view, self.W_grid, self.peak, self.cfg['gc_max_rate'])
        return {LAYER_GC: poisson(rates.unsqueeze(0), self.gran,
                                  self.cfg['dt'], device=self.device)}


# =========================================================================== #
# Critic
# =========================================================================== #
def critic(cfg, loss_prev, loss_now, moved, success):
    """Reward-modulated STDP needs a task signal; this is it.

    Same functional form as the paper's hand-crafted critic, with its
    (angle, angular velocity) state error replaced by dot-tracing's
    (distance, closing rate) -- see `episode_loss`.

        reward = 0                              if loss_prev < success_eps
               = -max_reward / eta_positivity   if no move was decided
               = +max_reward / eta_positivity   if loss_now < success_eps
               = loss_prev - loss_now           otherwise
        f(r)   = r * eta_positivity  if r > 0 else r
        critic = clip(f(reward) * gain, -max_reward, +max_reward)

    eta_positivity is the positivity bias and is REQUIRED: without it the
    critic is punishment-dominated early and the weights collapse to zero.
    """
    mr = cfg['max_reward']
    eta = cfg['eta_positivity']
    eps = cfg['success_eps']

    if loss_prev is not None and loss_prev < eps:
        r = 0.0
    elif not moved:
        r = -mr / eta
    elif success or loss_now < eps:
        r = mr / eta
    else:
        r = (loss_prev - loss_now) if loss_prev is not None else 0.0

    r = r * eta if r > 0 else r
    return float(np.clip(r * cfg['critic_gain'], -mr, mr))


def state_loss(cfg, dist, closing):
    """loss = sqrt(dist^2 + eta_closing * closing^2).

    The paper's loss is sqrt(angle^2 + eta_angvel * angvel^2): distance from
    the goal state plus how fast you are drifting. Here the goal is distance 0,
    and `closing` is the per-step change in distance. As in the paper this uses
    only these two terms -- notably NOT absolute position -- so expect the same
    measured bias: STDP-RL will form action associations mostly with the rate
    term, while EVOL picks up both.
    """
    return float(np.sqrt(dist ** 2 + cfg['eta_closing'] * closing ** 2))


def env_reward(prev, curr, target_prev, target_curr, intercept):
    """paper_replication1.py's dense reward: sign from outcome, magnitude from
    alignment. Selected by critic_mode='env'; still passed through the same
    positivity/clip machinery so the two modes stay comparable."""
    pr, pc = prev
    cr, cc = curr
    tr_p, tc_p = target_prev
    tr, tc = target_curr

    prev_dist = np.hypot(pr - tr_p, pc - tc_p)
    curr_dist = np.hypot(cr - tr, cc - tc)
    delta = prev_dist - curr_dist

    dr_p, dc_p = tr_p - pr, tc_p - pc
    dr_m, dc_m = cr - pr, cc - pc
    dmag, mmag = np.hypot(dr_p, dc_p), np.hypot(dr_m, dc_m)
    alignment = 0.0 if (mmag < 1e-8 or dmag < 1e-8) else \
        float(np.clip((dr_m * dr_p + dc_m * dc_p) / (dmag * mmag), -1, 1))
    scale = 0.5 + 0.5 * alignment
    r = delta * ((0.5 + scale) if delta >= 0 else (1.5 - scale))
    r += 0.2 * alignment
    if intercept:
        r += 20.0
    return r


# =========================================================================== #
# Weight normalization
# =========================================================================== #
class ReceptionTargetError(RuntimeError):
    pass


class WeightNormalizer:
    """The paper's three normalization mechanisms.

    In the pure-EVOL paper these were ambiguous. With STDP live they are not
    optional: weights "tend to increase without bound, leading to epileptic
    activity."

      1. Reception balancing -- every `reception_every` decision steps,
         renormalize the summed input weight onto each postsynaptic neuron back
         to its initial value.
      2. Transmission scaling -- scale each synapse's reinforcement by that
         presynaptic neuron's total outgoing weight, factor capped to
         [transmission_min, transmission_max].
      3. Homeostatic gain control -- every `homeostasis_every` steps, if a
         population's firing rate (over `rate_window` steps) deviates from
         target, nudge the expected total reception weight by
         `homeostasis_scale`.

    THE CRITICAL INTERACTION: reception balancing renormalizes to the INITIAL
    summed weight, and under the hybrid "initial" must mean w_j_start -- that
    offspring's EVOL-assigned weights -- NOT the global w_0. If it snapped back
    to w_0 it would silently undo the EVOL perturbation and the whole method
    would degenerate to plain STDP-RL. That is the single most likely silent
    failure here, so targets are captured from the live weights immediately
    after the gene is loaded, and `assert_targets_track_offspring` checks it.
    """

    def __init__(self, cfg, registry, n_neurons):
        import torch

        self.cfg = cfg
        self.registry = registry
        self.torch = torch

        # Captured from the weights AS THEY ARE NOW, i.e. w_j_start.
        self.recept_target = {p.name: p.col_sums().clone() for p in registry}
        self.trans_ref = {p.name: p.row_sums().clone() for p in registry}
        self.gain = {p.name: 1.0 for p in registry}

        self.n_neurons = n_neurons
        self.spike_hist = {pop: deque(maxlen=int(cfg['rate_window']))
                           for pop in (LAYER_AC, LAYER_MC)}
        self.rate_targets = {LAYER_AC: cfg['rate_target_ac'],
                             LAYER_MC: cfg['rate_target_mc']}
        self.last_rates = {}

    # -- guard ------------------------------------------------------------- #
    def assert_targets_track_offspring(self, w0_col_sums, w_start, w0, tol=1e-6):
        """Fail loudly if the renormalization targets came from w_0 rather than
        from this offspring's starting weights."""
        perturbation = float((w_start - w0).norm())
        if perturbation <= tol:
            return  # gene == w0, nothing to distinguish
        matches_w0 = all(
            self.torch.allclose(self.recept_target[p.name], w0_col_sums[p.name],
                                atol=tol)
            for p in self.registry)
        if matches_w0:
            raise ReceptionTargetError(
                'Reception-balancing targets equal the GLOBAL w_0 column sums '
                f'even though this offspring is {perturbation:.4g} away from '
                'w_0. Balancing would undo the EVOL perturbation and the '
                'hybrid would degenerate to plain STDP-RL. Targets must be '
                'captured from w_j_start.')

    # -- 2. transmission scaling ------------------------------------------- #
    def scale_reinforcement(self, before):
        """Rescale the STDP increment this step by presynaptic outgoing weight.

        MSTDPET has already written w_after; `before` is the snapshot from
        before net.run, so delta = w_after - before is exactly the increment.
        Rescaling the delta rather than patching the learning rule keeps this
        mechanism independent of BindsNET internals.

        Direction note: the spec says "scale each synapse's reinforcement by
        that presynaptic neuron's total outgoing weight", read here as
        proportional to current outgoing weight (factor 1.0 at the start of the
        lifetime). `transmission_invert: True` selects the opposite,
        conservation-style reading, where prolific presynaptic cells have their
        increments damped instead.
        """
        torch = self.torch
        lo, hi = self.cfg['transmission_min'], self.cfg['transmission_max']
        with torch.no_grad():
            for p in self.registry:
                ref = self.trans_ref[p.name]
                out_now = p.row_sums()
                factor = out_now / ref.clamp(min=1e-8)
                if self.cfg['transmission_invert']:
                    factor = 1.0 / factor.clamp(min=1e-8)
                factor = factor.clamp(lo, hi).unsqueeze(1)   # per presynaptic row
                w = p.tensor()
                delta = w - before[p.name]
                w.copy_(before[p.name] + delta * factor)
                p.enforce()

    # -- 1. reception balancing -------------------------------------------- #
    def balance_reception(self):
        torch = self.torch
        with torch.no_grad():
            for p in self.registry:
                target = self.recept_target[p.name] * self.gain[p.name]
                current = p.col_sums()
                scale = target / current.clamp(min=1e-8)
                # Leave silenced columns silenced: a target of 0 must not be
                # resurrected, and a column STDP drove to 0 has no direction to
                # be scaled in.
                scale = torch.where(current > 1e-8, scale, torch.ones_like(scale))
                p.tensor().mul_(scale.unsqueeze(0))
                p.enforce()

    # -- 3. homeostatic gain control --------------------------------------- #
    def record_spikes(self, counts):
        for pop, c in counts.items():
            self.spike_hist[pop].append(float(c))

    def apply_homeostasis(self, seconds_per_step):
        """Nudge the expected total reception weight of each plastic pathway's
        TARGET population toward its firing-rate set point."""
        s = self.cfg['homeostasis_scale']
        for p in self.registry:
            pop = _PATHWAY_TARGET_POP[p.name]
            hist = self.spike_hist[pop]
            if not hist:
                continue
            elapsed = len(hist) * seconds_per_step
            rate = sum(hist) / (self.n_neurons[pop] * max(elapsed, 1e-9))
            self.last_rates[pop] = rate
            tgt = self.rate_targets[pop]
            if rate > tgt:
                self.gain[p.name] *= (1.0 - s)
            elif rate < tgt:
                self.gain[p.name] *= (1.0 + s)


# =========================================================================== #
# Episode / lifetime
# =========================================================================== #
def make_env(cfg, out_path):
    from bindsnet.environment.dot_simulator import DotSimulator

    return DotSimulator(
        int(cfg['steps']), decay=cfg['decay'], herrs=cfg['herrs'],
        diag=cfg['diag'], randr=cfg['randr'], write=False, mute=True,
        bound_hand=cfg['bound_hand'], fit_func=cfg['fit_func'],
        allow_stay=cfg['allow_stay'], pandas=False, fpath=out_path,
        height=cfg['dim'], width=cfg['dim'],
    )


def wta_action(mc_spikes, n_actions, eps, rng):
    """Paper's action rule: WTA on aggregate subpopulation spike count over the
    decision window, random move on a tie (or on a silent motor pool).

    Returns (action, moved). `moved` is False when no move was decided -- the
    case the critic punishes.
    """
    import torch

    if eps > 0 and rng.random() < eps:
        return int(rng.integers(0, n_actions)), True
    counts = mc_spikes.sum(0)
    agg = torch.stack([p.sum() for p in torch.chunk(counts, n_actions)])
    if float(agg.max()) == 0:
        return int(rng.integers(0, n_actions)), False
    top = float(agg.max())
    winners = (agg == top).nonzero(as_tuple=False).squeeze(1)
    if winners.numel() > 1:
        choice = int(winners[rng.integers(0, winners.numel())].item())
        return choice, True
    return int(agg.argmax().item()), True


def run_episode(model, env, normalizer, cfg, episode_seed, learning, step_counter):
    """One episode. Weights are NOT reloaded here -- persistence across
    episodes within a lifetime is the point of the hybrid."""
    import torch

    net = model.net
    rng = np.random.default_rng(episode_seed)

    # DotSimulator.reset() reseeds from self.seed, so without this every
    # episode would be byte-identical. Setting it per episode is what makes
    # "random training initializations" and "fixed validation episodes" real.
    env.seed = int(episode_seed)
    env.reset()
    net.reset_state_variables()

    eps = cfg['eps_start'] if learning else 0.0
    action = int(rng.integers(0, model.n_actions))
    # Whether the action about to be executed was actually DECIDED by the motor
    # pool, or fell back to random because MC was silent. The critic punishes
    # the latter, so it has to be carried from the step that chose the action.
    moved = True
    total_reward, intercepts, steps, no_move = 0.0, 0, 0, 0
    loss_prev = None
    done = False
    seconds_per_step = model.gran * cfg['dt'] / 1000.0

    while not done:
        steps += 1
        step_counter[0] += 1

        prev = (env.netDot.row[0], env.netDot.col[0])
        target_prev = (env.dots[0].row[0], env.dots[0].col[0])
        prev_dist = float(np.hypot(prev[0] - target_prev[0], prev[1] - target_prev[1]))

        obs, _, done, intercept = env.step(action)

        curr = (env.netDot.row[0], env.netDot.col[0])
        target_curr = (env.dots[0].row[0], env.dots[0].col[0])
        curr_dist = float(np.hypot(curr[0] - target_curr[0], curr[1] - target_curr[1]))

        # --- encode the egocentric view (agent's own pixel removed) ---
        net_obs = obs.copy()
        net_obs[env.netDot.row, env.netDot.col] = 0.0
        view = torch.as_tensor(
            getLocalView(net_obs, curr[0], curr[1], cfg['view_r']),
            dtype=torch.float32, device=model.device)

        # --- critic ---
        if cfg['critic_mode'] == 'env':
            raw = env_reward(prev, curr, target_prev, target_curr, intercept)
            r = float(np.clip(
                (raw * cfg['eta_positivity'] if raw > 0 else raw) * cfg['critic_gain'],
                -cfg['max_reward'], cfg['max_reward']))
        else:
            loss_now = state_loss(cfg, curr_dist, curr_dist - prev_dist)
            r = critic(cfg, loss_prev, loss_now, moved=moved,
                       success=bool(intercept))
            loss_prev = loss_now

        reward = torch.tensor(r, dtype=torch.float32, device=model.device)

        # --- snapshot for transmission scaling, then run ---
        before = None
        if learning and cfg['norm_transmission']:
            before = {p.name: p.tensor().detach().clone() for p in model.registry}

        net.run(inputs=model.encode(view), time=model.gran, reward=reward)

        if learning:
            if before is not None:
                normalizer.scale_reinforcement(before)
            else:
                for p in model.registry:
                    p.enforce()

        # --- read out ---
        mc = model.monitors[LAYER_MC].get('s').squeeze()
        ac = model.monitors[LAYER_AC].get('s').squeeze()
        if normalizer is not None:
            normalizer.record_spikes({LAYER_AC: float(ac.sum()),
                                      LAYER_MC: float(mc.sum())})

        action, moved = wta_action(mc, model.n_actions, eps, rng)
        no_move += int(not moved)

        if learning and cfg['eps_decay'] < 1.0:
            eps = max(cfg['eps_min'], eps * cfg['eps_decay'])

        # --- periodic normalization, counted in DECISION steps ---
        if learning and normalizer is not None:
            n = step_counter[0]
            if cfg['norm_homeostasis'] and n % int(cfg['homeostasis_every']) == 0:
                normalizer.apply_homeostasis(seconds_per_step)
            if cfg['norm_reception'] and n % int(cfg['reception_every']) == 0:
                normalizer.balance_reception()

        total_reward += (r if cfg['critic_mode'] != 'env'
                         else env_reward(prev, curr, target_prev, target_curr, intercept))
        intercepts += int(bool(intercept))

    return {'reward': total_reward, 'intercepts': intercepts, 'steps': steps,
            'no_move': no_move}


def run_lifetime(model, cfg, w_start, out_path, iteration=0, offspring=0):
    """Evaluate ONE offspring: load its EVOL-assigned starting weights, let it
    live and learn, and measure fitness.

    Returns a dict of fitness variants and diagnostics.
    """
    import torch

    # (a) reset state, load w_j_start
    model.net.reset_state_variables()
    set_plastic_vector(model.registry, w_start)
    w_start_actual = get_plastic_vector(model.registry)

    # (b) STDP-RL is enabled by construction (MSTDPET on the plastic
    #     pathways). This is the hybrid's core departure from the paper.
    normalizer = WeightNormalizer(
        cfg, model.registry,
        n_neurons={LAYER_AC: cfg['ac'], LAYER_MC: model.n_mc})
    normalizer.assert_targets_track_offspring(
        model.w0_col_sums, w_start_actual, model.w0)

    env = make_env(cfg, out_path)
    metric = cfg['fitness_metric']
    X = int(cfg['episodes'])

    # Episode seeds. 'common' gives every offspring in a generation the same
    # episodes (common random numbers), so fitness differences reflect the
    # offspring rather than the draw -- this matters a lot at P=10.
    base = int(cfg['episode_seed_base'])
    if cfg['seed_mode'] == 'common':
        seed_rng = np.random.default_rng(base + 1000003 * int(iteration))
    else:
        seed_rng = np.random.default_rng(base + 1000003 * int(iteration)
                                         + 7919 * int(offspring))
    train_seeds = [int(s) for s in seed_rng.integers(1, 2 ** 31 - 1, size=X)]

    # (c) X episodes; weights PERSIST across them, only state resets.
    step_counter = [0]
    per_ep = []
    for ep, seed in enumerate(train_seeds):
        res = run_episode(model, env, normalizer, cfg, seed,
                          learning=cfg['stdp_enabled'], step_counter=step_counter)
        per_ep.append(res)
        if cfg['verbose']:
            print(f'    ep {ep + 1}/{X} seed={seed} '
                  f'intercepts={res["intercepts"]} reward={res["reward"]:+.2f}',
                  flush=True)

    scores = [e[metric] for e in per_ep]
    w_final = get_plastic_vector(model.registry)

    # (iii) freeze weights and run a separate held-out evaluation -- cleanest
    # separation of "what was learned" from "how fast it was learned".
    frozen_scores = []
    n_frozen = int(cfg['frozen_episodes'])
    if n_frozen > 0:
        held_rng = np.random.default_rng(base + 99991 + 1000003 * int(iteration))
        held_seeds = [int(s) for s in held_rng.integers(1, 2 ** 31 - 1, size=n_frozen)]
        frozen_counter = [0]
        for seed in held_seeds:
            res = run_episode(model, env, None, cfg, seed,
                              learning=False, step_counter=frozen_counter)
            frozen_scores.append(res[metric])
        # The held-out run must not leak into the lifetime it is measuring.
        set_plastic_vector(model.registry, w_final)

    # --- the MANDATORY diagnostic: did STDP actually do anything? ---
    dw = float((w_final - w_start_actual).norm())
    perturb = float((w_start_actual - model.w0).norm())
    k = max(1, min(int(cfg['fitness_last_k']), len(scores)))

    out = {
        # (i) mean over all X episodes -- selects for a mix of innate ability
        # and learning speed; closest to the paper's definition.
        'fitness_mean_all': float(np.mean(scores)),
        # (ii) mean over the last k -- selects for post-learning ceiling.
        'fitness_last_k': float(np.mean(scores[-k:])),
        # (iii) frozen held-out evaluation.
        'fitness_frozen': float(np.mean(frozen_scores)) if frozen_scores else None,
        'median_all': float(np.median(scores)),
        'mean_reward': float(np.mean([e['reward'] for e in per_ep])),
        'mean_steps': float(np.mean([e['steps'] for e in per_ep])),
        # Fraction of decisions where MC was silent and the move was random.
        # High values mean the motor pool is not driving behaviour at all.
        'no_move_frac': float(np.sum([e['no_move'] for e in per_ep])
                              / max(1, np.sum([e['steps'] for e in per_ep]))),
        'per_episode': [float(s) for s in scores],
        # diagnostics
        'stdp_dw_l2': dw,
        'evol_perturbation_l2': perturb,
        'stdp_vs_evol_ratio': (dw / perturb) if perturb > 1e-12 else None,
        'w_start_l2': float(w_start_actual.norm()),
        'homeostatic_gain': {k2: float(v) for k2, v in normalizer.gain.items()},
        'firing_rates_hz': {k2: float(v) for k2, v in normalizer.last_rates.items()},
    }
    out['fitnessScore'] = out['fitness_mean_all']
    return out, w_final


# =========================================================================== #
# Gene <-> weight mapping
# =========================================================================== #
def gene_to_weights(param, model, bounds):
    """Decoded YAML params -> the flat plastic vector, in registry order.

    Para-HADES has already mapped the normalized gene through [min, max], so
    `param['param'][key]` is already in weight units.
    """
    import torch

    parts = []
    for p in model.registry:
        key = gene_key(p.name)
        if key not in param['param']:
            raise KeyError(f"gene is missing '{key}'; the YAML `param:` block "
                           f"must declare one array per plastic pathway "
                           f"({[gene_key(q.name) for q in model.registry]})")
        arr = np.asarray(param['param'][key], dtype=np.float32).ravel()
        expected = p.idx.numel()
        if arr.size != expected:
            raise ValueError(
                f"gene key '{key}' has {arr.size} values but this network has "
                f"{expected} live synapses on that pathway. Fix the YAML "
                f"`array:` length -- run `python {os.path.basename(__file__)} "
                f"--gene_lengths` to print the correct values.")
        lo, hi = bounds.get(key, (p.wmin, p.wmax))
        parts.append(torch.from_numpy(np.clip(arr, lo, hi)))
    return torch.cat(parts).to(model.device)


def weights_to_gene(vec, model, bounds, gene_min):
    """Inverse mapping, for Lamarckian write-back.

    Mirrors GA_class.paramTOgene: gene = 2*(w - min)/(max - min) - 1 when
    geneMin != 0, else (w - min)/(max - min).
    """
    params, gene = {}, []
    off = 0
    for p in model.registry:
        key = gene_key(p.name)
        n = p.idx.numel()
        w = vec[off:off + n].detach().cpu().numpy().astype(np.float32)
        off += n
        lo, hi = bounds.get(key, (p.wmin, p.wmax))
        span = (hi - lo) if hi > lo else 1.0
        g = (w - lo) / span
        if gene_min != 0:
            g = 2.0 * g - 1.0
            g = np.clip(g, gene_min, -gene_min)
        else:
            g = np.clip(g, 0.0, 1.0)
        params[key] = w.tolist()
        gene.append(g.astype(np.float32))
    return params, np.concatenate(gene).tolist()


def _apply_inheritance(cfg, outP, w_final, model, bounds, gene_min):
    """The single swappable recombination hook -- design (A) vs (B).

    (A) darwinian: return fitness only. Para-HADES recombines the gene it
        issued, so STDP-learned changes die with the offspring (Baldwin
        effect: learning shapes selection, but acquired traits are not
        inherited).
    (B) lamarckian: overwrite `gene`/`param` in the record with the post-STDP
        weights. GA.py copies the whole result record into the population
        history, and the ES generators read `p['gene']` from it, so this is
        what makes acquired traits heritable.
    """
    mode = str(cfg['inheritance']).lower()
    if mode == 'darwinian':
        outP['inheritance'] = 'darwinian'
        return outP
    if mode != 'lamarckian':
        raise ValueError(f"inheritance must be 'darwinian' or 'lamarckian', "
                         f"got {cfg['inheritance']!r}")
    params, gene = weights_to_gene(w_final, model, bounds, gene_min)
    outP['inheritance'] = 'lamarckian'
    outP['param'] = params
    outP['gene'] = gene
    return outP


# =========================================================================== #
# Para-HADES entry point
# =========================================================================== #
_MODEL_CACHE = {}


def get_model(cfg, bounds, cache_dir):
    """Build the model once per process; every offspring shares the topology."""
    key = json.dumps({'cfg': {k: cfg[k] for k in sorted(cfg)},
                      'bounds': {k: bounds[k] for k in sorted(bounds)}},
                     sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()
    if digest not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[digest] = DotSNN(cfg, bounds, cache_dir)
    return _MODEL_CACHE[digest]


def run_task(param, args):
    global taskPath

    try:
        with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
            f.write('.')
    except Exception:
        print(f'Problem creating agent {args.agent_idx} task {args.agent_test_num} '
              f'lock file: {taskPath}', flush=True)
        return {'fitnessScore': FAILURE_FITNESS_SCORE}

    print(f'---- Start Task ----- '
          f'{datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    outP = {'fitnessScore': FAILURE_FITNESS_SCORE}
    try:
        cfg = load_cfg(args.paramFile)
        cfg['gpu'] = bool(getattr(args, 'gpu', False))
        bounds = load_param_bounds(args.paramFile)

        model = get_model(cfg, bounds, cache_dir=taskPath)
        w_start = gene_to_weights(param, model, bounds)

        if cfg['verbose']:
            print(f'[DotSNN] GC={model.n_gc} AC={cfg["ac"]} MC={model.n_mc} '
                  f'actions={model.n_actions} plastic={cfg["plastic"]} '
                  f'|w|={w_start.numel()} device={model.device}', flush=True)

        result, w_final = run_lifetime(
            model, cfg, w_start, out_path=taskPath,
            iteration=int(getattr(args, 'epoch', 0)),
            offspring=int(getattr(args, 'agent_test_num', 0)))

        outP.update(result)
        outP = _apply_inheritance(cfg, outP, w_final, model, bounds,
                                  float(getattr(args, 'geneMin', -1.0)))

        # evolutionTarget -1 means GA.py minimizes; our metrics are all
        # better-when-larger, so flip the sign it optimizes.
        if int(getattr(args, 'evolutionTarget', 1)) == -1:
            outP['fitnessScore'] = float(-1.0 * outP['fitness_mean_all'])
        else:
            outP['fitnessScore'] = float(outP['fitness_mean_all'])

        ratio = outP.get('stdp_vs_evol_ratio')
        print(f'[DotSNN] fitness(i)={outP["fitness_mean_all"]:.3f} '
              f'(ii)={outP["fitness_last_k"]:.3f} '
              f'(iii)={outP["fitness_frozen"]} | '
              f'STDP |dw|={outP["stdp_dw_l2"]:.4g} vs EVOL '
              f'|dw|={outP["evol_perturbation_l2"]:.4g} '
              f'ratio={ratio if ratio is None else round(ratio, 4)}', flush=True)
        if ratio is not None and ratio < 0.01:
            print('[DotSNN] WARNING: within-lifetime STDP displacement is <1% of '
                  'the EVOL perturbation. The hybrid has effectively degenerated '
                  'to plain EVOL -- raise `episodes` or `nu`.', flush=True)

    except ReceptionTargetError as exc:
        # A silent-failure guard tripping is a bug, not a bad offspring.
        print(f'FATAL: {exc}', flush=True)
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'ERROR evaluating gene: {exc}', flush=True)
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE

    # ---------------------------------------------------------------------- #
    # Make sure the `fitnessScore` is in return dict and the fitness is a
    # float number!
    # ---------------------------------------------------------------------- #
    print(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}',
          flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    return outP


def main(args):
    global taskPath
    global params_config

    taskPath = args.path + '/running/' + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)

    try:
        with open(args.path + '/running/' + str(args.agent_idx)
                  + '.newMember.json', 'rt') as f:
            params_config = json.load(f)
        format = 'json'
    except Exception:
        with zstandard.open(args.path + '/running/' + str(args.agent_idx)
                            + '.newMember.pkl', 'rb') as f:
            params_config = pickle_loads_compat(f.read())
        format = 'pkl'

    for gene_num in range(len(params_config)):
        fitness = []
        outP = {'fitnessScore': FAILURE_FITNESS_SCORE}
        for test in range(args.total_test_with_same_param):
            outP = run_task(params_config[gene_num], args)
            if outP['fitnessScore'] is not None:
                fitness.append(outP['fitnessScore'])

        if len(fitness) > 0:
            outP['fitnessScore'] = float(np.mean(fitness))

        if outP is not None:
            for k, v in outP.items():
                params_config[gene_num][k] = v

    if format == 'json':
        with open(taskPath + '/' + str(args.agent_test_num) + '.resoult.json', 'wt') as f:
            json.dump(params_config, f)
    else:
        with zstandard.open(taskPath + '/' + str(args.agent_test_num)
                            + '.resoult.pkl', 'wb') as f:
            f.write(pickle.dumps(params_config))

    if args.parallel or args.slurm:
        run(args, self=False, params=params_config[-1].copy())

    os.system('rm ' + args.path + '/running/' + str(args.agent_idx)
              + '.newMember.' + format)

    print(f'Done Task: {args.agent_test_num} for agent: {args.agent_idx}', flush=True)
    return


def run(args, self=False, params=None):
    args.agent_counter = str(args.agent_counter) + '.' + str(args.epoch)

    common = [
        '--path', args.path,
        '--agent_idx', str(args.agent_idx),
        '--agent_test_num', str(args.agent_test_num),
        '--agent_counter', str(args.agent_counter),
        '--total_test_with_same_param', str(args.total_test_with_same_param),
        '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
        '--gpu', 'True' if args.gpu else 'False',
        '--slurm', 'True' if args.slurm else 'False',
        '--parallel', 'True' if args.parallel else 'False',
        '--evolutionTarget', str(args.evolutionTarget),
        '--paramFile', args.paramFile,
    ]
    if self:
        run_process = [
            __file__,
            '--callBackScript',
            '"' + args.callBackScript + '"' if args.parallel
            else args.callBackScript.replace('"', ''),
        ] + common
    else:
        run_process = [
            '"' + args.callBackScript + '"' if args.parallel
            else args.callBackScript.replace('"', ''),
            '--part', '2',
            '--taskFilename', '"' + __file__ + '"',
        ] + common

    if args.slurm:
        os.makedirs(args.path + '/scripts/', exist_ok=True)
        temp_file = (args.path + '/scripts/Agent_idx_' + str(args.agent_idx)
                     + '_Task_' + str(args.agent_test_num))
        temp_file += '_Epoch_' + str(args.epoch) + '.sh'

        job_name = args.path.split('/')[-1]
        if job_name == '':
            job_name = args.path.split('/')[-2]
        res = params['main']['Task_resource'] if self else params['main']['Main_resource']
        script = SlurmScript(
            jobName=('T_' if self else 'A_') + job_name,
            jobTime=res['time'],
            jobMemory=res['memory'],
            output_path=args.path + res['path'],
            jobCPUs=res['cpu'],
            jobGPUs=res['gpu'],
            excludeNodes=res['exclude_nodes'],
            condaEnv=res['conda_env'],
            partition=res['partition'],
            singularity_image=res.get(
                'singularity_image',
                '/cluster/tufts/levinlab/hhazan01/singularity/delayW.sif'),
            command='python ' + str(run_process).replace('[', '').replace(']', '')
                    .replace('\', \'', ' ').replace('\'', ''),
            nextTask=temp_file,
        )
        with open(temp_file, 'w') as file:
            file.write(script)
            file.flush()

    elif args.parallel:
        run_process = (str(['python3'] + run_process).replace('[', '').replace(']', '')
                       .replace('\'', '').replace(',', ' '))
        subprocess.Popen(run_process, shell=True)


# must be after run function
def signal_handler_SIGTERM(sig, frame):
    global args
    global taskPath
    global params_config
    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    run(args=argparse._copy_items(args), self=True, params=params_config.copy())
    print('got a SIGTERM!!', flush=True)
    print('Done signal_handler', flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
signal.signal(signal.SIGINT, signal_handler_SIGTERM)


# =========================================================================== #
# Standalone modes
# =========================================================================== #
def mode_gene_lengths(args):
    cfg = load_cfg(args.paramFile)
    lengths = plastic_lengths(cfg)
    n_gc = len(cfg['gc_scales']) * cfg['gc_rotations'] * cfg['gc_offsets']
    n_mc = (9 if cfg['diag'] else 5) * cfg['mc_pop']
    print(f'\nConfig: GC={n_gc}  AC={cfg["ac"]}  MC={n_mc} '
          f'({9 if cfg["diag"] else 5} actions x {cfg["mc_pop"]})')
    print(f'plastic pathways: {cfg["plastic"]}\n')
    print('Put these in the YAML `param:` block:')
    for key, n in lengths.items():
        print(f'  {key}:  array: [{n}]')
    print(f'\nTotal gene length: {sum(lengths.values())}')
    if sum(lengths.values()) > 300000:
        print('NOTE: a gene this large with P=10 offspring is severely '
              'underdetermined. Consider lowering `ac` / `mc_pop`.')
    return 0


def measure_rates(model, cfg, out_path, steps, seed=7):
    """Mean AC and MC firing rates (Hz) over `steps` decision steps, no learning.

    Actions are drawn RANDOMLY rather than by WTA. That is deliberate: with a
    WTA policy the trajectory depends on the MC weights, so the input statistics
    move as the weights move and the measured rate stops being a monotone
    function of the weight -- which silently breaks the bisection in
    --calibrate. A fixed random policy decouples the measurement from the
    weights being measured.

    Weights are left exactly as the caller set them.
    """
    import torch

    env = make_env(cfg, out_path)
    env.seed = int(seed)
    env.reset()
    model.net.reset_state_variables()
    rng = np.random.default_rng(seed)

    action = int(rng.integers(0, model.n_actions))
    ac_tot = mc_tot = 0.0
    done = False
    n = 0
    seconds_per_step = model.gran * cfg['dt'] / 1000.0

    while n < steps:
        if done:
            env.seed = int(rng.integers(1, 2 ** 31 - 1))
            env.reset()
            model.net.reset_state_variables()
            done = False
        n += 1
        obs, _, done, _ = env.step(action)
        net_obs = obs.copy()
        net_obs[env.netDot.row, env.netDot.col] = 0.0
        view = torch.as_tensor(
            getLocalView(net_obs, env.netDot.row[0], env.netDot.col[0],
                         cfg['view_r']),
            dtype=torch.float32, device=model.device)
        model.net.run(inputs=model.encode(view), time=model.gran, reward=0.0)
        ac_tot += float(model.monitors[LAYER_AC].get('s').sum())
        mc_tot += float(model.monitors[LAYER_MC].get('s').sum())
        action = int(rng.integers(0, model.n_actions))

    elapsed = n * seconds_per_step
    return (ac_tot / (cfg['ac'] * elapsed), mc_tot / (model.n_mc * elapsed))


def mode_calibrate(args):
    """Pick the weight boxes that reproduce the paper's target firing rates.

    The paper's Table 2 gives synaptic magnitudes in NEURON conductance units,
    which are meaningless for BindsNET's LIF -- carrying the numbers 10.0 and
    6.5 across saturated the motor pool at ~140 Hz against a 6 Hz target, which
    makes the WTA readout degenerate.

    What DOES transfer is the paper's structure (convergence) and its dynamics
    (the homeostatic set points, 5.5 Hz for EA/AC and 6.0 Hz for EM/MC). So
    convergence is fixed from Table 2 and the magnitude scale is solved for.

    The two pathways are calibrated in sequence, which is exact rather than
    approximate: there is no MC -> AC feedback, so AC's rate does not depend on
    the AC -> MC weights at all.
    """
    import torch

    cfg = load_cfg(args.paramFile)
    cfg['steps'] = args.calibrate_steps
    cfg['verbose'] = False
    out_path = args.selftest_out
    os.makedirs(out_path, exist_ok=True)

    # Start from a wide box; only the midpoint matters for w_0.
    bounds = {gene_key('gc_ac'): (0.0, 2.0), gene_key('ac_mc'): (0.0, 2.0)}
    print('\n=== building model ===', flush=True)
    model = DotSNN(cfg, bounds, cache_dir=out_path)
    print(f'GC={model.n_gc} AC={cfg["ac"]} MC={model.n_mc}  '
          f'convergence={model.convergence}', flush=True)

    gc_ac = model.all_params['gc_ac']
    ac_mc = model.all_params['ac_mc']

    def set_scale(param, value):
        with torch.no_grad():
            param.tensor().copy_(param.mask * value)

    def bisect(param, other_fixed, which, target, lo, hi):
        """Monotone bisection on the uniform weight value."""
        set_scale(ac_mc, other_fixed) if which == 'AC' else None
        for _ in range(args.calibrate_iters):
            mid = 0.5 * (lo + hi)
            set_scale(param, mid)
            ac, mc = measure_rates(model, cfg, out_path, args.calibrate_steps)
            rate = ac if which == 'AC' else mc
            print(f'  {which}: w={mid:8.4f} -> {rate:7.2f} Hz '
                  f'(target {target})', flush=True)
            if rate > target:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    print(f'\n=== calibrating GC->AC for AC = {cfg["rate_target_ac"]} Hz ===',
          flush=True)
    w_ac = bisect(gc_ac, 0.0, 'AC', cfg['rate_target_ac'],
                  args.calibrate_lo, args.calibrate_hi)
    set_scale(gc_ac, w_ac)

    print(f'\n=== calibrating AC->MC for MC = {cfg["rate_target_mc"]} Hz ===',
          flush=True)
    w_mc = bisect(ac_mc, w_ac, 'MC', cfg['rate_target_mc'],
                  args.calibrate_lo, args.calibrate_hi)
    set_scale(ac_mc, w_mc)

    ac, mc = measure_rates(model, cfg, out_path, args.calibrate_steps * 2)
    print(f'\n=== final check ===')
    print(f'  AC {ac:.2f} Hz (target {cfg["rate_target_ac"]})')
    print(f'  MC {mc:.2f} Hz (target {cfg["rate_target_mc"]})')

    print('\nPut these in the YAML `param:` block. max = 2 x the calibrated')
    print('value so the box MIDPOINT is the calibrated weight -- that is what')
    print('centres generation 0 and defines w_0.\n')
    print(f'  w_gc_ac:   min: 0.0   max: {2 * w_ac:.4f}')
    print(f'  w_ac_mc:   min: 0.0   max: {2 * w_mc:.4f}')
    return 0


def mode_selftest(args):
    """Build, perturb, live, and report the STDP-contribution diagnostic."""
    import torch

    cfg = load_cfg(args.paramFile)
    cfg['episodes'] = args.selftest_episodes
    cfg['steps'] = args.selftest_steps
    cfg['frozen_episodes'] = min(1, cfg['frozen_episodes'])
    cfg['verbose'] = True
    bounds = load_param_bounds(args.paramFile)

    out_path = args.selftest_out
    os.makedirs(out_path, exist_ok=True)

    print('\n=== building model ===', flush=True)
    t0 = time.time()
    model = DotSNN(cfg, bounds, cache_dir=out_path)
    print(f'built in {time.time() - t0:.1f}s  GC={model.n_gc} AC={cfg["ac"]} '
          f'MC={model.n_mc} plastic-dim={model.w0.numel()} '
          f'device={model.device}', flush=True)
    for key, n in model.gene_lengths.items():
        print(f'  {key}: {n}')

    # Synthesize one offspring the way the paper's EVOL would: multiplicative
    # perturbation of the current mean, w * (1 + sigma * e).
    sigma = args.selftest_sigma
    g = torch.Generator(device='cpu').manual_seed(1234)
    e = torch.randn(model.w0.numel(), generator=g).to(model.w0.device)
    w_start = (model.w0 * (1.0 + sigma * e))
    off = 0
    for p in model.registry:
        n = p.idx.numel()
        w_start[off:off + n] = w_start[off:off + n].clamp(p.wmin, p.wmax)
        off += n

    print(f'\n=== lifetime: {cfg["episodes"]} episodes x {cfg["steps"]} steps '
          f'===', flush=True)
    t0 = time.time()
    result, w_final = run_lifetime(model, cfg, w_start, out_path=out_path)
    dur = time.time() - t0

    print(f'\n=== results ({dur:.1f}s) ===')
    for k in ('fitness_mean_all', 'fitness_last_k', 'fitness_frozen',
              'median_all', 'mean_reward', 'mean_steps'):
        print(f'  {k:22s} {result[k]}')
    print(f'  per_episode            {result["per_episode"]}')
    print(f'  homeostatic_gain       {result["homeostatic_gain"]}')
    print(f'  firing_rates_hz        {result["firing_rates_hz"]}')

    print('\n=== STDP contribution diagnostic (spec: MANDATORY) ===')
    dw = result['stdp_dw_l2']
    pert = result['evol_perturbation_l2']
    ratio = result['stdp_vs_evol_ratio']
    print(f'  ||w_start||                     {result["w_start_l2"]:.4f}')
    print(f'  ||w_start - w_0||  (EVOL)       {pert:.4f}')
    print(f'  ||w_final - w_start|| (STDP)    {dw:.4f}')
    print(f'  STDP / EVOL                     '
          f'{ratio if ratio is None else round(ratio, 5)}')
    if ratio is None:
        print('  -> no perturbation applied; ratio undefined.')
    elif ratio < 0.01:
        print('  -> STDP is NOT contributing. The hybrid reduces to plain EVOL '
              'and the result will be a null.\n'
              '     Raise `episodes` (20-50), raise `nu`, or both.')
    else:
        print('  -> STDP is moving the weights a measurable amount relative to '
              'the EVOL perturbation.')

    # Sanity: the reception-balancing guard must actually fire when misused.
    print('\n=== reception-target guard check ===')
    try:
        bad = WeightNormalizer(cfg, model.registry,
                              n_neurons={LAYER_AC: cfg['ac'], LAYER_MC: model.n_mc})
        bad.recept_target = {k: v.clone() for k, v in model.w0_col_sums.items()}
        bad.assert_targets_track_offspring(model.w0_col_sums, w_start, model.w0)
        print('  FAIL: guard did not fire on w_0-derived targets.')
        return 1
    except ReceptionTargetError:
        print('  OK: guard fires when targets are derived from w_0.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='population')
    parser.add_argument('--agent_test_num', type=int, default=-1)
    parser.add_argument('--agent_idx', type=int, default=-1)
    parser.add_argument('--agent_counter', type=str, default='-1.0')
    parser.add_argument('--total_test_with_same_param', type=int, default=1)
    parser.add_argument('--total_testsGene_with_same_agent', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='False')
    parser.add_argument('--slurm', type=str, default='False')
    parser.add_argument('--parallel', type=str, default='True')
    parser.add_argument('--callBackScript', type=str, default=None)
    parser.add_argument('--evolutionTarget', type=int, default=1)   # 1 = Max, -1 = Min
    parser.add_argument('--paramFile', type=str, default=None)
    parser.add_argument('--geneMin', type=float, default=-1.0)      # must match GA.py
    # standalone modes
    parser.add_argument('--gene_lengths', action='store_true',
                        help='print the YAML array lengths for the current config')
    parser.add_argument('--selftest', action='store_true',
                        help='run one short offspring lifetime + STDP diagnostic')
    parser.add_argument('--selftest_episodes', type=int, default=3)
    parser.add_argument('--selftest_steps', type=int, default=30)
    parser.add_argument('--selftest_sigma', type=float, default=0.1)
    parser.add_argument('--selftest_out', type=str, default='dotSNN_selftest')
    parser.add_argument('--calibrate', action='store_true',
                        help='solve for the weight boxes that hit the paper\'s '
                             'AC/MC firing-rate targets')
    parser.add_argument('--calibrate_steps', type=int, default=60)
    parser.add_argument('--calibrate_iters', type=int, default=9)
    parser.add_argument('--calibrate_lo', type=float, default=0.0)
    parser.add_argument('--calibrate_hi', type=float, default=4.0)

    args = parser.parse_args()

    args.gpu = True if args.gpu in ('True', 'true') else False
    args.slurm = True if args.slurm in ('True', 'true') else False
    args.parallel = True if args.parallel in ('True', 'true') else False

    if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in \
            ['', 'none', 'null']:
        args.paramFile = None
    if args.paramFile is None:
        task_root, _ = os.path.splitext(os.path.abspath(__file__))
        default_yaml = task_root + '.yaml'
        default_yml = task_root + '.yml'
        args.paramFile = default_yaml
        if (not os.path.exists(default_yaml)) and os.path.exists(default_yml):
            args.paramFile = default_yml

    if args.gene_lengths:
        sys.exit(mode_gene_lengths(args))
    if args.calibrate:
        sys.exit(mode_calibrate(args))
    if args.selftest:
        sys.exit(mode_selftest(args))

    if args.callBackScript is not None:
        args.callBackScript = args.callBackScript.replace('"', '')

    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.slurm:
        args.parallel = False

    main(argparse._copy_items(args))

#!/usr/bin/env python3
"""compute_bounds.py -- weight statistics + ES bound B for a CoGameAgent net.pt

Stand-in for the CoGameAgent scorer's `--dump-weight-stats` helper (work order
§1/§4) for the case where only the trained net.pt is on hand. It computes the
flattened weight vector exactly as CoGameAgent's codec does (state_dict keys
sorted by NAME, each tensor flattened row-major, concatenated as float32 -- see
scripts/stage_b/contest_spatialmem.py:net_to_gene / gene_to_state_dict) and
reports:

    total_params, max_abs, std, and the recommended symmetric bound
        B = max(3.0, 1.5 * max_abs)

B must satisfy B >= max|weight| so that Para-HADES' paramTOgene (which maps
weights in [-B, B] -> gene in [-1, 1] as gene = weight / B) does not clip the
warm-start. Put the printed B into the task YAML's `param:` block (min = -B,
max = +B). This reads ONLY tensor values -- it never imports game code.

Usage:
    python3 compute_bounds.py <net.pt> [--factor 1.5] [--floor 3.0]
"""
import argparse
import sys

import numpy as np
import torch


def flat_gene_from_state_dict(sd):
    """Concatenate a state_dict into one float32 vector, keys sorted by name."""
    tensor_keys = sorted(k for k in sd.keys() if hasattr(sd[k], 'shape'))
    parts = [sd[k].detach().cpu().numpy().reshape(-1).astype(np.float32) for k in tensor_keys]
    return np.concatenate(parts), tensor_keys


def load_state_dict(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict'], obj.get('val')
    return obj, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('net_pt', help='path to a CoGameAgent net.pt')
    ap.add_argument('--factor', type=float, default=1.5, help='B = max(floor, factor * max_abs)')
    ap.add_argument('--floor', type=float, default=3.0, help='minimum bound B')
    args = ap.parse_args(argv)

    sd, val = load_state_dict(args.net_pt)
    gene, keys = flat_gene_from_state_dict(sd)

    max_abs = float(np.abs(gene).max())
    std = float(gene.std())
    B = max(args.floor, args.factor * max_abs)

    print(f'net.pt         : {args.net_pt}')
    if val is not None:
        print(f'checkpoint val : {val}')
    print(f'num tensors    : {len(keys)}')
    print(f'total_params   : {gene.size}')
    print(f'max_abs        : {max_abs:.6f}')
    print(f'std            : {std:.6f}')
    print(f'recommended B  : {B:.6f}   (= max({args.floor}, {args.factor} * {max_abs:.6f}))')
    print()
    print('YAML param bounds:')
    print(f'    min: {-B:.6f}')
    print(f'    max: {B:.6f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Read-only self-test for the GA history loader (Work Order Issue 1, section 7).

Runs ChunkedGeneHistoryLoader.loadAllGeneHistory against an existing experiment
directory exactly as GA.py part=1 does, and reports how many valid genes come
back plus the max epoch/iteration. It NEVER writes to the history directory
(the cache file it touches lives under <path>/running/, same as the real loader).

Run in the delayW env on the login node, e.g.:

    conda activate delayW
    python GA_history_loader_selftest.py \
        --path /cluster/tufts/levinlab/hhazan01/git_repo/Alternative-Tasks/07-11-Heb_CoGame/GA \
        --geneFormat pickle

Before the loader fix a broken framed history returned 0 SILENTLY; after the fix
the loader prints a LOUD "GA - history loader - WARNING: ..." line naming the
actual decode exception (e.g. UnpicklingError, zstd error, length mismatch), and
falls back to the .log.csv summary sidecar to recover the scalar fields.
"""
import argparse
import glob
import os
import types

from GA_utils_history_loader import ChunkedGeneHistoryLoader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path', required=True, help='experiment dir containing *.log.csv history')
    ap.add_argument('--geneFormat', default='pickle', help="ea.geneFormat (usually 'pickle' or 'json')")
    ap.add_argument('--epoch', type=int, default=0, help='args.epoch (only used when --currentEpochOnly)')
    ap.add_argument('--currentEpochOnly', action='store_true',
                    help='filter to --epoch instead of allEpochs=True (part=1 uses allEpochs=True)')
    args_ns = ap.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args_ns.path, '*.log.csv')))
    print(f"Found {len(csv_files)} *.log.csv file(s) under {args_ns.path}")

    loader = ChunkedGeneHistoryLoader(
        savePath=args_ns.path,
        agent_id=-1,
        geneFormat=args_ns.geneFormat,
        required_keys_and_types=['fitnessScore', 'epoch', 'iteration', 'loss'],
        shuffle=True,
    )

    loader_args = types.SimpleNamespace(epoch=args_ns.epoch, agent_counter=0)
    population = loader.loadAllGeneHistory(loader_args, allEpochs=not args_ns.currentEpochOnly)

    with_fitness = [g for g in population if g.get('fitnessScore') is not None]
    epochs = [g['epoch'] for g in population if isinstance(g.get('epoch'), (int, float))]
    iters = [g['iteration'] for g in population if isinstance(g.get('iteration'), (int, float))]

    print("-" * 60)
    print(f"Total records returned : {len(population)}")
    print(f"With non-None fitness  : {len(with_fitness)}")
    print(f"Max epoch              : {max(epochs) if epochs else 'n/a'}")
    print(f"Max iteration          : {max(iters) if iters else 'n/a'}")
    print("-" * 60)
    if csv_files and len(with_fitness) == 0:
        print("RESULT: history files exist but 0 valid genes loaded -> resume would "
              "fall back to random. See the WARNING line(s) above for the real cause.")
    elif len(with_fitness) > 0:
        print("RESULT: OK -- loader returns a non-empty population; part=1 will "
              "resume with source: evolution.")
    else:
        print("RESULT: no history files -> fresh run (source: random). Expected for an empty dir.")


if __name__ == '__main__':
    main()

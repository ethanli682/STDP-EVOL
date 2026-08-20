"""composite_v6 NS-ES post-fitness hook for GA.py.

Plugged into GA.py via the generic `post_fitness_hook` yaml block:

  # In task_cogames_*.yaml at the top level:
  post_fitness_hook:
    module:   composite_v6_novelty_hook
    function: apply

Adds a Lehman-Stanley NS-ES novelty bonus to fitnessScore using the
`behavior_vector` field that composite_v6 writes into each per-test
sidecar. Bonus magnitude is controlled by env vars (defaults to no-op):

  V6_NOVELTY_WEIGHT        scale factor; 0.0 = OFF (default).
                           Try 0.1-1.0; raise if BV distances are tiny
                           vs fitness magnitudes.
  V6_NOVELTY_K             number of nearest neighbours to average.
                           Default 15. ~10% of typical batch is healthy.
  V6_NOVELTY_ARCHIVE_SIZE  cap on the archive. Default 500. Random-
                           replacement insertion (Lehman & Stanley 2017).

The archive is THIS WORKER'S log of past behavior_vectors. Cross-worker
novelty would need a shared archive with file locking; the per-worker
archive is biased toward in-worker diversity but avoids races.

Even when V6_NOVELTY_WEIGHT=0, the hook still stashes
`v6_behavior_vector` on every record so that turning the weight on later
can immediately use historical logs to seed the archive.
"""

import json
import os
import random

import numpy as np


# Per-process archive cache, keyed by (log_path, evolutionTarget). Loaded
# lazily on the first call for each key; grown in-process across calls.
_ARCHIVE_CACHE = {}


def _config_from_env():
    return {
        'weight':       float(os.environ.get('V6_NOVELTY_WEIGHT', '0.0')),
        'k':            int(os.environ.get('V6_NOVELTY_K', '15')),
        'archive_size': int(os.environ.get('V6_NOVELTY_ARCHIVE_SIZE', '500')),
    }


def _extract_behavior_vector_mean(reportParam):
    """Return per-test-mean v6 behavior_vector, or None if no v6 sidecars."""
    cv6_list = reportParam.get('components_v6_per_test') or []
    bvs = []
    for cv6 in cv6_list:
        if not isinstance(cv6, dict):
            continue
        agg = cv6.get('aggregate') or {}
        bv = agg.get('behavior_vector')
        if bv:
            try:
                bvs.append([float(x) for x in bv])
            except (TypeError, ValueError):
                continue
    if not bvs:
        return None
    return np.asarray(bvs, dtype=float).mean(axis=0).tolist()


def _load_archive_from_log(log_path, max_size):
    """Read v6_behavior_vector entries from a JSONL log; return the last
    max_size of them. Tolerant of malformed lines and missing files.
    """
    if not log_path or not os.path.exists(log_path):
        return []
    bvs = []
    try:
        with open(log_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                bv = rec.get('v6_behavior_vector')
                if bv:
                    try:
                        bvs.append([float(x) for x in bv])
                    except (TypeError, ValueError):
                        continue
    except Exception:
        return bvs
    return bvs[-max_size:] if len(bvs) > max_size else bvs


def _novelty_vs_archive(new_bv, archive_bvs, k):
    """Mean Euclidean distance from new_bv to its k nearest neighbours in
    archive_bvs. Returns 0.0 if archive is empty, shape-mismatched, or
    contains ragged rows (defensive: changing the BV dimension between
    runs while reusing an old log would otherwise raise on every call).
    """
    if not archive_bvs:
        return 0.0
    try:
        arr = np.asarray(archive_bvs, dtype=float)
        new = np.asarray(new_bv, dtype=float)
    except (ValueError, TypeError):
        return 0.0
    if (arr.ndim != 2 or new.ndim != 1
            or arr.shape[0] == 0 or arr.shape[1] != new.shape[0]):
        return 0.0
    dists = np.sqrt(((arr - new) ** 2).sum(axis=1))
    k_eff = max(1, min(int(k), dists.shape[0]))
    return float(np.partition(dists, k_eff - 1)[:k_eff].mean())


def _archive_append(archive_bvs, new_bv, max_size):
    """Random-replacement append — Lehman & Stanley recommend random
    replacement over FIFO so older diverse exemplars aren't always
    evicted by recent additions.
    """
    if len(archive_bvs) < max_size:
        archive_bvs.append(list(new_bv))
    else:
        idx = random.randrange(max_size)
        archive_bvs[idx] = list(new_bv)


def apply(reportParam, *, agent_idx=None, log_path=None,
          evolutionTarget=1, params_config=None, **_ignored):
    """Post-fitness hook entry point (GA.py contract). Mutates reportParam
    in place.

    - Always stashes `v6_behavior_vector` (when available) so future runs
      can rebuild the archive even if novelty was off this run.
    - When V6_NOVELTY_WEIGHT > 0, also computes novelty vs the archive
      and adjusts `fitnessScore`. Sign follows `evolutionTarget`
      (+1 = maximise → add; -1 = minimise → subtract, so novel genomes
      always look better to the optimiser).
    """
    cfg = _config_from_env()
    bv = _extract_behavior_vector_mean(reportParam)
    if bv is None:
        return
    reportParam['v6_behavior_vector'] = bv

    if cfg['weight'] <= 0.0:
        return

    cache_key = (str(log_path), int(evolutionTarget))
    if cache_key not in _ARCHIVE_CACHE:
        _ARCHIVE_CACHE[cache_key] = _load_archive_from_log(
            log_path, cfg['archive_size'])
    archive = _ARCHIVE_CACHE[cache_key]

    raw_nov = _novelty_vs_archive(bv, archive, cfg['k'])
    sign = 1.0 if int(evolutionTarget) >= 0 else -1.0
    scaled = sign * cfg['weight'] * raw_nov
    reportParam['v6_novelty_raw']   = float(raw_nov)
    reportParam['v6_novelty_bonus'] = float(scaled)
    reportParam['fitnessScore'] = (
        float(reportParam['fitnessScore']) + scaled)
    _archive_append(archive, bv, cfg['archive_size'])

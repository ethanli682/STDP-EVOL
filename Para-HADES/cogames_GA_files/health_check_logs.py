#!/usr/bin/env python3
"""
Quick health-check for CMA-ES cogames runs.

Scans .log.csv files (JSON-lines, despite the .csv extension) and reports:
  - whether stagnation_boost ever fired (should be never after 2026-04-22 tuning)
  - max per-group sigma observed (should stay <= 1.5)
  - per-candidate fitness trajectory (min/mean/max per window)
  - generations where fitnessScore_std collapsed (population convergence)

Usage:
    python cogames_GA_files/health_check_logs.py <run_dir>
    python cogames_GA_files/health_check_logs.py <run_dir> --window 20
    python cogames_GA_files/health_check_logs.py file1.log.csv file2.log.csv ...

Exits non-zero if any regression flags fire.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

SIGMA_CAP = 1.5
STD_COLLAPSE_THRESHOLD = 1e-3  # per-generation fitness std below this = population collapse


def _iter_records(path):
    with open(path) as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [warn] {path}:{line_no}: malformed JSON: {e}", file=sys.stderr)


def _resolve_paths(inputs):
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "**", "*.log.csv"), recursive=True)))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            print(f"  [warn] not found: {inp}", file=sys.stderr)
    return paths


def analyze(path, window):
    agent_idx = None
    fitness = []           # (iteration, fitnessScore)
    fit_std = []           # (iteration, fitnessScore_std)
    ep_reward = []         # (iteration, ep_reward_mean)  — real task signal
    sigma_max_seen = 0.0
    sigma_group_max = {}   # group -> max sigma
    boost_iters = []       # iterations where stagnation_boost fired
    restart_iters = []

    for rec in _iter_records(path):
        if agent_idx is None:
            agent_idx = rec.get("candidate_agent_idx")
        it = rec.get("iteration", -1)
        if "fitnessScore" in rec:
            fitness.append((it, rec["fitnessScore"]))
        if "fitnessScore_std" in rec:
            fit_std.append((it, rec["fitnessScore_std"]))
        if rec.get("ep_reward_mean") is not None:
            ep_reward.append((it, float(rec["ep_reward_mean"])))

        cma_log = (rec.get("build_in_param", {}) or {}).get("CMA_ES-log", {}) or {}
        if cma_log.get("stagnation_boost"):
            boost_iters.append(it)
        if cma_log.get("restart"):
            restart_iters.append(it)
        sigma = cma_log.get("sigma")
        if isinstance(sigma, (int, float)):
            sigma_max_seen = max(sigma_max_seen, float(sigma))
        for g, s in (cma_log.get("sigma_per_group", {}) or {}).items():
            try:
                s = float(s)
            except Exception:
                continue
            if s > sigma_group_max.get(g, 0.0):
                sigma_group_max[g] = s

    # windowed fitness summary
    summaries = []
    if fitness:
        last_it = fitness[-1][0]
        start = max(0, last_it - window + 1)
        windowed = [f for it, f in fitness if it >= start]
        if windowed:
            summaries.append(
                f"last {len(windowed)} iters (it≥{start}): "
                f"min={min(windowed):.3f} mean={sum(windowed)/len(windowed):.3f} max={max(windowed):.3f}"
            )
        all_f = [f for _, f in fitness]
        summaries.append(
            f"overall: n={len(all_f)} min={min(all_f):.3f} mean={sum(all_f)/len(all_f):.3f} max={max(all_f):.3f}"
        )

    # task-reward (ep_reward_mean) summary — the real signal
    if ep_reward:
        last_it = ep_reward[-1][0]
        start = max(0, last_it - window + 1)
        w_er = [v for it, v in ep_reward if it >= start]
        all_er = [v for _, v in ep_reward]
        summaries.append(
            f"ep_reward (real task signal): "
            f"last-{len(w_er)} mean={sum(w_er)/max(len(w_er),1):.3f} max={max(w_er) if w_er else 0:.3f}  |  "
            f"overall n={len(all_er)} mean={sum(all_er)/len(all_er):.3f} max={max(all_er):.3f}"
        )
    else:
        summaries.append("ep_reward (real task signal): not logged (pre-2026-04-23 run, or fitness_stat != composite_exploration_v2)")

    # population-collapse generations
    collapsed = [it for it, s in fit_std if s is not None and s < STD_COLLAPSE_THRESHOLD]

    return {
        "agent_idx": agent_idx,
        "summaries": summaries,
        "sigma_max": sigma_max_seen,
        "sigma_group_max": sigma_group_max,
        "boost_iters": boost_iters,
        "restart_iters": restart_iters,
        "collapsed_iters": collapsed,
    }


def report(path, res):
    flags = []
    notes = []
    # HARD regressions: indicate the run is actually broken, not just converged.
    if res["sigma_max"] > SIGMA_CAP + 1e-6:
        flags.append(f"sigma reached {res['sigma_max']:.3f} (cap {SIGMA_CAP})")
    bad_groups = {g: s for g, s in res["sigma_group_max"].items() if s > SIGMA_CAP + 1e-6}
    if bad_groups:
        flags.append("groups over cap: " + ", ".join(f"{g}={s:.2f}" for g, s in bad_groups.items()))
    if len(res["collapsed_iters"]) > 0:
        flags.append(f"population std collapsed in {len(res['collapsed_iters'])} iters "
                     f"(e.g. {res['collapsed_iters'][:5]})")
    # SOFT notes: observed but not necessarily a failure.
    # stagnation_boost alone is NOT a regression — it fires on any plateau,
    # including the healthy kind where the run has legitimately converged.
    # Only flag it as a problem if it coincides with population collapse.
    if res["boost_iters"]:
        if res["collapsed_iters"]:
            flags.append(f"stagnation_boost fired {len(res['boost_iters'])}x "
                         f"alongside {len(res['collapsed_iters'])} collapsed iters "
                         f"(symptom of stuck population, not just a plateau)")
        else:
            notes.append(f"stagnation_boost fired {len(res['boost_iters'])}x "
                         f"(expected once fitness plateaus; no population collapse, so healthy)")

    status = "OK" if not flags else "REGRESSION"
    print(f"\n[{status}] {path}   agent_idx={res['agent_idx']}")
    for s in res["summaries"]:
        print(f"   {s}")
    print(f"   sigma_max={res['sigma_max']:.3f}  groups_max={ {k: round(v,2) for k,v in res['sigma_group_max'].items()} }")
    print(f"   restarts={len(res['restart_iters'])}  boosts={len(res['boost_iters'])}  "
          f"std-collapsed-iters={len(res['collapsed_iters'])}")
    for f in flags:
        print(f"   ! {f}")
    for n in notes:
        print(f"   . {n}")
    return not flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="Directory to walk, or one or more *.log.csv files")
    ap.add_argument("--window", type=int, default=20,
                    help="Trailing-iteration window for summary (default 20)")
    args = ap.parse_args()

    paths = _resolve_paths(args.inputs)
    if not paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(2)

    ok = True
    for p in paths:
        res = analyze(p, args.window)
        if not report(p, res):
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

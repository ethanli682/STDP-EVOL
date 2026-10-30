"""Dot-tracing parameter screening, controlled sweeps, and history plots.

Run --help or a subcommand's --help. Imports the simulator only for new evaluations.
Historical stdout is never assigned invented generation numbers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
import yaml


def read_jsonl(path):
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            warnings.warn(f"{path}:{i}: skipped malformed/incomplete record: {exc}")
    return records


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fitness(record):
    try:
        a = np.asarray(record["fitnessScore"], dtype=float).reshape(-1)
        return float(a.mean()) if len(a) and np.isfinite(a).all() else None
    except (KeyError, TypeError, ValueError):
        return None


def features(record):
    result = {}
    for name, value in record.get("param", {}).items():
        a = np.asarray(value, dtype=float).reshape(-1)
        for i, v in enumerate(a):
            result[name if len(a) == 1 else f"{name}[{i}]"] = float(v)
    return result


def set_feature(param, name, value):
    match = re.fullmatch(r"(.+)\[(\d+)\]", name)
    key, idx = (match[1], int(match[2])) if match else (name, 0)
    param[key][idx] = float(value)


def bounds_from_yaml(path):
    specs = yaml.safe_load(Path(path).read_text())["param"]
    bounds, baseline = {}, {}
    for key, spec in specs.items():
        count = int(np.prod(spec.get("array", [1])))
        lo, hi = float(spec["min"]), float(spec["max"])
        baseline[key] = [(lo + hi) / 2] * count
        for i in range(count):
            bounds[key if count == 1 else f"{key}[{i}]"] = (lo, hi)
    return bounds, baseline


def parse_stdout(path):
    """Recover rounded genes, test episodes and returned fitness from worker logs."""
    records, current, phase = [], None, None
    scalar_names = {"gc_pc": ("scaler_gc_pc1", "scaler_gc_pc2"),
                    "pc_ac": ("scaler_pc_ac1", "scaler_pc_ac2"),
                    "ac_mc": ("scaler_ac_mc",), "inhib_ac": ("scaler_inhib_ac",),
                    "inhib_mc": ("scaler_inhib_mc",)}
    sparse_names = {"gc_pc": ("sparsity_gc_pc_a", "sparsity_gc_pc_t"),
                    "pc_ac": ("sparsity_pc_ac_a", "sparsity_pc_ac_t"),
                    "rec": ("sparsity_rec",), "ac_mc": ("sparsity_ac_mc",)}
    for line in Path(path).read_text().splitlines():
        if "---- Start Task" in line:
            current = {"param": {}, "test_episodes": [], "source": str(path)}
            phase = None
        if current is None:
            continue
        if line.startswith("[gene]"):
            mapping = sparse_names if "sparsity" in line else scalar_names
            for label, names in mapping.items():
                m = re.search(r"\b" + label + r"=([-+\d.eE/]+)", line)
                if m:
                    for name, v in zip(names, m[1].split("/")):
                        current["param"][name] = [float(v)]
            m = re.search(r"mc_opp=(\[[^\]]+\])", line)
            if m:
                current["param"]["sparsity_mc_opp"] = json.loads(m[1])
        if line.strip() in ("Training:", "Testing:"):
            phase = line.strip()
        if phase == "Testing:" and line.startswith("Episode"):
            m = re.search(r"AC active ([\d.]+)%", line)
            if m:
                v = float(m[1]) / 100
                current["test_episodes"].append({"ac_active_fraction": v,
                                                "ac_in_band": .25 <= v <= .65})
        if line.startswith("RETURNED FITNESS:"):
            current["fitnessScore"] = float(line.split(":", 1)[1])
            current["evaluation"] = len(records)
            eps = current["test_episodes"]
            current["ac_active_fraction"] = float(np.mean([e["ac_active_fraction"] for e in eps])) if eps else None
            current["ac_all_episodes_in_band"] = all(e["ac_in_band"] for e in eps) if eps else None
            records.append(current)
            current = None
    return records


def load_records(path):
    path = Path(path)
    if path.suffix == ".out":
        return parse_stdout(path)
    if path.is_dir():
        files = sorted(path.glob("*.log.csv"))
        if not files:
            raise ValueError(f"No *.log.csv generation histories in {path}")
        return [r for f in files for r in load_records(f)]
    # Framed files contain genes omitted from their lightweight JSONL sidecars.
    if path.name.endswith(".log.csv"):
        from GA_utils_framed_records import FramedRecordReader
        if FramedRecordReader.exists_for_csv(str(path)):
            with FramedRecordReader.from_csv_path(str(path)) as reader:
                return list(reader.iter_records())
    return read_jsonl(path)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, out, name):
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    pyplot().close(fig)


def rank_parameters(records, out, metric="fitnessScore"):
    """Exploratory nonlinear sensitivity: repeated held-out permutation importance.

    This estimates predictive importance under the sampled distribution, not causality.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import ShuffleSplit
    records = [r for r in records if fitness({"fitnessScore": r.get(metric)}) is not None and features(r)]
    if len(records) < 20:
        raise ValueError("Need at least 20 complete evaluations for exploratory parameter ranking; specify --axes instead.")
    names = sorted(set.intersection(*(set(features(r)) for r in records)))
    X = np.array([[features(r)[n] for n in names] for r in records])
    keep = np.isfinite(X).all(0) & (np.ptp(X, axis=0) > 0)
    names, X = [n for n, k in zip(names, keep) if k], X[:, keep]
    if X.shape[1] < 2:
        raise ValueError("Need at least two varying parameters")
    y = np.array([fitness({"fitnessScore": r[metric]}) for r in records])
    if np.ptp(y) == 0:
        raise ValueError(f"{metric} is constant: no parameter importance can be inferred; specify --axes or sample more widely.")
    importances, scores = [], []
    for split, (train, test) in enumerate(ShuffleSplit(n_splits=5, test_size=.25, random_state=42).split(X)):
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=split, n_jobs=1)
        model.fit(X[train], y[train])
        scores.append(float(model.score(X[test], y[test])))
        result = permutation_importance(model, X[test], y[test], n_repeats=10,
                                        random_state=split, scoring="neg_mean_squared_error", n_jobs=1)
        importances.append(result.importances_mean)
    values = np.array(importances)
    order = np.argsort(-values.mean(0))
    rows = [{"parameter": names[i], "importance_delta_mse": float(values[:, i].mean()),
             "split_std": float(values[:, i].std())} for i in order]
    write_csv(out / "parameter_importance.csv", rows)
    write_json(out / "ranking_validation.json", {"metric": metric, "evaluations": len(records), "held_out_r2": scores,
               "mean_held_out_r2": float(np.mean(scores)),
               "interpretation": "Exploratory predictive ranking, not causal. Weak validation or overlapping uncertainties makes the top-two choice tentative."})
    if np.mean(scores) < .3:
        warnings.warn(f"Weak held-out R² ({np.mean(scores):.2f}); selected axes are tentative, not established as most important.")
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    show = rows[:12][::-1]
    ax.barh([r["parameter"] for r in show], [r["importance_delta_mse"] for r in show],
            xerr=[r["split_std"] for r in show], color="#487b9e")
    ax.set(xlabel="Held-out MSE increase after shuffling (mean ± split SD)",
           title=f"Exploratory {metric} sensitivity · n={len(records)} · mean held-out R²={np.mean(scores):.2f}")
    fig.tight_layout()
    save_figure(fig, out, "parameter_importance")
    return [r["parameter"] for r in rows[:2]]


def landscape(records, axes, out, bins=6, controlled=False, title=""):
    rows = [r for r in records if fitness(r) is not None and all(n in features(r) for n in axes)]
    if not rows:
        raise ValueError("No valid fitness records with both selected parameters")
    x, y = (np.array([features(r)[n] for r in rows]) for n in axes)
    f = np.array([fitness(r) for r in rows])
    active = np.array([r.get("ac_active_fraction") if r.get("ac_active_fraction") is not None else np.nan for r in rows])
    healthy = np.array([float(r["ac_all_episodes_in_band"]) if r.get("ac_all_episodes_in_band") is not None else np.nan for r in rows])
    if controlled:
        def edges(v):
            u = np.unique(v)
            if len(u) == 1:
                return np.array([u[0] - .5, u[0] + .5])
            return np.r_[u[0] - (u[1] - u[0]) / 2, (u[:-1] + u[1:]) / 2, u[-1] + (u[-1] - u[-2]) / 2]
        xe, ye = edges(x), edges(y)
    else:
        xe, ye = np.linspace(x.min(), x.max(), bins + 1), np.linspace(y.min(), y.max(), bins + 1)
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError("Both heatmap axes must vary")
    count = np.histogram2d(y, x, bins=(ye, xe))[0]
    plt = pyplot()
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    for ax, value, label, cmap, limits in zip(axs, [f, active, healthy, np.ones(len(rows))],
            ["Mean returned fitness ↑", "Mean AC active fraction", "Fraction passing every test episode", "Evaluations per cell"],
            ["viridis", "magma", "YlGn", "Blues"], [{}, {"vmin": 0, "vmax": 1}, {"vmin": 0, "vmax": 1}, {}]):
        valid = np.isfinite(value)
        n = np.histogram2d(y[valid], x[valid], bins=(ye, xe))[0]
        total = np.histogram2d(y[valid], x[valid], bins=(ye, xe), weights=value[valid])[0]
        z = np.divide(total, n, out=np.full_like(total, np.nan), where=n > 0)
        if label == "Evaluations per cell":
            z = np.where(count > 0, count, np.nan)
            limits = {"vmin": 0, "vmax": max(1, count.max())}
        im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(z), cmap=cmap, shading="flat", **limits)
        ax.set_facecolor("#e9e9e9")
        ax.set(xlabel=axes[0], ylabel=axes[1], title=label)
        if controlled:
            if len(np.unique(x)) <= 15:
                ax.set_xticks(np.unique(x))
            if len(np.unique(y)) <= 15:
                ax.set_yticks(np.unique(y))
        colorbar = fig.colorbar(im, ax=ax, shrink=.8)
        if label == "Evaluations per cell":
            from matplotlib.ticker import MaxNLocator
            colorbar.locator = MaxNLocator(integer=True)
            colorbar.update_ticks()
    kind = "Controlled slice · other genes fixed" if controlled else "Observed samples · other genes vary · not a controlled slice"
    fig.suptitle(f"{title}\n{kind} · n={len(rows)} · gray = no data")
    save_figure(fig, out, "fitness_landscape")
    write_csv(out / "landscape_samples.csv", [{axes[0]: float(a), axes[1]: float(b), "fitness": float(c),
               "ac_active_fraction": float(d) if np.isfinite(d) else "",
               "all_test_episodes_in_band": bool(e) if np.isfinite(e) else ""}
               for a, b, c, d, e in zip(x, y, f, active, healthy)])


def generation_summary(runs):
    rows = []
    for label, records in runs.items():
        groups = defaultdict(list)
        for r in records:
            score = fitness(r)
            if score is None:
                continue
            if "iteration" not in r:
                raise ValueError(f"{label}: missing iteration metadata. Worker stdout cannot establish generations; supply *.log.csv histories.")
            coordinates = [r.get("epoch", 0), r["iteration"]]
            if any(v is None or not np.isfinite(float(v)) or float(v) != int(float(v)) for v in coordinates):
                raise ValueError(f"{label}: epoch and iteration must be finite integer coordinates")
            epoch, iteration = map(lambda v: int(float(v)), coordinates)
            groups[epoch, iteration].append(score)
        for generation, ((epoch, iteration), scores) in enumerate(sorted(groups.items())):
            rows.append({"run": label, "generation": generation, "epoch": epoch,
                         "iteration": iteration, "n_individuals": len(scores),
                         "best_fitness": max(scores), "mean_fitness": float(np.mean(scores))})
    return rows


def plot_evolution(runs, out):
    rows = generation_summary(runs)
    if not rows:
        raise ValueError("No valid evaluated individuals in run histories")
    write_csv(out / "generation_summary.csv", rows)
    plt = pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, constrained_layout=True)
    for label in runs:
        subset = [r for r in rows if r["run"] == label]
        for ax, key in zip(axs, ("best_fitness", "mean_fitness")):
            ax.plot([r["generation"] for r in subset], [r[key] for r in subset],
                    label=label, marker="o" if len(subset) < 20 else None, markersize=3)
    for ax, name in zip(axs, ("Best individual in generation", "Mean individual fitness in generation")):
        ax.set(xlabel="Observed generation (ordered epoch, iteration)", ylabel="Returned fitness ↑", title=name)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Evolution by independent starting seed · each candidate counts once")
    save_figure(fig, out, "evolution")
    write_json(out / "evolution_metadata.json", {"runs": list(runs), "aggregation": "All supplied agents pooled within each run by (epoch, iteration); missing generations are not filled; source coordinates in CSV. Best is per-generation, not cumulative. Histories can contain partial generations."})


def prepare_simulator(args):
    import dot_tracingtask as sim
    import torch
    torch.set_num_threads(args.threads)
    sim.args.render_replays = False
    sim.args.raster_replays = False
    sim.args.plot_every = 0
    sim.args.mute = True
    sim.args.write = False
    sim.DEVICE = torch.device(args.device)
    if args.steps is not None:
        sim.args.steps = args.steps
    if args.episodes is not None:
        sim.args.trn_eps = sim.args.tst_eps = args.episodes
    return sim


def evaluate(sim, param, seed, out, index):
    import copy
    from types import SimpleNamespace
    sim.args.seed = seed
    # DotSimulator treats zero as wall-clock. Use a stable, nonzero environment seed.
    sim.args.env_seed = seed + 1
    task_args = SimpleNamespace(path=str(out), agent_idx="analysis", agent_test_num=index)
    log_dir = out / "evaluation_logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / f"{index}.log").open("w") as log, contextlib.redirect_stdout(log):
        result = sim.do_task({"param": copy.deepcopy(param)}, task_args)
    pyplot().close("all")
    if fitness(result) is None:
        raise ValueError(f"Non-finite fitness at evaluation {index}")
    return {"param": copy.deepcopy(param), "seed": seed, "env_seed": seed + 1, **result}


# todo: prep plots for the search

def init_worker(args):
    global worker_sim
    worker_sim = prepare_simulator(args)


def evaluate_job(job):
    idx, param, seed, out = job
    result = evaluate(worker_sim, param, seed, out, f"{idx}_seed{seed}")
    result["candidate"] = idx
    return result


def run_experiment(args):
    import copy
    bounds, baseline = bounds_from_yaml(args.config)
    if args.baseline:
        raw = json.loads(Path(args.baseline).read_text())
        supplied = raw.get("param", raw)
        if set(supplied) != set(baseline):
            raise ValueError("Baseline must contain all YAML parameter keys")
        for key, default in baseline.items():
            values = np.asarray(supplied[key], dtype=float).reshape(-1)
            if len(values) != len(default):
                raise ValueError(f"Baseline {key} needs {len(default)} values, got {len(values)}")
            baseline[key] = values.tolist()
    axes = args.axes
    if axes and (len(set(axes)) != 2 or any(n not in bounds for n in axes)):
        raise ValueError("--axes must name two distinct YAML parameters (arrays use name[index])")
    for name, value in features({"param": baseline}).items():
        if name not in bounds or not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError(f"Baseline value outside YAML bounds: {name}={value}")
    sim = prepare_simulator(args)
    out = Path(args.output)
    manifest = {"command": args.command, "baseline": baseline, "bounds": bounds,
                "seeds": args.seeds, "axes": axes, "samples": args.samples,
                "grid_size": args.grid_size, "sampling_seed": args.sampling_seed,
                "simulator_settings": vars(sim.args), "device": args.device,
                "simulator_sha256": hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
    # Seed fields in settings are updated at every evaluation; seeds list is authoritative.
    manifest_path = out / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Output directory belongs to a different experiment. Choose a new --output.")
    else:
        write_json(manifest_path, manifest)
    candidates = []
    if args.command == "screen":
        # Independent Latin-hypercube coordinates cover every gene's full YAML range.
        rng = np.random.default_rng(args.sampling_seed)
        candidates = [copy.deepcopy(baseline) for _ in range(args.samples)]
        for name, (lo, hi) in bounds.items():
            values = (rng.permutation(args.samples) + rng.random(args.samples)) / args.samples
            for param, value in zip(candidates, values):
                set_feature(param, name, lo + value * (hi - lo))
    else:
        for y in np.linspace(*bounds[axes[1]], args.grid_size):
            for x in np.linspace(*bounds[axes[0]], args.grid_size):
                param = copy.deepcopy(baseline)
                set_feature(param, axes[0], x)
                set_feature(param, axes[1], y)
                candidates.append(param)
    data_path = out / "evaluations.jsonl"
    records = read_jsonl(data_path) if data_path.exists() else []
    done = {(r["candidate"], r["seed"]) for r in records}
    total = len(candidates) * len(args.seeds)
    # Rewrite parsed complete records to repair an interrupted final JSONL line.
    repaired_path = data_path.with_suffix(".jsonl.tmp")
    with repaired_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    repaired_path.replace(data_path)
    with data_path.open("a") as f:
        jobs = [(idx, param, seed, out) for idx, param in enumerate(candidates)
                for seed in args.seeds if (idx, seed) not in done]
        with contextlib.ExitStack() as stack:
            if args.workers > 1:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor, as_completed
                pool = stack.enter_context(ProcessPoolExecutor(
                    max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=init_worker, initargs=(args,)))
                futures = [pool.submit(evaluate_job, job) for job in jobs]
                results = (future.result() for future in as_completed(futures))
            else:
                global worker_sim
                worker_sim = sim
                results = map(evaluate_job, jobs)
            for record in results:
                f.write(json.dumps(record, allow_nan=False) + "\n")
                f.flush()
                records.append(record)
                print(f"Completed {len(records)}/{total}: candidate={record['candidate']}, seed={record['seed']}", flush=True)
    if args.command == "screen":
        # Average replicates before splitting to avoid the same gene leaking across train/test.
        averaged = []
        for idx in range(len(candidates)):
            group = [r for r in records if r["candidate"] == idx]
            averaged.append({**group[0], "fitnessScore": float(np.mean([fitness(r) for r in group])),
                             "ac_active_fraction": float(np.mean([r["ac_active_fraction"] for r in group]))})
        axes = rank_parameters(averaged, out, args.rank_metric)
        write_json(out / "selected_axes.json", axes)
    label = (f"{args.command.capitalize()} · {sim.args.steps} decisions/episode · "
             f"train/test episodes {sim.args.trn_eps}/{sim.args.tst_eps} · simulation seeds {args.seeds}")
    landscape(records, axes, out, controlled=args.command == "sweep", title=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    observed = sub.add_parser("observed", help="Explore one consistent run/model version; do not mix changed scoring rules")
    observed.add_argument("input")
    observed.add_argument("--axes", nargs=2)
    observed.add_argument("--bins", type=int, default=5)
    observed.add_argument("--rank-metric", choices=["fitnessScore", "ac_active_fraction"], default="fitnessScore")
    observed.add_argument("--output", required=True)
    evolution = sub.add_parser("evolution", help="Best and mean per generation, one line per independent seeded run")
    evolution.add_argument("--run", action="append", required=True, metavar="LABEL=PATH", help="Repeat for each independent run directory or JSONL history")
    evolution.add_argument("--output", required=True)
    for name in ("screen", "sweep"):
        p = sub.add_parser(name, help="Latin-hypercube gene screening" if name == "screen" else "Controlled two-gene slice with all other genes fixed")
        p.add_argument("--config", default=str(Path(__file__).with_name("task_dotTracing.yaml")))
        p.add_argument("--baseline", help="JSON containing all decoded genes; defaults to YAML midpoints")
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--sampling-seed", type=int, default=42)
        p.add_argument("--samples", type=int, default=64)
        p.add_argument("--rank-metric", choices=["fitnessScore", "ac_active_fraction"], default="fitnessScore")
        p.add_argument("--grid-size", type=int, default=11)
        p.add_argument("--axes", nargs=2, required=name == "sweep")
        p.add_argument("--steps", type=int, help="Override task default; shortened runs are pilot experiments")
        p.add_argument("--episodes", type=int, help="Override both training and test episode counts")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--workers", type=int, default=1, help="Independent evaluation processes; use workers × threads <= allocated CPUs")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "observed":
        if args.axes and len(set(args.axes)) != 2:
            parser.error("--axes must name two distinct parameters")
        records = load_records(args.input)
        axes = args.axes or rank_parameters(records, out, args.rank_metric)
        landscape(records, axes, out, args.bins, title=Path(args.input).name)
        write_json(out / "selected_axes.json", axes)
        write_json(out / "observations.json", records)
        scores = [fitness(r) for r in records if fitness(r) is not None]
        write_json(out / "summary.json", {"source": str(Path(args.input).resolve()), "n": len(scores),
                   "best_fitness": max(scores), "mean_fitness": float(np.mean(scores)),
                   "healthy_test_candidates": sum(r.get("ac_all_episodes_in_band") is True for r in records),
                   "note": "Stdout genes/activity are rounded. Fitness sign does not identify healthy activity. No generation inference from evaluation order."})
    elif args.command == "evolution":
        runs = {}
        for spec in args.run:
            label, sep, path = spec.partition("=")
            if not sep or not label or label in runs:
                parser.error("Each --run must have a unique LABEL=PATH")
            runs[label] = load_records(path)
        plot_evolution(runs, out)
    else:
        if args.samples < 20 or args.grid_size < 2 or args.threads < 1 or args.workers < 1:
            parser.error("Need samples >= 20, grid-size >= 2, threads >= 1 and workers >= 1")
        if any(s < 0 for s in args.seeds) or len(args.seeds) != len(set(args.seeds)):
            parser.error("Seeds must be distinct nonnegative integers")
        if (args.steps is not None and args.steps < 1) or (args.episodes is not None and args.episodes < 1):
            parser.error("Steps and episodes must be positive")
        run_experiment(args)
    print(f"Saved analysis to {out.resolve()}")


if __name__ == "__main__":
    main()

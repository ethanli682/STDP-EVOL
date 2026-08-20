import argparse
import json
import os
import shutil
import tempfile
import time
from statistics import mean

import numpy as np

from GA_utils_framed_records import FramedRecordReader, FramedRecordWriter


def dependency_available(codec: str) -> bool:
    if codec == "pickle":
        return True
    if codec == "msgpack":
        try:
            import msgpack  # noqa: F401

            return True
        except Exception:
            return False
    if codec == "msgspec":
        try:
            import msgspec  # noqa: F401

            return True
        except Exception:
            return False
    return False


def synthetic_record(record_id: int, gene_length: int) -> dict:
    rng = np.random.default_rng(seed=record_id)
    gene = rng.standard_normal(gene_length, dtype=np.float32)
    return {
        "gene": gene,
        "fitnessScore": [float(rng.normal(0.0, 1.0)) for _ in range(4)],
        "loss": [float(abs(rng.normal(0.0, 0.5))) for _ in range(4)],
        "epoch": int(record_id // 100),
        "iteration": int(record_id),
        "build_in_param": {
            "history": [float(rng.random()) for _ in range(8)],
            "flags": [int(rng.integers(0, 2)) for _ in range(8)],
        },
        "metadata": [float(rng.random()) for _ in range(16)],
    }


def one_run(codec: str, records: list, compression_level: int, out_dir: str):
    csv_path = os.path.join(out_dir, f"bench_{codec}.log.csv")
    data_path = csv_path.replace(".log.csv", ".log.framed.bin")
    index_path = csv_path.replace(".log.csv", ".log.framed.idx")

    t0 = time.perf_counter()
    with FramedRecordWriter(
        data_path=data_path,
        index_path=index_path,
        codec=codec,
        compression_level=compression_level,
    ) as writer:
        with open(csv_path, "wt") as fcsv:
            for rec in records:
                slim = dict(rec)
                slim.pop("gene", None)
                fcsv.write(json.dumps(slim) + "\n")
                writer.append(rec)
    t_write = time.perf_counter() - t0

    bytes_total = os.path.getsize(data_path) + os.path.getsize(index_path)

    t1 = time.perf_counter()
    read_count = 0
    gene_dtype_ok = True
    gene_shape_ok = True
    with FramedRecordReader(data_path=data_path, index_path=index_path, codec=codec) as reader:
        for idx in range(len(reader)):
            rec = reader.read(idx)
            read_count += 1
            if "gene" in rec:
                gene_dtype_ok = gene_dtype_ok and isinstance(rec["gene"], np.ndarray) and rec["gene"].dtype == np.float32
                gene_shape_ok = gene_shape_ok and rec["gene"].ndim == 1
    t_read = time.perf_counter() - t1

    return {
        "codec": codec,
        "records": len(records),
        "compression_level": compression_level,
        "write_seconds": t_write,
        "read_seconds": t_read,
        "write_records_per_sec": (len(records) / t_write) if t_write > 0 else 0.0,
        "read_records_per_sec": (read_count / t_read) if t_read > 0 else 0.0,
        "bytes_total": bytes_total,
        "bytes_per_record": bytes_total / max(1, len(records)),
        "gene_dtype_float32_ok": gene_dtype_ok,
        "gene_shape_1d_ok": gene_shape_ok,
    }


def summarize(results: list):
    grouped = {}
    for r in results:
        grouped.setdefault(r["codec"], []).append(r)

    summary = []
    for codec, rows in grouped.items():
        summary.append(
            {
                "codec": codec,
                "runs": len(rows),
                "records": rows[0]["records"],
                "compression_level": rows[0]["compression_level"],
                "write_seconds_mean": mean([x["write_seconds"] for x in rows]),
                "read_seconds_mean": mean([x["read_seconds"] for x in rows]),
                "write_rps_mean": mean([x["write_records_per_sec"] for x in rows]),
                "read_rps_mean": mean([x["read_records_per_sec"] for x in rows]),
                "bytes_total_mean": mean([x["bytes_total"] for x in rows]),
                "bytes_per_record_mean": mean([x["bytes_per_record"] for x in rows]),
                "gene_dtype_float32_ok": all([x["gene_dtype_float32_ok"] for x in rows]),
                "gene_shape_1d_ok": all([x["gene_shape_1d_ok"] for x in rows]),
            }
        )
    summary.sort(key=lambda x: x["write_rps_mean"], reverse=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark framed appendable record codecs under same compression and record size")
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--gene-length", type=int, default=1024)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    records = [synthetic_record(i, args.gene_length) for i in range(args.records)]

    codecs = ["pickle", "msgpack", "msgspec"]
    available = [c for c in codecs if dependency_available(c)]
    missing = [c for c in codecs if c not in available]

    print(f"Benchmark config: records={args.records}, gene_length={args.gene_length}, compression_level={args.compression_level}, repeats={args.repeats}")
    if missing:
        print(f"Skipping unavailable codecs: {missing}")

    all_results = []
    workspace_tmp = tempfile.mkdtemp(prefix="framed_bench_")
    try:
        for codec in available:
            for rep in range(args.repeats):
                out_dir = os.path.join(workspace_tmp, f"{codec}_{rep}")
                os.makedirs(out_dir, exist_ok=True)
                result = one_run(codec, records, args.compression_level, out_dir)
                all_results.append(result)
                print(
                    f"{codec:7s} rep={rep+1}/{args.repeats} "
                    f"write={result['write_records_per_sec']:.1f} rec/s "
                    f"read={result['read_records_per_sec']:.1f} rec/s "
                    f"size={result['bytes_per_record']:.1f} B/rec"
                )

        summary = summarize(all_results)
        print("\n=== Summary (mean across repeats) ===")
        for row in summary:
            print(
                f"{row['codec']:7s} "
                f"write={row['write_rps_mean']:.1f} rec/s "
                f"read={row['read_rps_mean']:.1f} rec/s "
                f"size={row['bytes_per_record_mean']:.1f} B/rec "
                f"gene_f32={row['gene_dtype_float32_ok']}"
            )

        payload = {
            "config": {
                "records": args.records,
                "gene_length": args.gene_length,
                "compression_level": args.compression_level,
                "repeats": args.repeats,
            },
            "raw_runs": all_results,
            "summary": summary,
            "missing_codecs": missing,
        }

        if args.output:
            with open(args.output, "wt") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved benchmark JSON to: {args.output}")

    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

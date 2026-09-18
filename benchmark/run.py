#!/usr/bin/env python3
"""Run serial, correctness-checked local OLS benchmarks; retain every sample."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "benchmark"
TOLERANCE = 0.00001


def dataset(n: int):
    import numpy as np
    i = np.arange(n, dtype=np.uint32)
    x = ((i * 17 % 4096).astype(np.float32) - np.float32(2048)) / np.float32(256)
    noise = ((i * 13 % 1024).astype(np.float32) - np.float32(512)) / np.float32(8192)
    y = np.float32(3) * x + np.float32(2) + noise
    return x, y


def reference(n: int) -> dict:
    import numpy as np
    x, y = dataset(n)
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    dx, dy = x64 - x64.mean(), y64 - y64.mean()
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    intercept = float(y64.mean() - slope * x64.mean())
    return {"slope": slope, "intercept": intercept,
            "dataset_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}


class AccuracyError(ValueError):
    """Finite model coefficients missed the declared numerical tolerance."""


def check_samples(samples: list[dict], ref: dict, trials: int, warmups: int) -> None:
    errors = []
    for phase, expected in (("setup", 1), ("warmup", warmups), ("fit", trials)):
        if sum(s.get("phase") == phase for s in samples) != expected:
            raise ValueError(f"Expected {expected} {phase} records")
    for sample in samples:
        if sample["phase"] not in {"setup", "warmup", "fit"}:
            raise ValueError("Unknown sample phase")
        if not isinstance(sample["ns"], (int, float)) or not math.isfinite(sample["ns"]) or sample["ns"] <= 0:
            raise ValueError(f"Invalid timing: {sample}")
        if sample["phase"] != "setup":
            for name in ("slope", "intercept"):
                value = sample[name]
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name}: {value}")
                error = abs(value - ref[name])
                if error > TOLERANCE:
                    errors.append(error)
    if errors:
        raise AccuracyError(f"Maximum coefficient error {max(errors):.9g} exceeds {TOLERANCE}")


def worker(method: str, n: int, warmups: int, trials: int) -> None:
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(os.environ["OMP_NUM_THREADS"]))
    # Import/JIT/runtime initialization is excluded; per-trial temporary arrays,
    # estimator allocation, evaluation, and host-readable coefficients are included.
    if method == "sklearn":
        from sklearn.linear_model import LinearRegression
    elif method == "scipy":
        from scipy.stats import linregress
    elif method.startswith("mlx"):
        import mlx.core as mx
        mx.set_default_device(mx.gpu if method == "mlx_gpu" else mx.cpu)

    start = time.perf_counter_ns()
    x, y = dataset(n)
    if method == "sklearn":
        X = x.reshape(-1, 1)
        def fit():
            m = LinearRegression().fit(X, y)
            return float(m.coef_[0]), float(m.intercept_)
    elif method == "numpy":
        def fit():
            xm, ym = x.mean(), y.mean()
            dx, dy = x - xm, y - ym
            slope = np.dot(dx, dy) / np.dot(dx, dx)
            return float(slope), float(ym - slope * xm)
    elif method == "scipy":
        def fit():
            m = linregress(x, y)
            return float(m.slope), float(m.intercept)
    else:
        xx, yy = mx.array(x), mx.array(y)
        mx.eval(xx, yy)
        @mx.compile
        def coefficients(a, b):
            xm, ym = mx.mean(a), mx.mean(b)
            dx, dy = a - xm, b - ym
            slope = mx.sum(dx * dy) / mx.sum(dx * dx)
            return slope, ym - slope * xm
        def fit():
            slope, intercept = coefficients(xx, yy)
            mx.eval(slope, intercept)  # Wait for device work; never time graph creation alone.
            return slope.item(), intercept.item()
    print(json.dumps({"phase": "setup", "ns": time.perf_counter_ns() - start}), flush=True)
    for i in range(warmups + trials):
        start = time.perf_counter_ns()
        slope, intercept = fit()
        elapsed = time.perf_counter_ns() - start
        print(json.dumps({"phase": "warmup" if i < warmups else "fit", "ns": elapsed,
                          "slope": slope, "intercept": intercept}), flush=True)


def command(args: list[str], *, timeout: int = 600) -> str:
    completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {args}\n"
                           f"{completed.stdout}\n{completed.stderr}")
    return completed.stdout.strip()


def executable(name: str, fallback: Path | None = None) -> str:
    path = shutil.which(name)
    if path:
        return path
    if fallback and fallback.is_file():
        return str(fallback)
    raise RuntimeError(f"Required executable not found: {name}")


def metadata(bend: str, threads: int) -> dict:
    import numpy as np
    from threadpoolctl import threadpool_info
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "bend": command([bend, "--version"]),
            "clang": command(["clang", "--version"]).splitlines()[0],
            "node": command(["node", "--version"]),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
            "cpu_thread_limit": threads,
            "thread_environment": {k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "numpy_build": np.show_config(mode="dicts"),
            "threadpools": [{k: v for k, v in entry.items() if k != "filepath"}
                            for entry in threadpool_info()]}
    # Avoid embedding local build paths in the public report.
    info["numpy_build"] = {k: v for k, v in info["numpy_build"].items()
                           if k in {"Build Dependencies", "SIMD Extensions"}}
    if sys.platform == "darwin":
        info["hardware"] = {key: command(["sysctl", "-n", key]) for key in
            ("machdep.cpu.brand_string", "hw.physicalcpu", "hw.memsize",
             "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu")}
        info["macos"] = command(["sw_vers", "-productVersion"])
        info["power_source"] = command(["pmset", "-g", "batt"]).splitlines()[0]
        info["thermal_status_before_run"] = command(["pmset", "-g", "therm"])
        displays = json.loads(command(["system_profiler", "SPDisplaysDataType", "-json"]))
        info["gpu"] = [{k: d[k] for k in ("sppci_model", "sppci_cores", "spdisplays_metal") if k in d}
                       for d in displays["SPDisplaysDataType"]]
    try:
        info["packages"]["mlx"] = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        pass
    return info


LABELS = {
    "bend_1": "Bend CPU · 1 thread",
    "bend_4": "Bend CPU · 4 threads",
    "bend_all": "Bend CPU · all requested threads",
    "bend_gpu": "Bend Metal GPU",
    "sklearn": "scikit-learn LinearRegression · F32",
    "numpy": "NumPy centered OLS · F32",
    "scipy": "SciPy linregress · mixed F32/F64",
    "mlx_cpu": "MLX compiled OLS · CPU F32",
    "mlx_gpu": "MLX compiled OLS · Metal F32",
    "c": "C pairwise OLS · 1 thread F32",
    "javascript": "Node.js pairwise OLS · 1 thread F32",
}


def markdown(report: dict) -> str:
    sizes = report["config"]["sizes"]
    lines = ["# Local linear regression benchmark", "",
             "Median warmed fit latency in milliseconds; lower is better. "
             "Brackets show p10–p90 across all recorded fit samples.", "",
             "| Implementation | " + " | ".join(f"{n:,} rows" for n in sizes) + " |",
             "| --- | " + " | ".join("---:" for _ in sizes) + " |"]
    for method in report["config"]["methods"]:
        cells = []
        for n in sizes:
            stats = report["summary"][method][str(n)]
            cells.append(f"{stats['median_ms']:.3f} [{stats['p10_ms']:.3f}–{stats['p90_ms']:.3f}]" + (" †" if not stats['accuracy_passed'] else ""))
        label = f"Bend CPU · {report['config']['threads']} threads" if method == "bend_all" else LABELS[method]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "Data generation, Bend tree construction, process startup, imports, compilation, "
              "and warmups are excluded from the fit table. Setup timing, first fits, every measured "
              "fit, commands, correctness checks, and environment metadata are in the JSON file.", ""]
    failed = [(method, n, report["summary"][method][str(n)]["max_coefficient_error"])
              for method in report["config"]["methods"] for n in sizes
              if not report["summary"][method][str(n)]["accuracy_passed"]]
    if failed:
        lines += ["† Outside the declared 1e-5 coefficient tolerance; retained for transparency, not treated as an equivalent-accuracy speed comparison.", ""]
        lines += [f"- {LABELS[m]}, {n:,} rows: maximum coefficient error {err:.9g}." for m, n, err in failed]
    return "\n".join(lines) + "\n"


def run(args) -> None:
    import numpy as np
    bend = executable("bend", Path.home() / ".bend/bin/bend")
    node = executable("node")
    BUILD.mkdir(parents=True, exist_ok=True)
    sizes = [2 ** d for d in args.depths]
    methods = list(LABELS)
    if args.skip_mlx:
        methods = [m for m in methods if not m.startswith("mlx")]
    if args.skip_gpu:
        methods = [m for m in methods if m not in {"bend_gpu", "mlx_gpu"}]
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": command(["git", "rev-parse", "HEAD"]),
              "compiler_flags": ["clang", "-O3", "-ffp-contract=off", "-std=c11"],
              "bend_build_command": "bend build/benchmark/rows_DEPTH.bend -o build/benchmark/rows_DEPTH",
              "working_tree_dirty": bool(command(["git", "status", "--porcelain"])),
              "config": {"sizes": sizes, "warmups_per_process": args.warmups,
                         "trials_per_process": args.trials, "rounds": args.rounds,
                         "threads": args.threads, "order_seed": args.seed, "methods": methods,
                         "coefficient_absolute_tolerance": TOLERANCE},
              "environment": metadata(bend, args.threads), "source_sha256": {},
              "references": {str(n): reference(n) for n in sizes}, "runs": []}
    for directory in ("benchmark", "src"):
        for p in sorted((ROOT / directory).glob("*")):
            if p.is_file() and p.suffix in {".py", ".bend", ".c", ".js", ".mjs", ".txt"}:
                report["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    command(["clang", "-O3", "-ffp-contract=off", "-std=c11", "benchmark/native.c", "-o", "build/benchmark/native"])
    for depth in args.depths:
        source = BUILD / f"rows_{depth}.bend"
        source.write_text("import Base\nimport ../../benchmark/bend.bend as Bench\n\n"
                          f"def main() -> IO(Unit):\n  Bench.run({depth}n, {args.warmups}n, {args.trials}n)\n")
        print(f"Compiling Bend for {2**depth:,} rows", flush=True)
        command([bend, str(source.relative_to(ROOT)), "-o", f"build/benchmark/rows_{depth}"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for round_id in range(args.rounds):
        jobs = [(m, depth) for m in methods for depth in args.depths]
        rng.shuffle(jobs)
        for method, depth in jobs:
            n = 2 ** depth
            if method.startswith("bend"):
                threads = {"bend_1": 1, "bend_4": 4}.get(method, args.threads)
                cmd = [f"build/benchmark/rows_{depth}", "--threads", str(threads),
                       "--gpu", "1GB" if method == "bend_gpu" else "off"]
            elif method in {"c", "javascript"}:
                cmd = (["build/benchmark/native"] if method == "c" else [node, "benchmark/javascript.mjs"])
                cmd += [str(n), str(args.warmups), str(args.trials)]
            else:
                cmd = [sys.executable, "benchmark/run.py", "--worker", method, "--n", str(n),
                       "--warmups", str(args.warmups), "--trials", str(args.trials), "--threads", str(args.threads)]
            print(f"Round {round_id+1}/{args.rounds}: {method}, {n:,} rows", flush=True)
            wall_start = time.perf_counter_ns()
            raw = command(cmd)
            wall_ns = time.perf_counter_ns() - wall_start
            samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
            accuracy_error = None
            try:
                check_samples(samples, report["references"][str(n)], args.trials, args.warmups)
            except AccuracyError as error:
                accuracy_error = str(error)
                print(f"  ACCURACY FLAG: {accuracy_error}", flush=True)
            portable_cmd = ["python" if x == sys.executable else "node" if x == node else x for x in cmd]
            report["runs"].append({"round": round_id + 1, "method": method, "n": n,
                                   "command": portable_cmd, "process_wall_ns": wall_ns,
                                   "accuracy_passed": accuracy_error is None,
                                   "accuracy_error": accuracy_error, "samples": samples})
            out.write_text(json.dumps(report, indent=2) + "\n")  # Preserve completed work on failure.
    report["summary"] = {}
    for method in methods:
        report["summary"][method] = {}
        for n in sizes:
            records = [r for r in report["runs"] if r["method"] == method and r["n"] == n]
            values = [s["ns"] / 1e6 for r in records for s in r["samples"] if s["phase"] == "fit"]
            report["summary"][method][str(n)] = {
                "accuracy_passed": all(r["accuracy_passed"] for r in records),
                "median_ms": statistics.median(values), "p10_ms": float(np.percentile(values, 10)),
                "p90_ms": float(np.percentile(values, 90)), "samples": len(values),
                "round_medians_ms": [statistics.median(s["ns"] / 1e6 for s in r["samples"] if s["phase"] == "fit") for r in records],
                "max_coefficient_error": max(abs(s[k] - report["references"][str(n)][k])
                    for r in records for s in r["samples"] if s["phase"] != "setup" for k in ("slope", "intercept"))}
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, nargs="+", default=[12, 16, 20])
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--output", default="build/benchmark/results.json")
    p.add_argument("--skip-mlx", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--worker", choices=["numpy", "sklearn", "scipy", "mlx_cpu", "mlx_gpu"])
    p.add_argument("--n", type=int, default=4096)
    args = p.parse_args()
    if any(d < 2 or d > 20 for d in args.depths) or len(set(args.depths)) != len(args.depths):
        p.error("depths must be distinct integers from 2 to 20")
    if args.trials < 1 or args.warmups < 0 or args.rounds < 1 or args.threads < 1 or args.threads > 128:
        p.error("invalid trial, warmup, round, or thread count")
    if args.worker and (args.n < 2 or args.n > 2**20):
        p.error("worker n must be between 2 and 2^20")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["BEND_NO_TELEMETRY"] = "1"
    if args.worker:
        worker(args.worker, args.n, args.warmups, args.trials)
    else:
        run(args)


if __name__ == "__main__":
    main()

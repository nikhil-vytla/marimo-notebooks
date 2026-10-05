"""Run the full Fig 2A sweep by launching train.py as parallel subprocesses."""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--aws", default="0,1,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_jobs(args):
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    aws = [float(a) for a in args.aws.split(",") if a.strip()]

    root = Path(args.results_root)
    logs = Path("logs")
    jobs = []
    # Seeds outermost so partial results cover every setting early.
    for seed in range(args.seeds):
        for variant in variants:
            for aw in aws:
                stem = f"aw{aw:g}_h{args.hidden}_s{seed}"
                out = root / variant / f"{stem}.json"
                log = logs / f"{variant}_aw{aw:g}_s{seed}.log"
                cmd = [
                    sys.executable,
                    "train.py",
                    "--aw",
                    f"{aw:g}",
                    "--seed",
                    str(seed),
                    "--hidden",
                    str(args.hidden),
                    "--epochs",
                    str(args.epochs),
                    "--out",
                    str(out),
                ]
                if variant == "b":
                    cmd.append("--center")
                jobs.append(
                    {
                        "variant": variant,
                        "aw": aw,
                        "seed": seed,
                        "out": out,
                        "log": log,
                        "cmd": cmd,
                    }
                )
    return jobs


def main():
    args = parse_args()
    jobs = build_jobs(args)
    total = len(jobs)

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))
        print(f"total: {total} jobs")
        return 0

    Path("logs").mkdir(exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)

    pending = [j for j in jobs if not j["out"].exists()]
    skipped = total - len(pending)
    if skipped:
        print(f"skipping {skipped} existing runs")

    if not pending:
        print("nothing to do")
        return 0

    state = {"done": 0}
    failed = []
    lock = threading.Lock()
    start = time.perf_counter()

    def run(job):
        with open(job["log"], "w") as log_file:
            t0 = time.perf_counter()
            result = subprocess.run(
                job["cmd"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            elapsed = time.perf_counter() - t0
        with lock:
            state["done"] += 1
            print(
                f"[done {state['done']}/{len(pending)}] {job['variant']} "
                f"{job['aw']:g} {job['seed']}  rc={result.returncode}  "
                f"{elapsed:.1f}s"
            )
            if result.returncode != 0:
                failed.append(job)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, pending))

    print(f"total wall time {time.perf_counter() - start:.1f}s")
    if failed:
        print(f"{len(failed)} job(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

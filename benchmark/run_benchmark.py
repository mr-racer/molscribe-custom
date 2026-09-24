"""Run a MolScribe checkout on a benchmark set and record predictions + timing.

The molscribe package is imported from --code_root, so the same script benchmarks any revision (e.g. a baseline
worktree and the current branch).

usage:
  python benchmark/run_benchmark.py --code_root <repo checkout> --ckpt <.pth> --data <dir with *.csv>
      --set general --batch_size 1 --device cuda --out results/<name>
"""
import argparse
import csv
import inspect
import json
import os
import platform
import sys
import time

import cv2
import numpy as np


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--set", required=True, help="general | metal")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt", action="append", default=[],
                    help="extra MolScribe(...) kwargs of the new build, e.g. --opt precision=fp16")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.code_root))
    import torch
    from molscribe import MolScribe

    if args.threads:
        torch.set_num_threads(args.threads)
    kwargs = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        kwargs[k] = {"true": True, "false": False}.get(v.lower(), v)
    supported = inspect.signature(MolScribe.__init__).parameters
    unknown = [k for k in kwargs if k not in supported]
    if unknown:
        raise SystemExit(f"this checkout does not support options {unknown}")

    t0 = time.perf_counter()
    model = MolScribe(args.ckpt, device=torch.device(args.device), **kwargs)
    load_s = time.perf_counter() - t0

    with open(os.path.join(args.data, f"{args.set}.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    images = [load_image(os.path.join(args.data, r["image"])) for r in rows]

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    # warm-up (cudnn autotune, lazy module loading, CUDA graph capture in the new build)
    if args.warmup:
        model.predict_images(images[:args.warmup], batch_size=min(args.batch_size, args.warmup))
        sync()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    preds, per_batch = [], []
    t_start = time.perf_counter()
    for i in range(0, len(images), args.batch_size):
        chunk = images[i:i + args.batch_size]
        tb = time.perf_counter()
        out = model.predict_images(chunk, batch_size=args.batch_size)
        sync()
        per_batch.append((time.perf_counter() - tb, len(chunk)))
        preds += [o["smiles"] for o in out]
    total_s = time.perf_counter() - t_start

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.set}_bs{args.batch_size}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"])
        for r, p in zip(rows, preds):
            w.writerow([r["id"], p])
    per_img = [t / n for t, n in per_batch]
    timing = dict(
        set=args.set, n=len(rows), batch_size=args.batch_size, device=args.device, options=kwargs,
        model_load_s=round(load_s, 2), total_s=round(total_s, 3),
        ms_per_image=round(1000 * total_s / len(rows), 2),
        images_per_s=round(len(rows) / total_s, 3),
        ms_per_image_median_batch=round(1000 * float(np.median(per_img)), 2),
        peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if args.device.startswith("cuda") else None,
        torch=torch.__version__, threads=torch.get_num_threads(), host=platform.node(),
        gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else platform.processor(),
    )
    with open(os.path.join(args.out, f"{args.set}_bs{args.batch_size}_timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps(timing))


if __name__ == "__main__":
    main()

"""Consistency checks for the inference changes of this branch.

1. `import molscribe` works without onmt / torchtext / albumentations / sklearn / skimage / matplotlib / pandas.
2. molscribe.transforms.InferenceTransform == dataset.get_transforms(augment=False), bitwise (needs albumentations).
3. Fast decoding (static batch + preallocated KV cache, optionally CUDA graphs / fp16) vs the reference GreedySearch
   path: identical atoms, coordinates and bonds; reported per batch size.

usage: python benchmark/check_equivalence.py --ckpt <.pth> --data <data dir> --set general_cpu100 [--device cuda]
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKED = ["onmt", "torchtext", "albumentations", "sklearn", "skimage", "matplotlib", "pandas"]


def check_imports():
    code = f"""
import sys
sys.path.insert(0, {ROOT!r})
from molscribe import MolScribe
print('loaded:', sorted({{m.split('.')[0] for m in sys.modules}} & set({BLOCKED!r})))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok = out.returncode == 0 and out.stdout.strip().endswith("loaded: []")
    print(f"[imports] molscribe does not load {BLOCKED}: {'OK' if ok else 'FAIL'} {out.stdout.strip()}")
    if out.returncode != 0:
        print(out.stderr[-2000:])
    return ok


def load_image(path):
    import cv2
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def check_transforms(images, device):
    try:
        from molscribe.dataset import get_transforms
    except ImportError as e:
        print(f"[transforms] skipped (albumentations not installed: {e})")
        return True
    import torch
    from molscribe.transforms import InferenceTransform
    ref, new = get_transforms(384, augment=False), InferenceTransform(384)
    bad = bad_dev = 0
    for img in images:
        expected = ref(image=img, keypoints=[])["image"]
        bad += not torch.equal(expected, new(img))
        on_device = new.normalize(torch.from_numpy(new.prepare(img)).unsqueeze(0).to(device))[0].cpu()
        bad_dev += not torch.equal(expected, on_device)
    print(f"[transforms] bitwise equal on {len(images) - bad}/{len(images)} images (CPU), "
          f"{len(images) - bad_dev}/{len(images)} ({device} normalize)")
    return bad == 0 and bad_dev == 0


def raw_predictions(model, images, batch_size, fast, precision="fp32", cuda_graph=False):
    import torch
    from molscribe.model import TransformerDecoderAR
    for module in model.decoder.modules():
        if isinstance(module, TransformerDecoderAR):
            module.fast_decoding = fast
            module.use_cuda_graph = cuda_graph
    model.precision = precision
    preds = []
    for i in range(0, len(images), batch_size):
        x = model._to_model_input(model._prepare(images[i:i + batch_size]))
        with torch.no_grad(), model._autocast():
            feats, hid = model.encoder(x)
            preds += model.decoder.decode(feats, hid)
    return [(p["chartok_coords"]["symbols"], p["chartok_coords"]["coords"], p["edges"]) for p in preds]


def compare(ref, new):
    return sum(r != n for r, n in zip(ref, new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--set", default="general_cpu100")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, ROOT)

    ok = check_imports()
    with open(os.path.join(args.data, f"{args.set}.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    images = [load_image(os.path.join(args.data, r["image"])) for r in rows]
    ok &= check_transforms(images, args.device)

    import torch
    from molscribe import MolScribe
    model = MolScribe(args.ckpt, device=torch.device(args.device))
    ref = raw_predictions(model, images, 1, fast=False)
    n = len(images)
    configs = [(1, "fp32", False), (8, "fp32", False)]
    if args.device.startswith("cuda"):
        configs += [(1, "fp32", True), (32, "fp32", True), (32, "tf32", True), (32, "fp16", True), (32, "bf16", True)]
    for bs, precision, graph in configs:
        new = raw_predictions(model, images, bs, fast=True, precision=precision, cuda_graph=graph)
        diff = compare(ref, new)
        exact = precision == "fp32"  # tf32/fp16/bf16 round matmul inputs
        print(f"[decode] fast bs={bs:<3d} {precision} cuda_graph={graph}: {n - diff}/{n} graphs identical to reference"
              + ("" if exact else "  (reduced precision: differences expected, check accuracy instead)"))
        ok &= diff == 0 or not exact
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()

"""Where does the time go? Per-stage wall time of MolScribe.predict_images on a batch stream.

Stages: preprocessing (CPU), encoder (GPU), autoregressive decoding (GPU), sequence parsing + edge head,
RDKit postprocessing (CPU). CUDA work is synchronized at stage boundaries, so each number is the stage's own time.

usage: python benchmark/profile_stages.py --ckpt <.pth> --data <data dir> --set general --batch_size 32 --device cuda
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--set", default="general")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--limit", type=int, default=256)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from molscribe import MolScribe
    from molscribe.chemistry import convert_graph_to_smiles

    rows = list(csv.DictReader(open(os.path.join(args.data, f"{args.set}.csv"), encoding="utf-8")))[:args.limit]
    images = [cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(args.data, r["image"]), np.uint8), 1),
                           cv2.COLOR_BGR2RGB) for r in rows]
    model = MolScribe(args.ckpt, device=torch.device(args.device), precision=args.precision)
    model.predict_images(images[:args.batch_size], batch_size=args.batch_size)  # warm-up / graph capture

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    dec = model.decoder
    fmt = "chartok_coords"
    t = dict(preprocess=0.0, encoder=0.0, decode=0.0, parse_edges=0.0, rdkit=0.0)
    start = time.perf_counter()
    for i in range(0, len(images), args.batch_size):
        batch = images[i:i + args.batch_size]
        t0 = time.perf_counter()
        x = torch.stack([model.transform(img) for img in batch]).to(model.device)
        sync(); t1 = time.perf_counter()
        with torch.no_grad(), model._autocast():
            feats, hid = model.encoder(x)
            sync(); t2 = time.perf_counter()
            out = dec.decoder[fmt].decode(feats, 1, 1, max_length=480)
            sync(); t3 = time.perf_counter()
            # the rest of Decoder.decode: token -> atoms, edge head
            dec.decoder[fmt].decode = lambda *a, **k: out
            try:
                preds = dec.decode(feats, hid)
            finally:
                del dec.decoder[fmt].decode
            sync(); t4 = time.perf_counter()
        convert_graph_to_smiles([p[fmt]["coords"] for p in preds], [p[fmt]["symbols"] for p in preds],
                                [p["edges"] for p in preds], images=batch, num_workers=1)
        t5 = time.perf_counter()
        for k, a, b in (("preprocess", t0, t1), ("encoder", t1, t2), ("decode", t2, t3), ("parse_edges", t3, t4),
                        ("rdkit", t4, t5)):
            t[k] += b - a
    total = time.perf_counter() - start
    n = len(images)
    print(f"{n} images, batch {args.batch_size}, {args.precision}: {1000 * total / n:.1f} ms/img total")
    for k, v in t.items():
        print(f"  {k:12s} {1000 * v / n:7.2f} ms/img  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()

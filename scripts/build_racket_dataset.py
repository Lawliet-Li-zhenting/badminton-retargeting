"""Self-labelled racket dataset: run the racket detector over a whole broadcast video (pose cache
from retarget_broadcast / pose_pass), in chunks with their own median background.

    python scripts/build_racket_dataset.py data/videos/standard_video.mp4 [--chunk 1500]
-> results/racket_dataset/<stem>_det.npz (uv, conf, hand per frame of the pose cache)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.calib import Camera  # noqa: E402
from badrt.racket import detect_rackets  # noqa: E402

PER_FRAME = ("ok", "main", "score", "world", "image", "vis", "ground", "lat", "n_people", "fallback", "reset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pose", default=None, help="pose cache (default results/broadcast/<stem>_0_0_pose.npz)")
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--out", default="results/racket_dataset")
    a = ap.parse_args()
    stem = Path(a.video).stem
    cam = Camera.load(f"assets/calib/{stem}.json")
    data = dict(np.load(a.pose or f"results/broadcast/{stem}_0_0_pose.npz"))
    start0 = int(data["start"])
    T = len(data["ok"])
    uv = np.full((T, 2), np.nan); conf = np.zeros(T); hand = np.full(T, -1)
    t0 = time.time()
    for a0 in range(0, T, a.chunk):
        b0 = min(T, a0 + a.chunk)
        chunk = {k: (data[k][a0:b0] if k in PER_FRAME else data[k]) for k in data}
        if (chunk["main"].astype(bool) & chunk["ok"].astype(bool)).sum() < 30:
            continue
        r = detect_rackets(a.video, cam, chunk, start0 + a0, log=False)
        uv[a0:b0], conf[a0:b0], hand[a0:b0] = r["uv"], r["conf"], r["hand"]
        print(f"  {b0}/{T}  {time.time() - t0:.0f}s  detected {np.mean(r['hand'] >= 0):.2f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}_det.npz", uv=uv, conf=conf, hand=hand, start=start0)
    print("->", out / f"{stem}_det.npz", f"detected {np.mean(hand >= 0):.2f}, right wrist {np.mean(hand == 0):.2f}")


if __name__ == "__main__":
    main()

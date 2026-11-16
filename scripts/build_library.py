"""Cut stroke segments from retargeted videos into assets/stroke_library.npz.

    python scripts/build_library.py results/clear_tutorial results/lcw_jumpsmash ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.library import extract_segments, save_library  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "assets" / "stroke_library.npz"))
    ap.add_argument("--target-peak", type=float, default=10.0)
    args = ap.parse_args()
    segs = []
    for d in args.dirs:
        s = extract_segments(Path(d) / "retarget.npz", target_peak=args.target_peak)
        print(f"{d}: {len(s)} strokes")
        segs += s
    save_library(segs, args.out)
    stats = dict(total=len(segs),
                 by_kind={k: sum(s.kind == k for s in segs) for k in ("overhead", "mid", "low")},
                 by_side={k: sum(s.side == k for s in segs) for k in ("forehand", "backhand")},
                 contact_height_m=[round(s.height, 2) for s in segs],
                 peak_speed_mps=[round(s.peak_speed, 1) for s in segs],
                 fps=sorted({round(s.fps, 1) for s in segs}))
    print(json.dumps(stats, indent=1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

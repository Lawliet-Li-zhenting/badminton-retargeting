"""Offline player track from a front-facing video (stand-in for the live camera).

    python scripts/build_player_track.py data/videos/N9jukZlmkns.mp4
    python scripts/build_player_track.py data/videos/N9jukZlmkns.mp4 --flip --handedness left   # left-hander test

Writes assets/tracks/<stem>[_left].npz and prints the stroke statistics used by the match director.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.recorded import PlayerTrack, build_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--flip", action="store_true", help="mirror the frames (simulates a left-handed person)")
    ap.add_argument("--court-gain", type=float, default=2.0)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()
    stem = Path(args.video).stem + ("_left" if args.handedness == "left" else "")
    out = Path(args.out) if args.out else ROOT / "assets" / "tracks" / f"{stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_track(args.video, out, model=args.model, handedness=args.handedness, court_gain=args.court_gain,
                hfov=args.hfov, start=args.start, duration=args.duration, flip=args.flip)
    tr = PlayerTrack(out)
    print(json.dumps(tr.summary(), indent=1))


if __name__ == "__main__":
    main()

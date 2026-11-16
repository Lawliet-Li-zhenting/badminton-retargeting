"""Grab still frames from the result videos into docs/figures/ for the report."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def grab(video, t, out, width=960):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("missing", video)
        return
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    if not ok:
        print("no frame", video, t)
        return
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    cv2.imwrite(str(FIG / out), img)
    print("wrote", out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    R = ROOT / "results"
    for name, t in (("clear_tutorial", 20.0), ("clear_tutorial", 41.0), ("lcw_jumpsmash", 4.0),
                    ("jumpsmash_doubles", 6.0)):
        grab(R / name / "side_by_side.mp4", t, f"retarget_{name}_{int(t)}s.png", width=1280)
    demo = json.loads((R / "demo" / "demo.json").read_text())
    hits = [e for e in demo["events"] if e.get("kind") in ("smash", "clear", "drop")][:6]
    for e in hits[:3]:
        grab(R / "demo" / "demo.mp4", e["t"] - 0.05, f"demo_{e['who']}_{e['kind']}_{e['t']:.1f}s.png")
    grab(R / "demo" / "demo.mp4", 30.0, "demo_rally_30s.png")
    grab(R / "human_video" / "match.mp4", 8.0, "human_video_8s.png")
    for name in ("shadow_footwork", "clear_tutorial", "footwork6_scale"):
        p = R / name / "root_track.png"
        if p.exists():
            (FIG / f"root_track_{name}.png").write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()

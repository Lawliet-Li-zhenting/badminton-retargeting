"""Contact sheet of detected hit moments (player crops from the video) for manual checking of the
stroke classification: python scripts/stroke_contact_sheet.py <category|all> [--n 12] [--seed 0]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEOS = {"standard_video": ("data/videos/standard_video.mp4", 0),
          "denmark2025_asy": ("data/videos/denmark2025_asy.mp4", 18000)}
POSE = {"standard_video": "results/broadcast/standard_video_0_0_pose.npz",
        "denmark2025_asy": "results/broadcast/denmark2025_asy_18000_10800_pose.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cat")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default="results/strokes")
    ap.add_argument("--offsets", default="-6,0", help="frames relative to the hit moment shown per event")
    a = ap.parse_args()
    ev = json.loads(str(np.load(Path(a.dir) / "events.npz")["events"]))
    if a.cat != "all":
        z, s = a.cat.split("_")
        ev = [e for e in ev if e["zone"] == z and e["side"] == s]
    rng = np.random.default_rng(a.seed)
    sel = sorted(rng.choice(len(ev), min(a.n, len(ev)), replace=False)) if ev else []
    offs = [int(o) for o in a.offsets.split(",")]
    tiles = []
    caps, imgs = {}, {}
    for i in sel:
        e = ev[i]
        path, start = VIDEOS[e["video"]]
        cap = caps.setdefault(e["video"], cv2.VideoCapture(path))
        if e["video"] not in imgs:
            z = np.load(POSE[e["video"]])
            imgs[e["video"]] = z["image"][..., :2] * [float(z["width"]), float(z["height"])]
        uv = imgs[e["video"]]
        row = []
        for o in offs:
            k = e["frame"] + o
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + k)
            img = cap.read()[1]
            c = np.nanmedian(uv[k, :, :2], 0)
            hgt = max(60, np.nanmax(uv[k, :, 1]) - np.nanmin(uv[k, :, 1]))
            r = int(hgt * 1.0)
            x0, y0 = int(c[0] - r), int(c[1] - r * 1.1)
            pad = cv2.copyMakeBorder(img, 400, 400, 400, 400, cv2.BORDER_CONSTANT)
            crop = pad[y0 + 400:y0 + 400 + 2 * r, x0 + 400:x0 + 400 + 2 * r]
            crop = cv2.resize(crop, (240, 240))
            cv2.putText(crop, f"{o:+d}", (4, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            row.append(crop)
        t = np.hstack(row)
        cv2.putText(t, f"#{i} {e['zone']}_{e['side']} f{e['frame']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        tiles.append(t)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = Path(a.dir) / f"sheet_{a.cat}_{a.seed}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()

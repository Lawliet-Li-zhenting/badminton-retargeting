"""Calibrate a static broadcast camera from the court lines of a badminton video.

    python scripts/calibrate_video.py data/videos/standard_video.mp4 --net-posts 176,229 751,229
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.calib import calibrate, draw_court, median_background  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--net-posts", nargs=2, default=None, help="pixel u,v of the two net-post tops (check only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--range", nargs=2, type=int, default=None, help="frame range for the background median")
    ap.add_argument("--ref", type=int, default=None, help="a main-camera frame index (background selection)")
    a = ap.parse_args()
    out = Path(a.out or f"assets/calib/{Path(a.video).stem}.json")
    bg = median_background(a.video, n=240 if a.ref is not None else 120, frame_range=a.range, ref_frame=a.ref)
    posts = [tuple(float(v) for v in p.split(",")) for p in a.net_posts] if a.net_posts else None
    cam, dbg = calibrate(bg, net_post_px=posts)
    cam.save(out)
    cv2.imwrite(str(out.with_suffix(".png")), draw_court(bg, cam))
    print(f"{len(dbg['obj'])} points  rms {cam.reproj_px:.2f}px  net-post check {cam.net_check_px:.1f}px")
    print(f"f {cam.Kmat[0, 0]:.1f}px  fovy {cam.fovy_deg:.1f} deg  camera at {cam.center.round(2)}  -> {out}")


if __name__ == "__main__":
    main()

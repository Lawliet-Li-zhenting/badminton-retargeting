"""Report figures of the game: stills from the recorded matches -> docs/figures/game_*.jpg

    python scripts/make_game_figures.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"

# figure -> (video, [seconds]); several times make a 2-column sheet
SHOTS = {
    "game_title": ("match_right_vs_lin_dan.mp4", [1.6]),
    "game_select": ("match_left_vs_momota.mp4", [5.6]),
    "game_settings": ("match_pad_vs_taufik.mp4", [10.6]),
    "game_walk_on": ("match_right_vs_lin_dan.mp4", [8.9]),
    "game_rally": ("match_right_vs_lin_dan.mp4", [100.0]),
    "game_result": ("match_right_vs_lin_dan.mp4", [388.0]),
    "game_match_sheet": ("match_right_vs_lin_dan.mp4", [36.0, 100.2, 164.0, 180.0, 308.0, 356.0]),
    "game_left_handed": ("match_left_vs_momota.mp4", [50.0, 242.0]),
    "game_ai_exhibition": ("match_ai_exhibition.mp4", [55.0, 83.0]),
    "game_pad": ("match_pad_vs_taufik.mp4", [96.0, 108.0]),
}


def frame_at(cap, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS))))
    ok, f = cap.read()
    return f if ok else None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for name, (video, times) in SHOTS.items():
        path = RES / video
        if not path.exists():
            print("missing", path)
            continue
        cap = cv2.VideoCapture(str(path))
        ims = [f for f in (frame_at(cap, t) for t in times) if f is not None]
        cap.release()
        if not ims:
            continue
        if len(ims) == 1:
            out = ims[0]
        else:
            ims = [cv2.resize(f, (800, 450), interpolation=cv2.INTER_AREA) for f in ims]
            if len(ims) % 2:
                ims.append(np.zeros_like(ims[0]))
            out = np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)])
        cv2.imwrite(str(FIG / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(name, out.shape[1], "x", out.shape[0])


if __name__ == "__main__":
    main()

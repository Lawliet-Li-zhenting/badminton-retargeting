"""Figures for the broadcast retargeting section of the report (docs/figures/broadcast_*.png).

    python scripts/make_broadcast_figures.py --tag standard_video_0_0 --frames 3140 3160 3170 3250 3300 3400
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from badrt.calib import Camera, draw_court  # noqa: E402
from badrt.character import CharacterIndex  # noqa: E402
from badrt.court import build_broadcast_model  # noqa: E402
from retarget_broadcast import game_camera, mj_camera  # noqa: E402

FIG = Path("docs/figures")


DIMS = None


def compare_grid(video, cam, Q, ok, frames, start, path, L=150):
    m = build_broadcast_model(2 * cam.width, 2 * cam.height, dims=DIMS)
    d = mujoco.MjData(m); ci = CharacterIndex(m, "p_")
    r = mujoco.Renderer(m, 2 * cam.height, 2 * cam.width)
    mc, fov = mj_camera(cam); m.vis.global_.fovy = fov
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for f in frames:
        k = f - start
        if not ok[k]:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); _, im = cap.read()
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        c = cam.project(d.site_xpos[ci.kp_sites]).mean(0)
        x0 = int(np.clip(c[0] - L, 0, cam.width - 2 * L)); y0 = int(np.clip(c[1] - L, 0, cam.height - 2 * L))
        crop = cv2.resize(im[y0:y0 + 2 * L, x0:x0 + 2 * L], (300, 300))
        r.update_scene(d, mc); ren = r.render()[:, :, ::-1]
        rc = cv2.resize(ren[2 * y0:2 * (y0 + 2 * L), 2 * x0:2 * (x0 + 2 * L)], (300, 300))
        cv2.putText(crop, f"#{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(np.vstack([crop, rc]))
    cv2.imwrite(str(path), np.hstack(tiles))


def game_and_match(video, cam, Q, ok, frame, start, path):
    W, H = cam.width, cam.height
    m = build_broadcast_model(W, H, dims=DIMS)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    d.qpos[:] = Q[frame - start]; mujoco.mj_forward(m, d)
    mc, fov = mj_camera(cam); m.vis.global_.fovy = fov
    r.update_scene(d, mc); match = r.render()[:, :, ::-1].copy()
    gc, gfov = game_camera(W / H); m.vis.global_.fovy = gfov
    r.update_scene(d, gc); game = r.render()[:, :, ::-1].copy()
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame); _, im = cap.read()
    im = draw_court(im, cam, (0, 0, 255), 1)
    cv2.imwrite(str(path), np.vstack([np.hstack([im, match]), np.hstack([game, np.zeros_like(game)])])[:H * 2, :])


def character_views(path):
    m = build_broadcast_model(560, 720, dims=DIMS)
    d = mujoco.MjData(m); ci = CharacterIndex(m, "p_")
    d.qpos[ci.qadr:ci.qadr + 3] = [0, -4, 0.98]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 560)
    imgs = []
    for az, el in ((270, -5), (0, -5), (120, -15), (90, -5)):
        c = mujoco.MjvCamera(); c.lookat[:] = [0, -4, 0.9]; c.distance = 2.5; c.azimuth = az; c.elevation = el
        r.update_scene(d, c); imgs.append(r.render()[:, :, ::-1])
    opt = mujoco.MjvOption(); opt.sitegroup[5] = 1                              # collision spheres
    c = mujoco.MjvCamera(); c.lookat[:] = [0, -4, 0.9]; c.distance = 2.5; c.azimuth = 120; c.elevation = -15
    r.update_scene(d, c, opt); imgs.append(r.render()[:, :, ::-1])
    cv2.imwrite(str(path), np.hstack(imgs))


def timeline(npz, start, a, b, path, fps=30.0):
    z = np.load(npz)
    c, air = z["contact"][a - start:b - start], z["airborne"][a - start:b - start]
    t = np.arange(a, b) / fps
    fig, ax = plt.subplots(figsize=(11, 2.4))
    names = ["left heel", "right heel", "left toe", "right toe"]
    for i in range(4):
        on = c[:, i]
        ax.fill_between(t, i + 0.1, i + 0.9, where=on, step="mid", color=["C0", "C1", "C0", "C1"][i],
                        alpha=0.85 if i < 2 else 0.5)
    ax.fill_between(t, 4.1, 4.9, where=air, step="mid", color="k")
    ax.set_yticks(np.arange(5) + 0.5); ax.set_yticklabels(names + ["airborne"])
    ax.set_xlabel("time (s)"); ax.set_xlim(t[0], t[-1]); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/videos/standard_video.mp4")
    ap.add_argument("--tag", default="standard_video_0_0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[3140, 3160, 3170, 3250, 3300, 3400])
    ap.add_argument("--game-frame", type=int, default=3250)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    cam = Camera.load(f"assets/calib/{Path(a.video).stem}.json")
    npz = Path("results/broadcast") / f"{a.tag}_qpos.npz"
    z = np.load(npz)
    Q, ok = z["qpos"], z["ok"]
    global DIMS
    DIMS = json.loads((Path("results/broadcast") / f"{a.tag}.json").read_text()).get("character_dims")
    compare_grid(a.video, cam, Q, ok, a.frames, a.start, FIG / "broadcast_compare.png")
    game_and_match(a.video, cam, Q, ok, a.game_frame, a.start, FIG / "broadcast_views.png")
    character_views(FIG / "character.png")
    timeline(npz, a.start, a.frames[0] - 60, a.frames[0] + 240, FIG / "broadcast_contacts.png")
    bg = cv2.imread(f"assets/calib/{Path(a.video).stem}.png")
    if bg is not None:
        cv2.imwrite(str(FIG / "broadcast_calib.png"), bg)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()

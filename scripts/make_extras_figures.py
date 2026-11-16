"""Figures of the wardrobe, the ladder, the menus and the self-play results (docs/figures/extras_*.jpg/png).
The app itself is run without a window (scripted thumbs play), nothing is read from or written to the save.

    python scripts/make_extras_figures.py             # all, about 4 minutes
    python scripts/make_extras_figures.py sheet curve # only these (sheet, menus, match, curve)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import modes, wardrobe as W  # noqa: E402
from badrt import ui as U  # noqa: E402
from badrt.app import App, Settings  # noqa: E402
from badrt.fonts import font  # noqa: E402
from badrt.footwork import yaw_of_quat  # noqa: E402
from badrt.styles import PLAYER_STYLE  # noqa: E402

FIG = ROOT / "docs" / "figures"
MINE = dict(suit="bee", head="crown", racket="pan", back="wings", trail="rainbow", quip="meme", name="n5", body="bighead")


def save(name, img):
    cv2.imwrite(str(FIG / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(name, img.shape[1], "x", img.shape[0])


def figure(st, qpos, size=(300, 400)):
    """A character alone, whole body, seen from the front-left."""
    m, d = U._figure(st, size, qpos, podium=True)
    m.vis.global_.fovy = 36
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 1.08]
    cam.distance, cam.elevation = 4.5, -8
    cam.azimuth = -90 + float(np.degrees(yaw_of_quat(d.qpos[3:7]))) - 28
    r.update_scene(d, cam)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, cam)
    alpha = (r.render()[:, :, 0] >= 0).astype(np.uint8) * 255
    r.close()
    return Image.fromarray(np.dstack([rgb, cv2.GaussianBlur(alpha, (0, 0), 0.8)]), "RGBA")


def sheet(app):
    """The ladder's characters and a few things the player can put together."""
    q = app.attract.opp.ready_q
    looks = [(r["style"].name, r["style"]) for r in modes.LADDER[:-1]]
    for name, o in (("恐龙 + 雪糕筒 + 人字拖", dict(suit="dino", head="cone", racket="slipper", back="tail", face="shades")),
                    ("土豪金 + 皇冠 + 咸鱼", dict(suit="gold", head="crown", racket="fish", back="cape")),
                    ("隐身衣 + 纸袋头 + 气球", dict(suit="ghost", head="bag", back="balloon", racket="glow")),
                    ("大头筋肉 + 绿巨人 + 板砖", dict(body="mega", skin="green", racket="brick", suit="tank", head="mohawk", hair="h4")),
                    ("睡衣 + 兔耳朵 + 棒棒糖", dict(suit="pajama", head="bunny", racket="lolly", face="blush", body="bighead")),
                    ("本命年 + 西瓜头盔 + 大葱", dict(suit="longjohns", head="melon", racket="leek", back="shell")),
                    ("小蜜蜂 + 竹蜻蜓 + 苍蝇拍", dict(suit="bee", head="copter", racket="swatter", body="tiny")),
                    ("校服 + 红鼻子 + 键盘", dict(suit="school", face="nose", racket="keyboard", head="afro", hair="h5"))):
        looks.append((name, replace(PLAYER_STYLE, outfit=dict(W.DEFAULT, **o))))
    cols, cw, ch = 6, 300, 440
    rows = (len(looks) + cols - 1) // cols
    im = Image.new("RGB", (cols * cw, rows * ch), (22, 28, 44))
    d = ImageDraw.Draw(im)
    for i, (name, st) in enumerate(looks):
        x, y = (i % cols) * cw, (i // cols) * ch
        d.rounded_rectangle([x + 8, y + 8, x + cw - 8, y + ch - 8], radius=16, fill=(30, 38, 60))
        f = figure(st, q)
        im.paste(f, (x, y + 4), f)
        d.text((x + cw / 2, y + ch - 28), name, font=font(21, True, True), fill=(255, 210, 60) if i < 9 else (236, 240, 248), anchor="mm")
    save("extras_wardrobe_sheet.jpg", cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))


def keys(app, ks, n=6):
    img = None
    for k in ks:
        app.inputs.append(k)
        img = app.frame(app.fixed_dt)
    for _ in range(n):
        img = app.frame(app.fixed_dt)
    return img


def menus(app):
    save("extras_title.jpg", keys(app, [], 70))
    keys(app, ["tab", "right"])
    save("extras_wardrobe.jpg", keys(app, ["down"] * 3, 12))
    keys(app, ["esc"] + ["down"] * 4 + ["right"] + ["down"] * 9, 4)
    save("extras_wardrobe_racket.jpg", keys(app, [], 14))
    keys(app, ["esc", "esc"])
    save("extras_mode.jpg", keys(app, ["enter", "right", "right"], 6))
    save("extras_ladder.jpg", keys(app, ["enter", "down", "down", "right"], 6))
    keys(app, ["esc", "esc"])


def match(app):
    """A ladder match (rung 6) played by the scripted thumbs: walk-on, a taunt after a point, the result."""
    app.mode, app.stage = "ladder", 5
    app.progress.ladder[:5] = [1] * 5
    app.state = "ladder"
    keys(app, [], 2)
    save("extras_walk_on.jpg", keys(app, ["enter"], 46))
    n = k = 0
    while app.state == "match" and k < 30 * 900:
        img = app.frame(app.fixed_dt)
        k += 1
        s = app.sess
        m = s.match
        if n < 1 and m.state == "point" and m.winner == "p" and s.t - m.t_state > 0.9 and m.taunt(s.t):
            for _ in range(22):
                img = app.frame(app.fixed_dt)
            save("extras_taunt.jpg", img)
            n += 1
        o = m.bubbles.get("o")
        if n == 1 and o and o[1] + 0.4 < s.t < o[2] and m.state == "point" and m.winner == "o":
            save("extras_opponent_line.jpg", img)
            n += 1
    save("extras_result.jpg", keys(app, [], 60))
    print("ladder rung 6:", app.results[-1]["score"], app.results[-1].get("verdict"), app.results[-1].get("coins"))


def curve():
    """Points share of each generation against the hand-written tactics: the per-generation check of the training
    run and the held-out evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Microsoft YaHei", "SimHei"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False
    log = json.loads((ROOT / "results" / "selfplay_log.json").read_text(encoding="utf-8"))
    g = [r["gen"] for r in log["generations"]]
    s = np.array([r["vs_plain"]["share"] for r in log["generations"]])
    e = np.array([r["vs_plain"]["se"] for r in log["generations"]])
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.axhline(0.5, color="#888", lw=1, ls="--")
    ax.errorbar(g, 100 * s, yerr=100 * e, fmt="o-", color="#2a6fdb", capsize=3, label="训练中的检查（14 组对阵 × 2，每代相同）")
    p = ROOT / "results" / "brain_eval.json"
    if p.exists():
        ev = json.loads(p.read_text(encoding="utf-8"))
        gg = sorted(int(k) for k in ev["generations"])
        ax.errorbar([x + 0.12 for x in gg], [100 * ev["generations"][str(k)]["share"] for k in gg],
                    yerr=[100 * ev["generations"][str(k)]["se"] for k in gg], fmt="s", color="#e2552b", capsize=3,
                    label=f"留出评测（{ev['pairs']} 组新对阵 × 2）")
    ax.set_xlabel("自对弈代数")
    ax.set_ylabel("对手写战术的得分率 [%]")
    ax.set_xticks(g)
    ax.set_ylim(40, 80)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "extras_selfplay_curve.png")
    print("extras_selfplay_curve.png")


if __name__ == "__main__":
    only = sys.argv[1:]
    want = lambda k: not only or k in only  # noqa: E731
    if want("sheet") or want("menus") or want("match"):
        app = App(headless=True, save=None, pad_bot=True, settings=Settings(quality=1), outfit=MINE, seed=4)
        if want("sheet"):
            sheet(app)
        if want("menus"):
            menus(app)
        if want("match"):
            match(app)
        if app.sess is not None:
            app.sess.close()
        if app.preview is not None:
            app.preview.close()
    if want("curve"):
        curve()

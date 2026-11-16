"""Report figures of the arcade layer, the campaign and the training drills -> docs/figures/game_*.jpg

    python scripts/make_arcade_figures.py [menus] [play] [cards] [drill]      (default: all four)

The frames come from the app itself, run without a window at a fixed time step with the scripted thumbs
(PadBot) on the pad -- nobody is playing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.app import App, Settings  # noqa: E402
from badrt.modes import DRILL_KEYS  # noqa: E402
from badrt.opponent import PadBot  # noqa: E402
from badrt.styles import ROSTER  # noqa: E402

FIG = ROOT / "docs" / "figures"
Q = 1                                                                # "high"


def run(app, n, keys=()):
    app.inputs += list(keys)
    img = None
    for _ in range(n):
        img = app.frame(app.fixed_dt)
    return img.copy()


def sheet(name, ims, cols=2, cell=(800, 450), crop=None):
    ims = [im if crop is None else im[crop[1]:crop[3], crop[0]:crop[2]] for im in ims]
    ims = [cv2.resize(im, cell, interpolation=cv2.INTER_AREA) for im in ims]
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    out = np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])
    cv2.imwrite(str(FIG / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(name, out.shape[1], "x", out.shape[0], flush=True)


def menus():
    app = App(headless=True, save=None, pad_bot=True, settings=Settings(quality=Q))
    run(app, 20)
    run(app, 2, ["enter"])
    mode = run(app, 2, ["right"])                                    # the mode screen, "campaign" selected
    stages = run(app, 3, ["enter"])
    assert app.state == "stages", app.state
    run(app, 2, ["esc"])
    run(app, 2, ["right"])                                           # (the ladder: scripts/make_extras_figures.py)
    run(app, 2, ["right"])
    drills = run(app, 3, ["enter"])
    assert app.state == "drills", app.state
    run(app, 2, ["esc"])
    for _ in range(3):
        run(app, 2, ["left"])
    run(app, 3, ["enter"])
    assert app.state == "select", app.state
    select = run(app, 3, ["right"] * 7)                              # second page of the roster: the new players
    sheet("game_modes", [mode, stages, drills, select])


def play():
    """A normal controller match (legend difficulty, so that the opponent dives too): the player's dive,
    a jump smash, the opponent's dive."""
    app = App(headless=True, save=None, pad_bot=True, start="match", seed=2,
              settings=Settings(quality=Q, opponent=[s.key for s in ROSTER].index("axelsen"), difficulty=3, points=0))
    m = app.sess.match
    got, pending, seen, marked = {}, [], 0, set()

    def want(kind, t0, offsets):
        if kind not in got:
            got[kind] = [None] * len(offsets)
            pending.extend((kind, k, t0 + o) for k, o in enumerate(offsets))

    for _ in range(30 * 420):
        img = app.frame(app.fixed_dt)
        if app.state == "result":
            break
        t = app.sess.t
        for e in m.events[seen:]:
            if e.get("pad") == "dive" and e.get("ok"):
                want("dive", e["t"], (0.12, 0.3, 0.5, 0.95))
        seen = len(m.events)
        for who, pl in (("p", m.pad), ("o", m.opp)):
            pn = pl.plan
            if pn is None or id(pn) in marked:
                continue
            marked.add(id(pn))
            if who == "p" and pn.shot == "smash" and pn.info.get("jump", 0.0) > 0.2:
                want("jump", pn.t_hit, (-0.3, -0.03))
            if who == "o" and pn.shot == "dive":
                want("odive", pn.t_hit, (-0.05, 0.3))
        for item in list(pending):
            kind, k, tt = item
            if t >= tt:
                pending.remove(item)
                got[kind][k] = img.copy()
        if all(k in got for k in ("dive", "jump", "odive")) and not pending:
            break
    for kind in ("dive", "jump", "odive"):
        print(kind, "captured" if kind in got and all(f is not None for f in got[kind]) else "MISSING")
    if "dive" in got:
        sheet("game_dive", got["dive"])
    if "jump" in got and "odive" in got:
        sheet("game_jump_dive", got["jump"] + got["odive"])


def cards():
    """The thumbs stroll about for 40 s after every point: shot clock, yellow, red (a point), black (the match)."""
    app = App(headless=True, save=None, pad_bot=True, start="match", seed=2, settings=Settings(quality=Q, opponent=0, points=0))
    m = app.sess.match
    m.pad_bot = PadBot(np.random.default_rng(5), dawdle=40.0)
    got, pending, seen = {}, [], 0
    for _ in range(30 * 400):
        img = app.frame(app.fixed_dt)
        if app.state == "result":
            got["result"] = run(app, 30)
            break
        t = app.sess.t
        for e in m.events[seen:]:
            if "card" in e and e["card"] not in got:
                pending.append((e["card"], e["t"] + 0.9))
        seen = len(m.events)
        for item in list(pending):
            if t >= item[1]:
                pending.remove(item)
                got[item[0]] = img.copy()
        if "clock" not in got and m.umpire.waited(t) > 12.5:
            got["clock"] = img.copy()
    print("cards", m.stats().get("cards"), "forfeit", m.forfeit, sorted(got))
    sheet("game_umpire", [got[k] for k in ("clock", "yellow", "red", "result") if k in got])


def drill():
    ims = []
    for key, at in (("smash", (6.5, 8.2)), ("net", (7.0,))):
        app = App(headless=True, save=None, pad_bot=True, settings=Settings(quality=Q), start="match", mode="training",
                  drill=DRILL_KEYS.index(key))
        want = list(at)
        for i in range(30 * 200):
            img = app.frame(app.fixed_dt)
            if want and i / 30.0 >= want[0]:
                want.pop(0)
                ims.append(img.copy())
            if app.state == "result":
                if key == "smash":
                    ims.append(run(app, 20))
                break
            if not want and key != "smash":
                break
    sheet("game_training", ims)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["menus", "play", "cards", "drill"]
    for name in todo:
        dict(menus=menus, play=play, cards=cards, drill=drill)[name]()

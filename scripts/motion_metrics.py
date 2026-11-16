"""How the players move: metrics of the body motion and the footwork figures of the report.

    python scripts/motion_metrics.py                 # -> results/motion_metrics.json
    python scripts/motion_metrics.py --figures       # + docs/figures/game_footwork_*.jpg

Players measured: both sides of an AI exhibition, the controller player (played by the scripted
thumbs) and the recorded stand-in of the camera player (carried by the footwork assist).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.live import DEMO_TRACKS  # noqa: E402
from badrt.character import KP  # noqa: E402
from badrt.game import Assist, Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.opponent import PadBot  # noqa: E402
from badrt.recorded import Director, PlayerTrack  # noqa: E402
from badrt.styles import get_style  # noqa: E402
from badrt.viz import SceneRenderer  # noqa: E402

LEG_HINGES = [35, 36, 37, 42, 43, 44]            # knees, ankles, subtalar joints


def make(kind, lib, seed=0):
    """(match, per-frame pose source or None, fps)."""
    if kind == "ai":
        return Match(lib, get_style("axelsen"), player="ai", player_style=get_style("lee_chong_wei"), arena=False,
                     seed=seed), None, 60.0
    if kind == "pad":
        return Match(lib, get_style("taufik"), player="pad", difficulty="normal", arena=False, seed=seed, points_to_win=21,
                     pad_bot=PadBot(np.random.default_rng(100 + seed))), None, 60.0
    path, start, dur = DEMO_TRACKS["right"]
    d = Director(PlayerTrack(ROOT / path, start=start, duration=dur))
    return Match(lib, get_style("lin_dan"), player="human", player_handed="right", assist=Assist.camera(), director=d,
                 difficulty="easy", arena=False, seed=seed, points_to_win=21), d, 30.0


def run(kind, lib, seconds, on_frame=None):
    m, director, fps = make(kind, lib)
    rec = {w: [] for w in "po"}
    k = 0
    while k / fps < seconds and m.state != "over":
        t = k / fps
        k += 1
        m.update(t, director.qpos(t) if director is not None else None)
        for w, ci in (("p", m.pi), ("o", m.oi)):
            rec[w].append((m.d.qpos[ci.qslice].copy(), m.d.site_xpos[ci.kp_sites].copy(), m.d.site_xpos[ci.sole_sites].copy()))
        if on_frame is not None:
            on_frame(m, t)
    return m, rec, fps


def metrics(frames, hand, fps, events=(), who=None):
    h = 1.0 / fps
    Q, K, S = (np.array([f[i] for f in frames]) for i in range(3))
    off = "l" if hand == "r" else "r"
    wr, sh, head = K[:, KP[off + "_wrist"]], K[:, KP[off + "_shoulder"]], K[:, KP["nose"]]
    step = np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1)
    v = step / h
    a = np.linalg.norm(np.diff(Q[:, :2], n=2, axis=0), axis=1) / h ** 2
    dj = np.abs(np.diff(Q[:, LEG_HINGES], axis=0)).max(1) / h      # fastest leg hinge [rad/s]
    out = dict(seconds=round(len(frames) * h, 1), fps=fps,
               off_wrist_above_shoulder_pct=round(100 * float(np.mean(wr[:, 2] > sh[:, 2])), 1),
               off_wrist_above_head_pct=round(100 * float(np.mean(wr[:, 2] > head[:, 2])), 1),
               root_speed_mean=round(float(v.mean()), 2), root_speed_p95=round(float(np.percentile(v, 95)), 2),
               root_speed_max=round(float(v.max()), 2), root_accel_p95=round(float(np.percentile(a, 95)), 1),
               root_steps_over_25cm=int(np.sum(step > 0.25)), root_step_max_m=round(float(step.max()), 3),
               leg_hinge_speed_p95=round(float(np.percentile(dj, 95)), 1),
               leg_hinge_speed_p99=round(float(np.percentile(dj, 99)), 1), feet={})
    # how much of the raised free hand belongs to an overhead stroke (pointing at the shuttle before the contact)
    up = wr[:, 2] > sh[:, 2]
    oh = np.array([e["t"] for e in events if e.get("who") == who and "shot" in e and e["shot"] != "deflect"
                   and e["pos"][2] >= 1.9])
    if len(oh) and up.any():
        t = np.arange(len(frames)) * h
        near = ((oh[None, :] - t[:, None] > -0.15) & (oh[None, :] - t[:, None] < 0.8)).any(1)
        out["raised_frames_before_an_overhead_contact_pct"] = round(100 * float((up & near).sum() / up.sum()), 1)
    still = v < 0.2
    out["pelvis_still_pct"] = round(100 * float(still.mean()), 1)
    for f, name in ((slice(0, 4), "left"), (slice(4, 8), "right")):
        c, low = S[:, f].mean(1), S[:, f, 2].min(1)
        sp = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1) / h
        g = low[1:] < 0.03
        g1 = low[1:] < 0.01
        out["feet"][name] = dict(on_floor_pct=round(100 * float(g.mean()), 1),
                                 on_floor_pct_pelvis_still=round(100 * float(g[still].mean()), 1) if still.any() else 0.0,
                                 on_floor_pct_pelvis_moving=round(100 * float(g[~still].mean()), 1) if (~still).any() else 0.0,
                                 skate_mean_1cm=round(float(sp[g1].mean()), 3) if g1.any() else 0.0,
                                 skate_p95_1cm=round(float(np.percentile(sp[g1], 95)), 3) if g1.any() else 0.0,
                                 skate_mean=round(float(sp[g].mean()), 2) if g.any() else 0.0,
                                 skate_p95=round(float(np.percentile(sp[g], 95)), 2) if g.any() else 0.0,
                                 lowest_sole_mm=round(1000 * float(low.min()), 1))
    return out


# ------------------------------------------------------------------------------------- figures
def pick_strokes(m, rec, fps):
    """Times of two strokes of the far player: a lunge after a long run, an overhead with a jump."""
    root = np.array([f[0][:2] for f in rec["o"]])
    best = {}
    for e in m.events:
        if e.get("who") != "o" or "shot" not in e or e.get("serve") or e["shot"] == "deflect":
            continue
        k1 = int(round(e["t"] * fps))
        k0 = max(k1 - int(1.2 * fps), 0)
        if k1 >= len(root) - int(0.6 * fps) or k0 == 0:
            continue
        travel = float(np.linalg.norm(root[k1] - root[k0]))
        z = e["pos"][2]
        if z < 1.0 and travel > best.get("lunge", (0, 0))[0]:
            best["lunge"] = (travel, e["t"], e["shot"])
        if z > 2.8 and travel > 0.8 and z > best.get("overhead", (0, 0))[0]:
            best["overhead"] = (z, e["t"], e["shot"])
    return best


def figures(lib, seconds, fig_dir):
    m, rec, fps = run("ai", lib, seconds)
    picks = pick_strokes(m, rec, fps)
    offsets = dict(lunge=[-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0, 0.25],
                   overhead=[-1.1, -0.8, -0.55, -0.35, -0.2, -0.08, 0.05, 0.3])
    root = np.array([f[0][:2] for f in rec["o"]])
    want = []
    for name, (_, t_hit, shot) in picks.items():
        k1, k0 = int(round(t_hit * fps)), int(round((t_hit + offsets[name][0]) * fps))
        look = 0.5 * (root[k0] + root[k1])
        want += [(t_hit + o, name, o, look, shot) for o in offsets[name]]
        print(f"{name}: {shot} at {t_hit:.2f} s")
    want.sort(key=lambda w: w[0])
    sr, ims = None, {n: [] for n in picks}

    def on_frame(mm, t):
        nonlocal sr
        if sr is None:
            sr = SceneRenderer(mm.m, 640, 520)
        while want and t >= want[0][0] - 1e-9:
            _, name, o, look, shot = want.pop(0)
            sr.set_view(dict(lookat=[look[0], look[1], 1.05], distance=5.4, azimuth=205.0, elevation=-10.0))
            im = sr.render(mm.d)
            cv2.putText(im, "%+.2f s" % o, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            ims[name].append(cv2.resize(im, (480, 390), interpolation=cv2.INTER_AREA))

    run("ai", lib, max(w[0] for w in want) + 0.1, on_frame)
    for name, fr in ims.items():
        if len(fr) == 8:
            cv2.imwrite(str(fig_dir / f"game_footwork_{name}.jpg"), np.vstack([np.hstack(fr[:4]), np.hstack(fr[4:])]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("game_footwork_" + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "motion_metrics.json"))
    args = ap.parse_args()
    lib = load_library(str(ROOT / "assets" / "stroke_library.npz"))
    res = {}
    for kind, players in (("ai", "po"), ("pad", "p"), ("recorded", "p")):
        m, rec, fps = run(kind, lib, args.seconds)
        for w in players:
            ci = m.pi if w == "p" else m.oi
            key = {"ai": "ai_near" if w == "p" else "ai_far", "pad": "controller", "recorded": "recorded"}[kind]
            res[key] = metrics(rec[w], ci.hand, fps, m.events, w)
            print(key, json.dumps(res[key]))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if args.figures:
        figures(lib, 70.0, ROOT / "docs" / "figures")


if __name__ == "__main__":
    main()

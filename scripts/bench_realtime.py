"""Latency of the live loop on a video that plays the part of the webcam.

    python scripts/bench_realtime.py data/videos/N9jukZlmkns.mp4 --start 840 --frames 900
    python scripts/bench_realtime.py data/videos/N9jukZlmkns.mp4 --start 840 --frames 900 --threaded

Default: every stage in sequence on each video frame (decode, pose, source, retarget, match, render,
HUD) -> per-stage milliseconds. --threaded: the game's own arrangement (live.LiveInput: capture and
pose in a worker thread paced at the video rate) -> display fps, pose rate, age of the pose on screen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import ui as U  # noqa: E402
from badrt.live import LiveInput  # noqa: E402
from badrt.character import MP_IDS, mirror_qpos  # noqa: E402
from badrt.config import RetargetConfig, SourceConfig  # noqa: E402
from badrt.court import game_view  # noqa: E402
from badrt.game import Assist, Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.pose import PoseEstimator  # noqa: E402
from badrt.retarget import BodyRetargeter  # noqa: E402
from badrt.source import SourceBuilder  # noqa: E402
from badrt.styles import get_style  # noqa: E402
from badrt.viz import SceneRenderer  # noqa: E402


def make_match(args):
    m = Match(load_library(ROOT / "assets" / "stroke_library.npz"), get_style(args.opponent), player="human",
              player_handed=args.handedness, assist=Assist.camera(), difficulty=args.difficulty,
              width=args.size[0], height=args.size[1], arena=args.quality)
    sr = SceneRenderer(m.m, *args.size)
    sr.set_view(game_view(args.size[0] / args.size[1]))
    names = {"p": "你", "o": m.opp_style.name}
    colors = {"p": U.BLUE, "o": U.RED}
    return m, sr, U.UI(*args.size), names, colors


def stats(v, skip=30):
    v = np.asarray(v[skip:])
    return dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                p99=float(np.percentile(v, 99)), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--model", default="lite", choices=["lite", "full", "heavy"])
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--opponent", default="lin_dan")
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--quality", default="high", choices=["fast", "high", "ultra"])
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--out", default=str(ROOT / "results" / "bench_realtime.json"))
    ap.add_argument("--threaded", action="store_true", help="pose in a worker thread, as in the game")
    args = ap.parse_args()
    if args.threaded:
        return bench_threaded(args)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    est = PoseEstimator(args.model, num_poses=1)
    rt = BodyRetargeter(RetargetConfig())
    src = SourceBuilder(rt.neutral_kp, SourceConfig(court_gain=(2.0, 2.0), handedness=args.handedness), origin=(0.0, -3.6))
    match, sr, ui, names, colors = make_match(args)
    stages = {k: [] for k in ("decode", "pose", "source", "retarget", "match", "render", "hud", "total")}
    for k in range(args.frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        now = k / fps
        t1 = time.perf_counter()
        pf = est(frame, now)
        t2 = time.perf_counter()
        q, kp = None, None
        t3 = t4 = t2
        if pf is not None:
            kp = pf.image[MP_IDS, :2]
            S, vis, info = src(pf)
            t3 = t4 = time.perf_counter()
            if S is not None:
                q = rt.step(S, vis, 1.0 / fps, info["contact"])["qpos"]
                if args.handedness == "left":
                    q = mirror_qpos(q)
                t4 = time.perf_counter()
        match.update(now, q)
        t5 = time.perf_counter()
        img = sr.render(match.d, decor=U.scene_fx(match, now, colors))
        t6 = time.perf_counter()
        ui.hud(img, match, now, names, colors, inset=dict(frame=frame, kp2d=kp, label="摄像头", ok=pf is not None, mirror=True))
        t7 = time.perf_counter()
        for name, a, b in (("decode", t0, t1), ("pose", t1, t2), ("source", t2, t3), ("retarget", t3, t4), ("match", t4, t5),
                           ("render", t5, t6), ("hud", t6, t7), ("total", t0, t7)):
            stages[name].append((b - a) * 1e3)
    res = {name: stats(v) for name, v in stages.items()}
    res.update(fps_equiv=1000.0 / res["total"]["mean"], frames=len(stages["total"]), model=args.model, size=args.size,
               quality=args.quality, score=match.score, player_hits=match.stats()["p"]["hits"],
               opponent_plan_ms=stats(match.opp.stats["plan_ms"], 0) if match.opp.stats["plan_ms"] else None)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for name in stages:
        r = res[name]
        print(f"{name:9s} mean {r['mean']:6.2f}  p50 {r['p50']:6.2f}  p95 {r['p95']:6.2f}  p99 {r['p99']:6.2f}  max {r['max']:7.2f} ms")
    print(f"=> {res['fps_equiv']:.1f} fps   score {match.score}  player hits {res['player_hits']}")


def bench_threaded(args):
    """Video paced at its native rate; reports display fps, pose rate and the age of the pose on screen."""
    match, sr, ui, names, colors = make_match(args)
    live = LiveInput(args.video, args.handedness, args.model, start=args.start)
    fps = live.worker.video_fps
    t0 = live.worker.t0
    loop_ms, ages, n_pose = [], [], 0
    duration = args.frames / fps
    t_game, t_prev = 0.0, time.perf_counter()
    while time.perf_counter() - t0 < duration and not live.finished:
        ta = time.perf_counter()
        t_game += min(ta - t_prev, 0.05)                           # the game clock never jumps more than 50 ms
        t_prev = ta
        q = live.qpos(t_game)
        if q is not None:
            n_pose += 1
        match.update(t_game, q)
        img = sr.render(match.d, decor=U.scene_fx(match, t_game, colors))
        ui.hud(img, match, t_game, names, colors, inset=live.inset(t_game))
        if q is not None:
            ages.append((time.perf_counter() - t0 - live.t_prev) * 1e3)     # capture -> on screen
        loop_ms.append((time.perf_counter() - ta) * 1e3)
    live.close()
    el = time.perf_counter() - t0
    res = dict(threaded=True, display_fps=len(loop_ms) / el, pose_fps=n_pose / el, video_fps=fps,
               loop_ms_p50=float(np.median(loop_ms)), loop_ms_p95=float(np.percentile(loop_ms, 95)),
               loop_ms_max=float(np.max(loop_ms)), game_clock_ratio=t_game / el,
               capture_to_screen_ms_p50=float(np.median(ages)), capture_to_screen_ms_p95=float(np.percentile(ages, 95)),
               model=args.model, size=args.size, quality=args.quality, score=match.score,
               player_hits=match.stats()["p"]["hits"], assisted=match.stats()["p"]["assisted"])
    Path(args.out).with_name(Path(args.out).stem + "_threaded.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

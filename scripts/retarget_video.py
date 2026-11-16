"""Retarget the player in a video onto the virtual character.

    python scripts/retarget_video.py data/videos/S6BumK0A5TM.mp4 --tag footwork --render
Outputs results/<tag>/: pose.npz (cached landmarks), retarget.npz, summary.json,
side_by_side.mp4 (with --render) and a few PNG frames.
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

from badrt.character import BONE_DRAW, KP, MP_IDS, SOLE_KP  # noqa: E402
from badrt.config import RetargetConfig, SourceConfig  # noqa: E402
from badrt.pose import PoseEstimator, PoseFrame, draw_pose  # noqa: E402
from badrt.retarget import BodyRetargeter  # noqa: E402
from badrt.source import SourceBuilder  # noqa: E402


def run_pose(video, start, max_frames, model, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return [None if not ok else PoseFrame(w, im, v, int(z["width"]), int(z["height"]), t)
                for ok, w, im, v, t in zip(z["ok"], z["world"], z["image"], z["vis"], z["t"])], float(z["fps"]), z["lat"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    est = PoseEstimator(model)
    frames, lat = [], []
    k = 0
    while max_frames <= 0 or k < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        pf = est(img, (start + k) / fps)
        frames.append(pf)
        lat.append(pf.latency_ms if pf else np.nan)
        k += 1
        if k % 200 == 0:
            print(f"  pose {k} frames")
    est.close()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok = np.array([f is not None for f in frames])
    z = lambda attr, shape: np.array([getattr(f, attr) if f else np.zeros(shape) for f in frames])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ok=ok, world=z("world", (33, 3)), image=z("image", (33, 3)), vis=z("vis", 33),
                        t=np.array([(start + i) / fps for i in range(len(frames))]), fps=fps, width=w, height=h,
                        lat=np.array(lat))
    return frames, fps, np.array(lat)


def detect_handedness(frames, fps):
    """The racket hand is the wrist with the larger peak speed (hip-centred world landmarks)."""
    sp = {15: [], 16: []}
    prev = None
    for f in frames:
        if f is None:
            prev = None
            continue
        if prev is not None:
            for j in sp:
                sp[j].append(np.linalg.norm(f.world[j] - prev.world[j]) * fps)
        prev = f
    if not sp[15]:
        return "right", {}
    p = {j: float(np.percentile(v, 97)) for j, v in sp.items()}
    return ("left" if p[15] > 1.15 * p[16] else "right"), {"left_wrist_p97": p[15], "right_wrist_p97": p[16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tag")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--handedness", default="auto", choices=["auto", "right", "left"])
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--no-translation", action="store_true")
    ap.add_argument("--translation", default="scale", choices=["ground_ray", "scale"])
    ap.add_argument("--real-fps", type=float, default=0, help="true capture rate of slow-motion footage")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--png", type=int, nargs="*", default=[])
    args = ap.parse_args()

    video = Path(args.video)
    tag = args.tag or video.stem
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames, fps, pose_lat = run_pose(video, args.start, args.max_frames, args.model,
                                     out / f"pose_{args.model}_{args.start}_{args.max_frames}.npz")
    print(f"pose: {len(frames)} frames, {sum(f is not None for f in frames)} detected, {time.time() - t0:.1f}s")
    video_fps = fps
    if args.real_fps > 0:
        fps = args.real_fps
        for i, f in enumerate(frames):
            if f is not None:
                f.t = (args.start + i) / fps
    hand, hand_info = (args.handedness, {}) if args.handedness != "auto" else detect_handedness(frames, fps)
    print("handedness:", hand, hand_info)

    rcfg = RetargetConfig.preset(args.preset)
    scfg = SourceConfig(hfov_deg=args.hfov, handedness=hand, track_translation=not args.no_translation,
                        translation_mode=args.translation)
    rt = BodyRetargeter(rcfg)
    src = SourceBuilder(rt.neutral_kp, scfg)

    Q, S_all, V_all, ok_all, lat, errs, bone_errs, rack, sole, scale = [], [], [], [], [], [], [], [], [], []
    bone_errs_all, lifts, air, locks = [], [], [], []
    rejected = 0
    for f in frames:
        S = vis = info = None
        if f is not None:
            S, vis, info = src(f)
            rejected += int(S is None)
        if S is None:
            ok_all.append(False)
            Q.append(Q[-1] if Q else rt.q_rest.copy()); S_all.append(S_all[-1] if S_all else rt.neutral_kp)
            V_all.append(np.zeros(23)); rack.append(rack[-1] if rack else np.zeros(3))
            sole.append(sole[-1] if sole else np.zeros((8, 3)))
            continue
        res = rt.step(S, vis, 1.0 / fps, info["contact"])
        locks.append(res["locked"])
        ev = rt.evaluate(res["qpos"], S, vis)
        ok_all.append(True)
        Q.append(res["qpos"]); S_all.append(S); V_all.append(vis); lat.append(res["latency_ms"])
        good = vis > 0.5
        errs.append(ev["kp_err"][good].mean() if good.any() else np.nan)
        bone_errs.append(ev["bone_err_deg"].mean())
        bone_errs_all.append(ev["bone_err_deg"])
        rack.append(ev["racket"]); sole.append(ev["sole"]); scale.append(info["scale"])
        lifts.append(info["lift"]); air.append(info["airborne"])
    Q, S_all, V_all, rack, sole = map(np.array, (Q, S_all, V_all, rack, sole))
    ok_all = np.array(ok_all)

    # quality metrics
    lat = np.array(lat)
    rv = np.linalg.norm(np.diff(rack, axis=0), axis=1) * fps
    rv[~(ok_all[1:] & ok_all[:-1])] = np.nan
    both_ok = (ok_all[1:] & ok_all[:-1])[:, None]
    soles = sole                                            # (T, 8, 3) character sole corners
    contact = soles[:, :, 2] < 0.02
    slide = np.linalg.norm(np.diff(soles[:, :, :2], axis=0), axis=2) * fps
    cmask = contact[1:] & contact[:-1] & both_ok
    toes = [KP["l_toe"], KP["r_toe"]]
    src_soles = S_all[:, toes]
    src_slide = np.linalg.norm(np.diff(src_soles[:, :, :2], axis=0), axis=2) * fps
    src_cmask = (src_soles[1:, :, 2] < 0.02) & (src_soles[:-1, :, 2] < 0.02) & both_ok
    per_bone = np.mean(bone_errs_all, axis=0)
    root_v = np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1)[both_ok[:, 0]] * fps
    hinge = Q[:, rt.hinge_qadr]
    jit = np.abs(np.diff(hinge, 2, axis=0)).mean(1) * fps * fps
    summary = dict(
        video=str(video), tag=tag, fps=fps, video_fps=video_fps, frames=len(frames),
        detected=int(sum(f is not None for f in frames)), rejected_implausible=rejected, used=int(ok_all.sum()),
        handedness=hand, handedness_info=hand_info, preset=args.preset, config=rcfg.to_dict(), source=scfg.to_dict(),
        pose_latency_ms=dict(mean=float(np.nanmean(pose_lat)), p50=float(np.nanpercentile(pose_lat, 50)),
                             p99=float(np.nanpercentile(pose_lat, 99))),
        retarget_latency_ms=dict(mean=float(lat.mean()), p50=float(np.percentile(lat, 50)),
                                 p99=float(np.percentile(lat, 99)), max=float(lat.max()), first=float(lat[0])),
        keypoint_error_mm=dict(mean=float(np.nanmean(errs) * 1e3), p95=float(np.nanpercentile(errs, 95) * 1e3)),
        bone_direction_error_deg=dict(mean=float(np.mean(bone_errs)), p95=float(np.percentile(bone_errs, 95))),
        min_sole_height_mm=float(soles[ok_all][:, :, 2].min() * 1e3),
        foot_skating_mps=float(slide[cmask].mean()) if cmask.any() else None,
        source_foot_skating_mps=float(src_slide[src_cmask].mean()) if src_cmask.any() else None,
        source_lowest_sole_mm=dict(p5=float(np.percentile(src_soles[ok_all][:, :, 2].min(1), 5) * 1e3),
                                   p50=float(np.percentile(src_soles[ok_all][:, :, 2].min(1), 50) * 1e3)),
        bone_error_per_bone_deg=[round(float(x), 1) for x in per_bone],
        hinge_jitter_rad_s2=float(np.nanmean(jit)),
        racket_head_speed_mps=dict(p95=float(np.nanpercentile(rv, 95)), max=float(np.nanmax(rv))),
        scale=float(np.median(scale)),
        root_speed_mps=dict(p50=float(np.percentile(root_v, 50)), p95=float(np.percentile(root_v, 95)),
                            max=float(root_v.max())),
        max_lift_m=float(np.max(lifts)), airborne_frames=int(np.sum(air)),
        locked_sole_points_mean=float(np.mean(locks)),
    )
    plot_root(Q, ok_all, fps, out / "root_track.png")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "source")}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "retarget.npz", qpos=Q, targets=S_all, vis=V_all, ok=ok_all, fps=fps,
                        racket=rack, racket_speed=np.r_[np.nan, rv], handedness=hand)

    if args.render or args.png:
        render(video, args.start, frames, Q, S_all, ok_all, rt, out, video_fps, args.render, set(args.png))


def plot_root(Q, ok, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(Q)) / fps
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, n in enumerate("xyz"):
        ax.plot(t, Q[:, i], label=f"pelvis {n}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("m"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def render(video, start, frames, Q, S_all, ok_all, rt, out, fps, write_video, png_frames):
    import mujoco
    from badrt.viz import SceneRenderer
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    H = 480
    sr = SceneRenderer(rt.m, 640, H)
    writer = None
    lines_idx = [(KP[a], KP[b]) for a, b in BONE_DRAW]
    for k in range(len(Q)):
        ok, img = cap.read()
        if not ok:
            break
        if frames[k] is not None:
            draw_pose(img, frames[k])
        img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
        rt.d.qpos[:] = Q[k]
        mujoco.mj_kinematics(rt.m, rt.d)
        sr.follow(rt.d.qpos[:3] * [1, 1, 0] + [0, 0, 1.0])
        S = S_all[k]
        rimg = sr.render(rt.d, points=S if ok_all[k] else None,
                         lines=[(S[a], S[b]) for a, b in lines_idx] if ok_all[k] else None)
        frame = np.hstack([img, rimg])
        cv2.putText(frame, f"frame {start + k}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if write_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out / "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if (start + k) in png_frames:
            cv2.imwrite(str(out / f"frame_{start + k:05d}.png"), frame)
    if writer:
        writer.release()


if __name__ == "__main__":
    main()

"""Train the pose -> racket direction prior (badrt.racket_prior) on self-labelled broadcast data and
evaluate it on held-out frames.

    python scripts/train_racket_prior.py [--videos standard_video denmark2025_asy]
Needs results/racket_dataset/<stem>_det.npz (build_racket_dataset.py) and the pose cache.
Hold-out: standard_video frames 3000-3900 (the retargeting test segment) +-60 frames margin.
Evaluation (held-out detected frames, treating the racket as unseen): 2D angle between the
projected predicted racket and the detected one, for the prior and simple baselines.
-> assets/racket_prior.npz, results/racket_prior/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.broadcast import reconstruct  # noqa: E402
from badrt.calib import Camera  # noqa: E402
from badrt.racket import R_WRIST, TIP_DIST  # noqa: E402
from badrt.racket_prior import (R_EL, R_INDEX, R_PINKY, KnnPrior, _unit, fit_em, front_init,  # noqa: E402
                                lift_candidates, pose_features, to_body, to_world)

POSE = {"standard_video": "results/broadcast/standard_video_0_0_pose.npz",
        "denmark2025_asy": "results/broadcast/denmark2025_asy_18000_10800_pose.npz"}
HOLD = {"standard_video": (3000 - 60, 3900 + 60)}


def load_video(stem, log=True):
    cam = Camera.load(f"assets/calib/{stem}.json")
    cache = Path(f"results/racket_dataset/{stem}_rec.npz")
    data = dict(np.load(POSE[stem]))
    det = dict(np.load(f"results/racket_dataset/{stem}_det.npz"))
    if cache.exists():
        z = np.load(cache)
        class R:  # noqa: E701
            pass
        rec = R(); rec.P, rec.ok, rec.contact, rec.airborne, rec.fps = z["P"], z["ok"], z["contact"], z["airborne"], float(z["fps"])
    else:
        rec = reconstruct(cam, data)
        np.savez_compressed(cache, P=rec.P, ok=rec.ok, contact=rec.contact, airborne=rec.airborne, fps=rec.fps)
    if log:
        print(f"{stem}: {len(rec.ok)} frames, ok {rec.ok.mean():.2f}, detections {np.mean(det['hand'] == 0):.2f}")
    return cam, data, det, rec


def angle2d(a, b):
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def eval_2d(cam, rec, det, idx, d_world):
    wr = rec.P[idx, R_WRIST]
    w2 = cam.project(wr)
    t2 = cam.project(wr + TIP_DIST * d_world)
    return angle2d(t2 - w2, det["uv"][idx] - w2)


def summary(e):
    return dict(median=float(np.median(e)), mean=float(np.mean(e)), lt20=float(np.mean(e < 20)),
                lt45=float(np.mean(e < 45)), n=int(len(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["standard_video"])
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="assets/racket_prior.npz")
    ap.add_argument("--test-videos", nargs="*", default=[], help="videos used only for testing (cross-match)")
    ap.add_argument("--eval-out", default="results/racket_prior/eval.json")
    a = ap.parse_args()

    tr = dict(X=[], t=[], s=[], cand=[], conf=[], init=[])
    te = []
    for si, stem in enumerate(a.videos + a.test_videos):
        cam, data, det, rec = load_video(stem)
        X, Rb, pel = pose_features(rec.P, rec.fps)
        idx, cand, ncand, conf = lift_candidates(cam, rec, det)
        cb = np.einsum("nij,nkj->nki", Rb[idx], cand)                 # body frame
        init = front_init(rec, idx, cand)
        lo, hi = HOLD.get(stem, (-1, -1))
        hold = ((idx >= lo) & (idx < hi)) | (stem in a.test_videos)
        if hold.any():
            te.append(dict(stem=stem, cam=cam, rec=rec, det=det, idx=idx[hold], X=X[idx[hold]], Rb=Rb[idx[hold]],
                           cand=cand[hold], init=init[hold]))
        keep = ~hold
        tr["X"].append(X[idx[keep]]); tr["t"].append(idx[keep]); tr["s"].append(np.full(keep.sum(), si))
        tr["cand"].append(cb[keep]); tr["conf"].append(conf[keep]); tr["init"].append(init[keep])
        print(f"  train samples {keep.sum()}, held out {hold.sum()}, two candidates {np.mean(ncand == 2):.2f}")
    tr = {k: np.concatenate(v) for k, v in tr.items()}

    print("EM (front-rule init)")
    prior, choice, hist = fit_em(tr["X"], tr["t"], tr["s"], tr["cand"], tr["conf"], n_iter=a.iters, k=a.k,
                                 init=tr["init"])
    flipped = float(np.mean(choice != tr["init"]))
    print(f"  labels changed vs front rule: {flipped:.3f}")
    prior0 = KnnPrior(tr["X"], tr["cand"][np.arange(len(tr["X"])), tr["init"]], np.clip(tr["conf"], 0.1, 1), k=a.k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prior.save(a.out)

    res = dict(train_n=int(len(tr["X"])), em=hist, flipped_vs_front=flipped)
    mean_dir = _unit(prior.Y.mean(0))
    for T in te:
        cam, rec, det, idx = T["cam"], T["rec"], T["det"], T["idx"]
        P = rec.P[idx]
        preds = {
            "forearm": _unit(P[:, R_WRIST] - P[:, R_EL]),
            "hand": _unit(P[:, [R_PINKY, R_INDEX]].mean(1) - P[:, R_WRIST]),
            "mean_body_dir": to_world(T["Rb"], np.broadcast_to(mean_dir, (len(idx), 3))),
        }
        for name, pr in (("knn_front_labels", prior0), ("knn_em", prior)):
            d, c, _ = pr.predict(T["X"])
            preds[name] = to_world(T["Rb"], d)
            if name == "knn_em":
                conf_te = c
        r = {name: summary(eval_2d(cam, rec, det, idx, d)) for name, d in preds.items()}
        e = eval_2d(cam, rec, det, idx, preds["knn_em"])
        hiC = conf_te > np.median(conf_te)
        r["knn_em_confident_half"] = summary(e[hiC])
        # lift choice on held-out frames: prior vs front rule (agreement and 1-frame flips)
        cand = T["cand"]
        cb = np.einsum("nij,nkj->nki", T["Rb"], cand)
        dp, _, _ = prior.predict(T["X"])
        ch = np.argmax(np.einsum("nj,nkj->nk", dp, cb), 1)
        r["lift_agree_front_rule"] = float(np.mean(ch == T["init"]))

        def flips(c):
            u = cand[np.arange(len(c)), c]
            cons = np.diff(idx) == 1
            return float(np.mean((u[1:] * u[:-1]).sum(1)[cons] < 0))
        r["lift_flip_rate"] = dict(front_rule=flips(T["init"]), prior=flips(ch))
        res[T["stem"]] = r
        print(T["stem"], "held-out 2D racket direction error (deg):")
        for name, v in r.items():
            if isinstance(v, dict) and "median" in v:
                print(f"  {name:24s} median {v['median']:5.1f}  <20deg {v['lt20']:.2f}  <45deg {v['lt45']:.2f}  n {v['n']}")
        print("  lift agree with front rule", r["lift_agree_front_rule"], " flips", r["lift_flip_rate"])
    Path(a.eval_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.eval_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

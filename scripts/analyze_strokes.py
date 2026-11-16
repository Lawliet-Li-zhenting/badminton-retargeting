"""Stroke-type analysis of the camera-side player: body posture and racket relative to the body at
the hitting moments, per court zone (front / mid / rear) and side (forehand / backhand / overhead).

    python scripts/analyze_strokes.py [--videos standard_video denmark2025_asy]
Racket direction per frame: detected frames use the lift candidate closest to the learned prior
(assets/racket_prior.npz), undetected frames the prior itself.
Hit moments: peaks of the racket-tip speed (>= 0.6 s apart) inside valid main-camera runs.
-> results/strokes/{stats.json, skeletons.png, racket_dirs.png, court_map.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.racket import R_WRIST, TIP_DIST  # noqa: E402
from badrt.racket_prior import (L_ANK, L_HIP, L_SH, L_WR, NOSE, R_ANK, R_EL, R_HIP, R_SH, KnnPrior,  # noqa: E402
                                _unit, body_frames, lift_candidates, pose_features, to_body, to_world)
from scripts.train_racket_prior import load_video  # noqa: E402

L_KNEE, R_KNEE = 25, 26
EDGES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27),
         (24, 26), (26, 28), (27, 31), (28, 32)]
ZONES = (("front", "前场"), ("mid", "中场"), ("rear", "后场"))
SIDES = (("fh", "正手"), ("bh", "反手"), ("oh", "头顶/高手"))
FRONT_MAX, REAR_MIN = 2.6, 4.6            # front-foot distance to the net (m)


def racket_dirs(cam, rec, det, prior):
    dp, conf = prior.predict_world(rec.P, rec.fps)
    d = dp.copy(); seen = np.zeros(len(d), bool)
    idx, cand, _, _ = lift_candidates(cam, rec, det)
    c = np.einsum("nj,nkj->nk", dp[idx], cand)
    best = cand[np.arange(len(idx)), c.argmax(1)]
    ok = c.max(1) > np.cos(np.radians(75))          # strongly disagreeing detections: keep the prior
    d[idx[ok]] = best[ok]; seen[idx[ok]] = True
    d = _unit(gaussian_filter1d(d, 1.0, axis=0))
    return d, seen, conf


def angle(a, b, c):
    u, v = _unit(a - b), _unit(c - b)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def frame_stats(P, Rb, pel, d, air):
    """Per-frame posture / racket descriptors (dict of (T,) arrays)."""
    yb = Rb[:, 1]
    ank_b = to_body(Rb, P[:, R_ANK] - P[:, L_ANK])
    knee = np.minimum(angle(P[:, L_HIP], P[:, L_KNEE], P[:, L_ANK]), angle(P[:, R_HIP], P[:, R_KNEE], P[:, R_ANK]))
    trunk = _unit(P[:, [L_SH, R_SH]].mean(1) - pel)
    shl = P[:, R_SH] - P[:, L_SH]
    db = to_body(Rb, d)
    fore = _unit(P[:, R_WRIST] - P[:, R_EL])
    return dict(
        pelvis_h=pel[:, 2],
        stance_w=np.abs(ank_b[:, 0]),
        stride=np.abs(ank_b[:, 1]),
        knee_min=knee,
        trunk_lean=np.degrees(np.arctan2((trunk * yb).sum(1), trunk[:, 2])),
        shoulder_yaw=np.degrees(np.abs(np.arctan2(shl[:, 1], shl[:, 0]))),   # 0 = shoulders parallel to the net
        elbow=angle(P[:, R_SH], P[:, R_EL], P[:, R_WRIST]),
        arm_elev=np.degrees(np.arcsin(np.clip(_unit(P[:, R_EL] - P[:, R_SH])[:, 2], -1, 1))),
        wrist_h=P[:, R_WRIST, 2],
        free_wrist_h=P[:, L_WR, 2],
        racket_elev=np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1))),
        racket_azim=np.degrees(np.arctan2(db[:, 0], db[:, 1])),             # 0 forward, + player's right
        racket_forearm=np.degrees(np.arccos(np.clip((d * fore).sum(1), -1, 1))),
        tip_front=((P[:, R_WRIST] + TIP_DIST * d - pel) * yb).sum(1),
        airborne=air.astype(float),
    )


def detect_hits(P, d, ok, fps, min_gap=1.0, pct=90):
    """Singles: the same player hits at most about once per second (the shuttle has to reach the
    opponent and come back), so only the strongest swing within min_gap is kept."""
    tip = P[:, R_WRIST] + TIP_DIST * d
    v = np.linalg.norm(np.gradient(gaussian_filter1d(tip, 1.0, axis=0), axis=0), axis=1) * fps
    v[~ok] = 0
    thr = np.percentile(v[ok], pct)
    pk, _ = find_peaks(v, height=thr, distance=int(min_gap * fps), prominence=thr * 0.4)
    good = np.array([ok[max(0, k - 8):k + 9].all() for k in pk], bool)
    return pk[good], v


def classify(P, Rb, pel, d, k, net_y=0.0, pre=8, lag=2):
    """The tip-speed peak trails the contact by a few frames (follow-through), so the stroke is
    judged on [k-pre, k]: overhead if the wrist goes above the head there (contact = highest wrist
    frame), else contact = k-lag. Returns zone, side, front-foot distance, contact frame."""
    w = np.arange(max(0, k - pre), k + 1)
    above = P[w, R_WRIST, 2] - P[w, NOSE, 2]
    if above.max() > 0:
        side, c = "oh", int(w[above.argmax()])
    else:
        c = max(0, k - lag)
        tip = P[c, R_WRIST] + TIP_DIST * d[c]
        side = "fh" if to_body(Rb[c:c + 1], (tip - pel[c])[None])[0, 0] > 0 else "bh"
    front_d = min(abs(P[c, L_ANK, 1] - net_y), abs(P[c, R_ANK, 1] - net_y))
    zone = "front" if front_d < FRONT_MAX else ("rear" if front_d > REAR_MIN else "mid")
    return zone, side, front_d, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["standard_video"])
    ap.add_argument("--out", default="results/strokes")
    a = ap.parse_args()
    prior = KnnPrior.load("assets/racket_prior.npz")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    events = []
    for stem in a.videos:
        cam, data, det, rec = load_video(stem, log=False)
        ok = rec.ok & data["main"].astype(bool)
        d, seen, pconf = racket_dirs(cam, rec, det, prior)
        Rb, pel = body_frames(rec.P)
        st = frame_stats(rec.P, Rb, pel, d, rec.airborne)
        hits, v = detect_hits(rec.P, d, ok, rec.fps)
        for kp in hits:
            zone, side, fd, k = classify(rec.P, Rb, pel, d, kp)
            Pb = to_body(np.repeat(Rb[k:k + 1], 33, 0), rec.P[k] - pel[k])
            events.append(dict(video=stem, frame=int(k), zone=zone, side=side, front_d=float(fd),
                               seen=bool(seen[max(0, k - 2):k + 3].any()), speed=float(v[k]),
                               court_xy=pel[k, :2].tolist(), skel=Pb.tolist(), rdir=to_body(Rb[k:k + 1], d[k:k + 1])[0].tolist(),
                               **{n: float(x[k]) for n, x in st.items()}))
        print(f"{stem}: {len(hits)} hit moments in {ok.sum() / rec.fps / 60:.1f} min of main-camera play")
    keys = [k for k in events[0] if k not in ("video", "frame", "zone", "side", "front_d", "seen", "speed",
                                              "court_xy", "skel", "rdir")]
    stats = {}
    for zk, zn in ZONES:
        for sk, sn in SIDES:
            E = [e for e in events if e["zone"] == zk and e["side"] == sk]
            if len(E) < 5:
                continue
            s = dict(name=zn + sn, n=len(E), seen=float(np.mean([e["seen"] for e in E])))
            for k in keys:
                x = np.array([e[k] for e in E])
                if k == "airborne":                    # binary: rate (same value in all three slots)
                    s[k] = [float(x.mean())] * 3
                else:
                    s[k] = [float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))]
            stats[f"{zk}_{sk}"] = s
    (out / "stats.json").write_text(json.dumps(dict(n_events=len(events), categories=stats), indent=1,
                                               ensure_ascii=False))
    np.savez_compressed(out / "events.npz", events=json.dumps(events))
    print_table(stats)
    plot(events, stats, out)


def print_table(stats):
    cols = ["n", "pelvis_h", "stride", "knee_min", "trunk_lean", "shoulder_yaw", "elbow", "wrist_h",
            "racket_elev", "racket_azim", "racket_forearm", "airborne"]
    print("category        " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for k, s in stats.items():
        row = [f"{s['n']:9d}"] + [f"{s[c][1]:9.2f}" for c in cols[1:]]
        print(f"{s['name']:<12s}" + " ".join(row))


def plot(events, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cats = list(stats)
    fig, ax = plt.subplots(2, len(cats), figsize=(2.3 * len(cats), 6.4))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, c in enumerate(cats):
        zk, sk = c.split("_")
        E = [e for e in events if e["zone"] == zk and e["side"] == sk]
        S = np.median(np.array([e["skel"] for e in E]), 0)
        r = _unit(np.median(np.array([e["rdir"] for e in E]), 0))
        for row, (i0, lab) in enumerate(((0, "x 右 (m)"), (1, "y 前 (m)"))):
            a_ = ax[row, j]
            for e in E[:: max(1, len(E) // 25)]:                       # faint individual samples
                Si = np.array(e["skel"]); ri = np.array(e["rdir"])
                a_.plot([Si[R_WRIST, i0], Si[R_WRIST, i0] + TIP_DIST * ri[i0]],
                        [Si[R_WRIST, 2], Si[R_WRIST, 2] + TIP_DIST * ri[2]], color="tab:orange", alpha=0.15, lw=1)
            for p, q in EDGES:
                a_.plot(S[[p, q], i0], S[[p, q], 2], color="tab:blue" if p % 2 == 0 and q % 2 == 0 else "0.35", lw=2)
            a_.plot(S[NOSE, i0], S[NOSE, 2], "o", color="0.35", ms=8)
            w = S[R_WRIST]
            a_.plot([w[i0], w[i0] + TIP_DIST * r[i0]], [w[2], w[2] + TIP_DIST * r[2]], color="tab:red", lw=3)
            a_.axhline(S[[27, 28, 29, 30, 31, 32], 2].min(), color="0.8", lw=1)
            a_.set_xlim(-1.1, 1.1); a_.set_ylim(-1.15, 1.35); a_.set_aspect("equal")
            a_.set_xlabel(lab, fontsize=8); a_.tick_params(labelsize=7)
            if row == 0:
                a_.set_title(f"{stats[c]['name']}\n(n={stats[c]['n']})", fontsize=9)
    ax[0, 0].set_ylabel("正面视图 z (m)"); ax[1, 0].set_ylabel("侧面视图 z (m)")
    fig.suptitle("击球瞬间的中位姿态（蓝：持拍侧/右侧肢体，红：球拍）", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "skeletons.png", dpi=130); plt.close(fig)

    fig, a_ = plt.subplots(figsize=(6.5, 4.2))
    for c in cats:
        zk, sk = c.split("_")
        E = [e for e in events if e["zone"] == zk and e["side"] == sk]
        a_.scatter([e["racket_azim"] for e in E], [e["racket_elev"] for e in E], s=10, alpha=0.6,
                   label=stats[c]["name"])
    a_.set_xlabel("球拍方位角（身体系，0=正前，+=右侧）/ deg"); a_.set_ylabel("球拍仰角 / deg")
    a_.legend(fontsize=7, ncol=2); a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "racket_dirs.png", dpi=130); plt.close(fig)

    fig, a_ = plt.subplots(figsize=(4.2, 5.2))
    for c in cats:
        zk, sk = c.split("_")
        E = [e for e in events if e["zone"] == zk and e["side"] == sk]
        xy = np.array([e["court_xy"] for e in E])
        a_.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.7, label=stats[c]["name"])
    for y in (0, -1.98, -5.94, -6.7):
        a_.plot([-3.05, 3.05], [y, y], "k-", lw=0.8)
    for x in (-3.05, -2.59, 2.59, 3.05):
        a_.plot([x, x], [0, -6.7], "k-", lw=0.8)
    a_.plot([0, 0], [-1.98, -6.7], "k-", lw=0.8)
    a_.axhline(-FRONT_MAX, color="r", ls=":", lw=0.8); a_.axhline(-REAR_MIN, color="r", ls=":", lw=0.8)
    a_.set_aspect("equal"); a_.set_xlim(-3.6, 3.6); a_.set_ylim(-7.6, 0.4)
    a_.legend(fontsize=6, loc="lower center", ncol=2); a_.set_title("击球时骨盆位置（本方半场）", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "court_map.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

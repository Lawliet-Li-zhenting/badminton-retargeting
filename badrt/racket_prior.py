"""Learned racket prior: body pose -> racket direction, trained without manual labels.

Labels come from the image racket detector (badrt.racket). A 2D detection lifted onto the camera ray
gives two 3D candidates (in front of / behind the wrist depth); which one is correct is a latent
variable, so the prior is fitted by hard EM:
    M-step  kNN regressor   pose features -> unit racket direction in the body frame
    E-step  per detected run, Viterbi over the two candidates with cost
            (angle to the leave-out kNN prediction) + (angle change between frames)
Body frame: origin pelvis centre, x = player's right (horizontal shoulder line), y = forward,
z = world up. Directions are wrist -> racket tip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .racket import R_WRIST, TIP_DIST, ray_sphere

L_SH, R_SH, L_EL, R_EL, L_WR = 11, 12, 13, 14, 15
L_HIP, R_HIP, L_ANK, R_ANK, NOSE = 23, 24, 27, 28, 0
R_PINKY, R_INDEX = 18, 20


def body_frames(P):
    """(T,3,3) rotation rows [x right, y forward, z up] and (T,3) pelvis origin, from (T,33,3)."""
    pel = P[:, [L_HIP, R_HIP]].mean(1)
    x = (P[:, R_SH] - P[:, L_SH]) + 0.5 * (P[:, R_HIP] - P[:, L_HIP])
    x[:, 2] = 0
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    z = np.broadcast_to([0.0, 0.0, 1.0], x.shape)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1), pel


def to_body(Rb, v):
    return np.einsum("tij,tj->ti", Rb, v)


def to_world(Rb, v):
    return np.einsum("tji,tj->ti", Rb, v)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


FEATURE_GROUPS = dict(r_upper=3, r_fore=3, r_wrist=3, hand=3, l_wrist=3, trunk=3, pelvis_h=1, wrist_vel=3,
                      zone=2)
GROUP_WEIGHT = dict(r_upper=1.0, r_fore=1.5, r_wrist=1.0, hand=1.0, l_wrist=0.0, trunk=0.3, pelvis_h=0.0,
                    wrist_vel=0.3, zone=0.2)       # held-out sweep: whole-body features hurt (results/racket_prior)


def pose_features(P, fps=30.0, net_y=0.0):
    """(T,F) features in the body frame (see FEATURE_GROUPS), plus (Rb, pelvis)."""
    P = np.asarray(P, float)
    Rb, pel = body_frames(P)
    sh = P[:, R_SH]
    f = dict(
        r_upper=_unit(to_body(Rb, P[:, R_EL] - sh)),
        r_fore=_unit(to_body(Rb, P[:, R_WRIST] - P[:, R_EL])),
        r_wrist=to_body(Rb, P[:, R_WRIST] - sh) / 0.6,
        hand=_unit(to_body(Rb, P[:, [R_PINKY, R_INDEX]].mean(1) - P[:, R_WRIST])),
        l_wrist=to_body(Rb, P[:, L_WR] - P[:, L_SH]) / 0.6,
        trunk=_unit(to_body(Rb, P[:, [L_SH, R_SH]].mean(1) - pel)),
        pelvis_h=(pel[:, 2:3] - 0.9) / 0.2,
        wrist_vel=to_body(Rb, gaussian_filter1d(np.gradient(P[:, R_WRIST], axis=0), 1.5, axis=0) * fps) / 5.0,
        zone=np.column_stack([(np.abs(pel[:, 1] - net_y) - 3.5) / 2.0, pel[:, 0] / 2.0]),
    )
    X = np.concatenate([f[k] * GROUP_WEIGHT[k] for k in FEATURE_GROUPS], 1)
    return X, Rb, pel


def lift_candidates(cam, rec, det, conf_lo=0.15, max_miss=0.12, tip_dist=TIP_DIST):
    """Per detected frame (MediaPipe right wrist only): (idx, cand (n,2,3) world unit directions,
    n_cand (n,), conf (n,)). Grazing rays give one candidate (duplicated)."""
    idx, cands, ncand, conf = [], [], [], []
    for k in np.flatnonzero(rec.ok & (det["hand"] == 0) & (det["conf"] >= conf_lo)):
        wr = rec.P[k, R_WRIST]
        cs, miss = ray_sphere(cam, det["uv"][k], wr, tip_dist)
        if miss > max_miss:
            continue
        d = [_unit(c - wr) for c in cs]
        idx.append(k); ncand.append(len(d)); conf.append(det["conf"][k])
        cands.append(d if len(d) == 2 else d * 2)
    return np.array(idx, int), np.array(cands).reshape(-1, 2, 3), np.array(ncand), np.array(conf)


def _forearm_of(X):
    """Body-frame forearm unit direction stored in the feature vector (r_fore group)."""
    o = 0
    for g, n in FEATURE_GROUPS.items():
        if g == "r_fore":
            return _unit(X[:, o:o + n])
        o += n


@dataclass
class KnnPrior:
    """Racket direction as the query forearm direction plus the neighbours' mean (racket - forearm)
    offset: the wrist angle generalises better than absolute directions (held-out 2D error)."""
    X: np.ndarray          # (N,F) training features
    Y: np.ndarray          # (N,3) body-frame unit directions
    W: np.ndarray          # (N,) sample weights
    k: int = 48
    bw: float = 0.0        # Gaussian bandwidth on feature distance (0: median kNN distance)

    def predict(self, Xq, exclude=None, chunk=1024):
        """(M,3) unit directions, (M,) confidence (resultant length of the weighted neighbour
        directions, 1 = all neighbours agree) and (M,) mean neighbour distance.
        exclude: fn(i0, i1) -> (i1-i0, N) bool mask of training samples to ignore."""
        M = len(Xq)
        out = np.zeros((M, 3)); conf = np.zeros(M); dist = np.zeros(M)
        x2 = (self.X ** 2).sum(1)
        R = self.Y - _forearm_of(self.X)
        Fq = _forearm_of(Xq)
        for i0 in range(0, M, chunk):
            i1 = min(M, i0 + chunk)
            D = (Xq[i0:i1] ** 2).sum(1)[:, None] + x2[None] - 2 * Xq[i0:i1] @ self.X.T
            if exclude is not None:
                D[exclude(i0, i1)] = np.inf
            nn = np.argpartition(D, self.k, axis=1)[:, :self.k]
            d = np.sqrt(np.maximum(np.take_along_axis(D, nn, 1), 0))
            bw = self.bw if self.bw > 0 else np.median(d, 1, keepdims=True) + 1e-6
            w = np.exp(-0.5 * (d / bw) ** 2) * self.W[nn]
            w /= w.sum(1, keepdims=True) + 1e-9
            out[i0:i1] = _unit(Fq[i0:i1] + (w[..., None] * R[nn]).sum(1))
            conf[i0:i1] = np.linalg.norm((w[..., None] * self.Y[nn]).sum(1), axis=1)
            dist[i0:i1] = d.mean(1)
        return out, conf, dist

    def save(self, path):
        np.savez_compressed(path, X=self.X, Y=self.Y, W=self.W, k=self.k, bw=self.bw,
                            weights=json.dumps(GROUP_WEIGHT))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        if json.loads(str(z["weights"])) != GROUP_WEIGHT:
            raise ValueError(f"{path}: trained with other feature weights, retrain")
        return cls(z["X"], z["Y"], z["W"], int(z["k"]), float(z["bw"]))

    def predict_world(self, P, fps=30.0):
        """Prior for a whole sequence (T,33,3): world unit directions (T,3), confidence (T,)."""
        X, Rb, _ = pose_features(P, fps)
        d, c, _ = self.predict(X)
        return to_world(Rb, d), c


def _viterbi_runs(t, cost_u, cand, w_smooth=1.0, max_gap=3):
    """Choose 0/1 per sample; t sorted frame indices. cost_u (n,2) unary; consecutive samples
    (gap <= max_gap) pay w_smooth * (1 - cos) between their chosen directions."""
    n = len(t)
    choice = np.zeros(n, int)
    a = 0
    while a < n:
        b = a + 1
        while b < n and t[b] - t[b - 1] <= max_gap:
            b += 1
        acc = cost_u[a].copy(); back = []
        for i in range(a + 1, b):
            pair = w_smooth * (1 - cand[i - 1] @ cand[i].T)          # (2 prev, 2 cur)
            tot = acc[:, None] + pair
            back.append(tot.argmin(0)); acc = tot.min(0) + cost_u[i]
        s = int(acc.argmin()); choice[b - 1] = s
        for i in range(b - 1, a, -1):
            s = int(back[i - a - 1][s]); choice[i - 1] = s
        a = b
    return choice


def front_init(rec, idx, cand):
    """Initial choice: the candidate in front of the torso (the old hand-made rule)."""
    Rb, pel = body_frames(rec.P[idx])
    yb = np.einsum("nj,nkj->nk", Rb[:, 1], cand)                    # forward component (n,2)
    return (yb[:, 1] > yb[:, 0]).astype(int)


def fit_em(X, t_frame, seq_id, cand_body, conf, n_iter=5, k=48, excl=30, w_smooth=2.0, init=None, log=True):
    """Hard EM over the lift ambiguity. X (n,F), t_frame (n,) frame index, seq_id (n,) video id,
    cand_body (n,2,3) body-frame candidate directions, conf (n,) detector confidence.
    Returns (KnnPrior, choice (n,), history list of dict)."""
    n = len(X)
    choice = np.zeros(n, int) if init is None else init.copy()
    W = np.clip(conf, 0.1, 1.0)
    hist = []

    def exclude(i0, i1):
        return (seq_id[i0:i1, None] == seq_id[None]) & (np.abs(t_frame[i0:i1, None] - t_frame[None]) < excl)

    order = np.lexsort((t_frame, seq_id))
    for it in range(n_iter):
        Y = cand_body[np.arange(n), choice]
        prior = KnnPrior(X, Y, W, k=k)
        pred, pconf, _ = prior.predict(X, exclude=exclude)
        cos = np.einsum("nj,nkj->nk", pred, cand_body)               # (n,2)
        cost_u = (1 - cos) * (0.3 + pconf[:, None])
        new = np.zeros(n, int)
        for s in np.unique(seq_id):
            o = order[seq_id[order] == s]
            new[o] = _viterbi_runs(t_frame[o], cost_u[o], cand_body[o], w_smooth)
        changed = float(np.mean(new != choice))
        ang = np.degrees(np.arccos(np.clip(cos[np.arange(n), new], -1, 1)))
        hist.append(dict(iter=it, changed=changed, median_angle=float(np.median(ang)),
                         mean_conf=float(pconf.mean())))
        if log:
            print(f"  EM {it}: changed {changed:.3f}  median angle to prior {np.median(ang):.1f} deg  "
                  f"conf {pconf.mean():.2f}", flush=True)
        choice = new
        if changed < 0.002:
            break
    Y = cand_body[np.arange(n), choice]
    return KnnPrior(X, Y, W, k=k), choice, hist

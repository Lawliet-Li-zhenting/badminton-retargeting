"""Stroke library: contact-centred motion segments cut from retargeted expert videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d

from .character import CharacterIndex, build_single_character_model


@dataclass
class Segment:
    qpos: np.ndarray          # (T, 45), facing +y and pelvis xy at the origin at the contact frame
    racket: np.ndarray        # (T, 3) racket head centre
    racket_R: np.ndarray      # (T, 3, 3)
    contact: int              # contact frame index
    fps: float                # playback rate (after the per-video time calibration)
    height: float             # racket height at contact
    lateral: float            # racket x at contact (>0: forehand side of a right-hander)
    peak_speed: float         # racket speed at contact at this fps
    source: str
    kind: str                 # overhead | mid | low

    @property
    def side(self):
        return "forehand" if self.lateral >= 0 else "backhand"


def yaw_of(qpos):
    """Heading of the pelvis (+y of the body) about z."""
    w, x, y, z = qpos[3:7]
    fy = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)])     # R[:, 1] xy
    return float(np.arctan2(-fy[0], fy[1]))


def rotate_qpos(qpos, yaw, shift_xy=(0.0, 0.0)):
    """Rotate a (T,45) clip about z through the origin by yaw, then shift."""
    q = np.array(qpos, float, copy=True)
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = q[..., 0].copy(), q[..., 1].copy()
    q[..., 0] = c * x - s * y + shift_xy[0]
    q[..., 1] = s * x + c * y + shift_xy[1]
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    w0, x0, y0, z0 = q[..., 3].copy(), q[..., 4].copy(), q[..., 5].copy(), q[..., 6].copy()
    a, _, _, d = qz
    q[..., 3] = a * w0 - d * z0
    q[..., 4] = a * x0 - d * y0
    q[..., 5] = a * y0 + d * x0
    q[..., 6] = a * z0 + d * w0
    return q


def align_quats(Q):
    """Make the quaternion blocks of a (T,45) clip sign-continuous in time (q and -q are the same
    rotation, but interpolation needs neighbours on the same hemisphere)."""
    from .character import quat_blocks
    Q = np.array(Q, float, copy=True)
    for adr in quat_blocks():
        for k in range(1, len(Q)):
            if Q[k, adr:adr + 4] @ Q[k - 1, adr:adr + 4] < 0:
                Q[k, adr:adr + 4] *= -1
    return Q


def sample_clip(Q, k):
    """C1 (Catmull-Rom) sample of a sign-aligned clip at fractional frame k. A piecewise-linear
    clip has velocity jumps at every frame, which is fatal for a racket whose speed at contact
    decides the shot."""
    from .character import quat_blocks
    n = len(Q)
    k = min(max(float(k), 0.0), n - 1.0)
    i = min(int(k), n - 2)
    f = k - i
    p0, p1, p2, p3 = Q[max(i - 1, 0)], Q[i], Q[i + 1], Q[min(i + 2, n - 1)]
    f2, f3 = f * f, f * f * f
    q = 0.5 * ((2 * p1) + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 + (3 * p1 - p0 - 3 * p2 + p3) * f3)
    for adr in quat_blocks():
        q[adr:adr + 4] /= np.linalg.norm(q[adr:adr + 4])
    return q


def classify(h):
    return "overhead" if h >= 1.9 else ("low" if h < 1.0 else "mid")


def extract_segments(npz_path, target_peak=10.0, pre_s=0.75, post_s=0.55, min_gap_s=0.7, min_peak=4.0,
                     min_height=0.25, min_width_frames=3, min_pelvis_up=0.5):
    z = np.load(npz_path, allow_pickle=True)
    qpos, ok, fps = z["qpos"], z["ok"], float(z["fps"])
    m = build_single_character_model()
    d = mujoco.MjData(m)
    ci = CharacterIndex(m, "")
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rc = np.zeros((len(qpos), 3)); rR = np.zeros((len(qpos), 3, 3))
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        rc[k] = d.site_xpos[ci.racket_site]; rR[k] = d.site_xmat[ci.racket_site].reshape(3, 3)
        ok[k] &= d.xmat[pelvis][8] >= min_pelvis_up          # upright pelvis (z-z entry of R)
    sp = np.r_[0, np.linalg.norm(np.diff(rc, axis=0), axis=1) * fps]
    sp = uniform_filter1d(sp, 3)
    good = ok & np.r_[True, ok[:-1]]
    sp[~good] = 0
    pre, post, gap = int(pre_s * fps), int(post_s * fps), int(min_gap_s * fps)
    thr = max(min_peak, np.percentile(sp[good], 85))
    peaks = []
    for k in np.argsort(-sp):
        if sp[k] < thr:
            break
        if k - pre < 0 or k + post >= len(qpos) or not ok[k - pre:k + post].all():
            continue
        if rc[k, 2] < min_height:
            continue
        w = slice(max(k - 4, 0), k + 5)
        if np.sum(sp[w] > 0.5 * sp[k]) < min_width_frames:      # 1-2 frame spikes are tracking flips
            continue
        if all(abs(k - p) > gap for p in peaks):
            peaks.append(int(k))
    if not peaks:
        return []
    # slow-motion calibration: retargeted real-time footage peaks around 10 m/s (motion blur and
    # filtering hide the true ~40 m/s racket head speed), so slower clips are sped up to match
    gain = float(np.clip(target_peak / np.median(sp[peaks]), 0.5, 5.0))
    segs = []
    for k in sorted(peaks):
        sl = slice(k - pre, k + post)
        yaw = yaw_of(qpos[k])
        q = rotate_qpos(qpos[sl], -yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        shift = -q[pre, :2]
        q[:, :2] += shift
        r = rc[sl] @ Rz.T; r[:, :2] += shift
        RR = np.einsum("ij,tjk->tik", Rz, rR[sl])
        h = float(r[pre, 2])
        segs.append(Segment(q, r, RR, pre, fps * gain, h, float(r[pre, 0]), float(sp[k] * gain),
                            Path(npz_path).parent.name, classify(h)))
    return segs


def save_library(segs, path):
    np.savez_compressed(path, n=len(segs), **{f"s{i}_{k}": getattr(s, k) for i, s in enumerate(segs)
                                              for k in Segment.__dataclass_fields__})


def load_library(path):
    z = np.load(path, allow_pickle=True)
    out = []
    for i in range(int(z["n"])):
        kw = {}
        for k in Segment.__dataclass_fields__:
            v = z[f"s{i}_{k}"]
            kw[k] = v.item() if v.ndim == 0 else v
        kw["contact"] = int(kw["contact"])
        out.append(Segment(**kw))
    return out

"""Physical swings for the command-driven opponent.

A stroke clip from the library gets the racket roughly to the shuttle; this module warps it so
that the string bed really is at the contact point, at the contact time, with the face normal
and the head speed demanded by the inverse impact model (shuttle.inverse_impact). The shuttle
then leaves the racket through the ordinary collision code -- nothing sets its velocity.

Warp of the racket head trajectory of the clip (t_c = contact time, tau = t - t_c):
    x(t) = x_clip(t) + w(tau) (c - x_clip(t_c)) + (s n - v_clip(t_c)) g(tau)
    w(tau) = cos^2 window (1 at contact, zero slope)      -> position correction
    g(tau) = tau exp(-tau^2 / 2 sigma^2)                  -> velocity correction, slope 1 at contact
    R(t)   = exp(w(tau) rho) R_clip(t),  exp(rho) n_clip(t_c) = n
sigma is chosen so that the extra excursion 0.61 sigma |dv| stays within what the arm can absorb.
The targets are tracked by damped-least-squares IK on the racket arm (+ a little lumbar), on a
time grid that is fine (sigma / 12) around the contact; poses between nodes are interpolated,
so the game and the planner evaluate exactly the same racket motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .character import CharacterIndex, blend_qpos, build_single_character_model, quat_slerp
from .shuttle import RacketPose

ARM_JOINTS = ["r_shoulder", "r_elbow", "r_pronation", "r_wrist_fe", "r_wrist_dev"]
TRUNK_JOINTS = ["lumbar"]
ROT_LEN = 0.12          # metres per radian: orientation vs position error (position wins a conflict)


def rotvec(R):
    """Rotation vector of a rotation matrix."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if np.pi - ang < 1e-4:                        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return ax * ang
    return v * (ang / (2.0 * np.sin(ang)))


def expmap(r):
    ang = np.linalg.norm(r)
    if ang < 1e-9:
        return np.eye(3)
    k = r / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rot_between(a, b):
    """Rotation vector of the smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.zeros(3)
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        return p / np.linalg.norm(p) * np.pi
    return v / s * np.arctan2(s, c)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class SwingIK:
    """Kinematics of one (right-handed, internally) character: FK of the racket and arm IK."""

    def __init__(self, dims=None):
        self.m = build_single_character_model(dims=dims)
        self.d = mujoco.MjData(self.m)
        self.ci = CharacterIndex(self.m, "")
        m = self.m
        dofs, wts, lim = [], [], []
        for names, w in ((ARM_JOINTS, 1.0), (TRUNK_JOINTS, 0.3)):
            for nm in names:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
                nd = 3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
                dofs += list(range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nd))
                wts += [w] * nd
                if nd == 1:
                    lim.append((len(dofs) - 1, int(m.jnt_qposadr[j]), *m.jnt_range[j]))
        self.dofs = np.array(dofs)
        self.wdof = np.array(wts)
        self.lim = lim                       # (index into dofs, qpos address, lo, hi) of the hinges
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        self.dq = np.zeros(m.nv)
        self.nq = m.nq

    def fk(self, q):
        d = self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.m, d)
        s = self.ci.racket_site
        return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()

    def solve(self, q, pos, R_tgt, warm=None, w_tilt=1.0, w_twist=0.15, iters=14, tol=2e-4, damp=0.015):
        """Move the racket of pose q to (pos, R_tgt). Tilt of the face normal is weighted w_tilt,
        rotation about the normal (which only swings the shaft inside the face plane) w_twist.
        warm: tangent-space offset (nv,) of the previous solution. Returns (q, offset, pos_err, tilt_err)."""
        m, d, s = self.m, self.d, self.ci.racket_site
        q = np.array(q, float)
        off = np.zeros(m.nv)
        if warm is not None:
            off[self.dofs] = warm[self.dofs]
            mujoco.mj_integratePos(m, q, off, 1.0)
            self._clamp(q)
        e_pos = tilt = 0.0
        wd = self.wdof.copy()                 # a hinge pushed against its limit is locked (active set)
        I6 = damp * damp * np.eye(6)
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, R = d.site_xpos[s], d.site_xmat[s].reshape(3, 3)
            ep = pos - p
            r = rotvec(R_tgt @ R.T)
            n = R[:, 0]
            r_tw = (r @ n) * n
            r_ti = r - r_tw
            e_pos, tilt = float(np.linalg.norm(ep)), float(np.linalg.norm(r_ti))
            if e_pos < tol and tilt < 2e-3:
                break
            mujoco.mj_jacSite(m, d, self.jp, self.jr, s)
            Wr = ROT_LEN * (w_tilt * (np.eye(3) - np.outer(n, n)) + w_twist * np.outer(n, n))
            e = np.r_[ep, Wr @ r]
            ne = np.linalg.norm(e)
            if ne > 0.2:
                e *= 0.2 / ne
            J = np.vstack([self.jp[:, self.dofs], Wr @ self.jr[:, self.dofs]]) * wd
            step = wd * (J.T @ np.linalg.solve(J @ J.T + I6, e))
            for i, adr, lo, hi in self.lim:
                v = q[adr] + step[i]
                if v < lo or v > hi:
                    step[i] = (lo if v < lo else hi) - q[adr]
                    wd[i] = 0.0
            self.dq[:] = 0
            self.dq[self.dofs] = step
            mujoco.mj_integratePos(m, q, self.dq, 1.0)
            off[self.dofs] += step
        return q, off, e_pos, tilt

    def _clamp(self, q):
        for _, adr, lo, hi in self.lim:
            q[adr] = min(max(q[adr], lo), hi)


@dataclass
class Swing:
    """Pre-computed pose nodes of one stroke; everything else interpolates between them."""
    t: np.ndarray            # (N,) node times
    q: np.ndarray            # (N, 45)
    rc: np.ndarray           # (N, 3) racket head centre
    rq: np.ndarray           # (N, 4) racket orientation quaternions (sign-continuous)
    t_hit: float
    pos_err: float           # IK residual at the contact node [m]
    tilt_err: float          # [rad]
    sigma: float
    head_speed: float        # commanded string-bed speed along the normal
    rc_box: tuple = (0.0, 0.0, 0.0, 1e9)   # (x, y, z, half-size) around every racket position of the stroke

    def _locate(self, t):
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        i = min(max(i, 0), len(self.t) - 2)
        f = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        return i, min(max(f, 0.0), 1.0)

    def qpos_at(self, t):
        i, f = self._locate(t)
        return blend_qpos(self.q[i], self.q[i + 1], f)

    def racket_at(self, t):
        i, f = self._locate(t)
        c = self.rc[i] + f * (self.rc[i + 1] - self.rc[i])
        return RacketPose(c, quat2mat(quat_slerp(self.rq[i], self.rq[i + 1], f)))

    def covers(self, t):
        return self.t[0] <= t <= self.t[-1]


def build_swing(ik: SwingIK, base_fn, t0, t1, t_hit, contact, normal, speed, window=0.30, max_excursion=0.35,
                coarse_dt=1 / 60, fine_per_sigma=8, shuttle_radius=0.03):
    """Warp the pose function base_fn(t) (t in [t0, t1], C1 in t) so that the string bed meets a shuttle
    whose centre is at `contact` at t_hit, with face normal +-`normal` (pointing at the incoming
    shuttle) and velocity `speed * normal`. The collision triggers when the shuttle centre is one
    shuttle radius off the string plane, so the head centre is aimed that far behind the contact."""
    h = 0.002
    p_c, R_c = ik.fk(base_fn(t_hit))
    p_a, _ = ik.fk(base_fn(max(t_hit - h, t0)))
    p_b, _ = ik.fk(base_fn(min(t_hit + h, t1)))
    v_base = (p_b - p_a) / (min(t_hit + h, t1) - max(t_hit - h, t0))
    dp = np.asarray(contact, float) - shuttle_radius * np.asarray(normal, float) - p_c
    dv = speed * np.asarray(normal, float) - v_base
    ndv = float(np.linalg.norm(dv))
    sigma = float(np.clip(max_excursion / (0.607 * max(ndv, 1e-6)), 0.012, 0.06))
    n_clip = R_c[:, 0]
    n_t = np.asarray(normal, float) * (1.0 if n_clip @ normal >= 0 else -1.0)       # the strings have two faces
    rho = rot_between(n_clip, n_t)
    w_pre = max(min(window, t_hit - t0), 1e-3)
    w_post = max(min(window, t1 - t_hit), 1e-3)

    coarse = np.arange(t0, t1, coarse_dt)
    fine = t_hit + sigma * np.linspace(-3, 3, 6 * fine_per_sigma + 1)
    ts = np.unique(np.r_[coarse, fine[(fine > t0) & (fine < t1)], t1])
    keep = np.r_[True, np.diff(ts) > 1e-5]
    ts = ts[keep]
    N = len(ts)
    Q = np.zeros((N, ik.nq)); RC = np.zeros((N, 3)); RQ = np.zeros((N, 4))
    warm = None
    pos_err = tilt_err = 0.0
    i_hit = int(np.argmin(np.abs(ts - t_hit)))
    for i, t in enumerate(ts):
        qb = base_fn(t)
        tau = t - t_hit
        W = w_pre if tau < 0 else w_post
        if abs(tau) < W:
            w = 0.5 * (1 + np.cos(np.pi * tau / W))
            gate = min(1.0, (t - t0) / 0.05)
            g = tau * np.exp(-0.5 * (tau / sigma) ** 2) * gate
            pb, Rb = ik.fk(qb)
            q, warm, ep, et = ik.solve(qb, pb + w * dp + g * dv, expmap(w * rho) @ Rb, warm=warm)
            if i == i_hit:
                pos_err, tilt_err = ep, et
        else:
            q, warm = qb, None
        Q[i] = q
        RC[i], R = ik.fk(q)
        RQ[i] = mat2quat(R)
        if i and RQ[i] @ RQ[i - 1] < 0:
            RQ[i] = -RQ[i]
    sw = Swing(ts, Q, RC, RQ, float(t_hit), float(pos_err), float(tilt_err), sigma, float(speed))
    mid = 0.5 * (RC.min(0) + RC.max(0))
    sw.rc_box = (mid[0], mid[1], mid[2], float(0.5 * np.ptp(RC[:, :2], axis=0).max() + 0.7))   # coarse reach, for culling
    return sw


def contact_error(ik: SwingIK, base_fn, t_hit, contact, normal, shuttle_radius=0.03):
    """Can the arm get the string bed to the contact at all? One IK solve at the contact instant
    (a millisecond) instead of the whole stroke: (position error [m], tilt error [rad])."""
    qb = base_fn(t_hit)
    p_c, R_c = ik.fk(qb)
    n = np.asarray(normal, float)
    n_t = n * (1.0 if R_c[:, 0] @ n >= 0 else -1.0)
    _, _, ep, et = ik.solve(qb, np.asarray(contact, float) - shuttle_radius * n, expmap(rot_between(R_c[:, 0], n_t)) @ R_c,
                            iters=24)
    return ep, et


def free_swing(ik: SwingIK, base_fn, t0, t1, t_hit, dt=1 / 60):
    """A stroke played as it is, with nothing to aim at (a swing at thin air): the nodes are the base poses."""
    ts = np.unique(np.r_[np.arange(t0, t1, dt), t1])
    N = len(ts)
    Q = np.zeros((N, ik.nq)); RC = np.zeros((N, 3)); RQ = np.zeros((N, 4))
    for i, t in enumerate(ts):
        Q[i] = base_fn(t)
        RC[i], R = ik.fk(Q[i])
        RQ[i] = mat2quat(R)
        if i and RQ[i] @ RQ[i - 1] < 0:
            RQ[i] = -RQ[i]
    sw = Swing(ts, Q, RC, RQ, float(t_hit), 0.0, 0.0, 0.03, 0.0)
    mid = 0.5 * (RC.min(0) + RC.max(0))
    sw.rc_box = (mid[0], mid[1], mid[2], float(0.5 * np.ptp(RC[:, :2], axis=0).max() + 0.7))
    return sw

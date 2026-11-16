"""Procedural badminton body motion: everything a player does except the racket arm's stroke.

The stroke library (monocular retargeting of web videos) is good enough for the racket arm, but
its roots and legs are not animation: clips float a metre above the floor, slide and drift. So
the body is generated here from how badminton is actually moved (BWF coaching material and the
biomechanics literature: see docs/technical_report.md, section 3.12):

  ready stance   feet a little wider than the shoulders, racket foot half a foot ahead, knees
                 bent, weight forward, racket up in front at waist height, the other arm in
                 front of the body and slightly out;
  split step     a shallow hop that lands wide as the opponent hits; the first foot leaves the
                 floor about 0.25 s later (the reaction time);
  travelling     short distances with chasse steps (one foot chases the other, never passes it),
                 long ones by running (cross-over steps), body turned towards the corner;
  lunge          front court and wide shots: the racket foot lands as the shuttle is struck,
                 toes pointing at the shuttle, hips low, rear foot turned out, the rear arm
                 extended as a counterbalance;
  overhead       rear court: side-on with the racket foot back and the other arm pointing at the
                 shuttle; a scissor kick swaps the feet through the stroke (with a jump when the
                 shuttle is high).

Pieces:
  Rig      joint addresses and leg geometry of a character, analytic two-bone leg IK
  Gait     stateful foot-step generator driven by the pelvis trajectory (no foot ever slides
           while it is planted)
  Body     authored upper-body poses (ready, relaxed, non-racket arm key poses)
  Loco     pelvis motion with acceleration limits (seek a point / follow a stick)
  Carry    stepping legs for a player whose pose comes from a camera or a recording while the
           footwork assist moves the pelvis across the court
All of it works for the internal right-handed character; left-handers are mirrored afterwards.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .character import quat_slerp

PI = np.pi


# ---------------------------------------------------------------------- small maths
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def sstep(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qaxis(axis, ang):
    return np.r_[np.cos(ang / 2), np.asarray(axis, float) * np.sin(ang / 2)]


def q2m(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def m2q(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return q


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def yaw_of_quat(q):
    """Heading about z of a body whose forward axis is +y."""
    w, x, y, z = q
    return float(np.arctan2(-2 * (x * y - w * z), 1 - 2 * (x * x + z * z)))


def split_yaw_tilt(q):
    """q = q_yaw * q_tilt (twist about z, then the rest)."""
    w, x, y, z = q
    n = np.hypot(w, z)
    if n < 1e-9:
        return np.array([1.0, 0, 0, 0]), np.asarray(q, float)
    tw = np.array([w / n, 0.0, 0.0, z / n])
    return tw, qmul(tw * [1, -1, -1, -1], q)


def arm_quat(u, f):
    """Shoulder quaternion (in the torso frame) and elbow flexion that put the upper arm along u and
    the forearm along f. Rest pose: arm hanging along -z, the elbow hinge (+x) bends the forearm forwards."""
    u, f = unit(u), unit(f)
    c = float(np.clip(u @ f, -1.0, 1.0))
    th = float(np.arccos(c))
    z = -u
    if np.sin(th) > 1e-3:
        y = (f - c * u) / np.sin(th)
    else:
        y = np.array([0.0, 1.0, 0.0]) - u[1] * u
    y = unit(y - (y @ z) * z)
    return m2q(np.column_stack([np.cross(y, z), y, z])), th


# ---------------------------------------------------------------------- rig
ARM = ("shoulder", "elbow", "pronation", "wrist_fe", "wrist_dev")
LEG = ("hip", "knee", "ankle", "subtalar")


class Rig:
    """Joint addresses (relative to the character's root) and leg geometry, read from a compiled model."""

    def __init__(self, m, prefix=""):
        J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
        jid = lambda n: mujoco.mj_name2id(m, J, prefix + n)
        bid = lambda n: mujoco.mj_name2id(m, B, prefix + n)
        root = int(m.jnt_qposadr[jid("root")])
        names = ["lumbar", "neck"] + [f"{s}_{j}" for s in "lr" for j in ARM + LEG]
        self.adr = {n: int(m.jnt_qposadr[jid(n)]) - root for n in names}
        self.rng = {n: m.jnt_range[jid(n)].copy() for n in names
                    if m.jnt_type[jid(n)] == mujoco.mjtJoint.mjJNT_HINGE}
        self.quats = [3] + [self.adr[n] for n in ("lumbar", "neck", "l_shoulder", "r_shoulder", "l_hip", "r_hip")]
        self.hip = np.array([m.body_pos[bid("l_thigh")], m.body_pos[bid("r_thigh")]], float)
        self.L1 = float(-m.body_pos[bid("l_shin")][2])
        self.L2 = float(-m.body_pos[bid("l_foot")][2])
        self.sole, self.toe, self.heel = 0.08, 0.16, 0.07
        self.half, self.toe_tip, self.heel_tip = 0.04, 0.17, 0.075          # corners of the sole (the model's sole sites)
        self.z_stand = self.sole + self.L1 + self.L2 - float(self.hip[0][2])
        self.scale = (self.L1 + self.L2) / 0.87
        self.nq = 45
        self.leg_adr = np.array([a for s in "lr" for a in (list(range(self.adr[f"{s}_hip"], self.adr[f"{s}_hip"] + 4))
                                                            + [self.adr[f"{s}_{j}"] for j in LEG[1:]])])
        self.q0 = np.zeros(self.nq)
        for a in self.quats:
            self.q0[a] = 1.0

    # -- arms
    def set_arm(self, q, side, quat, elbow, wrist=(0.0, 0.0, 0.0)):
        a = self.adr
        q[a[f"{side}_shoulder"]:a[f"{side}_shoulder"] + 4] = quat
        q[a[f"{side}_elbow"]] = float(np.clip(elbow, *self.rng[f"{side}_elbow"]))
        for k, n in enumerate(ARM[2:]):
            q[a[f"{side}_{n}"]] = float(np.clip(wrist[k], *self.rng[f"{side}_{n}"]))

    def arm_slice(self, side):
        a = self.adr[f"{side}_shoulder"]
        return slice(a, a + 8)                                  # shoulder quat, elbow, pronation, wrist x2

    # -- legs
    def leg(self, i, p, Rp, ankle, Rf):
        """Analytic leg IK. i: 0 left, 1 right; p, Rp: pelvis position / rotation; ankle: world position of
        the ankle joint; Rf: world rotation of the foot. Returns (hip quat, knee, ankle angle (not clipped),
        subtalar, reachable). (Scalar maths on purpose: this runs a few hundred times per planned stroke.)"""
        L1, L2 = self.L1, self.L2
        d = ankle - (p + Rp @ self.hip[i])
        dist = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        dh = d / max(dist, 1e-9)
        dc = min(max(dist, 0.42 * (L1 + L2)), 0.997 * (L1 + L2))
        th = PI - math.acos(min(max((L1 * L1 + L2 * L2 - dc * dc) / (2 * L1 * L2), -1.0), 1.0))
        al = math.acos(min(max((L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc), -1.0), 1.0))
        hint = Rf[:, 1]                  # the leg plane has to contain the foot's forward axis: only then can
        npr = hint - (hint @ dh) * dh    # ankle + subtalar lay the sole flat ("the knee goes where the toes point")
        nn = math.sqrt(npr @ npr)
        if nn < 1e-3:
            npr = Rp[:, 2] - (Rp[:, 2] @ dh) * dh
            nn = math.sqrt(npr @ npr)
        npr = npr / nn
        t = math.cos(al) * dh + math.sin(al) * npr              # thigh direction
        sth, cth = math.sin(th), math.cos(th)
        z_t = -t
        if sth > 1e-3:
            y_t = (cth * t - (dh * dc - L1 * t) / L2) / sth     # (dh dc - L1 t) / L2 = shin direction
            y_t = y_t - (y_t @ z_t) * z_t
            y_t = y_t / math.sqrt(y_t @ y_t)
        else:
            y_t = npr
        x_t = np.array([y_t[1] * z_t[2] - y_t[2] * z_t[1], y_t[2] * z_t[0] - y_t[0] * z_t[2], y_t[0] * z_t[1] - y_t[1] * z_t[0]])
        ys, zs = cth * y_t - sth * z_t, sth * y_t + cth * z_t   # shin frame = thigh frame turned by the knee
        f0, f1, f2 = Rf[:, 0], Rf[:, 1], Rf[:, 2]
        ax = math.atan2(zs @ f1, ys @ f1)
        lo, hi = self.rng["l_subtalar"]
        sb = min(max(math.atan2(x_t @ f2, x_t @ f0), lo), hi)
        self._shin = (x_t[2], ys[2], zs[2], ankle[2] + (dc - dist) * dh[2])     # for sole_low()
        return m2q(Rp.T @ np.column_stack([x_t, y_t, z_t])), th, ax, sb, dist <= 0.997 * (L1 + L2)

    def sole_low(self, ax, sb):
        """Height of the lowest corner of the sole for the leg solved last, with the ankle / subtalar angles
        actually set (a joint at its limit rolls or tips the foot, and then one edge would be under the floor)."""
        xz, yz, zz, az = self._shin
        ca, sa, cs, ss = math.cos(ax), math.sin(ax), math.cos(sb), math.sin(sb)
        fy, z1 = ca * yz + sa * zz, ca * zz - sa * yz           # foot axes (z components): forwards, ...
        fx, fz = cs * xz - ss * z1, ss * xz + cs * z1           # ... sideways, up
        return az - self.sole * fz - self.half * abs(fx) + min(self.toe_tip * fy, -self.heel_tip * fy)

    def _foot(self, x, y, lift, Rz, beta):
        """Ankle position and foot rotation for a foot whose floor point is (x, y, lift): beta > 0 lifts the
        heel (the foot pivots on its toes), beta < 0 lifts the toes (it pivots on the heel)."""
        Rf = Rz @ rot_x(-beta) if beta else Rz
        piv = self.toe if beta >= 0 else -self.heel
        return np.array([x, y, lift]) + Rz @ np.array([0.0, piv, 0.0]) - Rf @ np.array([0.0, piv, -self.sole]), Rf

    def set_legs(self, q, feet):
        """feet: (2, 5) rows left / right: x, y of the ankle's floor point, lift, yaw, toe-down pitch.
        A foot the leg cannot reach flat, or only with the ankle beyond its range, comes up on its toes
        (a trailing foot) or on its heel (a foot planted far in front, as a lunge lands)."""
        p, Rp = q[:3], q2m(q[3:7])
        lo, hi = self.rng["l_ankle"]
        a = self.adr
        for i, s in enumerate("lr"):
            x, y, lift, yaw, pitch = feet[i]
            Rz = rot_z(yaw)
            tgt = self._foot(x, y, lift, Rz, pitch)
            sol = self.leg(i, p, Rp, *tgt)
            if not sol[4] or sol[2] > hi:
                for beta in (0.25, 0.5, 0.75, 1.0):
                    if beta > pitch:
                        tgt = self._foot(x, y, lift, Rz, beta)
                        sol = self.leg(i, p, Rp, *tgt)
                        if sol[4] and sol[2] <= hi:
                            break
            elif sol[2] < lo:
                for beta in (-0.2, -0.4, -0.6):
                    tgt = self._foot(x, y, lift, Rz, beta)
                    sol = self.leg(i, p, Rp, *tgt)
                    if sol[2] >= lo:
                        break
            for _ in range(2):                                   # a joint at its limit has left an edge of the sole
                low = self.sole_low(min(max(sol[2], lo), hi), sol[3])           # under the floor: lift the foot by that much
                if low >= lift - 0.001:
                    break
                tgt = (tgt[0] + np.array([0.0, 0.0, lift - low]), tgt[1])
                sol = self.leg(i, p, Rp, *tgt)
            q[a[f"{s}_hip"]:a[f"{s}_hip"] + 4] = sol[0]
            q[a[f"{s}_knee"]], q[a[f"{s}_ankle"]], q[a[f"{s}_subtalar"]] = sol[1], min(max(sol[2], lo), hi), sol[3]
        return q

    def reach_z(self, xy, yaw, feet, planted):
        """Highest pelvis for which the planted feet stay on the floor: (nearest foot, every foot)."""
        Lr = 0.985 * (self.L1 + self.L2)
        R = rot2(yaw)
        zs = []
        for i in (0, 1):
            if planted[i]:
                d = np.linalg.norm(feet[i, :2] - (xy + R @ self.hip[i, :2]))
                zs.append(np.sqrt(max(Lr * Lr - d * d, 0.0)) + self.sole - self.hip[i, 2])
        if not zs:
            return np.inf, np.inf
        return max(zs), min(zs)

    def blend_legs(self, q_a, q_b, w):
        """Leg joints of q_a moved a fraction w towards those of q_b (returns a copy of q_a)."""
        q = q_a.copy()
        for s in "lr":
            a = self.adr[f"{s}_hip"]
            q[a:a + 4] = quat_slerp(q_a[a:a + 4], q_b[a:a + 4], w)
            for j in LEG[1:]:
                k = self.adr[f"{s}_{j}"]
                q[k] = q_a[k] + w * (q_b[k] - q_a[k])
        return q


# ---------------------------------------------------------------------- stances and the step generator
# feet frame: x to the player's right, y forwards; rows left foot, right (= racket) foot; then toe yaw
STANCES = {
    "ready": (((-0.29, -0.04), (0.29, 0.08)), (0.22, -0.22)),
    "wide": (((-0.39, -0.04), (0.39, 0.08)), (0.27, -0.27)),          # split-step landing: half the body height apart
    "walk": (((-0.10, 0.0), (0.10, 0.0)), (0.08, -0.08)),
    "overhead": (((-0.12, 0.24), (0.18, -0.30)), (-0.35, -0.95)),     # side-on, racket foot back
    "after": (((-0.18, -0.30), (0.14, 0.24)), (0.15, -0.10)),         # the scissor kick has swapped the feet
    "serve": (((-0.10, 0.22), (0.14, -0.18)), (-0.05, -0.60)),
}


class Step:
    __slots__ = ("t0", "T", "p0", "y0", "z0", "p1", "y1", "h", "fixed")

    def __init__(self, t0, T, p0, y0, z0, p1, y1, h, fixed):
        self.t0, self.T, self.p0, self.y0, self.z0 = t0, T, p0, y0, z0
        self.p1, self.y1, self.h, self.fixed = p1, y1, h, fixed

    def copy(self):
        return Step(self.t0, self.T, self.p0.copy(), self.y0, self.z0, self.p1.copy(), self.y1, self.h, self.fixed)


class Gait:
    """Where the feet are. A planted foot stays where it is; it is lifted when the pelvis has moved
    half a stride away from it and put down half a stride ahead of where the pelvis will be, so a
    slow pelvis gives chasse steps (the feet never pass each other), a fast one a run."""

    def __init__(self, scale=1.0):
        self.s = scale
        self.p = np.zeros((2, 2))
        self.yaw = np.zeros(2)
        self.z = np.zeros(2)
        self.pitch = np.zeros(2)
        self.sw = [None, None]
        self.t = None
        self.last = -1
        self.steps = 0

    def copy(self):
        g = Gait(self.s)
        g.p, g.yaw, g.z, g.pitch = self.p.copy(), self.yaw.copy(), self.z.copy(), self.pitch.copy()
        g.sw = [None if s is None else s.copy() for s in self.sw]
        g.t, g.last, g.steps = self.t, self.last, self.steps
        return g

    def homes(self, root, yaw, stance, narrow=0.0):
        """World positions and toe yaws of the two feet standing in `stance` around the pelvis `root`.
        narrow (0..1): the feet come in under the hips, as they do when running."""
        off, toe = STANCES[stance] if isinstance(stance, str) else stance
        off = np.asarray(off, float) * self.s
        if narrow > 0:
            off = off * [1.0 - 0.6 * narrow, 1.0 - 0.5 * narrow]
        return np.asarray(root, float) + off @ rot2(yaw).T, yaw + np.asarray(toe, float) * (1.0 - 0.7 * narrow)

    def reset(self, root, yaw, stance="ready", t=None):
        self.p, self.yaw = self.homes(root, yaw, stance)
        self.z[:] = 0
        self.pitch[:] = 0
        self.sw = [None, None]
        self.t = t

    def set_feet(self, p, yaw, z):
        """Adopt feet measured elsewhere (a camera player's own legs)."""
        self.p, self.yaw, self.z = np.array(p, float), np.array(yaw, float), np.array(z, float)
        self.pitch[:] = 0
        self.sw = [None, None]

    def start(self, i, t, target, yaw, T, h, fixed=True):
        self.sw[i] = Step(t, max(T, 0.05), self.p[i].copy(), float(self.yaw[i]), float(self.z[i]),
                          np.array(target, float), float(yaw), h, fixed)
        self.last = i
        self.steps += 1

    def hop(self, t, root, yaw, stance="wide", T=0.2, h=0.05):
        """Split step: both feet leave the floor together and land in `stance`."""
        H, Y = self.homes(root, yaw, stance)
        for i in (0, 1):
            self.start(i, t, H[i], Y[i], T, h * self.s)

    def planted(self):
        return [s is None for s in self.sw]

    def feet(self):
        return np.column_stack([self.p, self.z, self.yaw, self.pitch])

    def update(self, t, root, vel, yaw, stance="ready", auto=True):
        root, vel = np.asarray(root, float), np.asarray(vel, float)
        v = float(np.linalg.norm(vel))
        H, Y = self.homes(root, yaw, stance, sstep((v - 1.5) / 2.5))
        Ts = float(np.clip(0.30 - 0.028 * v, 0.16, 0.30))
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t
        for i in (0, 1):
            s = self.sw[i]
            if s is None:
                continue
            u = (t - s.t0) / s.T
            if not s.fixed and u < 0.7:                          # keep aiming at where the pelvis will be
                rem = max(s.t0 + s.T - t, 0.0)
                s.p1 += (H[i] + vel * (rem + 0.5 * Ts) - s.p1) * min(1.0, dt / 0.05)
                s.y1 = float(Y[i])
                s.h = max(s.h, self._arc(np.linalg.norm(s.p1 - s.p0)))
            if u >= 1.0:
                self.p[i], self.yaw[i], self.z[i], self.pitch[i] = s.p1, s.y1, 0.0, 0.0
                self.sw[i] = None
            else:
                e = u * u * u * (u * (6 * u - 15) + 10)
                self.p[i] = s.p0 + (s.p1 - s.p0) * e
                self.yaw[i] = s.y0 + wrap(s.y1 - s.y0) * e
                self.z[i] = s.z0 * (1 - u) + s.h * 4 * u * (1 - u)
                self.pitch[i] = 0.45 * np.sin(PI * u) * min(1.0, np.linalg.norm(s.p1 - s.p0) / (0.5 * self.s)) * (1 - u)
        if not auto:
            return
        allow = (True, True) if auto is True else auto
        free = [i for i in (0, 1) if self.sw[i] is None and allow[i]]
        if not free:
            return
        err = {i: float(np.linalg.norm(H[i] - self.p[i]) + 0.12 * abs(wrap(Y[i] - self.yaw[i]))) for i in free}
        thr = max(0.075 * self.s, 0.45 * v * Ts)
        cands = [i for i in free if err[i] > thr]
        if not cands:
            return
        if self.sw[0] is None and self.sw[1] is None:
            if len(cands) == 2 and abs(err[0] - err[1]) < 0.06:
                if v > 0.3:                                      # the foot on the side of travel leads
                    i = int(np.argmax([(H[k] - root) @ vel for k in (0, 1)]))
                else:
                    i = 1 - self.last if self.last >= 0 else 1
            else:
                i = max(cands, key=lambda k: err[k])
        else:
            i = cands[0]
            o = self.sw[1 - i]
            late = (t - o.t0) / o.T > float(np.clip(0.8 - 0.12 * (v - 2.2), 0.45, 0.8))   # the faster, the longer the flight
            if not ((late and v > 2.2) or err[i] > 0.45 * self.s):   # both feet in the air only when running
                                                                    # (or when the pelvis is leaving this foot behind)
                return
        tgt = H[i] + vel * (1.5 * Ts)
        self.start(i, t, tgt, Y[i], Ts, self._arc(np.linalg.norm(tgt - self.p[i])), fixed=False)

    def _arc(self, d):
        """How high a foot is lifted for a step of length d."""
        return (0.04 + 0.09 * min(1.0, d / (0.9 * self.s))) * self.s


def travel_turn(direction, facing_yaw, speed):
    """Feet-frame yaw offset while travelling: the body turns towards the corner it is going to
    (side-on for sideways and diagonal movement, square for straight forwards / backwards)."""
    if speed < 1e-6:
        return 0.0
    th = wrap(np.arctan2(direction[1], direction[0]) - (facing_yaw + PI / 2))
    return float(0.85 * np.sin(th) * sstep((speed - 1.2) / 2.0))


# ---------------------------------------------------------------------- upper body
class Body:
    """Authored upper-body poses of one character (root at the origin facing +y)."""

    # non-racket arm key poses: upper-arm and forearm directions in the torso frame (x right, y forward, z up)
    OFF = {
        "guard": ((-0.30, 0.22, -0.93), (0.10, 0.90, 0.42)),       # in front of the body, slightly out
        "point": ((-0.38, 0.50, 0.78), (-0.12, 0.42, 0.90)),       # raised at the falling shuttle
        "tuck": ((-0.32, 0.05, -0.95), (0.72, 0.62, 0.30)),        # pulled in as the body rotates through
        "balance": ((-0.66, -0.42, -0.62), (-0.62, -0.62, -0.48)), # extended behind: counterweight of a lunge
        "hold": ((-0.18, 0.50, -0.85), (0.22, 0.86, 0.46)),        # holding the shuttle out for a serve
        "relax": ((-0.10, 0.03, -0.99), (-0.04, 0.26, -0.96)),
        "run": ((-0.22, -0.20, -0.95), (0.05, 0.92, 0.38)),
    }
    # non-racket arm through a stroke: (seconds from contact, key pose)
    OFF_TIMELINE = {
        "overhead": ((-0.66, "guard"), (-0.44, "point"), (-0.22, "point"), (0.02, "tuck"), (0.45, "tuck")),
        "lunge": ((-0.50, "guard"), (-0.14, "balance"), (0.16, "balance"), (0.45, "guard")),
        "serve": ((-1.0, "hold"), (-0.14, "hold"), (0.15, "guard"), (0.45, "guard")),
    }

    # what a player does between rallies (the arcade layer of controller matches):
    # name -> lean, lumbar, neck (forward bends [rad]; neck > 0 lifts the chin), the two arms as (upper arm, forearm)
    # directions (l = the free arm, r = the racket arm; None = hanging), and what moves: (arm, axis index, amplitude, Hz)
    RELAX_R = ((0.10, 0.03, -0.99), (0.04, 0.26, -0.96))
    EMOTES = {
        # a clenched fist pumped in front of the chest (the hand stays below the shoulder) / the fist above the head
        "fist": dict(lean=0.05, lumbar=0.02, neck=0.10, l=((-0.26, 0.30, -0.92), (0.22, 0.62, 0.75)), wave=("l", 2, 0.30, 3.0)),
        "roar": dict(lean=-0.03, lumbar=-0.06, neck=0.18, l=((-0.22, 0.62, 0.12), (0.12, 0.18, 0.97)), wave=("l", 2, 0.22, 3.2)),
        "dejected": dict(lean=0.24, lumbar=0.22, neck=-0.55, l=((-0.08, 0.10, -0.99), (-0.02, 0.22, -0.97))),
        "shrug": dict(lean=0.0, lumbar=-0.04, neck=0.10, l=((-0.42, 0.12, -0.90), (-0.72, 0.52, 0.22)),
                      r=((0.42, 0.12, -0.90), (0.72, 0.52, 0.10)), wave=("lr", 2, 0.10, 1.3)),
        "impatient": dict(lean=0.02, lumbar=0.0, neck=0.05, l=((-0.62, -0.30, -0.72), (0.58, 0.42, -0.70))),
        "clock": dict(lean=0.06, lumbar=0.06, neck=-0.38, l=((-0.20, 0.42, -0.88), (0.80, 0.52, 0.30)), wave=("l", 2, 0.05, 2.4)),
        "appeal": dict(lean=0.0, lumbar=-0.05, neck=0.12, l=((-0.52, 0.22, 0.82), (-0.22, 0.18, 0.96)), wave=("l", 0, 0.22, 2.2)),
        "wipe": dict(lean=0.08, lumbar=0.08, neck=-0.22, l=((-0.30, 0.62, 0.40), (0.86, 0.10, 0.50)), wave=("l", 0, 0.10, 1.6)),
    }

    def __init__(self, rig: Rig, fk):
        self.rig = rig
        self.off = {k: arm_quat(*v) for k, v in self.OFF.items()}
        wrist_up = self._wrist(fk, ((0.26, 0.30, -0.92), (-0.30, 0.85, 0.43)), (-0.30, 0.45, 0.84))
        self.ready = self.pose(0.20, 0.10, 0.14, ((0.26, 0.30, -0.92), (-0.30, 0.85, 0.43)), wrist_up, "guard")
        self.relax = self.pose(0.04, 0.02, 0.0, self.RELAX_R, (0.0, 0.0, 0.0), "relax")
        # a dive: the body stretched out behind the racket arm, head up, the free arm ahead to break the fall
        self.off["dive"] = arm_quat((-0.30, 0.30, 0.90), (-0.10, 0.55, 0.83))
        self.dive = self.pose(0.0, -0.16, 0.55, ((0.20, 0.22, 0.95), (0.08, 0.30, 0.95)), (0.0, 0.25, 0.0), "dive")

    def emote(self, name, tau):
        """Upper-body pose of a gesture, tau seconds after it began."""
        e = self.EMOTES[name]
        r, q = self.rig, self.rig.q0.copy()
        q[3:7] = qaxis([1, 0, 0], -e["lean"])
        q[r.adr["lumbar"]:r.adr["lumbar"] + 4] = qaxis([1, 0, 0], -e["lumbar"])
        q[r.adr["neck"]:r.adr["neck"] + 4] = qaxis([1, 0, 0], e["neck"])
        who, ax, amp, hz = e.get("wave", ("", 0, 0.0, 0.0))
        for side, arm in (("l", e.get("l")), ("r", e.get("r") or self.RELAX_R)):
            u, f = np.array(arm[0], float), np.array(arm[1], float)
            if side in who:
                d = amp * np.sin(2 * PI * hz * tau)
                u[ax] += 0.5 * d
                f[ax] += d
            r.set_arm(q, side, *arm_quat(u, f))
        return q

    def pose(self, lean, lumbar, neck, racket_arm, wrist, off):
        r, q = self.rig, self.rig.q0.copy()
        q[3:7] = qaxis([1, 0, 0], -lean)
        q[r.adr["lumbar"]:r.adr["lumbar"] + 4] = qaxis([1, 0, 0], -lumbar)
        q[r.adr["neck"]:r.adr["neck"] + 4] = qaxis([1, 0, 0], neck)
        r.set_arm(q, "r", *arm_quat(*racket_arm), wrist=wrist)
        r.set_arm(q, "l", *self.off[off])
        return q

    def _wrist(self, fk, racket_arm, shaft_dir):
        """Pronation / wrist angles that hold the racket head along shaft_dir with the strings facing the net."""
        r, q = self.rig, self.rig.q0.copy()
        r.set_arm(q, "r", *arm_quat(*racket_arm))
        d = unit(shaft_dir)
        best, arg = -9.0, (0.0, 0.0, 0.0)
        rg = [r.rng[f"r_{n}"] for n in ARM[2:]]
        for pr in np.linspace(rg[0][0], rg[0][1], 13):
            for fe in np.linspace(rg[1][0], rg[1][1], 11):
                for dv in np.linspace(rg[2][0], rg[2][1], 5):
                    q[r.adr["r_pronation"]], q[r.adr["r_wrist_fe"]], q[r.adr["r_wrist_dev"]] = pr, fe, dv
                    R = fk(q)[1]
                    sc = R[:, 2] @ d + 0.3 * abs(R[1, 0]) - 0.1 * (abs(fe) + abs(dv))
                    if sc > best:
                        best, arg = sc, (float(pr), float(fe), float(dv))
        return arg

    def off_arm(self, kind, tau):
        """(shoulder quat, elbow) of the non-racket arm `tau` seconds from contact of a stroke of `kind`."""
        keys = self.OFF_TIMELINE[kind]
        if tau <= keys[0][0]:
            return self.off[keys[0][1]]
        for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
            if tau <= t1:
                f = sstep((tau - t0) / (t1 - t0))
                (qa, ea), (qb, eb) = self.off[a], self.off[b]
                return quat_slerp(qa, qb, f), ea + f * (eb - ea)
        return self.off[keys[-1][1]]


def set_root(q, xy, z, yaw, lean=None):
    """Place a pose built at the origin: position, heading, and optionally replace its forward lean."""
    q = q.copy()
    q[0], q[1], q[2] = xy[0], xy[1], z
    tilt = q[3:7] if lean is None else qaxis([1, 0, 0], -lean)
    q[3:7] = qmul(qaxis([0, 0, 1], yaw), tilt)
    return q


# ---------------------------------------------------------------------- pelvis motion
class Loco:
    """Pelvis position / velocity with human-like acceleration limits."""

    def __init__(self, pos, acc=15.0, dec=13.0):
        self.pos = np.array(pos, float)
        self.vel = np.zeros(2)
        self.acc, self.dec = acc, dec

    def copy(self):
        o = Loco(self.pos, self.acc, self.dec)
        o.vel = self.vel.copy()
        return o

    def seek(self, goal, vmax, dt):
        """Go to `goal` and stop there."""
        d = np.asarray(goal, float) - self.pos
        dist = float(np.linalg.norm(d))
        v_des = np.zeros(2) if dist < 1e-4 else d / dist * min(vmax, np.sqrt(2 * self.dec * dist), dist / max(dt, 1e-4))
        self._steer(v_des, dt)
        return dist

    def drive(self, v_cmd, dt):
        """Follow a commanded velocity (stick input)."""
        self._steer(np.asarray(v_cmd, float), dt)

    def _steer(self, v_des, dt):
        dv = v_des - self.vel
        n = float(np.linalg.norm(dv))
        lim = (self.acc if np.linalg.norm(v_des) >= np.linalg.norm(self.vel) else max(self.acc, self.dec)) * dt
        if n > lim:
            dv *= lim / n
        self.vel = self.vel + dv
        self.pos = self.pos + self.vel * dt


def travel_dist(T, vmax, acc, dec):
    """Distance covered in T seconds from rest to rest with these limits."""
    if T <= 0:
        return 0.0
    k = 0.5 * (1.0 / acc + 1.0 / dec)
    return vmax * T - vmax * vmax * k if T >= 2 * k * vmax else T * T / (4 * k)


class Carry:
    """The legs of a player whose pose comes from a camera or a recording.

    Such a player stands in a living room (or was filmed from one viewpoint and scaled to the court) while
    the avatar covers a court: the pelvis is moved by more than the person's own steps, so the tracked legs
    would glide over the floor. While the pelvis travels, the legs are therefore generated by the same step
    generator as the computer players' (feet planted, chasse steps or a run by speed); when it comes to rest
    the person's own legs fade back in, so lunges, squats and jumps on the spot stay the person's."""
    ON, OFF = 0.6, 0.25              # m/s pelvis speed at which the stepping legs take over / hand back

    def __init__(self, m, d, ci, facing=0.0, crouch=0.10):
        """crouch [of the body scale]: how far the pelvis may come down at a run (0: the person's height, exactly)."""
        self.m, self.d, self.ci = m, d, ci
        self.crouch = crouch
        self.rig = Rig(m, ci.prefix)
        B = mujoco.mjtObj.mjOBJ_BODY
        self._foot = [mujoco.mj_name2id(m, B, ci.prefix + n) for n in ("l_foot", "r_foot")]
        self.facing = float(facing)
        off, toe = STANCES["ready"]
        if ci.hand == "l":                                      # the racket foot is the one half a step ahead
            off, toe = ((-off[1][0], off[1][1]), (-off[0][0], off[0][1])), (-toe[1], -toe[0])
        self.stance = (off, toe)
        self.gait = Gait(self.rig.scale)
        self.w = 0.0
        self.on = False
        self.xy = None
        self.vel = np.zeros(2)
        self.yaw = self.facing
        self._slow_t = 0.0

    def feet_of(self, q):
        """World floor points, toe yaws and heights of the two feet of pose q."""
        self.d.qpos[self.ci.qslice] = q
        mujoco.mj_kinematics(self.m, self.d)
        P, Y, Z = np.zeros((2, 2)), np.zeros(2), np.zeros(2)
        for i, b in enumerate(self._foot):
            R = self.d.xmat[b].reshape(3, 3)
            P[i], Y[i] = self.d.xpos[b][:2], np.arctan2(-R[0, 1], R[1, 1])
            lift = self.d.xpos[b][2] - self.rig.sole
            Z[i] = lift if lift > 0.05 else 0.0
        return P, Y, Z

    def step(self, q, now, dt):
        """q: tracked pose, pelvis already placed on the court. Returns the pose to show."""
        xy = q[:2].copy()
        if self.xy is None or dt <= 0:
            self.xy = xy
            return q
        s = self.rig.scale
        self.vel += ((xy - self.xy) / dt - self.vel) * (1.0 - np.exp(-dt / 0.08))
        self.xy = xy
        sp = float(np.linalg.norm(self.vel))
        self._slow_t = self._slow_t + dt if sp < self.OFF else 0.0
        if not self.on and sp > self.ON:
            self.on = True
            if self.w <= 0.0:
                self.gait.set_feet(*self.feet_of(q))            # start from where the person's feet are
                self.gait.t = now - dt
                self.yaw = self.facing + float(np.clip(wrap(yaw_of_quat(q[3:7]) - self.facing), -1.0, 1.0))
        elif self.on and self._slow_t > 0.3:
            self.on = False
        self.w = float(np.clip(self.w + (dt / 0.12 if self.on else -dt / 0.35), 0.0, 1.0))
        if self.w <= 0.0:
            return q
        yaw = self.facing + float(np.clip(wrap(yaw_of_quat(q[3:7]) - self.facing), -1.0, 1.0))
        self.yaw += wrap(yaw - self.yaw) * (1.0 - np.exp(-dt / 0.15))
        g = self.gait
        g.update(now, xy, self.vel, self.yaw, self.stance, auto=self.on)
        w = sstep(self.w)
        qg = q.copy()
        # nobody runs with straight legs, but a person tracked standing in a room is upright: at speed the
        # pelvis comes down towards the ready height (by `crouch` at most -- it carries the racket). Where it
        # is still too high for a foot to stay down (a jump), the foot leaves the floor instead of the leg
        # overstretching
        over = q[2] - (self.rig.z_stand - 0.10 * s)
        qg[2] = q[2] - w * min(sp / 2.0, 1.0) * min(max(over, 0.0), self.crouch * s)
        feet = g.feet()
        Lr = 0.985 * (self.rig.L1 + self.rig.L2)
        R = rot2(yaw_of_quat(q[3:7]))
        for j in (0, 1):
            hz = qg[2] + self.rig.hip[j, 2] - self.rig.sole
            if hz > 0.93 * Lr:
                dh = min(float(np.linalg.norm(feet[j, :2] - (xy + R @ self.rig.hip[j, :2]))), 0.95 * Lr)
                feet[j, 2] = max(feet[j, 2], hz - np.sqrt(Lr * Lr - dh * dh))
        return self.rig.blend_legs(qg, self.rig.set_legs(qg.copy(), feet), w)


def profile(u, a=0.3):
    """Distance fraction covered at time fraction u: accelerate over a, cruise, brake over a (C1)."""
    u = min(max(float(u), 0.0), 1.0)
    if u < a:
        return u * u / (2 * a * (1 - a))
    if u > 1 - a:
        return 1 - (1 - u) ** 2 / (2 * a * (1 - a))
    return (u - a / 2) / (1 - a)

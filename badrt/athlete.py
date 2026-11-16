"""A player's body: where it stands, how it gets to the shuttle, and what the legs, the trunk
and the non-racket arm do through a stroke.

The stroke library contributes what a monocular video can be trusted with: the racket arm, the
trunk bend and the rotation of the hips through the stroke (Clip.U). Everything else comes from
footwork.py and the timeline built here:

    plan (opponent has hit)      split step: a shallow hop landing wide
    + reaction                   push off; chasse / running steps carry the pelvis along a
                                 minimum-time profile (accelerate, cruise, brake) that respects
                                 the player's acceleration and top speed
    lunge strokes (mid / low)    the racket foot is thrown at the shuttle and lands as it is
                                 struck, toes pointing at it; hips sink, rear arm goes back
    overhead strokes             side-on with the racket foot back, the other arm pointing up;
                                 scissor kick (feet swap in the air) through the hit, with a
                                 real jump when the shuttle is higher than the standing reach
    afterwards                   back to the base in the ready stance (online: Athlete.step)

The timeline is baked on a 10 ms grid (Script) before the arm is solved, so the swing planner
sees exactly the pelvis the game will show; the legs are filled in afterwards (they do not move
the racket). Everything here is for the internal right-handed character.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .character import blend_qpos, quat_slerp
from .footwork import (PI, Body, Gait, Loco, Rig, arm_quat, q2m, qaxis, qmul, rot2, set_root, split_yaw_tilt, sstep,
                       travel_turn, unit, wrap, yaw_of_quat)
from .library import Segment, align_quats, sample_clip

G = 9.81
DT = 0.01                    # bake grid [s]
JUMP_MAX = 0.32              # m (times the leg scale)
ARM_RES = 0.30               # m of height the arm warp may be asked to absorb


def _soft(x, lim):
    return lim * np.tanh(np.asarray(x, float) / lim)


def _limit_quat(q, lim):
    """Same axis, rotation angle soft-limited to lim."""
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    sn = float(np.linalg.norm(q[1:]))
    if sn < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = 2 * np.arctan2(sn, q[0])
    a2 = float(_soft(ang, lim))
    return np.r_[np.cos(a2 / 2), q[1:] / sn * np.sin(a2 / 2)]


def _smooth(x, n=3):
    k = np.ones(n) / n
    return np.convolve(np.pad(x, n // 2, mode="edge"), k, mode="valid")


@dataclass
class Clip:
    seg: Segment
    U: np.ndarray            # (T, 45) upper body of the stroke: column 0 = pelvis yaw relative to the contact frame,
    #                          root quaternion = pelvis tilt, lumbar, racket arm; the rest is the ready pose
    fps: float
    contact: int
    c_loc: np.ndarray        # racket head centre at contact relative to the pelvis (x right, y forward, z up)
    n_loc: np.ndarray        # string-bed normal at contact
    h_rel: float             # = c_loc[2]
    kind: str                # overhead | mid | low: which footwork goes with it
    side: str                # forehand | backhand
    z_rng: tuple             # pelvis heights this stroke is played from (feet on the floor)
    height: float            # contact height from the middle of that range
    s_loc: np.ndarray = None # racket shoulder at contact relative to the pelvis


@dataclass
class Snap:
    """Body state at one instant: what a new plan starts from."""
    xy: np.ndarray
    vel: np.ndarray
    z: float
    turn: float
    yaw_off: float
    gait: Gait
    q: np.ndarray


class Travel:
    """Straight move of length d in minimum time: accelerate at A to at most vmax, brake at D to
    v_end on arrival (a lunge is hit on the move) and to rest just after."""

    def __init__(self, d, A, D, vmax, v_end):
        d = max(float(d), 0.0)
        vf = np.sqrt(2 * A * d)
        if vf <= v_end:
            vp = ve = vf
        else:
            ve = v_end
            vp = min(np.sqrt((2 * d + ve * ve / D) / (1 / A + 1 / D)), vmax)
        self.A, self.D, self.vp, self.ve = A, D, vp, ve
        self.ta, self.da = vp / A, vp * vp / (2 * A)
        self.td = (vp - ve) / D
        self.dc = max(d - self.da - (vp * vp - ve * ve) / (2 * D), 0.0)
        self.tc = self.dc / max(vp, 1e-9)
        self.T = self.ta + self.tc + self.td
        self.tb = vp / D                                         # braking all the way to rest

    def at(self, t):
        """(distance, speed) t seconds after the start."""
        if t <= 0:
            return 0.0, 0.0
        if t < self.ta:
            return 0.5 * self.A * t * t, self.A * t
        t -= self.ta
        if t < self.tc:
            return self.da + self.vp * t, self.vp
        t = min(t - self.tc, self.tb)
        return self.da + self.dc + self.vp * t - 0.5 * self.D * t * t, self.vp - self.D * t


class Script:
    """Baked body timeline of one stroke."""

    def __init__(self, ath, clip, kind, t_plan, t_hit, t_start, t_end, k0, rate, tg, XY, Z, TURN, FEET, snaps,
                 up_plan, dyaw0, info):
        self.ath, self.clip, self.kind = ath, clip, kind
        self.t_plan, self.t_hit, self.t_start, self.t_end, self.k0, self.rate = t_plan, t_hit, t_start, t_end, k0, rate
        self.tg, self.XY, self.Z, self.TURN, self.FEET, self.snaps = tg, XY, Z, TURN, FEET, snaps
        self.up_plan, self.dyaw0, self.info = up_plan, dyaw0, info
        sl = ath.rig.arm_slice("l")
        self._l0 = up_plan[sl].copy()
        self._lsl = sl
        self.hold = None
        self._cache = {}

    def covers(self, t):
        return self.tg[0] <= t <= self.tg[-1]

    def _at(self, t):
        tg = self.tg
        x = (min(max(t, tg[0]), tg[-1]) - tg[0]) / (tg[1] - tg[0])
        i = min(int(x), len(tg) - 2)
        return i, x - i

    def base(self, t):
        """Pose at time t before the arm is warped (legs are filled in later). Cached: the swing builder asks
        for the same times again when it corrects its command; the result must not be modified."""
        q = self._cache.get(t)
        if q is None:
            q = self._cache[t] = self._base(t)
        return q

    def _base(self, t):
        a = self.ath
        i, f = self._at(t)
        xy = self.XY[i] + f * (self.XY[i + 1] - self.XY[i])
        z = self.Z[i] + f * (self.Z[i + 1] - self.Z[i])
        turn = self.TURN[i] + f * (self.TURN[i + 1] - self.TURN[i])
        el = max(t - self.t_plan, 0.0)
        q = blend_qpos(self.up_plan, a.body.ready, 1.0 - np.exp(-6.0 * el))
        yaw = 0.6 * turn + self.dyaw0 * np.exp(-el / 0.15)
        if t >= self.t_start:
            c = sample_clip(self.clip.U, self.k0 + (t - self.t_start) * self.clip.fps * self.rate)
            yc, c[0] = c[0], 0.0
            b = sstep((t - self.t_start) / 0.2)
            q = blend_qpos(q, c, b) if b < 1.0 else c
            yaw = (1 - b) * yaw + b * yc
        qa, el_a = a.body.off_arm(self.kind, t - self.t_hit)
        w = sstep(el / 0.22)
        s0 = self._lsl.start
        q[s0:s0 + 4] = quat_slerp(self._l0[:4], qa, w) if w < 1.0 else qa
        q[s0 + 4] = self._l0[4] + w * (el_a - self._l0[4])
        q[s0 + 5:s0 + 8] = self._l0[5:8] * (1 - w)
        q[0], q[1], q[2] = xy[0], xy[1], z
        q[3:7] = qmul(qaxis((0.0, 0.0, 1.0), a.facing + yaw), q[3:7])
        if self.hold is not None:                         # serve: the free hand holds the shuttle out until it is dropped
            wh = sstep(el / 0.35) * (1.0 - sstep((t - self.t_hit + 0.14) / 0.16))
            if wh > 0:
                qa, el_a = a.hand_to(q, self.hold)
                q[s0:s0 + 4] = quat_slerp(q[s0:s0 + 4], qa, wh)
                q[s0 + 4] += wh * (el_a - q[s0 + 4])
        return q

    def feet(self, t):
        i, f = self._at(t)
        return self.FEET[i] + f * (self.FEET[i + 1] - self.FEET[i])

    def fill(self, swing):
        """Put the legs into the nodes of the finished swing."""
        rig = self.ath.rig
        for j, t in enumerate(swing.t):
            rig.set_legs(swing.q[j], self.feet(t))

    def snap(self, t, q):
        i, f = self._at(t)
        a = self.ath
        vel = (self.XY[i + 1] - self.XY[i]) / (self.tg[i + 1] - self.tg[i])
        return Snap(self.XY[i] + f * (self.XY[i + 1] - self.XY[i]), vel, float(self.Z[i] + f * (self.Z[i + 1] - self.Z[i])),
                    float(self.TURN[i] + f * (self.TURN[i + 1] - self.TURN[i])),
                    float(wrap(yaw_of_quat(q[3:7]) - a.facing)), self.snaps[i if f < 0.5 else i + 1].copy(), q.copy())


class DiveScript(Script):
    """Baked timeline of a dive: the body is thrown at the shuttle behind the racket arm, lands flat, slides
    and gets up again. The pelvis is pitched about the horizontal axis across the direction of the dive."""

    def __init__(self, ath, t_plan, t_hit, t_end, tg, XY, Z, YAW, PITCH, W, FEET, snaps, up_plan, axis, info):
        super().__init__(ath, None, "dive", t_plan, t_hit, t_plan, t_end, 0.0, 1.0, tg, XY, Z, np.zeros(len(tg)), FEET, snaps,
                         up_plan, 0.0, info)
        self.YAW, self.PITCH, self.W, self.axis = YAW, PITCH, W, np.asarray(axis, float)

    def _base(self, t):
        a = self.ath
        i, f = self._at(t)
        xy = self.XY[i] + f * (self.XY[i + 1] - self.XY[i])
        z = self.Z[i] + f * (self.Z[i + 1] - self.Z[i])
        yaw = self.YAW[i] + f * (self.YAW[i + 1] - self.YAW[i])
        pitch = self.PITCH[i] + f * (self.PITCH[i + 1] - self.PITCH[i])
        w = self.W[i] + f * (self.W[i + 1] - self.W[i])
        q = blend_qpos(self.up_plan, a.body.ready, 1.0 - np.exp(-6.0 * max(t - self.t_plan, 0.0)))
        if w > 0:
            q = blend_qpos(q, a.body.dive, w)
        q[0], q[1], q[2] = xy[0], xy[1], z
        q[3:7] = qmul(qaxis(self.axis, pitch), qmul(qaxis((0.0, 0.0, 1.0), a.facing + yaw), q[3:7]))
        return q


def _keys(keys, t):
    """Smoothstep interpolation through (time, value) keys, at the times t."""
    ks = [keys[0]]
    for k in keys[1:]:
        if k[0] > ks[-1][0] + 0.02:
            ks.append(k)
    T, V = np.array([k[0] for k in ks]), np.array([k[1] for k in ks])
    if len(T) == 1:
        return np.full(len(t), V[0])
    i = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2)
    u = np.clip((t - T[i]) / (T[i + 1] - T[i]), 0.0, 1.0)
    return V[i] + (V[i + 1] - V[i]) * u * u * (3 - 2 * u)


class Athlete:
    def __init__(self, ik, facing, base, v_max, reaction, reach=0.9):
        self.ik = ik
        self.rig = Rig(ik.m)
        self.body = Body(self.rig, ik.fk)
        self.facing = float(facing)
        bid = lambda n: mujoco.mj_name2id(ik.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self._torso, self._lua, self._rua = bid("torso"), bid("l_upperarm"), bid("r_upperarm")
        self._arm = (float(-ik.m.body_pos[bid("l_forearm")][2]), float(-ik.m.body_pos[bid("l_hand")][2]) + 0.08)
        self.fwd = np.array([-np.sin(facing), np.cos(facing)])
        self.R2 = rot2(facing)
        s = self.rig.scale
        self.z_ready = self.rig.z_stand - 0.09 * s
        self.z_relax = self.rig.z_stand - 0.015 * s
        self.z_run = self.rig.z_stand - 0.11 * s
        self.v0, self.k = v_max, 1.0                     # the player's own top speed; set_speed scales it (fatigue)
        self.v_max, self.acc, self.dec = v_max, 2.9 * v_max, 2.6 * v_max
        self.v_end = 0.4 * v_max
        self.reaction = reaction
        self.emote_now = None                            # (name, start, end): a gesture between rallies
        self.hop_now = None                              # (take-off time, half the air time, height)
        self.crouch = False                              # gathering for a jump
        self.bounds = None                               # (x half-width, y min, y max) the pelvis is kept inside
        self.dz = 0.0
        self.slack = 0.28 * reach * s                    # the arm and trunk stretch when the feet are late
        self._d = np.linspace(0.0, 10.0, 201)
        self._T = np.array([Travel(d, self.acc, self.dec, v_max, self.v_end).T for d in self._d])    # hit on the move (lunge)
        self._T0 = np.array([Travel(d, self.acc, self.dec, v_max, 0.0).T for d in self._d])          # stop under it (overhead)
        # longest shoulder-to-string-bed distance: straight arm, over the wrist's range (+ 5 cm of trunk)
        q, far = self.rig.q0.copy(), 0.0
        for fe in np.linspace(*self.rig.rng["r_wrist_fe"], 7):
            for dv in np.linspace(*self.rig.rng["r_wrist_dev"], 5):
                q[self.rig.adr["r_wrist_fe"]], q[self.rig.adr["r_wrist_dev"]] = fe, dv
                c = ik.fk(q)[0]
                far = max(far, float(np.linalg.norm(c - ik.d.xpos[self._rua])))
        self.arm_len = 0.97 * far + 0.05
        self.base = np.array(base, float)
        self.goal, self.goal_speed, self.mode = self.base.copy(), 0.8 * v_max, "ready"
        self.drive = None                                # commanded velocity (stick), overrides the goal
        self.script: Script | None = None
        self.gait = Gait(s)
        self.reset(self.base)

    # ------------------------------------------------------------------ clips
    def clip(self, seg: Segment) -> Clip:
        rig, s = self.rig, self.rig.scale
        Q = align_quats(seg.qpos)
        n, c = len(Q), int(seg.contact)
        yaw = np.unwrap([yaw_of_quat(q[3:7]) for q in Q])
        yaw = _smooth(_soft(yaw - yaw[c], 1.3))
        yaw -= yaw[c]
        U = np.tile(self.body.ready, (n, 1))
        la, ra = rig.adr["lumbar"], rig.arm_slice("r")
        for k in range(n):
            U[k, 3:7] = _limit_quat(split_yaw_tilt(Q[k, 3:7])[1], 0.5)
            U[k, la:la + 4] = _limit_quat(Q[k, la:la + 4], 0.6)
            U[k, ra] = Q[k, ra]
        U = align_quats(U)
        cc, R = self.ik.fk(U[c])
        s_loc = self.ik.d.xpos[self._rua].copy()
        U[:, 0] = yaw
        h = float(cc[2])
        zs = rig.z_stand
        if h >= 0.95 * s:
            kind, rng = "overhead", (zs - 0.10 * s, zs - 0.02 * s)
        elif h < 0.16 * s:
            kind, rng = "low", (zs - 0.42 * s, zs - 0.20 * s)
        else:
            kind, rng = "mid", (zs - 0.24 * s, zs - 0.07 * s)
        return Clip(seg, U, float(seg.fps), c, cc, R[:, 0].copy(), h, kind, "forehand" if cc[0] >= 0 else "backhand",
                    rng, h + 0.5 * (rng[0] + rng[1]), s_loc)

    def arm_reaches(self, clip: Clip, root, z, p):
        """Is the contact point p within the racket arm's length of where this stroke puts the shoulder
        (pelvis at root, height z)? The trunk gives a little."""
        sh = np.array([root[0], root[1], z]) + np.r_[self.R2 @ clip.s_loc[:2], clip.s_loc[2]]
        return float(np.linalg.norm(np.asarray(p, float) - sh)) <= self.arm_len

    def fit(self, clip: Clip, z_contact, avail, may_jump=True):
        """Pelvis height (and jump) from which this clip meets a shuttle at z_contact:
        (pelvis z on the ground, jump, height left to the arm), or None."""
        s = self.rig.scale
        zr = z_contact - clip.h_rel
        z_p = float(np.clip(zr, *clip.z_rng))
        res, jump = zr - z_p, 0.0
        if may_jump and clip.kind == "overhead" and res > 0.10 * s:
            jump = min(res - 0.10 * s, JUMP_MAX * s, 0.5 * G * max(avail - 0.25, 0.0) ** 2)
            res -= jump
        if abs(res) > ARM_RES * s:
            return None
        return z_p, float(jump), float(res)

    def set_speed(self, k):
        """Top speed = k times the player's own (the controller's share, fatigue); the accelerations go with it, so the
        travel tables stay exact: T_k(d) = T_1(d / k)."""
        k = float(k)
        if abs(k - self.k) > 1e-4:
            self.k = k
            self.v_max, self.acc, self.dec, self.v_end = self.v0 * k, 2.9 * self.v0 * k, 2.6 * self.v0 * k, 0.4 * self.v0 * k

    def travel_time(self, d, rest=False):
        return float(np.interp(d / self.k, self._d, self._T0 if rest else self._T))

    def travel_range(self, T, rest=False):
        return self.k * float(np.interp(T, self._T0 if rest else self._T, self._d))

    def coast(self, vel):
        """Where the current momentum carries the pelvis while it is braked."""
        return 0.5 * np.asarray(vel, float) * (np.linalg.norm(vel) / self.dec)

    # ------------------------------------------------------------------ state
    def reset(self, xy, t=None):
        self.loco = Loco(xy, self.acc, self.dec)
        self.gait.reset(xy, self.facing, "ready", t)
        self.z, self.turn, self.yaw_off = self.z_ready, 0.0, 0.0
        self.up = self.body.ready.copy()
        self.script = None
        self.t = t
        self.hop_now, self.dz, self.crouch = None, 0.0, False
        self.q = self.compose()
        return self.q

    def compose(self):
        q = set_root(self.up, self.loco.pos, self.z + self.dz, self.facing + self.yaw_off)
        feet = self.gait.feet()
        if self.dz > 0:                                  # off the floor: the feet come up with the body, toes down
            feet[:, 2] += max(self.dz - 0.02, 0.0)
            feet[:, 4] = 0.6 * min(self.dz / 0.1, 1.0)
        return self.rig.set_legs(q, feet)

    def emote(self, name, t, seconds=2.2):
        self.emote_now = (name, t, t + seconds)

    def hop(self, t, h=None):
        """A jump on the spot (nothing to hit): take-off after a 0.08 s push."""
        h = 0.24 * self.rig.scale if h is None else h
        self.hop_now = (t + 0.08, float(np.sqrt(2 * h / G)), h)

    def airborne(self, t):
        return self.hop_now is not None and t < self.hop_now[0] + 2 * self.hop_now[1]

    def strip(self, q):
        up = q.copy()
        up[:3] = 0.0
        up[3:7] = split_yaw_tilt(q[3:7])[1]
        return up

    def snap(self, t, q):
        if self.script is not None and self.script.covers(t):
            return self.script.snap(t, q)
        back = 0.0 if self.t is None else min(max(self.t - t, 0.0), 0.05)     # the online state is one frame ahead
        return Snap(self.loco.pos - self.loco.vel * back, self.loco.vel.copy(), float(self.z), float(self.turn),
                    float(wrap(yaw_of_quat(q[3:7]) - self.facing)), self.gait.copy(), q.copy())

    def adopt(self, t, q):
        """Leave the script at time t (pose q) and carry on online from there."""
        sn = self.script.snap(t, q)
        self.loco.pos, self.loco.vel = sn.xy.copy(), sn.vel.copy()
        self.gait, self.z, self.turn, self.yaw_off = sn.gait, sn.z, sn.turn, sn.yaw_off
        self.up = self.strip(q)
        self.script = None
        self.t = t

    def hand_to(self, q, target):
        """(shoulder quat, elbow) that put the free hand of pose q on a world point: two-bone IK with the
        elbow hanging down and out."""
        m, d = self.ik.m, self.ik.d
        self.ik.fk(q)
        R, sh = d.xmat[self._torso].reshape(3, 3), d.xpos[self._lua]
        v = R.T @ (np.asarray(target, float) - sh)
        Lu, Lf = self._arm
        dist = float(np.clip(np.linalg.norm(v), 0.25 * (Lu + Lf), 0.98 * (Lu + Lf)))
        vh = unit(v)
        al = np.arccos(np.clip((Lu * Lu + dist * dist - Lf * Lf) / (2 * Lu * dist), -1.0, 1.0))
        hint = np.array([-0.5, -0.3, -1.0])
        n = unit(hint - (hint @ vh) * vh)
        u = np.cos(al) * vh + np.sin(al) * n
        return arm_quat(u, vh * dist - Lu * u)

    def _keep_side(self, xy, margin=0.30):
        over = xy @ self.fwd + margin
        return xy - self.fwd * over if over > 0 else xy

    def _support(self, xy, z, g, drag=True):
        """Pelvis height the planted legs allow (nearest foot decides, never below a deep lunge);
        a foot the pelvis has left too far behind is dragged along, as the rear foot of a lunge is."""
        pl = g.planted()
        if not any(pl):
            return z
        rig = self.rig
        Lr = 0.985 * (rig.L1 + rig.L2)
        hips = [xy + self.R2 @ rig.hip[j, :2] for j in (0, 1)]
        dist = [float(np.linalg.norm(g.p[j] - hips[j])) if pl[j] else np.inf for j in (0, 1)]
        jn = int(np.argmin(dist))
        zlim = np.sqrt(max(Lr * Lr - dist[jn] ** 2, 0.0)) + rig.sole - rig.hip[jn, 2]
        z = max(min(z, zlim), z - 0.06 * rig.scale)
        for j in (0, 1):
            if pl[j] and drag:
                hz = z + rig.hip[j, 2] - rig.sole
                dmax = np.sqrt(max(Lr * Lr - hz * hz, 0.0)) + (0.0 if j == jn else 0.06)
                if dist[j] > dmax:
                    g.p[j] = hips[j] + (g.p[j] - hips[j]) * (dmax / dist[j])
        return float(z)

    def go(self, xy, speed=None, mode="ready"):
        self.goal = np.array(xy, float)
        self.goal_speed = 0.8 * self.v_max if speed is None else speed
        self.mode = mode

    def arrived(self, tol=0.12):
        return np.linalg.norm(self.goal - self.loco.pos) < tol and np.linalg.norm(self.loco.vel) < 0.4

    # ------------------------------------------------------------------ online motion (no stroke in progress)
    def step(self, now):
        dt = 0.0 if self.t is None else max(now - self.t, 0.0)
        self.t = now
        lo, s = self.loco, self.rig.scale
        relax = self.mode == "relax"
        lo.acc = 5.0 if relax else 0.6 * self.acc
        lo.dec = 6.0 if relax else 0.6 * self.dec
        if dt > 0:
            if self.drive is not None:
                lo.drive(self.drive, dt)
            else:
                lo.seek(self.goal, self.goal_speed, dt)
            lo.pos = self._keep_side(lo.pos)
            if self.bounds is not None:
                bx, y0, y1 = self.bounds
                lo.pos = np.array([min(max(lo.pos[0], -bx), bx), min(max(lo.pos[1], y0), y1)])
        sp = float(np.linalg.norm(lo.vel))
        k = 1.0 - np.exp(-dt / 0.12)
        self.turn += (travel_turn(lo.vel, self.facing, sp) - self.turn) * k
        far = np.linalg.norm(self.goal - lo.pos) > 0.5
        stance = "serve" if self.mode == "serve" else ("walk" if relax and (far or sp > 0.5) else "ready")
        self.gait.update(now, lo.pos, lo.vel, self.facing + self.turn, stance)
        tgt, rate = (self.body.relax, 3.5) if relax else (self.body.ready, 6.0)
        if self.emote_now is not None:
            name, e0, e1 = self.emote_now
            if now >= e1:
                self.emote_now = None
            elif now >= e0:
                tgt, rate = self.body.emote(name, now - e0), 7.0
        self.up = blend_qpos(self.up, tgt, 1.0 - np.exp(-rate * dt))
        self.yaw_off += (0.6 * self.turn - self.yaw_off) * (1.0 - np.exp(-dt / 0.15))
        z_t = (self.z_relax if relax else self.z_ready) - 0.02 * s * min(sp / 3.0, 1.0) - (0.09 * s if self.crouch else 0.0)
        self.dz = 0.0
        if self.hop_now is not None:
            h0, th, hh = self.hop_now
            u = (now - h0) / (2 * th)
            if u >= 1.0:
                self.hop_now = None
            elif u > 0:
                self.dz = hh * 4 * u * (1 - u)
            else:
                z_t -= 0.10 * s                          # the push before the take-off
        self.z += (self._support(lo.pos, z_t, self.gait, drag=False) - self.z) * (1.0 - np.exp(-dt / 0.10))
        self.z = self._support(lo.pos, self.z, self.gait)
        self.q = self.compose()
        return self.q

    # ------------------------------------------------------------------ a stroke
    def plan(self, snap: Snap, t_plan, clip: Clip, root_hit, t_hit, z_p, jump=0.0, serve=False, hold=None) -> Script:
        rig, s, facing = self.rig, self.rig.scale, self.facing
        fps = clip.fps
        pre = clip.contact / fps
        avail = max(t_hit - t_plan, 1e-3)
        rate, k0 = 1.0, 0.0
        if pre > avail:
            rate = min(pre / avail, 1.8)
            k0 = max(0.0, clip.contact - avail * rate * fps)
        t_start = t_hit - (clip.contact - k0) / (fps * rate)
        t_end = min(t_start + (len(clip.U) - 1 - k0) / (fps * rate), t_hit + 0.34)
        overhead = clip.kind == "overhead" and not serve
        lunge = not overhead and not serve
        kind = "serve" if serve else ("overhead" if overhead else "lunge")

        # ---- pelvis path: what is left of the current momentum, plus a minimum-time move
        here, v0 = snap.xy, snap.vel
        sp0 = float(np.linalg.norm(v0))
        t_c = sp0 / self.dec
        dvec = np.asarray(root_hit, float) - here - 0.5 * v0 * t_c
        d = float(np.linalg.norm(dvec))
        u = dvec / d if d > 1e-6 else np.zeros(2)
        t_go = min(t_plan + (0.0 if serve else self.reaction), t_hit - 0.03)
        t_arr = t_hit - 0.02
        # a lunge is hit on the move; for an overhead the player has to be standing under the shuttle
        tr = Travel(d, self.acc, self.dec, self.v_max, self.v_end if lunge else 0.0)
        kt = max(t_arr - t_go, 0.02) / max(tr.T, 1e-6)
        if kt > 1.0:                                     # time to spare: less than full effort
            kt = min(kt, 2.0 if lunge else 1.35)
        t_mv = t_arr - kt * tr.T if lunge else t_go      # a lunge arrives with the shuttle, an overhead waits under it

        def root(t):
            tau = min(max(t - t_plan, 0.0), t_c)
            xy = here + v0 * (tau - 0.5 * tau * tau / t_c) if t_c > 1e-6 else here.copy()
            vel = v0 * (1.0 - tau / t_c) if t_c > 1e-6 and tau < t_c else np.zeros(2)
            ds, v = tr.at((t - t_mv) / kt)
            xy = xy + u * ds
            over = xy @ self.fwd + 0.30
            if over > 0:
                return xy - self.fwd * over, np.zeros(2)
            return xy, vel + u * (v / kt)

        # ---- timing of the feet
        hop = (not serve) and avail >= 0.45 and all(snap.gait.planted()) and sp0 < 2.0
        T_hop = float(np.clip(self.reaction + 0.07, 0.15, 0.22))
        hop_t1 = t_plan + T_hop
        T_half = np.sqrt(2 * jump / G) if jump > 0.02 else 0.0
        T_sc = 2 * max(T_half, 0.12)
        t_to, t_ld = t_hit - T_sc / 2, t_hit + T_sc / 2
        t_side = max(t_mv + 0.45 * kt * tr.T, t_hit - 0.9)
        low = float(np.clip((rig.z_stand - z_p) / s, 0.0, 0.45))
        reach = s * float(np.clip(0.26 + 1.0 * low, 0.30, 0.68))
        t_L1 = t_hit - 0.03
        t_L0 = max(t_L1 - (0.26 + 0.2 * reach), (hop_t1 if hop else t_plan) + 0.02)
        cdir = unit(self.R2 @ clip.c_loc[:2])
        end_xy = root(t_end)[0]
        tgt = end_xy + cdir * reach
        over = tgt @ self.fwd + 0.22
        if over > 0:
            tgt = tgt - self.fwd * over
        yaw_foot = facing + float(np.clip(wrap(np.arctan2(cdir[1], cdir[0]) - PI / 2 - facing), -1.25, 1.25))

        # ---- pelvis height
        zk = [(t_plan, snap.z)]
        if hop:
            zk.append((hop_t1, self.z_ready - 0.05 * s))
        if lunge:
            zk += [(t_L0, min(self.z_run, z_p + 0.12 * s)), (t_hit - 0.02, z_p), (t_end, z_p + 0.02 * s)]
        elif overhead:
            if jump > 0.02:
                zk += [(t_to - 0.16, z_p - 0.10 * s), (t_to, z_p), (t_ld, z_p - 0.01), (t_ld + 0.12, z_p - 0.12 * s)]
            else:
                zk += [(t_hit - 0.25, z_p - 0.03 * s), (t_hit, z_p), (t_hit + 0.22, z_p - 0.05 * s)]
        else:
            zk += [(t_plan + 0.4, z_p), (t_end, z_p)]
        n = max(int(np.ceil((t_end - t_plan) / DT)), 2)
        tg = np.linspace(t_plan, t_end, n + 1)
        Z = _keys(zk, tg)
        if jump > 0.02:
            Z = Z + jump * np.clip(1.0 - ((tg - t_hit) / T_half) ** 2, 0.0, None)

        # ---- bake
        g = snap.gait.copy()
        XY, TURN, FEET, snaps = np.zeros((n + 1, 2)), np.zeros(n + 1), np.zeros((n + 1, 2, 5)), []
        turn = snap.turn
        done = set()
        for i, t in enumerate(tg):
            xy, vel = root(t)
            sp = float(np.linalg.norm(vel))
            side_on = overhead and t >= t_side
            tt = 0.0 if (serve or side_on) else travel_turn(vel, facing, sp)
            turn += (tt - turn) * (1.0 - np.exp(-(tg[1] - tg[0]) / 0.12)) if i else 0.0
            auto, stance = True, "ready"
            if hop:
                if "hop" not in done:
                    g.hop(t, root(hop_t1)[0], facing + turn, "wide", T_hop)
                    done.add("hop")
                if t < hop_t1:
                    auto = False
            if serve:
                stance = "serve"
            elif overhead:
                if side_on:
                    stance = "overhead"
                if t >= t_to - 0.10:
                    auto = False
                if t >= t_to and "sc" not in done:
                    H, Y = g.homes(root(t_ld)[0], facing, "after")
                    for j in (0, 1):
                        g.start(j, t, H[j], Y[j], max(t_ld - t, 0.1), 0.8 * jump + 0.05 * s)
                    done.add("sc")
            else:
                if t >= t_L1:
                    auto = False
                elif t >= t_L0 - 0.06:
                    auto = (True, False)                  # the other foot may still run; the racket foot is for the lunge
                if t >= t_L0 and "lunge" not in done:
                    g.start(1, t, tgt, yaw_foot, max(t_L1 - t, 0.08), (0.05 + 0.08 * reach) * s)
                    done.add("lunge")
            g.update(t, xy, vel, facing + turn, stance, auto)
            Z[i] = self._support(xy, Z[i], g)
            XY[i], TURN[i], FEET[i] = xy, turn, g.feet()
            snaps.append(g.copy())
        Z = np.convolve(np.pad(Z, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        up_plan = self.strip(snap.q)
        info = dict(kind=kind, jump=float(jump), travel=d, effort=float(1.0 / kt), hop=bool(hop), lunge=float(reach))
        sc = Script(self, clip, kind, t_plan, t_hit, t_start, t_end, k0, rate, tg, XY, Z, TURN, FEET, snaps, up_plan,
                    snap.yaw_off - 0.6 * snap.turn, info)
        sc.hold = None if hold is None else np.array(hold, float) + [0.0, 0.0, -0.05]
        return sc

    # ------------------------------------------------------------------ a dive
    DIVE_PITCH = (1.35, 1.15, 0.95, 0.75)            # how far the body is tipped over at the contact [rad], tried in order

    def dive_limits(self):
        """(acceleration, top speed, speed kept through the contact, braking on the floor) of a dive: a leap
        off both legs is quicker than any step, and the player's own top speed decides how far it carries."""
        return 6.4 * self.v0, 1.25 * self.v0, 0.75 * self.v0, 8.0

    def plan_dive(self, snap: Snap, t_plan, t_hit, p=None, direction=None):
        """Throw the body at the contact point p (racket arm first) so that the strings are there at t_hit; with
        p = None just dive along `direction` (a swing at thin air). Returns the DiveScript, or None when the
        point cannot be reached like this."""
        rig, s, facing = self.rig, self.rig.scale, self.facing
        if getattr(self, "_dive_c0", None) is None:
            q0 = self.body.dive.copy()
            q0[:3] = 0.0
            self._dive_c0 = self.ik.fk(q0)[0]
        c0 = self._dive_c0
        A, V, VE, D = self.dive_limits()
        here, v0 = snap.xy, snap.vel
        sp0 = float(np.linalg.norm(v0))
        t_c = sp0 / self.dec
        start = here + 0.5 * v0 * t_c
        z_lo, z_hi, z_lie = 0.30 * s, 0.66 * s, 0.19 * s

        def frame(d):
            yaw = float(np.clip(wrap(np.arctan2(d[1], d[0]) - PI / 2 - facing), -1.2, 1.2))
            return yaw, np.array([-d[1], d[0], 0.0])

        def rot(axis, pitch, yaw):
            return q2m(qmul(qaxis(axis, pitch), qaxis((0.0, 0.0, 1.0), facing + yaw)))

        if p is None:
            d = unit(direction)
            yaw_d, axis = frame(d)
            pitch_h, z_h = 1.2, 0.42 * s
            root_hit = start + d * 1.5 * s
        else:
            p = np.asarray(p, float)
            d = unit(p[:2] - start) if direction is None else unit(direction)
            best = None
            for _ in range(3):
                yaw_d, axis = frame(d)
                best = None
                for pitch in self.DIVE_PITCH:
                    c = rot(axis, pitch, yaw_d) @ c0
                    z_need = p[2] - c[2]
                    z_h = float(np.clip(z_need, z_lo, z_hi))
                    if best is None or abs(z_need - z_h) < best[0] - 0.02:
                        best = (abs(z_need - z_h), pitch, z_h, p[:2] - c[:2])
                root_hit = best[3]
                if np.linalg.norm(root_hit - start) < 0.15:
                    break
                d = unit(root_hit - start)
            res, pitch_h, z_h, root_hit = best
            if res > 0.22 * s or root_hit @ self.fwd > -0.35:
                return None
        dist = float(np.linalg.norm(root_hit - start))
        u = (root_hit - start) / dist if dist > 1e-6 else d
        tr = Travel(dist, A, D, V, VE)
        t_go, t_arr = t_plan + 0.03, t_hit - 0.01
        if tr.T > t_arr - t_go + 1e-6:
            return None
        kt = float(np.clip((t_arr - t_go) / max(tr.T, 1e-6), 1.0, 1.5))
        t_mv = t_arr - kt * tr.T
        t_to = t_mv + min(0.12, 0.35 * (t_hit - t_mv))
        t_lie, t_up, t_end = t_hit + 0.20, t_hit + 0.50, t_hit + 1.25

        def root(t):
            tau = min(max(t - t_plan, 0.0), t_c)
            xy = here + v0 * (tau - 0.5 * tau * tau / t_c) if t_c > 1e-6 else here.copy()
            ds, _ = tr.at((t - t_mv) / kt)
            xy = xy + u * ds
            over = xy @ self.fwd + 0.30
            return xy - self.fwd * over if over > 0 else xy

        n = max(int(np.ceil((t_end - t_plan) / DT)), 2)
        tg = np.linspace(t_plan, t_end, n + 1)
        z0 = min(snap.z, self.z_ready)
        Z = _keys([(t_plan, snap.z), (t_to, z0 - 0.06 * s), (t_hit, z_h), (t_lie, z_lie), (t_up, z_lie),
                   (t_up + 0.38, 0.6 * rig.z_stand), (t_end, self.z_ready)], tg)
        PITCH = _keys([(t_plan, 0.0), (t_to, 0.3), (t_hit, pitch_h), (t_lie, 1.45), (t_up, 1.45), (t_up + 0.40, 0.55),
                       (t_end - 0.12, 0.0), (t_end, 0.0)], tg)
        YAW = _keys([(t_plan, snap.yaw_off), (t_hit, yaw_d), (t_up, yaw_d), (t_end, 0.0)], tg)
        W = _keys([(t_plan, 0.0), (t_hit - 0.04, 1.0), (t_up, 1.0), (t_end - 0.2, 0.0), (t_end, 0.0)], tg)
        XY = np.array([root(t) for t in tg])
        perp = np.array([u[1], -u[0]]) if dist > 1e-6 else np.array([d[1], -d[0]])
        g = snap.gait.copy()
        H, Y = g.homes(XY[-1], facing, "ready")
        FEET, snaps = np.zeros((n + 1, 2, 5)), []
        leg = 0.80 * s
        for i, t in enumerate(tg):
            if t < t_to:
                g.update(t, XY[i], np.zeros(2), facing, "ready", auto=False)
            planted = g.feet()
            wf = sstep((t - t_to) / max(0.12, 0.5 * (t_hit - t_to)))
            wg = sstep((t - t_up - 0.08) / 0.55)
            row = np.zeros((2, 5))
            for j, sx in ((0, -1.0), (1, 1.0)):
                th = PITCH[i]
                trail = np.r_[XY[i] - u * leg * np.sin(th) + perp * sx * 0.10, max(Z[i] - leg * np.cos(th) - rig.sole, 0.04),
                              facing + YAW[i], min(th, 1.0)]
                a = planted[j].copy()
                a[3] = trail[3] + wrap(a[3] - trail[3])
                r = a + wf * (trail - a)
                if wg > 0:
                    home = np.r_[H[j], 0.0, r[3] + wrap(Y[j] - r[3]), 0.0]
                    r = r + wg * (home - r)
                    r[2] += 0.05 * s * np.sin(PI * wg)
                row[j] = r
            FEET[i] = row
            if t < t_to:
                snaps.append(g.copy())
            else:
                g2 = Gait(s)
                g2.set_feet(row[:, :2], row[:, 3], np.where(row[:, 2] > 0.02, row[:, 2], 0.0))
                g2.t = t
                snaps.append(g2)
        info = dict(kind="dive", jump=0.0, travel=dist, effort=float(1.0 / kt), hop=False, lunge=0.0, lock=float(t_end - 0.15))
        return DiveScript(self, t_plan, t_hit, t_end, tg, XY, Z, YAW, PITCH, W, FEET, snaps, self.strip(snap.q), axis, info)


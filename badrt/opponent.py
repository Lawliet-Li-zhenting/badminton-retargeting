"""Command-driven player ("读指令的对手"): tactics pick a shot, physics has to deliver it.

For every incoming shuttle the player
  1. predicts the flight and lists where it could be met (height, time, can the feet get there),
  2. picks a command: either its own tactical shot (styles.Style) or, when the match director
     asks for it, a feed that reaches a given point at a given time (recorded human input),
  3. inverts the impact model for the string-bed normal and racket-head speed of that shot,
  4. warps a stroke clip from the library so the racket really does that (swing.build_swing),
  5. replays the collision with the game's own contact code and corrects the command once.
The shuttle is never assigned a velocity: in the match it leaves the strings through
RacketImpact like any other hit, so a late, stretched or badly oriented swing gives a bad shot.

Everything is computed for a right-handed player; a left-hander plans in the mirrored world
(x -> -x) and the resulting poses are mirrored back (character.mirror_qpos).
side = +1 plays the far half (y > 0) facing -y; side = -1 the near half facing +y.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import brain
from .athlete import Athlete, Clip
from .character import RACKET, blend_qpos, mirror_qpos
from .court import LENGTH_HALF, SHORT_SERVICE, WIDTH_SINGLES_HALF, in_singles, net_height
from .footwork import set_root
from .shuttle import (ImpactParams, RacketImpact, RacketPose, ShuttleParams, ShuttlePhysics, ShuttleState, aim_shot,
                      inverse_impact, racket_point_velocity, solve_arrival, solve_launch)
from .styles import DIFFICULTY, SHOTS, Style
from .swing import Swing, SwingIK, build_swing, contact_error, free_swing, mat2quat, quat2mat
from .character import quat_slerp

TICK = 1e-3
MIRROR = np.diag([-1.0, 1.0, 1.0])


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp_pose(a: RacketPose, b: RacketPose, f):
    qa, qb = mat2quat(a.R), mat2quat(b.R)
    return RacketPose(a.center + f * (b.center - a.center), quat2mat(quat_slerp(qa, qb, f)))


def mirror_state(s: ShuttleState):
    m = np.array([-1.0, 1.0, 1.0])
    return ShuttleState(s.pos * m, s.vel * m, s.axis * m, s.omega * [1.0, -1.0, -1.0], s.spin, s.spin_angle)


def situation(z, depth):
    if z >= 1.9:
        return "kill" if depth < 3.8 else "rear"
    if z >= 1.0:
        return "mid"
    return "net" if depth < 2.6 else "defence"


@dataclass
class Plan:
    shot: str
    t_plan: float
    t_hit: float
    contact: np.ndarray              # world
    v_out: np.ndarray                # world, what the verified collision will produce
    land: np.ndarray | None          # predicted landing point (world)
    situation: str
    feed: bool = False               # shot dictated by the match director
    release: tuple | None = None     # serve: (t_release, ShuttleState) -- the shuttle is dropped onto the swing
    info: dict = field(default_factory=dict)


class AIPlayer:
    def __init__(self, segments, side, style: Style, rng=None, phys: ShuttlePhysics | None = None,
                 difficulty="pro", impact_params: ImpactParams | None = None):
        self.side, self.style = side, style
        self.rng = rng or np.random.default_rng(0)
        self.phys = phys or ShuttlePhysics()
        self.sp = self.phys.p
        self.ip = impact_params or ImpactParams()
        self.impact = RacketImpact(RACKET["head_half_width"], RACKET["head_half_length"], params=self.ip,
                                   shuttle=self.sp, rng=np.random.default_rng(1))
        d = DIFFICULTY[difficulty]
        self.max_speed = style.move_speed * d["speed"]
        self.reaction = style.reaction + d["reaction"]
        self.err = style.error * d["error"]
        self.aggr = style.aggression * d["aggression"]
        self.attack, self.power, self.width = d["attack"], d["power"], d["width"]
        self.reach = style.reach * d["reach"]
        self.beaten = d["beaten"]
        self.smart = d.get("smart", 0.0)         # how often the target is chosen by where the other player is / is going
        self.dive_p = d.get("dive", 0.0)         # how often a shuttle out of reach is dived for
        self.mind = None                         # brain.Mind: learned tactics instead of the style's menu
        self.stam = None                         # arcade.Stamina (controller matches)
        self.resting = False                     # between rallies (set by the match)
        self.locked = -1.0                       # on the floor after a dive until then: no new stroke
        self.pending = False                     # a reply could not be planned because of that
        self.mx = -1.0 if style.handed == "left" else 1.0
        self.ik = SwingIK(style.dims)
        self.leg = style.leg_scale
        self.yaw = 0.0 if side < 0 else np.pi
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.R2 = np.array([[c, -s], [s, c]])
        self.R3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.base = np.array([0.0, side * style.base_depth])
        self.body = Athlete(self.ik, self.yaw, self.base, self.max_speed, self.reaction, self.reach)
        self.clips = [self.body.clip(sg) for sg in segments]
        self.q = self.body.q.copy()
        self.q_prev = self.q.copy()
        self.t = self.t_prev = None
        self.swing: Swing | None = None
        self.plan: Plan | None = None
        self.rk_now = self.rk_prev = None
        self.message = ""
        self.stats = dict(plans=0, fails=0, corrections=0, plan_ms=[])
        self.fail = dict(not_coming=0, leave=0, no_window=0, no_reach=0, no_command=0, no_swing=0, mishit=0, beaten=0, arm_short=0)
        self.dives = 0

    # ------------------------------------------------------------------ stamina (arcade matches)
    def _may_jump(self):
        return self.stam is None or self.stam.can("jump")

    def _speed_k(self):
        return 1.0 if self.stam is None else self.stam.speed()

    def _pay(self, plan):
        """What the stroke that has just been planned costs."""
        if self.stam is None or plan is None:
            return
        self.stam.take(plan.shot if plan.shot in self.stam.COST else "stroke")
        if plan.info.get("jump", 0.0) > 0.02:
            self.stam.take("jump")

    @property
    def ready_q(self):
        """The ready stance at the base position (internal, right-handed)."""
        b = self.body
        g = b.gait.copy()
        g.reset(self.base, self.yaw, "ready")
        return b.rig.set_legs(set_root(b.body.ready, self.base, b.z_ready, self.yaw), g.feet())

    # ------------------------------------------------------------------ mirroring (left-handers)
    def _in(self, s: ShuttleState):
        return s if self.mx > 0 else mirror_state(s)

    def _vec(self, v):
        v = np.array(v, float)
        v[0] *= self.mx
        return v

    def _out_pose(self, rp: RacketPose):
        if self.mx > 0:
            return rp
        return RacketPose(rp.center * [-1.0, 1.0, 1.0], MIRROR @ rp.R @ MIRROR)

    def qpos(self):
        """Pose for the (handed) character model."""
        return self.q if self.mx > 0 else mirror_qpos(self.q)

    def root_xy(self):
        return self._vec(np.r_[self.q[:2], 0.0])[:2]

    def place(self, xy):
        """Put the player at a serving / receiving position at once (start of a match; world coordinates)."""
        self.q = self.body.reset(np.array([self.mx * xy[0], xy[1]]), self.t).copy()
        self.body.go(self.q[:2])
        self.q_prev = self.q.copy()
        self.swing = None
        self.plan = None
        if self.t is not None:
            self.rk_now = self.rk_prev = (self.t, RacketPose(*self.ik.fk(self.q)))

    def walk_to(self, xy, speed=2.4, mode="relax"):
        """Between points: walk to the next serving / receiving position (world coordinates)."""
        self.body.go(np.array([self.mx * xy[0], xy[1]]), speed, mode)

    def set_ready(self):
        self.body.mode = "ready"

    def in_position(self):
        return self.swing is None and self.body.arrived()

    def _take(self, swing):
        """Start executing a planned swing."""
        self.swing = swing
        self.body.script = swing.script
        self.body.go(self.base)

    # ------------------------------------------------------------------ per-frame update
    def update(self, now):
        """Advance to `now` and return the pose. Calling it again with the same time re-reads a swing
        that was planned in between (the match plans inside its physics ticks)."""
        if self.t is None or now > self.t:
            if self.stam is not None and self.t is not None:
                span = self.t - self.t_prev if self.t_prev is not None else 0.0
                v = float(np.linalg.norm(self.q[:2] - self.q_prev[:2])) / span if span > 1e-6 else 0.0
                self.stam.tick(now - self.t, v / self.body.v0, self.resting)
                self.body.set_speed(self._speed_k())
            self.q_prev, self.t_prev = self.q.copy(), self.t if self.t is not None else now
            if self.swing is not None and now > self.swing.t[-1]:
                self.body.adopt(self.swing.t[-1], self.swing.q[-1])      # stroke over: recover from where it ended
                self.swing = None
            if self.swing is not None:
                self.q = self.swing.qpos_at(now)
            else:
                self.q = self.body.step(now).copy()
            self.t = now
            pose = RacketPose(*self.ik.fk(self.q))
            self.rk_prev = self.rk_now if self.rk_now is not None else (now, pose)
            self.rk_now = (now, pose)
        elif self.swing is not None and self.swing.covers(now):
            self.q = self.swing.qpos_at(now)
            self.rk_now = (now, RacketPose(*self.ik.fk(self.q)))
        return self.qpos()

    def pose_internal(self, t):
        if self.swing is not None and self.swing.covers(t):
            return self.swing.qpos_at(t)
        if self.t is None or self.t_prev is None or self.t <= self.t_prev:
            return self.q.copy()
        f = min(max((t - self.t_prev) / (self.t - self.t_prev), 0.0), 1.0)
        return blend_qpos(self.q_prev, self.q, f)

    def racket_at(self, t):
        """Racket pose (world) at any time inside the current frame interval."""
        if self.swing is not None and self.swing.covers(t):
            return self._out_pose(self.swing.racket_at(t))
        (t0, a), (t1, b) = self.rk_prev, self.rk_now
        if t1 <= t0:
            return self._out_pose(b)
        return self._out_pose(lerp_pose(a, b, min(max((t - t0) / (t1 - t0), 0.0), 1.0)))

    def racket_center(self):
        c = self.rk_now[1].center
        return c * [self.mx, 1.0, 1.0]

    # ------------------------------------------------------------------ planning: returns
    def plan_return(self, shuttle: ShuttleState, now, other_xy=None, feed=None, rally=0, other_vel=None):
        """Decide and prepare the reply to the shuttle in flight. feed: upcoming (t, pos) pairs the
        shot should be delivered to (recorded human input); None = play the style's own game.
        Returns the Plan, or None when the shuttle is left (going out) or cannot be reached."""
        import time
        tic = time.perf_counter()
        self.pending = False
        if now < self.locked:                    # still getting up from a dive: the match asks again
            self.pending = True
            self.message = "on the floor"
            return None
        self._other_vel = None if other_vel is None else self._vec(np.r_[other_vel, 0])[:2]
        plan = self._plan_return(self._in(shuttle), now, None if other_xy is None else self._vec(np.r_[other_xy, 0])[:2],
                                 feed, rally)
        self.stats["plans"] += 1
        self.stats["fails"] += plan is None
        self.stats["plan_ms"].append((time.perf_counter() - tic) * 1e3)
        self._pay(plan)
        return plan

    def _plan_return(self, s, now, other, feed, rally):
        T, P, V, C, W = self.phys.predict(s, 3.6, dt=0.004, full=True)
        side = self.side
        mine = P[:, 1] * side > 0.3
        self.message = ""
        if not mine.any():
            self.message = "not coming over"
            self.fail["not_coming"] += 1
            return None
        land = P[-1]
        if land[2] <= 0.0 and land[1] * side > 0:
            out_by = max(abs(land[0]) - WIDTH_SINGLES_HALF, abs(land[1]) - LENGTH_HALF)
            if out_by > 0.02 and (out_by > 0.35 or self.rng.random() < self.style.judgement):
                self.message = "leave (out)"
                self.fail["leave"] += 1
                return None
        # difficulty: a hard downward shot is sometimes simply not read in time
        if s.vel[2] < 0 and np.linalg.norm(s.vel) > 33.0 and self.rng.random() < self.beaten:
            self.message = "beaten for pace"
            self.fail["beaten"] += 1
            return None
        snap = self.body.snap(now, self.pose_internal(now))
        z_max = 2.75 * self.leg
        cands, last_t = [], -1.0
        for i in np.flatnonzero(mine):
            z = P[i, 2]
            if z > z_max or z < 0.22 or T[i] < self.reaction + 0.06:
                continue
            if V[i, 2] > 0 and z < 2.0:
                continue                              # still rising: wait unless it is already overhead
            if T[i] - last_t >= 0.05:
                cands.append(int(i)); last_t = T[i]
        attack = feed is None or self.rng.random() < self._attack_prob(rally)
        if feed is not None:
            feed = [(t, self._vec(p)) for t, p in feed]
        fine = None
        why = "no_window"
        tried = 0
        for i in cands:
            p, v_in, axis, omega = P[i], V[i], C[i], W[i]
            sit = situation(p[2], abs(p[1]))
            reachable = self._reachable(p, T[i], snap)
            if not reachable:
                why = "no_reach"
                continue
            tried += 1
            if tried > 5:
                break
            stretch = min(c[3] for c in reachable)
            cmd = None
            if feed is not None and not attack:
                cmd = self._feed_shot(p, now + T[i], v_in, axis, omega, feed, sit)
            if cmd is None:
                cmd = self._tactic_shot(p, v_in, axis, omega, sit, other, stretch)
            if cmd is None:
                why = "no_command"
                continue
            shot, v_des, is_feed = cmd
            if fine is None or fine[0][-1] < T[i] + 0.045:
                fine = self.phys.predict(s, T[i] + 0.06, dt=TICK, full=True, stop_z=-1.0)
            j = int(round(T[i] / TICK))
            if j + 1 >= len(fine[0]):
                continue
            res = self._realise(now, now, fine, j, v_des, reachable, snap, max_rel=0.05 if is_feed else 0.15)
            if res is None:
                why = "no_swing"
                continue
            swing, t_c, st_out, info = res
            self._take(swing)
            info["intent"] = "style" if feed is None else ("feed" if is_feed else ("attack" if attack else "no_feed"))
            self.plan = self._make_plan(shot, now, t_c, st_out, sit, is_feed, info)
            self.message = f"{shot} ({sit})"
            return self.plan
        self.fail[why] += 1
        self.message = {"no_window": "no hitting window", "no_reach": "can't reach", "no_command": "no shot",
                        "no_swing": "swing failed"}[why]
        if why == "no_reach" and self.dive_p > 0 and self.rng.random() < self.dive_p and (
                self.stam is None or self.stam.can("dive")):
            return self._dive(s, now, (T, P, V, C, W), other)
        return None

    def _attack_prob(self, rally):
        """With a director feeding a recorded human, how likely this shot is played to win instead."""
        return float(np.clip(0.07 * self.aggr * (1 + 0.12 * rally), 0.0, 0.6))

    def _reachable(self, p, t_avail, snap):
        """Clips that put the racket at p in time: (clip, pelvis xy at the hit, cost, stretch, pelvis z, jump).
        The feet have to get there with this player's acceleration and top speed (Athlete.travel_time);
        when they are a little short the pelvis stops short and the arm and trunk have to stretch."""
        b = self.body
        t_move = max(t_avail - self.reaction - 0.02, 0.0)
        d_lunge, d_rest = b.travel_range(t_move), b.travel_range(t_move, rest=True)
        start = snap.xy + b.coast(snap.vel)
        out = []
        for c in self.clips:
            fit = b.fit(c, p[2], t_avail, self._may_jump())
            if fit is None:
                continue
            z_p, jump, res = fit
            root = p[:2] - self.R2 @ c.c_loc[:2]
            if root[1] * self.side < 0.45:                      # never step over (or under) the net
                continue
            dv = root - start
            dist = float(np.linalg.norm(dv))
            rest = c.kind == "overhead"
            short = dist - (d_rest if rest else d_lunge)
            if short > b.slack:
                continue
            if short > 0:
                root = root - dv / dist * short
            if not b.arm_reaches(c, root, z_p + jump, p):       # (the swing builder would only find out 50 ms later)
                continue
            stretch = 1.0 if short > 0 else min(1.0, b.travel_time(dist, rest) / max(t_move, 1e-3))
            cost = abs(res) / 0.25 + dist / 2.0 + 0.5 * jump / 0.3 + max(short, 0.0) / b.slack
            out.append((c, root, cost, stretch, z_p, jump))
        return out

    # ------------------------------------------------------------------ commands
    def _after_hit(self, p, v_in, axis, omega):
        def make(v0):
            n = inverse_impact(v_in, v0, axis, self.ip, self.sp)[0]
            return ShuttleState(np.array(p, float), np.array(v0, float), np.array(axis, float),
                                np.array(omega, float) + np.cross(axis, n) * 30.0)
        return make

    def _shot_speed(self, shot):
        speed = SHOTS[shot][0]
        pw = self.power * (1.0 if self.stam is None else self.stam.power())
        if shot == "smash":
            return self.style.smash_speed * pw
        if shot == "half_smash":
            return 0.64 * self.style.smash_speed * pw
        return speed * pw if shot == "kill" else speed

    def _brain_shot(self, p, v_in, axis, omega, sit, other, stretch):
        """Shot, side and depth from the learned tactics (brain.Mind); None when none of its choices can be played."""
        st, mind = self.style, self.mind
        s, c = brain.state(sit, p, other, stretch)
        recv = -self.side
        after = self._after_hit(p, v_in, axis, omega)
        for a in mind.order(s, brain.prior(st, sit, s, self.attack), self.rng):
            shot, side, deep = brain.ACTIONS[sit][a]
            if shot in brain.DOWN and p[2] < 1.75:
                continue
            _, high, (y0, y1) = SHOTS[shot]
            speed = self._shot_speed(shot)
            x = (WIDTH_SINGLES_HALF - st.corner - self.rng.uniform(0, 0.25)) * self.width
            xt = float(np.clip(other[0], -x, x)) if side == 2 else (c if side == 0 else -c) * x
            f = self.rng.uniform(0.05, 0.3) if deep == 0 else self.rng.uniform(0.7, 0.95)
            for attempt in range(2):
                v0 = aim_shot(self.phys, after, p, (xt, recv * (y0 + f * (y1 - y0))), speed * (1 + 0.1 * attempt),
                              high=high, net_margin=0.12)
                if v0 is not None:
                    mind.took(s, a)
                    return shot, self._execution_error(v0, stretch), False
        return None

    def _tactic_shot(self, p, v_in, axis, omega, sit, other, stretch):
        if self.mind is not None and other is not None:
            cmd = self._brain_shot(p, v_in, axis, omega, sit, other, stretch)
            if cmd is not None:
                return cmd
        st = self.style
        menu = dict(st.shots[sit])
        for k in ("smash", "half_smash", "kill", "slice"):      # difficulty: how readily the rally is finished
            if k in menu and len(menu) > 1:
                menu[k] *= self.attack
        if stretch > 0.8:                                       # late: forget the fancy shots
            for k in ("smash", "half_smash", "slice", "net"):
                if k in menu:
                    menu[k] *= 0.3
        vel = getattr(self, "_other_vel", None)
        if self.smart > 0 and other is not None:                # read the court: play to where the other player is not
            for k in menu:
                y0, y1 = SHOTS[k][2]
                menu[k] *= 1.0 + 0.5 * self.smart * abs(0.5 * (y0 + y1) - abs(other[1]))
                if k in ("smash", "half_smash", "kill") and abs(other[0]) > 1.2:
                    menu[k] *= 1.0 + self.smart
        names = list(menu)
        w = np.array([menu[k] for k in names], float)
        order = list(self.rng.choice(names, size=len(names), replace=False, p=w / w.sum()))
        order += [k for k in ("clear", "lift", "drive") if k not in order]
        recv = -self.side
        after = self._after_hit(p, v_in, axis, omega)
        for shot in order[:4]:
            speed, high, (y0, y1) = SHOTS[shot]
            pw = self.power * (1.0 if self.stam is None else self.stam.power())
            if shot == "smash":
                speed = st.smash_speed * pw
            elif shot == "half_smash":
                speed = 0.64 * st.smash_speed * pw
            elif shot == "kill":
                speed *= pw
            if shot in ("smash", "half_smash", "kill", "slice") and p[2] < 1.75:
                continue                                        # cannot hit down from below the tape
            for attempt in range(2):
                x = (WIDTH_SINGLES_HALF - st.corner - self.rng.uniform(0, 0.25)) * self.width
                sgn = 1.0 if self.rng.random() < 0.5 else -1.0
                body = False
                if other is not None:
                    if shot in ("smash", "half_smash") and self.rng.random() < st.body:
                        x, sgn, body = abs(other[0]), np.sign(other[0] + 1e-6), True   # at the body
                    elif self.rng.random() < st.cross:
                        sgn = -np.sign(other[0] + 1e-3)
                y = recv * self.rng.uniform(y0, y1)
                if self.smart > 0 and other is not None and not body and self.rng.random() < self.smart:
                    sgn, y = self._smart_target(x, recv, y0, y1, other, vel)
                v0 = aim_shot(self.phys, after, p, (sgn * x, y), speed * (1 + 0.1 * attempt), high=high,
                              net_margin=0.12)
                if v0 is not None:
                    return shot, self._execution_error(v0, stretch), False
        return None

    def _smart_target(self, x, recv, y0, y1, other, vel):
        """The corner (side, depth of this shot's range) farthest from where the other player will be -- and
        behind them when they are on the move (wrong-footing)."""
        v = np.zeros(2) if vel is None else np.asarray(vel, float)
        at = np.asarray(other, float) + 0.3 * v
        sp = float(np.linalg.norm(v))
        best = None
        for sgn in (-1.0, 1.0):
            for y in (y0 + 0.15 * (y1 - y0), y1 - 0.1 * (y1 - y0)):
                tgt = np.array([sgn * x, recv * y])
                sc = float(np.linalg.norm(tgt - at)) + self.rng.uniform(0, 0.4)
                if sp > 1.5:
                    d = tgt - np.asarray(other, float)
                    sc -= 0.25 * sp * float(v @ d) / (sp * max(np.linalg.norm(d), 1e-6))
                if best is None or sc > best[0]:
                    best = (sc, sgn, recv * y)
        return best[1], best[2]

    def _execution_error(self, v0, stretch):
        """Aiming error of the command (the swing adds its own through IK residuals)."""
        sigma = self.err * (1 + 2.5 * stretch ** 2) * (1.0 if self.stam is None else self.stam.error())
        speed = np.linalg.norm(v0)
        d = v0 / speed
        a = np.cross(d, [0, 0, 1.0])
        a /= max(np.linalg.norm(a), 1e-9)
        b = np.cross(a, d)
        th_h, th_v = self.rng.normal(0, np.radians(4.0) * sigma, 2)
        d = d + th_h * a + th_v * b
        return d / np.linalg.norm(d) * speed * (1 + self.rng.normal(0, 0.06 * sigma))

    def arrive(self, p, target, T, v_in, axis, omega, v_max, margin=0.08):
        """Launch velocity reaching `target` T seconds after contact at p, with the full flight model
        (post-impact turnover drag). None if too fast for this stroke or it would not clear the net."""
        after = self._after_hit(p, v_in, axis, omega)
        bias = np.zeros(3)
        for _ in range(3):
            sol = solve_arrival(p, target, T, self.sp, bias=bias)
            if sol is None:
                return None
            v = sol[0]
            if np.linalg.norm(v) > v_max:
                return None
            Tt, Pp, _ = self.phys.predict(after(v), T + 0.004, dt=0.004, stop_z=-1.0)
            end = np.array([np.interp(T, Tt, Pp[:, k]) for k in range(3)])
            err = end - np.asarray(target, float)
            cross = np.flatnonzero(np.sign(Pp[:, 1]) != np.sign(Pp[0, 1]))
            if len(cross) == 0:
                return None
            a, b = Pp[cross[0] - 1], Pp[cross[0]]
            f = a[1] / (a[1] - b[1])
            pn = a + f * (b - a)
            if pn[2] < net_height(pn[0]) + margin or Pp[:cross[0], 2].min() < 0.05:
                return None
            if np.linalg.norm(err) < 0.06:
                return v
            bias = bias + err
        return v if np.linalg.norm(err) < 0.2 else None

    def _feed_shot(self, p, t_hit, v_in, axis, omega, feed, sit):
        """Deliver the shuttle to the recorded player's next reachable stroke."""
        v_max = self.style.smash_speed if p[2] >= 1.9 else 46.0
        tried = 0
        for t_k, p_k in feed:
            T = t_k - t_hit
            if T < 0.42:
                continue
            if T > 2.8 or tried >= 2:
                break
            tried += 1
            v = self.arrive(p, p_k, T, v_in, axis, omega, v_max)
            if v is not None:
                e = np.degrees(np.arcsin(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
                sp = np.linalg.norm(v)
                if p[2] >= 1.9:
                    shot = "clear" if e > 8 else ("smash" if sp > 45 else ("slice" if sp > 24 else "drop"))
                elif p[2] < 1.0:
                    shot = "lift" if e > 25 and sp > 18 else "net"
                else:
                    shot = "lift" if e > 30 else ("drive" if sp > 22 else "push")
                return shot, v, True
        return None

    # ------------------------------------------------------------------ swing construction
    def _realise(self, t_plan, t_ref, fine, j, v_des, reachable, snap, max_rel=0.15, serve=False, hold=None):
        """Build the swing for the commanded outgoing velocity and replay the contact.
        fine: 1 ms prediction (T, P, V, C, W) starting at t_ref; j: tick index of the planned contact.
        max_rel: largest accepted |v_realised - v_commanded| / |v_commanded| (a feed has to arrive
        on the spot, so it is held to a tighter one)."""
        Tf, Pf, Vf, Cf, Wf = fine
        p, v_in, axis = Pf[j], Vf[j], Cf[j]
        t_hit = t_ref + Tf[j]
        n, s, _ = inverse_impact(v_in, v_des, axis, self.ip, self.sp)
        scored = []
        for c, root, cost, _, z_p, jump in reachable:
            # how far the strings of this clip face away from what the shot needs. A bent arm has the freedom to
            # turn the racket; an arm stretched overhead has not (measured: 7 % of such swings fail below 15 deg
            # of mismatch, half at 30-40, four in five above 50), so there it decides the order
            mis = np.arccos(min(abs(float((self.R3 @ c.n_loc) @ n)), 1.0))
            scored.append((cost + mis / np.radians(15.0 if c.kind == "overhead" else 50.0) + self.rng.uniform(0, 0.6),
                           c, root, z_p, jump))
        scored.sort(key=lambda x: x[0])
        built = 0
        for _, clip, root, z_p, jump in scored[:6]:
            if built >= 3:
                break
            script = self.body.plan(snap, t_plan, clip, root, t_hit, z_p, jump, serve=serve, hold=hold)
            base, t_end = script.base, script.t_end
            if contact_error(self.ik, base, t_hit, p, n, self.sp.radius)[0] > 0.03:
                self.fail["arm_short"] += 1                     # this stroke does not reach from there: next clip
                continue
            built += 1
            best, n = self._fit_swing(script, t_plan, t_ref, fine, j, v_des, clip.kind, clip.side)
            if best is not None and best[0] <= max_rel:
                script.fill(best[1])                            # legs: they do not move the racket
                best[1].script = script
                return best[1:]
            if best is not None:
                self.fail["mishit"] += 1
        return None

    def _fit_swing(self, script, t_plan, t_ref, fine, j, v_des, kind, side):
        """Warp the arm of a baked body timeline so that the racket gives the shuttle the velocity v_des, replay
        the contact and correct the command (fixed point, at most three rounds).
        Returns ((relative error, swing, contact time, shuttle after, info) or None, last string-bed normal)."""
        Tf, Pf, Vf, Cf, Wf = fine
        p, v_in, axis = Pf[j], Vf[j], Cf[j]
        t_hit = t_ref + Tf[j]
        best, n = None, None
        v_cmd = np.array(v_des, float)
        for it in range(3):
            n, s, _ = inverse_impact(v_in, v_cmd, axis, self.ip, self.sp)
            swing = build_swing(self.ik, script.base, t_plan, script.t_end, t_hit, p, n, s, shuttle_radius=self.sp.radius)
            res = self._replay(swing, t_ref, fine, j)
            if res is None:
                break
            t_c, st, info = res
            dv = st.vel - v_des
            rel = float(np.linalg.norm(dv) / max(np.linalg.norm(v_des), 1e-6))
            if rel > 0.3 or info["kind"] != "strings" or info["racket_speed"] > 1.4 * abs(s) + 6.0:
                self.fail["mishit"] += 1                    # the arm cannot do this one: not a usable plan
                if it == 0:
                    v_cmd = v_cmd - 0.7 * dv
                    continue
                break
            if best is None or rel < best[0]:
                info = dict(info, rel_err=rel, ik_pos=swing.pos_err, ik_tilt=swing.tilt_err, clip=kind,
                            clip_side=side, head_cmd=float(s), corrected=it > 0, **script.info)
                best = (rel, swing, t_c, st, info)
            if rel < 0.03:
                break
            v_cmd = v_cmd - dv                              # fixed-point correction of the command
            self.stats["corrections"] += 1
        return best, n

    # ------------------------------------------------------------------ a dive
    def _dive(self, s, now, pred, other, button=None, direction=None, t_lo=0.14, t_hi=0.8):
        """Throw the body at a shuttle the feet cannot get to: the first point of its flight a dive reaches
        (direction: only towards that side). The reply is a desperate lift (or a block with the soft button);
        the player then lies on the floor for a second. Returns the Plan or None."""
        T, P, V, C, W = pred
        b = self.body
        snap = b.snap(now, self.pose_internal(now))
        fine, last, tried = None, -1.0, 0
        for i in np.flatnonzero((P[:, 1] * self.side > 0.3) & (T >= t_lo) & (T <= t_hi) & (P[:, 2] >= 0.12) & (P[:, 2] <= 1.3)):
            if T[i] - last < 0.04:
                continue
            last = T[i]
            p = P[i]
            script = b.plan_dive(snap, now, now + T[i], p)
            if script is None:
                continue
            if direction is not None:
                k = int(np.argmin(np.abs(script.tg - (now + T[i]))))
                if (script.XY[k] - snap.xy) @ np.asarray(direction, float) < 0.25 * np.linalg.norm(script.XY[k] - snap.xy):
                    continue
            cmd = self._dive_shot(button, p, V[i], C[i], W[i], other)
            if cmd is None:
                continue
            tried += 1
            if tried > 3:
                break
            if fine is None or fine[0][-1] < T[i] + 0.045:
                fine = self.phys.predict(s, min(T[i] + 0.3, t_hi + 0.06), dt=TICK, full=True, stop_z=-1.0)
            j = int(round(T[i] / TICK))
            if j + 1 >= len(fine[0]):
                continue
            n = inverse_impact(V[i], cmd, C[i], self.ip, self.sp)[0]
            if contact_error(self.ik, script.base, now + T[i], p, n, self.sp.radius)[0] > 0.04:
                self.fail["arm_short"] += 1
                continue
            best, _ = self._fit_swing(script, now, now, fine, j, cmd, "dive", "forehand")
            if best is None or best[0] > 0.3:
                continue
            _, swing, t_c, st_out, info = best
            script.fill(swing)
            swing.script = script
            self._take(swing)
            self.locked = script.info["lock"]
            self.dives += 1
            info["intent"] = "dive"
            self.plan = self._make_plan("dive", now, t_c, st_out, "defence", False, info)
            self.message = "dive"
            if self.stam is not None:
                self.stam.take("dive")
            return self.plan
        return None

    def _dive_shot(self, button, p, v_in, axis, omega, other):
        after = self._after_hit(p, v_in, axis, omega)
        recv = -self.side
        sgn = -float(np.sign(other[0])) if other is not None and abs(other[0]) > 0.25 else (1.0 if self.rng.random() < 0.5 else -1.0)
        for shot in (("block", "lift") if button == "soft" else ("lift", "block")):
            speed, high, (y0, y1) = SHOTS[shot]
            x = sgn * (WIDTH_SINGLES_HALF - 0.9 - self.rng.uniform(0, 0.4))
            y = recv * (self.rng.uniform(4.9, 5.9) if shot == "lift" else self.rng.uniform(y0, y1))
            v0 = aim_shot(self.phys, after, p, (x, y), speed, high=high, net_margin=0.15)
            if v0 is not None:
                return self._execution_error(v0, 1.0)
        return None

    def _replay(self, swing: Swing, t_ref, fine, j, before=160, after=45, tol=6):
        """The match's contact loop on the predicted shuttle ticks. Returns (t, state after, info),
        or None when there is no contact or the racket clips the shuttle before the planned moment
        (the window starts well inside the backswing)."""
        Tf, Pf, Vf, Cf, Wf = fine
        j0, j1 = max(j - before, 0), min(j + after, len(Tf) - 1)
        r0 = swing.racket_at(t_ref + Tf[j0])
        for k in range(j0, j1):
            if abs(Pf[k + 1, 0] - swing.rc_box[0]) > swing.rc_box[3] or abs(Pf[k + 1, 1] - swing.rc_box[1]) > swing.rc_box[3]:
                r0 = None
                continue
            r1 = swing.racket_at(t_ref + Tf[k + 1])
            if r0 is None:
                r0 = swing.racket_at(t_ref + Tf[k])
            if np.linalg.norm(Pf[k + 1] - r1.center) < 0.6:
                s0 = ShuttleState(Pf[k], Vf[k], Cf[k], Wf[k])
                s1 = ShuttleState(Pf[k + 1].copy(), Vf[k + 1].copy(), Cf[k + 1].copy(), Wf[k + 1].copy())
                hit = self.impact.check(s0, s1, r0, r1)
                if hit is not None:
                    kind, point, normal = hit
                    v_r = racket_point_velocity(r0, r1, Tf[k + 1] - Tf[k], point)
                    info = self.impact.respond(s1, kind, point, normal, v_r)
                    if info is not None:
                        if abs(k + 1 - j) > tol:
                            return None
                        return t_ref + Tf[k + 1], s1, info
            r0 = r1
        return None

    def _make_plan(self, shot, t_plan, t_c, st_out, sit, is_feed, info, release=None):
        T, P, _ = self.phys.predict(st_out, 4.0, dt=0.004)
        w = lambda v: self._vec(v)
        return Plan(shot, t_plan, t_c, w(st_out.pos), w(st_out.vel), w(P[-1]), sit, is_feed, release, info)

    # ------------------------------------------------------------------ planning: serve
    def plan_serve(self, now, right_court, feed=None, lead=1.3, tick0=None, shot=None, drill=None):
        """Serve from the current position into the diagonal service court (or, with a director
        feed, to the receiver's next stroke). The shuttle is dropped from just above the contact
        point 0.12 s before the swing arrives: Plan.release = (time, state). tick0: origin of the
        match's physics ticks, so that the release falls on a tick.
        drill: a training feed instead -- dict(shot, speed, high, x=(lo, hi) world, y=(lo, hi) depth in the
        receiver's half, over=hit from above the head, margin=net clearance): the shuttle is hit out of the
        hand to that place, with no aiming error."""
        side, st = self.side, self.style
        recv = -side
        if feed is not None:
            feed = [(t, self._vec(p)) for t, p in feed]
        b = self.body
        z_p = b.rig.z_stand - 0.05 * b.rig.scale
        low = ([c for c in self.clips if c.side == "forehand" and c.kind != "overhead" and 0.55 <= c.h_rel + z_p <= 1.35]
               or [c for c in self.clips if c.kind != "overhead"] or self.clips)
        over = drill is not None and drill.get("over")
        if over:
            low = [c for c in self.clips if c.side == "forehand" and c.kind == "overhead"] or low
        root = self.q[:2].copy()
        snap = b.snap(now, self.q.copy())
        axis = np.array([0.0, -side, -0.6]); axis /= np.linalg.norm(axis)
        shot_asked = shot
        for attempt in range(6):
            clip = low[int(self.rng.integers(len(low)))]
            if over:
                z_p = 0.5 * (clip.z_rng[0] + clip.z_rng[1])
            cz = clip.h_rel + z_p if over else min(clip.h_rel + z_p, 1.1)
            contact = np.r_[root + self.R2 @ clip.c_loc[:2], cz]
            drop = ShuttleState(contact + [0.0, 0.0, 0.5 * 9.81 * 0.12 ** 2], np.zeros(3), axis.copy())
            fine = self.phys.predict(drop, 0.2, dt=TICK, full=True, stop_z=-1.0)
            j = 120
            p, v_in = fine[1][j], fine[2][j]
            cmd = None
            if feed is not None:
                for t_k, p_k in feed:
                    if t_k - now < 1.5:
                        continue
                    for T in (1.8, 1.4, 2.2, 1.0, 0.8):
                        if t_k - T - now < 0.95:
                            continue
                        v = self.arrive(p, p_k, T, v_in, axis, np.zeros(3), 44.0)
                        if v is not None:
                            cmd = ("serve", v, t_k - T, True)
                            break
                    if cmd is not None or t_k - now > 6.0:
                        break
            if cmd is None and drill is not None:
                tgt = np.array([self.mx * self.rng.uniform(*drill["x"]), recv * self.rng.uniform(*drill["y"])])
                after = self._after_hit(p, v_in, axis, np.zeros(3))
                v = aim_shot(self.phys, after, p, tgt, drill["speed"], high=drill["high"], net_margin=drill.get("margin", 0.2))
                if v is None:
                    continue
                cmd = (drill["shot"], v, now + lead, False)
            if cmd is None:
                shot = shot_asked or ("serve_low" if self.rng.random() < st.serve_low else "serve_high")
                speed, high, (y0, y1) = SHOTS[shot]
                # diagonal: from the server's right court to the receiver's right court
                tx = (1.3 if right_court else -1.3) * (1 if side > 0 else -1) * self.mx
                tgt = np.array([np.clip(tx + self.rng.normal(0, 0.25), -2.2, 2.2), recv * self.rng.uniform(y0, y1)])
                after = self._after_hit(p, v_in, axis, np.zeros(3))
                v = aim_shot(self.phys, after, p, tgt, speed, high=high, net_margin=0.06 if shot == "serve_low" else 0.3)
                if v is None:
                    shot = "serve_high"
                    tgt[1] = recv * self.rng.uniform(*SHOTS[shot][2])
                    v = aim_shot(self.phys, after, p, tgt, SHOTS[shot][0], high=True)
                if v is None:
                    continue
                cmd = (shot, self._execution_error(v, 0.0), now + lead, False)
            shot, v_des, t_hit, is_feed = cmd
            t_rel = t_hit - 0.12
            if tick0 is not None:
                t_rel = tick0 + round((t_rel - tick0) / TICK) * TICK
            res = self._realise(now, t_rel, fine, j, v_des, [(clip, root, 0.0, 0.0, z_p, 0.0)], snap,
                                max_rel=0.05 if is_feed else 0.15, serve=True, hold=drop.pos)
            if res is None:
                continue
            swing, t_c, st_out, info = res
            self._take(swing)
            self.body.go(root)
            rel_state = ShuttleState(drop.pos.copy(), drop.vel.copy(), drop.axis.copy())
            if self.mx < 0:
                rel_state = mirror_state(rel_state)
            self.plan = self._make_plan(shot, now, t_c, st_out, "serve", is_feed, info, release=(t_rel, rel_state))
            return self.plan
        return None


# ---------------------------------------------------------------------------- a player driven by buttons
# button -> situation -> shots tried in order (the first one physics allows is played)
PAD_SHOTS = {
    "soft": dict(kill=("drop", "slice"), rear=("drop", "slice"), mid=("block", "push"), net=("net", "block"),
                 defence=("block", "net")),
    "high": dict(kill=("clear",), rear=("clear",), mid=("lift",), net=("lift",), defence=("lift",)),
    "power": dict(kill=("kill", "smash", "half_smash"), rear=("smash", "half_smash"), mid=("drive", "push"),
                  net=("push",), defence=("drive",)),
}
PAD_CN = {"soft": "轻", "high": "高", "power": "力"}


class PadPlayer(AIPlayer):
    """The same body and the same physics as the AI, but the decisions come from a game pad / keyboard /
    touch screen: a stick to run (or automatic footwork) and three buttons.

        soft  (轻)  drop from the back, block in the mid court, tumbling net shot at the net
        high  (高)  clear from the back, lift from anywhere else
        power (力)  smash / kill when the shuttle is above the tape, flat drive or push when it is not

    Pressing a button says "hit it now": the stroke is played at the first point of the incoming flight the
    player can still get the racket to (not later than WINDOW seconds ahead). Early on a high shuttle means a
    high contact; late means digging it out from below, where a smash is no longer possible. The stick at the
    moment of the press aims: left / right for the side (no direction: away from the opponent), up / down for
    deep / short. Nothing reachable: a swing at thin air.

        jump    gathers for JUMP_HOLD seconds; a shot button in that time is played from a jump (the feet leave
                the floor only for a shuttle above standing reach -- so press early; power = jump smash)
        dive    throws the body at a shuttle the feet cannot reach (lift, or a block with the soft button)

    Between rallies the stick walks the player about (roam); the umpire's clock is the match's business.
    """
    WINDOW = 0.9          # s: how far ahead of the contact a press is accepted
    LEAD = 0.14           # s: the fastest a racket can be brought to the shuttle
    SWING = 0.30          # s: what an unhurried swing takes
    RUN = 0.90            # top speed, as a fraction of the player's own
    JUMP_HOLD = 0.45      # s: how long the jump key waits for a shot button
    JUMP_LEAD = 0.50      # s: a stroke from a jump is met at least this far ahead (crouch + take-off)
    JUMP_SMASH = 1.08     # a smash from a jump is that much faster
    STROLL = 2.4          # m/s between rallies (walking about)

    def __init__(self, segments, side, style, rng=None, phys=None, difficulty="pro", auto_move=True):
        super().__init__(segments, side, style, rng, phys, difficulty)
        self.reaction = 0.03                     # the human has already reacted when the button goes down
        self.body.reaction = 0.03
        self.auto_move = auto_move
        self.live = False
        self.roam = False                        # between rallies: the stick walks the player about
        self.raw_stick = np.zeros(2)             # as given (world: x right, y to the net)
        self._jump_until = None
        self.stick = np.zeros(2)                 # internal frame: x to the (right-handed) player's right, y to the net
        self.track = None                        # (t0, prediction) of the incoming shuttle
        self.hint = None                         # where it can be met next: dict(t, pos, root)
        self._hint_t = -1.0
        self.whiffs = 0

    # ------------------------------------------------------------------ inputs
    def control(self, stick, live):
        s = np.array(stick, float)
        n = np.linalg.norm(s)
        self.raw_stick = s / max(n, 1.0)
        self.stick = np.array([self.mx * s[0], s[1]]) / max(n, 1.0)
        self.live = bool(live)

    def _speed_k(self):
        return self.RUN * (1.0 if self.stam is None else self.stam.speed())

    def _may_jump(self):
        return self._jump_until is not None

    def arm_jump(self, now):
        """The jump key: for the next JUMP_HOLD seconds a shot button plays its stroke from a jump (a higher
        contact, a harder smash). No button in that time: a hop on the spot."""
        if self.busy(now) or self._jump_until is not None:
            return False
        if self.stam is not None and not self.stam.can("jump"):
            self.message = "too tired to jump"
            return False
        self._jump_until = now + self.JUMP_HOLD
        self.body.crouch = True
        return True

    def _unarm(self):
        self._jump_until = None
        self.body.crouch = False

    def aim(self):
        """Where the stick points the next shot: (side in world x: -1 / +1, or 0 = the open side is chosen for
        you; depth 0..1 inside the range of whatever shot it will be)."""
        sx, sy = self.raw_stick
        return (float(np.sign(sx)) if abs(sx) > 0.3 else 0.0), float(np.clip(0.5 + 0.45 * sy, 0.08, 0.92))

    def incoming(self, shuttle: ShuttleState, now):
        """The opponent has hit: remember the flight (running target, timing of the press)."""
        s = self._in(shuttle)
        self.track = (now, self.phys.predict(s, 3.6, dt=0.004, full=True))
        self.hint, self._hint_t = None, -1.0

    def clear(self):
        self.track = self.hint = None

    def hint_world(self):
        h = self.hint
        if h is None:
            return None
        return dict(t=h["t"], pos=self._vec(h["pos"]), z=h["pos"][2])

    def menu(self):
        """What each button would play at the next meeting point: {button: shot} (HUD labels)."""
        h = self.hint
        if h is None or h["t"] is None:
            return {}
        sit = situation(h["pos"][2], abs(h["pos"][1]))
        return {b: PAD_SHOTS[b][sit][0] for b in PAD_SHOTS}

    def _windows(self, T, P, V, t_lo, t_hi, step=0.05):
        """Indices of the predicted flight where the shuttle could be met between t_lo and t_hi."""
        z_max = 2.75 * self.leg
        out, last = [], -1.0
        for i in np.flatnonzero((P[:, 1] * self.side > 0.3) & (T >= t_lo) & (T <= t_hi)):
            z = P[i, 2]
            if z > z_max or z < 0.22 or (V[i, 2] > 0 and z < 2.0):
                continue
            if T[i] - last >= step:
                out.append(int(i)); last = T[i]
        return out

    def _refresh_hint(self, now):
        self._hint_t = now
        t0, (T, P, V, C, W) = self.track
        if t0 + T[-1] < now:
            self.track = self.hint = None
            return
        snap = self.body.snap(now, self.pose_internal(now))
        self.hint = None
        for i in self._windows(T, P, V, now - t0 + 0.2, 9.0)[:14]:
            reach = self._reachable(P[i], t0 + T[i] - now, snap)
            if reach:
                c = min(reach, key=lambda r: r[2])
                self.hint = dict(t=t0 + T[i], pos=P[i].copy(), root=c[1])
                return
        end = P[-1]
        if end[1] * self.side > 0:                              # cannot be met in the air: at least run towards it
            self.hint = dict(t=None, pos=np.array([end[0], end[1], 0.6]), root=end[:2] - self.body.fwd * 0.6)

    # ------------------------------------------------------------------ per frame
    def update(self, now):
        b = self.body
        if self.t is None or now > self.t:
            b.set_speed(self._speed_k())
            if self._jump_until is not None and now > self._jump_until:      # nothing was hit: a hop on the spot
                self._unarm()
                if self.swing is None and (self.stam is None or self.stam.can("hop")):
                    b.hop(now)
                    if self.stam is not None:
                        self.stam.take("hop")
            if self.swing is None or now > self.swing.t[-1]:
                moving = np.linalg.norm(self.stick) > 0.15
                if self.live:
                    if self.track is not None and now - self._hint_t > 0.12:
                        self._refresh_hint(now)
                    if self.auto_move:                           # the stick only aims; the feet find the shuttle
                        b.drive = None
                        if self.hint is not None:
                            b.go(self.hint["root"], b.v_max)
                        else:
                            b.go(self.base)
                    elif moving:
                        b.drive, b.mode = self.stick * b.v_max, "ready"
                    else:
                        b.drive, b.mode = np.zeros(2), "ready"
                elif self.roam and moving:                       # between rallies: walk about (the umpire has a clock)
                    b.drive, b.mode = self.stick * self.STROLL, "relax"
                else:
                    b.drive = None
        return super().update(now)

    # ------------------------------------------------------------------ buttons
    def busy(self, now):
        return (self.swing is not None and now < self.swing.t_hit + 0.1) or now < self.locked or self.body.airborne(now)

    def press(self, button, shuttle, now, other_xy=None):
        """A shot button went down at `now`. shuttle: the incoming shuttle (None: nothing to hit).
        Returns the Plan, or None when the swing meets nothing."""
        if self.busy(now):
            return None
        import time
        tic = time.perf_counter()
        plan = None
        if shuttle is not None:
            plan = self._press(button, self._in(shuttle), now, None if other_xy is None else self._vec(np.r_[other_xy, 0])[:2])
            self.stats["plans"] += 1
            self.stats["fails"] += plan is None
            self.stats["plan_ms"].append((time.perf_counter() - tic) * 1e3)
        if plan is None:
            self._whiff(now, shuttle)
        self._pay(plan)
        self._unarm()
        return plan

    def dive(self, shuttle, now, other_xy=None, button=None):
        """The dive key: throw the body at the incoming shuttle (with a direction on the stick: only towards
        that side). Nothing a dive could reach: a dive at nothing, along the stick. Either way the player is on
        the floor for a second afterwards. Returns the Plan or None."""
        if self.busy(now):
            return None
        if self.stam is not None and not self.stam.can("dive"):
            self.message = "too tired to dive"
            return None
        import time
        tic = time.perf_counter()
        d = self.stick / np.linalg.norm(self.stick) if np.linalg.norm(self.stick) > 0.3 else None
        plan = None
        if shuttle is not None:
            s = self._in(shuttle)
            other = None if other_xy is None else self._vec(np.r_[other_xy, 0])[:2]
            plan = self._dive(s, now, self.phys.predict(s, 0.9, dt=0.004, full=True), other, button, d)
            self.stats["plans"] += 1
            self.stats["fails"] += plan is None
            self.stats["plan_ms"].append((time.perf_counter() - tic) * 1e3)
        if plan is None:
            b = self.body
            snap = b.snap(now, self.pose_internal(now))
            if d is None and shuttle is not None:                # no direction: at least towards where it comes down
                to = self._in(shuttle).pos[:2] - snap.xy
                d = to / max(np.linalg.norm(to), 1e-6)
            script = b.plan_dive(snap, now, now + 0.42, None, b.fwd if d is None else d)
            swing = free_swing(self.ik, script.base, now, script.t_end, now + 0.42)
            script.fill(swing)
            swing.script = script
            self._take(swing)
            self.locked = script.info["lock"]
            self.plan = None
            self.whiffs += 1
            self.dives += 1
            self.message = "dive at nothing"
            if self.stam is not None:
                self.stam.take("dive")
        self._unarm()
        return plan

    def _press(self, button, s, now, other):
        T, P, V, C, W = self.phys.predict(s, self.WINDOW + 0.05, dt=0.004, full=True)
        snap = self.body.snap(now, self.pose_internal(now))
        fine = None
        self.message = "too early / too far"
        # a proper swing needs about 0.3 s: the first contact that leaves that much is tried first; pressed too
        # late for that, the nearest thing to it (a snatched, compressed swing -- which often fails, as it should)
        win = self._windows(T, P, V, self.LEAD, self.WINDOW, step=0.04)
        if self._may_jump():                                    # gathered for a jump: the take-off needs its time too,
            first = [i for i in win if T[i] >= self.JUMP_LEAD]  # and the contact is wanted as high as it can be
            win = first + [i for i in win if self.SWING <= T[i] < self.JUMP_LEAD] + [i for i in reversed(win) if T[i] < self.SWING]
        else:
            win = [i for i in win if T[i] >= self.SWING] + [i for i in reversed(win) if T[i] < self.SWING]
        for i in win[:8]:
            p, v_in, axis, omega = P[i], V[i], C[i], W[i]
            reachable = self._reachable(p, T[i], snap)
            if not reachable:
                continue
            sit = situation(p[2], abs(p[1]))
            cmd = self._pad_shot(button, p, v_in, axis, omega, sit, other, min(c[3] for c in reachable))
            if cmd is None:
                continue
            shot, v_des = cmd
            if fine is None or fine[0][-1] < T[i] + 0.045:
                fine = self.phys.predict(s, T[i] + 0.06, dt=TICK, full=True, stop_z=-1.0)
            j = int(round(T[i] / TICK))
            if j + 1 >= len(fine[0]):
                continue
            res = self._realise(now, now, fine, j, v_des, reachable, snap, max_rel=0.2)
            if res is None:
                continue
            swing, t_c, st_out, info = res
            self._take(swing)
            info["intent"] = "pad:" + button
            self.plan = self._make_plan(shot, now, t_c, st_out, sit, False, info)
            self.message = f"{shot} ({sit})"
            return self.plan
        return None

    def _pad_shot(self, button, p, v_in, axis, omega, sit, other, stretch):
        st = self.style
        after = self._after_hit(p, v_in, axis, omega)
        recv = -self.side
        sx, sy = self.stick
        if abs(sx) > 0.3:
            sgn = float(np.sign(sx))
        elif other is not None and abs(other[0]) > 0.25:
            sgn = -float(np.sign(other[0]))                     # no direction held: the open side
        else:
            sgn = 1.0 if self.rng.random() < 0.5 else -1.0
        deep = float(np.clip(0.5 + 0.45 * sy, 0.08, 0.92))
        for shot in PAD_SHOTS[button][sit] + (("clear",) if p[2] >= 1.75 else ("lift",)):
            speed, high, (y0, y1) = SHOTS[shot]
            pw = self.power * (1.0 if self.stam is None else self.stam.power()) * (self.JUMP_SMASH if self._may_jump() else 1.0)
            if shot == "smash":
                speed = st.smash_speed * pw
            elif shot == "half_smash":
                speed = 0.64 * st.smash_speed * pw
            if shot in ("smash", "half_smash", "kill", "slice") and p[2] < 1.75:
                continue                                        # cannot hit down from below the tape
            x = sgn * (WIDTH_SINGLES_HALF - st.corner - self.rng.uniform(0, 0.2))
            y = recv * (y0 + (y1 - y0) * deep)
            for attempt in range(2):
                v0 = aim_shot(self.phys, after, p, (x, y), speed * (1 + 0.1 * attempt), high=high, net_margin=0.12)
                if v0 is not None:
                    return shot, self._execution_error(v0, stretch)
        return None

    def _whiff(self, now, shuttle):
        """Swing anyway: the button always does something (and a swing at nothing costs time)."""
        b = self.body
        snap = b.snap(now, self.pose_internal(now))
        high = shuttle is not None and self._in(shuttle).pos[2] > 2.3
        pool = [c for c in self.clips if c.side == "forehand" and c.kind == ("overhead" if high else "mid")] or self.clips
        clip = pool[int(self.rng.integers(len(pool)))]
        script = b.plan(snap, now, clip, snap.xy + b.coast(snap.vel), now + 0.36, 0.5 * (clip.z_rng[0] + clip.z_rng[1]))
        swing = free_swing(self.ik, script.base, now, script.t_end, now + 0.36)
        script.fill(swing)
        swing.script = script
        self._take(swing)
        self.plan = None
        self.whiffs += 1
        self.message = "whiff"

    def serve(self, button, now, right_court, tick0=None):
        return self.plan_serve(now, right_court, lead=0.85, tick0=tick0,
                               shot="serve_low" if button == "soft" else "serve_high")


class PadBot:
    """Scripted thumbs for a PadPlayer: plays whole matches for tests and demo videos, through exactly
    the interface a person uses (stick, three shot buttons, jump / dive keys, once per frame).

    extras: use the jump and dive keys (otherwise only the stick and the three buttons);
    dawdle [s]: after every point walk about for that long before getting ready (the umpire's patience)."""
    MENU = dict(kill=(("power", 0.75), ("soft", 0.25)), rear=(("power", 0.42), ("high", 0.33), ("soft", 0.25)),
                mid=(("power", 0.4), ("soft", 0.35), ("high", 0.25)), net=(("soft", 0.55), ("high", 0.35), ("power", 0.1)),
                defence=(("high", 0.55), ("soft", 0.35), ("power", 0.1)))

    def __init__(self, rng=None, lead=0.42, jitter=0.05, miss=0.0, extras=True, dawdle=0.0):
        self.rng = rng or np.random.default_rng(0)
        self.lead, self.jitter, self.miss = lead, jitter, miss
        self.extras, self.dawdle = extras, dawdle
        self._for = None                         # the incoming flight the current decision belongs to
        self._when = self._button = self._aim = None
        self._jump = False
        self._dived = None
        self._serve_t = None

    def act(self, match, now):
        """(stick, button or None, dict(jump, dive)) for this frame."""
        pad = match.pad
        stick, press, extra = np.zeros(2), None, {}
        if match.state in ("serve", "point") and self.dawdle > 0 and pad.roam and match.umpire is not None \
                and match.umpire.waited(now) < self.dawdle:
            to = np.array([1.2 * np.cos(0.9 * now), -3.2 + 0.8 * np.sin(1.3 * now)]) - match._player_xy()
            return to / max(np.linalg.norm(to), 0.3), None, extra
        if match.state == "serve":
            self._for = None
            if match.server == "p" and match.serve_plan is None and match.serve_started and pad.in_position():
                if self._serve_t is None:
                    self._serve_t = now + self.rng.uniform(0.5, 1.1)
                elif now >= self._serve_t:
                    press, self._serve_t = ("soft" if self.rng.random() < 0.45 else "high"), None
            return stick, press, extra
        self._serve_t = None
        h = pad.hint
        if match.state != "rally" or pad.track is None or h is None or pad.busy(now):
            return stick, press, extra
        d = h["root"] - pad.body.loco.pos
        n = np.linalg.norm(d)
        if not pad.auto_move and n > 0.12:                       # run there with the stick
            stick = np.array([pad.mx * d[0], d[1]]) / n * min(1.0, n / 0.5)
        if h["t"] is None:
            # cannot be met on the feet: dive for it just before it comes down
            left = pad.track[0] + pad.track[1][0][-1] - now
            if self.extras and self._dived is not pad.track and 0.25 < left < 0.55:
                self._dived = pad.track
                extra["dive"] = True
            return stick, press, extra
        if self._for is not pad.track:                           # a new flight: decide once what to do with it
            self._for = pad.track
            self._when = self.lead + self.rng.uniform(-self.jitter, self.jitter)
            sit = situation(h["pos"][2], abs(h["pos"][1]))
            names, w = zip(*self.MENU[sit])
            self._button = str(self.rng.choice(names, p=np.array(w) / sum(w)))
            self._aim = np.array([self.rng.choice([-1.0, 0.0, 1.0]), self.rng.choice([-1.0, 0.0, 1.0])])
            self._jump = bool(self.extras and self._button == "power" and sit in ("rear", "kill") and self.rng.random() < 0.5)
            if self._jump:
                self._when = 0.75 + self.rng.uniform(-self.jitter, self.jitter)       # a jump is called earlier
            if self.rng.random() < self.miss:
                self._when = -1.0                                # a lapse: no press at all
        if self._when > 0 and h["t"] - now <= self._when:
            stick, press = self._aim.copy(), self._button
            if self._jump and (pad.stam is None or pad.stam.can("jump")):
                extra["jump"] = True
            self._when = -1.0
        return stick, press, extra

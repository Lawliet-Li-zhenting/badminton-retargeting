"""Shuttlecock flight and racket impact.

Translation (Cohen et al., "The physics of badminton", New J. Phys. 17, 2015):
    a = g - (g / U_inf^2) |v| v * (1 + k_side sin^2(theta))
U_inf ~ 6.8 m/s for a feather shuttle; theta is the angle between the cork axis and the
velocity. A shuttle flying skirt-first or sideways (right after an impact) has more drag.

Orientation: the cork axis c (skirt -> cork) is aerodynamically stable and turns towards the
velocity like a damped pendulum whose natural frequency scales with speed (spatial wavelength
2*pi*l_turn): theta'' = -(|v|/l)^2 sin(theta) - 2 zeta |v|/l theta'. This reproduces the
"turnover" after a hit, when the cork points back at the racket and must flip by 180 degrees.

Spin about the axis relaxes to kappa |v| (feather vanes are angled); it is integrated and
rendered but, the shuttle being axisymmetric, it does not change the trajectory here.

Impact with the string bed (racket frame x = face normal):
    v' = v + mu [ -(1+e) (v_rel.n) n - beta v_rel_t ],   mu = m_eff / (m_s + m_eff)
with e and beta depending on whether the cork or the skirt meets the strings, and a frame
hit (outside the string ellipse but inside the frame) being weaker and deflected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = np.array([0.0, 0.0, -9.81])


@dataclass
class ShuttleParams:
    mass: float = 5.1e-3
    terminal_velocity: float = 6.8
    k_side: float = 2.0
    l_turn: float = 0.25
    zeta: float = 0.35
    spin_per_metre: float = 2 * np.pi * 2.0     # rad per metre travelled at steady state
    spin_tau: float = 0.08
    radius: float = 0.03                        # contact radius against the string plane
    cork_offset: float = 0.025                  # centre of mass -> cork tip


@dataclass
class ImpactParams:
    racket_eff_mass: float = 0.07
    e_cork: float = 0.62           # shuttle/racket-head speed ratio ~1.5 for a stationary shuttle
    e_skirt: float = 0.25
    e_frame: float = 0.30
    beta_strings: float = 0.35     # tangential grip of the string bed
    frame_margin: float = 1.15     # ellipse scale of the outer edge of the frame
    frame_scatter_deg: float = 15.0


@dataclass
class ShuttleState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))   # skirt -> cork
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))                  # tumbling rate
    spin: float = 0.0
    spin_angle: float = 0.0

    def copy(self):
        return ShuttleState(self.pos.copy(), self.vel.copy(), self.axis.copy(), self.omega.copy(),
                            self.spin, self.spin_angle)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class ShuttlePhysics:
    def __init__(self, params: ShuttleParams | None = None):
        self.p = params or ShuttleParams()

    def accel(self, vel, axis):
        p = self.p
        sp = np.linalg.norm(vel)
        if sp < 1e-9:
            return G.copy()
        cos_t = np.clip(axis @ vel / sp, -1, 1)
        drag = (9.81 / p.terminal_velocity ** 2) * (1 + p.k_side * (1 - cos_t ** 2))
        return G - drag * sp * vel

    def step(self, s: ShuttleState, dt):
        p = self.p
        # translation: midpoint rule (drag is stiff at smash speeds)
        a1 = self.accel(s.vel, s.axis)
        vm = s.vel + 0.5 * dt * a1
        a2 = self.accel(vm, s.axis)
        s.pos = s.pos + dt * vm
        s.vel = s.vel + dt * a2
        # orientation
        sp = np.linalg.norm(s.vel)
        if sp > 1e-6:
            vh = s.vel / sp
            wn = sp / p.l_turn
            torque_dir = np.cross(s.axis, vh)          # rotates the cork towards the velocity
            if np.linalg.norm(torque_dir) < 1e-6 and s.axis @ vh < 0:   # exactly backwards: unstable equilibrium
                perp = np.cross(s.axis, [0.0, 0.0, 1.0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(s.axis, [1.0, 0.0, 0.0])
                torque_dir = 1e-3 * _normalize(perp)
            w_perp = s.omega - (s.omega @ s.axis) * s.axis
            # semi-implicit Euler (omega first, then the axis) is stable for wn*dt < 2
            s.omega = w_perp + dt * (wn * wn * torque_dir - 2 * p.zeta * wn * w_perp)
            ang = np.linalg.norm(s.omega) * dt
            if ang > 1e-12:
                k = s.omega / np.linalg.norm(s.omega)
                c = s.axis
                s.axis = _normalize(c * np.cos(ang) + np.cross(k, c) * np.sin(ang) + k * (k @ c) * (1 - np.cos(ang)))
        target_spin = p.spin_per_metre * sp
        s.spin += (target_spin - s.spin) * min(1.0, dt / p.spin_tau)
        s.spin_angle = (s.spin_angle + s.spin * dt) % (2 * np.pi)
        return s

    def predict(self, s: ShuttleState, duration, dt=0.002, stop_z=0.0, full=False):
        """Future positions/velocities (until the floor). Returns times, pos, vel
        (and with full=True also the cork axis and tumbling rate).
        Same equations as step() (spin omitted: it does not affect the path), written with
        Python floats because tiny numpy arrays dominate the cost of planning."""
        p = self.p
        k0 = 9.81 / p.terminal_velocity ** 2
        ks, zeta, lt = p.k_side, p.zeta, p.l_turn
        x, y, z = map(float, s.pos)
        vx, vy, vz = map(float, s.vel)
        cx, cy, cz = map(float, s.axis)
        wx, wy, wz = map(float, s.omega)
        T, P, V = [0.0], [(x, y, z)], [(vx, vy, vz)]
        C, W = [(cx, cy, cz)], [(wx, wy, wz)]
        sqrt, cos, sin = math.sqrt, math.cos, math.sin

        def acc(ux, uy, uz):
            sp = sqrt(ux * ux + uy * uy + uz * uz)
            if sp < 1e-9:
                return 0.0, 0.0, -9.81
            ct = (cx * ux + cy * uy + cz * uz) / sp
            dr = k0 * (1 + ks * (1 - ct * ct)) * sp
            return -dr * ux, -dr * uy, -9.81 - dr * uz

        t, n = 0.0, int(math.ceil(duration / dt))
        for _ in range(n):
            ax, ay, az = acc(vx, vy, vz)
            mx, my, mz = vx + 0.5 * dt * ax, vy + 0.5 * dt * ay, vz + 0.5 * dt * az
            ax, ay, az = acc(mx, my, mz)
            x, y, z = x + dt * mx, y + dt * my, z + dt * mz
            vx, vy, vz = vx + dt * ax, vy + dt * ay, vz + dt * az
            sp = sqrt(vx * vx + vy * vy + vz * vz)
            if sp > 1e-6:
                hx, hy, hz = vx / sp, vy / sp, vz / sp
                wn = sp / lt
                tx, ty, tz = cy * hz - cz * hy, cz * hx - cx * hz, cx * hy - cy * hx
                if tx * tx + ty * ty + tz * tz < 1e-12 and cx * hx + cy * hy + cz * hz < 0:
                    px, py, pz = cy, -cx, 0.0
                    if px * px + py * py < 1e-12:
                        px, py, pz = 0.0, cz, -cy
                    pn = sqrt(px * px + py * py + pz * pz)
                    tx, ty, tz = 1e-3 * px / pn, 1e-3 * py / pn, 1e-3 * pz / pn
                wa = wx * cx + wy * cy + wz * cz
                px, py, pz = wx - wa * cx, wy - wa * cy, wz - wa * cz
                d2 = 2 * zeta * wn
                wx = px + dt * (wn * wn * tx - d2 * px)
                wy = py + dt * (wn * wn * ty - d2 * py)
                wz = pz + dt * (wn * wn * tz - d2 * pz)
                wl = sqrt(wx * wx + wy * wy + wz * wz)
                ang = wl * dt
                if ang > 1e-12:
                    kx, ky, kz = wx / wl, wy / wl, wz / wl
                    ca, sa = cos(ang), sin(ang)
                    kc = (kx * cx + ky * cy + kz * cz) * (1 - ca)
                    nx = cx * ca + (ky * cz - kz * cy) * sa + kx * kc
                    ny = cy * ca + (kz * cx - kx * cz) * sa + ky * kc
                    nz = cz * ca + (kx * cy - ky * cx) * sa + kz * kc
                    nl = sqrt(nx * nx + ny * ny + nz * nz)
                    cx, cy, cz = nx / nl, ny / nl, nz / nl
            t += dt
            T.append(t); P.append((x, y, z)); V.append((vx, vy, vz))
            if full:
                C.append((cx, cy, cz)); W.append((wx, wy, wz))
            if z <= stop_z:
                break
        if full:
            return np.array(T), np.array(P), np.array(V), np.array(C), np.array(W)
        return np.array(T), np.array(P), np.array(V)


def fly(p0, v0, T, k_drag, dt=0.004):
    """Drag-only flight (cork aligned with the velocity) for T seconds, Python floats.
    Returns (x, y, z, z_net) with z_net the height when crossing the net plane y = 0 (None if it
    does not cross) -- the inner loop of the timed-arrival solver."""
    x, y, z = p0
    vx, vy, vz = v0
    n = max(1, int(math.ceil(T / dt)))
    h = T / n
    sqrt = math.sqrt
    z_net = x_net = None
    for _ in range(n):
        sp = sqrt(vx * vx + vy * vy + vz * vz)
        d = k_drag * sp
        mx, my, mz = vx - 0.5 * h * d * vx, vy - 0.5 * h * d * vy, vz + 0.5 * h * (-9.81 - d * vz)
        sp = sqrt(mx * mx + my * my + mz * mz)
        d = k_drag * sp
        y1 = y + h * my
        if z_net is None and (y > 0.0) != (y1 > 0.0) and y1 != y:
            f = y / (y - y1)
            z_net, x_net = z + f * h * mz, x + f * h * mx
        x, y, z = x + h * mx, y1, z + h * mz
        vx, vy, vz = vx - h * d * mx, vy - h * d * my, vz + h * (-9.81 - d * mz)
    return x, y, z, z_net, x_net


def solve_arrival(p0, target, T, params: ShuttleParams, iters=14, tol=0.015, bias=None):
    """Launch velocity that puts the shuttle at `target` exactly T seconds after leaving p0
    (drag-only model, Newton with a finite-difference Jacobian). bias (3,) is added to the
    simulated end point (correction from the full model). Returns (v0, z_net, x_net) or None."""
    k = 9.81 / params.terminal_velocity ** 2
    L = 1.0 / k
    p0 = tuple(float(v) for v in p0)
    tgt = np.asarray(target, float) - (0.0 if bias is None else np.asarray(bias, float))
    d = tgt - np.asarray(p0)
    D = math.hypot(d[0], d[1])
    vh = (L / T) * (math.exp(min(D / L, 6.0)) - 1.0)                 # 1-D quadratic drag, inverted
    u = d[:2] / max(D, 1e-6)
    v = np.array([vh * u[0], vh * u[1], (d[2] + 0.5 * 9.81 * T * T) / T])
    eps = 0.05
    for _ in range(iters):
        x, y, z, z_net, x_net = fly(p0, v, T, k)
        err = tgt - np.array([x, y, z])
        if err @ err < tol * tol:
            return v, z_net, x_net
        J = np.empty((3, 3))
        for j in range(3):
            vp = v.copy(); vp[j] += eps
            xj, yj, zj, _, _ = fly(p0, vp, T, k)
            J[:, j] = (np.array([xj, yj, zj]) - (x, y, z)) / eps
        try:
            step = np.linalg.solve(J + 1e-6 * np.eye(3), err)
        except np.linalg.LinAlgError:
            return None
        nrm = np.linalg.norm(step)
        lim = 0.6 * max(np.linalg.norm(v), 8.0)                       # trust region: the map saturates with speed
        if nrm > lim:
            step *= lim / nrm
        v = v + step
        if not np.isfinite(v).all() or np.linalg.norm(v) > 200.0:
            return None
    return None


def inverse_impact(v_in, v_out, axis, ip: ImpactParams | None = None, sp: ShuttleParams | None = None):
    """String-bed normal and racket speed that turn v_in into v_out, for a racket moving along its
    own face normal (the cheapest swing). From the impact law with v_racket = s n:
        v_out = (1 - mu beta) v_in + [mu beta (v_in.n) - mu (1+e) (v_in.n - s)] n
    so n is parallel to w = v_out - (1 - mu beta) v_in and
        s = v_in.n + (|w| - mu beta v_in.n) / (mu (1 + e)).
    Returns (n, s, e): n points from the strings towards the incoming shuttle side."""
    ip = ip or ImpactParams()
    sp = sp or ShuttleParams()
    mu = ip.racket_eff_mass / (ip.racket_eff_mass + sp.mass)
    w = np.asarray(v_out, float) - (1 - mu * ip.beta_strings) * np.asarray(v_in, float)
    c = float(np.linalg.norm(w))
    n = w / max(c, 1e-9)
    e = ip.e_cork if (np.asarray(axis) @ -n) > 0.0 else ip.e_skirt
    vin_n = float(np.asarray(v_in) @ n)
    s = vin_n + (c - mu * ip.beta_strings * vin_n) / (mu * (1 + e))
    return n, s, e


def launch_batch(p0, v0s, params: ShuttleParams, dt=0.005, t_max=4.0, net_y=0.0):
    """Vectorised flight of N shuttles assumed aligned with the velocity (used for aiming).
    Returns landing points (N,3), flight times (N,), and height when crossing the net plane."""
    N = len(v0s)
    p = np.repeat(np.asarray(p0, float)[None], N, 0)
    v = np.asarray(v0s, float).copy()
    k = 9.81 / params.terminal_velocity ** 2
    traj = [p]
    for i in range(int(np.ceil(t_max / dt))):
        a1 = -k * np.sqrt(np.einsum("ij,ij->i", v, v))[:, None] * v
        a1[:, 2] -= 9.81
        vm = v + 0.5 * dt * a1
        a2 = -k * np.sqrt(np.einsum("ij,ij->i", vm, vm))[:, None] * vm
        a2[:, 2] -= 9.81
        p = p + dt * vm
        v = v + dt * a2
        traj.append(p)
        if i % 32 == 31 and (p[:, 2] <= 0).all():
            break
    X = np.stack(traj, 1)                                # (N, steps, 3)
    rows = np.arange(N)
    below = X[:, :, 2] <= 0
    j = np.where(below.any(1), below.argmax(1), -1)      # first sample at/below the floor
    landed = j > 0
    land = np.full((N, 3), np.nan)
    t_land = np.full(N, np.nan)
    a, b = X[rows[landed], j[landed] - 1], X[rows[landed], j[landed]]
    f = a[:, 2] / (a[:, 2] - b[:, 2])
    land[landed] = a + f[:, None] * (b - a)
    t_land[landed] = (j[landed] - 1 + f) * dt
    # net plane crossing before landing
    side = np.sign(X[:, :, 1] - net_y) != np.sign(X[:, :1, 1] - net_y)
    side &= np.arange(X.shape[1])[None] <= np.where(landed, j, X.shape[1] - 1)[:, None]
    net_z = np.full(N, np.nan)
    c = side.any(1)
    jc = side.argmax(1)[c]
    a, b = X[rows[c], jc - 1], X[rows[c], jc]
    f = (net_y - a[:, 1]) / (b[:, 1] - a[:, 1])
    net_z[c] = a[:, 2] + f * (b[:, 2] - a[:, 2])
    return land, t_land, net_z


def solve_launch(p0, target_xy, speed, params: ShuttleParams, high=False, net_clear=1.60, n=48, tol=0.3):
    """Elevation for a launch from p0 with the given speed towards target_xy.
    high=True picks the lob branch. Returns (v0, landing, flight_time), or None when the
    target is out of reach (landing error > tol) or the net cannot be cleared."""
    p0 = np.asarray(p0, float)
    d = np.asarray(target_xy, float) - p0[:2]
    dist = np.linalg.norm(d)
    if dist < 1e-3:
        return None
    u = d / dist
    for _ in range(2):
        phis = np.linspace(-np.pi / 3, np.radians(85), n)
        v0s = np.column_stack([np.cos(phis) * u[0], np.cos(phis) * u[1], np.sin(phis)]) * speed
        land, tl, net_z = launch_batch(p0, v0s, params)
        rng = np.einsum("ij,j->i", land[:, :2] - p0[:2], u)
        ok = ~np.isnan(rng) & ((np.isnan(net_z)) | (net_z > net_clear))
        if not ok.any():
            return None
        err = np.where(ok, np.abs(rng - dist), np.inf)
        imax = int(np.nanargmax(np.where(ok, rng, -np.inf)))
        branch = np.arange(n) >= imax if high else np.arange(n) <= imax
        err = np.where(branch, err, np.inf)
        if not np.isfinite(err).any():
            return None
        i = int(np.argmin(err))
        if err[i] < 0.1:
            return v0s[i], land[i], tl[i]
        if err[i] > 1.0:
            return None
        # refine around the best angle
        lo, hi = phis[max(i - 1, 0)], phis[min(i + 1, n - 1)]
        phis_ref = np.linspace(lo, hi, n)
        v0s = np.column_stack([np.cos(phis_ref) * u[0], np.cos(phis_ref) * u[1], np.sin(phis_ref)]) * speed
        land, tl, net_z = launch_batch(p0, v0s, params)
        rng = np.einsum("ij,j->i", land[:, :2] - p0[:2], u)
        ok = ~np.isnan(rng) & ((np.isnan(net_z)) | (net_z > net_clear))
        err = np.where(ok, np.abs(rng - dist), np.inf)
        i = int(np.argmin(err))
        if err[i] <= tol:
            return v0s[i], land[i], tl[i]
        return None
    return None


def aim_shot(phys: "ShuttlePhysics", state_after_hit, p0, target_xy, speed, high=False, iters=3, net_margin=0.05):
    """solve_launch, then correct with the full model (turnover drag after the hit) by shifting
    the aim point by the landing error. state_after_hit(v0) -> ShuttleState right after contact."""
    from .court import net_height
    tgt = np.asarray(target_xy, float).copy()
    best = None
    net_clear = 1.60
    for _ in range(iters):
        sol = None
        for f in (1.0, 0.88, 1.12, 0.77, 1.25):          # too deep -> softer, too short -> harder
            sol = solve_launch(p0, tgt, speed * f, phys.p, high=high, net_clear=net_clear)
            if sol is None and not high and speed * f < 30.0:
                # slow shots (drops, net shots) that cannot go flat over the net take the arched branch
                sol = solve_launch(p0, tgt, speed * f, phys.p, high=True, net_clear=net_clear)
            if sol is not None:
                speed *= f
                break
        if sol is None:
            break
        T, P, V = phys.predict(state_after_hit(sol[0]), 4.0, dt=0.002)
        side0 = np.sign(P[0, 1])
        cross = np.flatnonzero(np.sign(P[:, 1]) != side0)
        if len(cross) == 0 or P[cross[0], 2] < net_height(P[cross[0], 0]) + net_margin:
            net_clear += 0.2                # into the net: a higher, softer arc ...
            speed *= 1.05                   # ... and a little more pace
            continue
        err = np.asarray(target_xy) - P[-1, :2]
        best = sol[0]
        if np.linalg.norm(err) < 0.2:
            break
        tgt = tgt + err
    return best


@dataclass
class RacketPose:
    center: np.ndarray
    R: np.ndarray            # columns: face normal, width axis, shaft axis


def interp_pose(a: RacketPose, b: RacketPose, f):
    from scipy.spatial.transform import Rotation, Slerp
    c = a.center + f * (b.center - a.center)
    rots = Rotation.from_matrix(np.stack([a.R, b.R]))
    R = Slerp([0, 1], rots)(f).as_matrix()
    return RacketPose(c, R)


def racket_point_velocity(a: RacketPose, b: RacketPose, dt, point):
    """Velocity of a point rigidly attached to the racket, from two poses dt apart."""
    v_c = (b.center - a.center) / dt
    dR = b.R @ a.R.T
    from scipy.spatial.transform import Rotation
    w = Rotation.from_matrix(dR).as_rotvec() / dt
    return v_c + np.cross(w, point - b.center)


def hitting_speed(v, side=-1):
    """Speed of a racket-head velocity that can send a shuttle over the net: forward (towards the net)
    plus across the body (backhand / round-the-head swings travel mostly sideways). Swings going
    away from the net count for nothing. v: (..., 3) relative to the pelvis; side: -1 for the near player."""
    v = np.asarray(v, float)
    fwd = -side * v[..., 1]
    h = np.sqrt(np.maximum(fwd, 0.0) ** 2 + (0.7 * v[..., 0]) ** 2)
    return np.where(fwd > -0.35 * np.linalg.norm(v[..., :2], axis=-1), h, 0.0)


class RacketImpact:
    def __init__(self, half_width=0.10, half_length=0.115, params: ImpactParams | None = None,
                 shuttle: ShuttleParams | None = None, rng=None):
        self.hw, self.hl = half_width, half_length
        self.p = params or ImpactParams()
        self.sp = shuttle or ShuttleParams()
        self.rng = rng or np.random.default_rng(0)

    def check(self, s0: ShuttleState, s1: ShuttleState, r0: RacketPose, r1: RacketPose):
        """Swept test over one sub-step. Returns (kind, contact_point, normal) or None."""
        n0, n1 = r0.R[:, 0], r1.R[:, 0]
        d0 = (s0.pos - r0.center) @ n0
        d1 = (s1.pos - r1.center) @ n1
        rad = self.sp.radius
        if d0 * d1 > 0 and min(abs(d0), abs(d1)) > rad:
            return None
        f = 0.5 if d0 == d1 else np.clip(d0 / (d0 - d1), 0.0, 1.0)
        rp = interp_pose(r0, r1, f) if (d0 * d1 <= 0) else (r1 if abs(d1) < abs(d0) else r0)
        p = s0.pos + f * (s1.pos - s0.pos)
        loc = rp.R.T @ (p - rp.center)
        e = (loc[1] / self.hw) ** 2 + (loc[2] / self.hl) ** 2
        if e <= 1.0:
            kind = "strings"
        elif e <= self.p.frame_margin ** 2:
            kind = "frame"
        else:
            return None
        return kind, p, rp.R[:, 0] * (1.0 if d0 >= 0 else -1.0)

    def respond(self, s: ShuttleState, kind, point, normal, v_racket):
        """normal points from the racket towards the side the shuttle came from."""
        p = self.p
        v_rel = s.vel - v_racket
        vn = v_rel @ normal
        if vn >= 0:                              # separating already
            return None
        mu = p.racket_eff_mass / (p.racket_eff_mass + self.sp.mass)
        cork_first = (s.axis @ -normal) > 0.0    # cork pointing into the strings
        e = p.e_frame if kind == "frame" else (p.e_cork if cork_first else p.e_skirt)
        v_t = v_rel - vn * normal
        dv = mu * (-(1 + e) * vn * normal - p.beta_strings * v_t)
        v_new = s.vel + dv
        if kind == "frame":
            ax = _normalize(np.cross(normal, self.rng.normal(size=3)))
            ang = np.radians(self.rng.normal(0, p.frame_scatter_deg))
            v_new = v_new * np.cos(ang) + np.cross(ax, v_new) * np.sin(ang) + ax * (ax @ v_new) * (1 - np.cos(ang))
        s.vel = v_new
        s.pos = point + normal * (self.sp.radius + 1e-3)
        # the strings also kick the shuttle's tumbling (contact is off its centre of mass)
        s.omega = s.omega + np.cross(s.axis, normal) * 30.0
        return dict(kind=kind, cork_first=bool(cork_first), e=e, speed_in=float(np.linalg.norm(v_rel)),
                    speed_out=float(np.linalg.norm(v_new)), racket_speed=float(np.linalg.norm(v_racket)))

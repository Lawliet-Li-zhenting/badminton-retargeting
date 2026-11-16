"""Turn per-frame MediaPipe landmarks into global, character-scaled 3D keypoint targets.

Steps: One-Euro filtering -> root translation from a perspective fit of the 2D landmarks
-> ground plane fit in camera space (levels a tilted camera, gives jump height)
-> camera axes to z-up world (person facing the camera faces +y) -> optional left/right
mirror -> uniform scale to the character.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .character import KP, MIRROR_PERM, MP_IDS, SOLE_KP
from .config import SourceConfig
from .pose import PoseFrame

# camera (x right, y down, z away) -> world (x = person's right when facing the camera, y towards camera, z up)
CAM_TO_WORLD = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])

SCALE_SEGMENTS = [("l_hip", "l_knee"), ("r_hip", "r_knee"), ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
                  ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"), ("l_elbow", "l_wrist"),
                  ("r_elbow", "r_wrist")]


class OneEuro:
    """Vectorised One-Euro filter (Casiez et al. 2012)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def segment_length(P):
    return sum(np.linalg.norm(P[KP[a]] - P[KP[b]]) for a, b in SCALE_SEGMENTS)


def solve_camera_translation(world, image, vis, w, h, hfov_deg):
    """Least-squares t with  pi(world_i + t) ~ pixel_i  (pinhole, principal point at centre)."""
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(hfov_deg))
    u = image[:, 0] * w - 0.5 * w
    v = image[:, 1] * h - 0.5 * h
    X, Y, Z = world.T
    wt = np.sqrt(np.clip(vis, 0.05, 1.0))
    A = np.zeros((2 * len(u), 3)); b = np.zeros(2 * len(u))
    A[0::2, 0] = f; A[0::2, 2] = -u; b[0::2] = u * Z - f * X
    A[1::2, 1] = f; A[1::2, 2] = -v; b[1::2] = v * Z - f * Y
    A *= np.repeat(wt, 2)[:, None]; b *= np.repeat(wt, 2)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def rotation_between(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class GroundPlane:
    """Ground plane y = a x + b z + c in camera coordinates (y down).

    Fitted to the lowest foot of every frame as a lower envelope (airborne samples are
    down-weighted); the slopes are ridge-regularised towards the body's average up axis,
    which is all we have while the player stands still.
    """

    def __init__(self, maxlen=900, refit_every=10, ridge=30.0, air_tol=0.03):
        self.buf = deque(maxlen=maxlen)
        self.refit_every, self.ridge, self.air_tol = refit_every, ridge, air_tol
        self.up_sum = np.zeros(3)
        self.coef = None
        self.k = 0

    def prior(self):
        n = self.up_sum / max(np.linalg.norm(self.up_sum), 1e-9)
        if n[1] > -0.3:                                   # degenerate: assume an upright camera
            n = np.array([0.0, -1.0, 0.0])
        return -n[0] / n[1], -n[2] / n[1]

    def update(self, cam_pts, torso_up, sole_idx):
        self.up_sum += torso_up / max(np.linalg.norm(torso_up), 1e-9)
        sole = cam_pts[sole_idx]
        up = self.up()
        self.buf.append(sole[np.argmin(sole @ up)])
        if self.coef is None or self.k % self.refit_every == 0:
            self._fit()
        self.k += 1

    def _fit(self):
        X = np.array(self.buf)
        a0, b0 = self.prior()
        A = np.column_stack([X[:, 0], X[:, 2], np.ones(len(X))])
        y = X[:, 1]
        w = np.ones(len(X))
        reg = np.diag([self.ridge, self.ridge, 0.0])
        prior = np.array([a0, b0, 0.0])
        for _ in range(4):
            Aw = A * w[:, None]
            coef = np.linalg.solve(Aw.T @ A + reg + 1e-9 * np.eye(3), Aw.T @ y + reg @ prior)
            res = A @ coef - y                     # > 0: sample above the plane
            w = np.where(res > self.air_tol, 0.05, 1.0)
        self.coef = coef

    def up(self):
        if self.coef is None:
            a, b = self.prior()
        else:
            a, b = self.coef[:2]
        n = np.array([a, -1.0, b])
        return n / np.linalg.norm(n)

    def height(self, p):
        a, b, c = self.coef
        return (a * p[..., 0] + b * p[..., 2] + c - p[..., 1]) / np.sqrt(a * a + 1 + b * b)


def pixel_ray(image_xy, w, h, hfov_deg):
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(hfov_deg))
    return np.array([(image_xy[0] * w - 0.5 * w) / f, (image_xy[1] * h - 0.5 * h) / f, 1.0])


class GroundRayTranslator:
    """Root translation from the planted foot: its pixel ray intersected with the ground plane.

    Much less noisy than the body-scale depth for distant players. Jumps are detected without
    any depth: if the foot were grounded, the hip's own pixel ray at the same distance should
    pass at the posture hip height; a hip seen higher than that means the player is airborne.
    While airborne the in-plane position is held and the height is read off the foot ray.
    """

    def __init__(self, jump_tol_m=0.15, max_flight_s=0.8, min_frames=2):
        self.jump_tol, self.max_flight, self.min_frames = jump_tol_m, max_flight_s, min_frames
        self.last_contact = None
        self.airborne = False
        self.flight_start = None
        self.hip_excess = 0.0
        self.history = deque(maxlen=150)       # the bias (unknown FOV, posture error) is removed by its median
        self.above = 0

    def __call__(self, world, image, sole_mp, plane: GroundPlane, up, t_scale, w, h, hfov, t):
        a, b, c = plane.coef
        k = sole_mp[int(np.argmin(world[sole_mp] @ up))]
        r = pixel_ray(image[k, :2], w, h, hfov)
        den = r[1] - a * r[0] - b * r[2]
        if den <= 0.02:                        # ray almost parallel to the ground: no usable intersection
            return t_scale, False
        foot = r * (c / den)
        t_ray = foot - world[k]
        hor = lambda p: p - (p @ up) * up
        hip_w = world[[23, 24]].mean(0)
        r_hip = pixel_ray(image[[23, 24], :2].mean(0), w, h, hfov)
        hip_on_ray = r_hip * (np.linalg.norm(hor(hip_w + t_ray)) / max(np.linalg.norm(hor(r_hip)), 1e-6))
        self.hip_excess = float(plane.height(hip_on_ray) - plane.height(hip_w + t_ray))
        bias = np.median(self.history) if len(self.history) > 20 else 0.0
        self.above = self.above + 1 if self.hip_excess - bias > self.jump_tol else 0
        airborne = self.above >= self.min_frames
        if not airborne:
            self.history.append(self.hip_excess)
        if airborne and not self.airborne:
            self.flight_start = t
        if airborne and t - self.flight_start > self.max_flight:
            airborne = False                  # real jumps are short: treat as a depth error instead
        self.airborne = airborne
        if not airborne or self.last_contact is None:
            self.last_contact = foot
            return t_ray, False
        # airborne: point on the foot ray with the same in-plane distance as the last contact
        target = np.linalg.norm(hor(self.last_contact))
        hr = np.linalg.norm(hor(r))
        p = r * (target / max(hr, 1e-6))
        return p - world[k], True


class SourceBuilder:
    def __init__(self, char_neutral_kp, cfg: SourceConfig | None = None, origin=(0.0, 0.0), yaw=0.0):
        self.cfg = cfg or SourceConfig()
        self.char_len = segment_length(char_neutral_kp)
        self.char_sole_z = float(np.min(char_neutral_kp[SOLE_KP, 2]))
        self.sole_rest = char_neutral_kp[SOLE_KP, 2].copy()
        self.origin = np.asarray(origin, float)
        c, s = np.cos(yaw), np.sin(yaw)
        self.R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.reset()

    def reset(self):
        c = self.cfg
        self.f_world = OneEuro(c.filter_min_cutoff, c.filter_beta)
        self.f_root = OneEuro(c.root_min_cutoff, c.root_beta)
        self.f_lift = OneEuro(3.0, 0.5)
        self.f_img = OneEuro(2.5, 0.0)
        self.t0 = None
        self.ground = GroundPlane()
        self.ray = GroundRayTranslator()
        self.t_raw = self.t_prev = None
        self.img_prev = self.t_img = None
        self.lengths = deque(maxlen=120)
        self.u_valid = self.t_valid = None
        self.rejected = 0

    def plausible(self, pf: PoseFrame):
        """MediaPipe sometimes flips the skeleton (person partly out of frame, occlusion).
        A badminton player never has the shoulders below the hips, and the torso cannot turn
        by more than ~70 deg within 0.3 s."""
        w = pf.world
        u = w[[11, 12]].mean(0) - w[[23, 24]].mean(0)
        u = u / max(np.linalg.norm(u), 1e-6)
        ref = self.ground.up() if len(self.ground.buf) >= 15 else np.array([0.0, -1.0, 0.0])
        if u @ ref < self.cfg.min_torso_up:
            return False, "inverted"
        if self.u_valid is not None and pf.t - self.t_valid < 0.3 and u @ self.u_valid < np.cos(np.radians(70)):
            return False, "torso jump"
        self.u_valid, self.t_valid = u, pf.t
        return True, ""

    def __call__(self, pf: PoseFrame):
        """Returns (P, vis, info); P and vis are None when the frame is rejected (info['reason'])."""
        cfg = self.cfg
        ok, why = self.plausible(pf)
        if not ok:
            self.rejected += 1
            return None, None, dict(valid=False, reason=why)
        world = self.f_world(pf.world, pf.t)
        inside = (pf.image[:, 0] > 0) & (pf.image[:, 0] < 1) & (pf.image[:, 1] > 0) & (pf.image[:, 1] < 1)
        t_scale = solve_camera_translation(world, pf.image, pf.vis * inside, pf.width, pf.height, cfg.hfov_deg)
        torso_up = world[[11, 12]].mean(0) - world[[27, 28]].mean(0)
        sole_mp = MP_IDS[SOLE_KP]
        self.ground.update(world + t_scale, torso_up, sole_mp)
        up = self.ground.up()
        airborne, cam_t = False, t_scale
        if len(self.ground.buf) >= 15:
            t_ray, airborne = self.ray(world, pf.image, sole_mp, self.ground, up, t_scale,
                                       pf.width, pf.height, cfg.hfov_deg, pf.t)
            if cfg.translation_mode == "ground_ray":
                cam_t = t_ray
        if self.t_raw is not None:
            dt = max(pf.t - self.t_prev, 1e-3)
            step = cam_t - self.t_raw
            lim = cfg.max_root_speed_mps * dt
            if np.linalg.norm(step) > lim:
                cam_t = self.t_raw + step * (lim / np.linalg.norm(step))
        self.t_raw, self.t_prev = cam_t.copy(), pf.t
        cam_t = self.f_root(cam_t, pf.t)
        if self.t0 is None:
            self.t0 = cam_t.copy()
        lift = 0.0
        if airborne:
            lift = max(0.0, float(self.ground.height(world[sole_mp] + cam_t).min()) - cfg.jump_deadband_m)
        lift = self.f_lift(np.array([lift]), pf.t)[0]
        R_level = rotation_between(up, np.array([0.0, -1.0, 0.0]))

        P = (world[MP_IDS] @ R_level.T) @ CAM_TO_WORLD.T
        vis = pf.vis[MP_IDS].copy()
        img = pf.image[MP_IDS, :2] * [pf.width, pf.height]
        if cfg.handedness == "left":
            P[:, 0] *= -1
            P, vis, img = P[MIRROR_PERM], vis[MIRROR_PERM], img[MIRROR_PERM]
        self.lengths.append(segment_length(P))
        scale = self.char_len / max(np.median(self.lengths), 1e-3)

        P = P * scale
        P[:, 2] += self.char_sole_z - P[SOLE_KP, 2].min()
        contact = self._contact(P, img, pf.t)
        P[:, 2] += lift * scale
        if cfg.track_translation:
            d = (R_level @ (cam_t - self.t0)) @ CAM_TO_WORLD.T
            if cfg.handedness == "left":
                d[0] *= -1
            P[:, 0] += d[0] * cfg.court_gain[0]
            P[:, 1] += d[1] * cfg.court_gain[1]
        P = P @ self.R_yaw.T
        P[:, :2] += self.origin
        if lift > 0.05:
            contact[:] = False
        return P, vis, dict(valid=True, scale=scale, lift=lift, cam_t=cam_t, t_scale=t_scale, airborne=airborne,
                            contact=contact)

    def _contact(self, P, img, t):
        """Sole keypoints that are low in the body posture and nearly still in the image."""
        sole = np.array(SOLE_KP)
        rel = P[sole, 2] - self.sole_rest
        low = rel - rel.min() < self.cfg.contact_height_m
        px_h = max(np.ptp(img[:, 1]), 1.0)
        img = self.f_img(img, t)
        if self.img_prev is None:
            still = np.ones(len(sole), bool)
        else:
            v = np.linalg.norm(img[sole] - self.img_prev[sole], axis=1) / max(t - self.t_img, 1e-3) / px_h
            still = v < self.cfg.contact_image_speed
        self.img_prev, self.t_img = img.copy(), t
        return low & still

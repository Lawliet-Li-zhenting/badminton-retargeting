"""Calibrated broadcast video -> court-frame body keypoints of the camera-side player.

Pose pass
    every frame: main-camera test (court-line model vs white pixels), MediaPipe with several
    candidates, keep the person whose feet project into the camera-side half (TemPose-style court
    filtering) and who is closest to the previous position.

Reconstruction (offline, whole clip)
    1. MediaPipe world landmarks are metric and hip-centred in the frame of a virtual camera looking
       at the person; rotate them by (hip-ray correction) and the calibrated camera rotation into
       the court frame -> P_rel (z up).
    2. Body scale s (MediaPipe metres -> real metres) and root T from a linear least-squares fit of
       every keypoint's pixel (u - cx) Xc_z - f Xc_x = 0 ... plus a ground row for the stance foot.
       s is solved per standing frame and its median is used for the whole clip.
    3. Airborne test: with the low camera a foot in the air projects far behind the body, so the
       ground-anchored depth and the image-size depth disagree (hysteresis on their difference).
    4. Constant-velocity Kalman filter + RTS smoother on T, with each frame's measurement covariance
       from the LSQ normal matrix (depth is the weak direction).
    5. Footplants per heel / toe (Kovar et al. 2002; contact labels from height + speed with
       hysteresis as in WHAM): intervals of >= 3 frames.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calib import ROWS_Y, SIDES_X, Camera, line_mask
from .court import LENGTH_HALF, SHORT_SERVICE, WIDTH_DOUBLES_HALF

MP_L_HEEL, MP_R_HEEL, MP_L_TOE, MP_R_TOE = 29, 30, 31, 32
MP_SOLE = [MP_L_HEEL, MP_R_HEEL, MP_L_TOE, MP_R_TOE]
SOLE_H = np.array([0.045, 0.045, 0.02, 0.02])       # landmark height above the floor when planted


# ---------------------------------------------------------------------------------------- pose pass
def court_line_samples(cam: Camera, step=0.25):
    pts = []
    for y in ROWS_Y:
        for x in np.arange(-WIDTH_DOUBLES_HALF, WIDTH_DOUBLES_HALF + 1e-6, step):
            pts.append([x, y, 0.0])
    for x in SIDES_X:
        for y in np.arange(-LENGTH_HALF, LENGTH_HALF + 1e-6, step):
            pts.append([x, y, 0.0])
    uv = cam.project(pts)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return np.round(uv[ok]).astype(int)


def main_camera_score(frame, samples):
    """Fraction of court-line model pixels that are white-ish in this frame (any 3x3 neighbour)."""
    m = cv2.dilate(line_mask(frame), np.ones((3, 3), np.uint8))
    return float((m[samples[:, 1], samples[:, 0]] > 0).mean())


def foot_ground_xy(cam: Camera, pf):
    uv = pf.image[MP_SOLE, :2] * [pf.width, pf.height]
    g = cam.ground_point(uv, 0.03)
    w = pf.vis[MP_SOLE] + 1e-3
    return (g[:, :2] * w[:, None]).sum(0) / w.sum()


class CropTracker:
    """Square crop around the tracked player, upscaled so the pose detector sees a large person.
    The crop only moves when the player leaves a dead zone, which keeps the landmarker's own temporal
    tracking (VIDEO mode) valid; when the player is lost a fixed crop of the camera-side half is used."""

    def __init__(self, cam: Camera, size=448):
        self.cam, self.size = cam, size
        self.box = None                                  # (cx, cy, L) in pixels
        # default: camera-side half of the court including players' height
        corners = np.array([[x, y, z] for x in (-3.8, 3.8) for y in (-7.8, 0.0) for z in (0.0, 2.4)])
        uv = cam.project(corners)
        u0, v0 = np.clip(uv.min(0), 0, None)
        u1, v1 = np.minimum(uv.max(0), [cam.width, cam.height])
        L = max(u1 - u0, v1 - v0)
        self.default = ((u0 + u1) / 2, (v0 + v1) / 2, L)
        # re-acquisition tiles: with a high / wide camera the half-court crop makes the player too
        # small for the detector, so search overlapping tiles of about 3 player heights instead
        ph = np.ptp(cam.project([[0, -LENGTH_HALF, 0], [0, -LENGTH_HALF, 1.8]])[:, 1])
        Lt = float(np.clip(3.0 * ph, 220, L))
        nx = int(np.ceil(max(u1 - u0 - Lt, 0) / (0.6 * Lt))) + 1
        ny = int(np.ceil(max(v1 - v0 - Lt, 0) / (0.6 * Lt))) + 1
        self.tiles = [(cx, cy, Lt) for cx in np.linspace(u0 + Lt / 2, max(u1 - Lt / 2, u0 + Lt / 2), nx)
                      for cy in np.linspace(v0 + Lt / 2, max(v1 - Lt / 2, v0 + Lt / 2), ny)]
        if len(self.tiles) == 1:
            self.tiles = [self.default]

    def crop(self, frame):
        cx, cy, L = self.box or self.default
        s = self.size / L
        M = np.array([[s, 0, self.size / 2 - s * cx], [0, s, self.size / 2 - s * cy]])
        self.M = M
        return cv2.warpAffine(frame, M, (self.size, self.size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    def to_full(self, pf):
        """PoseFrame on the crop -> PoseFrame on the full frame (normalised image coords)."""
        M = self.M
        s = M[0, 0]
        uv = (pf.image[:, :2] * self.size - M[:, 2]) / s
        im = pf.image.copy()
        im[:, 0] = uv[:, 0] / self.cam.width
        im[:, 1] = uv[:, 1] / self.cam.height
        im[:, 2] = pf.image[:, 2] * self.size / s / self.cam.width
        pf.image, pf.width, pf.height = im, self.cam.width, self.cam.height
        return pf

    def update(self, pf):
        if pf is None:
            self.box = None
            return
        uv = pf.image[:, :2] * [pf.width, pf.height]
        good = pf.vis > 0.3
        lo, hi = uv[good].min(0), uv[good].max(0)
        c = (lo + hi) / 2
        L = float(np.clip(1.5 * max(hi[1] - lo[1], 1.2 * (hi[0] - lo[0])), 200, 544))
        if self.box is None:
            self.box = (c[0], c[1], L)
            return
        bx, by, bL = self.box
        dz = 0.12 * bL
        nx = bx + np.sign(c[0] - bx) * max(abs(c[0] - bx) - dz, 0)
        ny = by + np.sign(c[1] - by) * max(abs(c[1] - by) - dz, 0)
        nL = bL if 0.85 < L / bL < 1.2 else 0.7 * bL + 0.3 * L
        self.box = (nx, ny, nL)


def _select(cam, cands, prev):
    best, best_d, bg = None, np.inf, None
    for pf in cands:
        g = foot_ground_xy(cam, pf)
        if not (-LENGTH_HALF - 2.5 < g[1] < -0.2 and abs(g[0]) < WIDTH_DOUBLES_HALF + 1.5):
            continue
        d = np.linalg.norm(g - prev) if prev is not None else -np.ptp(pf.image[:, 1])
        if d < best_d:
            best, best_d, bg = pf, d, g
    if best is None or (prev is not None and best_d > 1.5):
        return None, None
    return best, bg


def pose_pass(video, cam: Camera, model="heavy", start=0, count=0, cache=None, min_score=0.55, log_every=600):
    """Returns dict of arrays (T frames) for the camera-side player."""
    from .pose import PoseEstimator
    if cache and Path(cache).exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = count or (N - start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    est = PoseEstimator(model, num_poses=2, min_conf=0.3)
    fallback = PoseEstimator(model, num_poses=3, min_conf=0.3, video=False)
    tracker = CropTracker(cam)
    samples = court_line_samples(cam)
    out = dict(ok=np.zeros(count, bool), main=np.zeros(count, bool), score=np.zeros(count),
               world=np.zeros((count, 33, 3)), image=np.zeros((count, 33, 3)), vis=np.zeros((count, 33)),
               ground=np.full((count, 2), np.nan), lat=np.zeros(count), n_people=np.zeros(count, int),
               fallback=np.zeros(count, bool), reset=np.zeros(count, bool))
    prev, lost, last_small = None, 0, None
    t0 = time.time()
    for k in range(count):
        ok, frame = cap.read()
        if not ok:
            for key in out:
                out[key] = out[key][:k]
            break
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
        if last_small is not None and np.abs(small - last_small).mean() > 18:   # hard cut
            prev = None
            out["reset"][k] = True
        last_small = small
        if lost > 6 and prev is not None:                  # track lost for 0.2 s: re-acquire anywhere
            prev = None
            out["reset"][k] = True
        sc = main_camera_score(frame, samples)
        out["score"][k] = sc
        out["main"][k] = sc >= min_score
        if not out["main"][k]:
            prev = None
            tracker.update(None)
            continue
        t = (start + k) / fps
        cands = [tracker.to_full(pf) for pf in est.detect_all(tracker.crop(frame), t)]
        best, bg = _select(cam, cands, prev)
        if best is None:                                  # retry on the default half-court crop / tiles
            box = tracker.box
            cands = []
            for tile in [tracker.default] + [b for b in tracker.tiles if b != tracker.default]:
                tracker.box = tile
                cands += [tracker.to_full(pf) for pf in fallback.detect_all(tracker.crop(frame), t)]
                best, bg = _select(cam, cands, prev)
                if best is not None:
                    break
            tracker.box = box
            out["fallback"][k] = True
        out["n_people"][k] = len(cands)
        tracker.update(best)
        if best is None:
            lost += 1
            continue
        lost = 0
        prev = bg
        out["ok"][k] = True
        out["world"][k], out["image"][k], out["vis"][k] = best.world, best.image, best.vis
        out["ground"][k] = bg
        out["lat"][k] = best.latency_ms
        if log_every and k % log_every == 0:
            print(f"  pose {k}/{count}  {time.time() - t0:.0f}s  main={out['main'][:k + 1].mean():.2f} "
                  f"ok={out['ok'][:k + 1].mean():.2f}", flush=True)
    out["fps"] = np.array(fps)
    out["start"] = np.array(start)
    out["width"] = np.array(cam.width); out["height"] = np.array(cam.height)
    if cache:
        np.savez_compressed(cache, **out)
    return out


# ----------------------------------------------------------------------------------- reconstruction
def _rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def to_court(cam: Camera, world, image, width, height):
    """MediaPipe world landmarks (33,3) -> court-frame offsets from the hip centre."""
    hip_uv = image[[23, 24], :2].mean(0) * [width, height]
    K = cam.Kmat
    ray = np.array([(hip_uv[0] - K[0, 2]) / K[0, 0], (hip_uv[1] - K[1, 2]) / K[1, 1], 1.0])
    R_ray = _rot_between(np.array([0.0, 0.0, 1.0]), ray)     # virtual (person-centred) camera -> real camera
    return world @ (cam.R.T @ R_ray).T


def _pixel_rows(cam: Camera, P_rel, uv, w, s=None):
    """Rows of A [s?, Tx, Ty, Tz] = b for (u-cx) Xc_z - fx Xc_x = 0 and (v-cy) Xc_z - fy Xc_y = 0,
    Xc = R (s P + T) + t. Rows are divided by the current depth estimate so residuals are pixels."""
    K, R, t = cam.Kmat, cam.R, cam.t
    rows_A, rows_b = [], []
    du = uv[:, 0] - K[0, 2]; dv = uv[:, 1] - K[1, 2]
    for comp, d, f in ((0, du, K[0, 0]), (1, dv, K[1, 1])):
        a = d[:, None] * R[2][None] - f * R[comp][None]           # coefficient on X (court)
        c = d * t[2] - f * t[comp]                                 # constant
        if s is None:
            A = np.column_stack([np.einsum("ij,ij->i", a, P_rel), a])
            b = -c
        else:
            A = a
            b = -c - s * np.einsum("ij,ij->i", a, P_rel)
        rows_A.append(A * w[:, None]); rows_b.append(b * w)
    return np.vstack(rows_A), np.concatenate(rows_b)


@dataclass
class Reconstruction:
    fps: float
    ok: np.ndarray            # (T,) usable frames
    P: np.ndarray             # (T,33,3) court-frame keypoints (real metres)
    vis: np.ndarray           # (T,33)
    T: np.ndarray             # (T,3) smoothed hip centre
    airborne: np.ndarray      # (T,)
    contact: np.ndarray       # (T,4) heel L, heel R, toe L, toe R footplants
    scale: float
    sole_h: np.ndarray        # (4,) floor height of the sole landmarks when planted
    info: dict = field(default_factory=dict)
    uv: np.ndarray | None = None      # (T,33,2) lightly smoothed image keypoints (px)
    uv_w: np.ndarray | None = None    # (T,33) their weights (0 where not observed)


def reconstruct(cam: Camera, data, sigma_px=3.0, sigma_ground=0.03, accel_std=12.0, air_on=0.55, air_off=0.3,
                max_gap=8, smooth_sigma=1.0, sigma_scale=0.06, resolve_flips=False,
                accel_std_depth=None, plant_min_len=5):
    ok = data["ok"].astype(bool)
    Tn = len(ok)
    fps = float(data["fps"]); dt = 1.0 / fps
    W, H = int(data["width"]), int(data["height"])
    P_rel = np.zeros((Tn, 33, 3))
    for k in np.flatnonzero(ok):
        P_rel[k] = to_court(cam, data["world"][k], data["image"][k], W, H)
    vis = data["vis"].copy()
    uv_all = data["image"][:, :, :2] * [W, H]
    # body below the visible frame: MediaPipe still predicts, but trust it less
    inside = (uv_all[..., 1] < H - 2) & (uv_all[..., 0] > 1) & (uv_all[..., 0] < W - 2)
    wpx = np.clip(vis, 0.05, 1.0) * inside / sigma_px
    flips = None
    if resolve_flips:
        P_rel, flips = resolve_arm_flips(P_rel, uv_all, vis, ok, cam, fps)

    # 1. global scale from frames where the lowest sole landmark is clearly the stance foot
    s_est = []
    for k in np.flatnonzero(ok)[::3]:
        A, b = _pixel_rows(cam, P_rel[k], uv_all[k], wpx[k])
        depth = _depth_guess(cam, uv_all[k])
        A = A / depth; b = b / depth
        j = int(np.argmin(P_rel[k, MP_SOLE, 2]))
        lo = np.sort(P_rel[k, MP_SOLE, 2])
        if lo[1] - lo[0] > 0.03 and np.ptp(P_rel[k, MP_SOLE, 2]) > 0.15:
            continue                                   # one foot clearly off the floor and other unsure
        g = np.r_[P_rel[k, MP_SOLE[j], 2], 0, 0, 1.0] / sigma_ground
        A = np.vstack([A, g]); b = np.r_[b, SOLE_H[j] / sigma_ground]
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        if 0.7 < x[0] < 1.4:
            s_est.append(x[0])
    scale = float(np.median(s_est)) if s_est else 1.0

    # 2. per-frame measurements: free (size-based depth) and ground-anchored
    meas_T = np.full((Tn, 3), np.nan); meas_C = np.zeros((Tn, 3, 3))
    free_T = np.full((Tn, 3), np.nan); free_C = np.zeros((Tn, 3, 3)); diff = np.full(Tn, np.nan)
    fwd = cam.R[2]                                       # optical axis in court frame
    scale_k = np.full(Tn, scale)
    for k in np.flatnonzero(ok):
        depth = _depth_guess(cam, uv_all[k])
        # rows are in pixels * depth; normalise by the rough depth so residuals are pixels
        A, b = _pixel_rows(cam, P_rel[k], uv_all[k], wpx[k], s=scale)
        A = A / depth; b = b / depth
        N = A.T @ A
        free_T[k] = np.linalg.solve(N + 1e-9 * np.eye(3), A.T @ b)
        free_C[k] = np.linalg.inv(N + 1e-9 * np.eye(3))
        # ground-anchored: per-frame scale with a prior, so the size cue cannot override the floor
        A, b = _pixel_rows(cam, P_rel[k], uv_all[k], wpx[k])
        A = A / depth; b = b / depth
        j = int(np.argmin(P_rel[k, MP_SOLE, 2]))
        Ag = np.vstack([A, np.r_[P_rel[k, MP_SOLE[j], 2], 0, 0, 1.0] / sigma_ground,
                        np.r_[1.0, 0, 0, 0] / sigma_scale])
        bg = np.r_[b, SOLE_H[j] / sigma_ground, scale / sigma_scale]
        Ng = Ag.T @ Ag
        x = np.linalg.solve(Ng, Ag.T @ bg)
        scale_k[k] = x[0]
        meas_T[k] = x[1:]
        meas_C[k] = np.linalg.inv(Ng)[1:, 1:]
        diff[k] = (meas_T[k] - free_T[k]) @ fwd          # > 0: ground solution is further away

    # 3. airborne: robust baseline of the depth disagreement, hysteresis, short gaps closed
    d = diff.copy()
    base = np.nanmedian(d)
    dd = d - base
    # feet must be really observed: near the baseline they leave the bottom of the frame and
    # MediaPipe extrapolates them, which looks exactly like a lifted foot
    feet_seen = ((uv_all[:, MP_SOLE, 1] < H - 12).all(1) & (vis[:, MP_SOLE] > 0.5).all(1))
    air = np.zeros(Tn, bool)
    state = False
    for k in range(Tn):
        if not ok[k] or np.isnan(dd[k]) or not feet_seen[k]:
            state = False
        elif not state and dd[k] > air_on:
            state = True
        elif state and dd[k] < air_off:
            state = False
        air[k] = state
    # require >= 3 frames and <= 0.9 s flights
    air = _runs_filter(air, 3, int(0.9 * fps))

    # 4. Kalman + RTS on T, independently per continuous segment (cuts / long gaps split it)
    z = np.where(air[:, None], free_T, meas_T)
    Rm = np.where(air[:, None, None], free_C, meas_C)
    reset = data["reset"] if "reset" in data else np.zeros(Tn, bool)
    segs = segments(ok, reset, max_gap)
    valid = np.zeros(Tn, bool)
    T_s = np.zeros((Tn, 3))
    uv_sole = uv_all[:, MP_SOLE].copy()
    for a, b in segs:
        T_s[a:b] = _kalman_rts(z[a:b], Rm[a:b], ok[a:b], dt, accel_std, accel_std_depth, cam.R[2])
        valid[a:b] = True
        P_rel[a:b], vis[a:b] = _fill_gaps(P_rel[a:b], vis[a:b], ok[a:b])
        uv_sole[a:b] = _fill_gaps(uv_sole[a:b], np.zeros(uv_sole[a:b].shape[:2]), ok[a:b])[0]
        idx = np.flatnonzero(ok[a:b]) + a
        scale_k[a:b] = np.interp(np.arange(a, b), idx, scale_k[idx])
    P_rel = _smooth(P_rel, valid, smooth_sigma)
    scale_k = _smooth(scale_k[:, None, None], valid, 4.0)[:, 0, 0]
    P = scale_k[:, None, None] * P_rel + T_s[:, None, :]

    # 5. footplants: relative sole height (common-mode depth/height errors cancel) + image speed
    uv_sole = _smooth(uv_sole, valid, 1.0)
    dist = np.linalg.norm(P[:, 0] - cam.center, axis=1)
    contact = footplants(P[:, MP_SOLE], uv_sole, dist, valid & ~air, fps, min_len=plant_min_len)
    sole_h = SOLE_H.copy()
    info = dict(scale=scale, air_frames=int(air.sum()), depth_diff_median=float(base), segments=len(segs),
                valid_frac=float(valid.mean()), contact_frac=contact[valid].mean(0).round(3).tolist(),
                arm_flip_frac=None if flips is None else {s: dict(zip(["upper", "fore"], np.mean(
                    [[FLIP_STATES[x][0], FLIP_STATES[x][1]] for x in flips[s][ok]], 0).round(3).tolist()))
                    for s in flips},
                sole_h=sole_h.round(3).tolist())
    uv_w = np.where(ok[:, None], np.clip(data["vis"], 0, 1) * inside, 0.0)
    uv_s = uv_all.copy()
    from scipy.ndimage import gaussian_filter1d
    for a, b in segs:
        w = np.maximum(uv_w[a:b], 1e-3)[..., None]
        uv_s[a:b] = (gaussian_filter1d(uv_all[a:b] * w, 1.0, axis=0, mode="nearest")
                     / gaussian_filter1d(w, 1.0, axis=0, mode="nearest"))
    return Reconstruction(fps, valid, P, vis, T_s, air, contact, scale, sole_h, info, uv_s, uv_w)


def segments(ok, reset, max_gap=8, min_len=15):
    segs, a, last = [], None, None
    for k in range(len(ok)):
        if reset[k] and a is not None:
            segs.append((a, last + 1)); a = None
        if not ok[k]:
            if a is not None and k - last > max_gap:
                segs.append((a, last + 1)); a = None
            continue
        if a is None:
            a = k
        last = k
    if a is not None:
        segs.append((a, last + 1))
    return [(a, b) for a, b in segs if b - a >= min_len]


def _fill_gaps(P, vis, ok):
    idx = np.flatnonzero(ok)
    P = P.copy(); vis = vis.copy()
    miss = np.flatnonzero(~ok)
    if len(miss):
        for j in range(P.shape[1]):
            for c in range(P.shape[2]):
                P[miss, j, c] = np.interp(miss, idx, P[idx, j, c])
            vis[miss, j] = np.interp(miss, idx, vis[idx, j]) * 0.5
    return P, vis


def _smooth(P, valid, sigma):
    """Light zero-phase Gaussian smoothing of the body-relative pose inside valid runs."""
    if sigma <= 0:
        return P
    from scipy.ndimage import gaussian_filter1d
    out = P.copy()
    for a, b in segments(valid, np.zeros(len(valid), bool), 0, 2):
        out[a:b] = gaussian_filter1d(P[a:b], sigma, axis=0, mode="nearest")
    return out


# arm chains (MediaPipe ids): shoulder, elbow, wrist, hand points that follow the wrist
ARMS = {"l": (11, 13, 15, (17, 19, 21)), "r": (12, 14, 16, (18, 20, 22))}
# candidate states: (flip upper arm, flip forearm)
FLIP_STATES = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _reflect(v, n):
    return v - 2.0 * (v @ n)[..., None] * n


def resolve_arm_flips(P_rel, uv, vis, ok, cam, fps, w_mp=1.5, w_occ=3.0, w_anat=0.0, w_time=1.0,
                      v_max=6.0):
    """Monocular forward/backward ambiguity of the arm links (Sminchisescu & Triggs 2003): every link
    has two 3D solutions with (nearly, telephoto lens) the same projection - its depth component along
    the viewing ray reflected. MediaPipe often picks the wrong one when the player faces away.
    Per arm and frame choose one of 4 states (upper / forearm reflected or not) by Viterbi with
      - occlusion ordering (cf. RePOSE 2024): a joint whose image lies inside the torso is behind
        the torso if MediaPipe calls it invisible, in front of it (camera side) if visible;
      - MediaPipe's own depth, trusted less when the torso faces away from the camera;
      - a weak anatomical prior (hands rarely far behind the torso's frontal plane);
      - temporal continuity of elbow and wrist.
    Returns corrected P_rel and the per-frame chosen states."""
    P = P_rel.copy()
    Tn = len(P)
    n_ray = cam.R[2]
    states = {s: np.zeros(Tn, int) for s in ARMS}
    sig = v_max / fps
    for side, (S, E, W, hand) in ARMS.items():
        cand = np.zeros((Tn, 4, 2 + len(hand), 3))              # elbow, wrist, hand points
        unary = np.zeros((Tn, 4))
        for k in np.flatnonzero(ok):
            p = P[k]
            up = p[[11, 12]].mean(0) - p[[23, 24]].mean(0)
            right = p[12] - p[11]
            fwd = np.cross(up, right); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
            facing_cam = float(-fwd @ n_ray)                        # +1 facing the camera, -1 back view
            torso_c = p[[11, 12, 23, 24]].mean(0)
            quad = uv[k, [11, 12, 24, 23]].astype(np.float32)
            cq = quad.mean(0)
            quad = (cq + 1.15 * (quad - cq)).astype(np.float32)
            vu, vf = p[E] - p[S], p[W] - p[E]
            vh = p[list(hand)] - p[W]
            for si, (fu, ff) in enumerate(FLIP_STATES):
                u = _reflect(vu, n_ray) if fu else vu
                f_ = _reflect(vf, n_ray) if ff else vf
                h = _reflect(vh, n_ray) if ff else vh
                e = p[S] + u; w = e + f_
                cand[k, si, 0], cand[k, si, 1], cand[k, si, 2:] = e, w, w + h
                c = 0.0
                # MediaPipe preference: proportional to how much depth the flip actually changes
                trust = w_mp * (0.6 + 0.4 * max(0.0, facing_cam))
                c += trust * (fu * min(abs(vu @ n_ray) / 0.1, 1.0) + ff * min(abs(vf @ n_ray) / 0.1, 1.0))
                # occlusion ordering against the torso
                for j, x in ((E, e), (W, w)):
                    if cv2.pointPolygonTest(quad, (float(uv[k, j, 0]), float(uv[k, j, 1])), False) < 0:
                        continue
                    rel = (x - torso_c) @ n_ray                     # > 0: further from the camera than the torso
                    if vis[k, j] < 0.5 and rel < 0.05:
                        c += w_occ * min((0.05 - rel) / 0.1, 1.0)
                    elif vis[k, j] > 0.8 and rel > -0.05:
                        c += w_occ * 0.5 * min((rel + 0.05) / 0.1, 1.0)
                # anatomy: wrist well behind the frontal plane of the torso is rare
                back = -((w - torso_c) @ fwd) - 0.15
                c += w_anat * max(0.0, back) / 0.1
                unary[k, si] = c
        # Viterbi over each run of valid frames
        k = 0
        while k < Tn:
            if not ok[k]:
                k += 1
                continue
            a = k
            while k < Tn and ok[k]:
                k += 1
            b = k
            L = b - a
            cost = unary[a].copy()
            back_ptr = np.zeros((L, 4), int)
            for t in range(1, L):
                d = cand[a + t][None, :, :2] - cand[a + t - 1][:, None, :2]      # (prev, cur, 2, 3)
                trans = w_time * (np.linalg.norm(d, axis=3) ** 2).sum(2) / sig ** 2
                tot = cost[:, None] + trans
                back_ptr[t] = np.argmin(tot, 0)
                cost = tot[back_ptr[t], np.arange(4)] + unary[a + t]
            s = int(np.argmin(cost))
            for t in range(L - 1, -1, -1):
                states[side][a + t] = s
                P[a + t, E], P[a + t, W] = cand[a + t, s, 0], cand[a + t, s, 1]
                P[a + t, list(hand)] = cand[a + t, s, 2:]
                s = back_ptr[t, s] if t > 0 else s
    return P, states


def _depth_guess(cam, uv):
    g = cam.ground_point(uv[MP_SOLE].mean(0, keepdims=True), 0.03)[0]
    return float(np.clip(np.linalg.norm(g - cam.center), 4.0, 40.0))


def _runs_filter(mask, min_len, max_len):
    out = np.zeros_like(mask)
    k = 0
    while k < len(mask):
        if mask[k]:
            e = k
            while e < len(mask) and mask[e]:
                e += 1
            if min_len <= e - k <= max_len:
                out[k:e] = True
            k = e
        else:
            k += 1
    return out


def _kalman_rts(z, R, ok, dt, accel_std, accel_std_depth=None, n_depth=None):
    """Constant-velocity Kalman filter + RTS smoother. z (T,3), R (T,3,3).
    accel_std_depth: separate (lower) process noise along the horizontal camera depth axis n_depth,
    where monocular measurements are weakest."""
    Tn = len(z)
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    Qa = np.eye(3) * accel_std ** 2
    if accel_std_depth is not None and n_depth is not None:
        n = np.r_[n_depth[:2], 0.0]; n /= np.linalg.norm(n)
        Qa += (accel_std_depth ** 2 - accel_std ** 2) * np.outer(n, n)
    Q = np.block([[Qa * dt ** 4 / 4, Qa * dt ** 3 / 2], [Qa * dt ** 3 / 2, Qa * dt ** 2]])
    Hm = np.zeros((3, 6)); Hm[:, :3] = np.eye(3)
    xs = np.zeros((Tn, 6)); Ps = np.zeros((Tn, 6, 6))
    xp = np.zeros((Tn, 6)); Pp = np.zeros((Tn, 6, 6))
    first = int(np.flatnonzero(ok)[0])
    x = np.r_[z[first], 0, 0, 0]; Pc = np.diag([1, 1, 1, 4, 4, 4.0])
    for k in range(Tn):
        if k > 0:
            x = F @ x; Pc = F @ Pc @ F.T + Q
        xp[k], Pp[k] = x, Pc
        if ok[k] and np.all(np.isfinite(z[k])):
            S = Hm @ Pc @ Hm.T + R[k]
            Kg = Pc @ Hm.T @ np.linalg.inv(S)
            innov = z[k] - x[:3]
            # gate gross outliers (wrong person / bad frame)
            if innov @ np.linalg.solve(S, innov) < 60.0:
                x = x + Kg @ innov
                Pc = (np.eye(6) - Kg @ Hm) @ Pc
        xs[k], Ps[k] = x, Pc
    for k in range(Tn - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, :3]


def footplants(sole, uv, dist, valid, fps, h_on=0.05, h_off=0.09, v_on=1.3, v_off=2.6, min_len=3):
    """Footplant flags (T,4) for heel L/R, toe L/R.
    Height: each sole landmark relative to the lowest one (minus its own planted height), so the
    common-mode root height error cancels. Speed: image speed in px/frame normalised to 12 m."""
    Tn = len(sole)
    v = np.zeros((Tn, 4))
    v[1:] = np.linalg.norm(np.diff(uv, axis=0), axis=2) * (dist[1:, None] / 12.0)
    v[0] = v[1]
    v = np.minimum(v, np.r_[v[1:], v[-1:]])              # min of backward / forward speed
    hz = sole[:, :, 2] - SOLE_H
    h = hz - hz.min(1, keepdims=True)
    c = np.zeros((Tn, 4), bool)
    for i in range(4):
        st = False
        for k in range(Tn):
            if not valid[k]:
                st = False
            elif not st and h[k, i] < h_on and v[k, i] < v_on:
                st = True
            elif st and (h[k, i] > h_off or v[k, i] > v_off):
                st = False
            c[k, i] = st
        c[:, i] = _runs_filter(c[:, i], min_len, 10 ** 9)
    return c

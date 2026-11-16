"""Camera calibration from badminton court lines (static broadcast camera behind a baseline).

1. median of frames sampled over the video -> clean court without players
2. white-line mask on the green court -> Hough segments -> clustered lines -> refit (total least squares)
3. lines are labelled by geometry: 6 "horizontal" lines (baselines, doubles long service lines, short
   service lines) sorted by image row and 4 side lines sorted by their position at the bottom row
4. their intersections have known court coordinates (BWF) -> single-view calibration (focal length,
   rotation, translation; principal point at the image centre, no distortion)

Court frame (same as the game): net plane y = 0, the camera-side half is y < 0, x to the right when
looking from the camera-side baseline towards the net, z up.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .court import (DOUBLES_LONG_SERVICE, LENGTH_HALF, NET_TOP_POST, SHORT_SERVICE, WIDTH_DOUBLES_HALF,
                    WIDTH_SINGLES_HALF)

# camera-side half first (larger image row = closer to the camera)
ROWS_Y = [-LENGTH_HALF, -DOUBLES_LONG_SERVICE, -SHORT_SERVICE, SHORT_SERVICE, DOUBLES_LONG_SERVICE, LENGTH_HALF]
SIDES_X = [-WIDTH_DOUBLES_HALF, -WIDTH_SINGLES_HALF, WIDTH_SINGLES_HALF, WIDTH_DOUBLES_HALF]


@dataclass
class Camera:
    width: int
    height: int
    K: list            # 3x3
    rvec: list         # world (court) -> camera
    tvec: list
    reproj_px: float
    net_check_px: float

    @property
    def Kmat(self):
        return np.array(self.K, float)

    @property
    def R(self):
        return cv2.Rodrigues(np.array(self.rvec, float))[0]

    @property
    def t(self):
        return np.array(self.tvec, float).reshape(3)

    @property
    def center(self):
        """Camera position in court coordinates."""
        return -self.R.T @ self.t

    @property
    def fovy_deg(self):
        return float(np.degrees(2 * np.arctan(self.height / 2 / self.Kmat[1, 1])))

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xc = X @ self.R.T + self.t
        uv = Xc[:, :2] / Xc[:, 2:3]
        return uv * [self.Kmat[0, 0], self.Kmat[1, 1]] + self.Kmat[:2, 2]

    def ray(self, uv):
        """Unit rays (court frame) through pixels uv (N,2)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        K = self.Kmat
        d = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))])
        d = d @ self.R                         # R^T d for row vectors
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ground_point(self, uv, z=0.0):
        """Intersection of the pixel rays with the horizontal plane at height z."""
        d = self.ray(uv)
        c = self.center
        s = (z - c[2]) / d[:, 2]
        return c + s[:, None] * d

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def median_background(video, n=120, skip_outliers=True, frame_range=None, ref_frame=None):
    """Median over n frames. ref_frame: a frame index known to be the main camera; only frames
    close to it are used (matches with several camera zooms / replays)."""
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a, b = frame_range if frame_range is not None else (0, N - 1)
    frames = []
    for f in np.linspace(a, min(b, N - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    frames = np.array(frames)
    if ref_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref_frame))
        ref = cap.read()[1].astype(np.int16)
        d = np.array([np.mean(np.abs(f.astype(np.int16) - ref)) for f in frames])
        frames = frames[d < max(np.percentile(d, 30), 1.0) * 1.5]
    med = np.median(frames, 0).astype(np.uint8)
    if skip_outliers:                          # drop replays / other camera angles and recompute
        d = np.array([np.mean(np.abs(f.astype(np.int16) - med)) for f in frames])
        keep = d < np.median(d) * 2.0
        med = np.median(frames[keep], 0).astype(np.uint8)
    return med


def line_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 170)) | ((tophat > 25) & (hsv[..., 1] < 110))
    green = cv2.dilate(cv2.inRange(hsv, (40, 60, 60), (90, 255, 255)), np.ones((15, 15), np.uint8)) > 0
    return (white & green).astype(np.uint8) * 255


def _refit(mask, n, rho, band=4.0):
    """Total least squares line through mask pixels within `band` px of n.x = rho."""
    ys, xs = np.nonzero(mask)
    P = np.column_stack([xs, ys]).astype(float)
    for b in (band, band / 2):
        sel = np.abs(P @ n - rho) < b
        Q = P[sel]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c)
        n = vt[1]
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n = -n
        rho = float(n @ c)
    return n, rho, Q


def detect_lines(mask, min_len=100):
    s = mask.shape[0] / 544.0                  # thresholds were tuned at 544p
    min_len *= s
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, int(round(60 * s)), minLineLength=int(round(40 * s)),
                           maxLineGap=int(round(8 * max(s, 1.0))))
    segs = segs.reshape(-1, 4).astype(float)
    cl = []
    for x1, y1, x2, y2 in segs:
        n = np.array([-(y2 - y1), x2 - x1])
        n /= np.linalg.norm(n)
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n = -n
        rho = n @ [x1, y1]
        ln = np.hypot(x2 - x1, y2 - y1)
        for c in cl:
            if c["n"] @ n > np.cos(np.radians(2.5)) and abs(c["rho"] - rho) < 6:
                c["len"] += ln
                break
        else:
            cl.append(dict(n=n, rho=rho, len=ln))
    lines = []
    for c in sorted(cl, key=lambda c: -c["len"]):
        if c["len"] < min_len:
            continue
        n, rho, Q = _refit(mask, c["n"], c["rho"])
        if any(l["n"] @ n > np.cos(np.radians(1.5)) and abs(l["rho"] - rho) < 3 for l in lines):
            continue
        lines.append(dict(n=n, rho=rho, len=c["len"], ext=(Q.min(0), Q.max(0))))
    return lines


def _intersect(a, b):
    A = np.array([a["n"], b["n"]])
    return np.linalg.solve(A, [a["rho"], b["rho"]])


def label_lines(lines, width, height):
    """Split into the 6 transverse court lines (by row) and the 4 side lines (by x at the bottom)."""
    horiz = [l for l in lines if abs(l["n"][1]) > np.cos(np.radians(12))]
    side = [l for l in lines if abs(l["n"][1]) < np.cos(np.radians(20)) and abs(l["n"][1]) > np.sin(np.radians(15))]
    horiz.sort(key=lambda l: -l["rho"] / l["n"][1])                 # bottom (camera side) first
    if len(horiz) < 6:
        raise RuntimeError(f"found {len(horiz)} transverse lines, need 6")
    horiz = horiz[:6] if len(horiz) == 6 else _best_rows(horiz)
    yb = height - 1.0

    def x_at(l, y):
        return (l["rho"] - l["n"][1] * y) / l["n"][0]
    side.sort(key=lambda l: x_at(l, yb))
    if len(side) < 4:
        raise RuntimeError(f"found {len(side)} side lines, need 4")
    if len(side) > 4:                           # keep the two outermost on each side of the centre
        left = [l for l in side if x_at(l, yb) < width / 2][:2]
        right = [l for l in side if x_at(l, yb) >= width / 2][-2:]
        side = left + right
    return horiz, side


def _best_rows(horiz):
    """More than 6 transverse lines (e.g. ad boards): keep the 6 longest, in row order."""
    keep = sorted(horiz, key=lambda l: -l["len"])[:6]
    return sorted(keep, key=lambda l: -l["rho"] / l["n"][1])


def _solve(horiz, side, w, h):
    obj, imgp = [], []
    for li, yw in zip(horiz, ROWS_Y):
        for ls, xw in zip(side, SIDES_X):
            p = _intersect(li, ls)
            if -5 <= p[0] <= w + 5 and -5 <= p[1] <= h + 5:
                obj.append([xw, yw, 0.0]); imgp.append(p)
    obj = np.array(obj, np.float32); imgp = np.array(imgp, np.float32)
    if len(obj) < 8:
        return np.inf, None, obj, imgp
    f0 = 1.2 * w
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, dist, rv, tv = cv2.calibrateCamera([obj], [imgp], (w, h), K0, np.zeros(5), flags=flags)
    cam = Camera(w, h, K.tolist(), rv[0].ravel().tolist(), tv[0].ravel().tolist(), float(rms), float("nan"))
    return float(rms), cam, obj, imgp


def calibrate(img, net_post_px=None):
    """Returns (Camera, debug dict). net_post_px: optional [(u,v) left post top, (u,v) right post top]."""
    h, w = img.shape[:2]
    mask = line_mask(img)
    lines = detect_lines(mask)
    horiz, side = label_lines(lines, w, h)
    rms, cam, obj, imgp = _solve(horiz, side, w, h)
    if rms > 3.0:                               # floor text / net tape / ad boards among the candidates
        cands = sorted([l for l in lines if abs(l["n"][1]) > np.cos(np.radians(12))],
                       key=lambda l: -l["rho"] / l["n"][1])
        for sub in itertools.combinations(cands, 6):
            r = _solve(list(sub), side, w, h)
            if r[0] < rms:
                rms, cam, obj, imgp = r
                horiz = list(sub)
    if net_post_px is not None:
        pr = cam.project([[-WIDTH_DOUBLES_HALF, 0, NET_TOP_POST], [WIDTH_DOUBLES_HALF, 0, NET_TOP_POST]])
        cam.net_check_px = float(np.linalg.norm(pr - np.asarray(net_post_px, float), axis=1).mean())
    return cam, dict(mask=mask, lines=lines, horiz=horiz, side=side, obj=obj, imgp=imgp)


def draw_court(img, cam: Camera, color=(0, 0, 255), thick=1):
    """Overlay the BWF court model projected with the calibrated camera."""
    out = img.copy()
    segs = []
    for y in ROWS_Y:
        segs.append(([-WIDTH_DOUBLES_HALF, y, 0], [WIDTH_DOUBLES_HALF, y, 0]))
    for x in SIDES_X:
        segs.append(([x, -LENGTH_HALF, 0], [x, LENGTH_HALF, 0]))
    segs += [([0, -LENGTH_HALF, 0], [0, -SHORT_SERVICE, 0]), ([0, SHORT_SERVICE, 0], [0, LENGTH_HALF, 0])]
    segs += [([-WIDTH_DOUBLES_HALF, 0, NET_TOP_POST], [WIDTH_DOUBLES_HALF, 0, NET_TOP_POST])]
    for sx in (-1, 1):
        segs.append(([sx * WIDTH_DOUBLES_HALF, 0, 0], [sx * WIDTH_DOUBLES_HALF, 0, NET_TOP_POST]))
    for a, b in segs:
        pa, pb = cam.project([a, b])
        cv2.line(out, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, thick, cv2.LINE_AA)
    return out

"""Racket detection in broadcast video (fixed main camera), without a learned detector.

Foreground = |frame - median background| (Lab), minus the player's own silhouette (MediaPipe
skeleton + torso/shorts polygons) and court-line pixels. At 960x544 the racket is a thin segment
from the hand (the head is mostly seen edge-on), so it is searched on a polar grid (angle x
projected wrist->tip length) from both wrists, scored as a thin foreground line with background
on both sides; a Viterbi pass picks one temporally consistent hypothesis (hand, angle, length)
per frame, or 'not seen'. The tip is lifted to 3D on its camera ray at the known wrist->tip
distance.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calib import draw_court, line_mask

# MediaPipe pose landmark ids
L_WRIST, R_WRIST = 15, 16
BODY_LINES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
              (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (27, 29), (28, 30),
              (0, 11), (0, 12), (15, 19), (16, 20), (15, 17), (16, 18)]
N_ANG = 72
TIP_DIST = 0.62           # wrist -> racket tip (m), character racket
LENS = np.array([0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64])   # projected wrist -> tip length (m)


def median_background(video, frames, main=None, n=60):
    cap = cv2.VideoCapture(str(video))
    idx = [f for f in frames if main is None or main[f - frames[0]]]
    idx = idx[:: max(1, len(idx) // n)]
    imgs = []
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            imgs.append(im)
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def foreground(frame, bg, thr=22.0):
    a = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(bg, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm((a - b) * [0.5, 1.0, 1.0], axis=2)
    return np.clip((d - thr * 0.5) / thr, 0, 1).astype(np.float32)


def body_mask(shape, uv, px_per_m):
    m = np.zeros(shape[:2], np.uint8)
    th = max(3, int(0.16 * px_per_m))
    for a, b in BODY_LINES:
        if a in (15, 16) or b in (15, 16):
            t = max(2, int(0.08 * px_per_m))
        else:
            t = th
        cv2.line(m, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), 255, t, cv2.LINE_AA)
    for poly in ((11, 12, 24, 23), (23, 24, 26, 25)):             # torso, shorts
        cv2.fillConvexPoly(m, np.int32(uv[list(poly)]), 255)
    head = uv[[0, 7, 8]].mean(0)
    cv2.circle(m, tuple(np.int32(head)), int(0.17 * px_per_m), 255, -1)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return m > 0


def score_frame(fg, body, wrist_uv, px_per_m, lat=5):
    """Thin-segment racket model from the hand. Scores (N_ANG, len(LENS)) and tip pixels.
    Score = mean over the unmasked samples of [on-line fg (max over +-1 px) - side fg (+-lat px)]
    along the visible length, minus fg continuing beyond the tip."""
    H, W = fg.shape
    ang = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)                     # (A,2)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    R = int(np.ceil((LENS[-1] + 0.12) * px_per_m))
    r = np.arange(R, dtype=np.float32)
    base = wrist_uv[None, None] + d[:, None] * r[None, :, None]     # (A,R,2)

    def sample(off):
        p = base + nrm[:, None] * off
        x = np.clip(np.round(p[..., 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(p[..., 1]).astype(int), 0, H - 1)
        inside = (p[..., 0] >= 0) & (p[..., 0] < W) & (p[..., 1] >= 0) & (p[..., 1] < H)
        return fg[y, x] * inside, body[y, x] | ~inside

    on = np.max([sample(o)[0] for o in (-1, 0, 1)], 0)
    masked = sample(0)[1]
    side = 0.5 * (np.max([sample(o)[0] for o in (lat, lat + 1)], 0) +
                  np.max([sample(-o)[0] for o in (lat, lat + 1)], 0))
    v = np.where(masked, 0.0, on - side)
    m = (~masked).astype(np.float32)
    cv, cm, con = np.cumsum(v, 1), np.cumsum(m, 1), np.cumsum(np.where(masked, 0.0, on), 1)
    S = np.full((N_ANG, len(LENS)), -1.0); C = np.zeros((N_ANG, len(LENS), 2))
    r0 = int(0.06 * px_per_m)
    for j, L in enumerate(LENS):
        l1 = int(L * px_per_m)
        l2 = min(R - 1, l1 + int(0.1 * px_per_m))
        n = cm[:, l1] - cm[:, r0]
        mean = (cv[:, l1] - cv[:, r0]) / np.maximum(n, 1)
        beyond = (con[:, l2] - con[:, l1]) / max(l2 - l1, 1)
        frac = n / max(l1 - r0, 1)
        S[:, j] = np.where(frac > 0.25, mean - 0.6 * beyond, -1.0)
        C[:, j] = wrist_uv + d * l1
    return S, C


def viterbi(scores, valid, w_ang=0.08, w_dist=0.15, null=0.15, w_switch=1.0, hand_cost=(0.0, 0.05)):
    """scores (T, Hh, A, D): hand x angle x projected length; returns flat state per frame,
    Hh*A*D = not seen. Hand switches (MediaPipe left/right label swaps) cost w_switch."""
    hand_cost = np.asarray(hand_cost, np.float32)
    T, Hh, A, D = scores.shape
    n = Hh * A * D + 1
    h_idx = np.r_[np.repeat(np.arange(Hh), A * D), -1]
    a_idx = np.r_[np.tile(np.repeat(np.arange(A), D), Hh), -1]
    d_idx = np.r_[np.tile(np.arange(D), Hh * A), -1]
    da = np.abs(a_idx[:, None] - a_idx[None, :]); da = np.minimum(da, A - da) * (2 * np.pi / A)
    dd = np.abs(d_idx[:, None] - d_idx[None, :]).astype(float)
    trans = w_ang * da ** 2 / 0.1 + w_dist * dd ** 2
    sw = h_idx[:, None] != h_idx[None, :]
    trans[sw] = w_switch
    null_mask = (a_idx[:, None] < 0) | (a_idx[None, :] < 0)
    trans[null_mask] = 0.25
    trans[-1, -1] = 0.0
    trans = trans.astype(np.float32)
    # hand memory through 'not seen' runs is lost with a single null state; acceptable
    cost = np.zeros(n, np.float32); back = np.zeros((T, n), np.int32)
    for k in range(T):
        u = (np.r_[-scores[k].ravel(), -null] if valid[k] else np.r_[np.full(n - 1, 9.0), 0.0]).astype(np.float32)
        u[:-1] += hand_cost[h_idx[:-1]]
        if k == 0:
            cost = u
            continue
        tot = cost[:, None] + trans
        back[k] = np.argmin(tot, 0)
        cost = tot[back[k], np.arange(n)] + u
    s = np.zeros(T, int); s[-1] = int(np.argmin(cost))
    for k in range(T - 1, 0, -1):
        s[k - 1] = back[k, s[k]]
    return s


def detect_rackets(video, cam, data, start, bg=None, log=True):
    """Racket tip per frame, searched from both MediaPipe wrists (right-handed player: a racket on
    MediaPipe's 'left' wrist means the left/right labels are swapped in that frame).
    Returns dict(uv (T,2) tip px (nan if unseen), conf (T,), hand (T,) 0 = MP right, 1 = MP left,
    -1 unseen, bg)."""
    T = len(data["ok"])
    W, H = int(data["width"]), int(data["height"])
    main = data["main"].astype(bool) & data["ok"].astype(bool)
    if bg is None:
        bg = median_background(video, list(range(start, start + T)), main)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    court_zone = cv2.dilate(draw_court(np.zeros((H, W, 3), np.uint8), cam, (255, 255, 255), 1)[..., 0],
                            np.ones((15, 15), np.uint8))
    shape = (T, 2, N_ANG, len(LENS))
    scores = np.full(shape, -1.0, np.float32); cents = np.zeros(shape + (2,), np.float32)
    for k in range(T):
        ok, fr = cap.read()
        if not ok:
            main[k:] = False
            break
        if not main[k]:
            continue
        uv = data["image"][k][:, :2] * [W, H]
        dist = np.linalg.norm(cam.ground_point(uv[[27, 28]].mean(0, keepdims=True), 0.0)[0] + [0, 0, 1.0]
                              - cam.center)
        ppm = cam.Kmat[0, 0] / max(dist, 3.0)
        mg = (LENS[-1] + 0.25) * ppm
        x0, y0 = [int(max(0, v)) for v in uv.min(0) - mg]
        x1, y1 = int(min(W, uv[:, 0].max() + mg)), int(min(H, uv[:, 1].max() + mg))
        fg = np.zeros((H, W), np.float32); lines = np.zeros((H, W), np.uint8)
        roi = fr[y0:y1, x0:x1]
        fg[y0:y1, x0:x1] = foreground(roi, bg[y0:y1, x0:x1])
        # white court-line pixels near the calibrated lines (small camera pans make them foreground)
        lines[y0:y1, x0:x1] = cv2.dilate(line_mask(roi), np.ones((3, 3), np.uint8)) & court_zone[y0:y1, x0:x1]
        mask = body_mask(fg.shape, uv, ppm) | (lines > 0)
        for h, wj in enumerate((R_WRIST, L_WRIST)):
            scores[k, h], cents[k, h] = score_frame(fg, mask, uv[wj], ppm)
        if log and k % 300 == 0:
            print(f"  racket detect {k}/{T}", flush=True)
    s = viterbi(scores, main)
    uv = np.full((T, 2), np.nan); conf = np.zeros(T); hand = np.full(T, -1)
    A, D = N_ANG, len(LENS)
    for k in range(T):
        if s[k] < 2 * A * D and main[k]:
            h, r = divmod(int(s[k]), A * D)
            a, d = divmod(r, D)
            uv[k] = cents[k, h, a, d]; conf[k] = scores[k, h, a, d]; hand[k] = h
    return dict(uv=uv, conf=conf, hand=hand, bg=bg)


def ray_sphere(cam, uv, center, radius):
    """Points on the camera ray through uv at distance `radius` from `center`: list of 2, or the
    single closest ray point when the ray misses (projected length too long for the racket)."""
    o = cam.center
    d = cam.ray(np.asarray(uv, float)[None])[0]
    w = center - o
    tc = w @ d
    h2 = radius ** 2 - (w @ w - tc ** 2)
    if h2 < 0:
        return [o + d * tc], np.sqrt(-h2)
    h = np.sqrt(h2)
    return [o + d * (tc - h), o + d * (tc + h)], 0.0


def racket_targets(cam, rec, det, tip_dist=TIP_DIST, conf_lo=0.15, conf_hi=0.45, max_miss=0.12, prior=None,
                   prior_reject_deg=70.0, prior_reject_conf=0.75, prior_fill=1.0):
    """Per-frame racket tip targets for the retargeter from detections on MediaPipe's right wrist.
    Depth ambiguity of the lifted tip (two ray-sphere intersections) is resolved by temporal
    continuity plus either the learned pose prior (badrt.racket_prior; prior = (dir (T,3) world
    unit, conf (T,))) or, without it, the side in front of the torso when the hand is below the
    shoulder (the same rule as the retargeter's racket-front term).
    With a prior, detections far from a confident prediction are rejected (false hits on legs /
    shadows) and frames without a detection get a target along the prior (weight
    prior_fill * conf; weaker fills lose against the arm terms, see the dropout test in
    docs/technical_report.md) so the racket keeps a plausible direction while unseen.
    Returns dict(tip (T,3) nan if none, uv (T,2) nan if none, w (T,) in [0,1])."""
    T = len(rec.ok)
    tip = np.full((T, 3), np.nan); uv = np.full((T, 2), np.nan); w = np.zeros(T)
    prev, prev_k = None, -10
    n_prior_rej = 0
    cos_rej = np.cos(np.radians(prior_reject_deg))
    for k in range(T):
        if not rec.ok[k] or det["hand"][k] != 0 or det["conf"][k] < conf_lo:
            continue
        P = rec.P[k]
        cands, miss = ray_sphere(cam, det["uv"][k], P[R_WRIST], tip_dist)
        if miss > max_miss:
            continue
        up = P[[11, 12]].mean(0) - P[[23, 24]].mean(0)
        fwd = np.cross(up, P[12] - P[11]); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        tcen = P[[11, 12, 23, 24]].mean(0)
        low = P[R_WRIST, 2] < P[12, 2]
        if prior is not None:
            pc = [(c - P[R_WRIST]) @ prior[0][k] / (np.linalg.norm(c - P[R_WRIST]) + 1e-9) for c in cands]
            if prior[1][k] > prior_reject_conf and max(pc) < cos_rej:
                n_prior_rej += 1
                continue

        def cost(i, c):
            e = 0.0
            if prev is not None and k - prev_k <= 3:
                e += (np.linalg.norm(c - prev) / 0.3) ** 2
            if prior is not None:
                e += 2.0 * prior[1][k] * (1.0 - pc[i])
            elif low and (c - tcen) @ fwd < -0.1:
                e += 1.0
            return e
        c = cands[min(range(len(cands)), key=lambda i: cost(i, cands[i]))]
        tip[k] = c; uv[k] = det["uv"][k]
        w[k] = float(np.clip((det["conf"][k] - conf_lo) / (conf_hi - conf_lo), 0.2, 1.0)) * (1.0 - miss / max_miss * 0.5)
        prev, prev_k = c, k
    out = _temporal_clean(rec, tip, uv, w, tip_dist, prior=prior, prior_fill=prior_fill)
    out["n_prior_reject"] = n_prior_rej
    return out


def _temporal_clean(rec, tip, uv, w, tip_dist, win=3, max_dev_deg=50.0, max_gap=6, sigma=2.0, fade=3,
                    prior=None, prior_fill=1.0):
    """Wrist->tip directions: drop detections that disagree with their temporal neighbours
    (false hits on legs / the other player), fill short gaps by interpolation (no image term there,
    lower weight) and smooth lightly, so the racket target does not switch on and off."""
    from scipy.ndimage import gaussian_filter1d
    T = len(w)
    wr = rec.P[:, R_WRIST]
    u = np.full((T, 3), np.nan)
    has = w > 0
    u[has] = (tip[has] - wr[has]) / np.linalg.norm(tip[has] - wr[has], axis=1, keepdims=True)
    keep = has.copy()
    cos_t = np.cos(np.radians(max_dev_deg))
    for k in np.flatnonzero(has):
        nb = [j for j in range(max(0, k - win), min(T, k + win + 1)) if j != k and has[j]]
        if len(nb) < 2:
            continue
        m = np.median(u[nb], axis=0); m /= np.linalg.norm(m) + 1e-9
        keep[k] = u[k] @ m > cos_t
    w = np.where(keep, w, 0.0); uv = np.where(keep[:, None], uv, np.nan)
    # fill gaps <= max_gap between kept frames inside valid runs
    ku = np.flatnonzero(keep)
    wf = w.copy(); uf = np.where(keep[:, None], u, np.nan)
    for a, b in zip(ku[:-1], ku[1:]):
        if 1 < b - a <= max_gap + 1 and rec.ok[a:b + 1].all():
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                v = (1 - t) * u[a] + t * u[b]
                if np.linalg.norm(v) < 0.3:
                    break
                uf[k] = v / np.linalg.norm(v)
                wf[k] = 0.5 * min(w[a], w[b])
    # fade in / out: hold the end directions of each run for a few frames with decaying weight
    on = wf > 0
    base_on = on.copy()
    for k in np.flatnonzero(base_on):
        for step in (-1, 1):
            j = k + step
            if 0 <= j < T and base_on[j]:
                continue
            for i in range(1, fade + 1):
                j = k + step * i
                if not (0 <= j < T) or base_on[j] or not rec.ok[j]:
                    break
                wn = wf[k] * 0.5 ** i
                if wn > wf[j]:
                    wf[j] = wn; uf[j] = uf[k]
    fill = np.zeros(T, bool)
    if prior is not None and prior_fill > 0:
        fill = (wf == 0) & rec.ok.astype(bool)
        wf[fill] = prior_fill * prior[1][fill]; uf[fill] = prior[0][fill]
    # smoothing of the direction (normalised convolution)
    on = wf > 0
    num = gaussian_filter1d(np.where(on[:, None], uf * wf[:, None], 0.0), sigma, axis=0, mode="constant")
    den = gaussian_filter1d(wf, sigma, mode="constant")
    us = num / np.maximum(den, 1e-6)[:, None]
    us /= np.linalg.norm(us, axis=1, keepdims=True) + 1e-9
    tip_o = np.where(on[:, None], wr + tip_dist * us, np.nan)
    return dict(tip=tip_o, uv=uv, w=wf, prior_only=fill, n_detect=int(has.sum()), n_reject=int((has & ~keep).sum()))

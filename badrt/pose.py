"""MediaPipe PoseLandmarker (Tasks API) wrapper with simple single-person tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "assets" / "models"


@dataclass
class PoseFrame:
    world: np.ndarray        # (33,3) metres, hip-centred, camera-aligned axes (x right, y down, z away)
    image: np.ndarray        # (33,3) normalised image coords (x, y in [0,1], z relative depth)
    vis: np.ndarray          # (33,) visibility * presence
    width: int
    height: int
    t: float                 # seconds
    latency_ms: float = 0.0


class PoseEstimator:
    def __init__(self, model="full", num_poses=2, min_conf=0.5, video=True):
        path = MODELS / f"pose_landmarker_{model}.task"
        self.video = video
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ms = -1
        self._prev_center = None

    def close(self):
        self.lm.close()

    def detect_all(self, frame_bgr, t_s) -> list[PoseFrame]:
        """Every detected person (no tracking choice)."""
        h, w = frame_bgr.shape[:2]
        ts = int(round(t_s * 1000))
        if ts <= self._last_ms:
            ts = self._last_ms + 1
        self._last_ms = ts
        t0 = time.perf_counter()
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ts) if self.video else self.lm.detect(img)
        lat = (time.perf_counter() - t0) * 1e3
        out = []
        for lm2d, lm3d in zip(res.pose_landmarks, res.pose_world_landmarks):
            im = np.array([[p.x, p.y, p.z] for p in lm2d])
            wd = np.array([[p.x, p.y, p.z] for p in lm3d])
            vis = np.array([(p.visibility or 0.0) * (p.presence or 1.0) for p in lm2d])
            out.append(PoseFrame(wd, im, vis, w, h, t_s, lat))
        return out

    def __call__(self, frame_bgr, t_s) -> PoseFrame | None:
        cands = self.detect_all(frame_bgr, t_s)
        if not cands:
            return None
        pf = cands[self._pick([(c.image, c.world, c.vis) for c in cands])]
        self._prev_center = pf.image[[23, 24], :2].mean(0)
        return pf

    def _pick(self, cands):
        if len(cands) == 1:
            return 0
        if self._prev_center is None:
            size = [np.ptp(c[0][:, 1]) for c in cands]
            return int(np.argmax(size))
        dist = [np.linalg.norm(c[0][[23, 24], :2].mean(0) - self._prev_center) for c in cands]
        return int(np.argmin(dist))


def draw_pose(img, pf: PoseFrame, color=(0, 255, 0), mirror=False):
    from .character import BONE_DRAW, KP, MP_IDS
    h, w = img.shape[:2]
    pts = pf.image[MP_IDS, :2] * [w, h]
    if mirror:
        pts[:, 0] = w - pts[:, 0]
    vis = pf.vis[MP_IDS]
    for a, b in BONE_DRAW:
        ia, ib = KP[a], KP[b]
        if min(vis[ia], vis[ib]) > 0.3:
            cv2.line(img, tuple(int(v) for v in pts[ia]), tuple(int(v) for v in pts[ib]), color, 2, cv2.LINE_AA)
    for p, v in zip(pts, vis):
        if v > 0.3:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return img

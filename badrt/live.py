"""Camera / video input: pose estimation and retargeting live in a worker thread, and the offline-retargeted
"demo" track re-timed to the rally. The game itself is played with the keyboard (app.py) and does not import
this module; it is kept for the retargeting side of the project (scripts/bench_realtime.py, and
game.Match(player="human") which takes the poses these classes produce)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .character import MP_IDS, mirror_qpos
from .config import RetargetConfig, SourceConfig
from .recorded import Director, PlayerTrack

ROOT = Path(__file__).resolve().parents[1]

# offline player tracks: racket hand -> (file, start [s], duration [s])
DEMO_TRACKS = {
    "right": ("assets/tracks/N9jukZlmkns.npz", 840.0, 480.0),
    "left": ("assets/tracks/N9jukZlmkns_left.npz", 0.0, 0.0),
}


# ---------------------------------------------------------------------------- input sources
class PoseWorker(threading.Thread):
    """Capture + MediaPipe in a background thread (both release the GIL), so pose estimation overlaps
    with physics and rendering. Only the newest result is kept. Video files are paced to their own
    frame rate and frames are dropped when behind."""

    def __init__(self, cap, est, video_fps=None):
        super().__init__(daemon=True)
        self.cap, self.est, self.video_fps = cap, est, video_fps
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.latest = None                     # (frame, PoseFrame | None, t)
        self.seq = 0
        self.done = False
        self.stop = False

    def run(self):
        k = 0
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.video_fps:
                t = k / self.video_fps
                k += 1
                lag = (time.perf_counter() - self.t0) - t
                if lag > 1.5 / self.video_fps:
                    continue                   # behind: drop this frame
                if lag < 0:
                    time.sleep(-lag)
            else:
                t = time.perf_counter() - self.t0
            pf = self.est(frame, t)
            with self.lock:
                self.latest = (frame, pf, t)
                self.seq += 1
        self.done = True
        self.cap.release()

    def get(self):
        with self.lock:
            return self.seq, self.latest


class LiveInput:
    """Camera (index) or video file (path) -> pose -> retargeted qpos of the handed character."""
    kind = "camera"

    def __init__(self, source, handed="right", model="lite", court_gain=2.0, hfov=60.0, start=0.0):
        from .pose import PoseEstimator
        from .retarget import BodyRetargeter
        from .source import SourceBuilder
        is_cam = isinstance(source, int)
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if is_cam else cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {'camera' if is_cam else 'video'} {source}")
        if is_cam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        self.handed = handed
        self.rt = BodyRetargeter(RetargetConfig())
        scfg = SourceConfig(hfov_deg=hfov, court_gain=(court_gain, court_gain), handedness=handed)
        self.src = SourceBuilder(self.rt.neutral_kp, scfg, origin=(0.0, -3.6))
        self.worker = PoseWorker(cap, PoseEstimator(model, num_poses=1), None if is_cam else (cap.get(cv2.CAP_PROP_FPS) or 30.0))
        self.worker.start()
        self.seen, self.t_prev = 0, None
        self.frame, self.kp2d, self.ok = None, None, False
        self.lat = [0.0, 0.0]
        self.label = "摄像头" if is_cam else "视频输入"

    def qpos(self, t):
        seq, latest = self.worker.get()
        if latest is None or seq == self.seen:
            return None
        self.seen = seq
        self.frame, pf, tp = latest
        self.ok = pf is not None
        if pf is None:
            self.kp2d = None
            return None
        self.kp2d = pf.image[MP_IDS, :2]
        S, vis, info = self.src(pf)
        if S is None:
            self.ok = False
            return None
        dt = 1 / 30 if self.t_prev is None else tp - self.t_prev
        self.t_prev = tp
        out = self.rt.step(S, vis, float(np.clip(dt, 1e-3, 0.2)), info["contact"])
        self.lat = [pf.latency_ms, out["latency_ms"]]
        return mirror_qpos(out["qpos"]) if self.handed == "left" else out["qpos"]

    def inset(self, t):
        if self.frame is None:
            return None
        return dict(frame=self.frame, kp2d=self.kp2d, label=self.label, ok=self.ok, mirror=True)

    @property
    def finished(self):
        return self.worker.done

    def close(self):
        self.worker.stop = True
        self.worker.join(timeout=1.0)


class VideoFrames:
    """Frames of a video by index for the inset (small, optionally mirrored). Decoding and seeking run
    in a thread unless sync=True (deterministic output for recordings)."""

    def __init__(self, path, flip=False, width=400, sync=False):
        self.cap = cv2.VideoCapture(str(path))
        self.flip, self.width, self.sync = flip, width, sync
        self.pos, self.last, self.want = -1, None, None
        self.stop = False
        if not sync:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def _read(self, index):
        if index == self.pos:
            return self.last
        if index < self.pos or index > self.pos + 15:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index - 1
        while self.pos < index - 1:
            self.cap.grab()
            self.pos += 1
        ok, f = self.cap.read()
        self.pos = index
        if ok:
            f = cv2.resize(f, (self.width, int(round(f.shape[0] * self.width / f.shape[1]))), interpolation=cv2.INTER_AREA)
            self.last = cv2.flip(f, 1) if self.flip else f
        return self.last

    def _loop(self):
        while not self.stop:
            w = self.want
            if w is None or w == self.pos:
                time.sleep(0.003)
            else:
                self._read(w)

    def get(self, index):
        if self.sync:
            return self._read(index)
        self.want = index
        return self.last

    def close(self):
        self.stop = True
        if not self.sync:
            self.th.join(timeout=1.0)
        self.cap.release()


class DemoInput:
    """An offline-retargeted front-facing video, re-timed to the rally by the director."""
    kind = "demo"

    def __init__(self, track: PlayerTrack, sync=False):
        self.track = track
        self.director = Director(track)
        self.handed = track.handed
        video = Path(track.video)
        if not video.is_absolute():
            video = ROOT / video
        self.frames = VideoFrames(video, flip=track.flip, sync=sync) if video.exists() else None
        self.lat = [float(np.median(track.latency[track.ok, 0])), float(np.median(track.latency[track.ok, 1]))]
        self.finished = False

    def qpos(self, t):
        return self.director.qpos(t)

    def inset(self, t):
        if self.frames is None:
            return None
        k = self.director.frame_index(t)
        frame = self.frames.get(k)
        if frame is None:
            return None
        i = k - self.track.frame0
        return dict(frame=frame, kp2d=self.track.kp2d[i], label="演示视频 · 摄像头替身", ok=bool(self.track.ok[i]), mirror=True)

    def close(self):
        if self.frames is not None:
            self.frames.close()

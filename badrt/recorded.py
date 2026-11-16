"""A front-facing video standing in for the live camera: offline player track + stroke look-ahead.

The game treats the recorded person exactly like a camera user (same pose -> source -> retarget
chain, same hit physics). The only difference is that a recording cannot react to the shuttle,
so the match "director" (game.Director) reads the upcoming strokes from here and has the
opponent deliver the shuttle to where and when the recorded player swings.

Handedness: the retargeter works on the right-handed character. For a left-hander the source is
mirrored (SourceConfig.handedness), retargeted, and the resulting pose mirrored back onto the
left-handed character (character.mirror_qpos), so the avatar moves like the person does.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .character import MP_IDS, CharacterIndex, blend_qpos, build_single_character_model, mirror_qpos
from .config import RetargetConfig, SourceConfig
from .shuttle import hitting_speed


def build_track(video, out, model="full", handedness="right", court_gain=2.0, hfov=60.0, start=0.0,
                duration=0.0, flip=False, origin=(0.0, -3.6), smooth_sigma=1.0, log=print):
    """Pose estimation + retargeting of a whole video; saves an npz read by PlayerTrack.
    flip mirrors the frames first (turns a right-handed person into a left-handed one, for tests)."""
    from .pose import PoseEstimator
    from .retarget import BodyRetargeter, smooth_qpos
    from .source import SourceBuilder

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(start * fps))
    f1 = n_total if duration <= 0 else min(n_total, f0 + int(round(duration * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    est = PoseEstimator(model, num_poses=1)
    rt = BodyRetargeter(RetargetConfig())
    scfg = SourceConfig(hfov_deg=hfov, court_gain=(court_gain, court_gain), handedness=handedness)
    src = SourceBuilder(rt.neutral_kp, scfg, origin=origin)
    n = f1 - f0
    Q = np.zeros((n, rt.m.nq))
    ok = np.zeros(n, bool)
    kp2d = np.full((n, len(MP_IDS), 2), np.nan, np.float32)
    lat = np.zeros((n, 2), np.float32)
    q_last, t_prev = None, None
    t_wall = time.perf_counter()
    k = 0
    for k in range(n):
        got, frame = cap.read()
        if not got:
            n = k
            break
        if flip:
            frame = cv2.flip(frame, 1)
        t = k / fps
        pf = est(frame, t)
        S = None
        if pf is not None:
            kp2d[k] = pf.image[MP_IDS, :2]
            S, vis, info = src(pf)
        if S is not None:
            dt = 1 / fps if t_prev is None else t - t_prev
            t_prev = t
            res = rt.step(S, vis, float(np.clip(dt, 1e-3, 0.2)), info["contact"])
            q_last = res["qpos"].copy()
            ok[k] = True
            lat[k] = pf.latency_ms, res["latency_ms"]
        if q_last is not None:
            Q[k] = q_last
        if k % 1500 == 0 and k:
            el = time.perf_counter() - t_wall
            log(f"  {k}/{n} frames, {el:.0f}s ({el / k * 1e3:.1f} ms/frame), valid {ok[:k].mean():.3f}")
    cap.release()
    Q, ok, kp2d, lat = Q[:n], ok[:n], kp2d[:n], lat[:n]
    first = int(np.argmax(ok)) if ok.any() else 0
    Q[:first] = Q[first]
    if smooth_sigma > 0:                      # zero-phase: the look-ahead stroke list uses the same smoothed track
        Q = smooth_qpos(rt.m, Q, np.ones(n, bool), smooth_sigma)
    np.savez_compressed(out, qpos=Q, ok=ok, fps=fps, frame0=f0, kp2d=kp2d, latency=lat, handedness=handedness,
                        flip=flip, court_gain=court_gain, video=str(video), model=model)
    log(f"track: {n} frames ({n / fps:.0f} s), valid {ok.mean():.3f}, pose {np.median(lat[ok, 0]):.1f} ms, "
        f"retarget {np.median(lat[ok, 1]):.1f} ms -> {out}")
    return out


@dataclass
class Stroke:
    k: int                 # track frame of the fastest racket-head sample
    t: float               # seconds from the start of the track
    pos: np.ndarray        # racket head centre (court frame)
    vel: np.ndarray        # racket head velocity
    speed: float
    kind: str              # overhead | mid | low


class PlayerTrack:
    """Retargeted player motion with racket kinematics and the list of detected strokes."""

    def __init__(self, path, start=0.0, duration=0.0, fit_court=True, home=(0.0, -3.9),
                 half_x=2.0, depth=(1.7, 5.9), root_smooth=0.1):
        """start / duration [s]: part of the recording to use. fit_court: place the player's usual
        standing spot on `home` and scale the root motion so that the area they actually cover maps
        onto the playable part of the half court (|x| < half_x, depth[0] < |y| < depth[1]) -- the
        offline equivalent of a "walk to the corners of your room" calibration for a live camera."""
        z = np.load(path, allow_pickle=True)
        self.path = Path(path)
        self.fps = float(z["fps"])
        a = int(round(start * self.fps))
        b = len(z["qpos"]) if duration <= 0 else min(len(z["qpos"]), a + int(round(duration * self.fps)))
        self.ok = z["ok"][a:b]
        self.frame0 = int(z["frame0"]) + a
        self.kp2d = z["kp2d"][a:b]
        self.handed = str(z["handedness"])
        self.flip = bool(z["flip"])
        self.video = str(z["video"])
        self.latency = z["latency"][a:b]
        q = z["qpos"][a:b].copy()
        self.fit = None
        if fit_court:
            xy = q[:, :2]
            lo, med, hi = np.percentile(xy, [3, 50, 97], axis=0)
            gx = half_x / max(max(med[0] - lo[0], hi[0] - med[0]), 0.2)
            gy = min((-depth[0] - home[1]) / max(hi[1] - med[1], 0.2), (home[1] + depth[1]) / max(med[1] - lo[1], 0.2))
            g = np.clip([gx, gy], 0.5, 2.5)
            q[:, :2] = np.asarray(home) + (xy - med) * g
            q[:, 0] = np.clip(q[:, 0], -2.9, 2.9)
            q[:, 1] = np.clip(q[:, 1], -6.9, -0.45)
            # a pelvis seen by one camera and scaled up to the court jitters by metres per second: zero-phase
            # smoothing over about 0.1 s (no lag; the strokes below are detected on the smoothed motion)
            q[:, :2] = gaussian_filter1d(q[:, :2], root_smooth * self.fps, axis=0, mode="nearest")
            self.fit = dict(gain=(g * float(z["court_gain"])).round(2).tolist(), home=list(home))
        self.qpos = mirror_qpos(q) if self.handed == "left" else q        # pose of the handed character
        self.n = len(q)
        self.t = np.arange(self.n) / self.fps
        self.duration = self.n / self.fps
        m = build_single_character_model(handed=self.handed)
        d = mujoco.MjData(m)
        ci = CharacterIndex(m, "")
        self.rc = np.zeros((self.n, 3)); self.rR = np.zeros((self.n, 3, 3))
        for k in range(self.n):
            d.qpos[:] = self.qpos[k]
            mujoco.mj_kinematics(m, d)
            self.rc[k] = d.site_xpos[ci.racket_site]
            self.rR[k] = d.site_xmat[ci.racket_site].reshape(3, 3)
        v = np.gradient(self.rc, 1 / self.fps, axis=0)
        self.rv = v
        self.speed = uniform_filter1d(np.linalg.norm(v, axis=1), 3)
        # swing speed: racket head relative to the pelvis (a lunge carries the racket along at 4 m/s
        # without any swing)
        v_root = np.gradient(self.qpos[:, :3], 1 / self.fps, axis=0)
        self.swing_speed = uniform_filter1d(np.linalg.norm(v - v_root, axis=1), 3)
        # hitting speed: the part of the swing that goes towards the net. The fastest racket sample of a
        # stroke is often the loop behind the back or the racket being raised, neither of which hits anything.
        self.hit_speed = uniform_filter1d(hitting_speed(v - v_root), 3)
        self.strokes = self.detect_strokes()

    def detect_strokes(self, min_gap_s=0.9, min_speed=4.0, min_forward=3.0, min_height=0.2):
        """Peaks of the forward racket-head speed (non-maximum suppression in time), on valid frames."""
        sp = self.hit_speed.copy()
        good = self.ok & np.r_[True, self.ok[:-1]] & np.r_[self.ok[1:], True]
        sp[~good] = 0
        sp[self.swing_speed < min_speed] = 0
        thr = min_forward
        gap = int(min_gap_s * self.fps)
        peaks = []
        for k in np.argsort(-sp):
            if sp[k] < thr:
                break
            if k < 2 or k > self.n - 3 or self.rc[k, 2] < min_height:
                continue
            if np.sum(self.hit_speed[max(k - 3, 0):k + 4] > 0.5 * sp[k]) < 3:   # 1-2 frame spikes are tracking flips
                continue
            if all(abs(k - p) > gap for p in peaks):
                peaks.append(int(k))
        out = []
        for k in sorted(peaks):
            h = self.rc[k, 2]
            kind = "overhead" if h >= 1.9 else ("low" if h < 1.0 else "mid")
            out.append(Stroke(k, k / self.fps, self.rc[k].copy(), self.rv[k].copy(), float(self.swing_speed[k]), kind))
        return out

    # ------------------------------------------------------------------ sampling
    def index(self, t):
        return int(np.clip(round(t * self.fps), 0, self.n - 1))

    def qpos_at(self, t):
        x = np.clip(t * self.fps, 0, self.n - 1 - 1e-9)
        i = int(x)
        return blend_qpos(self.qpos[i], self.qpos[i + 1], x - i)

    def next_strokes(self, t, horizon=8.0):
        return [s for s in self.strokes if t < s.t <= t + horizon]

    def summary(self):
        s = self.strokes
        gaps = np.diff([x.t for x in s]) if len(s) > 1 else np.zeros(1)
        kinds = {k: sum(x.kind == k for x in s) for k in ("overhead", "mid", "low")}
        return dict(frames=self.n, seconds=round(self.duration, 1), valid=float(self.ok.mean()),
                    strokes=len(s), kinds=kinds, gap_median=float(np.median(gaps)),
                    gap_p10=float(np.percentile(gaps, 10)), gap_p90=float(np.percentile(gaps, 90)),
                    speed_median=float(np.median([x.speed for x in s])) if s else 0.0,
                    speed_max=float(max([x.speed for x in s])) if s else 0.0, handed=self.handed, fit=self.fit,
                    root_x=np.percentile(self.qpos[:, 0], [1, 50, 99]).round(2).tolist(),
                    root_y=np.percentile(self.qpos[:, 1], [1, 50, 99]).round(2).tolist())


class Director:
    """Stands a recording in for a live player.

    A person in front of a camera swings when the shuttle arrives; a recording swings when it
    swings. The director closes that gap in the only way that leaves the recorded motion intact:
    it edits *time*. When the opponent has hit, `react` looks for the next recorded stroke whose
    contact height the shuttle comes down through at a spot the avatar can get to, and cuts the
    waiting out of the recording so that this swing happens when the shuttle is there (poses are
    cross-faded over the cut; the recording loops when it runs out). Where on court the swing
    happens is the footwork assist's business (game.Match), exactly as for a camera player in a
    living room. Whether the racket then meets the shuttle and what shot comes out is decided by
    the same code as for live input.

    mode "feed" is the practice-partner variant: no cuts during rallies; the opponent is told
    where and when the next swings are (`upcoming`) and delivers the shuttle there.
    """

    def __init__(self, track: PlayerTrack, offset=0.0, mode="react", lookahead=7.0, lead=3.0, blend=0.35,
                 min_lead=0.38, search=12, reach_z=(0.0, -0.15, -0.3)):
        """min_lead [s]: shortest notice on which a swing can be produced (reaction + start of the
        swing). search: how many recorded strokes ahead may be used (the others are cut out).
        reach_z: offsets from the recorded racket-head height at which the shuttle may be met."""
        self.track, self.offset, self.mode, self.lookahead = track, float(offset), mode, lookahead
        self.lead, self.blend, self.min_lead, self.search, self.reach_z = lead, blend, min_lead, search, reach_z
        self._pending = None                   # scheduled cut: (match time, seconds to skip, cross-fade [s])
        self._fade = None                      # running cross-fade: (match time, duration, offset before the cut | pose)
        self._loop = 0
        self._shown = None                     # (number of cuts, root xy) at the last pose handed out
        self.jump = np.zeros(2)                # root displacement introduced by cuts / loops, not yet collected
        self.cuts = []                         # executed cuts: (match time, seconds skipped)
        self.reactions = dict(asked=0, found=0, too_fast=0, no_height=0, too_far=0, timing=0)

    # ------------------------------------------------------------------ time
    def _apply(self, now):
        if self._pending is not None and now >= self._pending[0]:
            t_cut, skip, dur = self._pending
            self._pending = None
            self._fade = (t_cut, dur, self.offset)
            self.offset += skip
            self.cuts.append((float(t_cut), float(skip)))

    def track_time(self, now):
        """Recording time shown at match time `now` (keeps counting through loops of the recording)."""
        self._apply(now)
        return now + self.offset

    def _strokes_after(self, tt, count):
        """[(recording time incl. loops, Stroke)] of the next `count` strokes after tt."""
        S, D = self.track.strokes, self.track.duration
        out = []
        if not S:
            return out
        n = int(tt // D)
        while len(out) < count:
            for s in S:
                if s.t + n * D > tt:
                    out.append((s.t + n * D, s))
                    if len(out) >= count:
                        break
            n += 1
        return out

    def frame_index(self, now):
        return self.track.frame0 + self.track.index(self.track_time(now) % self.track.duration)

    # ------------------------------------------------------------------ pose
    def qpos(self, now):
        tt, D = self.track_time(now), self.track.duration
        q = self.track.qpos_at(tt % D)
        loop = int(tt // D)
        edit = (len(self.cuts), loop)
        if loop != self._loop:                                      # ran through the end of the recording
            if self._fade is None or now - self._fade[0] >= self._fade[1]:
                self._fade = (now, self.blend, self.track.qpos[-1].copy())
            self._loop = loop
        # where the recorded player stands changes at a cut. A body cannot do that: the pose is cross-faded,
        # the position is not -- the displacement is reported (pop_jump) and whoever places the avatar keeps
        # it where it was and lets it travel the difference
        xy = q[:2].copy()
        if self._shown is not None and self._shown[0] != edit:
            self.jump += xy - self._shown[1]
        self._shown = (edit, xy)
        if self._fade is not None:
            t0, dur, src = self._fade
            f = (now - t0) / dur
            if f >= 1.0:
                self._fade = None
            else:
                old = src if isinstance(src, np.ndarray) else self.track.qpos_at((now + src) % D)
                q = blend_qpos(old, q, f * f * (3 - 2 * f))
                q[:2] = xy
        return q

    def pop_jump(self):
        """Root displacement (xy) that cuts have introduced into the poses since the last call."""
        j, self.jump = self.jump, np.zeros(2)
        return j

    # ------------------------------------------------------------------ between points
    def skip_idle(self, now):
        """Between points: cut to `lead` seconds before the next stroke if the recorded player does
        nothing for a while (rest between sets, end of the recording). Returns the seconds skipped."""
        tt = self.track_time(now)
        nxt = self._strokes_after(tt + 1.2, 1)
        if not nxt:
            return 0.0
        skip = nxt[0][0] - self.lead - tt
        if skip < 1.0:
            return 0.0
        self._pending = (now, skip, self.blend)
        self._apply(now)
        return skip

    # ------------------------------------------------------------------ rallies
    def react(self, now, T, P, root_now, reach, reaction=0.15):
        """The opponent has just hit; T, P: predicted flight (seconds from now, positions).
        root_now: where the avatar's pelvis is on the court (xy); reach(seconds) -> metres it can travel.
        Picks the next recorded stroke that can meet this shuttle and schedules the cut that puts the
        swing on the arrival. Returns dict(t=arrival time, pos=meeting point, offset=footwork offset
        needed there, stroke=Stroke), or None when no recorded stroke fits (too fast, too far)."""
        self.reactions["asked"] += 1
        tt = self.track_time(now)
        why = "no_height"
        cands = [(tu, s, dz) for dz in self.reach_z for tu, s in self._strokes_after(tt + 0.2, self.search)]
        for tu, s, dz in cands:
            # the shuttle is met at the racket-head height of the recorded swing (a stretch below or
            # above it if nothing else works: the hit is then a scrappy one)
            z = s.pos[2] + dz
            idx = np.flatnonzero((P[:-1, 2] >= z) & (P[1:, 2] < z) & (P[1:, 1] < -0.25))
            if idx.size == 0:
                continue
            i = int(idx[0])
            f = (P[i, 2] - z) / max(P[i, 2] - P[i + 1, 2], 1e-9)
            lead = float(T[i] + f * (T[i + 1] - T[i]))
            p_x = P[i] + f * (P[i + 1] - P[i])
            if lead < self.min_lead:
                why = "too_fast"
                continue
            skip = (tu - tt) - lead
            if skip < -0.03:                                        # this swing comes before the shuttle can
                why = "timing" if why == "no_height" else why
                continue
            off = p_x[:2] - s.pos[:2]
            root = self.track.qpos[s.k, :2] + off
            if (root[1] > -0.4 or root[1] < -7.3 or abs(root[0]) > 3.4
                    or np.linalg.norm(root - root_now) > reach(lead - reaction)):
                why = "too_far"
                continue
            if skip > 0.03:
                t_cut = now + float(np.clip(lead - 0.9, 0.0, 0.3))    # let the previous follow-through finish
                self._pending = (t_cut, skip, float(np.clip(now + lead - t_cut - 0.25, 0.12, self.blend)))
            else:
                self._pending = None
            self._apply(now)
            self.reactions["found"] += 1
            return dict(t=now + lead, pos=p_x, offset=off, stroke=s)
        self.reactions[why] += 1
        return None

    def upcoming(self, now):
        """[(match time, racket-head position)] of the strokes ahead (feed mode)."""
        tt = self.track_time(now)
        return [(tu - self.offset, s.pos) for tu, s in self._strokes_after(tt, 8) if tu <= tt + self.lookahead]

"""Retarget the camera-side player of a calibrated broadcast video onto the character, on the court.

    python scripts/retarget_broadcast.py data/videos/standard_video.mp4 --start 3000 --frames 900

Pipeline: pose pass (cached) -> court-frame reconstruction (badrt.broadcast) -> retarget pass 1
(self-collision, floor, court bounds) -> footplant anchors from the plant intervals (median of the
pass-1 sole positions, Kovar et al. 2002) -> retarget pass 2 with anchors -> metrics + video.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.broadcast import pose_pass, reconstruct  # noqa: E402
from badrt.calib import Camera, draw_court  # noqa: E402
from badrt.character import BONE_DRAW, KP, MP_IDS, CharacterIndex, personalize  # noqa: E402
from badrt.config import RetargetConfig  # noqa: E402
from badrt.court import LENGTH_HALF, WIDTH_DOUBLES_HALF, build_broadcast_model, game_view  # noqa: E402
from badrt.racket import detect_rackets, racket_targets  # noqa: E402
from badrt.racket_prior import KnnPrior  # noqa: E402
from badrt.retarget import BodyRetargeter, smooth_qpos  # noqa: E402

# source contact columns: heel L, heel R, toe L, toe R -> character sole-site indices (ci.sole_sites order:
# l heel i/o, l toe i/o, r heel i/o, r toe i/o)
CONTACT_TO_SOLE = {0: [0, 1], 1: [4, 5], 2: [2, 3], 3: [6, 7]}
BOUNDS = dict(root=(-WIDTH_DOUBLES_HALF - 0.9, WIDTH_DOUBLES_HALF + 0.9, -LENGTH_HALF - 1.6, -0.35), sole_ymax=-0.06)


def source_dims(rec, min_vis=0.8):
    """Median source bone lengths (both sides pooled) over frames where both ends are well visible."""
    P, V = rec.P[rec.ok], rec.vis[rec.ok]

    def L(a, b):
        ia, ib = MP_IDS[KP[a]], MP_IDS[KP[b]]
        good = (V[:, ia] > min_vis) & (V[:, ib] > min_vis)
        return np.linalg.norm(P[good, ia] - P[good, ib], axis=1)

    side = lambda a, b: float(np.median(np.r_[L("l_" + a, "l_" + b), L("r_" + a, "r_" + b)]))
    torso = float(np.median(np.r_[L("l_shoulder", "l_hip"), L("r_shoulder", "r_hip")]))
    return dict(thigh=side("hip", "knee"), shin=side("knee", "ankle"), upperarm=side("shoulder", "elbow"),
                forearm=side("elbow", "wrist"), torso=torso,
                shoulder=float(np.median(L("l_shoulder", "r_shoulder"))), hip=float(np.median(L("l_hip", "r_hip"))))


def run_pass(rt, rec, anchors=None, log="", cam=None, q_ref=None, ref_weight=0.0, shift=None, racket=None):
    T = len(rec.ok)
    Q = np.zeros((T, rt.m.nq)); lat = np.zeros(T); ok = np.zeros(T, bool)
    prev_valid = False
    t0 = time.time()
    for k in range(T):
        if not rec.ok[k]:
            prev_valid = False
            continue
        if not prev_valid:
            rt.reset()
        S = rec.P[k][MP_IDS] if shift is None else rec.P[k][MP_IDS] + shift[k]
        vis = rec.vis[k][MP_IDS]
        obs = None
        if cam is not None and rec.uv is not None:
            obs = (rec.uv[k][MP_IDS], rec.uv_w[k][MP_IDS], cam)
        r = rt.step(S, vis, 1 / rec.fps, anchors=None if anchors is None else anchors[k], bounds=BOUNDS, obs2d=obs,
                    q_ref=q_ref[k] if q_ref is not None and np.any(q_ref[k, 3:7]) else None, ref_weight=ref_weight,
                    racket=racket_frame(racket, k, cam))
        Q[k], lat[k], ok[k] = r["qpos"], r["latency_ms"], r["success"]
        prev_valid = True
    print(f"  {log}: {ok.sum()} frames, {time.time() - t0:.1f}s, median {np.median(lat[ok]):.1f} ms/frame", flush=True)
    return Q, lat, ok


def racket_tip_distance(m, ci):
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(d.site_xpos[ci.racket_tip] - d.site_xpos[ci.kp_sites[KP["r_wrist"]]]))


def racket_metrics(rt, cam, data, det, Q, ok, hidden=None):
    """Against the image racket detections (right wrist, conf >= 0.15; ~80-85% precision, so
    medians): tip reprojection error, 2D wrist->tip direction error; and tip smoothness.
    hidden: (T,) frames whose detections were withheld from the retargeter (dropout test); the
    direction error there is reported separately."""
    W, H = cam.width, cam.height
    err, ang, hid = [], [], []
    tips = np.full((len(Q), 3), np.nan)
    for k in np.flatnonzero(ok):
        rt.d.qpos[:] = Q[k]
        mujoco.mj_kinematics(rt.m, rt.d)
        tips[k] = rt.d.site_xpos[rt.ci.racket_tip]
        if det["hand"][k] != 0 or det["conf"][k] < 0.15:
            continue
        uv_t, uv_w = cam.project(np.stack([tips[k], rt.d.site_xpos[rt.ci.kp_sites[KP["r_wrist"]]]]))
        err.append(np.linalg.norm(uv_t - det["uv"][k]))
        a = det["uv"][k] - data["image"][k][16, :2] * [W, H]
        b = uv_t - uv_w
        c = np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)
        ang.append(np.degrees(np.arccos(c)))
        hid.append(hidden is not None and bool(hidden[k]))
    err = np.array(err) if err else np.full(1, np.nan)
    ang, hid = np.array(ang), np.array(hid, bool)
    acc = np.linalg.norm(np.diff(tips, 2, axis=0), axis=1) * 30.0 ** 2
    r = dict(racket_tip_px_median=float(np.median(err)), racket_tip_inlier=float(np.mean(err < 15)),
             racket_dir_deg_median=float(np.median(ang)) if len(ang) else float("nan"),
             racket_tip_accel_mps2=float(np.nanmean(acc)), racket_eval_frames=int(len(ang)))
    if hid.any():
        r.update(racket_dir_deg_hidden=float(np.median(ang[hid])), racket_dir_hidden_lt30=float(np.mean(ang[hid] < 30)),
                 racket_dir_deg_visible=float(np.median(ang[~hid])), racket_hidden_frames=int(hid.sum()))
    return r


def dropout_mask(T, frac, run=15, seed=0):
    """Random runs of `run` frames covering about `frac` of the sequence."""
    rng = np.random.default_rng(seed)
    hid = np.zeros(T, bool)
    for s in rng.choice(max(1, T - run), int(frac * T / run), replace=False):
        hid[s:s + run] = True
    return hid


def racket_frame(racket, k, cam):
    if racket is None or racket["w"][k] <= 0:
        return None
    return dict(tip=racket["tip"][k], w=racket["w"][k], uv=racket["uv"][k], cam=cam,
                prior_only=bool(racket.get("prior_only", np.zeros(len(racket["w"]), bool))[k]))


def sole_positions(rt, Q, ok):
    out = np.full((len(Q), 8, 3), np.nan)
    for k in np.flatnonzero(ok):
        rt.d.qpos[:] = Q[k]
        mujoco.mj_kinematics(rt.m, rt.d)
        out[k] = rt.d.site_xpos[rt.ci.sole_sites]
    return out


def plant_intervals(c):
    out = []
    k = 0
    while k < len(c):
        if c[k]:
            e = k
            while e < len(c) and c[e]:
                e += 1
            out.append((k, e))
            k = e
        else:
            k += 1
    return out


def build_anchors(rec, sole1, ramp=4):
    """Per frame (idx, targets, weights): plant target = median pass-1 position over the interval, z = 0."""
    T = len(rec.ok)
    per = [([], [], []) for _ in range(T)]
    n_plants = 0
    for c, sites in CONTACT_TO_SOLE.items():
        for a, b in plant_intervals(rec.contact[:, c]):
            n_plants += 1
            for s in sites:
                tgt = np.nanmedian(sole1[a:b, s], axis=0)
                tgt[2] = 0.0
                for k in range(a, b):
                    w = min(1.0, (k - a + 1) / (ramp + 1), (b - k) / (ramp + 1))
                    per[k][0].append(s); per[k][1].append(tgt); per[k][2].append(w)
    anchors = [(np.array(i, int), np.array(t).reshape(-1, 3), np.array(w)) for i, t, w in per]
    return anchors, n_plants


def plant_shift(rec, sole1, anchors, sigma=4.0):
    """Smooth horizontal root-offset field (Kovar et al. 2002): the planted-sole corrections
    (target - pass-1 sole position), Gaussian-interpolated in time, so the source moves with the
    plants instead of the anchors fighting the root term frame by frame."""
    from scipy.ndimage import gaussian_filter1d
    T = len(rec.ok)
    num, den = np.zeros((T, 2)), np.zeros(T)
    for k, (idx, tgt, w) in enumerate(anchors):
        if len(idx) and np.isfinite(sole1[k, idx]).all():
            num[k] = (w[:, None] * (tgt[:, :2] - sole1[k, idx, :2])).sum(0)
            den[k] = w.sum()
    out = np.zeros((T, 3))
    for a, b in segments_of(rec.ok):
        n = gaussian_filter1d(num[a:b], sigma, axis=0, mode="constant")
        dd = gaussian_filter1d(den[a:b], sigma, mode="constant")
        out[a:b, :2] = n / np.maximum(dd, 1e-3)[:, None] * np.clip(dd / 0.05, 0, 1)[:, None]
    return out


def segments_of(ok):
    return plant_intervals(np.asarray(ok, bool))


def metrics(rt, cam, rec, data, Q, ok, sole):
    W, H = cam.width, cam.height
    rep, rep_feet, pen, npen, minz, out_root = [], [], [], [], [], 0
    KPS = np.full((len(Q), len(rt.ci.kp_sites), 3), np.nan)
    n_ray = cam.R[2]
    occ_tests = occ_bad = rk_tests = rk_behind = 0
    for k in np.flatnonzero(ok & data["ok"].astype(bool)):
        rt.d.qpos[:] = Q[k]
        mujoco.mj_kinematics(rt.m, rt.d)
        P = rt.d.site_xpos[rt.ci.kp_sites]
        KPS[k] = P
        up = P[[KP["l_shoulder"], KP["r_shoulder"]]].mean(0) - P[[KP["l_hip"], KP["r_hip"]]].mean(0)
        fwd = np.cross(up, P[KP["r_shoulder"]] - P[KP["l_shoulder"]]); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
        tcen = P[[KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]].mean(0)
        if P[KP["r_wrist"], 2] < P[KP["r_shoulder"], 2]:
            rk_tests += 1
            rk_behind += (rt.d.site_xpos[rt.ci.racket_site] - tcen) @ fwd < -0.1
        # occlusion ordering: MediaPipe says a wrist/elbow inside the torso outline is hidden (behind
        # the torso seen from the camera) or visible (in front); does the character agree?
        uv_all = data["image"][k][:, :2] * [W, H]
        quad = uv_all[[11, 12, 24, 23]].astype(np.float32)
        tc = P[[KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]].mean(0)
        for name in ("l_wrist", "r_wrist", "l_elbow", "r_elbow"):
            j = MP_IDS[KP[name]]
            if cv2.pointPolygonTest(quad, (float(uv_all[j, 0]), float(uv_all[j, 1])), False) < 0:
                continue
            vj = data["vis"][k][j]
            if 0.5 <= vj <= 0.8:
                continue
            rel = (P[KP[name]] - tc) @ n_ray
            occ_tests += 1
            occ_bad += (vj < 0.5 and rel < -0.02) or (vj > 0.8 and rel > 0.08)
        uv = cam.project(P)
        uv_mp = data["image"][k][MP_IDS, :2] * [W, H]
        v = data["vis"][k][MP_IDS] > 0.6
        e = np.linalg.norm(uv - uv_mp, axis=1)
        rep.append(np.mean(e[v])) if v.any() else None
        f = [KP[n] for n in ("l_heel", "r_heel", "l_toe", "r_toe", "l_ankle", "r_ankle")]
        vf = v[f]
        if vf.any():
            rep_feet.append(np.mean(e[f][vf]))
        p, n = rt.penetration(Q[k])
        pen.append(p); npen.append(n)
        minz.append(rt.d.site_xpos[rt.ci.sole_sites, 2].min())
        x, y = Q[k][0], Q[k][1]
        out_root += not (BOUNDS["root"][0] <= x <= BOUNDS["root"][1] and BOUNDS["root"][2] <= y <= BOUNDS["root"][3])
    # foot skating during source footplants: horizontal speed of the matching sole points
    skate = []
    for c, sites in CONTACT_TO_SOLE.items():
        for a, b in plant_intervals(rec.contact[:, c] & ok):
            if b - a < 2:
                continue
            for s in sites:
                v = np.linalg.norm(np.diff(sole[a:b, s, :2], axis=0), axis=1) * rec.fps
                skate.extend(v.tolist())
    skate = np.array(skate) if skate else np.zeros(1)
    # jitter: mean |second difference| of joint angles (rad/frame^2)
    qd = Q[ok][:, 7:]
    jit = float(np.mean(np.abs(np.diff(qd, 2, axis=0)))) if len(qd) > 3 else 0.0
    acc = np.linalg.norm(np.diff(KPS, 2, axis=0), axis=2) * rec.fps ** 2        # (T-2, K), nan across gaps
    wr = [KP["l_wrist"], KP["r_wrist"]]
    hips = KPS[:, [KP["l_hip"], KP["r_hip"]]].mean(1)
    acc_hip = np.linalg.norm(np.diff(hips, 2, axis=0), axis=1) * rec.fps ** 2
    src = np.where(rec.ok[:, None, None], rec.P[:, MP_IDS], np.nan)
    acc_src = np.linalg.norm(np.diff(src, 2, axis=0), axis=2) * rec.fps ** 2
    return dict(frames=int(ok.sum()), reproj_px_mean=float(np.mean(rep)), reproj_px_median=float(np.median(rep)),
                reproj_feet_px=float(np.mean(rep_feet)), skate_mps_mean=float(skate.mean()),
                skate_mps_p90=float(np.percentile(skate, 90)), self_pen_max_m=float(np.max(pen)),
                self_pen_frames=float(np.mean(np.array(npen) > 0)), sole_min_z=float(np.min(minz)),
                root_out_of_bounds=int(out_root), jitter=jit,
                kp_accel_mps2=float(np.nanmean(acc)), wrist_accel_mps2=float(np.nanmean(acc[:, wr])),
                hip_accel_mps2=float(np.nanmean(acc_hip)), src_kp_accel_mps2=float(np.nanmean(acc_src)),
                src_hip_accel_mps2=float(np.nanmean(acc_src[:, [KP["l_hip"], KP["r_hip"]]])),
                occlusion_order_violation=float(occ_bad / max(occ_tests, 1)), occlusion_tests=int(occ_tests),
                racket_behind_low_hand=float(rk_behind / max(rk_tests, 1)))


def mj_camera(cam: Camera, back=0.0, fovy=None, lift=0.0, look_dist=14.0):
    f = cam.R[2]                                     # optical axis (court frame)
    c = cam.center - back * np.r_[f[:2] / np.linalg.norm(f[:2]), 0] + [0, 0, lift]
    mc = mujoco.MjvCamera()
    mc.type = mujoco.mjtCamera.mjCAMERA_FREE
    mc.lookat[:] = c + look_dist * f
    mc.distance = look_dist
    mc.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
    mc.elevation = float(np.degrees(np.arcsin(f[2])))
    return mc, (fovy or cam.fovy_deg)


def draw_skeleton(img, uv, color, thick=2, vis=None):
    for a, b in BONE_DRAW:
        i, j = KP[a], KP[b]
        if vis is not None and (vis[i] < 0.5 or vis[j] < 0.5):
            continue
        cv2.line(img, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), color, thick, cv2.LINE_AA)


def game_camera(aspect):
    gv = game_view(aspect)
    mc = mujoco.MjvCamera()
    mc.type = mujoco.mjtCamera.mjCAMERA_FREE
    mc.lookat[:] = gv["lookat"]; mc.distance = gv["distance"]
    mc.azimuth, mc.elevation = gv["azimuth"], gv["elevation"]
    return mc, gv["fovy"]


def render_video(path, video, cam, data, rec, rt_model, ci, Q, ok, start, metrics_txt="", racket=None):
    m = rt_model
    d = mujoco.MjData(m)
    W, H = cam.width, cam.height
    r = mujoco.Renderer(m, H, W)
    cam_match, fov_match = mj_camera(cam)
    cam_game, fov_game = game_camera(W / H)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), rec.fps, (2 * W, H + H // 2))
    opt = mujoco.MjvOption()
    last = None
    for k in range(len(Q)):
        okf, frame = cap.read()
        if not okf:
            break
        if ok[k]:
            last = Q[k]
        if last is not None:
            d.qpos[:] = last
        mujoco.mj_forward(m, d)
        left = draw_court(frame.copy(), cam, (0, 0, 255), 1)
        if ok[k]:
            if data["ok"][k]:
                uv_mp = data["image"][k][MP_IDS, :2] * [W, H]
                draw_skeleton(left, uv_mp, (0, 255, 0), 1, data["vis"][k][MP_IDS])
            uv_c = cam.project(d.site_xpos[ci.kp_sites])
            draw_skeleton(left, uv_c, (255, 160, 0), 2)
            tw = cam.project(np.stack([d.site_xpos[ci.kp_sites[KP["r_wrist"]]], d.site_xpos[ci.racket_tip]]))
            cv2.line(left, tuple(np.int32(tw[0])), tuple(np.int32(tw[1])), (255, 160, 0), 1, cv2.LINE_AA)
            if racket is not None and np.isfinite(racket["uv"][k, 0]):
                cv2.circle(left, tuple(np.int32(racket["uv"][k])), 4, (0, 0, 255), -1, cv2.LINE_AA)
            for c in range(4):
                if rec.contact[k, c]:
                    uv = cam.project(d.site_xpos[ci.sole_sites[CONTACT_TO_SOLE[c]]].mean(0, keepdims=True))[0]
                    cv2.circle(left, tuple(np.int32(uv)), 5, (0, 255, 255), -1, cv2.LINE_AA)
        m.vis.global_.fovy = fov_game
        r.update_scene(d, cam_game, opt)
        game = r.render()[:, :, ::-1].copy()
        m.vis.global_.fovy = fov_match
        r.update_scene(d, cam_match, opt)
        match = r.render()[:, :, ::-1].copy()
        for img, txt in ((left, "court (red) MediaPipe (green) character (blue) racket det (red dot)"),
                         (game, "virtual low camera (game view)")):
            cv2.putText(img, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(left, f"frame {start + k}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # bottom strip: matched camera render (half size) + original (half size), for direct comparison
        strip = np.hstack([cv2.resize(frame, (W // 2, H // 2)), cv2.resize(match, (W // 2, H // 2)),
                           np.zeros((H // 2, W, 3), np.uint8)])
        y = 24
        for line in metrics_txt.splitlines():
            cv2.putText(strip, line, (W + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            y += 22
        cv2.putText(strip, "air" if rec.airborne[k] else "", (W + 12, H // 2 - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)
        out.write(np.vstack([np.hstack([left, game]), strip]))
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--model", default="heavy")
    ap.add_argument("--out", default="results/broadcast")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--no-personalize", action="store_true", help="keep the default character proportions")
    ap.add_argument("--no-ablation", action="store_true", help="only the full method (no baseline variants)")
    ap.add_argument("--rec-opts", default="{}", help='JSON kwargs for reconstruct(), e.g. {"smooth_sigma": 1.5}')
    ap.add_argument("--cfg-opts", default="{}", help='JSON overrides of the broadcast RetargetConfig')
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--post-opts", default="{}",
                    help='offline smoothing + constraint re-projection pass, e.g. {"sigma": 2, "ref_weight": 30}; '
                         'sigma 0 disables')
    ap.add_argument("--racket-prior", default="assets/racket_prior.npz", help="learned pose->racket prior, or none")
    ap.add_argument("--prior-ablation", action="store_true", help="also run the full method without the racket prior")
    ap.add_argument("--racket-dropout", type=float, default=0.0,
                    help="withhold racket detections in random 15-frame runs covering this fraction (evaluation)")
    a = ap.parse_args()
    stem = Path(a.video).stem
    cam = Camera.load(a.calib or f"assets/calib/{stem}.json")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{stem}_{a.start}_{a.frames}"
    t0 = time.time()
    data = pose_pass(a.video, cam, a.model, a.start, a.frames, cache=out / f"{tag}_pose.npz")
    tag += a.tag_suffix
    print(f"pose pass {time.time() - t0:.0f}s: main-camera {data['main'].mean():.2f}, player found {data['ok'].mean():.2f}")
    rec_opts = json.loads(a.rec_opts)
    cfg_opts = json.loads(a.cfg_opts)
    rec = reconstruct(cam, data, **rec_opts)
    print("reconstruction", rec.info)

    dims = personalize(source_dims(rec)) if not a.no_personalize else None
    print("character dims", None if dims is None else {k: round(v, 3) for k, v in dims.items()})
    m = build_broadcast_model(cam.width, cam.height, dims=dims)
    ci = CharacterIndex(m, "p_")
    if a.render_only:
        z = np.load(out / f"{tag}_qpos.npz")
        render_video(out / f"{tag}.mp4", a.video, cam, data, rec, m, ci, z["qpos"], z["ok"], a.start,
                     _metrics_text(json.loads((out / f"{tag}.json").read_text())))
        print("video ->", out / f"{tag}.mp4")
        return
    tip_dist = racket_tip_distance(m, ci)
    det_path = out / f"{tag.removesuffix(a.tag_suffix) if a.tag_suffix else tag}_racket.npz"
    if det_path.exists():
        det = dict(np.load(det_path))
    else:
        t1 = time.time()
        det = detect_rackets(a.video, cam, data, a.start)
        det.pop("bg")
        np.savez_compressed(det_path, **det)
        print(f"racket detection {time.time() - t1:.0f}s")
    hidden = dropout_mask(len(rec.ok), a.racket_dropout) if a.racket_dropout > 0 else None
    det_in = det if hidden is None else {**det, "hand": np.where(hidden, -1, det["hand"])}
    prior = None
    if a.racket_prior != "none" and Path(a.racket_prior).exists():
        prior = KnnPrior.load(a.racket_prior).predict_world(rec.P, rec.fps)
    rk = racket_targets(cam, rec, det_in, tip_dist, prior=prior)
    print(f"racket: detected {np.mean(det['hand'] >= 0):.2f} of frames, right-wrist lifted {rk['n_detect']}, "
          f"rejected by temporal check {rk['n_reject']}, by prior {rk['n_prior_reject']}, target frames "
          f"{np.mean(rk['w'] > 0):.2f} (image term {np.mean(np.isfinite(rk['uv'][:, 0])):.2f}; wrist->tip "
          f"{tip_dist:.3f} m; prior {'on' if prior is not None else 'off'}"
          f"{f', dropout {hidden.mean():.2f}' if hidden is not None else ''})")
    variants = {"full": (rec, RetargetConfig.preset("broadcast", **cfg_opts), True, True, rk)}
    if a.prior_ablation and prior is not None:
        variants["full_no_rkprior"] = (rec, RetargetConfig.preset("broadcast", **cfg_opts), True, True,
                                       racket_targets(cam, rec, det_in, tip_dist))
    post = json.loads(a.post_opts)
    post_sigma, post_root, ref_w = post.get("sigma", 2.0), post.get("sigma_root", 1.5), post.get("ref_weight", 60.0)
    post_leg = post.get("leg_frac", 0.3)
    shift_sigma = post.get("shift_sigma", 0.0)
    post_iters = int(post.get("iters", 1))
    if not a.no_ablation:
        rec_plain = reconstruct(cam, data)
        plain = dict(accel_weight=0.0, reproj_weight=0.0, racket_front_weight=0.0)
        no_rk = dict(racket_weight=0.0, racket_reproj_weight=0.0)
        variants = {
            "baseline": (rec_plain, RetargetConfig.preset("broadcast", self_collision=False, **plain, **no_rk),
                         False, False, None),
            "v1_contact_collision": (rec_plain, RetargetConfig.preset("broadcast", **plain, **no_rk), True, False, None),
            "+2d_reproj": (rec_plain, RetargetConfig.preset("broadcast", accel_weight=0.0, racket_front_weight=0.0,
                                                            **no_rk), True, True, None),
            "+racket_prior": (rec_plain, RetargetConfig.preset("broadcast", accel_weight=0.0, **no_rk), True, True,
                              None),
            "full_no_racket": (rec, RetargetConfig.preset("broadcast", **{**cfg_opts, **no_rk}), True, True, None),
            **variants}
    res = dict(video=str(a.video), start=a.start, frames=len(rec.ok), reconstruction=rec.info, character_dims=dims,
               camera=dict(f=cam.Kmat[0, 0], fovy=cam.fovy_deg, center=cam.center.tolist(), rms_px=cam.reproj_px),
               pose=dict(main_camera=float(data["main"].mean()), found=float(data["ok"].mean()),
                         latency_ms=float(np.median(data["lat"][data["ok"].astype(bool)]))),
               latency_ms={})
    for name, (rc, cfg, use_anchors, use_2d, rkv) in variants.items():
        rt = BodyRetargeter(cfg, m, "p_")
        Q, lat, okq = run_pass(rt, rc, None, f"{name}: pass 1", cam if use_2d else None, racket=rkv)
        shift = None
        if use_anchors:
            sole1 = sole_positions(rt, Q, okq)
            anchors, n_pl = build_anchors(rc, sole1)
            if name == "full" and shift_sigma > 0:
                shift = plant_shift(rc, sole1, anchors, shift_sigma)
            Q, lat, okq = run_pass(rt, rc, anchors, f"{name}: pass 2 ({n_pl} footplants)", cam if use_2d else None,
                                   shift=shift, racket=rkv)
        runs = [(name, Q, okq)]
        if name.startswith("full") and post_sigma > 0:
            # alternating projection: smooth the trajectory, re-impose plants / floor / collisions
            # with a pull toward the smoothed pose (strong on the upper body, weaker on root + legs)
            wv = np.full(m.nv, float(ref_w))
            leg = [m.jnt_dofadr[j] + i for j in range(m.njnt) for i in range(3 if m.jnt_type[j] == 1 else 1)
                   if any(s in m.joint(j).name for s in ("hip", "knee", "ankle", "subtalar"))]
            wv[:6] *= post_leg; wv[leg] *= post_leg
            Q3, ok3 = Q, okq
            for it in range(post_iters):
                Qs = smooth_qpos(m, Q3, ok3, post_sigma, post_root)
                anchors, n_pl = build_anchors(rc, sole_positions(rt, Qs, ok3))
                Q3, lat3, ok3 = run_pass(rt, rc, anchors, f"{name}: pass 3.{it} (smoothed ref, sigma {post_sigma})",
                                         cam if use_2d else None, q_ref=Qs, ref_weight=wv, shift=shift, racket=rkv)
            runs = ([("full_no_post", Q, okq)] if name == "full" else []) + [(name, Q3, ok3)]
            lat = lat + lat3
        for nm, Qr, okr in runs:
            res[nm] = metrics(rt, cam, rc, data, Qr, okr, sole_positions(rt, Qr, okr))
            res[nm].update(racket_metrics(rt, cam, data, det, Qr, okr, hidden))
            r = res[nm]
            print(f"{nm:22s} reproj {r['reproj_px_mean']:.1f}px (feet {r['reproj_feet_px']:.1f})  skate "
                  f"{r['skate_mps_mean']:.2f} m/s  self-pen {100 * r['self_pen_frames']:.1f}%  "
                  f"kp accel {r['kp_accel_mps2']:.1f} m/s2 (wrist {r['wrist_accel_mps2']:.1f}, hip "
                  f"{r['hip_accel_mps2']:.1f}; src {r['src_kp_accel_mps2']:.1f}/{r['src_hip_accel_mps2']:.1f})  "
                  f"occl-order viol {100 * r['occlusion_order_violation']:.1f}%  racket behind "
                  f"{100 * r['racket_behind_low_hand']:.0f}%  racket tip {r['racket_tip_px_median']:.1f}px "
                  f"(<15px {100 * r['racket_tip_inlier']:.0f}%, dir {r['racket_dir_deg_median']:.0f} deg)  "
                  f"racket accel {r['racket_tip_accel_mps2']:.0f}"
                  + (f"  hidden-racket dir {r['racket_dir_deg_hidden']:.0f} deg (<30deg "
                     f"{100 * r['racket_dir_hidden_lt30']:.0f}%)" if "racket_dir_deg_hidden" in r else ""), flush=True)
        res["latency_ms"][name] = float(np.median(lat[okq]))
        if name == "full":
            Qf, okf = runs[-1][1], runs[-1][2]
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"{tag}_qpos.npz", qpos=Qf, ok=okf, contact=rec.contact, airborne=rec.airborne,
                        P=rec.P.astype(np.float32))
    if not a.no_video:
        render_video(out / f"{tag}.mp4", a.video, cam, data, rec, m, ci, Qf, okf, a.start, _metrics_text(res),
                     racket=rk)
        print("video ->", out / f"{tag}.mp4")


def _metrics_text(res):
    f = res["full"]
    b = res.get("baseline", f)
    return (f"reprojection {f['reproj_px_mean']:.1f}px (feet {f['reproj_feet_px']:.1f}px)\n"
            f"foot skate in plants {f['skate_mps_mean']:.2f} m/s (baseline {b['skate_mps_mean']:.2f})\n"
            f"self-penetration frames {100 * f['self_pen_frames']:.1f}% (baseline {100 * b['self_pen_frames']:.1f}%)\n"
            f"racket tip vs detection {f['racket_tip_px_median']:.1f}px, direction {f['racket_dir_deg_median']:.0f} deg "
            f"(baseline {b['racket_dir_deg_median']:.0f} deg)\n"
            f"yellow dots: footplants   bottom: video | matched-camera render")


if __name__ == "__main__":
    main()

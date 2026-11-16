"""Synthetic checks of the broadcast pipeline: calibration, court-frame reconstruction, self-collision
and court bounds in the retargeter.  python tests/test_broadcast.py"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badrt.broadcast import _rot_between, reconstruct  # noqa: E402
from badrt.calib import ROWS_Y, SIDES_X, Camera, calibrate  # noqa: E402
from badrt.character import KP, MP_IDS, CharacterIndex  # noqa: E402
from badrt.config import RetargetConfig  # noqa: E402
from badrt.court import LENGTH_HALF, WIDTH_DOUBLES_HALF, build_broadcast_model  # noqa: E402
from badrt.retarget import BodyRetargeter  # noqa: E402

W, H = 960, 544


def true_camera():
    """Broadcast-like camera 10.5 m behind the near baseline, 2.8 m high, pitched down ~9 deg."""
    c = np.array([0.2, -17.2, 2.8]); look = np.array([0.0, -1.0, 0.3])
    f = look - c; f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.vstack([x, y, f])
    K = [[1620.0, 0, W / 2], [0, 1620.0, H / 2], [0, 0, 1]]
    return Camera(W, H, K, cv2.Rodrigues(R)[0].ravel().tolist(), (-R @ c).tolist(), 0.0, 0.0)


def render_court(cam):
    img = np.zeros((H, W, 3), np.uint8); img[:] = (70, 140, 30)          # green court (BGR)
    segs = [((-WIDTH_DOUBLES_HALF, y, 0), (WIDTH_DOUBLES_HALF, y, 0)) for y in ROWS_Y]
    segs += [((x, -LENGTH_HALF, 0), (x, LENGTH_HALF, 0)) for x in SIDES_X]
    for a, b in segs:
        pts = cam.project(np.linspace(a, b, 60))
        cv2.polylines(img, [np.int32(pts)], False, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def test_calibration():
    cam = true_camera()
    est, _ = calibrate(render_court(cam))
    d_center = float(np.linalg.norm(est.center - cam.center))
    d_f = float(abs(est.Kmat[0, 0] - cam.Kmat[0, 0]) / cam.Kmat[0, 0])
    assert est.reproj_px < 1.5 and d_center < 0.3 and d_f < 0.03, (est.reproj_px, d_center, d_f)
    return dict(rms_px=est.reproj_px, center_err_m=d_center, focal_rel_err=d_f)


def synthetic_sequence(cam, T=150, fps=30.0, seed=0):
    """Rest-pose character keypoints moved along a path with a 0.3 s jump, rendered as MediaPipe output."""
    rng = np.random.default_rng(seed)
    m = build_broadcast_model(W, H)
    d = mujoco.MjData(m); ci = CharacterIndex(m, "p_")
    mujoco.mj_kinematics(m, d)
    kp0 = d.site_xpos[ci.kp_sites].copy()
    hip0 = kp0[[KP["l_hip"], KP["r_hip"]]].mean(0)
    t = np.arange(T) / fps
    # +2 cm: MediaPipe's foot_index sits on the shoe sole ~2 cm above the floor (SOLE_H convention)
    path = np.column_stack([1.2 * np.sin(1.3 * t), -4.0 + 1.5 * np.sin(0.9 * t), np.full(T, 0.02)])
    jump = (t > 2.0) & (t < 2.3)
    path[jump, 2] += 0.25 * np.sin(np.pi * (t[jump] - 2.0) / 0.3)
    data = dict(ok=np.ones(T, bool), main=np.ones(T, bool), world=np.zeros((T, 33, 3)), image=np.zeros((T, 33, 3)),
                vis=np.zeros((T, 33)), fps=np.array(fps), width=np.array(W), height=np.array(H),
                reset=np.zeros(T, bool))
    Ktrue = cam.Kmat
    hips = []
    for k in range(T):
        P = kp0 - np.r_[hip0[:2], 0] + path[k]
        hip = P[[KP["l_hip"], KP["r_hip"]]].mean(0)
        hips.append(hip)
        uv = cam.project(P) + rng.normal(0, 1.0, (len(P), 2))
        hip_uv = cam.project(hip[None])[0]
        ray = np.array([(hip_uv[0] - Ktrue[0, 2]) / Ktrue[0, 0], (hip_uv[1] - Ktrue[1, 2]) / Ktrue[1, 1], 1.0])
        R_ray = _rot_between(np.array([0, 0, 1.0]), ray)
        world = (P - hip) @ (cam.R.T @ R_ray)                        # inverse of to_court
        world += rng.normal(0, 0.01, world.shape)
        data["world"][k][MP_IDS] = world
        data["image"][k][MP_IDS, 0] = uv[:, 0] / W
        data["image"][k][MP_IDS, 1] = uv[:, 1] / H
        data["vis"][k][MP_IDS] = 0.95
    return data, np.array(hips), jump


def test_reconstruction():
    cam = true_camera()
    data, hips, jump = synthetic_sequence(cam)
    rec = reconstruct(cam, data)
    err = np.linalg.norm(rec.T - hips, axis=1)
    err_xy = np.linalg.norm(rec.T[:, :2] - hips[:, :2], axis=1)
    hit = rec.airborne[jump].mean()
    false_air = rec.airborne[~jump].mean()
    assert np.median(err) < 0.08 and np.percentile(err, 95) < 0.2, (np.median(err), np.percentile(err, 95))
    comp = np.median(np.abs(rec.T - hips), axis=0)
    return dict(T_err_median_m=float(np.median(err)), T_err_p95_m=float(np.percentile(err, 95)),
                T_err_xy_median_m=float(np.median(err_xy)), T_err_axis_median_m=comp.round(4).tolist(),
                scale=rec.scale,
                jump_detected_frac=float(hit), false_airborne_frac=float(false_air),
                contact_frac=rec.info["contact_frac"])


def _arm_through_torso_target(rt):
    """Rest keypoints with the left wrist / hand pushed into the chest."""
    S = rt.neutral_kp.copy()
    chest = S[[KP["l_shoulder"], KP["r_shoulder"]]].mean(0) + [0, 0.0, -0.18]
    for n in ("l_elbow",):
        S[KP[n]] = S[KP["l_shoulder"]] + [0.05, 0.05, -0.25]
    for n in ("l_wrist", "l_index", "l_pinky"):
        S[KP[n]] = chest + [0.05 if n != "l_wrist" else 0.0, 0.0, 0.0]
    return S


def test_self_collision():
    m = build_broadcast_model(W, H)
    out = {}
    for name, on in (("off", False), ("on", True)):
        rt = BodyRetargeter(RetargetConfig.preset("broadcast", self_collision=on), m, "p_")
        S = _arm_through_torso_target(rt)
        for _ in range(5):
            r = rt.step(S)
        out[name] = rt.penetration(r["qpos"])[0]
    assert out["on"] < 0.02 and out["off"] > out["on"] + 0.02, out
    return dict(penetration_off_m=out["off"], penetration_on_m=out["on"])


def test_court_bounds():
    m = build_broadcast_model(W, H)
    rt = BodyRetargeter(RetargetConfig.preset("broadcast"), m, "p_")
    S = rt.neutral_kp.copy() + [0, -0.2, 0]                         # target straddling the net (y = 0)
    bounds = dict(root=(-3.9, 3.9, -8.3, -0.35), sole_ymax=-0.06)
    for _ in range(5):
        r = rt.step(S, bounds=bounds)
    rt.d.qpos[:] = r["qpos"]; mujoco.mj_kinematics(m, rt.d)
    ymax = float(rt.d.site_xpos[rt.ci.sole_sites, 1].max())
    assert ymax <= -0.05 and r["qpos"][1] <= -0.34, (ymax, r["qpos"][1])
    return dict(sole_y_max=ymax, root_y=float(r["qpos"][1]))


if __name__ == "__main__":
    res = {}
    for fn in (test_calibration, test_reconstruction, test_self_collision, test_court_bounds):
        res[fn.__name__] = fn()
        print(fn.__name__, "ok", flush=True)
    print(json.dumps(res, indent=1))
    Path("results/stats").mkdir(parents=True, exist_ok=True)
    Path("results/stats/broadcast_tests.json").write_text(json.dumps(res, indent=1))

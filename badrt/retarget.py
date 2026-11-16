"""Per-frame interaction-mesh Laplacian SQP retargeting of the body (no fingers).

Unknown: increment dq in MuJoCo's velocity space (nv = 38), integrated with mj_integratePos,
so ball joints need no special quaternion handling.

    min  w_lap ||diag(vis) (L V(q+dq) - L V_src)||^2      interaction-mesh Laplacian
       + w_bone sum vis_b ||u_b(q+dq) - u_b_src||^2        bone directions
       + w_kp ||diag(vis)(P(q+dq) - P_src)||^2             weak absolute anchor
       + w_root ||hip_mid(q+dq) - hip_mid_src||^2          root position
       + w_s ||(q+dq) - q_prev||^2 + w_r ||(q+dq) - q_rest||^2 (joints only)
    s.t. hinge limits, sole points z >= 0, ||dq|| <= step
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .character import KP, SOLE_KP, CharacterIndex, build_single_character_model
from .config import RetargetConfig
from .laplacian import delaunay_adjacency, laplacian_matrix
from .qp import ParametricQP


def _mid(*names):
    return tuple(KP[n] for n in names)


def _gauss_quat(q, sigma):
    from scipy.ndimage import gaussian_filter1d
    q = q.copy()
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0:
            q[k] = -q[k]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def smooth_qpos(m, Q, ok, sigma=1.5, sigma_root=None):
    """Zero-phase Gaussian smoothing of a qpos trajectory per contiguous valid run
    (root translation, quaternions with sign alignment, hinge angles)."""
    from scipy.ndimage import gaussian_filter1d
    sigma_root = sigma if sigma_root is None else sigma_root
    out = Q.copy()
    k = 0
    while k < len(ok):
        if not ok[k]:
            k += 1
            continue
        e = k
        while e < len(ok) and ok[e]:
            e += 1
        if e - k >= 3:
            seg = Q[k:e]
            for j in range(m.njnt):
                a = m.jnt_qposadr[j]
                t = m.jnt_type[j]
                if t == mujoco.mjtJoint.mjJNT_FREE:
                    out[k:e, a:a + 3] = gaussian_filter1d(seg[:, a:a + 3], sigma_root, axis=0, mode="nearest")
                    out[k:e, a + 3:a + 7] = _gauss_quat(seg[:, a + 3:a + 7], sigma)
                elif t == mujoco.mjtJoint.mjJNT_BALL:
                    out[k:e, a:a + 4] = _gauss_quat(seg[:, a:a + 4], sigma)
                else:
                    out[k:e, a] = gaussian_filter1d(seg[:, a], sigma, mode="nearest")
        k = e
    return out


# bones as (from, to); a tuple means the mean of those keypoints
BONES = [
    (_mid("l_hip", "r_hip"), _mid("l_shoulder", "r_shoulder")),
    (KP["l_shoulder"], KP["r_shoulder"]), (KP["l_hip"], KP["r_hip"]),
    (_mid("l_shoulder", "r_shoulder"), KP["nose"]), (KP["l_ear"], KP["r_ear"]),
]
for s in "lr":
    BONES += [(KP[f"{s}_shoulder"], KP[f"{s}_elbow"]), (KP[f"{s}_elbow"], KP[f"{s}_wrist"]),
              (KP[f"{s}_wrist"], KP[f"{s}_index"]), (KP[f"{s}_wrist"], KP[f"{s}_pinky"]),
              (KP[f"{s}_pinky"], KP[f"{s}_index"]),
              (KP[f"{s}_hip"], KP[f"{s}_knee"]), (KP[f"{s}_knee"], KP[f"{s}_ankle"]),
              (KP[f"{s}_heel"], KP[f"{s}_toe"]), (KP[f"{s}_ankle"], KP[f"{s}_heel"])]


def _selector(spec, k):
    row = np.zeros(k)
    idx = spec if isinstance(spec, tuple) else (spec,)
    row[list(idx)] = 1.0 / len(idx)
    return row


def bone_matrices(k):
    A = np.array([_selector(a, k) for a, _ in BONES])
    B = np.array([_selector(b, k) for _, b in BONES])
    return A, B


# SOLE_KP order (heel L, heel R, toe L, toe R) -> sole-corner sites (l heel i/o, l toe i/o, r heel i/o, r toe i/o)
ONLINE_SOLE = {0: [0, 1], 1: [4, 5], 2: [2, 3], 3: [6, 7]}

FLOOR_OFFSETS = np.array([[x, y, 0.0] for x in (-0.6, 0.0, 0.6) for y in (-0.6, 0.6)])


class BodyRetargeter:
    def __init__(self, cfg: RetargetConfig | None = None, model=None, prefix=""):
        self.cfg = cfg = cfg or RetargetConfig()
        self.m = model or build_single_character_model()
        self.d = mujoco.MjData(self.m)
        self.ci = CharacterIndex(self.m, prefix)
        m = self.m
        self.n = m.nv
        self.K = len(self.ci.kp_sites)
        self.lb = np.full(m.nv, -np.inf); self.ub = np.full(m.nv, np.inf)
        self.hinge_dof, self.hinge_qadr = [], []
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
                da, qa = m.jnt_dofadr[j], m.jnt_qposadr[j]
                self.lb[da], self.ub[da] = m.jnt_range[j]
                self.hinge_dof.append(da); self.hinge_qadr.append(qa)
        self.hinge_dof = np.array(self.hinge_dof); self.hinge_qadr = np.array(self.hinge_qadr)
        # elbows and knees slightly bent: at exactly 0 the limb twist has zero gradient
        self.q_rest = m.qpos0.copy()
        for name, val in (("l_elbow", 0.35), ("r_elbow", 0.35), ("l_knee", 0.2), ("r_knee", 0.2)):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            self.q_rest[m.jnt_qposadr[j]] = val
        self.neutral_kp = self._fk_points(self.q_rest)
        self.A_bone, self.B_bone = bone_matrices(self.K)
        hand_kp = [KP[f"{s}_{n}"] for s in "lr" for n in ("index", "pinky")]
        self.kp_scale = np.ones(self.K); self.kp_scale[hand_kp] = cfg.hand_weight
        self.bone_scale = np.array([cfg.hand_weight if (isinstance(b, int) and b in hand_kp) else 1.0
                                    for _, b in BONES])
        self.sole_rest_z = self.neutral_kp[SOLE_KP, 2]
        n_col = cfg.max_collision_rows if (cfg.self_collision and len(self.ci.col_sites)) else 0
        self.qp = ParametricQP(self.n, 2 * len(self.ci.sole_sites), n_col, cfg.collision_soft_weight, cfg.step_size)
        self.n_col = n_col
        self._jp = np.zeros((3, m.nv))
        # the racket terms act only through the racket arm, so detection noise cannot shake the body
        self.rk_mask = np.ones(m.nv) if not cfg.racket_arm_only else np.zeros(m.nv)
        for nm in [] if not cfg.racket_arm_only else ("r_shoulder", "r_elbow", "r_pronation", "r_wrist_fe", "r_wrist_dev"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + nm)
            if j >= 0:
                k = 3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
                self.rk_mask[m.jnt_dofadr[j]:m.jnt_dofadr[j] + k] = 1.0
        # optional DOF set for racket targets from the pose prior only (cfg.racket_prior_dofs)
        self.rk_mask_wrist = np.zeros(m.nv)
        for nm in ("r_pronation", "r_wrist_fe", "r_wrist_dev"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + nm)
            if j >= 0:
                self.rk_mask_wrist[m.jnt_dofadr[j]] = 1.0
        if cfg.prewarm:
            self.qp.solve(np.eye(self.n), np.zeros(self.n), -np.ones(self.n), np.ones(self.n), [], [])
        self.reset()

    def reset(self):
        self.frame = 0
        self.adj = None
        self.q_prev = None
        self.S_prev = None
        self.P_prev = None
        self._sole_prev = None
        self._plants = {}
        self.v_prev = None
        self._adj_tip = False

    def _fk_points(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.ci.kp_sites].copy()

    def _points_jac(self):
        m, d = self.m, self.d
        P = d.site_xpos[self.ci.kp_sites].copy()
        J = np.empty((self.K, 3, self.n))
        for i, s in enumerate(self.ci.kp_sites):
            mujoco.mj_jacSite(m, d, self._jp, None, int(s))
            J[i] = self._jp
        return P, J

    def _seed(self, S):
        q = self.q_rest.copy()
        hips = S[[KP["l_hip"], KP["r_hip"]]]
        q[0:2] = hips.mean(0)[:2]
        right = hips[1] - hips[0]
        yaw = np.arctan2(right[1], right[0])
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        return q

    def _collision_rows(self):
        """Linearised gap rows a.dq >= -gap for close sphere pairs and spheres near the floor."""
        m, d, ci, cfg = self.m, self.d, self.ci, self.cfg
        Pc = d.site_xpos[ci.col_sites]
        i, j = ci.col_pairs.T
        diff = Pc[i] - Pc[j]
        dist = np.linalg.norm(diff, axis=1) + 1e-9
        gap = dist - ci.col_radius[i] - ci.col_radius[j] - cfg.collision_margin
        cand = [(g, 0, k) for k, g in enumerate(gap) if g < cfg.collision_activation]
        fl = ci.col_floor
        gz = Pc[fl, 2] - ci.col_radius[fl] - cfg.collision_margin
        cand += [(g, 1, k) for k, g in enumerate(gz) if g < cfg.collision_activation]
        cand.sort()
        cand = cand[:self.n_col]
        jac = {}

        def J(s):
            if s not in jac:
                mujoco.mj_jacSite(m, d, self._jp, None, int(ci.col_sites[s]))
                jac[s] = self._jp.copy()
            return jac[s]

        rows = []
        for g, kind, k in cand:
            if kind == 0:
                nrm = diff[k] / dist[k]
                rows.append((nrm @ (J(i[k]) - J(j[k])), -g))
            else:
                rows.append((J(fl[k])[2].copy(), -g))
        return rows

    def penetration(self, qpos):
        """Deepest sphere-pair penetration (m, >= 0) and number of penetrating pairs."""
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        ci = self.ci
        if not len(ci.col_pairs):
            return 0.0, 0
        Pc = self.d.site_xpos[ci.col_sites]
        i, j = ci.col_pairs.T
        gap = np.linalg.norm(Pc[i] - Pc[j], axis=1) - ci.col_radius[i] - ci.col_radius[j]
        return float(max(0.0, -gap.min())), int((gap < -0.005).sum())

    def step(self, S, vis=None, dt=1 / 30, contact=None, anchors=None, bounds=None, obs2d=None,
             q_ref=None, ref_weight=0.0, racket=None):
        """S: (23,3) target keypoints (world, character scale); contact: optional (4,) flags
        for SOLE_KP from the source (otherwise detected from S).
        anchors: optional (idx, targets, weights) with idx into ci.sole_sites (8 sole corners).
        bounds: optional dict(root=(xmin, xmax, ymin, ymax), sole_ymax=float) court limits.
        obs2d: optional (uv (K,2) px, weights (K,), camera with Kmat/R/t): image reprojection term.
        q_ref: optional reference pose (e.g. offline-smoothed); used as seed and pulled toward with
        ref_weight, so the solve only re-imposes contacts / floor / collisions / bounds on it.
        racket: optional dict(tip=(3,) source racket tip or None, w=confidence, uv=(2,) detected tip
        px or None, cam=camera, prior_only=bool): the racket tip as an extra joint (mesh vertex +
        wrist->tip direction + robust image term); prior_only targets (no detection) only steer the
        wrist DOFs and stay out of the mesh.
        Returns dict with qpos."""
        cfg, m, d = self.cfg, self.m, self.d
        t0 = time.perf_counter()
        K, n = self.K, self.n
        vis = np.ones(K) if vis is None else np.clip(np.asarray(vis, float), 0.05, 1.0)
        S = np.asarray(S, float)
        rk_on = racket is not None and cfg.racket_weight > 0 and self.ci.racket_tip >= 0
        tip_src = racket.get("tip") if rk_on else None
        rk_w = float(racket.get("w", 1.0)) if rk_on else 0.0
        has_tip = tip_src is not None and rk_w > 0 and np.all(np.isfinite(tip_src))
        rk_uv = racket.get("uv") if rk_on else None
        if rk_uv is not None and not np.all(np.isfinite(rk_uv)):
            rk_uv = None
        prior_only = bool(racket.get("prior_only", False)) if rk_on else False
        in_mesh = has_tip and cfg.racket_in_mesh and (cfg.racket_prior_in_mesh or not prior_only)
        rk_dir_mask = self.rk_mask_wrist if prior_only and cfg.racket_prior_dofs == "wrist" else self.rk_mask
        Kv = K + int(in_mesh)                       # mesh vertices from tracked points

        if cfg.use_laplacian:
            if cfg.floor_points:
                fc = S[SOLE_KP].mean(0); fc[2] = 0.0
                F = fc + FLOOR_OFFSETS
            else:
                F = np.zeros((0, 3))
            V_src = np.vstack(([S, np.asarray(tip_src)[None]] if in_mesh else [S]) + [F])
            if self.adj is None or self.frame % cfg.topology_reuse_frames == 0 or in_mesh != self._adj_tip:
                self.adj = delaunay_adjacency(V_src)
                self._adj_tip = in_mesh
            L = laplacian_matrix(V_src, self.adj, cfg.laplacian_kappa)
            delta = L @ V_src
            w_rows = np.sqrt(cfg.laplacian_weight) * np.r_[vis, [cfg.racket_weight * rk_w] * int(in_mesh),
                                                           np.ones(len(F))]
        bv = np.minimum(self.A_bone @ vis, self.B_bone @ vis) * self.bone_scale
        u_src = self.B_bone @ S - self.A_bone @ S
        u_src /= np.linalg.norm(u_src, axis=1, keepdims=True) + 1e-9
        hip_sel = np.zeros(K); hip_sel[[KP["l_hip"], KP["r_hip"]]] = 0.5
        root_src = hip_sel @ S

        first = self.q_prev is None
        lock = np.zeros(0, int)
        if anchors is None and not first and cfg.contact_lock_weight > 0:
            # online footplants: a planted sole point is anchored where it was at plant onset (no drift)
            sole = np.array(SOLE_KP)
            if contact is None:
                v = np.linalg.norm(S[sole, :2] - self.S_prev[sole, :2], axis=1) / max(dt, 1e-3)
                contact = (S[sole, 2] - self.sole_rest_z < cfg.contact_height_m) & (v < cfg.contact_speed_mps)
            contact = np.asarray(contact, bool)
            idx, tgt, wts = [], [], []
            for c in range(4):
                if not contact[c]:
                    self._plants.pop(c, None)
                    continue
                if c not in self._plants:
                    p = self._sole_prev[ONLINE_SOLE[c]].copy()
                    p[:, 2] = np.minimum(p[:, 2], 0.0)
                    self._plants[c] = [p, 0]
                p, age = self._plants[c]
                self._plants[c][1] += 1
                idx += ONLINE_SOLE[c]; tgt += list(p); wts += [min(1.0, (age + 1) / 3)] * 2
            if idx:
                anchors = (np.array(idx), np.array(tgt), np.array(wts) * cfg.contact_lock_weight / cfg.anchor_weight)
            lock = sole[contact]
        has_anchor = anchors is not None and len(anchors[0]) > 0
        if q_ref is not None:
            q = np.asarray(q_ref, float).copy()
        else:
            q = self._seed(S) if first else self.q_prev.copy()
        if bounds is not None and "root" in bounds:
            x0, x1, y0, y1 = bounds["root"]
            q[0], q[1] = np.clip(q[0], x0, x1), np.clip(q[1], y0, y1)
        q_prev = q.copy()
        iters = cfg.sqp_iters_first if first else cfg.sqp_iters
        e_prev = np.zeros(n); e_rest = np.zeros(n)
        n_done, success = 0, True
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            P, J = self._points_jac()
            if rk_on:
                Ptip = d.site_xpos[self.ci.racket_tip].copy()
                mujoco.mj_jacSite(m, d, self._jp, None, int(self.ci.racket_tip))
                Jtip = self._jp.copy()
            rows_J, rows_r = [], []
            if cfg.use_laplacian:
                Pv, Jv = (np.vstack([P, Ptip[None]]), np.concatenate([J, Jtip[None]])) if in_mesh else (P, J)
                V = np.vstack([Pv, F])
                r = (L @ V - delta) * w_rows[:, None]
                Jl = np.einsum("ik,kjn->ijn", L[:, :Kv], Jv) * w_rows[:, None, None]
                rows_J.append(Jl.reshape(-1, n)); rows_r.append(r.ravel())
            if has_tip:
                # wrist -> racket tip direction, like a bone
                wj = KP["r_wrist"]
                dv_t = Ptip - P[wj]; ln_t = np.linalg.norm(dv_t) + 1e-9; u_t = dv_t / ln_t
                u_s = tip_src - S[wj]; u_s = u_s / (np.linalg.norm(u_s) + 1e-9)
                Pj_t = (np.eye(3) - np.outer(u_t, u_t)) / ln_t
                wt = np.sqrt(cfg.bone_direction_weight * cfg.racket_weight * rk_w)
                rows_J.append(wt * Pj_t @ ((Jtip - J[wj]) * rk_dir_mask)); rows_r.append(wt * (u_t - u_s))
            if rk_uv is not None and cfg.racket_reproj_weight > 0:
                cam_r = racket["cam"]
                Xc = cam_r.R @ Ptip + cam_r.t
                fx, fy = cam_r.Kmat[0, 0], cam_r.Kmat[1, 1]
                uv_t = Xc[:2] / Xc[2] * [fx, fy] + cam_r.Kmat[:2, 2]
                Dt = np.stack([fx / Xc[2] * (cam_r.R[0] - Xc[0] / Xc[2] * cam_r.R[2]),
                               fy / Xc[2] * (cam_r.R[1] - Xc[1] / Xc[2] * cam_r.R[2])])
                e = uv_t - np.asarray(rk_uv, float)
                # Geman-McClure IRLS weight: false detections get bounded influence
                robust = 1.0 / (1.0 + (e @ e) / cfg.racket_reproj_sigma_px ** 2) ** 2
                w2t = np.sqrt(cfg.racket_reproj_weight * rk_w * robust)
                rows_J.append(w2t * Dt @ (Jtip * self.rk_mask)); rows_r.append(w2t * e)
            wk = np.sqrt(cfg.keypoint_position_weight) * vis * self.kp_scale
            rows_J.append((J * wk[:, None, None]).reshape(-1, n)); rows_r.append(((P - S) * wk[:, None]).ravel())
            if obs2d is not None and cfg.reproj_weight > 0:
                uv_o, w_o, cam = obs2d
                Kc, Rc, tc = cam.Kmat, cam.R, cam.t
                Xc = P @ Rc.T + tc
                z = Xc[:, 2:3]
                uv = Xc[:, :2] / z * [Kc[0, 0], Kc[1, 1]] + Kc[:2, 2]
                du = Kc[0, 0] / z * (Rc[0][None] - Xc[:, 0:1] / z * Rc[2][None])     # (K,3)
                dv = Kc[1, 1] / z * (Rc[1][None] - Xc[:, 1:2] / z * Rc[2][None])
                D = np.stack([du, dv], 1)                                            # (K,2,3)
                w2 = np.sqrt(cfg.reproj_weight * np.asarray(w_o, float))
                rows_J.append((np.einsum("kab,kbn->kan", D, J) * w2[:, None, None]).reshape(-1, n))
                rows_r.append(((uv - uv_o) * w2[:, None]).ravel())
            if cfg.racket_front_weight > 0:
                # racket head in front of the torso's frontal plane unless the racket hand is above the
                # shoulder (overhead backswing); arm twist and pronation are invisible in the image
                up = P[[KP["l_shoulder"], KP["r_shoulder"]]].mean(0) - P[[KP["l_hip"], KP["r_hip"]]].mean(0)
                fwd = np.cross(up, P[KP["r_shoulder"]] - P[KP["l_shoulder"]]); fwd[2] = 0
                fwd /= np.linalg.norm(fwd) + 1e-9
                tc = P[[KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]].mean(0)
                lift = P[KP["r_wrist"], 2] - P[KP["r_shoulder"], 2]
                gate = float(np.clip((0.0 - lift) / 0.15, 0.0, 1.0))
                if has_tip and not (prior_only and cfg.racket_prior_keep_front):
                    gate *= max(0.0, 1.0 - rk_w)    # an observed racket overrides the prior
                if gate > 0:
                    prk = d.site_xpos[self.ci.racket_site]
                    gap = (prk - tc) @ fwd - cfg.racket_front_margin
                    if gap < 0:
                        mujoco.mj_jacSite(m, d, self._jp, None, int(self.ci.racket_site))
                        wf = np.sqrt(cfg.racket_front_weight * gate)
                        rows_J.append(wf * (fwd @ self._jp)[None]); rows_r.append(np.array([wf * gap]))
            wr = np.sqrt(cfg.root_position_weight)
            rows_J.append(wr * np.einsum("k,kjn->jn", hip_sel, J)); rows_r.append(wr * (hip_sel @ P - root_src))
            # bone directions
            dv = self.B_bone @ P - self.A_bone @ P
            Jd = np.einsum("bk,kjn->bjn", self.B_bone - self.A_bone, J)
            ln = np.linalg.norm(dv, axis=1) + 1e-9
            u = dv / ln[:, None]
            Pj = (np.eye(3)[None] - u[:, :, None] * u[:, None, :]) / ln[:, None, None]
            wb = np.sqrt(cfg.bone_direction_weight * bv)
            rows_J.append((np.einsum("bij,bjn->bin", Pj, Jd) * wb[:, None, None]).reshape(-1, n))
            rows_r.append(((u - u_src) * wb[:, None]).ravel())
            Jt = np.vstack(rows_J); rt = np.concatenate(rows_r)
            H = Jt.T @ Jt
            g = Jt.T @ rt
            mujoco.mj_differentiatePos(m, e_prev, 1.0, q_prev, q)
            mujoco.mj_differentiatePos(m, e_rest, 1.0, self.q_rest, q)
            w_seed = cfg.smooth_weight + (ref_weight if q_ref is not None else 0.0)
            H[np.diag_indices(n)] += w_seed
            g += w_seed * e_prev
            if cfg.accel_weight > 0 and self.v_prev is not None:
                H[np.diag_indices(n)] += cfg.accel_weight
                g += cfg.accel_weight * (e_prev - self.v_prev)
            H[np.arange(6, n), np.arange(6, n)] += cfg.rest_weight
            g[6:] += cfg.rest_weight * e_rest[6:]
            if first:
                H[np.diag_indices(n)] += 1e-3
            Js = np.empty((len(self.ci.sole_sites), 3, n))
            for k, s in enumerate(self.ci.sole_sites):
                mujoco.mj_jacSite(m, d, self._jp, None, int(s))
                Js[k] = self._jp
            Ps = d.site_xpos[self.ci.sole_sites]
            if has_anchor:
                ai, at, aw = anchors
                wa = np.sqrt(cfg.anchor_weight * np.asarray(aw, float))
                Ja = Js[ai] * wa[:, None, None]
                ra = (Ps[ai] - at) * wa[:, None]
                H += np.einsum("kin,kim->nm", Ja, Ja)
                g += np.einsum("kin,ki->n", Ja, ra)
            hard = []
            if cfg.floor_constraint:
                hard += [(Js[k, 2].copy(), -Ps[k, 2]) for k in range(len(Ps))]
            lower = np.full(n, -np.inf); upper = np.full(n, np.inf)
            lower[self.hinge_dof] = self.lb[self.hinge_dof] - q[self.hinge_qadr]
            upper[self.hinge_dof] = self.ub[self.hinge_dof] - q[self.hinge_qadr]
            if bounds is not None:
                if "root" in bounds:
                    x0, x1, y0, y1 = bounds["root"]
                    lower[0], upper[0] = x0 - q[0], x1 - q[0]
                    lower[1], upper[1] = y0 - q[1], y1 - q[1]
                    lower[:2] = np.minimum(lower[:2], 0); upper[:2] = np.maximum(upper[:2], 0)
                if "sole_ymax" in bounds:
                    ym = bounds["sole_ymax"]
                    hard += [(-Js[k, 1].copy(), Ps[k, 1] - ym) for k in range(len(Ps)) if Ps[k, 1] > ym - 0.3]
            soft = self._collision_rows() if self.n_col else []
            dq, _ = self.qp.solve(H, g, lower, upper, hard, soft)
            if dq is None:
                success = False
                break
            mujoco.mj_integratePos(m, q, dq, 1.0)
            mujoco.mj_normalizeQuat(m, q)
            q[self.hinge_qadr] = np.clip(q[self.hinge_qadr], self.lb[self.hinge_dof], self.ub[self.hinge_dof])
            n_done += 1
        if self.q_prev is not None:
            self.v_prev = np.zeros(n)
            mujoco.mj_differentiatePos(m, self.v_prev, 1.0, self.q_prev, q)
        self.q_prev = q.copy()
        self.S_prev = S.copy()
        self.P_prev = self._fk_points(q)
        self._sole_prev = self.d.site_xpos[self.ci.sole_sites].copy()
        self.frame += 1
        lat = (time.perf_counter() - t0) * 1e3
        return dict(qpos=q, latency_ms=lat, success=success, iters=n_done, locked=len(lock))

    def evaluate(self, qpos, S, vis):
        """Keypoint error (m), bone direction error (deg), racket centre, sole height."""
        P = self._fk_points(qpos)
        err = np.linalg.norm(P - S, axis=1)
        u_s = self.B_bone @ S - self.A_bone @ S
        u_r = self.B_bone @ P - self.A_bone @ P
        cosang = np.sum(u_s * u_r, 1) / (np.linalg.norm(u_s, axis=1) * np.linalg.norm(u_r, axis=1) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        return dict(kp=P, kp_err=err, bone_err_deg=ang,
                    racket=self.d.site_xpos[self.ci.racket_site].copy(),
                    racket_mat=self.d.site_xmat[self.ci.racket_site].reshape(3, 3).copy(),
                    sole=self.d.site_xpos[self.ci.sole_sites].copy())

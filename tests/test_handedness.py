"""Left-handed character = mirror image of the right-handed one.

Run:  python tests/test_handedness.py
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from badrt.character import (KP_NAMES, MIRROR_PERM, CharacterIndex, build_single_character_model,  # noqa: E402
                             mirror_qpos)

M = np.diag([-1.0, 1.0, 1.0])


def random_pose(m, rng):
    q = np.zeros(m.nq)
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            q[a:a + 3] = rng.uniform([-2, -2, 0.7], [2, 2, 1.3])
            v = rng.normal(size=4); q[a + 3:a + 7] = v / np.linalg.norm(v)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            v = rng.normal(size=4) * [1, 0.5, 0.5, 0.5] + [1, 0, 0, 0]; q[a:a + 4] = v / np.linalg.norm(v)
        else:
            q[a] = rng.uniform(*m.jnt_range[j])
    return q


def test_mirror():
    mr = build_single_character_model(handed="right")
    ml = build_single_character_model(handed="left")
    ir, il = CharacterIndex(mr), CharacterIndex(ml)
    assert ir.hand == "r" and il.hand == "l"
    assert mr.nq == ml.nq == 45
    dr, dl = mujoco.MjData(mr), mujoco.MjData(ml)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = random_pose(mr, rng)
        dr.qpos[:] = q; mujoco.mj_kinematics(mr, dr)
        dl.qpos[:] = mirror_qpos(q); mujoco.mj_kinematics(ml, dl)
        # keypoints: mirrored and left/right swapped
        kr = dr.site_xpos[ir.kp_sites] @ M
        kl = dl.site_xpos[il.kp_sites][MIRROR_PERM]
        worst = max(worst, np.abs(kr - kl).max())
        # racket: centre, tip and frame (face normal up to sign)
        for a, b in ((ir.racket_site, il.racket_site), (ir.racket_tip, il.racket_tip)):
            worst = max(worst, np.abs(dr.site_xpos[a] @ M - dl.site_xpos[b]).max())
        Rr = M @ dr.site_xmat[ir.racket_site].reshape(3, 3) @ M          # proper rotation again
        Rl = dl.site_xmat[il.racket_site].reshape(3, 3)
        worst = max(worst, np.abs(Rr - Rl).max())
        # involution
        worst = max(worst, np.abs(mirror_qpos(mirror_qpos(q)) - q).max())
    assert worst < 1e-9, worst
    print(f"mirror_qpos: keypoints / racket / involution error {worst:.2e}  ({len(KP_NAMES)} keypoints, 200 poses)")


if __name__ == "__main__":
    test_mirror()
    print("ok")

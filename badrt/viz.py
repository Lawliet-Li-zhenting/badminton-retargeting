"""Offscreen MuJoCo rendering helpers."""
from __future__ import annotations

import mujoco
import numpy as np


class SceneRenderer:
    def __init__(self, m, width=640, height=480):
        self.m = m
        self.r = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.0, -90.0, -10.0
        self.cam.lookat[:] = [0, 0, 1.0]
        self.opt = mujoco.MjvOption()
        self._lookat = None

    def follow(self, target, alpha=0.15):
        target = np.asarray(target, float)
        self._lookat = target if self._lookat is None else (1 - alpha) * self._lookat + alpha * target
        self.cam.lookat[:] = self._lookat

    def set_view(self, view):
        """view: dict(lookat, distance, azimuth, elevation[, fovy]) as returned by court.game_view."""
        self.cam.lookat[:] = view["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = view["distance"], view["azimuth"], view["elevation"]
        if "fovy" in view:
            self.m.vis.global_.fovy = view["fovy"]

    def render(self, d, points=None, point_rgba=(1, 0.1, 0.1, 1), radius=0.025, lines=None, line_rgba=(1, 1, 0, 1),
               decor=None):
        """decor(scene): adds its own geoms (effects) to the scene before it is drawn."""
        mujoco.mj_camlight(self.m, d)
        self.r.update_scene(d, self.cam, self.opt)
        scn = self.r.scene
        if decor is not None:
            decor(scn)
        if points is not None:
            for p in np.asarray(points):
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    [radius, 0, 0], p, np.eye(3).ravel(), np.asarray(point_rgba, np.float32))
                scn.ngeom += 1
        if lines is not None:
            for a, b in lines:
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.asarray(line_rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, np.asarray(a, float), np.asarray(b, float))
                scn.ngeom += 1
        return self.r.render()[:, :, ::-1].copy()       # BGR for OpenCV

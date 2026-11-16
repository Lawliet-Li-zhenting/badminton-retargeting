"""Humanoid character (built with MjSpec) holding a badminton racket in the right hand.

World frame: z up. In its rest pose the character stands at the origin facing +y with
its right side towards +x, arms hanging with palms facing the thighs.

The 23 keypoints are MediaPipe / BlazePose body landmarks (fingers excluded: only the
index and pinky knuckles are used to orient the hand). Every keypoint is a site named
``{prefix}kp_{name}`` so positions and Jacobians come straight from MuJoCo.

Joint layout (nq = 45, nv = 38):
    pelvis free joint, ball joints for lumbar / neck / shoulders / hips,
    hinges for elbows, forearm pronation + wrist (2), knees and ankles (2).
"""
from __future__ import annotations

import mujoco
import numpy as np

KP_NAMES = [
    "nose", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_pinky", "r_pinky", "l_index", "r_index",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_heel", "r_heel", "l_toe", "r_toe",
]
KP = {n: i for i, n in enumerate(KP_NAMES)}
# MediaPipe pose landmark index of every keypoint
MP_IDS = np.array([0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
# left/right swap permutation (used when mirroring a left-handed player)
MIRROR_PERM = np.array([KP[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")]
                        for n in KP_NAMES])
SOLE_KP = [KP["l_heel"], KP["r_heel"], KP["l_toe"], KP["r_toe"]]

# skeleton connections for drawing
BONE_DRAW = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_shoulder", "l_hip"),
             ("r_shoulder", "r_hip"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
             ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_wrist", "l_index"),
             ("l_wrist", "l_pinky"), ("r_wrist", "r_index"), ("r_wrist", "r_pinky"),
             ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"),
             ("r_knee", "r_ankle"), ("l_ankle", "l_heel"), ("l_heel", "l_toe"),
             ("l_ankle", "l_toe"), ("r_ankle", "r_heel"), ("r_heel", "r_toe"),
             ("r_ankle", "r_toe"), ("l_ear", "r_ear"), ("nose", "l_ear"), ("nose", "r_ear")]

# racket (metres, racket frame: x = string-face normal, z = shaft towards the head)
RACKET = dict(
    grip_in_hand=np.array([0.0, 0.0, -0.07]),
    shaft_angle_deg=55.0,            # angle between the shaft and the finger direction
    head_center=0.46,                # grip point -> head centre along the shaft
    head_half_length=0.115,          # along the shaft
    head_half_width=0.100,
    butt=-0.12,
)

D2R = np.pi / 180.0


def _quat_from_axes(x, z):
    x = np.asarray(x, float); z = np.asarray(z, float)
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.ravel())
    return q


def _capsule(body, p0, p1, r, rgba, name=None, group=0):
    g = body.add_geom()
    if name:
        g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = np.r_[p0, p1]
    g.size = [r, 0, 0]
    g.rgba = rgba
    g.contype = g.conaffinity = 0
    g.group = group
    return g


def _geom(body, gtype, size, pos, rgba, quat=None, group=0):
    g = body.add_geom()
    g.type = gtype
    g.size = size
    g.pos = pos
    if quat is not None:
        g.quat = quat
    g.rgba = rgba
    g.contype = g.conaffinity = 0
    g.group = group
    return g


def _site(body, name, pos):
    s = body.add_site()
    s.name = name
    s.pos = pos
    s.size = [0.012, 0, 0]
    s.rgba = [0, 0, 0, 0]
    s.group = 4
    return s


def _ball(body, name):
    j = body.add_joint()
    j.name = name
    j.type = mujoco.mjtJoint.mjJNT_BALL
    return j


def _hinge(body, name, axis, lo_deg, hi_deg):
    j = body.add_joint()
    j.name = name
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = axis
    j.range = [lo_deg, hi_deg]               # spec.compiler.degree must be True
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    return j


def _col(body, name, pos, r):
    """Self-collision sphere (invisible site, radius in size[0])."""
    s = body.add_site()
    s.name = name
    s.pos = pos
    s.size = [r, 0, 0]
    s.rgba = [1, 0, 1, 0.25]
    s.group = 5
    return s


class _Paint:
    """Per-character materials (specular highlights) when a spec is available, plain rgba otherwise."""

    # roles that take another role's colour unless an outfit says otherwise (long sleeves, trousers, gloves)
    FALLBACK = {"sleeve": "shirt", "arm": "skin", "leg": "skin", "hand": "skin", "face": "skin", "band": "trim"}
    SHINE = {"skin": (0.25, 0.3), "hair": (0.35, 0.6), "racket": (0.8, 0.9), "metal": (0.9, 0.95), "shoe": (0.4, 0.5),
             "sole": (0.1, 0.2), "strings": (0.3, 0.5)}

    def __init__(self, spec, prefix, colors):
        self.spec, self.p, self.colors = spec, prefix, colors
        self.made = set()
        self.shine = {}                           # role -> (specular, shininess) set by an outfit
        self.tags = {}                            # tag -> geoms (what an outfit may hide)
        self.parts = {}                           # body name without the prefix -> body

    def __call__(self, g, role, tag=None):
        base = role if role in self.colors else self.FALLBACK[role]
        rgba = self.colors[base]
        if tag is not None:
            self.tags.setdefault(tag, []).append(g)
        if self.spec is None:
            g.rgba = rgba
            return g
        name = f"{self.p}mat_{role}"
        if name not in self.made:
            mat = self.spec.add_material()
            mat.name = name
            mat.rgba = rgba
            mat.specular, mat.shininess = self.shine.get(role) or self.SHINE.get(base, (0.15, 0.2))
            self.made.add(name)
        g.material = name
        g.rgba = [0.5, 0.5, 0.5, 1]               # default rgba -> the material colour is used
        return g

    def hide(self, *tags):
        for t in tags:
            for g in self.tags.get(t, []):
                g.group = 4                       # (groups 3-5 are not drawn)


E, C, S, B = (mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE,
              mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX)


def _g(body, paint, role, gtype, size, pos=(0, 0, 0), quat=None, fromto=None, tag=None):
    g = body.add_geom()
    g.type = gtype
    if fromto is not None:
        g.fromto = np.r_[fromto]
        g.size = [size, 0, 0]
    else:
        g.size = np.r_[size, np.zeros(3 - len(size))]
        g.pos = pos
    if quat is not None:
        g.quat = quat
    g.contype = g.conaffinity = 0
    return paint(g, role, tag)


DEFAULT_DIMS = dict(thigh=1.0, shin=1.0, upperarm=1.0, forearm=1.0, torso=1.0, shoulder=1.0, hip=1.0, girth=1.0)


def _scale_body(body, fr, fz, children=True):
    """Scale a body's geoms / sites: x, y (radial) by fr, z (along the segment) by fz; child body
    offsets too. Bodies named *racket keep their own geometry."""
    sv = np.array([fr, fr, fz])
    for g in body.geoms:
        ft = np.asarray(g.fromto, float)
        if (g.type == mujoco.mjtGeom.mjGEOM_CAPSULE and np.any(ft != 0)) or (np.all(np.isfinite(ft)) and np.any(ft != 0)):
            ft = ft.copy()
            g.fromto = np.r_[ft[:3] * sv, ft[3:] * sv]
            g.size = [g.size[0] * fr, 0, 0]
        else:
            g.pos = np.asarray(g.pos) * sv
            if g.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                g.size = [g.size[0] * fr, 0, 0]
            else:
                g.size = np.asarray(g.size) * sv
    for s in body.sites:
        s.pos = np.asarray(s.pos) * sv
        if s.group == 5:
            s.size = [s.size[0] * fr, 0, 0]
    if children:
        for b in body.bodies:
            b.pos = np.asarray(b.pos) * sv


def personalize(dims_src):
    """Character segment scale factors from source bone lengths (metres, medians), clamped to +-20%."""
    ref = dict(thigh=0.44, shin=0.43, upperarm=0.29, forearm=0.26, torso=0.49, shoulder=0.38, hip=0.18)
    out = dict(DEFAULT_DIMS)
    for k, v in dims_src.items():
        if k in ref and np.isfinite(v):
            out[k] = float(np.clip(v / ref[k], 0.8, 1.2))
    # girth follows overall size (mean of the long-bone factors), narrower range
    out["girth"] = float(np.clip(np.mean([out[k] for k in ("thigh", "shin", "upperarm", "forearm")]), 0.88, 1.1))
    return out


def add_character(parent, prefix="", shirt=(0.15, 0.35, 0.8, 1), shorts=(0.1, 0.1, 0.15, 1),
                  skin=(0.93, 0.75, 0.6, 1), racket=True, racket_color=(0.9, 0.2, 0.1, 1), spec=None,
                  hair=(0.08, 0.06, 0.05, 1), shoe=(0.96, 0.96, 0.97, 1), accent=None, dims=None,
                  handed="right", outfit=None):
    """handed: which hand holds the racket ("right" | "left"). The body is left/right symmetric,
    so a left-hander's pose is the mirror image of a right-hander's (see mirror_qpos).
    outfit: wardrobe.py dict (slot -> item): colours, things worn, racket skin, build. Looks only: the
    joints, sites and the racket the game computes with are the same whatever is worn."""
    look = None
    if outfit:
        from . import wardrobe
        look = wardrobe.Look(outfit)
    pelvis, paint = _add_character(parent, prefix, shirt, shorts, skin, racket, racket_color, spec, hair, shoe, accent,
                                   handed[0], look)
    if look is not None:
        look.dress(paint, handed[0])
        dims = dict(DEFAULT_DIMS, **(dims or {}))
        dims["girth"] = dims["girth"] * look.girth
        if look.head != 1.0:
            _scale_body(paint.parts["head"], look.head, look.head, children=False)
    if dims is None:
        return pelvis
    D = dict(DEFAULT_DIMS, **dims)
    gi = D["girth"]
    name = lambda b: b.name[len(prefix):]
    _scale_body(pelvis, gi * D["hip"], gi, children=False)
    for b in pelvis.bodies:
        if name(b) == "torso":
            _scale_body(b, gi, D["torso"], children=False)
            for c in b.bodies:
                if name(c) == "head":
                    c.pos = np.asarray(c.pos) * [1, 1, D["torso"]]
                else:                                            # upper arms
                    c.pos = np.asarray(c.pos) * [D["shoulder"], 1, D["torso"]]
                    _scale_body(c, gi, D["upperarm"])
                    fa = c.bodies[0]
                    _scale_body(fa, gi, D["forearm"])
                    _scale_body(fa.bodies[0], gi, 1.0, children=False)        # hand (keep racket grip offset)
        else:                                                    # thighs
            b.pos = np.asarray(b.pos) * [D["hip"], 1, 1]
            _scale_body(b, gi, D["thigh"])
            _scale_body(b.bodies[0], gi, D["shin"])
    pelvis.pos = [0, 0, 0.03 + 0.44 * D["thigh"] + 0.43 * D["shin"] + 0.08]
    return pelvis


def _add_character(parent, prefix, shirt, shorts, skin, racket, racket_color, spec, hair, shoe, accent,
                   racket_side="r", look=None):
    p = prefix
    accent = accent or racket_color
    paint = _Paint(spec, p, dict(shirt=shirt, shorts=shorts, skin=skin, hair=hair, shoe=shoe,
                                 sole=(0.85, 0.3, 0.15, 1) if accent is None else accent,
                                 sock=(0.97, 0.97, 0.97, 1), racket=racket_color, metal=(0.2, 0.2, 0.22, 1),
                                 grip=(0.95, 0.95, 0.95, 1), strings=(1, 1, 1, 0.45), eye=(0.1, 0.1, 0.12, 1),
                                 trim=(1, 1, 1, 1)))
    if look is not None:
        look.paint(paint)
    pelvis = parent.add_body()
    pelvis.name = p + "pelvis"
    pelvis.pos = [0, 0, 0.98]
    j = pelvis.add_freejoint()
    j.name = p + "root"
    _g(pelvis, paint, "shorts", E, [0.155, 0.105, 0.095], [0, -0.005, -0.015])
    _g(pelvis, paint, "shirt", E, [0.14, 0.095, 0.05], [0, 0, 0.06])                 # shirt hem
    for x in (-1, 1):
        _col(pelvis, f"{p}col_pelvis_{'l' if x < 0 else 'r'}", [0.06 * x, 0, 0.0], 0.095)

    torso = pelvis.add_body(); torso.name = p + "torso"; torso.pos = [0, 0, 0.07]
    _ball(torso, p + "lumbar")
    _g(torso, paint, "shirt", E, [0.125, 0.09, 0.13], [0, 0, 0.09])                  # waist
    _g(torso, paint, "shirt", E, [0.165, 0.105, 0.14], [0, 0.005, 0.26])             # chest
    for x in (-1, 1):
        _g(torso, paint, "shirt", E, [0.06, 0.058, 0.05], [0.16 * x, 0, 0.365])        # deltoids
    _g(torso, paint, "skin", C, 0.045, fromto=[0, 0.0, 0.38, 0, 0.0, 0.45])           # neck base
    _col(torso, p + "col_torso_low", [0, 0, 0.08], 0.12)
    for x in (-1, 1):
        _col(torso, f"{p}col_torso_{'l' if x < 0 else 'r'}", [0.075 * x, 0.0, 0.26], 0.11)
    _col(torso, p + "col_torso_up", [0, 0, 0.32], 0.11)

    head = torso.add_body(); head.name = p + "head"; head.pos = [0, 0, 0.43]
    _ball(head, p + "neck")
    _g(head, paint, "skin", C, 0.043, fromto=[0, 0, 0, 0, 0.01, 0.08])
    _g(head, paint, "face", E, [0.085, 0.095, 0.108], [0, 0.005, 0.15])               # skull
    _g(head, paint, "face", E, [0.06, 0.06, 0.05], [0, 0.04, 0.1])                     # jaw
    _g(head, paint, "hair", E, [0.092, 0.1, 0.085], [0, -0.012, 0.185], tag="hair")   # hair cap
    _g(head, paint, "hair", E, [0.06, 0.04, 0.07], [0, -0.075, 0.13], tag="hair")     # back hair
    _g(head, paint, "hair", E, [0.035, 0.03, 0.06], [0, -0.115, 0.08], tag="ponytail")
    _g(head, paint, "band", E, [0.094, 0.1, 0.016], [0, 0.0, 0.205], tag="band")      # headband
    for x in (-1, 1):
        _g(head, paint, "eye", S, [0.011], [0.032 * x, 0.093, 0.165], tag="eyes")
        _g(head, paint, "face", E, [0.012, 0.02, 0.026], [0.086 * x, -0.005, 0.15], tag="ears")
    _g(head, paint, "face", E, [0.009, 0.012, 0.016], [0, 0.098, 0.142], tag="nose")
    _site(head, p + "kp_nose", [0, 0.105, 0.15])
    _site(head, p + "kp_l_ear", [-0.078, -0.005, 0.15])
    _site(head, p + "kp_r_ear", [0.078, -0.005, 0.15])
    _col(head, p + "col_head", [0, 0.01, 0.15], 0.11)

    for side, sx in (("l", -1.0), ("r", 1.0)):
        ua = torso.add_body(); ua.name = f"{p}{side}_upperarm"; ua.pos = [0.19 * sx, 0, 0.38]
        _ball(ua, f"{p}{side}_shoulder")
        _g(ua, paint, "sleeve", C, 0.052, fromto=[0, 0, 0, 0, 0, -0.1])                # sleeve
        _g(ua, paint, "trim", C, 0.053, fromto=[0, 0, -0.1, 0, 0, -0.11], tag="cuffs")
        _g(ua, paint, "arm", E, [0.043, 0.046, 0.12], [0, 0, -0.16])                   # biceps
        _g(ua, paint, "arm", S, [0.036], [0, 0, -0.28])                                 # elbow
        _site(ua, f"{p}kp_{side}_shoulder", [0, 0, 0])
        _col(ua, f"{p}col_{side}_ua_0", [0, 0, -0.09], 0.05)
        _col(ua, f"{p}col_{side}_ua_1", [0, 0, -0.21], 0.043)
        fa = ua.add_body(); fa.name = f"{p}{side}_forearm"; fa.pos = [0, 0, -0.29]
        _hinge(fa, f"{p}{side}_elbow", [1, 0, 0], 0, 150)
        _g(fa, paint, "arm", E, [0.04, 0.038, 0.1], [0, 0, -0.085])                    # forearm muscle
        _g(fa, paint, "arm", C, 0.026, fromto=[0, 0, -0.1, 0, 0, -0.245])
        _g(fa, paint, "trim", C, 0.03, fromto=[0, 0, -0.205, 0, 0, -0.235], tag="cuffs")   # wristband
        _site(fa, f"{p}kp_{side}_elbow", [0, 0, 0])
        _col(fa, f"{p}col_{side}_fa_0", [0, 0, -0.07], 0.042)
        _col(fa, f"{p}col_{side}_fa_1", [0, 0, -0.19], 0.032)
        hand = fa.add_body(); hand.name = f"{p}{side}_hand"; hand.pos = [0, 0, -0.26]
        _hinge(hand, f"{p}{side}_pronation", [0, 0, sx], -85, 85)
        _hinge(hand, f"{p}{side}_wrist_fe", [0, sx, 0], -75, 75)
        _hinge(hand, f"{p}{side}_wrist_dev", [1, 0, 0], -25, 35)
        _g(hand, paint, "hand", E, [0.02, 0.043, 0.045], [0, 0.0, -0.045])             # palm
        holding = racket and side == racket_side
        if holding:                                   # fist around the grip
            _g(hand, paint, "hand", E, [0.026, 0.035, 0.03], [0, 0.03, -0.085])
        else:
            _g(hand, paint, "hand", E, [0.017, 0.04, 0.035], [0, 0.0, -0.105])         # fingers
        _g(hand, paint, "hand", C, 0.011, fromto=[0.012 * sx, 0.03, -0.03, 0.018 * sx, 0.05, -0.075])  # thumb
        _site(hand, f"{p}kp_{side}_wrist", [0, 0, 0])
        _site(hand, f"{p}kp_{side}_index", [0, 0.035, -0.09])
        _site(hand, f"{p}kp_{side}_pinky", [0, -0.035, -0.08])
        _col(hand, f"{p}col_{side}_hand", [0, 0.0, -0.06], 0.045)
        if holding:
            _add_racket(hand, p, paint)

        th = pelvis.add_body(); th.name = f"{p}{side}_thigh"; th.pos = [0.09 * sx, 0, -0.03]
        _ball(th, f"{p}{side}_hip")
        _g(th, paint, "shorts", E, [0.078, 0.08, 0.14], [0.006 * sx, 0, -0.09])        # shorts leg
        _g(th, paint, "leg", E, [0.064, 0.068, 0.17], [0, 0.005, -0.25])               # thigh
        _g(th, paint, "leg", S, [0.05], [0, 0.01, -0.43])                               # knee
        _site(th, f"{p}kp_{side}_hip", [0, 0, 0])
        _col(th, f"{p}col_{side}_th_0", [0, 0, -0.12], 0.072)
        _col(th, f"{p}col_{side}_th_1", [0, 0, -0.30], 0.06)
        sh = th.add_body(); sh.name = f"{p}{side}_shin"; sh.pos = [0, 0, -0.44]
        _hinge(sh, f"{p}{side}_knee", [-1, 0, 0], 0, 150)
        _g(sh, paint, "leg", E, [0.05, 0.055, 0.13], [0, -0.012, -0.13])               # calf
        _g(sh, paint, "leg", C, 0.034, fromto=[0, 0, -0.2, 0, 0, -0.34])
        _g(sh, paint, "sock", C, 0.037, fromto=[0, 0, -0.33, 0, 0, -0.41])              # sock
        _site(sh, f"{p}kp_{side}_knee", [0, 0, 0])
        _col(sh, f"{p}col_{side}_sh_0", [0, -0.01, -0.12], 0.052)
        _col(sh, f"{p}col_{side}_sh_1", [0, 0, -0.3], 0.042)
        ft = sh.add_body(); ft.name = f"{p}{side}_foot"; ft.pos = [0, 0, -0.43]
        _hinge(ft, f"{p}{side}_ankle", [1, 0, 0], -50, 30)
        _hinge(ft, f"{p}{side}_subtalar", [0, 1, 0], -25, 25)
        # shoe: sole slab (bottom at z = -0.08, same as the sole sites), upper, toe box, heel counter
        _g(ft, paint, "sole", B, [0.047, 0.122, 0.009], [0, 0.048, -0.071])
        _g(ft, paint, "shoe", E, [0.046, 0.105, 0.04], [0, 0.06, -0.05])
        _g(ft, paint, "shoe", E, [0.044, 0.05, 0.03], [0, 0.125, -0.052])
        _g(ft, paint, "shoe", E, [0.043, 0.05, 0.045], [0, -0.035, -0.04])
        _g(ft, paint, "sole", C, 0.006, fromto=[-0.046, 0.0, -0.045, -0.046, 0.1, -0.05])  # side stripe
        _g(ft, paint, "sole", C, 0.006, fromto=[0.046, 0.0, -0.045, 0.046, 0.1, -0.05])
        _site(ft, f"{p}kp_{side}_ankle", [0, 0, 0])
        # BlazePose "heel" lies on the back of the heel ~5 cm above the sole, "foot_index" at the sole
        _site(ft, f"{p}kp_{side}_heel", [0, -0.005, -0.05])
        _site(ft, f"{p}kp_{side}_toe", [0, 0.16, -0.08])
        for k, y in (("heel", -0.075), ("toe", 0.17)):
            for x in (-0.04, 0.04):
                _site(ft, f"{p}{side}_sole_{k}_{'i' if x * sx < 0 else 'o'}", [x, y, -0.08])
        _col(ft, f"{p}col_{side}_ft_0", [0, -0.01, -0.04], 0.05)
        _col(ft, f"{p}col_{side}_ft_1", [0, 0.11, -0.05], 0.045)
        paint.parts.update({f"{side}_upperarm": ua, f"{side}_forearm": fa, f"{side}_hand": hand, f"{side}_thigh": th,
                            f"{side}_shin": sh, f"{side}_foot": ft})
    paint.parts.update(pelvis=pelvis, torso=torso, head=head)
    return pelvis, paint


def _add_racket(hand, p, paint):
    rk = RACKET
    a = rk["shaft_angle_deg"] * D2R
    shaft = np.array([0.0, np.sin(a), -np.cos(a)])
    body = hand.add_body()
    body.name = p + "racket"
    body.pos = rk["grip_in_hand"]
    body.quat = _quat_from_axes([1.0, 0, 0], shaft)
    zc, hl, hw = rk["head_center"], rk["head_half_length"], rk["head_half_width"]
    paint.parts["racket"] = body
    _g(body, paint, "grip", C, 0.0135, fromto=[0, 0, rk["butt"] + 0.01, 0, 0, 0.08], tag="grip")      # overgrip
    _g(body, paint, "metal", C, 0.0155, fromto=[0, 0, rk["butt"], 0, 0, rk["butt"] + 0.012], tag="grip")  # butt cap
    _g(body, paint, "metal", C, 0.011, fromto=[0, 0, 0.08, 0, 0, 0.11], tag="shaft")       # handle taper
    _g(body, paint, "metal", C, 0.004, fromto=[0, 0, 0.11, 0, 0, zc - hl - 0.045], tag="shaft")   # shaft
    z_t = zc - hl - 0.045                                                                  # throat (T-joint)
    ang = np.linspace(0, 2 * np.pi, 33)
    # isometric head: slightly squarer than an ellipse
    sx_ = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** 0.85
    sz_ = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** 0.85
    ring = np.column_stack([np.zeros_like(ang), hw * sx_, zc + hl * sz_])
    for k in range(32):
        _g(body, paint, "racket", C, 0.0058, fromto=np.r_[ring[k], ring[k + 1]], tag="frame")
    for y in (-1, 1):
        _g(body, paint, "racket", C, 0.0045, fromto=[0, 0, z_t, 0, 0.028 * y, zc - hl * 0.965], tag="frame")
    _g(body, paint, "strings", E, [0.0008, hw * 0.97, hl * 0.97], [0, 0, zc], tag="frame")
    for v in np.linspace(-0.75, 0.75, 7):                                                  # a few string lines
        h = hl * np.sqrt(max(1 - v * v, 0)) * 0.96
        _g(body, paint, "trim", C, 0.0007, fromto=[0.001, v * hw, zc - h, 0.001, v * hw, zc + h], tag="frame")
        w = hw * np.sqrt(max(1 - v * v, 0)) * 0.96
        _g(body, paint, "trim", C, 0.0007, fromto=[0.001, -w, zc + v * hl, 0.001, w, zc + v * hl], tag="frame")
    s = _site(body, p + "racket_center", [0, 0, zc])
    s.size = [0.01, 0, 0]
    s = _site(body, p + "racket_tip", [0, 0, zc + hl])
    s.size = [0.008, 0, 0]
    _col(body, p + "col_racket", [0, 0, zc], 0.10)
    _col(body, p + "col_racket_throat", [0, 0, zc - hl - 0.05], 0.03)


def add_floor(spec, size=12.0, rgba=(0.35, 0.55, 0.4, 1)):
    g = spec.worldbody.add_geom()
    g.name = "floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [size, size, 0.1]
    g.rgba = rgba
    g.contype = g.conaffinity = 0


def add_light(spec, shadows=True):
    for k, (pos, d) in enumerate((([0, -4, 10], [0, 0.35, -1]), ([0, 6, 8], [0, -0.6, -1]))):
        lt = spec.worldbody.add_light()
        lt.pos = pos
        lt.dir = d
        lt.diffuse = [0.65, 0.65, 0.63] if k == 0 else [0.45, 0.45, 0.47]
        lt.specular = [0.35, 0.35, 0.35]
        lt.castshadow = bool(shadows and k == 0)
        lt.cutoff = 70
    spec.visual.headlight.ambient = [0.33, 0.33, 0.35]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.01


def new_spec(offwidth=1280, offheight=720):
    spec = mujoco.MjSpec()
    spec.compiler.degree = True
    spec.visual.global_.offwidth = offwidth
    spec.visual.global_.offheight = offheight
    return spec


def build_single_character_model(offwidth=1280, offheight=720, handed="right", dims=None):
    spec = new_spec(offwidth, offheight)
    add_floor(spec)
    add_light(spec)
    add_character(spec.worldbody, "", spec=spec, handed=handed, dims=dims)
    return spec.compile()


_MIRROR = None


def _mirror_map():
    """qpos index map of the left/right reflection (x -> -x) of the 45-dim character pose."""
    global _MIRROR
    if _MIRROR is None:
        m = build_single_character_model()
        name = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        ids = {name(j): j for j in range(m.njnt)}
        quats, hinges = [], []
        for nm, j in ids.items():
            pn = ("r_" + nm[2:]) if nm.startswith("l_") else (("l_" + nm[2:]) if nm.startswith("r_") else nm)
            a, b = int(m.jnt_qposadr[j]), int(m.jnt_qposadr[ids[pn]])
            t = m.jnt_type[j]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                quats.append((a + 3, b + 3))
            elif t == mujoco.mjtJoint.mjJNT_BALL:
                quats.append((a, b))
            else:
                # a rotation about axis u becomes one about -M u (pseudo-vector); the partner hinge
                # has axis u_p, so the angle keeps its sign when u_p == -M u
                ax = m.jnt_axis[j] * [1, -1, -1]
                hinges.append((a, b, float(np.sign(ax @ m.jnt_axis[ids[pn]]))))
        _MIRROR = (np.array(quats), np.array([h[:2] for h in hinges], int), np.array([h[2] for h in hinges]))
    return _MIRROR


QUAT_ADR = (3, 7, 11, 15, 23, 31, 38)      # set from the model on first use (root, then the ball joints)
_QUAT_READY = False


def quat_blocks():
    """qpos addresses of the quaternions of the 45-dim character pose (root + ball joints)."""
    global QUAT_ADR, _QUAT_READY
    if not _QUAT_READY:
        m = build_single_character_model()
        QUAT_ADR = (3,) + tuple(int(m.jnt_qposadr[j]) for j in range(m.njnt)
                                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL)
        _QUAT_READY = True
    return QUAT_ADR


def quat_slerp(a, b, f):
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + f * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * a + np.sin(f * th) * b) / np.sin(th)


def blend_qpos(a, b, f):
    """Interpolate two character poses (slerp on the quaternion blocks)."""
    q = a + f * (b - a)
    for adr in quat_blocks():
        q[adr:adr + 4] = quat_slerp(a[adr:adr + 4], b[adr:adr + 4], f)
    return q


def mirror_qpos(q):
    """Reflect a character pose (..., 45) across the sagittal plane x = 0: a right-handed stroke
    becomes the same stroke played left-handed (apply it to a model built with the other hand)."""
    q = np.asarray(q, float)
    out = q.copy()
    quats, hinges, sign = _mirror_map()
    out[..., 0] = -q[..., 0]
    for a, b in quats:
        out[..., b:b + 4] = q[..., a:a + 4] * [1, 1, -1, -1]
    out[..., hinges[:, 1]] = q[..., hinges[:, 0]] * sign
    return out


class CharacterIndex:
    """Ids of one (possibly prefixed) character inside a compiled model."""

    def __init__(self, m, prefix=""):
        self.prefix = prefix
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, prefix + n)
        self.kp_sites = np.array([sid("kp_" + n) for n in KP_NAMES])
        self.racket_site = sid("racket_center")
        self.racket_tip = sid("racket_tip")
        rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "racket")
        hand_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[rb]) if rb >= 0 else ""
        self.hand = "l" if hand_body == prefix + "l_hand" else "r"          # racket hand
        self.sole_sites = np.array([sid(f"{s}_sole_{k}_{io}") for s in "lr" for k in ("heel", "toe") for io in "io"])
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
        self.root_joint = jid
        self.qadr = m.jnt_qposadr[jid]
        self.dadr = m.jnt_dofadr[jid]
        last = [j for j in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(prefix)]
        self.joints = last
        self.nq = sum(int(_jnt_nq(m.jnt_type[j])) for j in last)
        self.nv = sum(int(_jnt_nv(m.jnt_type[j])) for j in last)
        self.qslice = slice(self.qadr, self.qadr + self.nq)
        # self-collision spheres
        names, ids = [], []
        for s in range(m.nsite):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ""
            if nm.startswith(prefix + "col_"):
                names.append(nm[len(prefix) + 4:]); ids.append(s)
        self.col_sites = np.array(ids, int)
        self.col_radius = m.site_size[self.col_sites, 0].copy() if len(ids) else np.zeros(0)
        self.col_names = names
        groups = [_col_group(n) for n in names]
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if frozenset((groups[i], groups[j])) in COL_PAIRS:
                    pairs.append((i, j))
        self.col_pairs = np.array(pairs, int).reshape(-1, 2)
        self.col_floor = np.array([i for i, g in enumerate(groups) if g in COL_FLOOR], int)


def _col_group(name):
    """'l_fa_0' -> 'l_fa', 'torso_up' -> 'trunk', 'racket_throat' -> 'racket'."""
    if name.startswith(("pelvis", "torso")):
        return "trunk"
    if name.startswith("racket"):
        return "racket"
    if name == "head":
        return "head"
    parts = name.split("_")
    return f"{parts[0]}_{parts[1]}"


def _pairs():
    P = set()
    arms = [f"{s}_{b}" for s in "lr" for b in ("fa", "hand")]
    legs = {s: [f"{s}_{b}" for b in ("th", "sh", "ft")] for s in "lr"}
    for a in arms:
        P.add(frozenset((a, "trunk"))); P.add(frozenset((a, "head")))
    for a in ("l_fa", "l_hand"):
        for b in ("r_fa", "r_hand"):
            P.add(frozenset((a, b)))
    for a in legs["l"]:
        for b in legs["r"]:
            P.add(frozenset((a, b)))
    for b in ["trunk", "head", "l_ua", "l_fa", "l_hand"] + legs["l"] + legs["r"]:
        P.add(frozenset(("racket", b)))
    return P


COL_PAIRS = _pairs()
COL_FLOOR = {"l_th", "r_th", "l_sh", "r_sh", "l_hand", "r_hand", "racket", "l_fa", "r_fa"}


def _jnt_nq(t):
    return {0: 7, 1: 4, 2: 1, 3: 1}[int(t)]


def _jnt_nv(t):
    return {0: 6, 1: 3, 2: 1, 3: 1}[int(t)]

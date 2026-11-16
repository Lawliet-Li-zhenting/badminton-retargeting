"""The wardrobe: what a character looks like, as data.

An outfit is a dict slot -> item key (the save file keeps the player's; styles.Style.outfit dresses an
opponent). Slots:

    suit     a whole look (colours of every part of the body, sometimes with pieces sewn on); "none" = the kit
             colours below
    shirt, shorts, shoes, skin, hair      colours
    head, face, back                      things worn (hats, glasses, capes ...)
    racket   what is held instead of a racket -- looks only: the string bed the game computes with stays the same
    body     build (big head, muscles, bean pole) -- looks only as well
    trail    the colour of the shuttle's trail          (read by ui.scene_fx)
    name, quip                            what the player is called and says (read by the app / the match)

Everything worn is made of the same primitives as the body (ellipsoids, capsules, boxes, cylinders) attached
to its segments, without collision. Look.dress() is called by character.add_character before the body is
scaled to its owner, so hats fit big heads and capes fit tall players.
"""
from __future__ import annotations

import mujoco
import numpy as np

E, C, S, B, Y = (mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                 mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER)

WHITE, BLACK, GOLD, RED = (0.96, 0.96, 0.97, 1), (0.05, 0.05, 0.06, 1), (0.98, 0.78, 0.15, 1), (0.88, 0.12, 0.14, 1)
PINK, GREEN, BLUE, GREY = (0.98, 0.5, 0.68, 1), (0.2, 0.62, 0.25, 1), (0.15, 0.4, 0.9, 1), (0.6, 0.63, 0.68, 1)
ORANGE, STRAW, KRAFT = (0.98, 0.5, 0.1, 1), (0.93, 0.8, 0.45, 1), (0.72, 0.56, 0.36, 1)


def rx(deg):
    a = np.radians(deg) / 2
    return [np.cos(a), np.sin(a), 0, 0]


def ry(deg):
    a = np.radians(deg) / 2
    return [np.cos(a), 0, np.sin(a), 0]


class Kit:
    """Adds pieces to the segments of one character."""

    def __init__(self, paint, hand):
        self.paint, self.parts, self.hand = paint, paint.parts, hand

    def role(self, rgba, shine=None):
        name = "w" + "".join("%02x" % int(round(255 * float(v))) for v in rgba)
        self.paint.colors[name] = tuple(rgba)
        if shine is not None:
            self.paint.shine[name] = shine
        return name

    def add(self, part, gtype, size, pos=(0, 0, 0), rgba=WHITE, quat=None, fromto=None, shine=None):
        from .character import _g
        return _g(self.parts[part], self.paint, self.role(rgba, shine), gtype, size, pos, quat, fromto)

    def both(self, part, gtype, size, pos, rgba, quat=None, fromto=None):
        """The same piece on the left and the right segment (x mirrored)."""
        for side, sx in (("l", -1.0), ("r", 1.0)):
            p = None if pos is None else [pos[0] * sx, pos[1], pos[2]]
            ft = None if fromto is None else [fromto[0] * sx, fromto[1], fromto[2], fromto[3] * sx, fromto[4], fromto[5]]
            self.add(f"{side}_{part}", gtype, size, p if p is not None else (0, 0, 0), rgba, quat, ft)

    def hide(self, *tags):
        self.paint.hide(*tags)


# ---------------------------------------------------------------------------- head
def _dome(k, rgba, hide=True):
    if hide:
        k.hide("hair", "ponytail", "band")
    k.add("head", E, [0.104, 0.112, 0.092], [0, -0.008, 0.2], rgba, shine=(0.6, 0.7))


def h_chef(k):
    k.add("head", Y, 0.09, rgba=WHITE, fromto=[0, 0, 0.245, 0, 0, 0.33])
    k.add("head", E, [0.125, 0.125, 0.065], [0, 0, 0.365], WHITE)


def h_crown(k):
    k.add("head", Y, 0.088, rgba=GOLD, fromto=[0, 0, 0.24, 0, 0, 0.285], shine=(0.9, 0.9))
    for a in np.linspace(0, 2 * np.pi, 7)[:-1]:
        x, y = 0.082 * np.sin(a), 0.082 * np.cos(a)
        k.add("head", C, 0.013, rgba=GOLD, fromto=[x, y, 0.28, x * 1.12, y * 1.12, 0.335], shine=(0.9, 0.9))
        k.add("head", S, [0.014], [x * 1.12, y * 1.12, 0.345], RED if abs(x) < 0.01 and y > 0 else (0.2, 0.5, 0.95, 1))


def h_cone(k):
    k.hide("ponytail")
    k.add("head", B, [0.13, 0.13, 0.008], [0, 0, 0.245], ORANGE)
    r0, n = 0.1, 6
    for i in range(n):
        z0 = 0.25 + i * 0.052
        k.add("head", Y, r0 * (1 - i / (n + 0.6)), rgba=WHITE if i in (2, 3) else ORANGE, fromto=[0, 0, z0, 0, 0, z0 + 0.052])


def h_horns(k):
    for sx in (-1, 1):
        k.add("head", C, 0.02, rgba=RED, fromto=[0.06 * sx, 0.0, 0.235, 0.105 * sx, 0.0, 0.3])
        k.add("head", C, 0.012, rgba=RED, fromto=[0.105 * sx, 0.0, 0.3, 0.1 * sx, 0.0, 0.36])


def h_bunny(k):
    for sx in (-1, 1):
        k.add("head", E, [0.026, 0.014, 0.1], [0.05 * sx, -0.01, 0.36], WHITE, quat=ry(8 * sx))
        k.add("head", E, [0.014, 0.008, 0.07], [0.05 * sx, 0.002, 0.36], PINK, quat=ry(8 * sx))


def h_halo(k):
    a = np.linspace(0, 2 * np.pi, 15)
    for i in range(14):
        k.add("head", C, 0.009, rgba=(1.0, 0.9, 0.3, 1), shine=(0.9, 0.9),
              fromto=[0.09 * np.cos(a[i]), 0.09 * np.sin(a[i]), 0.37, 0.09 * np.cos(a[i + 1]), 0.09 * np.sin(a[i + 1]), 0.37])


def h_straw(k):
    k.hide("band")
    k.add("head", Y, 0.2, rgba=STRAW, fromto=[0, 0, 0.238, 0, 0, 0.25])
    k.add("head", Y, 0.098, rgba=STRAW, fromto=[0, 0, 0.25, 0, 0, 0.325])
    k.add("head", Y, 0.1, rgba=RED, fromto=[0, 0, 0.252, 0, 0, 0.272])


def h_melon(k):
    _dome(k, (0.1, 0.45, 0.15, 1))
    for a in np.linspace(-60, 60, 5):
        k.add("head", E, [0.012, 0.114, 0.09], [0, -0.008, 0.203], (0.04, 0.22, 0.08, 1), quat=ry(a))
    k.add("head", C, 0.008, rgba=(0.35, 0.25, 0.1, 1), fromto=[0, -0.008, 0.29, 0.015, -0.008, 0.32])


def h_mohawk(k):
    k.hide("hair", "ponytail", "band")
    for i, y in enumerate(np.linspace(0.07, -0.09, 6)):
        k.add("head", E, [0.012, 0.022, 0.06 - 0.004 * abs(i - 2)], [0, y, 0.275 - 0.012 * max(i - 3, 0)], (0.98, 0.2, 0.6, 1))


def h_antenna(k):
    for sx in (-1, 1):
        k.add("head", C, 0.004, rgba=GREY, fromto=[0.04 * sx, 0, 0.25, 0.075 * sx, 0.015, 0.4])
        k.add("head", S, [0.022], [0.075 * sx, 0.015, 0.4], (0.3, 1.0, 0.4, 1))


def h_copter(k):
    k.add("head", C, 0.006, rgba=GOLD, fromto=[0, 0, 0.255, 0, 0, 0.35])
    k.add("head", B, [0.12, 0.013, 0.003], [0, 0, 0.352], GOLD, quat=rx(12))


def h_bag(k):
    k.hide("hair", "ponytail", "band", "eyes", "nose", "ears")
    k.add("head", B, [0.118, 0.122, 0.15], [0, 0.003, 0.175], KRAFT)
    for sx in (-1, 1):
        k.add("head", E, [0.02, 0.004, 0.02], [0.038 * sx, 0.125, 0.18], BLACK)
    k.add("head", C, 0.004, rgba=BLACK, fromto=[-0.03, 0.126, 0.1, 0.03, 0.126, 0.1])


def h_panda(k):
    k.hide("hair", "ponytail", "band", "eyes", "nose", "ears")
    k.add("head", S, [0.14], [0, 0.003, 0.175], WHITE)
    for sx in (-1, 1):
        k.add("head", S, [0.05], [0.1 * sx, 0, 0.29], BLACK)
        k.add("head", E, [0.034, 0.012, 0.044], [0.052 * sx, 0.128, 0.19], BLACK, quat=ry(-18 * sx))
        k.add("head", S, [0.01], [0.05 * sx, 0.139, 0.195], WHITE)
    k.add("head", E, [0.02, 0.01, 0.013], [0, 0.141, 0.15], BLACK)


def h_shuttle(k):
    k.hide("band")
    n = 5
    for i in range(n):
        z0 = 0.245 + i * 0.03
        k.add("head", Y, 0.105 - 0.014 * i, rgba=WHITE, fromto=[0, 0, z0, 0, 0, z0 + 0.03])
    k.add("head", Y, 0.05, rgba=RED, fromto=[0, 0, 0.395, 0, 0, 0.405])
    k.add("head", S, [0.04], [0, 0, 0.425], (0.85, 0.7, 0.5, 1))


def h_astro(k):
    k.add("head", S, [0.175], [0, 0.01, 0.165], (0.75, 0.88, 1.0, 0.22), shine=(0.9, 0.95))
    k.add("head", Y, 0.125, rgba=WHITE, fromto=[0, 0, 0.0, 0, 0, 0.04])


def h_robot(k):
    k.hide("hair", "ponytail", "band", "eyes", "nose", "ears")
    k.add("head", B, [0.105, 0.105, 0.105], [0, 0.003, 0.165], GREY, shine=(0.8, 0.8))
    k.add("head", B, [0.08, 0.004, 0.022], [0, 0.11, 0.185], (0.2, 0.95, 1.0, 1))
    k.add("head", B, [0.04, 0.004, 0.006], [0, 0.11, 0.12], BLACK)
    k.add("head", C, 0.005, rgba=GREY, fromto=[0, 0, 0.27, 0, 0, 0.34])
    k.add("head", S, [0.018], [0, 0, 0.345], RED)


def h_rider(k):
    _dome(k, (0.98, 0.82, 0.1, 1))
    h_bunny(k)


def h_bald(k):
    k.hide("hair", "ponytail", "band")


def h_afro(k):
    k.hide("ponytail", "band")
    k.add("head", S, [0.165], [0, -0.02, 0.24], (0.1, 0.07, 0.05, 1))


# ---------------------------------------------------------------------------- face
def f_shades(k):
    for sx in (-1, 1):
        k.add("head", B, [0.03, 0.004, 0.017], [0.036 * sx, 0.101, 0.166], BLACK, shine=(0.9, 0.9))
        k.add("head", C, 0.004, rgba=BLACK, fromto=[0.066 * sx, 0.1, 0.168, 0.088 * sx, 0.0, 0.165])
    k.add("head", C, 0.004, rgba=BLACK, fromto=[-0.01, 0.102, 0.17, 0.01, 0.102, 0.17])


def f_nose(k):
    k.add("head", S, [0.024], [0, 0.108, 0.14], RED, shine=(0.7, 0.8))


def f_mustache(k):
    for sx in (-1, 1):
        k.add("head", E, [0.026, 0.007, 0.009], [0.024 * sx, 0.096, 0.119], BLACK, quat=ry(18 * sx))


def f_mask(k):
    k.hide("nose")
    k.add("head", E, [0.062, 0.03, 0.04], [0, 0.075, 0.115], (0.6, 0.82, 0.95, 1))
    for sx in (-1, 1):
        k.add("head", C, 0.003, rgba=WHITE, fromto=[0.055 * sx, 0.07, 0.13, 0.088 * sx, -0.005, 0.15])


def f_blush(k):
    for sx in (-1, 1):
        k.add("head", E, [0.017, 0.005, 0.012], [0.056 * sx, 0.081, 0.132], (1.0, 0.45, 0.5, 1))


def f_patch(k):
    k.add("head", E, [0.022, 0.006, 0.022], [0.033, 0.097, 0.165], BLACK)
    k.add("head", C, 0.004, rgba=BLACK, fromto=[0.05, 0.09, 0.18, -0.07, 0.06, 0.225])


def f_monocle(k):
    a = np.linspace(0, 2 * np.pi, 11)
    for i in range(10):
        k.add("head", C, 0.003, rgba=GOLD, fromto=[0.033 + 0.022 * np.cos(a[i]), 0.103, 0.166 + 0.022 * np.sin(a[i]),
                                                   0.033 + 0.022 * np.cos(a[i + 1]), 0.103, 0.166 + 0.022 * np.sin(a[i + 1])])
    k.add("head", C, 0.002, rgba=GOLD, fromto=[0.05, 0.1, 0.148, 0.06, 0.09, 0.07])


# ---------------------------------------------------------------------------- back
def b_cape(k, rgba=RED):
    k.add("torso", C, 0.02, rgba=rgba, fromto=[-0.13, -0.04, 0.4, 0.13, -0.04, 0.4])
    k.add("torso", B, [0.165, 0.008, 0.13], [0, -0.14, 0.29], rgba, quat=rx(-14))
    k.add("torso", B, [0.185, 0.008, 0.14], [0, -0.215, 0.04], rgba, quat=rx(-20))
    k.add("torso", B, [0.2, 0.008, 0.13], [0, -0.32, -0.2], rgba, quat=rx(-27))


def b_wings(k):
    for sx in (-1, 1):
        for i, (ang, ln) in enumerate(((35, 0.2), (60, 0.24), (85, 0.2), (110, 0.15))):
            a = np.radians(ang)
            c = np.array([sx * (0.05 + 0.5 * ln * np.sin(a)), -0.135 - 0.012 * i, 0.3 + 0.5 * ln * np.cos(a)])
            k.add("torso", E, [0.035, 0.01, 0.5 * ln], c, WHITE, quat=ry(ang * sx))


def b_shell(k):
    k.add("torso", E, [0.175, 0.075, 0.2], [0, -0.13, 0.21], (0.12, 0.42, 0.18, 1))
    for x, z in ((0, 0.21), (-0.08, 0.3), (0.08, 0.3), (-0.08, 0.12), (0.08, 0.12)):
        k.add("torso", E, [0.045, 0.02, 0.05], [x, -0.195 + 0.3 * abs(x), z], (0.45, 0.7, 0.3, 1))


def b_jet(k):
    for sx in (-1, 1):
        k.add("torso", Y, 0.052, rgba=GREY, fromto=[0.07 * sx, -0.165, 0.1, 0.07 * sx, -0.165, 0.36], shine=(0.8, 0.8))
        k.add("torso", S, [0.052], [0.07 * sx, -0.165, 0.36], RED)
        k.add("torso", E, [0.035, 0.035, 0.08], [0.07 * sx, -0.165, 0.03], (1.0, 0.6, 0.1, 0.85))
        k.add("torso", E, [0.02, 0.02, 0.05], [0.07 * sx, -0.165, 0.03], (1.0, 0.95, 0.5, 0.9))


def b_tail(k, rgba=(0.25, 0.6, 0.25, 1)):
    p = np.array([0.0, -0.09, -0.02])
    for i in range(7):
        r = 0.06 - 0.006 * i
        k.add("pelvis", S, [r], p, rgba)
        p = p + np.array([0.0, -0.075 * np.cos(0.25 * i), -0.03 + 0.03 * i * 0.5])


def b_balloon(k):
    k.add("torso", C, 0.003, rgba=WHITE, fromto=[0.12, -0.08, 0.36, 0.3, -0.3, 1.0])
    k.add("torso", E, [0.14, 0.14, 0.17], [0.3, -0.3, 1.16], (0.95, 0.15, 0.25, 0.92), shine=(0.9, 0.9))


def b_bag(k):
    k.add("torso", B, [0.13, 0.065, 0.16], [0, -0.17, 0.22], (0.85, 0.15, 0.15, 1))
    k.add("torso", B, [0.09, 0.02, 0.06], [0, -0.24, 0.15], (0.95, 0.75, 0.15, 1))
    for sx in (-1, 1):
        k.add("torso", C, 0.014, rgba=BLACK, fromto=[0.1 * sx, 0.09, 0.36, 0.1 * sx, 0.1, 0.12])


def b_box(k):
    k.add("torso", B, [0.2, 0.15, 0.18], [0, -0.26, 0.24], (0.98, 0.82, 0.1, 1))
    k.add("torso", B, [0.12, 0.004, 0.06], [0, -0.412, 0.26], BLACK)
    k.add("torso", B, [0.202, 0.152, 0.012], [0, -0.26, 0.425], BLACK)


def b_spikes(k, rgba=(0.95, 0.6, 0.1, 1)):
    for z in (0.08, 0.2, 0.32):
        k.add("torso", E, [0.012, 0.05, 0.055], [0, -0.13, z], rgba, quat=rx(35))


def b_sword(k):
    k.add("torso", B, [0.012, 0.004, 0.3], [0, -0.13, 0.25], GREY, quat=ry(28), shine=(0.9, 0.95))
    k.add("torso", B, [0.07, 0.008, 0.012], [0.135, -0.13, 0.5], GOLD, quat=ry(28))


# ---------------------------------------------------------------------------- suits: colours of the roles + pieces
def s_boss(k):
    k.add("torso", B, [0.045, 0.01, 0.1], [0, 0.098, 0.3], WHITE)
    k.add("torso", B, [0.016, 0.006, 0.085], [0, 0.11, 0.28], RED)
    k.add("torso", S, [0.02], [0, 0.108, 0.375], RED)


def s_bones(k):
    k.hide("hair", "ponytail", "band", "cuffs")
    bone = (0.94, 0.94, 0.88, 1)
    for part, z0, z1, r in (("upperarm", -0.04, -0.26, 0.016), ("forearm", -0.03, -0.23, 0.013), ("thigh", -0.08, -0.4, 0.022),
                            ("shin", -0.05, -0.38, 0.018)):
        for y in (-1, 1):
            k.both(part, C, r, None, bone, fromto=[0, 0.75 * y * (0.045 if part in ("upperarm", "forearm") else 0.06), z0,
                                                    0, 0.75 * y * (0.035 if part in ("upperarm", "forearm") else 0.05), z1])
    for y in (-1, 1):
        k.add("torso", C, 0.016, rgba=bone, fromto=[0, 0.1 * y, 0.02, 0, 0.105 * y, 0.4])
        for z in (0.2, 0.26, 0.32, 0.38):
            k.add("torso", C, 0.012, rgba=bone, fromto=[-0.12, 0.085 * y, z - 0.02, 0.0, 0.11 * y, z])
            k.add("torso", C, 0.012, rgba=bone, fromto=[0.12, 0.085 * y, z - 0.02, 0.0, 0.11 * y, z])
    k.add("pelvis", E, [0.13, 0.03, 0.05], [0, 0.085, 0.0], bone)
    for sx in (-1, 1):
        k.add("head", E, [0.026, 0.012, 0.03], [0.034 * sx, 0.09, 0.166], BLACK)
    k.add("head", E, [0.012, 0.01, 0.016], [0, 0.1, 0.132], BLACK)
    for x in (-0.03, -0.015, 0.0, 0.015, 0.03):
        k.add("head", C, 0.003, rgba=BLACK, fromto=[x, 0.092, 0.092, x, 0.094, 0.112])


def s_astro(k):
    k.add("torso", B, [0.07, 0.012, 0.055], [0, 0.1, 0.27], GREY)
    for i, c in enumerate((RED, GREEN, BLUE)):
        k.add("torso", S, [0.012], [-0.035 + 0.035 * i, 0.114, 0.285], c)
    k.add("torso", B, [0.11, 0.07, 0.14], [0, -0.16, 0.25], WHITE)


def s_robot(k):
    k.add("torso", B, [0.09, 0.012, 0.07], [0, 0.1, 0.26], (0.2, 0.22, 0.26, 1))
    for i, c in enumerate(((0.2, 0.95, 1.0, 1), (0.3, 1.0, 0.4, 1), RED)):
        k.add("torso", S, [0.014], [-0.045 + 0.045 * i, 0.115, 0.28], c)
    k.both("upperarm", S, [0.05], [0, 0, -0.28], (0.95, 0.75, 0.1, 1))
    k.both("thigh", S, [0.058], [0, 0.01, -0.43], (0.95, 0.75, 0.1, 1))


def s_dino(k):
    k.add("torso", E, [0.105, 0.03, 0.17], [0, 0.085, 0.22], (0.9, 0.88, 0.55, 1))
    b_spikes(k)
    b_tail(k)


def s_bee(k):
    for z in (0.14, 0.26):
        k.add("torso", Y, 0.16, rgba=BLACK, fromto=[0, 0, z, 0, 0, z + 0.05])
    b_wings(k)


def _all(c, **kw):
    d = dict(shirt=c, sleeve=c, shorts=c, arm=c, leg=c, hand=c, sock=c, shoe=c)
    d.update(kw)
    return d


SKIN = (0.93, 0.75, 0.6, 1)
NAVY, INK = (0.07, 0.09, 0.16, 1), (0.09, 0.09, 0.11, 1)
# key: (name, price, line, colours of the roles, pieces, hair shown)
SUITS = {
    "none": ("自己搭配", 0, "上衣、短裤、球鞋各选各的", {}, None),
    "tank": ("老头乐", 0, "白背心、大裤衩、人字拖。公园里最不能惹的那种", dict(shirt=WHITE, sleeve=None, shorts=(0.25, 0.45, 0.75, 1), sock=None,
                                                    shoe=(0.2, 0.5, 0.9, 1), sole=(0.2, 0.5, 0.9, 1)), None),
    "school": ("校服", 80, "蓝白校服，全国统一款。跑操专用", dict(shirt=WHITE, sleeve=(0.1, 0.25, 0.7, 1), arm=(0.1, 0.25, 0.7, 1),
                                                shorts=(0.1, 0.25, 0.7, 1), leg=(0.1, 0.25, 0.7, 1)), None),
    "longjohns": ("本命年", 80, "红秋衣红秋裤，妈妈说穿上就不会输", _all(RED, hand=None, shoe=BLACK, sock=RED), None),
    "pajama": ("睡衣", 120, "刚睡醒，顺便打个球", _all((0.6, 0.8, 0.95, 1), hand=None, shoe=(0.98, 0.75, 0.8, 1), sock=None), None),
    "boss": ("霸总", 200, "西装革履。天凉了，该让对手破产了", _all(NAVY, hand=None, shoe=BLACK, sock=BLACK), s_boss),
    "square": ("广场舞", 120, "粉色上衣黑长裤，音响一响，全场归我", dict(shirt=PINK, sleeve=PINK, shorts=INK, leg=INK, shoe=WHITE), None),
    "rider": ("外卖骑手", 200, "您的杀球已送达，请给五星好评", dict(shirt=(0.98, 0.82, 0.1, 1), sleeve=(0.98, 0.82, 0.1, 1), arm=(0.98, 0.82, 0.1, 1),
                                                   shorts=INK, leg=INK, shoe=INK), None),
    "panda": ("国宝", 300, "黑白配。打球可以输，萌不能输", dict(shirt=WHITE, sleeve=BLACK, arm=BLACK, shorts=WHITE, leg=BLACK, hand=BLACK,
                                                 shoe=BLACK, sock=BLACK), None),
    "bee": ("小蜜蜂", 300, "嗡嗡嗡。蜇人很疼的那种杀球", dict(shirt=(0.98, 0.82, 0.1, 1), sleeve=BLACK, arm=BLACK, shorts=BLACK, leg=BLACK,
                                               shoe=BLACK, sock=BLACK), s_bee),
    "dino": ("恐龙", 400, "穿上它，心理年龄减二十岁", _all((0.25, 0.6, 0.25, 1), skin=None), s_dino),
    "astro": ("宇航员", 400, "地球上的羽毛球场太小了", _all((0.93, 0.94, 0.96, 1), hand=GREY, shoe=GREY), s_astro),
    "robot": ("机器人", 500, "哔——检测到你的反手是弱点", _all(GREY, face=GREY), s_robot),
    "bones": ("骨感", 500, "减肥减过头了", _all((0.04, 0.04, 0.05, 1), face=(0.94, 0.94, 0.88, 1)), s_bones),
    "gold": ("土豪金", 800, "从头到脚都是金的，包括球品", _all((0.95, 0.74, 0.18, 1), face=(0.95, 0.74, 0.18, 1), hair=(0.95, 0.74, 0.18, 1)), None),
    "ghost": ("隐身衣", 800, "看不见我，但是看得见拍子", _all((0.8, 0.9, 1.0, 0.1), face=(0.8, 0.9, 1.0, 0.1), hair=(0.8, 0.9, 1.0, 0.1),
                                              trim=(0.8, 0.9, 1.0, 0.1), band=(0.8, 0.9, 1.0, 0.1), sole=(0.8, 0.9, 1.0, 0.1)), None),
}
SUIT_SHINE = {"gold": (0.95, 0.95), "robot": (0.8, 0.8)}

HEADS = {
    "none": ("不戴", 0, "头带就挺好", None),
    "bald": ("光头", 0, "强者的发型", h_bald),
    "afro": ("爆炸头", 60, "被杀球电到之后就这样了", h_afro),
    "mohawk": ("鸡冠头", 60, "气势上先赢一半", h_mohawk),
    "straw": ("草帽", 60, "下地干活和打球两不误", h_straw),
    "chef": ("厨师帽", 100, "今天给对手炒个菜", h_chef),
    "bunny": ("兔耳朵", 100, "卖萌是战术的一部分", h_bunny),
    "horns": ("小恶魔", 100, "专打你够不着的地方", h_horns),
    "copter": ("竹蜻蜓", 150, "理论上可以飞，实际上不行", h_copter),
    "antenna": ("天线", 150, "正在接收外星教练的指导", h_antenna),
    "shuttle": ("羽毛球帽", 150, "把信仰顶在头上", h_shuttle),
    "melon": ("西瓜头盔", 200, "夏天限定，打累了可以吃", h_melon),
    "cone": ("雪糕筒", 200, "路上捡的，前方施工请绕行", h_cone),
    "halo": ("光环", 250, "打得太好，原地成仙", h_halo),
    "rider": ("骑手头盔", 250, "带兔耳朵的那种，超时要扣钱", h_rider),
    "crown": ("皇冠", 400, "欲戴皇冠，必先赢球", h_crown),
    "bag": ("纸袋头", 300, "社恐专用。输了也没人认得出", h_bag),
    "panda": ("熊猫头", 400, "表情包本包", h_panda),
    "astro": ("宇航头盔", 400, "自带氧气，三局不喘", h_astro),
    "robot": ("机器人头", 500, "面无表情地杀球", h_robot),
}
FACES = {
    "none": ("素颜", 0, "", None),
    "blush": ("腮红", 30, "不是害羞，是跑热了", f_blush),
    "shades": ("墨镜", 80, "太阳不大，主要是为了帅", f_shades),
    "mustache": ("八字胡", 80, "看起来像个高手", f_mustache),
    "mask": ("口罩", 80, "低调，别认出我", f_mask),
    "nose": ("红鼻子", 120, "小丑竟是我自己", f_nose),
    "patch": ("独眼罩", 150, "一只眼也能看清你的球路", f_patch),
    "monocle": ("单片眼镜", 200, "优雅，永不过时", f_monocle),
}
BACKS = {
    "none": ("不背", 0, "", None),
    "bag": ("书包", 60, "作业写完了才能来打球", b_bag),
    "cape": ("披风", 150, "超级英雄都有，虽然会绊脚", b_cape),
    "tail": ("尾巴", 150, "保持平衡用的", b_tail),
    "balloon": ("气球", 150, "别扎，扎了我跟你急", b_balloon),
    "shell": ("龟壳", 200, "防守反击的最高境界", b_shell),
    "box": ("外卖箱", 250, "里面是对手的盒饭", b_box),
    "sword": ("大宝剑", 250, "背着不用，主要是气质", b_sword),
    "wings": ("翅膀", 400, "跳杀的时候好像真的飞起来了", b_wings),
    "jet": ("火箭背包", 500, "还没点火，点了要出事", b_jet),
}


# ---------------------------------------------------------------------------- what is held (racket frame: z along
# the shaft, the string bed in the y-z plane around z = 0.46, 0.2 wide and 0.23 long)
def r_pan(k):
    k.hide("frame", "shaft")
    k.add("racket", Y, 0.108, rgba=(0.13, 0.13, 0.15, 1), fromto=[-0.012, 0, 0.46, 0.012, 0, 0.46], shine=(0.7, 0.8))
    k.add("racket", Y, 0.094, rgba=(0.3, 0.3, 0.33, 1), fromto=[0.012, 0, 0.46, 0.014, 0, 0.46])
    k.add("racket", C, 0.013, rgba=BLACK, fromto=[0, 0, 0.08, 0, 0, 0.35])


def r_fish(k):
    k.hide("frame", "shaft")
    c = (0.78, 0.66, 0.45, 1)
    k.add("racket", E, [0.022, 0.08, 0.17], [0, 0, 0.43], c)
    k.add("racket", E, [0.012, 0.035, 0.07], [0, 0, 0.21], c)
    for sy in (-1, 1):
        k.add("racket", E, [0.008, 0.03, 0.06], [0, 0.035 * sy, 0.13], c, quat=rx(-28 * sy))
    for sx in (-1, 1):
        k.add("racket", S, [0.012], [0.02 * sx, 0.035, 0.53], WHITE)
        k.add("racket", S, [0.006], [0.028 * sx, 0.037, 0.532], BLACK)
    k.add("racket", E, [0.006, 0.02, 0.05], [0, -0.085, 0.45], (0.6, 0.5, 0.32, 1))


def r_slipper(k):
    k.hide("frame", "shaft")
    k.add("racket", E, [0.014, 0.075, 0.15], [0, 0, 0.45], (0.15, 0.45, 0.9, 1))
    k.add("racket", C, 0.011, rgba=WHITE, fromto=[0.012, -0.06, 0.47, 0.03, 0.0, 0.52])
    k.add("racket", C, 0.011, rgba=WHITE, fromto=[0.012, 0.06, 0.47, 0.03, 0.0, 0.52])
    k.add("racket", C, 0.011, rgba=WHITE, fromto=[0.014, 0.0, 0.56, 0.03, 0.0, 0.52])
    k.add("racket", C, 0.012, rgba=(0.15, 0.45, 0.9, 1), fromto=[0, 0, 0.08, 0, 0, 0.31])


def r_swatter(k):
    k.hide("frame")
    k.add("racket", B, [0.003, 0.085, 0.09], [0, 0, 0.47], (0.3, 0.85, 0.35, 0.75))
    for v in np.linspace(-0.07, 0.07, 5):
        k.add("racket", C, 0.003, rgba=(0.15, 0.55, 0.2, 1), fromto=[0, v, 0.385, 0, v, 0.555])
        k.add("racket", C, 0.003, rgba=(0.15, 0.55, 0.2, 1), fromto=[0, -0.082, 0.47 + v, 0, 0.082, 0.47 + v])


def r_leek(k):
    k.hide("frame", "shaft")
    k.add("racket", C, 0.024, rgba=(0.97, 0.97, 0.92, 1), fromto=[0, 0, 0.06, 0, 0, 0.34])
    for y, t in ((-0.05, 0.57), (0.0, 0.6), (0.055, 0.56)):
        k.add("racket", C, 0.016, rgba=(0.2, 0.6, 0.22, 1), fromto=[0, 0, 0.34, 0, y, t])


def r_keyboard(k):
    k.hide("frame", "shaft")
    k.add("racket", B, [0.01, 0.075, 0.19], [0, 0, 0.4], (0.13, 0.14, 0.17, 1))
    for i in range(9):
        for j in range(4):
            k.add("racket", B, [0.003, 0.013, 0.015], [0.011, -0.054 + 0.036 * j, 0.24 + 0.04 * i],
                  (0.95, 0.3, 0.3, 1) if (i, j) == (8, 0) else (0.85, 0.87, 0.9, 1))


def r_brick(k):
    k.hide("frame", "shaft")
    k.add("racket", B, [0.03, 0.06, 0.115], [0, 0, 0.46], (0.72, 0.3, 0.2, 1))
    k.add("racket", C, 0.012, rgba=(0.5, 0.35, 0.2, 1), fromto=[0, 0, 0.08, 0, 0, 0.35])


def r_fan(k):
    k.hide("frame")
    k.add("racket", E, [0.004, 0.115, 0.125], [0, 0, 0.46], STRAW)
    for a in np.linspace(-50, 50, 7):
        s, c = np.sin(np.radians(a)), np.cos(np.radians(a))
        k.add("racket", C, 0.003, rgba=(0.6, 0.45, 0.2, 1), fromto=[0.004, 0, 0.34, 0.004, 0.2 * s, 0.34 + 0.23 * c])


def r_glow(k):
    k.hide("frame", "shaft")
    k.add("racket", C, 0.016, rgba=(0.3, 1.0, 0.9, 0.9), fromto=[0, 0, 0.09, 0, 0, 0.58], shine=(1.0, 1.0))
    k.add("racket", C, 0.007, rgba=WHITE, fromto=[0, 0, 0.09, 0, 0, 0.58])


def r_lolly(k):
    k.hide("frame")
    for i, c in enumerate(((0.98, 0.3, 0.5, 1), WHITE, (0.98, 0.3, 0.5, 1), WHITE, (0.98, 0.3, 0.5, 1))):
        k.add("racket", Y, 0.11 - 0.02 * i, rgba=c, fromto=[-0.012 - 0.002 * i, 0, 0.46, 0.012 + 0.002 * i, 0, 0.46])


RACKETS = {
    "none": ("球拍", 0, "正经球拍", None),
    "fan": ("蒲扇", 60, "夏天打球必备，还能扇风", r_fan),
    "swatter": ("苍蝇拍", 60, "羽毛球和苍蝇，本质上是一回事", r_swatter),
    "slipper": ("人字拖", 100, "妈妈的武器，童年的阴影", r_slipper),
    "pan": ("平底锅", 150, "吃鸡同款，杀球带锅气", r_pan),
    "leek": ("大葱", 150, "山东人的骄傲，打完还能蘸酱", r_leek),
    "lolly": ("棒棒糖", 200, "打累了舔一口", r_lolly),
    "keyboard": ("键盘", 250, "键盘侠的终极形态：用键盘打用键盘控制的人", r_keyboard),
    "brick": ("板砖", 300, "能动手就别吵吵", r_brick),
    "glow": ("荧光棒", 300, "演唱会顺回来的", r_glow),
    "fish": ("咸鱼", 500, "咸鱼也有梦想。翻身就靠这一拍", r_fish),
}

_C = lambda *c: tuple(c) + (1,)
COLOURS = [("白", _C(0.96, 0.96, 0.97)), ("黑", _C(0.08, 0.08, 0.1)), ("红", _C(0.86, 0.12, 0.14)), ("蓝", _C(0.12, 0.38, 0.9)),
           ("黄", _C(0.98, 0.82, 0.1)), ("绿", _C(0.15, 0.62, 0.3)), ("橙", _C(0.98, 0.5, 0.1)), ("紫", _C(0.55, 0.3, 0.85)),
           ("粉", _C(0.98, 0.5, 0.68)), ("青", _C(0.1, 0.7, 0.75)), ("藏青", _C(0.08, 0.1, 0.2)), ("荧光绿", _C(0.6, 1.0, 0.2))]
SHIRTS = {f"c{i}": (n, 0, "", c) for i, (n, c) in enumerate(COLOURS)}
SHORTS = dict(SHIRTS)
SHOES = dict(SHIRTS)
SKINS = {
    "s0": ("自然", 0, "", _C(0.93, 0.75, 0.6)), "s1": ("白皙", 0, "", _C(0.98, 0.84, 0.74)), "s2": ("小麦", 0, "", _C(0.8, 0.6, 0.45)),
    "s3": ("古铜", 0, "", _C(0.6, 0.42, 0.3)), "s4": ("深色", 0, "", _C(0.38, 0.26, 0.2)),
    "blue": ("阿凡达", 150, "潘多拉星球羽毛球队", _C(0.3, 0.55, 0.9)), "green": ("绿巨人", 150, "别惹我生气，我一生气就杀球", _C(0.4, 0.72, 0.3)),
    "purple": ("紫薯精", 150, "一个响指，对手减半", _C(0.55, 0.4, 0.75)), "grey": ("水泥灰", 150, "刚从工地下班", _C(0.62, 0.63, 0.65)),
    "red": ("关公", 150, "面如重枣，温酒斩对手", _C(0.8, 0.25, 0.2)),
}
HAIRS = {
    "h0": ("黑", 0, "", _C(0.08, 0.06, 0.05)), "h1": ("棕", 0, "", _C(0.35, 0.22, 0.12)), "h2": ("金", 0, "", _C(0.9, 0.75, 0.35)),
    "h3": ("银", 30, "", _C(0.85, 0.87, 0.9)), "h4": ("红", 30, "", _C(0.85, 0.2, 0.15)), "h5": ("粉", 30, "", _C(0.98, 0.5, 0.7)),
    "h6": ("蓝", 30, "", _C(0.2, 0.45, 0.95)), "h7": ("绿", 30, "原谅色，自己选的", _C(0.25, 0.75, 0.3)),
}
BODIES = {
    "none": ("标准", 0, "", (1.0, 1.0)),
    "bighead": ("大头", 100, "大头大头，下雨不愁", (1.5, 1.0)),
    "buff": ("筋肉人", 150, "练过，但只练了外观", (1.0, 1.2)),
    "slim": ("竹竿", 150, "风一吹就倒，但是倒不了", (0.95, 0.8)),
    "tiny": ("小头", 100, "九头身", (0.72, 1.0)),
    "mega": ("大头筋肉", 400, "头和肌肉一起练的", (1.45, 1.2)),
}
TRAILS = {
    "none": ("白金", 0, "", [(1.0, 0.95, 0.6)]),
    "fire": ("火焰", 100, "杀球自带特效", [(1.0, 0.35, 0.05), (1.0, 0.8, 0.1)]),
    "ice": ("冰霜", 100, "冷静，冷静", [(0.3, 0.8, 1.0), (0.9, 1.0, 1.0)]),
    "toxic": ("毒气", 150, "对手闻了都说好", [(0.4, 1.0, 0.2), (0.1, 0.5, 0.1)]),
    "love": ("粉红泡泡", 150, "少女心杀球", [(1.0, 0.4, 0.7), (1.0, 0.8, 0.9)]),
    "rainbow": ("彩虹", 300, "七种颜色，七种快乐", [(1.0, 0.2, 0.2), (1.0, 0.7, 0.1), (1.0, 1.0, 0.2), (0.2, 0.9, 0.3), (0.2, 0.6, 1.0),
                                          (0.6, 0.3, 1.0)]),
}
NAMES = {
    "n0": ("你", 0, "", None), "n1": ("键盘侠", 0, "", None), "n2": ("羽球小白", 0, "", None), "n3": ("退堂鼓一级选手", 0, "", None),
    "n4": ("公园球王", 50, "", None), "n5": ("杀球狂魔", 50, "", None), "n6": ("重在参与", 50, "", None), "n7": ("反手战神", 50, "", None),
    "n8": ("野生林丹", 100, "", None), "n9": ("这球我不接", 100, "", None), "n10": ("打工人", 100, "", None), "n11": ("我妈喊我吃饭", 100, "", None),
}
# what the player says: reaction (arcade.REACT key) -> lines
QUIPS = {
    "none": ("不说话", 0, "安静地打球", {}),
    "humble": ("谦虚", 0, "承让承让", dict(won=("承让", "运气运气"), won_big=("承让了", "碰巧碰巧"), won_long=("好球，你也打得好",),
                                    lost_error=("我的我的",), lost_winner=("好球！", "漂亮"), tired=("让我缓缓",))),
    "cocky": ("嚣张", 60, "就这？", dict(won=("就这？", "谢谢老板"), won_big=("就这？", "接啊，怎么不接"), won_long=("还有谁！", "我要打十个"),
                                   lost_error=("让你一个", "热身而已"), lost_winner=("风吹的", "地滑"), tired=("我没累，我装的",))),
    "versailles": ("凡尔赛", 100, "哎，随便打打", dict(won=("哎，又赢了", "没发力呢"), won_big=("随手一拍", "这也能得分？"), won_long=("打这么久，手都酸了",),
                                              lost_error=("昨晚没睡好",), lost_winner=("拍子线松了",), tired=("今天只发挥了三成",))),
    "worker": ("打工人", 100, "下班了才来的", dict(won=("今日 KPI 达成",), won_big=("这球算加班费",), won_long=("比开会还累",),
                                            lost_error=("这锅我背",), lost_winner=("老板说得对",), tired=("想下班",))),
    "chuuni": ("中二", 150, "封印解除", dict(won=("愚蠢的凡人",), won_big=("感受痛苦吧", "这就是我的领域"), won_long=("吾之右手，苏醒了",),
                                      lost_error=("封印还没解开",), lost_winner=("有点意思",), tired=("魔力不足……",))),
    "meme": ("抽象", 200, "家人们谁懂啊", dict(won=("家人们谁懂啊", "泰裤辣"), won_big=("遥遥领先", "退！退！退！"), won_long=("我真的会谢", "尊嘟假嘟"),
                                       lost_error=("我没惹你们任何人", "栓 Q"), lost_winner=("你礼貌吗", "绝绝子"), tired=("蚌埠住了", "已老实，求放过"))),
}

SLOTS = [("suit", "套装", SUITS), ("head", "头饰", HEADS), ("face", "脸", FACES), ("back", "背后", BACKS), ("racket", "手里拿的", RACKETS),
         ("body", "体型", BODIES), ("shirt", "上衣", SHIRTS), ("shorts", "短裤", SHORTS), ("shoes", "球鞋", SHOES), ("skin", "肤色", SKINS),
         ("hair", "发色", HAIRS), ("trail", "球的尾迹", TRAILS), ("name", "名字", NAMES), ("quip", "口头禅", QUIPS)]
CATALOG = {k: items for k, _, items in SLOTS}
DEFAULT = dict(suit="none", head="none", face="none", back="none", racket="none", body="none", shirt="c3", shorts="c10", shoes="c0",
               skin="s0", hair="h0", trail="none", name="n0", quip="none")


def item(slot, key):
    """(name, price, line, data) of an item; the slot's first item when the key is unknown."""
    items = CATALOG[slot]
    return items.get(key) or next(iter(items.values()))


def clean(outfit):
    """A complete outfit with only known items (an old or hand-edited save)."""
    out = dict(DEFAULT)
    for k, v in (outfit or {}).items():
        if k in CATALOG and v in CATALOG[k]:
            out[k] = v
    return out


def price(slot, key):
    return item(slot, key)[1]


def trail(outfit):
    return item("trail", (outfit or {}).get("trail", "none"))[3]


def name(outfit):
    return item("name", (outfit or {}).get("name", "n0"))[0]


def quips(outfit):
    return item("quip", (outfit or {}).get("quip", "none"))[3]


RANKS = [(0, "路人甲"), (1, "有点想法"), (250, "显眼包"), (700, "行为艺术家"), (1400, "抽象大师"), (2400, "抽象之神")]


def score(outfit):
    """(how odd the outfit is: the prices of what is worn, what that makes the wearer)."""
    o = clean(outfit)
    v = sum(price(k, o[k]) for k in CATALOG) + 20 * sum(1 for k in ("suit", "head", "face", "back", "racket", "body")
                                                         if o[k] != "none")
    return v, [r for lo, r in RANKS if v >= lo][-1]


def swatch(slot, key):
    """Colour of a colour item (r, g, b in 0..1), None for the rest."""
    d = item(slot, key)[3]
    if slot == "trail":
        return d[0]
    return d[:3] if slot in ("shirt", "shorts", "shoes", "skin", "hair") else None


def accent(outfit):
    """The colour that stands for this outfit in the HUD (0-255 BGR-free tuple r, g, b)."""
    o = clean(outfit)
    suit = SUITS[o["suit"]][3]
    c = suit.get("shirt") if o["suit"] != "none" else None
    c = c or SHIRTS[o["shirt"]][3]
    if max(c[:3]) < 0.35:
        c = (0.3, 0.55, 1.0, 1)
    return tuple(int(round(255 * v)) for v in c[:3])


class Look:
    """An outfit resolved for character.add_character."""

    def __init__(self, outfit):
        self.given = set(outfit)                 # (an opponent's outfit names only what differs from its style)
        self.o = clean(outfit)
        self.head, self.girth = BODIES[self.o["body"]][3]

    def paint(self, paint):
        """Colours of the roles (before the body is built)."""
        o, c = self.o, paint.colors
        for slot, role, table in (("skin", "skin", SKINS), ("hair", "hair", HAIRS), ("shirt", "shirt", SHIRTS),
                                  ("shorts", "shorts", SHORTS), ("shoes", "shoe", SHOES)):
            if slot in self.given:
                c[role] = table[o[slot]][3]
        suit = SUITS[o["suit"]]
        for role, rgba in suit[3].items():
            if rgba is None:                                     # None: bare (the skin colour)
                c[role] = c["skin"]
            else:
                c[role] = rgba
        if o["suit"] in SUIT_SHINE:
            for role in ("shirt", "sleeve", "shorts", "arm", "leg", "hand", "face", "hair", "shoe", "sock"):
                paint.shine[role] = SUIT_SHINE[o["suit"]]

    def dress(self, paint, hand):
        """The pieces (after the body is built, before it is scaled)."""
        k = Kit(paint, hand)
        for slot, table in (("suit", SUITS), ("head", HEADS), ("face", FACES), ("back", BACKS), ("racket", RACKETS)):
            build = table[self.o[slot]][-1]
            if build is not None and (slot != "racket" or "racket" in paint.parts):
                build(k)

"""The match venue: a dark indoor arena with a spot-lit court, LED boards and stands.

Everything is procedural (PIL textures + MuJoCo primitives), so the game ships without art assets.
No real sponsors or trade marks: the boards carry the game's own wordmarks.
"""
from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .court import (DOUBLES_LONG_SERVICE, LENGTH_HALF, LINE_W, NET_DEPTH, NET_TOP_POST, SHORT_SERVICE,
                    WIDTH_DOUBLES_HALF, WIDTH_SINGLES_HALF, net_height)
from .fonts import font

MAT_GREEN = (30, 128, 92)
SURROUND = (22, 52, 86)
FLOOR = (13, 17, 26)
BOARD_X = WIDTH_DOUBLES_HALF + 2.7          # side LED boards
BOARD_Y = LENGTH_HALF + 2.7                 # end LED boards
BOARD_H = 0.86


# ---------------------------------------------------------------------------- textures
def _tex(spec, name, img: Image.Image):
    img = img.convert("RGB")
    t = spec.add_texture()
    t.name = name
    t.type = mujoco.mjtTexture.mjTEXTURE_2D
    t.width, t.height, t.nchannel = img.width, img.height, 3
    t.data = np.asarray(img, np.uint8).tobytes()               # shown upright on the local -y face of a box
    return name


def _material(spec, name, tex=None, rgba=(1, 1, 1, 1), emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0,
              repeat=(1, 1)):
    m = spec.add_material()
    m.name = name
    m.rgba = rgba
    m.emission, m.specular, m.shininess, m.reflectance = emission, specular, shininess, reflectance
    if tex is not None:
        tx = list(m.textures)
        tx[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = tex
        m.textures = tx
        m.texrepeat = list(repeat)
    return name


def _noise(w, h, amp, rng, blur=0.0):
    n = rng.normal(0, amp, (h, w))
    im = Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im, np.float32) - 128


def mat_texture(rng, px=96):
    """Court mat and its surround, seen from above (x right, y up), with the pool of light baked in."""
    hx, hy = WIDTH_DOUBLES_HALF + 2.3, LENGTH_HALF + 2.3
    w, h = int(2 * hx * px), int(2 * hy * px)
    a = np.zeros((h, w, 3), np.float32)
    a[:] = SURROUND
    x0, x1 = int((hx - WIDTH_DOUBLES_HALF) * px), int((hx + WIDTH_DOUBLES_HALF) * px)
    y0, y1 = int((hy - LENGTH_HALF) * px), int((hy + LENGTH_HALF) * px)
    a[y0:y1, x0:x1] = MAT_GREEN
    # texture of the PVC mat: fine grain + the seams between the rolls (every 1.8 m, along the court)
    a += _noise(w, h, 5.0, rng)[..., None]
    a += _noise(w, h, 10.0, rng, blur=18)[..., None] * 0.6
    for xs in np.arange(-hx, hx, 1.8):
        c = int((xs + hx) * px)
        a[:, max(c - 1, 0):c + 1] *= 0.93
    # lighting: bright over the court, falling off into the run-off
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h / 2) / (h * 0.60)) ** 2)
    a *= np.clip(1.12 - 0.55 * r ** 2.2, 0.35, 1.2)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def floor_texture(rng, n=512):
    a = np.zeros((n, n, 3), np.float32)
    a[:] = FLOOR
    a += _noise(n, n, 3.0, rng)[..., None]
    a += _noise(n, n, 8.0, rng, blur=24)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


BOARD_THEMES = [((14, 96, 190), (255, 255, 255)), ((230, 236, 245), (16, 40, 84)), ((200, 30, 52), (255, 255, 255)),
                ((16, 22, 34), (255, 204, 40)), ((12, 132, 110), (255, 255, 255))]
BOARD_WORDS = ["羽你无瓜", "整活到底", "SMASH LAB", "键盘侠杯", "BADMINTON LIVE", "今日宜杀球", "MUJOCO ARENA",
               "输了换身衣服"]


def board_texture(words, themes, panel=(512, 128)):
    """A strip of LED panels: one wordmark each."""
    pw, ph = panel
    im = Image.new("RGB", (pw * len(words), ph))
    d = ImageDraw.Draw(im)
    for i, (txt, (bg, fg)) in enumerate(zip(words, themes)):
        x = i * pw
        d.rectangle([x, 0, x + pw - 1, ph - 1], fill=bg)
        d.rectangle([x, ph - 10, x + pw - 1, ph - 1], fill=tuple(int(c * 0.8) for c in bg))
        cjk = any(ord(c) > 0x2E80 for c in txt)
        f = font(64 if cjk else 60, bold=True, cjk=cjk)
        bb = d.textbbox((0, 0), txt, font=f)
        if bb[2] - bb[0] > pw - 40:
            f = font(int((64 if cjk else 60) * (pw - 40) / (bb[2] - bb[0])), bold=True, cjk=cjk)
            bb = d.textbbox((0, 0), txt, font=f)
        d.text((x + (pw - (bb[2] - bb[0])) / 2 - bb[0], (ph - 10 - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=f, fill=fg)
        d.line([x, 0, x, ph], fill=(0, 0, 0), width=3)
    return im


def crowd_texture(rng, w=1024, h=256, rows=7):
    """Rows of seated spectators in a dark stand: heads and shoulders as soft blobs."""
    im = Image.new("RGB", (w, h), (9, 11, 17))
    d = ImageDraw.Draw(im)
    rh = h / rows
    palette = [(150, 40, 46), (40, 70, 140), (190, 190, 196), (36, 38, 46), (176, 136, 44), (44, 112, 86), (90, 60, 110),
               (60, 60, 70), (24, 24, 30)]
    for r in range(rows):
        y = h - (r + 1) * rh
        d.rectangle([0, y + rh * 0.86, w, y + rh], fill=(20, 24, 34))            # the step
        x = rng.uniform(0, 14)
        dim = 0.34 + 0.5 * (rows - r) / rows                                     # the back rows sink into the dark
        while x < w:
            if rng.random() < 0.86:
                c = palette[int(rng.integers(len(palette)))]
                c = tuple(int(v * dim * rng.uniform(0.7, 1.1)) for v in c)
                skin = tuple(int(v * dim) for v in (200, 160, 130))
                bw = rng.uniform(15, 19)
                d.rounded_rectangle([x, y + rh * 0.42, x + bw, y + rh * 0.9], radius=5, fill=c)
                d.ellipse([x + bw * 0.24, y + rh * 0.1, x + bw * 0.76, y + rh * 0.48], fill=skin)
                d.pieslice([x + bw * 0.22, y + rh * 0.06, x + bw * 0.78, y + rh * 0.44], 180, 360,
                           fill=tuple(int(v * dim) for v in (30, 24, 22)))
            x += rng.uniform(19, 25)
    return im.filter(ImageFilter.GaussianBlur(0.7))


def screen_texture(w=1024, h=384):
    """The big screen over the far stand."""
    im = Image.new("RGB", (w, h), (6, 10, 20))
    d = ImageDraw.Draw(im)
    for i in range(h):
        t = i / h
        d.line([0, i, w, i], fill=(int(8 + 10 * t), int(22 + 40 * t), int(52 + 70 * t)))
    for k in range(-6, 16):                                                     # diagonal light streaks
        d.polygon([(k * 90, h), (k * 90 + 40, h), (k * 90 + 240, 0), (k * 90 + 200, 0)], fill=(16, 52, 104))
    f1, f2 = font(150, bold=True, cjk=True), font(54, bold=True, cjk=True)
    t1 = "羽你无瓜"
    bb = d.textbbox((0, 0), t1, font=f1)
    d.text(((w - bb[2] + bb[0]) / 2 - bb[0], 56 - bb[1]), t1, font=f1, fill=(255, 255, 255))
    t2 = "抽象羽毛球 · 键盘侠杯"
    bb = d.textbbox((0, 0), t2, font=f2)
    d.text(((w - bb[2] + bb[0]) / 2 - bb[0], 250 - bb[1]), t2, font=f2, fill=(255, 208, 60))
    d.rectangle([0, 0, w - 1, h - 1], outline=(40, 44, 54), width=10)
    return im


# ---------------------------------------------------------------------------- geometry
def _box(body, size, pos, rgba=(1, 1, 1, 1), material=None, quat=None):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = size
    g.pos = pos
    if quat is not None:
        g.quat = quat
    if material is not None:
        g.material = material
    else:
        g.rgba = rgba
    g.contype = g.conaffinity = 0
    return g


def _cyl(body, p0, p1, r, rgba):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.fromto = np.r_[p0, p1]
    g.size = [r, 0, 0]
    g.rgba = rgba
    g.contype = g.conaffinity = 0
    return g


def _quat(yaw_deg, roll_deg=0.0):
    """Yaw about z, then roll about the turned x axis. A textured box reads correctly on its local
    -y face, so: yaw 0 faces -y (the far end looking at the court), -90 faces -x, 90 faces +x."""
    a, b = np.radians(yaw_deg) / 2, np.radians(roll_deg) / 2
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array([np.cos(a), 0, 0, np.sin(a)]), np.array([np.cos(b), np.sin(b), 0, 0]))
    return out


def _lines_and_net(spec):
    wb = spec.worldbody
    white = _material(spec, "line_white", rgba=(0.96, 0.96, 0.94, 1), emission=0.25, specular=0.0)
    z, h = 0.0032, 0.001
    for x in (-WIDTH_DOUBLES_HALF, -WIDTH_SINGLES_HALF, WIDTH_SINGLES_HALF, WIDTH_DOUBLES_HALF):
        _box(wb, [LINE_W / 2, LENGTH_HALF, h], [x, 0, z], material=white)
    for y in (-LENGTH_HALF, -DOUBLES_LONG_SERVICE, -SHORT_SERVICE, SHORT_SERVICE, DOUBLES_LONG_SERVICE, LENGTH_HALF):
        _box(wb, [WIDTH_DOUBLES_HALF, LINE_W / 2, h], [0, y, z], material=white)
    for sgn in (-1, 1):
        y0, y1 = sgn * SHORT_SERVICE, sgn * LENGTH_HALF
        _box(wb, [LINE_W / 2, abs(y1 - y0) / 2, h], [0, (y0 + y1) / 2, z], material=white)
    # net: posts, a dark translucent mesh with a few cords, white tape
    post = (0.78, 0.8, 0.84, 1)
    for sx in (-1, 1):
        x = sx * (WIDTH_DOUBLES_HALF + 0.02)
        _cyl(wb, [x, 0, 0], [x, 0, NET_TOP_POST], 0.022, post)
        _box(wb, [0.16, 0.2, 0.012], [x + sx * 0.1, 0, 0.012], (0.2, 0.22, 0.26, 1))
    xs = np.linspace(-WIDTH_DOUBLES_HALF, WIDTH_DOUBLES_HALF, 9)
    for a, b in zip(xs[:-1], xs[1:]):
        top = net_height(0.5 * (a + b))
        _box(wb, [(b - a) / 2, 0.002, NET_DEPTH / 2], [(a + b) / 2, 0, top - NET_DEPTH / 2], (0.16, 0.1, 0.08, 0.5))
        _box(wb, [(b - a) / 2, 0.006, 0.019], [(a + b) / 2, 0, top - 0.019], material=white)
    cord = (0.34, 0.23, 0.16, 0.5)
    for k in range(1, 5):
        _box(wb, [WIDTH_DOUBLES_HALF, 0.0026, 0.003], [0, 0, net_height(0) - 0.03 - k * (NET_DEPTH - 0.03) / 5], cord)
    for x in np.linspace(-WIDTH_DOUBLES_HALF, WIDTH_DOUBLES_HALF, 21)[1:-1]:
        _box(wb, [0.003, 0.0026, NET_DEPTH / 2], [x, 0, net_height(x) - NET_DEPTH / 2], cord)


def _umpire_chair(spec, x):
    wb = spec.worldbody
    metal, seat = (0.72, 0.74, 0.78, 1), (0.1, 0.25, 0.5, 1)
    for dx in (-0.28, 0.28):
        for dy in (-0.3, 0.3):
            _cyl(wb, [x + dx * 1.25, dy * 1.25, 0], [x + dx, dy, 1.45], 0.02, metal)
    _box(wb, [0.3, 0.32, 0.025], [x, 0, 1.47], seat)
    _box(wb, [0.03, 0.32, 0.28], [x + 0.3, 0, 1.76], seat)
    _box(wb, [0.3, 0.2, 0.012], [x - 0.12, 0, 1.78], (0.85, 0.86, 0.88, 1))            # score desk


def _bench(spec, x, y, colour):
    wb = spec.worldbody
    _box(wb, [0.22, 0.75, 0.03], [x, y, 0.44], (0.16, 0.17, 0.2, 1))
    for dy in (-0.6, 0.6):
        _box(wb, [0.2, 0.03, 0.21], [x, y + dy, 0.21], (0.3, 0.31, 0.34, 1))
    _box(wb, [0.18, 0.36, 0.14], [x, y - 0.2, 0.61], colour)                           # racket bag


def _stand(spec, material, base, yaw_deg, half_len, length=6.2, slope_deg=32.0, z0=0.95):
    """A bank of seats as one sloped textured slab rising away from the court, on a dark plinth.
    base: where its foot meets the floor (centre of the long side)."""
    wb = spec.worldbody
    out = np.array([-np.sin(np.radians(yaw_deg)), np.cos(np.radians(yaw_deg)), 0.0])  # away from the court
    phi = np.radians(slope_deg)
    c = np.asarray(base, float) + out * (np.cos(phi) * length / 2) + [0, 0, z0 + np.sin(phi) * length / 2]
    _box(wb, [half_len, 0.03, length / 2], c, material=material, quat=_quat(yaw_deg, -(90.0 - slope_deg)))
    _box(wb, [half_len, 0.06, z0 / 2], np.asarray(base, float) + [0, 0, z0 / 2], (0.035, 0.04, 0.06, 1), quat=_quat(yaw_deg))


def add_arena(spec, seed=7, quality="high"):
    """quality: "fast" (integrated graphics: coarser round shapes, no floor reflection), "high"
    (default), "ultra" (offline recording). The cost is almost all in the two characters' ~230
    curved geoms, drawn once per pass (main, shadow, reflection)."""
    rng = np.random.default_rng(seed)
    wb = spec.worldbody
    v = spec.visual
    v.map.znear = 0.01
    v.map.zfar = 90.0
    v.quality.shadowsize = 4096
    v.quality.offsamples = 4
    v.quality.numslices, v.quality.numstacks, v.quality.numquads = {"fast": (14, 8, 2), "high": (18, 10, 3)}.get(
        quality, (28, 16, 4))
    v.headlight.ambient = [0.2, 0.2, 0.22]
    v.headlight.diffuse = [0.14, 0.14, 0.15]
    v.headlight.specular = [0.0, 0.0, 0.0]
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = [0.03, 0.04, 0.07], [0.0, 0.0, 0.01]
    sky.width = sky.height = 128

    # floor, court mat
    _material(spec, "hall_floor", _tex(spec, "hall_floor", floor_texture(rng)), specular=0.3, shininess=0.6,
              reflectance=0.0 if quality == "fast" else 0.14, repeat=(14, 14))
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [45, 45, 0.1]
    g.material = "hall_floor"
    g.contype = g.conaffinity = 0
    _material(spec, "court_mat", _tex(spec, "court_mat", mat_texture(rng)), specular=0.12, shininess=0.25)
    _box(wb, [WIDTH_DOUBLES_HALF + 2.3, LENGTH_HALF + 2.3, 0.001], [0, 0, 0.001], material="court_mat")
    _lines_and_net(spec)

    # LED boards: both sides, the far end, and the corners of the near end (the main camera looks
    # over the middle of the near end)
    order = rng.permutation(len(BOARD_WORDS))
    used = [0]

    def strip(n):
        words = [BOARD_WORDS[order[(used[0] + i) % len(order)]] for i in range(n)]
        themes = [BOARD_THEMES[(used[0] + i) % len(BOARD_THEMES)] for i in range(n)]
        used[0] += n
        name = f"board{used[0]}"
        _material(spec, name, _tex(spec, name, board_texture(words, themes)), emission=0.8, specular=0.0)
        return name

    dark = (0.04, 0.04, 0.05, 1)
    for sx in (-1, 1):
        _box(wb, [BOARD_Y, 0.05, BOARD_H / 2], [sx * BOARD_X, 0, BOARD_H / 2], material=strip(6), quat=_quat(-90 * sx))
        _box(wb, [0.75, 0.05, BOARD_H / 2], [sx * (BOARD_X - 0.75), -BOARD_Y, BOARD_H / 2], material=strip(1),
             quat=_quat(180))
    _box(wb, [BOARD_X, 0.05, BOARD_H / 2], [0, BOARD_Y, BOARD_H / 2], material=strip(4))

    # stands (far end and both sides; the near end is where the main camera sits) and the big screen
    for i, (base, yaw, half) in enumerate((([0, BOARD_Y + 1.3, 0], 0, BOARD_X + 4.5),
                                           ([BOARD_X + 1.3, 0, 0], -90, BOARD_Y + 1.5),
                                           ([-BOARD_X - 1.3, 0, 0], 90, BOARD_Y + 1.5))):
        nm = f"crowd{i}"
        _material(spec, nm, _tex(spec, nm, crowd_texture(rng, w=2048, h=512, rows=9)), emission=0.42, specular=0.0)
        _stand(spec, nm, base, yaw, half)
    _material(spec, "screen", _tex(spec, "screen", screen_texture()), emission=0.9, specular=0.0)
    _box(wb, [2.6, 0.06, 0.975], [0, BOARD_Y + 7.0, 5.45], material="screen")
    _box(wb, [2.75, 0.08, 1.1], [0, BOARD_Y + 7.12, 5.45], dark)

    # court-side furniture
    _umpire_chair(spec, WIDTH_DOUBLES_HALF + 0.95)
    _bench(spec, -(WIDTH_DOUBLES_HALF + 1.75), -3.2, (0.1, 0.3, 0.75, 1))
    _bench(spec, -(WIDTH_DOUBLES_HALF + 1.75), 3.2, (0.8, 0.15, 0.2, 1))

    # lighting rig: two rows of floodlights along the court (as in a real hall there is no light
    # behind the baselines, where players would look into it); one of them casts the shadows
    _material(spec, "lamp", rgba=(1, 1, 0.96, 1), emission=1.0, specular=0.0)
    for sx in (-1, 1):
        _box(wb, [0.3, LENGTH_HALF + 1.5, 0.05], [sx * 4.4, 0, 10.2], dark)
        _box(wb, [0.22, LENGTH_HALF + 1.2, 0.02], [sx * 4.4, 0, 10.14], material="lamp")
    lights = [([-4.4, -3.4, 10.0], [0.34, 0.2, -1.0], 0.52, True), ([4.4, 3.4, 10.0], [-0.34, -0.2, -1.0], 0.46, False),
              ([4.4, -3.4, 10.0], [-0.34, 0.2, -1.0], 0.3, False), ([-4.4, 3.4, 10.0], [0.34, -0.2, -1.0], 0.3, False)]
    for pos, dr, power, shadow in lights:
        lt = wb.add_light()
        lt.pos = pos
        lt.dir = dr
        lt.type = mujoco.mjtLightType.mjLIGHT_SPOT
        lt.diffuse = [power, power, power * 0.97]
        lt.specular = [0.25 * power] * 3
        lt.castshadow = shadow
        lt.cutoff = 58
        lt.exponent = 6
        lt.attenuation = [1.0, 0.0, 0.0]

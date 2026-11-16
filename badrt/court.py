"""BWF court geometry and the match scene (court, net, two characters, shuttle)."""
from __future__ import annotations

import mujoco
import numpy as np

from .character import add_character, add_light, new_spec

# BWF Laws of Badminton, metres. Net plane y = 0; player (near) side y < 0.
LENGTH_HALF = 6.70
WIDTH_DOUBLES_HALF = 3.05
WIDTH_SINGLES_HALF = 2.59
SHORT_SERVICE = 1.98
DOUBLES_LONG_SERVICE = 6.70 - 0.76
NET_TOP_CENTER = 1.524
NET_TOP_POST = 1.55
NET_DEPTH = 0.76
LINE_W = 0.04


def net_height(x):
    """Net top (it sags from 1.55 m at the posts to 1.524 m in the middle)."""
    return NET_TOP_CENTER + (NET_TOP_POST - NET_TOP_CENTER) * np.clip(abs(x) / WIDTH_DOUBLES_HALF, 0, 1) ** 2


def in_singles(x, y):
    return abs(x) <= WIDTH_SINGLES_HALF + LINE_W / 2 and abs(y) <= LENGTH_HALF + LINE_W / 2


def _box(body, size, pos, rgba, name=None):
    g = body.add_geom()
    if name:
        g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = size
    g.pos = pos
    g.rgba = rgba
    g.contype = g.conaffinity = 0
    return g


def add_court(spec, floor_half=14.0):
    wb = spec.worldbody
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [floor_half, floor_half, 0.1]
    g.rgba = [0.12, 0.2, 0.36, 1]
    g.contype = g.conaffinity = 0
    _box(wb, [WIDTH_DOUBLES_HALF + 2.0, LENGTH_HALF + 2.3, 0.0008], [0, 0, 0.0008], (0.09, 0.42, 0.33, 1))
    _box(wb, [WIDTH_DOUBLES_HALF + 0.02, LENGTH_HALF + 0.02, 0.001], [0, 0, 0.001], (0.1, 0.5, 0.38, 1))
    white = (0.95, 0.95, 0.95, 1)
    z, h = 0.0025, 0.0012
    for x in (-WIDTH_DOUBLES_HALF, -WIDTH_SINGLES_HALF, WIDTH_SINGLES_HALF, WIDTH_DOUBLES_HALF):
        _box(wb, [LINE_W / 2, LENGTH_HALF, h], [x, 0, z], white)
    for y in (-LENGTH_HALF, -DOUBLES_LONG_SERVICE, -SHORT_SERVICE, SHORT_SERVICE, DOUBLES_LONG_SERVICE, LENGTH_HALF):
        _box(wb, [WIDTH_DOUBLES_HALF, LINE_W / 2, h], [0, y, z], white)
    for sgn in (-1, 1):
        y0, y1 = sgn * SHORT_SERVICE, sgn * LENGTH_HALF
        _box(wb, [LINE_W / 2, abs(y1 - y0) / 2, h], [0, (y0 + y1) / 2, z], white)
    # net: mesh panel, white tape and posts
    for sx in (-1, 1):
        _box(wb, [0.02, 0.02, NET_TOP_POST / 2], [sx * (WIDTH_DOUBLES_HALF + 0.02), 0, NET_TOP_POST / 2],
             (0.85, 0.85, 0.85, 1))
    xs = np.linspace(-WIDTH_DOUBLES_HALF, WIDTH_DOUBLES_HALF, 13)
    for a, b in zip(xs[:-1], xs[1:]):
        top = net_height(0.5 * (a + b))
        _box(wb, [(b - a) / 2, 0.003, NET_DEPTH / 2], [(a + b) / 2, 0, top - NET_DEPTH / 2], (0.1, 0.1, 0.1, 0.55))
        _box(wb, [(b - a) / 2, 0.006, 0.02], [(a + b) / 2, 0, top - 0.02], white)


def add_hall(spec, half_x=14.0, half_y=14.0, height=9.0):
    """Gradient sky plus four dark walls with a light band, like an indoor sports hall."""
    t = spec.add_texture()
    t.name = "sky"
    t.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    t.rgb1, t.rgb2 = [0.55, 0.6, 0.68], [0.08, 0.09, 0.12]
    t.width = t.height = 256
    wb = spec.worldbody
    wall = (0.11, 0.2, 0.32, 1)
    band = (0.75, 0.78, 0.82, 1)
    for sx in (-1, 1):
        _box(wb, [0.1, half_y, height / 2], [sx * half_x, 0, height / 2], wall)
        _box(wb, [0.11, half_y, 0.15], [sx * half_x, 0, 2.4], band)
    for sy in (-1, 1):
        _box(wb, [half_x, 0.1, height / 2], [0, sy * half_y, height / 2], wall)
        _box(wb, [half_x, 0.11, 0.15], [0, sy * half_y, 2.4], band)


def _shuttle_skirt_mesh(spec, name="shuttle_skirt", n=16):
    """Truncated cone: cork end radius 0.013 at z=0, skirt radius 0.033 at z=-0.07 (local z = cork axis)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = np.column_stack([0.013 * np.cos(ang), 0.013 * np.sin(ang), np.zeros(n)])
    bot = np.column_stack([0.033 * np.cos(ang), 0.033 * np.sin(ang), np.full(n, -0.07)])
    verts = np.vstack([top, bot])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + i], [j, n + j, n + i]]
    m = spec.add_mesh()
    m.name = name
    m.uservert = verts.ravel().tolist()
    m.userface = np.array(faces).ravel().tolist()
    return name


def add_shuttle(spec):
    mesh = _shuttle_skirt_mesh(spec)
    b = spec.worldbody.add_body()
    b.name = "shuttle"
    b.mocap = True
    b.pos = [0, 0, -1]
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.014, 0, 0]
    g.pos = [0, 0, 0.012]
    g.rgba = [0.95, 0.9, 0.8, 1]
    g.contype = g.conaffinity = 0
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh
    g.rgba = [1, 1, 1, 0.9]
    g.contype = g.conaffinity = 0
    # a dark marker on the skirt makes the spin visible
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.002, 0.012, 0.025]
    g.pos = [0.025, 0, -0.045]
    g.rgba = [0.8, 0.1, 0.1, 1]
    g.contype = g.conaffinity = 0


def add_player(spec, prefix, style=None, handed=None):
    """Character dressed and sized after a styles.Style (None: default look)."""
    if style is None:
        return add_character(spec.worldbody, prefix, spec=spec, handed=handed or "right")
    return add_character(spec.worldbody, prefix, shirt=style.shirt, shorts=style.shorts, skin=style.skin,
                         hair=style.hair, racket_color=style.accent, accent=style.accent, spec=spec,
                         dims=style.dims, handed=handed or style.handed, outfit=style.outfit or None)


def build_match_model(width=1280, height=720, player=None, opponent=None, player_handed=None, arena=True):
    """Match scene: player "p_" (near half), opponent "o_" (far half), shuttle (mocap body).
    player / opponent: styles.Style (looks, size, racket hand). arena: True | "high" | "fast" for
    the game venue (arena.add_arena), False for the plain training hall."""
    spec = new_spec(width, height)
    spec.visual.map.znear = 0.01
    if arena:
        from .arena import add_arena
        add_arena(spec, quality="high" if arena is True else str(arena))
    else:
        add_light(spec)
        add_court(spec, floor_half=25.0)
        add_hall(spec, half_x=16.0, half_y=20.0, height=12.0)
    add_player(spec, "p_", player, player_handed)
    add_player(spec, "o_", opponent)
    add_shuttle(spec)
    return spec.compile()


def game_view(aspect=16 / 9, pos=(0.0, -13.8, 2.35), lookat=(0.0, -2.2, 0.55), margin=0.06):
    """Low camera behind the camera-side baseline, like a broadcast main camera but closer.
    The vertical FOV is the smallest that keeps both camera-side court corners (with run-off), the
    net posts and the far baseline inside the image. Returns dict(lookat, distance, azimuth,
    elevation, fovy) for an mjvCamera plus model.vis.global_.fovy."""
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    f = lookat - pos
    dist = float(np.linalg.norm(f)); f /= dist
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    pts = [(sx * (WIDTH_DOUBLES_HALF + 0.4), -LENGTH_HALF - 0.5, 0.0) for sx in (-1, 1)]
    pts += [(sx * WIDTH_DOUBLES_HALF, 0.0, NET_TOP_POST + 0.6) for sx in (-1, 1)]
    pts += [(sx * WIDTH_DOUBLES_HALF, LENGTH_HALF, 0.0) for sx in (-1, 1)]
    tan_v = 0.0
    for p in pts:
        v = np.asarray(p) - pos
        z = v @ f
        tan_v = max(tan_v, abs(v @ up) / z, abs(v @ right) / z / aspect)
    fovy = float(np.degrees(2 * np.arctan(tan_v * (1 + margin))))
    return dict(lookat=lookat.tolist(), distance=dist, azimuth=float(np.degrees(np.arctan2(f[1], f[0]))),
                elevation=float(np.degrees(np.arcsin(f[2]))), fovy=fovy)


def build_broadcast_model(width=960, height=544, shirt=(0.08, 0.08, 0.1, 1), shorts=(0.08, 0.08, 0.1, 1),
                          accent=(0.95, 0.3, 0.45, 1), dims=None):
    """Court, hall and the camera-side player only (prefix "p_"); dims: character.personalize()."""
    spec = new_spec(width, height)
    add_light(spec)
    add_court(spec, floor_half=30.0)
    add_hall(spec, half_x=16.0, half_y=24.0, height=12.0)
    add_character(spec.worldbody, "p_", shirt=shirt, shorts=shorts, racket_color=(0.95, 0.85, 0.15, 1),
                  accent=accent, spec=spec, dims=dims)
    return spec.compile()

"""The game: menus, the controller, match loop, recording.

    title -> mode -> quick match:  choose opponent -> settings -> (walk-on) match -> result
                     campaign:     twelve stages (modes.CAMPAIGN) -> match -> result
                     ladder:       ten odd characters (modes.LADDER) who play with the tactics learned in self-play
                                   (brain.py) and go on learning the player's habits -> match -> result
                     training:     drill list (modes.DRILLS) -> drill -> result
             wardrobe (Tab):       what the player's character wears (wardrobe.py), bought with the coins of matches
    Stars, coins, the wardrobe and what the opponents have learned are kept in save/progress.json.

The game is played with a controller: the keyboard (arrows / WASD aim, J K L shots, Space jump,
Ctrl dive) or the on-screen stick and buttons (mouse / touch). The body is the same procedural athlete the
opponents use (opponent.PadPlayer). The only other thing that can stand at the near end is a second AI
(exhibition). Camera and video input are not part of the game any more (live.py keeps that code for the
retargeting tools).

The frame loop is the same with a window (wall clock) and headless (fixed time step, for recording
videos and for tests); menus can be driven by a key script (`auto`), which is how the demo videos
show the whole flow.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import brain, modes, wardrobe
from . import ui as U
from .court import game_view
from .game import Match
from .library import load_library
from .opponent import PadBot
from .styles import DIFFICULTY, PLAYER_STYLE, ROSTER, SHOT_CN, get_style
from .viz import SceneRenderer

ROOT = Path(__file__).resolve().parents[1]

DIFF_KEYS = list(DIFFICULTY)                     # easy, normal, pro, legend
MODES = ["quick", "campaign", "ladder", "training"]
POINTS = [11, 15, 21]
SOURCES = ["pad", "ai"]
MOVES = ["auto", "stick"]
TACTICS = ["style", "brain"]
QUALITIES = ["fast", "high", "ultra"]

KEYS = {2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down", 13: "enter", 10: "enter", 27: "esc",
        32: "space", 9: "tab", ord("a"): "left", ord("d"): "right", ord("w"): "up", ord("s"): "down"}
for _c in "crhfqjklpiot":
    KEYS[ord(_c)] = _c
    KEYS[ord(_c.upper())] = _c


@dataclass
class Settings:
    opponent: int = 0
    handed: int = 0          # 0 right, 1 left
    difficulty: int = 0
    points: int = 2
    source: int = 0          # 0 the controller, 1 an AI plays in the player's place (exhibition)
    quality: int = 1
    move: int = 0            # controller: 0 automatic footwork, 1 run with the stick
    tactics: int = 0         # quick match: 0 the style's hand-written tactics, 1 the latest self-play generation

    def rows(self, mode="quick"):
        """[(attribute, title, options, note)] of the settings screen (campaign / training: what is left to choose)."""
        src = SOURCES[self.source] if mode == "quick" else "pad"
        gens = len(brain.generations())
        note_t = ["对手按自己的风格出球（手写战术）",
                  (f"对手用自对弈练出来的战术（第 {gens - 1} 代），并且边打边记你的习惯" if gens > 1
                   else "还没有训练结果：先运行 scripts/train_selfplay.py")][self.tactics]
        notes_d = ["对手慢半拍、失误多、很少下杀手 —— 适合第一次上手", "对手会调动你，抓住机会才进攻", "职业级：全速、几乎不失误、有机会就杀",
                   "传奇：看你的站位和重心打反方向，够不着的球会鱼跃去救，体力也更好"]
        notes_s = ["键盘 / 屏幕按键：方向键瞄准，J 轻 · K 高 · L 力，空格起跳，Ctrl 鱼跃", "AI 代打：坐下来看两位传奇对决"]
        notes_m = ["角色自己跑到击球点，方向键只管瞄准落点；你选球种、抓时机", "跑位也交给你：方向键 / 摇杆跑到落点，再按键击球（按键那一刻的方向同时是瞄准）"]
        if mode != "quick":
            return [
                ("handed", "惯用手", ["右手", "左手"], "你用哪只手握拍 —— 角色、发球站位和击球判定都会跟着换边"),
                ("move", "跑位", ["自动", "手动"], notes_m[self.move]),
                ("quality", "画质", ["流畅", "精致", "极致"], "只有集成显卡时建议“流畅”；独立显卡三档都流畅"),
            ]
        return [
            ("handed", "惯用手", ["右手", "左手"], "你用哪只手握拍 —— 角色、发球站位和击球判定都会跟着换边"),
            ("difficulty", "难度", [DIFFICULTY[k]["label"] for k in DIFF_KEYS], notes_d[self.difficulty]),
            ("points", "局分", [f"{p} 分" for p in POINTS], "先到该分数且领先 2 分获胜"),
            ("tactics", "对手战术", ["风格", "自学 AI"], note_t),
            ("source", "谁来打", ["我自己", "AI 表演"], notes_s[self.source]),
            ("move", "跑位", ["自动", "手动"], notes_m[self.move] if src == "pad" else "只在自己打的时候有效"),
            ("quality", "画质", ["流畅", "精致", "极致"], "只有集成显卡时建议“流畅”；独立显卡三档都流畅"),
        ]


# ---------------------------------------------------------------------------- input
class PadInput:
    """Stick, three shot buttons and jump / dive, from the keyboard and from the on-screen controls.

    Keyboard: arrows / WASD = stick, J K L = soft / high / power, Space (or O) = jump, Ctrl (or I) = dive: with a
    direction held or pressed within DIVE_WAIT seconds to that side, otherwise at the shuttle. On Windows the keys are read as real key state (GetAsyncKeyState, only
    while the game window is in front), so directions and Ctrl can be held while a button is pressed; elsewhere
    the key events of the window are used (a direction lasts a quarter of a second).
    Screen: drag inside the stick, tap a button (one pointer: OpenCV windows have no multi-touch)."""
    VK = dict(left=(0x25, 0x41), right=(0x27, 0x44), up=(0x26, 0x57), down=(0x28, 0x53), soft=(0x4A,), high=(0x4B,),
              power=(0x4C,), jump=(0x20, 0x4F), dive=(0x11, 0x49))
    KEY_BUTTON = {"j": "soft", "k": "high", "l": "power", "o": "jump", "i": "dive"}
    KEY_DIR = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}
    SHOTS = ("soft", "high", "power")
    DIVE_WAIT = 0.12

    def __init__(self, ui, title=None):
        self.stick_geom, self.button_geom = ui.pad_geometry()
        self.user32 = None
        if title is not None and sys.platform == "win32":
            try:
                import ctypes
                self._ct, self.user32, self.title = ctypes, ctypes.windll.user32, title
            except Exception:                                      # no user32: key events only
                self.user32 = None
        self.queue = []
        self.down = {b: False for b in self.VK}
        self.touch = np.zeros(2)
        self.dragging = False
        self.pulse = (np.zeros(2), -1.0)
        self._jump = self._dive = False
        self._dive_at = None

    def _focused(self):
        buf = self._ct.create_unicode_buffer(128)
        self.user32.GetWindowTextW(self.user32.GetForegroundWindow(), buf, 128)
        return buf.value == self.title

    def _button(self, name):
        if name in self.SHOTS:
            self.queue.append(name)
        elif name == "jump":
            self._jump = True
        elif name == "dive":
            self._dive = True

    def key(self, name, clock):
        """A key event of the window (used where real key state is not available)."""
        if self.user32 is not None:
            return
        if name in self.KEY_BUTTON:
            self._button(self.KEY_BUTTON[name])
        elif name in self.KEY_DIR:
            self.pulse = (np.array(self.KEY_DIR[name]), clock + 0.25)

    def mouse(self, event, x, y):
        cx, cy, r = self.stick_geom
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, br) in self.button_geom.items():
                if (x - bx) ** 2 + (y - by) ** 2 <= (1.15 * br) ** 2:
                    self._button(name)
                    return True
            if (x - cx) ** 2 + (y - cy) ** 2 <= (1.5 * r) ** 2:
                self.dragging = True
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.touch = np.zeros(2)
        if self.dragging and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            v = np.array([x - cx, -(y - cy)], float) / (0.6 * r)
            n = np.linalg.norm(v)
            self.touch = v / max(n, 1.0)
            return True
        return False

    def poll(self, clock):
        """(stick, shot button that went down or None, dict(jump / dive: went down))."""
        stick = self.touch.copy()
        if self.user32 is not None:
            if self._focused():
                now = {nm: any(self.user32.GetAsyncKeyState(vk) & 0x8000 for vk in vks) for nm, vks in self.VK.items()}
                edge = {nm: now[nm] and not self.down[nm] for nm in now}
                self.down = now
                kx, ky = float(now["right"]) - float(now["left"]), float(now["up"]) - float(now["down"])
                held = bool(kx or ky)
                if held:
                    stick = np.array([kx, ky]) / np.hypot(kx, ky)
                for b in self.SHOTS:
                    if edge[b]:
                        self.queue.append(b)
                self._jump |= edge["jump"]
                # the dive key: a direction that is held, or that follows within a moment, says where to
                if edge["dive"]:
                    if held:
                        self._dive = True
                    else:
                        self._dive_at = clock + self.DIVE_WAIT
                elif now["dive"] and any(edge[d] for d in self.KEY_DIR):
                    self._dive, self._dive_at = True, None
                if self._dive_at is not None and (held or clock >= self._dive_at):
                    self._dive, self._dive_at = True, None
            else:
                self.down = {b: False for b in self.VK}
        elif clock < self.pulse[1]:
            stick = self.pulse[0].copy()
        extra = dict(jump=self._jump, dive=self._dive)
        self._jump = self._dive = False
        return stick, (self.queue.pop(0) if self.queue else None), extra


class WinKeys:
    """The window's message loop without cv2.waitKey (Windows). OpenCV's own loop opens a "save image" dialog
    on Ctrl+S and copies the frame on Ctrl+C -- and Ctrl + S is "dive backwards" here. The messages are pumped
    with user32 instead; those two key strokes are swallowed, everything else goes to the window as before.
    poll() returns the names of the keys that went down (auto-repeat only for the arrows / WASD)."""
    NAMES = {0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down", 0x0D: "enter", 0x1B: "esc", 0x20: "space", 0x09: "tab",
             0x41: "left", 0x44: "right", 0x57: "up", 0x53: "down"}
    NAMES.update({ord(c.upper()): c for c in "crhfqjklpiot"})
    REPEAT = {"left", "right", "up", "down"}

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.u = ctypes, ctypes.windll.user32
        self.msg = wintypes.MSG()
        self.u.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        self.u.PeekMessageW.restype = wintypes.BOOL
        self.u.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.u.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.u.DispatchMessageW.restype = ctypes.c_ssize_t
        self.u.GetKeyState.restype = ctypes.c_short

    def poll(self):
        keys, ref = [], self.ct.byref(self.msg)
        for _ in range(400):                                       # (never spin on a flood of messages)
            if not self.u.PeekMessageW(ref, None, 0, 0, 1):
                break
            m = self.msg
            if m.message in (0x0100, 0x0104):                      # WM_KEYDOWN, WM_SYSKEYDOWN
                vk = int(m.wParam)
                name = self.NAMES.get(vk)
                ctrl = self.u.GetKeyState(0x11) & 0x8000
                if name and (name in self.REPEAT or not (int(m.lParam) >> 30) & 1) and not (ctrl and name in ("c", "q", "r", "f", "h", "p")):
                    keys.append(name)
                if ctrl and vk in (0x53, 0x43):                    # Ctrl+S, Ctrl+C: not for OpenCV
                    continue
                if m.message == 0x0104 and vk in (0x12, 0x79):     # Alt, F10 alone: would open the window menu
                    continue
            self.u.TranslateMessage(ref)
            self.u.DispatchMessageW(ref)
        return keys


# ---------------------------------------------------------------------------- one match
class Session:
    INTRO = 2.8

    def __init__(self, app, s: Settings, seed, mode="quick", stage=0, drill=0):
        self.settings = s
        self.mode, self.stage, self.drill = mode, stage, None
        self.opp = ROSTER[s.opponent]
        W, H = app.W, app.H
        pr = app.progress
        src = SOURCES[s.source]
        handed = "left" if s.handed else "right"
        self.notice = None
        diff, points = DIFF_KEYS[s.difficulty], POINTS[s.points]
        self.mind = None                                       # brain.Mind of the opponent (learned tactics)
        if mode == "campaign":
            k, diff, points = modes.stage(stage)
            self.opp, src = ROSTER[k], "pad"
        elif mode == "ladder":
            r = modes.LADDER[stage]
            self.opp, src, diff, points = modes.rung_style(stage, pr.outfit, handed), "pad", r["diff"], r["points"]
            self.mind = brain.Mind(modes.rung_brain(stage) or brain.Brain(), beta=r["beta"], personal=pr.personal, learn=True)
        elif mode == "training":
            self.drill = modes.DRILLS[drill]
            self.opp, src, diff = modes.COACH, "pad", "easy"
        elif TACTICS[s.tactics] == "brain":
            self.mind = brain.Mind(brain.generation(1 << 20) or brain.Brain(), beta=4.0,
                                   personal=pr.personal if src == "pad" else None, learn=src == "pad")
        kw = dict(points_to_win=points, seed=seed, difficulty=diff, width=W, height=H, arena=QUALITIES[s.quality])
        self.pad = None
        self.trail = None
        if src == "pad":
            me = app.player_style()
            bot = PadBot(np.random.default_rng(seed + 7)) if app.pad_bot else None
            self.match = Match(app.segs, self.opp, player="pad", player_style=me, player_handed=handed,
                               pad_auto=MOVES[s.move] == "auto", pad_bot=bot, drill=self.drill, **kw)
            self.pad = None if bot is not None else PadInput(app.ui, None if app.headless else app.win)
            self.names = {"p": me.name, "o": self.opp.name}
            self.colors = {"p": wardrobe.accent(pr.outfit), "o": _accent(self.opp)}
            self.trail = wardrobe.trail(pr.outfit)
            self.sub = (f"{'左手' if s.handed else '右手'}持拍 · {'自动' if MOVES[s.move] == 'auto' else '手动'}跑位 · "
                        + (f"训练：{self.drill['name']}" if self.drill is not None else f"{DIFFICULTY[diff]['label']}难度"))
            if mode == "campaign":
                self.sub = f"第 {stage + 1} 关 · " + self.sub
            elif mode == "ladder":
                self.sub = f"天梯第 {stage + 1} 层 · " + self.sub
        else:
            me = get_style("lee_chong_wei" if self.opp.key != "lee_chong_wei" else "lin_dan")
            self.match = Match(app.segs, self.opp, player="ai", player_style=me, **kw)
            self.names = {"p": me.name, "o": self.opp.name}
            self.colors = {"p": _accent(me), "o": _accent(self.opp)}
            self.sub = me.title
        self.match.opp.mind = self.mind
        self.source = src
        self.sr = SceneRenderer(self.match.m, W, H)
        self.t = 0.0                 # match clock
        self.age = 0.0               # seconds since the session started (walk-on)
        self.paused = False
        self.logged = False
        self.coins = None            # [(what for, coins)] once the match is over
        self.verdict = None
        self.match.update(0.0)

    def step(self, dt, clock=0.0):
        self.age += dt
        if self.pad is not None:
            stick, press, extra = self.pad.poll(clock)
            if self.age >= self.INTRO and not self.paused:
                self.match.pad_input(stick, press, **extra)
        if self.age >= self.INTRO and not self.paused:
            self.t += dt
        if not self.paused:
            self.match.update(self.t)

    def close(self):
        self.sr.r.close()


def _accent(st):
    return U._rgb255(st.accent) if max(st.accent[:3]) > 0.35 else U._rgb255(st.shirt)


def _mix(a, b, f):
    return {k: (np.asarray(a[k], float) * (1 - f) + np.asarray(b[k], float) * f) for k in a}


def compress_video(path, crf=23):
    """Re-encode an OpenCV mp4v recording as H.264 (about a tenth of the size) when the optional
    imageio-ffmpeg package is there; otherwise the file is left as it is. Returns the final size [MB]."""
    import subprocess
    path = Path(path)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = path.with_name(path.stem + "_h264.mp4")
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "medium",
                            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                           capture_output=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
        elif tmp.exists():
            tmp.unlink()
    except Exception as e:                                         # no ffmpeg: keep the mp4v file
        print("video not re-encoded:", e)
    return path.stat().st_size / 1e6


# ---------------------------------------------------------------------------- the application
class App:
    def __init__(self, size=(1280, 720), settings: Settings | None = None, library=None, seed=0, headless=False,
                 record=None, fps=30.0, auto=None, start="title", duration=0.0, log=None, show_perf=False, render=True,
                 realtime=False, pad_bot=False, mode="quick", stage=None, drill=0, save="default", outfit=None):
        """headless / record: fixed time step 1/fps (deterministic). auto: target Settings reached by a
        key script through the menus (demo videos, tests). start: "title" or "match". render=False:
        matches are simulated without drawing (statistics; needs start="match").
        mode / stage / drill: what start="match" (or the key script) plays -- "quick", "campaign" stage or
        "ladder" rung (0-based; None: the first one not won yet) or "training" drill (index into modes.DRILLS).
        save: the progress file ("default": save/progress.json; None: nothing is read or written).
        outfit: what the player wears in this run instead of the saved outfit (demo videos, tests)."""
        self.W, self.H = size
        self.render = render
        self._blank = np.zeros((self.H, self.W, 3), np.uint8)
        self.settings = settings or Settings()
        self.pad_bot = pad_bot                   # controller mode played by scripted thumbs (demo videos, tests)
        self.seed = seed
        self.headless, self.record, self.fps = headless, record, fps
        self.fixed_dt = 1.0 / fps if (headless or record) and not realtime else None
        self.duration, self.log_path, self.show_perf = duration, log, show_perf
        self.segs = load_library(library or ROOT / "assets" / "stroke_library.npz")
        self.ui = U.UI(self.W, self.H)
        self.views = [game_view(self.W / self.H),
                      dict(lookat=[0, 0, 1.0], distance=13.0, azimuth=0, elevation=-22, fovy=45.0),
                      dict(lookat=[0, 0, 0.5], distance=17.0, azimuth=90, elevation=-70, fovy=45.0)]
        self.view_k = 0
        self.sess = None
        self.attract = self.attract_sr = None
        self.portraits = {}
        self.progress = modes.Progress(ROOT / "save" / "progress.json" if save == "default" else save)
        if outfit is not None:
            self.progress.outfit = wardrobe.clean(outfit)
        self.mode = mode
        first = self.progress.next_rung() if mode == "ladder" else self.progress.next_open()
        self.stage = first if stage is None else int(stage)
        self.preview = None                      # ui.Preview: the player's character on the title and in the wardrobe
        self.w_slot = self.w_item = self.w_focus = 0
        self._w_back, self._w_toast = "title", None
        self._w_rng = np.random.default_rng(seed + 5)
        self._ladder_key = None
        self.drill = int(drill)
        self.keys = None                         # WinKeys once the window exists
        self._result_t = 0.0
        self.state = start
        self.row = 0
        self.clock = 0.0
        self.toast = None                        # (text, until)
        self.inputs = []
        self.writer = None
        self.win = "Yu Ni Wu Gua - Badminton"          # ASCII: OpenCV window titles are not Unicode-safe on Windows
        self.perf = dict(fps=fps, render=0.0, sim=0.0, hud=0.0)
        self.frames = 0
        self.script, self._script_t = self._make_script(auto) if auto is not None else [], 0.0
        self.results = []
        if start != "match":
            self._new_attract()
            self._make_portraits()
        else:
            self._start_match()

    # ------------------------------------------------------------------ resources
    def player_style(self, outfit=None):
        """The player's character: the saved outfit (or the one being tried on), its name and catch phrases."""
        o = wardrobe.clean(outfit or self.progress.outfit)
        return replace(PLAYER_STYLE, name=wardrobe.name(o), outfit=o, lines=wardrobe.quips(o),
                       handed="left" if self.settings.handed else "right")

    def _portraits_of(self, styles):
        """Cut-out portraits {style key: image}; kept as png files next to the save (a render takes half a second)."""
        from PIL import Image
        out, todo = {}, []
        cache = None if self.progress.path is None else self.progress.path.parent / "portraits"
        if cache is not None:
            src = hashlib.md5(b"".join((ROOT / "badrt" / f).read_bytes() for f in ("character.py", "wardrobe.py", "ui.py"))).hexdigest()
        for st in styles:
            f = None
            if cache is not None:
                h = hashlib.md5(repr((src, st.shirt, st.shorts, st.skin, st.hair, st.accent, sorted(st.dims.items()), st.handed,
                                      sorted(st.outfit.items()))).encode()).hexdigest()[:12]
                f = cache / f"{st.key}_{h}.png"
                if f.exists():
                    try:
                        out[st.key] = Image.open(f).convert("RGBA")
                        continue
                    except OSError:
                        pass
            todo.append((st, f))
        if todo:
            new = U.render_portraits([st for st, _ in todo], qpos=self.attract.opp.ready_q)
            for st, f in todo:
                out[st.key] = new[st.key]
                if f is not None:
                    try:
                        f.parent.mkdir(parents=True, exist_ok=True)
                        new[st.key].save(f)
                    except OSError:
                        pass
        return out

    def _make_portraits(self):
        self.portraits = self._portraits_of(ROSTER)
        self._ladder_key = None

    def _ladder_styles(self):
        handed = "left" if self.settings.handed else "right"
        return [modes.rung_style(i, self.progress.outfit, handed) for i in range(len(modes.LADDER))]

    def _figure(self, outfit=None):
        if self.preview is None:
            self.preview = U.Preview(self.ui)
        return self.preview.sprite(self.player_style(outfit), self.attract.opp.ready_q, self.clock)

    def _new_attract(self):
        """AI exhibition behind the menus."""
        rng = np.random.default_rng(self.seed + 100 + len(self.results))
        a, b = rng.choice(len(ROSTER), 2, replace=False)
        if self.attract_sr is not None:
            self.attract_sr.r.close()
        self.attract = Match(self.segs, ROSTER[a], player="ai", player_style=ROSTER[b], points_to_win=21,
                             seed=int(rng.integers(1 << 30)), width=self.W, height=self.H, arena="fast")
        self.attract_sr = SceneRenderer(self.attract.m, self.W, self.H)
        self.attract_t = 0.0

    def _start_match(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = Session(self, self.settings, self.seed + len(self.results), self.mode, self.stage, self.drill)
        if self.sess.notice:
            self.toast = (self.sess.notice, self.clock + 4.0)
        self.state = "match"

    def _menu_of_mode(self):
        return {"quick": "select", "campaign": "stages", "ladder": "ladder", "training": "drills"}[self.mode]

    def _leave_match(self, to=None):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        if self.attract is None:
            self._new_attract()
        if not self.portraits:
            self._make_portraits()
        self.state = to or self._menu_of_mode()
        return getattr(self, "_" + self.state)(0.0, [])

    # ------------------------------------------------------------------ scripted menus
    def _make_script(self, target: Settings):
        """Key presses (delay [s], key) that walk through the menus to `target`."""
        mode, drill = self.mode, self.drill
        self.mode, self.drill = MODES[0], 0                         # the cursors start at the top and are moved there
        sc = [(2.6, "enter")] + [(0.8, "right")] * MODES.index(mode) + [(1.4, "enter")]
        if mode in ("campaign", "ladder"):                          # they open on the first stage not won yet
            return sc + [(2.0, "enter"), ("wait", "result"), (7.0, "q")]
        if mode == "training":
            return sc + [(0.5, "down")] * drill + [(1.6, "enter"), ("wait", "result"), (7.0, "q")]
        cur = self.settings.opponent
        while cur != target.opponent:                               # browse the roster card by card
            sc.append((0.7, "right"))
            cur = (cur + 1) % len(ROSTER)
        sc.append((1.6, "enter"))
        rows = [r[0] for r in self.settings.rows()]
        for i, attr in enumerate(rows):
            d = getattr(target, attr) - getattr(self.settings, attr)
            sc += [(0.45, "right" if d > 0 else "left")] * abs(d)
            if i < len(rows) - 1:
                sc.append((0.35, "down"))
        sc += [(1.2, "enter"), ("wait", "result"), (7.0, "q")]
        return sc

    def _run_script(self):
        while self.script:
            item = self.script[0]
            if item[0] == "wait":
                if self.state != item[1]:
                    return
            elif self.clock - self._script_t < item[0]:
                return
            else:
                self.inputs.append(item[1])
            self.script.pop(0)
            self._script_t = self.clock

    # ------------------------------------------------------------------ input
    def _on_mouse(self, event, x, y, flags, param):
        s = self.sess
        if self.state == "match" and s is not None and s.pad is not None and not s.paused:
            if s.pad.mouse(event, x, y):
                return
        if event == cv2.EVENT_LBUTTONDOWN:
            name = self.ui.hit_test(x, y)
            if name:
                self.inputs.append("@" + name)

    def _poll_window(self):
        if self.keys is None and sys.platform == "win32":
            try:
                self.keys = WinKeys()
            except Exception as e:                                  # no user32: OpenCV's own loop
                print("own key loop not available:", e)
                self.keys = False
        if self.keys:
            self.inputs += self.keys.poll()
        else:
            k = cv2.waitKeyEx(1)
            if k != -1:
                name = KEYS.get(k)
                if name:
                    self.inputs.append(name)
        try:
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                self.inputs.append("quit")
        except cv2.error:
            self.inputs.append("quit")

    # ------------------------------------------------------------------ states
    def _menu_background(self, dt):
        a = self.attract
        self.attract_t += min(dt, 0.05)
        a.update(self.attract_t)
        if a.state == "over" and self.attract_t - a.t_state > 3.0:
            self._new_attract()
            a = self.attract
            a.update(0.0)
        t = self.clock
        self.attract_sr.set_view(dict(lookat=[0.0, 0.0, 1.1], distance=14.5, azimuth=90 + 38 * np.sin(0.11 * t),
                                      elevation=-13 + 3 * np.sin(0.07 * t), fovy=42.0))
        return self.attract_sr.render(a.d, decor=U.scene_fx(a, self.attract_t, {"p": U.BLUE, "o": U.RED}, show_target=False))

    def _title(self, dt, inputs):
        for k in inputs:
            if k in ("enter", "space", "@start"):
                self.state = "mode"
            elif k in ("tab", "@wardrobe"):
                self._open_wardrobe()
            elif k == "esc":
                return None
        img = self._menu_background(dt)
        self.ui.title(img, self.clock, self.progress.coins, self._figure(), wardrobe.accent(self.progress.outfit))
        return img

    # ------------------------------------------------------------------ wardrobe
    def _open_wardrobe(self):
        self._w_back, self.state = self.state, "wardrobe"
        self.w_focus, self._w_toast = 0, None
        self._w_cursor()

    def _w_cursor(self):
        """Put the item cursor on what is worn in the slot."""
        slot, _, table = wardrobe.SLOTS[self.w_slot]
        keys = list(table)
        self.w_item = keys.index(self.progress.outfit[slot]) if self.progress.outfit[slot] in keys else 0

    def _w_take(self, slot, key):
        pr = self.progress
        name, cost = wardrobe.item(slot, key)[:2]
        if pr.has(slot, key):
            pr.wear(slot, key)
            msg = f"穿上了：{name}"
        elif pr.buy(slot, key):
            pr.wear(slot, key)
            msg = f"买下了：{name}（-{cost}）"
        else:
            msg = f"金币不够，还差 {cost - pr.coins} —— 去打几场吧"
        self._w_toast = (msg, self.clock + 2.5)

    def _wardrobe(self, dt, inputs):
        pr, slots = self.progress, wardrobe.SLOTS
        for k in inputs:
            slot, _, table = slots[self.w_slot]
            keys = list(table)
            step = {"up": -1, "down": 1}.get(k)
            if k in ("esc", "@back"):
                if self.w_focus and k == "esc":
                    self.w_focus = 0
                    self._w_cursor()
                else:
                    self.state = self._w_back
            elif step is not None and not self.w_focus:
                self.w_slot = (self.w_slot + step) % len(slots)
                self._w_cursor()
            elif step is not None:
                self.w_item = (self.w_item + step) % len(keys)
            elif k == "left" and self.w_focus:
                self.w_focus = 0
                self._w_cursor()
            elif k in ("right", "enter", "@next") and not self.w_focus:
                self.w_focus = 1
            elif k in ("enter", "@next"):
                self._w_take(slot, keys[self.w_item])
            elif k.startswith("@slot"):
                self.w_slot, self.w_focus = int(k[5:]), 1
                self._w_cursor()
            elif k.startswith("@item"):
                i = int(k[5:])
                if i == self.w_item and self.w_focus:
                    self._w_take(slot, keys[i])
                self.w_item, self.w_focus = i, 1
            elif k in ("r", "@random"):                             # a random outfit from what is owned
                for sl, _, tb in slots:
                    if sl != "name":
                        own = [kk for kk in tb if pr.has(sl, kk)]
                        pr.outfit[sl] = own[int(self._w_rng.integers(len(own)))]
                pr.save()
                self._w_cursor()
                self._w_toast = ("随机整活：" + wardrobe.score(pr.outfit)[1], self.clock + 2.5)
        slot, _, table = slots[self.w_slot]
        keys = list(table)
        self.w_item = min(self.w_item, len(keys) - 1)
        trying = dict(pr.outfit)
        if self.w_focus:
            trying[slot] = keys[self.w_item]
        items = [(table[kk][0], table[kk][2], table[kk][1], pr.has(slot, kk), pr.outfit[slot] == kk, wardrobe.swatch(slot, kk))
                 for kk in keys]
        img = self._menu_background(dt)
        toast = self._w_toast[0] if self._w_toast and self.clock < self._w_toast[1] else None
        self.ui.wardrobe(img, [(t, wardrobe.item(sl, pr.outfit[sl])[0]) for sl, t, _ in slots], self.w_slot, items, self.w_item,
                         self.w_focus, self._figure(trying), pr.coins, wardrobe.name(trying), wardrobe.score(trying),
                         wardrobe.accent(trying), toast, self.clock)
        return img

    def _mode(self, dt, inputs):
        k0 = MODES.index(self.mode)
        for k in inputs:
            if k in ("left", "right"):
                k0 = (k0 + (1 if k == "right" else -1)) % len(MODES)
            elif k.startswith("@mode"):
                if int(k[5:]) == k0:
                    self.state = self._menu_of_mode()
                k0 = int(k[5:])
            elif k in ("enter", "@next"):
                self.mode = MODES[k0]
                self.state = self._menu_of_mode()
            elif k in ("tab", "@wardrobe"):
                self.mode = MODES[k0]
                self._open_wardrobe()
            elif k in ("esc", "@back"):
                self.state = "title"
            self.mode = MODES[k0]
        pr = self.progress
        if self.state == "stages":
            self.stage = pr.next_open()
        elif self.state == "ladder":
            self.stage = pr.next_rung()
        done = sum(1 for v in pr.stars if v > 0)
        gens = len(brain.generations())
        cards = [("快速比赛", "QUICK MATCH", ["自己挑对手、难度和局分", "对手用自己的风格，", "或者用自学 AI 的战术"],
                  f"{len(ROSTER)} 位对手 · {len(DIFF_KEYS)} 档难度"),
                 ("闯关模式", "CAMPAIGN", ["十二位名将一关比一关强", "赢了才能挑战下一位", "每关按表现评一到三星"],
                  f"已通过 {done} / {len(modes.CAMPAIGN)} 关 · {sum(pr.stars)} 星"),
                 ("整活天梯", "LADDER", ["大爷、大妈、外卖小哥、熊猫……", "战术是 AI 自己跟自己打出来的", "还会边打边记你的习惯"],
                  f"已击败 {sum(1 for v in pr.ladder if v > 0)} / {len(modes.LADDER)} 位"
                  + (f" · AI 第 {gens - 1} 代" if gens > 1 else "")),
                 ("训练模式", "TRAINING", ["教练喂球，一项一项练：", "高远、杀球、吊球、搓放", "接杀、四角跑动、多拍"],
                  f"{len(modes.DRILLS)} 个项目 · 练过 {len(pr.drills)} 个")]
        img = self._menu_background(dt)
        self.ui.mode(img, k0, cards, self.clock, pr.coins)
        return img

    def _ladder(self, dt, inputs):
        pr, n = self.progress, len(modes.LADDER)
        for k in inputs:
            if k in ("left", "right"):
                self.stage = (self.stage + (1 if k == "right" else -1)) % n
            elif k in ("up", "down"):
                self.stage = int(np.clip(self.stage + (4 if k == "down" else -4), 0, n - 1))
            elif k.startswith("@stage"):
                i = int(k[6:])
                if i == self.stage and pr.rung_unlocked(i):
                    self.state = "loading"
                self.stage = i
            elif k in ("enter", "@next"):
                if pr.rung_unlocked(self.stage):
                    self.state = "loading"
                else:
                    self.toast = ("先赢下前一位", self.clock + 2.0)
            elif k in ("tab", "@config"):
                self.state, self.row = "settings", 0
            elif k in ("esc", "@back"):
                self.state = "mode"
        self.stage = int(np.clip(self.stage, 0, n - 1))
        styles = self._ladder_styles()
        key = (tuple(sorted(pr.outfit.items())), self.settings.handed)
        if key != self._ladder_key:                                 # (the last opponent wears what the player wears)
            self.portraits.update(self._portraits_of(styles))
            self._ladder_key = key
        items = [(styles[i], r["diff"], r["points"], pr.ladder[i], not pr.rung_unlocked(i)) for i, r in enumerate(modes.LADDER)]
        b = modes.rung_brain(self.stage)
        seen = int(round(pr.personal.N.sum()))
        info = [("战术：自对弈第 %d 代（共训练了 %d 代）" % (b.gen, len(brain.generations()) - 1)) if b is not None and b.gen > 0
                else "战术：自己的风格（还没经过自对弈）",
                f"和你交手时出过 {seen} 拍，都记着结果" if seen else "还没和你交过手 —— 打过之后会记住哪些球路对你管用"]
        img = self._menu_background(dt)
        self.ui.ladder(img, self.stage, items, self.portraits, info, pr.coins)
        if self.toast and self.clock < self.toast[1]:
            self.ui.hint(img, self.toast[0], self.clock, y=640)
        return img

    def _stages(self, dt, inputs):
        n = len(modes.CAMPAIGN)
        for k in inputs:
            if k in ("left", "right"):
                self.stage = (self.stage + (1 if k == "right" else -1)) % n
            elif k in ("up", "down"):
                self.stage = (self.stage + (4 if k == "down" else -4)) % n
            elif k.startswith("@stage"):
                i = int(k[6:])
                if i == self.stage and self.progress.unlocked(i):
                    self.state = "loading"
                self.stage = i
            elif k in ("enter", "@next"):
                if self.progress.unlocked(self.stage):
                    self.state = "loading"
                else:
                    self.toast = ("先赢下前一关", self.clock + 2.0)
            elif k in ("tab", "@config"):
                self.state, self.row = "settings", 0
            elif k in ("esc", "@back"):
                self.state = "mode"
        items = []
        for i in range(n):
            j, diff, pts = modes.stage(i)
            items.append((ROSTER[j], diff, pts, self.progress.stars[i], not self.progress.unlocked(i)))
        img = self._menu_background(dt)
        self.ui.stages(img, self.stage, items, self.portraits, sum(self.progress.stars))
        if self.toast and self.clock < self.toast[1]:
            self.ui.hint(img, self.toast[0], self.clock, y=640)
        return img

    def _drills(self, dt, inputs):
        n = len(modes.DRILLS)
        for k in inputs:
            if k in ("up", "down"):
                self.drill = (self.drill + (1 if k == "down" else -1)) % n
            elif k.startswith("@drill"):
                i = int(k[6:])
                if i == self.drill:
                    self.state = "loading"
                self.drill = i
            elif k in ("enter", "@next"):
                self.state = "loading"
            elif k in ("tab", "@config"):
                self.state, self.row = "settings", 0
            elif k in ("esc", "@back"):
                self.state = "mode"
        img = self._menu_background(dt)
        self.ui.drills(img, self.drill, [(d, self.progress.drills.get(d["key"])) for d in modes.DRILLS])
        return img

    def _select(self, dt, inputs):
        s, n = self.settings, len(ROSTER)
        for k in inputs:
            if k == "left":
                s.opponent = (s.opponent - 1) % n
            elif k == "right":
                s.opponent = (s.opponent + 1) % n
            elif k in ("up", "down"):
                s.opponent = (s.opponent + (3 if k == "down" else -3)) % n
            elif k in ("@pageprev", "@pagenext"):
                s.opponent = (s.opponent + (6 if k == "@pagenext" else -6)) % n
            elif k.startswith("@opp"):
                i = int(k[4:])
                if i == s.opponent:
                    self.state = "settings"
                s.opponent = i
            elif k in ("enter", "@next"):
                self.state = "settings"
                self.row = 0
            elif k in ("esc", "@back"):
                self.state = "mode"
        img = self._menu_background(dt)
        self.ui.select(img, s.opponent, self.portraits, self.clock)
        return img

    def _settings(self, dt, inputs):
        s = self.settings
        quick = self.mode == "quick"
        rows = s.rows(self.mode)
        self.row = min(self.row, len(rows) - 1)
        for k in inputs:
            attr, _, opts, _ = rows[self.row]
            if k == "up":
                self.row = (self.row - 1) % len(rows)
            elif k == "down":
                self.row = (self.row + 1) % len(rows)
            elif k in ("left", "right"):
                setattr(s, attr, int(np.clip(getattr(s, attr) + (1 if k == "right" else -1), 0, len(opts) - 1)))
            elif k.startswith("@set"):
                i, j = (int(v) for v in k[4:].split("_"))
                self.row = i
                setattr(s, rows[i][0], j)
            elif k in ("enter", "@start"):
                self.state = "loading" if quick else self._menu_of_mode()
            elif k in ("esc", "@back"):
                self.state = self._menu_of_mode()
            rows = s.rows(self.mode)
        img = self._menu_background(dt)
        self.ui.settings(img, [(t, o, getattr(s, a), note) for a, t, o, note in rows], self.row,
                         ROSTER[s.opponent] if quick else None, self.clock)
        return img

    def _loading(self, dt, inputs):
        img = self._menu_background(0.0)
        if self.mode == "training":
            who = "教练"
        elif self.mode == "ladder":
            who = modes.LADDER[self.stage]["style"].name
        else:
            who = ROSTER[modes.stage(self.stage)[0] if self.mode == "campaign" else self.settings.opponent].name
        self.ui.loading(img, f"{who} 正在进场…")
        self._pending_start = True
        return img

    def _match(self, dt, inputs):
        s = self.sess
        m = s.match
        for k in inputs:
            if s.pad is not None:
                s.pad.key(k, self.clock)
            if k == "p" or (k == "space" and m.pad is None):      # (with a controller the space bar is the jump)
                s.paused = not s.paused
            elif k == "esc":
                if s.paused:
                    return self._leave_match()
                s.paused = True
            elif k == "r":
                self._start_match()
                s = self.sess
                m = s.match
            elif k == "t":
                m.taunt(s.t)
            elif k == "c":
                self.view_k = (self.view_k + 1) % len(self.views)
            elif k == "h":
                self.show_perf = not self.show_perf
        tic = time.perf_counter()
        s.step(dt if self.fixed_dt else min(dt, 0.05), self.clock)
        t1 = time.perf_counter()
        if m.state == "over" and s.t - m.t_state > 2.5:
            self._log_result()
            self.state, self._result_t = "result", self.clock
        if not self.render:                                        # statistics only
            return self._blank
        view = self.views[self.view_k]
        if s.age < s.INTRO:                                        # walk-on: the camera swings round to its place
            f = float(np.clip((s.age - 0.3) / (s.INTRO - 0.7), 0, 1))
            f = f * f * (3 - 2 * f)
            v0 = dict(lookat=[0.0, 1.0, 1.2], distance=11.0, azimuth=view["azimuth"] - 75.0, elevation=-9.0, fovy=view["fovy"])
            view = {k: (v.tolist() if np.ndim(v) else float(v)) for k, v in _mix(v0, view, f).items()}
        s.sr.set_view(view)
        img = s.sr.render(m.d, decor=U.scene_fx(m, s.t, s.colors, show_target=m.pad is not None, trail_rgb=s.trail))
        t2 = time.perf_counter()
        if s.age < s.INTRO:
            self.ui.vignette(img)
            self.ui.countdown(img, s.names, s.colors, s.opp, s.age, s.sub)
        else:
            hint = None
            if self.toast and self.clock < self.toast[1]:
                hint = self.toast[0]
            elif m.pad is not None and m.state == "serve" and m.server == "p" and m.serve_plan is None and m.pad.in_position():
                hint = "J 发网前球 · K 发高远球"
            elif m.pad is not None and m.pad.message.startswith("too tired") and m.state == "rally":
                hint = "体力不足 —— 跳不动也扑不动了"
            elif m.pad is not None and s.t < 16.0 and m.state != "point":       # (short: it sits between the stick and the buttons)
                hint = ("圈变绿时按 J 轻 · K 高 · L 力" if s.t < 5.5 else "方向键瞄准：左右选边，上下深浅" if s.t < 11.0
                        else "空格起跳 · Ctrl + 方向 鱼跃")
            elif m.pad is not None and m.state == "point" and m.winner == "p" and m.taunts == 0 and sum(m.score) <= 6:
                hint = "得分了！按 T 嘚瑟一下"
            elif m.pad is not None and not m.pad_view["ok"] and 0 <= s.t - m.pad_view["t"] < 0.9:
                hint = "扑空了" if m.pad_view["press"] == "dive" else "挥空了 —— 等圈变绿再按"
            pad = None
            delaying = False
            if m.pad is not None:
                v = m.pad_view
                held = ["jump"] if m.pad._jump_until is not None or m.pad.body.airborne(s.t) else []
                pad = dict(stick=v["stick"], lit=v["press"], lit_age=s.t - v["t"], held=held,
                           labels={b: SHOT_CN.get(sh, sh) for b, sh in m.pad.menu().items()} if m.state == "rally" else None)
                delaying = m.umpire is not None and m.state in ("point", "serve") and m.serve_plan is None and (
                    not m.pad.in_position() or (m.server == "p" and m.opp.in_position()))
            perf = None
            if self.show_perf:
                p = self.perf
                perf = [f"{p['fps']:.0f} fps", f"渲染 {p['render']:.0f} ms", f"物理+AI {p['sim']:.1f} ms", f"界面 {p['hud']:.1f} ms"]
            self.ui.hud(img, m, s.t, s.names, s.colors, hint=hint, perf=perf, pad=pad, view=view, delaying=delaying)
            if s.paused:
                self.ui.paused(img, pad=m.pad is not None)
        t3 = time.perf_counter()
        for key, v in (("sim", t1 - tic), ("render", t2 - t1), ("hud", t3 - t2)):
            self.perf[key] += 0.1 * (v * 1e3 - self.perf[key])
        return img

    def _result(self, dt, inputs):
        s = self.sess
        camp = None
        if s.mode in ("campaign", "ladder"):
            n = len(modes.CAMPAIGN) if s.mode == "campaign" else len(modes.LADDER)
            camp = dict(stage=s.stage, stars=modes.stars(s.match), next=s.stage + 1 < n, unit="关" if s.mode == "campaign" else "层")
        for k in inputs:
            if k in ("enter", "r", "@again"):
                if camp is not None and k != "r" and s.match.won() and camp["next"]:
                    self.stage = s.stage + 1
                self._start_match()
                return self._match(dt, [])
            if k in ("esc", "@menu"):
                return self._leave_match()
        if not self.render:
            return self._blank
        s.sr.set_view(self.views[0])
        img = s.sr.render(s.match.d)
        if s.mode == "training":
            self.ui.drill_result(img, s.match, self.progress.drills.get(s.drill["key"]), s.coins)
        else:
            self.ui.result(img, s.match, s.names, s.colors, self.clock - self._result_t, campaign=camp, verdict=s.verdict,
                           coins=s.coins, purse=self.progress.coins)
        return img

    def _log_result(self):
        s = self.sess
        if s.logged:
            return
        s.logged = True
        m = s.match
        rec = dict(score=m.score, opponent=s.opp.key, source=s.source, handed=m.player_handed, settings=asdict(s.settings),
                   difficulty=m.difficulty, game_seconds=round(s.t, 1), stats=m.stats(), events=m.events, mode=s.mode)
        if s.mind is not None:
            rec["brain"] = dict(generation=s.mind.brain.gen, beta=s.mind.beta, choices=len(s.mind.samples), rallies=s.mind.rallies,
                                personal=int(round(self.progress.personal.N.sum())))
        if m.state == "over":
            first = False
            if s.mode == "campaign":
                rec.update(stage=s.stage + 1, stars=modes.stars(m))
                self.progress.stage_done(s.stage, rec["stars"])
            elif s.mode == "ladder":
                rec.update(rung=s.stage + 1, stars=modes.stars(m))
                first = self.progress.rung_done(s.stage, rec["stars"])
            elif s.mode == "training":
                self.progress.drill_done(s.drill["key"], m.score[0])
            if s.source == "pad":                                   # coins, and what kind of match it was
                s.coins = modes.reward(s.mode, m, s.stage, first)
                s.verdict = modes.verdict(m) if s.mode != "training" else None
                self.progress.earn(sum(n for _, n in s.coins))
                rec.update(coins=s.coins, verdict=s.verdict, outfit=dict(self.progress.outfit))
        self.results.append(rec)
        out = Path(self.log_path) if self.log_path else (Path(self.record).with_suffix(".json") if self.record
                                                         else ROOT / "results" / "last_match.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")

    # ------------------------------------------------------------------ main loop
    def frame(self, dt):
        """Advance by dt and draw one frame; None when the app wants to quit."""
        self.clock += dt
        self._run_script()
        inputs, self.inputs = self.inputs, []
        if "quit" in inputs or "q" in inputs:
            return None
        if getattr(self, "_pending_start", False):
            self._pending_start = False
            self._start_match()
        img = getattr(self, "_" + self.state)(dt, inputs)
        self.frames += 1
        return img

    def run(self):
        if not self.headless:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.win, self.W, self.H)
            cv2.setMouseCallback(self.win, self._on_mouse)
        t_prev = time.perf_counter()
        full = False
        try:
            while True:
                now = time.perf_counter()
                wall = now - t_prev
                t_prev = now
                self.perf["fps"] += 0.1 * (1.0 / max(wall, 1e-3) - self.perf["fps"])
                img = self.frame(self.fixed_dt or wall)
                if img is None:
                    break
                if self.record:
                    if self.writer is None:
                        Path(self.record).parent.mkdir(parents=True, exist_ok=True)
                        self.writer = cv2.VideoWriter(str(self.record), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.W, self.H))
                    self.writer.write(img)
                if not self.headless:
                    cv2.imshow(self.win, img)
                    self._poll_window()
                    if "f" in self.inputs:
                        full = not full
                        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
                if self.duration and self.clock >= self.duration:
                    break
                if self.headless and not self.script and self.state == "result":
                    break
        finally:
            if self.sess is not None:
                if self.sess.match.events and not self.sess.logged:
                    self._log_result()
                self.sess.close()
            if self.preview is not None:
                self.preview.close()
            if self.writer is not None:
                self.writer.release()
                compress_video(self.record)
            if not self.headless:
                cv2.destroyAllWindows()
        return self.results

"""Presentation layer: HUD, banners, menus, result screen and the in-scene effects.

Text and panels are drawn once with PIL (Chinese needs a real font) into cached RGBA sprites and
alpha-blended onto the rendered frame with numpy, so a HUD frame costs about a millisecond.
All sizes are given for a 720-line frame and scaled.
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .arcade import CARD_CN
from .character import KP
from .court import LENGTH_HALF, WIDTH_DOUBLES_HALF, WIDTH_SINGLES_HALF
from .fonts import font
from .styles import DIFFICULTY, ROSTER, SHOT_CN, Style

# palette (RGB)
INK = (12, 16, 26)
PANEL = (16, 22, 36)
WHITE = (245, 247, 250)
GREY = (150, 160, 176)
GOLD = (255, 204, 40)
CYAN = (70, 200, 255)
BLUE = (64, 140, 255)
RED = (240, 70, 80)
GREEN = (70, 220, 140)
ORANGE = (255, 150, 60)
VIOLET = (176, 124, 255)
CARD_RGB = {"yellow": (250, 210, 30), "red": (232, 44, 52), "black": (18, 18, 22)}

WHY_CN = {"in": "落地得分", "out": "出界", "net": "下网", "short": "未过网", "serve_out": "发球出界", "serve_fault": "发球失误",
          "card": "拖延比赛，红牌罚分"}


def project(view, p, W, H):
    """Pixel position of a world point in a frame rendered with this view (None: behind the camera)."""
    az, el = np.radians(view["azimuth"]), np.radians(view["elevation"])
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    d = np.asarray(p, float) - (np.asarray(view["lookat"], float) - view["distance"] * f)
    z = float(d @ f)
    if z <= 0.1:
        return None
    k = (H / 2) / np.tan(np.radians(view.get("fovy", 45.0)) / 2)
    return W / 2 + k * float(d @ np.cross(f, up)) / z, H / 2 - k * float(d @ up) / z

def _rgb255(c):
    return tuple(int(round(float(v) * 255)) for v in c[:3])


def _is_cjk(s):
    return any(ord(ch) > 0x2E7F for ch in s)


class UI:
    def __init__(self, width, height):
        self.W, self.H = width, height
        self.s = height / 720.0
        self._cache = {}
        self.rects = {}                        # clickable regions of the screen being shown: name -> (x0, y0, x1, y1)

    # ------------------------------------------------------------------ sprites
    def px(self, v):
        return int(round(v * self.s))

    def _store(self, key, im: Image.Image):
        a = np.asarray(im.convert("RGBA"), np.uint8)
        spr = (np.ascontiguousarray(a[:, :, 2::-1]).astype(np.uint16), a[:, :, 3:4].astype(np.uint16))   # BGR, alpha
        self._cache[key] = spr
        return spr

    def text(self, txt, size, color=WHITE, bold=True, shadow=False, cjk=None):
        key = ("t", txt, size, color, bold, shadow, cjk)
        spr = self._cache.get(key)
        if spr is None:
            f = font(self.px(size), bold=bold, cjk=_is_cjk(txt) if cjk is None else cjk)
            bb = f.getbbox(txt)
            pad = self.px(3) + (self.px(2) if shadow else 0)
            w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
            im = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            if shadow:
                d.text((pad - bb[0] + self.px(1.5), pad - bb[1] + self.px(1.5)), txt, font=f, fill=(0, 0, 0, 170))
                im = im.filter(ImageFilter.GaussianBlur(self.px(1.2)))
                d = ImageDraw.Draw(im)
            d.text((pad - bb[0], pad - bb[1]), txt, font=f, fill=tuple(color) + (255,))
            spr = self._store(key, im)
        return spr

    def panel(self, w, h, fill=PANEL, alpha=200, radius=12, border=None, border_w=2):
        key = ("p", w, h, fill, alpha, radius, border, border_w)
        spr = self._cache.get(key)
        if spr is None:
            W, H = self.px(w), self.px(h)
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=self.px(radius), fill=tuple(fill) + (alpha,),
                                outline=None if border is None else tuple(border) + (255,), width=self.px(border_w))
            spr = self._store(key, im)
        return spr

    def custom(self, key, builder):
        spr = self._cache.get(key)
        if spr is None:
            spr = self._store(key, builder())
        return spr

    def blit(self, img, spr, x, y, opacity=1.0, anchor="lt"):
        """x, y in 720-line units (floats); anchor: l/c/r + t/m/b."""
        bgr, a = spr
        h, w = a.shape[:2]
        x, y = self.px(x), self.px(y)
        x -= {"l": 0, "c": w // 2, "r": w}[anchor[0]]
        y -= {"t": 0, "m": h // 2, "b": h}[anchor[1]]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.W), min(y + h, self.H)
        if x1 <= x0 or y1 <= y0 or opacity <= 0.003:
            return (x, y, x + w, y + h)
        sb, sa = bgr[y0 - y:y1 - y, x0 - x:x1 - x], a[y0 - y:y1 - y, x0 - x:x1 - x]
        if opacity < 0.997:
            sa = (sa * int(opacity * 256)) >> 8
        dst = img[y0:y1, x0:x1]
        dst[:] = ((dst.astype(np.uint16) * (255 - sa) + sb * sa + 127) // 255).astype(np.uint8)
        return (x, y, x + w, y + h)

    def label(self, img, txt, x, y, size, color=WHITE, anchor="lt", bold=True, shadow=False, opacity=1.0):
        return self.blit(img, self.text(txt, size, color, bold, shadow), x, y, opacity, anchor)

    def dim(self, img, k=0.45, blur=True):
        """Darken (and soften) the frame behind a menu."""
        if blur:
            small = cv2.resize(img, (self.W // 4, self.H // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 2.2)
            img[:] = cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img[:] = (img.astype(np.uint16) * int(k * 256) >> 8).astype(np.uint8)

    def vignette(self, img):
        key = ("vig",)
        v = self._cache.get(key)
        if v is None:
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            r = np.sqrt(((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2)
            v = (np.clip(1.0 - 0.34 * np.clip(r - 0.55, 0, 1) ** 1.5, 0, 1) * 255).astype(np.uint8)
            v = np.ascontiguousarray(np.repeat(v[..., None], 3, axis=2))
            self._cache[key] = v
        cv2.multiply(img, v, dst=img, scale=1.0 / 255.0)

    # ------------------------------------------------------------------ HUD
    def scoreboard(self, img, match, names, colors, now):
        """Top left: two rows (name, serve mark, points), like a broadcast bug."""
        w, row = 340, 40
        x0, y0 = 18, 16
        self.blit(img, self.panel(w, 2 * row + 8, PANEL, 215, 10), x0, y0)
        gp = self.game_point(match)
        for i, who in enumerate(("o", "p")):
            y = y0 + 4 + i * row
            sc = match.score[1] if who == "o" else match.score[0]
            self.blit(img, self.panel(6, row - 8, colors[who], 255, 3), x0 + 10, y + 4)
            self.label(img, names[who], x0 + 26, y + row / 2, 21, WHITE, "lm")
            if match.server == who and match.state in ("serve", "rally"):
                self.blit(img, self.shuttle_icon(), x0 + w - 86, y + row / 2, anchor="cm")
            lead = (sc > (match.score[0] if who == "o" else match.score[1]))
            self.blit(img, self.panel(54, row - 6, GOLD if lead else (36, 46, 68), 255, 6), x0 + w - 62, y + 3)
            self.label(img, str(sc), x0 + w - 35, y + row / 2 + 1, 27, INK if lead else WHITE, "cm")
        if gp is not None and match.state != "over":
            txt = "赛点" if gp else "局点"
            self.blit(img, self.panel(74, 26, RED, 235, 13), x0 + w + 10, y0 + row + 4, anchor="lm")
            self.label(img, txt, x0 + w + 47, y0 + row + 4, 16, WHITE, "cm")

    @staticmethod
    def game_point(match):
        """None, or True when somebody is one point from winning."""
        a, b = match.score
        n = match.points_to_win
        cap = n + 9
        for me, other in ((a, b), (b, a)):
            if (me >= n - 1 and me - other >= 1) or me == cap - 1:
                return True
        return None

    def shuttle_icon(self):
        def build():
            n = self.px(22)
            im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.polygon([(n * 0.5, n * 0.92), (n * 0.08, n * 0.2), (n * 0.92, n * 0.2)], fill=(255, 255, 255, 255))
            for k in range(1, 4):
                x = n * (0.08 + 0.84 * k / 4)
                d.line([(n * 0.5, n * 0.86), (x, n * 0.2)], fill=(170, 180, 196, 255), width=max(1, self.px(1)))
            d.ellipse([n * 0.34, n * 0.7, n * 0.66, n * 1.0], fill=GOLD + (255,))
            return im
        return self.custom(("shuttle",), build)

    def shot_ticker(self, img, match, names, colors, now):
        """Bottom centre: the last stroke (who, what, how fast), fading out."""
        e = match.last_event
        if not e or "who" not in e:
            return
        age = now - e["t"]
        if age > 2.2 or age < 0:
            return
        op = float(np.clip(age / 0.08, 0, 1) * np.clip((2.2 - age) / 0.5, 0, 1))
        kmh = e["speed"] * 3.6
        shot = SHOT_CN.get(e["shot"], e["shot"])
        if e["shot"] == "smash" and e.get("jump", 0.0) > 0.08:
            shot = "跳杀"
        big = e["shot"] in ("smash", "half_smash", "kill") and kmh > 180
        w = 330
        x0, y0 = 640 - w / 2, 646 - (8 * (1 - min(age / 0.12, 1)))
        self.blit(img, self.panel(w, 50, PANEL, 215, 25, border=colors[e["who"]] if big else None), x0, y0, op)
        self.blit(img, self.panel(10, 10, colors[e["who"]], 255, 5), x0 + 20, y0 + 25, op, "cm")
        self.label(img, names[e["who"]], x0 + 34, y0 + 25, 17, GREY, "lm", opacity=op)
        self.label(img, shot, x0 + w / 2 + 6, y0 + 25, 23, GOLD if big else WHITE, "cm", opacity=op)
        self.label(img, f"{kmh:.0f}", x0 + w - 62, y0 + 25, 27, GOLD if big else WHITE, "rm", opacity=op)
        self.label(img, "km/h", x0 + w - 16, y0 + 29, 13, GREY, "rm", opacity=op)

    def rally_counter(self, img, match):
        if match.state == "rally" and match.rally_shots >= 4:
            self.blit(img, self.panel(108, 34, PANEL, 190, 17), 1262, 668, anchor="rt")
            self.label(img, f"{match.rally_shots}", 1196, 685, 21, GOLD, "rm")
            self.label(img, "拍", 1202, 685, 16, WHITE, "lm")
            self.label(img, "回合", 1226, 685, 13, GREY, "lm")

    def point_banner(self, img, match, names, colors, now):
        e = next((x for x in reversed(match.events[-4:]) if "point" in x), None)
        if match.state != "point" or e is None:
            return
        age = now - e["t"]
        if age < 0.35:
            return
        a = age - 0.35
        op = float(np.clip(a / 0.18, 0, 1))
        slide = (1 - min(a / 0.22, 1)) ** 2 * 60
        w = e["point"]
        why = WHY_CN.get(e["why"], e["why"])
        y = 132                                             # above the far player: what the two do next is worth seeing
        if "drill" in e:                                     # a repetition of a training drill
            dr = e["drill"]
            if dr["void"]:
                return
            col = GREEN if dr["ok"] else RED
            self.blit(img, self.panel(460, 78, PANEL, 225, 14, border=col), 640 - slide, y, op, "cm")
            self.label(img, "到位！" if dr["ok"] else "再来", 640 - slide, y - 14, 29, col, "cm", opacity=op)
            self.label(img, dr["note"], 640 - slide, y + 20, 16, GREY, "cm", opacity=op)
            return
        if e["why"] == "card":
            title, detail = f"{names[w]} 得分", why
        elif e["why"] in ("out", "net", "short", "serve_out", "serve_fault"):
            loser = "o" if w == "p" else "p"
            title, detail = f"{names[w]} 得分", f"{names[loser]} {why} · {e['shots']} 拍"
        else:
            title, detail = f"{names[w]} 得分", f"{why} · {e['shots']} 拍"
        self.blit(img, self.panel(460, 78, PANEL, 225, 14, border=colors[w]), 640 - slide, y, op, "cm")
        self.label(img, title, 640 - slide, y - 14, 29, WHITE, "cm", opacity=op)
        self.label(img, detail, 640 - slide, y + 20, 16, GREY, "cm", opacity=op)

    def inset(self, img, frame, kp2d=None, label="视频输入", ok=True, mirror=False):
        """Top right (clear of the court): what the camera sees, with the tracked skeleton."""
        if frame is None:
            return
        w = 236
        h = w * frame.shape[0] / frame.shape[1]
        W, H = self.px(w), self.px(h)
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if kp2d is not None and np.isfinite(kp2d).all():
            from .character import BONE_DRAW, KP
            pts = (np.asarray(kp2d)[:, :2] * [W, H]).astype(int)       # the character's key points (MP_IDS order)
            for a, b in BONE_DRAW:
                cv2.line(small, tuple(pts[KP[a]]), tuple(pts[KP[b]]), (255, 200, 70), max(1, self.px(2)), cv2.LINE_AA)
            for n in ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "l_shoulder", "r_shoulder", "l_ankle", "r_ankle"):
                cv2.circle(small, tuple(pts[KP[n]]), max(2, self.px(2.5)), (40, 204, 255), -1, cv2.LINE_AA)
        if mirror:
            small = cv2.flip(small, 1)
        x, y = 1280 - 20 - w, 16 + 28
        self.blit(img, self.panel(w + 8, h + 8 + 24, PANEL, 235, 10, border=(60, 74, 100)), x - 4, y - 4 - 24)
        xs, ys = self.px(x), self.px(y)
        img[ys:ys + H, xs:xs + W] = small[:max(0, min(H, self.H - ys)), :max(0, min(W, self.W - xs))]
        self.blit(img, self.panel(8, 8, GREEN if ok else RED, 255, 4), x + 6, y - 14, anchor="lm")
        self.label(img, label, x + 20, y - 14, 13, WHITE, "lm")

    def hint(self, img, txt, now, y=596):
        op = 0.65 + 0.35 * np.sin(now * 4.0)
        w = 30 + 21 * len(txt)
        self.blit(img, self.panel(w, 40, PANEL, 200, 20, border=GOLD), 640, y, anchor="cm")
        self.label(img, txt, 640, y, 19, GOLD, "cm", opacity=float(op))

    def perf(self, img, lines):
        for i, s in enumerate(lines):
            self.label(img, s, 16, 706 - 17 * (len(lines) - 1 - i), 12, GREY, "lb", bold=False)

    # ------------------------------------------------------------------ on-screen controller
    PAD_STICK = (150.0, 556.0, 78.0)                                  # centre x, y, radius (720-line units)
    PAD_BUTTONS = {"soft": (1004.0, 596.0, 46.0, CYAN, "轻", "J"), "high": (1097.0, 520.0, 46.0, GREEN, "高", "K"),
                   "power": (1190.0, 596.0, 46.0, RED, "力", "L"), "jump": (1097.0, 662.0, 36.0, GOLD, "跳", "空格"),
                   "dive": (916.0, 654.0, 36.0, VIOLET, "扑", "Ctrl")}

    def _disc(self, r, fill, alpha, border=None, bw=3):
        def build():
            n = 2 * self.px(r) + 2
            im = Image.new("RGBA", (n * 2, n * 2), (0, 0, 0, 0))    # drawn at 2x and reduced: smooth edge
            d = ImageDraw.Draw(im)
            d.ellipse([2, 2, 2 * n - 3, 2 * n - 3], fill=tuple(fill) + (alpha,),
                      outline=None if border is None else tuple(border) + (255,), width=2 * self.px(bw))
            return im.resize((n, n), Image.LANCZOS)
        return self.custom(("disc", r, fill, alpha, border, bw), build)

    def pad_geometry(self):
        """Pixel geometry of the on-screen controller, for hit testing: stick (cx, cy, r), buttons {name: (cx, cy, r)}."""
        sx, sy, sr = self.PAD_STICK
        return (self.px(sx), self.px(sy), self.px(sr)), {k: (self.px(v[0]), self.px(v[1]), self.px(v[2]))
                                                         for k, v in self.PAD_BUTTONS.items()}

    def pad_controls(self, img, stick, lit=None, lit_age=9.0, labels=None, held=()):
        """Virtual stick (bottom left), the three shot buttons (bottom right) and jump / dive.
        stick: (x, y) in [-1, 1], lit: the button pressed last, lit_age seconds ago; labels: what each button
        would play right now; held: buttons that are on (gathered for a jump)."""
        sx, sy, sr = self.PAD_STICK
        self.blit(img, self._disc(sr, PANEL, 150, border=(70, 84, 112)), sx, sy, anchor="cm")
        k = np.asarray(stick, float)
        n = float(np.linalg.norm(k))
        if n > 1.0:
            k = k / n
        self.blit(img, self._disc(30, WHITE if n > 0.15 else (120, 132, 150), 215), sx + 0.6 * sr * k[0], sy - 0.6 * sr * k[1],
                  anchor="cm")
        for name, (bx, by, br, col, cn, key) in self.PAD_BUTTONS.items():
            on = (lit == name and lit_age < 0.18) or name in held
            big = br > 40
            self.blit(img, self._disc(br, col if on else PANEL, 235 if on else 170, border=col), bx, by, anchor="cm")
            self.label(img, cn, bx, by - (5 if big else 4), 27 if big else 20, INK if on else WHITE, "cm")
            self.label(img, key, bx, by + (24 if big else 17), 12 if big else 10, INK if on else GREY, "cm", bold=False)
            if labels and labels.get(name):
                self.label(img, labels[name], bx, by - br - 12, 13, col, "cm", shadow=True)

    # ------------------------------------------------------------------ arcade layer
    def stamina(self, img, match, colors):
        """Under the score: for both players the burst bar (this rally) and, thinner, the match reserve."""
        x0, y0, w, row = 18, 110, 340, 27
        self.blit(img, self.panel(w, 2 * row + 10, PANEL, 205, 10), x0, y0)
        bx, bw = x0 + 60, 186
        for i, (who, ai) in enumerate((("o", match.opp), ("p", match.pad))):
            st = ai.stam
            y = y0 + 5 + i * row
            self.blit(img, self.panel(6, row - 8, colors[who], 255, 3), x0 + 10, y + 4)
            self.label(img, "体力", x0 + 24, y + row / 2, 13, GREY, "lm", bold=False)
            col = RED if st.winded else (GREEN if st.burst > 0.5 else (GOLD if st.burst > 0.25 else ORANGE))
            for (yy, hh, val, c) in ((y + 6, 9, st.burst, col), (y + 18, 4, st.tank, CYAN)):
                p0, p1 = (self.px(bx), self.px(yy)), (self.px(bx + bw), self.px(yy + hh))
                cv2.rectangle(img, p0, p1, (78, 54, 44), -1)
                if val > 0.004:
                    cv2.rectangle(img, p0, (self.px(bx + bw * val), p1[1]), c[::-1], -1)
            cx = self.px(bx + bw * st.cap)                              # how far this bar can still refill
            cv2.line(img, (cx, self.px(y + 4)), (cx, self.px(y + 17)), (200, 200, 200), 1)
            if st.winded:
                self.label(img, "喘不过气", bx + bw + 10, y + row / 2, 13, RED, "lm")
            else:
                self.label(img, f"耐力 {100 * st.tank:.0f}%", bx + bw + 10, y + row / 2, 12, GREY, "lm", bold=False)
        cards = match.umpire.cards
        for k, c in enumerate(cards[-6:]):                             # the cards shown to the player so far
            x = self.px(x0 + w + 10 + 16 * k)
            cv2.rectangle(img, (x, self.px(y0 + row + 9)), (x + self.px(11), self.px(y0 + row + 25)), CARD_RGB[c][::-1], -1)
            cv2.rectangle(img, (x, self.px(y0 + row + 9)), (x + self.px(11), self.px(y0 + row + 25)), (230, 230, 230), 1)

    def umpire(self, img, match, now, delaying):
        """The umpire's voice (top centre), the shot clock while the player keeps the game waiting, the card."""
        u = match.umpire
        if u.lines:
            t, txt, kind = u.lines[-1]
            age = now - t
            if 0 <= age < 3.4:
                op = float(np.clip(age / 0.12, 0, 1) * np.clip((3.4 - age) / 0.5, 0, 1))
                col = {"info": WHITE, "warn": GOLD, "card": RED}[kind]
                w = 96 + 19 * len(txt)
                self.blit(img, self.panel(w, 38, PANEL, 225, 19, border=None if kind == "info" else col), 640, 34, op, "cm")
                self.blit(img, self.panel(48, 24, (44, 56, 84), 255, 12), 640 - w / 2 + 34, 34, op, "cm")
                self.label(img, "裁判", 640 - w / 2 + 34, 34, 13, GOLD, "cm", opacity=op)
                self.label(img, txt, 640 + 26, 34, 18, col, "cm", opacity=op)
        wait = u.waited(now)
        if delaying and u.running and wait > 4.0 and match.state != "over":
            f = min(wait / u.limit, 1.0)
            col = RED if wait > u.WARN * u.limit / u.LIMIT else GOLD
            self.blit(img, self.panel(300, 30, PANEL, 215, 15), 640, 76, anchor="cm")
            p0 = (self.px(640 - 134), self.px(84))
            cv2.rectangle(img, p0, (self.px(640 + 134), self.px(88)), (78, 54, 44), -1)
            cv2.rectangle(img, p0, (self.px(640 - 134 + 268 * f), self.px(88)), col[::-1], -1)
            self.label(img, f"准备时间 {wait:.0f} / {u.limit:.0f} 秒 · 超时吃牌", 640, 71, 12, col, "cm")
        if u.card_fx is not None:
            t, c = u.card_fx
            age = now - t
            if 0 <= age < 3.2:
                op = float(np.clip(age / 0.1, 0, 1) * np.clip((3.2 - age) / 0.4, 0, 1))
                pop = 1.0 + 0.25 * max(1.0 - age / 0.25, 0.0)
                pop = round(pop * 20) / 20
                self.blit(img, self.panel(84 * pop, 118 * pop, CARD_RGB[c], 255, 9, border=WHITE, border_w=3), 900, 300, op, "cm")
                self.label(img, CARD_CN[c], 900, 386, 24, WHITE, "cm", shadow=True, opacity=op)

    def bubbles(self, img, match, now, view):
        """What the players say, above their heads."""
        for who, (txt, t0, t1) in match.bubbles.items():
            if not t0 <= now < t1:
                continue
            ci = match.oi if who == "o" else match.pi
            pt = project(view, match.d.site_xpos[ci.kp_sites[KP["nose"]]] + [0.0, 0.0, 0.42], self.W, self.H)
            if pt is None:
                continue
            op = float(np.clip((now - t0) / 0.12, 0, 1) * np.clip((t1 - now) / 0.3, 0, 1))
            w = 30 + 17.5 * len(txt)
            x = float(np.clip(pt[0] / self.s, w / 2 + 8, 1280 - w / 2 - 8))
            y = float(np.clip(pt[1] / self.s, 110, 700))
            self.blit(img, self.panel(w, 34, WHITE, 240, 15), x, y, op, "cm")
            self.blit(img, self.panel(12, 12, WHITE, 240, 6), float(np.clip(pt[0] / self.s, x - w / 2 + 10, x + w / 2 - 10)),
                      y + 19, op, "cm")
            self.label(img, txt, x, y, 16, INK, "cm", opacity=op)

    def drill_panel(self, img, match, now):
        """Training: what the exercise asks for and how it is going (where the score would be)."""
        d = match.drill
        self.blit(img, self.panel(430, 104, PANEL, 215, 10), 18, 16)
        self.label(img, "训练 · " + d["name"], 34, 38, 21, WHITE, "lm")
        self.label(img, d["goal"], 34, 64, 13, GOLD, "lm", bold=False)
        self.label(img, d["tip"], 34, 84, 12, GREY, "lm", bold=False)
        ok, bad = match.score
        n = min(len(match.drill_log) + (match.state != "point"), d["reps"])
        self.label(img, f"第 {n} / {d['reps']} 球", 34, 104, 13, WHITE, "lm")
        self.label(img, f"成功 {ok}", 330, 104, 14, GREEN, "rm")
        self.label(img, f"失败 {bad}", 430, 104, 14, RED, "rm")

    def hud(self, img, match, now, names, colors, inset=None, hint=None, perf=None, pad=None, view=None, delaying=False):
        self.vignette(img)
        if match.drill is not None:
            self.drill_panel(img, match, now)
        else:
            self.scoreboard(img, match, names, colors, now)
        if match.umpire is not None:
            self.stamina(img, match, colors)
            self.umpire(img, match, now, delaying)
        if view is not None and match.bubbles:
            self.bubbles(img, match, now, view)
        self.shot_ticker(img, match, names, colors, now)
        self.rally_counter(img, match)
        self.point_banner(img, match, names, colors, now)
        if pad is not None:
            self.pad_controls(img, **pad)
            self.label(img, "P 暂停 / 按键说明", 640, 709, 11, GREY, "cm", bold=False)
        if inset is not None:
            self.inset(img, **inset)
        if hint:
            self.hint(img, hint, now)
        if perf:
            self.perf(img, perf)

    # ------------------------------------------------------------------ screens
    SLOGANS = ["今日宜：杀球 · 忌：鱼跃", "用键盘打羽毛球，膝盖永远不会疼", "穿最怪的衣服，打最狠的球", "对手是自己跟自己打出来的 AI",
               "输了不要紧，换身衣服再输一次", "裁判：再拖延时间我可掏牌了", "本游戏没有氪金，只有金币和脸皮", "咸鱼也能当球拍，梦想还是要有的"]

    def coins(self, img, n, x, y):
        """The purse, right-aligned at x."""
        txt = f"{n}"
        w = 56 + 12 * len(txt)
        self.blit(img, self.panel(w, 34, PANEL, 225, 17, border=(120, 96, 24), border_w=1), x - w, y, anchor="lm")
        self.blit(img, self._disc(10, GOLD, 255, border=(176, 122, 10), bw=2), x - w + 21, y, anchor="cm")
        self.label(img, txt, x - w + 38, y, 18, GOLD, "lm")

    def glow(self, w, h, color, alpha=110):
        def build():
            W, H = self.px(w), self.px(h)
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(im).ellipse([W * 0.2, H * 0.2, W * 0.8, H * 0.8], fill=tuple(color) + (alpha,))
            return im.filter(ImageFilter.GaussianBlur(min(W, H) * 0.11))
        return self.custom(("glow", w, h, color, alpha), build)

    def logo(self):
        """The name of the game: big gold letters with a dark edge, slightly tilted."""
        def build():
            f = font(self.px(150), True, True)
            txt = "羽你无瓜"
            bb = f.getbbox(txt)
            pad = self.px(26)
            W, H = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            o = (pad - bb[0], pad - bb[1])
            d.text((o[0] + self.px(7), o[1] + self.px(9)), txt, font=f, fill=(0, 0, 0, 150))
            im = im.filter(ImageFilter.GaussianBlur(self.px(5)))
            d = ImageDraw.Draw(im)
            d.text(o, txt, font=f, fill=(255, 255, 255, 255), stroke_width=self.px(9), stroke_fill=(20, 26, 44, 255))
            face = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(face).text(o, txt, font=f, fill=(255, 255, 255, 255))
            a = np.asarray(face)[:, :, 3:4].astype(np.float32) / 255.0
            t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
            top, bot = np.array([255, 238, 120], np.float32), np.array([255, 150, 30], np.float32)
            g = top * (1 - t) + bot * t
            arr = np.asarray(im).astype(np.float32)
            arr[:, :, :3] = arr[:, :, :3] * (1 - a) + g * a
            return Image.fromarray(arr.astype(np.uint8), "RGBA").rotate(3.5, resample=Image.BICUBIC, expand=True)
        return self.custom(("logo",), build)

    def title(self, img, now, coins=None, figure=None, accent=CYAN):
        """figure: sprite of the player's own character (Preview)."""
        self.dim(img, 0.45)
        self.rects = {}
        if figure is not None:
            self.blit(img, self.glow(560, 640, accent, 120), 1010, 380, anchor="cm")
            self.blit(img, figure, 1010, 380, anchor="cm")
        cx = 470 if figure is not None else 640
        self.blit(img, self.logo(), cx, 232 + 5 * np.sin(now * 1.6), anchor="cm")
        self.blit(img, self.panel(460, 40, GOLD, 255, 20), cx, 352, anchor="cm")
        self.label(img, "抽象羽毛球 · 键盘上场 · 整活到底", cx, 352, 21, INK, "cm")
        k = int(now / 3.5) % len(self.SLOGANS)
        f = (now / 3.5) % 1.0
        op = float(np.clip(f / 0.1, 0, 1) * np.clip((1 - f) / 0.1, 0, 1))
        self.label(img, "“" + self.SLOGANS[k] + "”", cx, 410, 21, (224, 230, 242), "cm", bold=False, opacity=op, shadow=True)
        pulse = 0.55 + 0.45 * np.sin(now * 3.2)
        r = self.blit(img, self.panel(300, 58, GOLD, 255, 29), cx - 120, 520, anchor="cm")
        self.label(img, "开始整活  Enter", cx - 120, 520, 24, INK, "cm", opacity=float(0.75 + 0.25 * pulse))
        self.rects["start"] = r
        self.rects["wardrobe"] = self.blit(img, self.panel(200, 58, (40, 50, 74), 245, 29, border=(90, 104, 136), border_w=1),
                                           cx + 150, 520, anchor="cm")
        self.label(img, "更衣室  Tab", cx + 150, 520, 20, WHITE, "cm")
        if coins is not None:
            self.coins(img, coins, 1220, 44)
        self.label(img, "KEYBOARD WARRIOR CUP", 60, 44, 15, GREY, "lm")
        self.label(img, "↑↓←→ 选择 · Enter 确认 · Esc 返回 · 鼠标可点", cx, 690, 14, GREY, "cm", bold=False)

    def star(self, size, on=True):
        def build():
            n = self.px(size) * 2
            im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
            ang = np.radians(-90 + 36 * np.arange(10))
            r = np.where(np.arange(10) % 2 == 0, 0.48, 0.2) * n
            pts = [(n / 2 + r[k] * np.cos(ang[k]), n / 2 + r[k] * np.sin(ang[k])) for k in range(10)]
            ImageDraw.Draw(im).polygon(pts, fill=(GOLD if on else (62, 72, 96)) + (255,))
            return im.resize((n // 2, n // 2), Image.LANCZOS)
        return self.custom(("star", size, on), build)

    def stars(self, img, n, x, y, size=22, gap=6, opacity=1.0):
        for k in range(3):
            self.blit(img, self.star(size, k < n), x + (k - 1) * (size + gap), y, opacity, "cm")

    MODE_RGB = [CYAN, GOLD, VIOLET, GREEN, ORANGE]

    def mode(self, img, sel, cards, now, coins=None):
        """cards: [(title, english, [lines], footer)] -- quick match, campaign, ladder, training."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "选择模式", 60, 44, 34, WHITE, "lm")
        self.label(img, "GAME MODE", 62, 78, 14, GREY, "lm", bold=False)
        if coins is not None:
            self.coins(img, coins, 1220, 50)
        self.rects["wardrobe"] = self.blit(img, self.panel(168, 34, (40, 50, 74), 245, 17, border=(90, 104, 136), border_w=1),
                                           1100 - 12 * len(str(coins or 0)), 50, anchor="rm")
        self.label(img, "更衣室  Tab", 1016 - 12 * len(str(coins or 0)), 50, 16, WHITE, "cm")
        n = len(cards)
        cw = min(340.0, (1160.0 - 20.0 * (n - 1)) / n)
        for i, (title, en, lines, foot) in enumerate(cards):
            on = i == sel
            col = self.MODE_RGB[i % len(self.MODE_RGB)]
            x, y = 60 + cw / 2 + (cw + 20) * i, 372 - (8 if on else 0)
            self.rects[f"mode{i}"] = self.blit(img, self.panel(cw, 380, (30, 42, 70) if on else PANEL, 240 if on else 215, 16,
                                                             border=col if on else (56, 68, 94), border_w=3 if on else 1),
                                               x, y, anchor="cm")
            self.blit(img, self.panel(cw - 44, 6, col, 255 if on else 150, 3), x, y - 168, anchor="cm")
            self.label(img, title, x, y - 122, 34, col if on else WHITE, "cm")
            self.label(img, en, x, y - 88, 13, GREY, "cm", bold=False)
            self.blit(img, self.panel(cw - 70, 2, (70, 84, 112), 255, 1), x, y - 66, anchor="cm")
            for k, ln in enumerate(lines):
                self.label(img, ln, x, y - 34 + 29 * k, 15, (214, 222, 236), "cm", bold=False)
            self.label(img, foot, x, y + 148, 14, col if on else GREY, "cm")
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, "进入  Enter", 1125, 672, 19, INK, "cm")
        self.rects["next"] = r
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    # ------------------------------------------------------------------ wardrobe
    def wardrobe(self, img, slots, si, items, ii, focus, figure, coins, name, score, accent, toast, now):
        """slots: [(title, name of what is worn)], si: the slot shown; items of that slot: [(name, line, price, owned,
        worn, colour or None)], ii: the item under the cursor; focus: 0 slots, 1 items; figure: Preview sprite;
        score: (value, rank)."""
        self.dim(img, 0.3)
        self.rects = {}
        self.label(img, "更衣室", 60, 44, 34, WHITE, "lm")
        self.label(img, "WARDROBE · 穿什么都行，反正裁判不管", 62, 78, 14, GREY, "lm", bold=False)
        self.coins(img, coins, 1220, 50)
        # the slots
        for i, (title, worn) in enumerate(slots):
            on = i == si
            y = 100 + i * 38
            self.rects[f"slot{i}"] = self.blit(img, self.panel(214, 34, (30, 42, 70) if on else PANEL, 240 if on else 200, 10,
                                                             border=(GOLD if focus == 0 else (110, 124, 156)) if on else None), 60, y)
            self.label(img, title, 74, y + 17, 16, GOLD if on else WHITE, "lm")
            self.label(img, worn[:6], 264, y + 17, 12, (214, 222, 236) if on else GREY, "rm", bold=False)
        # the items of the slot
        rows, pitch = 10, 52
        top = int(np.clip(ii - rows // 2, 0, max(len(items) - rows, 0)))
        x0, w = 290, 470
        for k in range(top, min(top + rows, len(items))):
            nm, line, cost, owned, worn, col = items[k]
            on = k == ii
            y = 100 + (k - top) * pitch
            self.rects[f"item{k}"] = self.blit(img, self.panel(w, pitch - 6, (30, 42, 70) if on else PANEL, 240 if on else 195, 10,
                                                             border=(GOLD if focus == 1 else (110, 124, 156)) if on else None), x0, y)
            tx = x0 + 18
            if col is not None:
                self.blit(img, self._disc(13, _rgb255(col), 255, border=(230, 232, 238), bw=2), x0 + 28, y + 23, anchor="cm")
                tx = x0 + 52
            self.label(img, nm, tx, y + (15 if line else 23), 18, WHITE, "lm")
            if line:
                self.label(img, line[:26], tx, y + 35, 12, GREY, "lm", bold=False)
            if worn:
                self.blit(img, self.panel(62, 24, GREEN, 255, 12), x0 + w - 14, y + 23, anchor="rm")
                self.label(img, "穿着", x0 + w - 45, y + 23, 13, INK, "cm")
            elif owned:
                self.label(img, "已拥有" if cost else "免费", x0 + w - 18, y + 23, 13, GREY, "rm", bold=False)
            else:
                self.label(img, f"{cost}", x0 + w - 18, y + 23, 17, GOLD if cost <= coins else RED, "rm")
                self.blit(img, self._disc(8, GOLD, 255, border=(176, 122, 10), bw=2), x0 + w - 30 - 11 * len(str(cost)), y + 23, anchor="cm")
        if len(items) > rows:
            self.label(img, f"{ii + 1} / {len(items)}", x0 + w, 632, 12, GREY, "rm", bold=False)
        # the mirror
        px0, pw = 784, 436
        self.blit(img, self.panel(pw, 526, PANEL, 215, 16, border=(56, 68, 94), border_w=1), px0, 100)
        self.blit(img, self.glow(pw, 500, accent, 120), px0 + pw / 2, 340, anchor="cm")
        if figure is not None:
            self.blit(img, figure, px0 + pw / 2, 336, anchor="cm")
        self.label(img, name, px0 + 22, 126, 22, WHITE, "lm", shadow=True)
        self.blit(img, self.panel(40 + 16 * len(score[1]), 28, GOLD, 255, 14), px0 + pw - 16, 126, anchor="rm")
        self.label(img, score[1], px0 + pw - 16 - (40 + 16 * len(score[1])) / 2, 126, 15, INK, "cm")
        self.label(img, f"抽象值 {score[0]}", px0 + pw - 20, 156, 13, GOLD, "rm")
        if toast:
            self.blit(img, self.panel(34 + 16 * len(toast), 34, PANEL, 235, 17, border=GOLD), px0 + pw / 2, 596, anchor="cm")
            self.label(img, toast, px0 + pw / 2, 596, 15, GOLD, "cm")
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, ("穿上 / 购买" if focus else "挑一件") + "  Enter", 1125, 672, 18, INK, "cm")
        self.rects["next"] = r
        self.rects["random"] = self.blit(img, self.panel(170, 46, (40, 50, 74), 255, 23, border=(90, 104, 136), border_w=1),
                                         1016, 672, anchor="rm")
        self.label(img, "随机整活  R", 931, 672, 17, WHITE, "cm")
        self.label(img, "↑↓ 选择 · → 进入 · ← 返回部位 · 光标停在哪件就先试穿哪件", 300, 672, 14, (214, 222, 236), "lm", bold=False)
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    # ------------------------------------------------------------------ ladder
    def ladder(self, img, sel, items, portraits, info, coins):
        """The ladder of odd characters. items: [(style, difficulty, points, stars, locked)]; info: lines about
        the highlighted one (what plays its tactics)."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "整活天梯", 60, 44, 34, WHITE, "lm")
        self.label(img, f"已击败 {sum(1 for it in items if it[3] > 0)} / {len(items)} 位 · 对手的战术是自对弈练出来的，越往上练得越久",
                   62, 80, 15, VIOLET, "lm")
        self.coins(img, coins, 1220, 50)
        for i, (st, diff, pts, n, locked) in enumerate(items):
            col, row = i % 4, i // 4
            x, y = 60 + col * 295, 108 + row * 179
            self.rects[f"stage{i}"] = self.blit(img, self.stage_tile(i, st, portraits.get(st.key), diff, pts, n, locked, i == sel, "层"),
                                                x, y - (5 if i == sel else 0))
        st = items[sel][0]
        k = len(items) % 4
        x0 = 60 + k * 295
        w = 1220 - x0
        y0 = 108 + (len(items) // 4) * 179
        self.blit(img, self.panel(w, 165, PANEL, 225, 14, border=(56, 68, 94), border_w=1), x0, y0)
        self.label(img, st.name, x0 + 20, y0 + 26, 22, WHITE, "lm")
        self.label(img, st.title, x0 + 34 + 22 * len(st.name), y0 + 28, 14, GOLD, "lm")
        maxc = int((w - 40) / 14.3)
        y = y0 + 56
        for kk in range(0, min(len(st.blurb), 3 * maxc), maxc):
            self.label(img, st.blurb[kk:kk + maxc], x0 + 20, y, 13, (214, 222, 236), "lm", bold=False)
            y += 21
        for ln in info:
            self.label(img, ln, x0 + 20, y + 4, 13, VIOLET, "lm")
            y += 20
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, "挑战  Enter", 1125, 672, 19, INK, "cm")
        self.rects["next"] = r
        self.rects["config"] = self.label(img, "Tab 设置（惯用手 / 跑位 / 画质）", 640, 672, 15, (214, 222, 236), "cm", bold=False)
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    def stage_tile(self, i, st: Style, portrait, diff, points, stars, locked, selected, unit="关"):
        def build():
            W, H = self.px(275), self.px(165)
            acc = _rgb255(st.accent) if max(st.accent[:3]) > 0.35 else _rgb255(st.shirt)
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=self.px(14), fill=(30, 42, 70, 250) if selected else (22, 30, 48, 240),
                                outline=GOLD + (255,) if selected else (56, 68, 94, 255), width=self.px(3 if selected else 1))
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse([W * 0.02, H * 0.1, W * 0.5, H * 0.95], fill=acc + (90,))
            im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(self.px(22))))
            if portrait is not None:
                p = portrait.resize((self.px(120), self.px(136)), Image.LANCZOS)
                im.alpha_composite(p, (self.px(4), self.px(24)))
            d = ImageDraw.Draw(im)
            f0, f1, f2 = font(self.px(13), True, True), font(self.px(24), True, True), font(self.px(14), False, True)
            d.text((self.px(14), self.px(16)), f"第 {i + 1} {unit}", font=f0, fill=GOLD + (255,), anchor="lm")
            if len(st.name) > 5:
                f1 = font(self.px(19), True, True)
            d.text((self.px(132), self.px(52)), st.name, font=f1, fill=WHITE + (255,), anchor="lm")
            d.text((self.px(132), self.px(82)), f"{DIFFICULTY[diff]['label']} · {points} 分", font=f2, fill=(214, 222, 236, 255), anchor="lm")
            d.text((self.px(132), self.px(104)), st.title, font=font(self.px(12), False, True), fill=GREY + (255,), anchor="lm")
            if locked:
                im = Image.blend(Image.new("RGBA", im.size, (10, 14, 22, im.getpixel((W // 2, H // 2))[3])), im, 0.38)
                d = ImageDraw.Draw(im)
                d.rounded_rectangle([self.px(132), self.px(122), self.px(212), self.px(148)], radius=self.px(13), fill=(50, 60, 84, 255))
                d.text((self.px(172), self.px(135)), "未解锁", font=f0, fill=GREY + (255,), anchor="mm")
            else:
                star = [Image.fromarray(np.dstack([self.star(20, k < stars)[0][:, :, ::-1].astype(np.uint8),
                                                   self.star(20, k < stars)[1].astype(np.uint8)]), "RGBA") for k in range(3)]
                for k in range(3):
                    im.alpha_composite(star[k], (self.px(132 + 26 * k), self.px(124)))
            return im
        return self.custom(("stage", i, st.key, diff, points, stars, locked, selected, unit, id(portrait)), build)

    def stages(self, img, sel, items, portraits, total):
        """Campaign ladder. items: [(style, difficulty, points, stars, locked)]."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "闯关模式", 60, 44, 34, WHITE, "lm")
        self.label(img, f"已通过 {sum(1 for it in items if it[3] > 0)} / {len(items)} 关", 62, 80, 16, GOLD, "lm")
        self.blit(img, self.star(22), 1100, 50, anchor="cm")
        self.label(img, f"{total} / {3 * len(items)}", 1120, 50, 22, WHITE, "lm")
        self.label(img, "一星：获胜 · 二星：没吃牌 · 三星：失分不到对手一半", 1220, 82, 13, GREY, "rm", bold=False)
        for i, (st, diff, pts, n, locked) in enumerate(items):
            col, row = i % 4, i // 4
            x, y = 60 + col * 295, 108 + row * 179
            self.rects[f"stage{i}"] = self.blit(img, self.stage_tile(i, st, portraits.get(st.key), diff, pts, n, locked, i == sel),
                                                x, y - (5 if i == sel else 0))
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, "挑战  Enter", 1125, 672, 19, INK, "cm")
        self.rects["next"] = r
        self.rects["config"] = self.label(img, "Tab 设置（惯用手 / 跑位 / 画质）", 640, 672, 15, (214, 222, 236), "cm", bold=False)
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    def drills(self, img, sel, items):
        """Training menu. items: [(drill dict, best score or None)]."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "训练模式", 60, 44, 34, WHITE, "lm")
        self.label(img, "教练喂球，每个球都按要求判定 · 训练里不计体力", 62, 80, 16, GOLD, "lm")
        pitch = 76
        for i, (d, best) in enumerate(items):
            on = i == sel
            y = 104 + i * pitch
            self.rects[f"drill{i}"] = self.blit(img, self.panel(1160, pitch - 10, (26, 36, 60) if on else PANEL, 235 if on else 205, 12,
                                                              border=GOLD if on else None), 60, y)
            self.label(img, f"{i + 1}", 92, y + 33, 26, GOLD if on else GREY, "cm")
            self.label(img, d["name"], 126, y + 22, 21, WHITE, "lm")
            self.label(img, d["goal"], 126, y + 48, 13, (214, 222, 236), "lm", bold=False)
            self.label(img, d["tip"], 640, y + 48, 13, GREY, "lm", bold=False)
            self.label(img, f"{d['reps']} 球", 1090, y + 22, 14, GREY, "rm", bold=False)
            if best is not None:
                self.label(img, f"最好 {best} / {d['reps']}", 1196, y + 22, 16, GREEN if best >= 0.8 * d["reps"] else WHITE, "rm")
            else:
                self.label(img, "还没练过", 1196, y + 22, 14, GREY, "rm", bold=False)
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, "开始  Enter", 1125, 672, 19, INK, "cm")
        self.rects["next"] = r
        self.rects["config"] = self.label(img, "Tab 设置（惯用手 / 跑位 / 画质）", 640, 672, 15, (214, 222, 236), "cm", bold=False)
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    def roster_card(self, st: Style, portrait, selected):
        key = ("card", st.key, selected)

        def build():
            W, H = self.px(180), self.px(250)
            acc = _rgb255(st.accent) if max(st.accent[:3]) > 0.35 else _rgb255(st.shirt)
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=self.px(14), fill=(22, 30, 48, 240) if not selected else (30, 42, 70, 250),
                                outline=GOLD + (255,) if selected else (56, 68, 94, 255), width=self.px(3 if selected else 1))
            # backdrop glow in the player's colour, then the portrait
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse([W * 0.1, H * 0.06, W * 0.9, H * 0.66], fill=acc + (120 if selected else 70,))
            im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(self.px(26))))
            if portrait is not None:
                p = portrait.resize((self.px(150), self.px(170)), Image.LANCZOS)
                if not selected:
                    p = Image.blend(Image.new("RGBA", p.size, (0, 0, 0, 0)), p, 0.82)
                im.alpha_composite(p, (int((W - p.width) / 2), self.px(8)))
            d = ImageDraw.Draw(im)
            d.rectangle([self.px(1), self.px(174), W - self.px(2), self.px(176)], fill=acc + (255,))
            f1, f2, f3 = font(self.px(23), True, True), font(self.px(12), False, False), font(self.px(13), True, True)
            d.text((W / 2, self.px(194)), st.name, font=f1, fill=WHITE + (255,), anchor="mm")
            d.text((W / 2, self.px(214)), st.name_en.upper(), font=f2, fill=GREY + (255,), anchor="mm")
            d.text((W / 2, self.px(234)), st.title, font=f3, fill=GOLD + (255,), anchor="mm")
            hand = "左手" if st.handed == "left" else "右手"
            d.rounded_rectangle([self.px(8), self.px(8), self.px(48), self.px(28)], radius=self.px(10),
                                fill=(RED if st.handed == "left" else (50, 62, 90)) + (235,))
            d.text((self.px(28), self.px(18)), hand, font=font(self.px(11), True, True), fill=WHITE + (255,), anchor="mm")
            return im
        return self.custom(key, build)

    def select(self, img, sel, portraits, now):
        """Opponent select: six cards a page and the detail panel of the highlighted one."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "选择对手", 60, 44, 34, WHITE, "lm")
        self.label(img, "CHOOSE YOUR OPPONENT", 62, 78, 14, GREY, "lm", bold=False)
        n = len(ROSTER)
        cw, gap = 180, 16
        x0 = 60
        page, pages = sel // 6, (n + 5) // 6
        if pages > 1:
            self.rects["pageprev"] = self.label(img, "‹", 470, 58, 30, WHITE if page > 0 else (70, 80, 100), "cm")
            self.label(img, f"{page + 1} / {pages}", 530, 58, 17, GOLD, "cm")
            self.rects["pagenext"] = self.label(img, "›", 590, 58, 30, WHITE if page < pages - 1 else (70, 80, 100), "cm")
        for i, st in enumerate(ROSTER):
            if i // 6 != page:
                continue
            col, row = i % 3, (i % 6) // 3
            x, y = x0 + col * (cw + gap), 106 + row * (250 + gap)
            lift = -6 if i == sel else 0
            self.rects[f"opp{i}"] = self.blit(img, self.roster_card(st, portraits.get(st.key), i == sel), x, y + lift)
        st = ROSTER[sel]
        px0, pw = 60 + 3 * (cw + gap) + 14, 1280 - 60 - (60 + 3 * (cw + gap) + 14)
        self.blit(img, self.panel(pw, 516, PANEL, 225, 14), px0, 106)
        acc = _rgb255(st.accent) if max(st.accent[:3]) > 0.35 else _rgb255(st.shirt)
        self.blit(img, self.panel(6, 58, acc, 255, 3), px0 + 24, 130)
        self.label(img, st.name, px0 + 44, 146, 38, WHITE, "lm")
        self.label(img, st.name_en, px0 + 46, 180, 15, GREY, "lm", bold=False)
        self.blit(img, self.panel(34 + 17 * len(st.title), 30, GOLD, 255, 15), px0 + 24, 214)
        self.label(img, st.title, px0 + 24 + (34 + 17 * len(st.title)) / 2, 229, 16, INK, "cm")
        # blurb, wrapped
        words, line, y = st.blurb, "", 268
        maxc = int((pw - 48) / 17.5)
        for k in range(0, len(words), maxc):
            self.label(img, words[k:k + maxc], px0 + 24, y, 16, (214, 222, 236), "lm", bold=False)
            y += 26
        y = max(y + 12, 336)
        for name, val in st.ratings.items():
            self.label(img, name, px0 + 24, y, 15, GREY, "lm")
            bw = pw - 48 - 96
            self.blit(img, self.panel(bw, 8, (44, 54, 78), 255, 4), px0 + 70, y, anchor="lm")
            self.blit(img, self.panel(max(bw * val / 100.0, 8), 8, GOLD if val >= 95 else CYAN, 255, 4), px0 + 70, y, anchor="lm")
            self.label(img, str(val), px0 + pw - 24, y, 17, WHITE, "rm")
            y += 30
        facts = [("身高", f"{st.height_m:.2f} m"), ("持拍", "左手" if st.handed == "left" else "右手"),
                 ("杀球", f"{st.smash_speed * 3.6:.0f} km/h"), ("移动", f"{st.move_speed:.1f} m/s")]
        fx = px0 + 24
        for k, (a, b) in enumerate(facts):
            self.label(img, a, fx + k * (pw - 48) / 4, y + 22, 12, GREY, "lm", bold=False)
            self.label(img, b, fx + k * (pw - 48) / 4, y + 44, 17, WHITE, "lm")
        r = self.blit(img, self.panel(190, 46, GOLD, 255, 23), 1220, 672, anchor="rm")
        self.label(img, "下一步  Enter", 1125, 672, 19, INK, "cm")
        self.rects["next"] = r
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    def settings(self, img, rows, cur, opponent: Style, now):
        """rows: [(title, [option labels], selected index, note)]; cur: highlighted row."""
        self.dim(img, 0.36)
        self.rects = {}
        self.label(img, "比赛设置", 60, 44, 34, WHITE, "lm")
        if opponent is not None:
            self.label(img, f"对手  {opponent.name} · {opponent.title}", 62, 80, 16, GOLD, "lm")
        y = 112
        pitch = min(90.0, 524.0 / max(len(rows), 1))
        rh = pitch - 10
        for i, (title, opts, k, note) in enumerate(rows):
            on = i == cur
            self.blit(img, self.panel(1160, rh, (26, 36, 60) if on else PANEL, 235 if on else 205, 12,
                                      border=GOLD if on else None), 60, y)
            self.label(img, title, 86, y + rh * 0.36, 21, WHITE, "lm")
            if note:
                self.label(img, note, 86, y + rh * 0.72, 13, GREY, "lm", bold=False)
            x = 1196
            for j in range(len(opts) - 1, -1, -1):
                w = 40 + 19 * max(len(opts[j]), 2)
                sel = j == k
                r = self.blit(img, self.panel(w, 42, GOLD if sel else (40, 50, 74), 255, 21), x, y + rh / 2, anchor="rm")
                self.label(img, opts[j], x - w / 2, y + rh / 2, 18, INK if sel else (200, 208, 222), "cm")
                self.rects[f"set{i}_{j}"] = r
                x -= w + 12
            y += pitch
        r = self.blit(img, self.panel(220, 50, GOLD, 255, 25), 1220, 672, anchor="rm")
        self.label(img, "开打！ Enter" if opponent is not None else "好了  Enter", 1110, 672, 21, INK, "cm")
        self.rects["start"] = r
        self.rects["back"] = self.label(img, "Esc 返回", 60, 672, 15, GREY, "lm", bold=False)

    def loading(self, img, txt="正在进场…"):
        self.dim(img, 0.25)
        self.label(img, txt, 640, 360, 30, WHITE, "cm")

    def countdown(self, img, names, colors, opponent: Style, t, sub=None):
        """Walk-on card before the first serve (t: seconds since the start)."""
        op = float(np.clip(min(t / 0.3, (2.6 - t) / 0.4), 0, 1))
        if op <= 0:
            return
        y = 566                                 # lower third: the players stay visible
        self.blit(img, self.panel(760, 150, PANEL, 225, 16), 640, y, op, "cm")
        self.blit(img, self.panel(8, 86, colors["p"], 255, 4), 300, y, op, "cm")
        self.blit(img, self.panel(8, 86, colors["o"], 255, 4), 980, y, op, "cm")
        self.label(img, names["p"], 326, y - 14, 36, WHITE, "lm", opacity=op)
        self.label(img, sub or "挑战者", 328, y + 28, 15, GREY, "lm", opacity=op, bold=False)
        self.label(img, "VS", 640, y, 44, GOLD, "cm", opacity=op)
        self.label(img, names["o"], 954, y - 14, 36, WHITE, "rm", opacity=op)
        self.label(img, opponent.title, 952, y + 28, 15, GOLD, "rm", opacity=op)

    def result(self, img, match, names, colors, now, campaign=None, verdict=None, coins=None, purse=None):
        """campaign: None, or dict(stage, stars, next: is there a next stage to go on to, unit: 关 / 层).
        verdict: (nickname, remark); coins: [(what for, coins)] earned; purse: coins now."""
        self.dim(img, 0.3)
        self.rects = {}
        st = match.stats()
        won = match.won()
        self.blit(img, self.panel(820, 580, PANEL, 235, 18, border=GOLD if won else (70, 84, 112)), 640, 356, anchor="cm")
        head = "胜利！" if won else ("被取消比赛资格" if match.forfeit == "p" else "惜败")
        self.label(img, head, 640, 112, 44 if match.forfeit is None else 36, GOLD if won else WHITE, "cm")
        if campaign is not None:
            self.label(img, f"第 {campaign['stage'] + 1} {campaign.get('unit', '关')}", 500, 154, 15, GREY, "rm")
            self.stars(img, campaign["stars"], 640, 154, 24, 8, opacity=float(np.clip(now * 2.0, 0, 1)))
        else:
            self.label(img, "MATCH COMPLETE", 640, 150, 13, GREY, "cm", bold=False)
        self.label(img, names["p"], 470, 206, 26, WHITE, "rm")
        self.label(img, f"{match.score[0]}", 560, 206, 58, GOLD if won else WHITE, "cm")
        self.label(img, ":", 640, 202, 44, GREY, "cm")
        self.label(img, f"{match.score[1]}", 720, 206, 58, WHITE if won else GOLD, "cm")
        self.label(img, names["o"], 810, 206, 26, WHITE, "lm")
        rows = [("最高球速", f"{st['p']['max_speed_kmh']:.0f} km/h", f"{st['o']['max_speed_kmh']:.0f} km/h"),
                ("制胜球", st["p"]["winners"], st["o"]["winners"]),
                ("失误送分", st["p"]["errors"], st["o"]["errors"]),
                ("击球数", st["p"]["hits"], st["o"]["hits"]),
                ("杀球", st["p"]["shots"].get("smash", 0) + st["p"]["shots"].get("half_smash", 0),
                 st["o"]["shots"].get("smash", 0) + st["o"]["shots"].get("half_smash", 0))]
        if "dives" in st:
            rows.append(("鱼跃", st["dives"]["p"], st["dives"]["o"]))
        y, pitch = 262, 34 if len(rows) <= 5 else 30
        for name, a, b in rows:
            self.blit(img, self.panel(700, 1, (56, 68, 94), 255, 0), 640, y - pitch / 2, anchor="cm")
            self.label(img, str(a), 470, y, 20, WHITE, "cm")
            self.label(img, name, 640, y, 15, GREY, "cm")
            self.label(img, str(b), 810, y, 20, WHITE, "cm")
            y += pitch
        tail = f"最长回合 {st.get('rally_max', 0)} 拍 · 平均 {st.get('rally_mean', 0):.1f} 拍 · 共 {st['points']} 分"
        if st.get("cards"):
            tail += " · 吃牌：" + "、".join({"yellow": "黄", "red": "红", "black": "黑"}[c] for c in st["cards"])
        self.label(img, tail, 640, 448, 14, GREY, "cm", bold=False)
        if verdict is not None:
            op = float(np.clip((now - 0.5) * 2.5, 0, 1))
            self.blit(img, self.panel(720, 54, (30, 42, 70), 245, 14, border=GOLD, border_w=1), 640, 496, op, "cm")
            w = 36 + 20 * len(verdict[0])
            self.blit(img, self.panel(w, 34, GOLD, 255, 17), 296 + w / 2, 496, op, "cm")
            self.label(img, verdict[0], 296 + w / 2, 496, 18, INK, "cm", opacity=op)
            self.label(img, verdict[1], 310 + w, 496, 15, (224, 230, 242), "lm", bold=False, opacity=op)
        if coins:
            total = sum(n for _, n in coins)
            self.blit(img, self._disc(10, GOLD, 255, border=(176, 122, 10), bw=2), 300, 546, anchor="cm")
            self.label(img, f"+{total}", 318, 546, 21, GOLD, "lm")
            self.label(img, " · ".join(f"{a} +{n}" for a, n in coins)[:44], 336 + 13 * len(str(total)) + 14, 547, 13, GREY, "lm", bold=False)
            if purse is not None:
                self.label(img, f"现有 {purse}", 980, 547, 14, GOLD, "rm")
        go_on = campaign is not None and won and campaign["next"]
        unit = (campaign or {}).get("unit", "关")
        r = self.blit(img, self.panel(210, 46, GOLD, 255, 23), 530, 604, anchor="cm")
        self.label(img, (f"下一{unit}  Enter") if go_on else "再来一局  Enter", 530, 604, 18, INK, "cm")
        self.rects["again"] = r
        r = self.blit(img, self.panel(210, 46, (40, 50, 74), 255, 23), 750, 604, anchor="cm")
        self.label(img, ("返回天梯  Esc" if unit == "层" else "返回关卡  Esc") if campaign is not None else "换个对手  Esc", 750, 604, 18, WHITE, "cm")
        self.rects["menu"] = r

    def drill_result(self, img, match, best, coins=None):
        self.dim(img, 0.3)
        self.rects = {}
        d, log = match.drill, match.drill_log
        ok = match.score[0]
        good = ok >= 0.8 * d["reps"]
        self.blit(img, self.panel(820, 520, PANEL, 235, 18, border=GOLD if good else (70, 84, 112)), 640, 352, anchor="cm")
        self.label(img, "训练完成 · " + d["name"], 640, 128, 34, GOLD if good else WHITE, "cm")
        self.label(img, d["goal"], 640, 166, 14, GREY, "cm", bold=False)
        self.label(img, f"{ok}", 600, 226, 64, GOLD if good else WHITE, "rm")
        self.label(img, f"/ {len(log)}", 612, 234, 30, GREY, "lm")
        if best is not None:
            self.label(img, f"历史最好 {best}", 760, 238, 15, GREY, "lm", bold=False)
        if coins:
            self.blit(img, self._disc(9, GOLD, 255, border=(176, 122, 10), bw=2), 770, 212, anchor="cm")
            self.label(img, f"+{sum(n for _, n in coins)}", 786, 212, 17, GOLD, "lm")
        for k, r in enumerate(log[:12]):
            col, row = k % 2, k // 2
            x, y = 270 + 380 * col, 290 + 38 * row
            self.blit(img, self.panel(10, 10, GREEN if r["ok"] else RED, 255, 5), x, y, anchor="cm")
            self.label(img, f"{k + 1}", x + 16, y, 14, GREY, "lm", bold=False)
            txt = r["note"] + (f" · {SHOT_CN.get(r['shot'], r['shot'])} {r['kmh']} km/h" if r["shot"] and r["ok"] else "")
            self.label(img, txt, x + 44, y, 15, WHITE if r["ok"] else (214, 222, 236), "lm", bold=False)
        r = self.blit(img, self.panel(210, 48, GOLD, 255, 24), 530, 566, anchor="cm")
        self.label(img, "再练一次  Enter", 530, 566, 18, INK, "cm")
        self.rects["again"] = r
        r = self.blit(img, self.panel(210, 48, (40, 50, 74), 255, 24), 750, 566, anchor="cm")
        self.label(img, "返回  Esc", 750, 566, 18, WHITE, "cm")
        self.rects["menu"] = r

    def paused(self, img, pad=False):
        self.dim(img, 0.4)
        self.label(img, "暂停", 640, 250 if pad else 320, 54, WHITE, "cm")
        self.label(img, ("P" if pad else "空格") + " 继续 · R 重新开始 · Esc 返回菜单", 640, 314 if pad else 384, 18, GREY, "cm", bold=False)
        if pad:
            keys = [("方向键 / WASD", "瞄准落点：左右选边，上下选深浅（手动跑位时也用来跑）"),
                    ("J · K · L", "轻（吊 / 搓 / 挡）· 高（高远 / 挑）· 力（杀 / 抽 / 扑）"),
                    ("空格", "起跳：0.45 秒内再按 J / K / L 就是跳起击球，空格 + L 是跳杀"),
                    ("Ctrl + 方向", "鱼跃救球（不按方向就扑向来球）；同时按 J 是挡网，否则挑高"),
                    ("得分之后", "可以用方向键随便走走，但 15 秒内要回到位置，否则吃牌"),
                    ("T", "得分之后嘚瑟一下：摆个姿势、喊句口头禅（更衣室里换）")]
            pitch = 40
            self.blit(img, self.panel(860, pitch * len(keys) + 24, PANEL, 225, 14), 640, 356 + pitch / 2 * len(keys) + 12, anchor="cm")
            for k, (a, b) in enumerate(keys):
                y = 356 + 12 + pitch / 2 + pitch * k
                self.label(img, a, 360, y, 17, GOLD, "rm")
                self.label(img, b, 384, y, 15, (214, 222, 236), "lm", bold=False)

    def hit_test(self, x, y):
        for name, (x0, y0, x1, y1) in self.rects.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None


# ---------------------------------------------------------------------------- in-scene effects
def _add(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.asarray(rgba, np.float32))
    g.emission = 0.8
    scn.ngeom += 1
    return g


def _line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    g.emission = 0.9
    scn.ngeom += 1


def _limb(scn, a, b, r, rgba, emission=0.12):
    n0 = scn.ngeom
    _line(scn, a, b, r, rgba)
    if scn.ngeom > n0:
        scn.geoms[n0].emission = emission


def _ball(scn, c, r, rgba, emission=0.12):
    g = _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], c, np.eye(3), rgba)
    if g is not None:
        g.emission = emission


UMPIRE_X = WIDTH_DOUBLES_HALF + 0.95


def _umpire(scn, card, age):
    """The umpire on the chair beside the net (arena.py builds the chair); card: colour being shown."""
    x = UMPIRE_X
    suit, skin, hair = (0.1, 0.13, 0.24, 1.0), (0.9, 0.72, 0.58, 1.0), (0.12, 0.1, 0.09, 1.0)
    for sy in (-0.1, 0.1):
        _limb(scn, [x + 0.14, sy, 1.57], [x - 0.2, sy, 1.57], 0.07, (0.14, 0.15, 0.2, 1.0))
        _limb(scn, [x - 0.2, sy, 1.57], [x - 0.22, sy, 1.16], 0.055, (0.14, 0.15, 0.2, 1.0))
        _limb(scn, [x - 0.22, sy, 1.12], [x - 0.32, sy, 1.1], 0.045, (0.05, 0.05, 0.06, 1.0))
    _limb(scn, [x + 0.14, 0.0, 1.62], [x + 0.11, 0.0, 2.02], 0.135, suit)
    _limb(scn, [x + 0.11, -0.13, 2.05], [x + 0.11, 0.13, 2.05], 0.075, suit)
    _limb(scn, [x + 0.1, 0.0, 2.1], [x + 0.09, 0.0, 2.2], 0.045, skin)
    _ball(scn, [x + 0.08, 0.0, 2.29], 0.1, skin)
    _ball(scn, [x + 0.1, 0.0, 2.32], 0.098, hair)
    up = 0.0 if card is None else float(np.clip(age / 0.25, 0, 1) * np.clip((3.0 - age) / 0.4, 0, 1))
    for sy in (-1.0, 1.0):
        sh = np.array([x + 0.11, 0.2 * sy, 2.02])
        el, hand = np.array([x + 0.03, 0.25 * sy, 1.86]), np.array([x - 0.2, 0.13 * sy, 1.83])
        if sy < 0 and up > 0:                                    # the arm on the player's side shows the card
            el = el + up * (np.array([x - 0.02, -0.34, 2.2]) - el)
            hand = hand + up * (np.array([x - 0.06, -0.38, 2.52]) - hand)
        _limb(scn, sh, el, 0.045, suit)
        _limb(scn, el, hand, 0.04, suit)
        _ball(scn, hand, 0.045, skin)
        if sy < 0 and up > 0.5:
            g = _add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.004, 0.055, 0.08], hand + [0.0, 0.0, 0.11], np.eye(3),
                     (*(np.array(CARD_RGB[card]) / 255.0), 1.0))
            if g is not None:
                g.emission = 0.6


def _rect(scn, x0, x1, y0, y1, rgba, width=0.012, z=0.014):
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    for a, b in zip(c[:-1], c[1:]):
        _line(scn, [a[0], a[1], z], [b[0], b[1], z], width, rgba)


def _ring(scn, c, r, rgba, n=20, width=0.012, z=0.012):
    ang = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.full(n + 1, z)])
    for a, b in zip(pts[:-1], pts[1:]):
        _line(scn, a, b, width, rgba)


def scene_fx(match, now, colors, show_target=True, trail_rgb=None):
    """Returns decor(scn) for SceneRenderer.render: shuttle trail, hit flashes, landing marks and
    the meeting-point marker of the footwork assist. trail_rgb: colours of the trail behind the player's own
    shots (wardrobe), from its tail to the shuttle."""
    tcols = trail_rgb if trail_rgb and match.last_hitter == "p" else [(1.0, 0.95, 0.6)]
    trail = np.asarray(match.trail)[::2] if (match.flying or match.state == "point") and len(match.trail) > 2 else None
    fx = list(match.fx)
    cp = np.array(colors["p"], float) / 255.0
    plan, target = match.move_plan, match.move_target
    incoming = match.flying and not match.dead and match.last_hitter == "o"
    pad_ring = None
    if getattr(match, "pad", None) is not None:                 # controller: where the shuttle can be met next, and when
        h = match.pad.hint_world()
        plan = None if h is None or h["t"] is None else dict(pos=h["pos"], t=h["t"])
        target = None
        if plan is not None:
            left = plan["t"] - now
            pad_ring = (0.3, 1.0, 0.55, 0.9) if match.pad.SWING <= left <= 0.65 else (1.0, 0.85, 0.25, 0.75)
    aim = land = zone = None
    pad = getattr(match, "pad", None)
    if pad is not None and show_target and match.state == "rally" and not match.dead:
        if incoming and not pad.busy(now):                      # where the stick points the next shot
            sx, deep = pad.aim()
            auto = sx == 0.0
            if auto:
                ox = float(match._opp_xy()[0])
                sx = -np.sign(ox) if abs(ox) > 0.25 else 0.0
            aim = (np.array([sx * (WIDTH_SINGLES_HALF - 0.65), 0.9 + deep * (LENGTH_HALF - 1.5)]), auto)
        elif match.last_hitter == "p" and pad.plan is not None and match.flying:
            land = pad.plan.land[:2]                            # where the shot that is in the air was sent
    if getattr(match, "drill", None) is not None and match.state != "over":
        zone = match.drill_zone()
    ump = None
    if getattr(match, "umpire", None) is not None and match.has_arena:
        fxc = match.umpire.card_fx
        ump = (None, 0.0) if fxc is None or now - fxc[0] > 3.0 else (fxc[1], now - fxc[0])

    def decor(scn):
        if ump is not None:
            _umpire(scn, *ump)
        if zone is not None:
            x0, x1, y0, y1 = zone[0] + 0.03, zone[1] - 0.03, max(zone[2], 0.06), zone[3] - 0.03
            g = _add(scn, mujoco.mjtGeom.mjGEOM_BOX, [(x1 - x0) / 2, (y1 - y0) / 2, 0.001], [(x0 + x1) / 2, (y0 + y1) / 2, 0.006],
                     np.eye(3), (1.0, 0.9, 0.25, 0.2))
            _rect(scn, x0, x1, y0, y1, (1.0, 0.9, 0.25, 0.95), width=0.028)
        if aim is not None:
            c, auto = aim
            col = (1.0, 1.0, 1.0, 0.45) if auto else (1.0, 0.85, 0.25, 0.95)
            _ring(scn, c, 0.42, col, n=24, width=0.012 if auto else 0.018)
            _line(scn, [c[0] - 0.6, c[1], 0.014], [c[0] + 0.6, c[1], 0.014], 0.006, col)
            _line(scn, [c[0], c[1] - 0.6, 0.014], [c[0], c[1] + 0.6, 0.014], 0.006, col)
        if land is not None:
            _ring(scn, land, 0.22, (1.0, 0.85, 0.25, 0.9), width=0.014)
        if trail is not None and len(trail) > 1:
            n = len(trail)
            for i in range(n - 1):
                f = (i + 1) / n
                _line(scn, trail[i], trail[i + 1], 0.004 + 0.012 * f,
                      (*tcols[min(int(f * len(tcols)), len(tcols) - 1)], 0.08 + 0.5 * f * f))
        for e in fx:
            age = now - e["t"]
            if e["type"] == "hit" and 0 <= age < 0.22:
                f = age / 0.22
                c = (1.0, 0.85, 0.3) if e["speed"] > 50 else (1.0, 1.0, 1.0)
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05 + 0.3 * f, 0, 0], e["pos"], np.eye(3), (*c, 0.55 * (1 - f) ** 2))
            elif e["type"] == "land" and 0 <= age < 2.2:
                f = min(age / 0.35, 1.0)
                fade = float(np.clip((2.2 - age) / 0.6, 0, 1))
                c = (0.3, 1.0, 0.55) if e["inside"] else (1.0, 0.25, 0.25)
                _ring(scn, e["pos"], 0.08 + 0.2 * f, (*c, 0.9 * fade), width=0.014)
                _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.05, 0.003, 0], [e["pos"][0], e["pos"][1], 0.008], np.eye(3),
                     (*c, 0.9 * fade))
            elif e["type"] in ("net", "cord") and 0 <= age < 0.3:
                f = age / 0.3
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.06 + 0.25 * f, 0, 0], e["pos"], np.eye(3), (1, 1, 1, 0.4 * (1 - f)))
        if show_target and incoming:
            p, t = (plan["pos"], plan["t"]) if plan is not None else ((np.r_[target, 1.2], None) if target is not None else (None, None))
            if p is not None:
                left = 0.0 if t is None else max(t - now, 0.0)
                r = 0.28 + 0.9 * min(left, 1.2) / 1.2
                _ring(scn, p, 0.28, (*cp, 0.9), width=0.012)
                _ring(scn, p, r, pad_ring or (1.0, 0.85, 0.25, 0.75), n=24, width=0.008 if pad_ring is None else 0.014)
                _line(scn, [p[0], p[1], 0.02], p, 0.004, (*cp, 0.35))
    return decor


# ---------------------------------------------------------------------------- portraits
def _figure(st, size, qpos=None, podium=False):
    """Model and data of one character alone, lit for a portrait."""
    from .character import add_character, mirror_qpos, new_spec
    spec = new_spec(size[0], size[1])
    spec.visual.headlight.ambient = [0.42, 0.42, 0.44]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.15, 0.15, 0.15]
    spec.visual.quality.offsamples = 8
    lt = spec.worldbody.add_light()
    lt.pos, lt.dir, lt.diffuse = [1.5, 3.0, 3.0], [-0.4, -0.8, -0.5], [0.5, 0.5, 0.48]
    if podium:
        g = spec.worldbody.add_geom()
        g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_CYLINDER, [0.62, 0.03, 0], [0, 0, -0.03], [0.16, 0.2, 0.3, 1]
        g.contype = g.conaffinity = 0
        g = spec.worldbody.add_geom()
        g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_CYLINDER, [0.66, 0.012, 0], [0, 0, -0.045], [1.0, 0.8, 0.16, 1]
        g.contype = g.conaffinity = 0
    add_character(spec.worldbody, "", shirt=st.shirt, shorts=st.shorts, skin=st.skin, hair=st.hair,
                  racket_color=st.accent, accent=st.accent, spec=spec, dims=st.dims, handed=st.handed,
                  outfit=st.outfit or None)
    m = spec.compile()
    d = mujoco.MjData(m)
    if qpos is not None:
        d.qpos[:] = mirror_qpos(qpos) if st.handed == "left" else qpos
        d.qpos[:2] = 0
    mujoco.mj_forward(m, d)
    return m, d


class Preview:
    """The player's own character, turning on a stand (wardrobe, title). The model is rebuilt when what it
    wears changes; a frame is two small renders (colour, and segmentation for the cut-out)."""

    def __init__(self, ui, size=(420, 500)):
        self.size = (ui.px(size[0]) // 2 * 2, ui.px(size[1]) // 2 * 2)
        self.key = self.r = None
        self.cam = mujoco.MjvCamera()

    def sprite(self, st, qpos, now):
        key = (tuple(sorted(st.outfit.items())), st.handed)
        if key != self.key:
            self.close()
            self.m, self.d = _figure(st, self.size, qpos, podium=True)
            self.m.vis.global_.fovy = 36
            self.r = mujoco.Renderer(self.m, self.size[1], self.size[0])
            self.key = key
        cam = self.cam
        cam.lookat[:] = [0.0, 0.0, 1.06]
        cam.distance, cam.elevation, cam.azimuth = 4.35, -8, -90 + 24 + 38.0 * now
        r = self.r
        r.update_scene(self.d, cam)
        rgb = r.render()
        r.enable_segmentation_rendering()
        r.update_scene(self.d, cam)
        alpha = (r.render()[:, :, 0] >= 0).astype(np.uint8) * 255
        r.disable_segmentation_rendering()
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
        return np.ascontiguousarray(rgb[:, :, ::-1]).astype(np.uint16), alpha[:, :, None].astype(np.uint16)

    def close(self):
        if self.r is not None:
            self.r.close()
            self.r = self.key = None


def render_portraits(styles=ROSTER, size=(300, 340), qpos=None):
    """Cut-out renders of each opponent in their kit (RGBA PIL images) for the select screen."""
    out = {}
    for st in styles:
        m, d = _figure(st, size, qpos)
        r = mujoco.Renderer(m, size[1], size[0])
        cam = mujoco.MjvCamera()
        if qpos is not None:                    # action pose: whole figure with the racket up
            cam.lookat[:] = [0.0, 0.0, d.qpos[2] + 0.62 * st.dims["torso"]]
            cam.distance = 2.75
        else:                                   # standing: head to thigh
            cam.lookat[:] = [0.0, 0.0, d.qpos[2] + 0.55 * st.dims["torso"]]
            cam.distance = 1.9
        from .footwork import yaw_of_quat
        face = float(np.degrees(yaw_of_quat(d.qpos[3:7])))      # (the pose may be that of a player facing the other way)
        cam.azimuth, cam.elevation = -90 + face + (24 if st.handed == "left" else -24), -6
        m.vis.global_.fovy = 36
        r.update_scene(d, cam)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(d, cam)
        seg = r.render()
        r.disable_segmentation_rendering()
        alpha = (seg[:, :, 0] >= 0).astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
        out[st.key] = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
        r.close()
    return out

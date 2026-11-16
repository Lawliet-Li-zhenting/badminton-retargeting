"""The arcade layer of controller matches: stamina, the umpire (delay of game, cards) and what the
players say to it.

None of this touches the physics of a stroke: stamina changes how fast a player runs, whether a jump or a
dive is available and how well a shot is aimed; the umpire only watches the clock between rallies.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Stamina:
    """Two reserves, both 0..1.

    burst  what a rally draws on. Running close to top speed, jumping, diving and hard hitting
           empty it; standing still and the break between two points refill it -- but only up to `cap`,
           which shrinks as the match reserve runs down. Empty = winded: slower, no jump or dive
           until a third of it is back.
    tank   the match reserve. A share of every bit of burst that is spent comes out of it, and it only comes
           back a little at the interval.
    """
    RUN = 0.16             # per second at the player's own top speed (nothing below 75 % of it)
    RALLY = 0.09           # per second recovered standing still during a rally
    REST = 0.28            # per second recovered between points
    SHARE = 0.10           # of every unit of burst spent, taken out of the match reserve
    COST = dict(jump=0.12, dive=0.25, hop=0.04, smash=0.07, half_smash=0.04, kill=0.04, stroke=0.015)

    def __init__(self, economy=1.0):
        """economy: multiplier on every cost (0 = tireless, for training)."""
        self.burst, self.tank = 1.0, 1.0
        self.economy = economy
        self.winded = False
        self.spent = 0.0

    @property
    def cap(self):
        return 0.35 + 0.65 * self.tank

    def can(self, what):
        return not self.winded and self.burst >= self.COST[what] * self.economy

    def take(self, what):
        """Pay for something that has happened (a stroke, a jump)."""
        self._take(self.COST[what] * self.economy)

    def _take(self, c):
        c = min(c, self.burst)
        self.burst -= c
        self.tank = max(self.tank - self.SHARE * c, 0.0)
        self.spent += c
        if self.burst <= 0.01:
            self.winded = True

    def tick(self, dt, effort, resting):
        """effort: pelvis speed over the player's own top speed."""
        if dt <= 0:
            return
        cost = self.RUN * max(effort - 0.75, 0.0) / 0.25 * dt * self.economy
        gain = (self.REST if resting else self.RALLY * max(1.0 - effort, 0.0)) * dt
        if cost > gain:
            self._take(cost - gain)
        else:
            self.burst = min(self.burst + gain - cost, self.cap)
        self.burst = min(self.burst, self.cap)
        if self.winded and self.burst >= 0.3 * self.cap:
            self.winded = False

    def interval(self):
        self.tank = min(self.tank + 0.15, 1.0)

    # what it does to the player
    def speed(self):
        return (0.90 + 0.10 * self.tank) * (0.84 if self.winded else 1.0)

    def error(self):
        return 1.0 + 0.5 * (1.0 - self.tank) + (0.4 if self.winded else 0.0)

    def power(self):
        return 0.92 + 0.08 * self.tank - (0.08 if self.winded else 0.0)


# what the opponent does and says while it is kept waiting / after a point: (gesture, lines to pick from)
WAITING = [(6.0, "impatient", ("……", "准备好了吗？")), (9.0, "clock", ("快点吧", "都等半天了")),
           (11.5, "appeal", ("裁判！他在拖延时间！", "裁判，管管啊！"))]
AFTER_CARD = {"yellow": ("fist", ("早该罚了", "谢谢裁判")), "red": ("fist", ("白送一分，谢了", "这分我收下了")),
              "black": ("shrug", ("这就结束了？",))}
REACT = {
    "won_long": ("roar", ("漂亮！", "这球值了！", "Yes！")),          # a long rally, or the last point of the match
    "won_big": ("fist", ("漂亮！", "好球！", "Yes！")),
    "won": ("fist", ("好！", "再来")),
    "lost_error": ("dejected", ("唉……", "失误了", "不该丢的")),
    "lost_winner": ("shrug", ("这也行？", "好球", "防不住")),
    "tired": ("wipe", ("呼……", "有点累了")),
}
SECONDS = {"fist": 1.5, "roar": 2.0}                              # how long a gesture is held (default 2.4 s)
LONG_RALLY = 9                                                     # strokes
CARD_CN = {"yellow": "黄牌警告", "red": "红牌 · 判罚一分", "black": "黑牌 · 取消比赛资格"}


class Umpire:
    """Keeps the clock between rallies (BWF: play shall be continuous) and hands out cards for delay:
    a warning (yellow), then a point to the opponent for every further offence (red), and after the
    second red the match (black). Also the voice of the match: `lines` are what it has said."""
    WARN, LIMIT, AGAIN = 10.5, 15.0, 10.0

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(0)
        self.cards = []                         # every card shown to the player, in order
        self.lines = deque(maxlen=6)            # (time, text, kind: info | warn | card)
        self.card_fx = None                     # (time, colour) of the card being shown
        self.reset(0.0)

    def reset(self, t):
        """A point has ended (or a card has just been dealt with): the clock starts again."""
        self.t0 = t
        self.limit = self.LIMIT if not self.cards else self.AGAIN
        self.warned = False
        self.stage = 0                          # how far the opponent's patience has run (index into WAITING)
        self.running = True

    def stop(self):
        self.running = False

    def say(self, t, text, kind="info"):
        self.lines.append((float(t), text, kind))

    def waited(self, t):
        return max(t - self.t0, 0.0) if self.running else 0.0

    def step(self, t):
        """Returns (card or None, gesture of the opponent or None)."""
        if not self.running:
            return None, None
        w = t - self.t0
        scale = self.limit / self.LIMIT
        gesture = None
        if self.stage < len(WAITING) and w >= WAITING[self.stage][0] * scale:
            _, name, lines = WAITING[self.stage]
            gesture = (name, lines[int(self.rng.integers(len(lines)))])
            self.stage += 1
        if not self.warned and w >= self.WARN * scale:
            self.warned = True
            self.say(t, "请尽快准备比赛", "warn")
        if w < self.limit:
            return None, gesture
        reds = self.cards.count("red")
        card = "yellow" if not self.cards else ("red" if reds < 2 else "black")
        self.cards.append(card)
        self.card_fx = (float(t), card)
        self.say(t, f"拖延比赛 · {CARD_CN[card]}", "card")
        name, lines = AFTER_CARD[card]
        self.reset(t)
        return card, (name, lines[int(self.rng.integers(len(lines)))])

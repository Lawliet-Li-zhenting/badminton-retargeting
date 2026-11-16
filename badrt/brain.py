"""Tactics learned by playing: which shot, to which side, how deep -- from where, against a player standing where.

The hand-written opponents (styles.Style) draw the shot from a menu per situation and the target from a few
probabilities. A Brain keeps, for every (state, action), how the rallies went in which that choice was made:

    state   situation (kill / rear / mid / net / defence)  x  where the other player stands (front / middle /
            back; on the side the shuttle is played from / in the middle / on the other side)  x  hit in time / late
    action  shot  x  side (straight / cross / at the body, for smashes)  x  depth (short / long end of the shot's range)

    Q(s, a) = mean of (+1 rally won, -1 rally lost), discounted by 0.9 per later stroke of the same player

and plays  p(a | s)  ~  prior(a | s) * exp(beta * Q(s, a)),  where the prior is the style's own menu. With no
data the Brain is the style; with data it leans towards what has won. Nothing else about the player changes:
speed, reach, error and power stay those of the style and difficulty, and every stroke is still planned and
struck through the same swing and collision code.

Two sources of data: self-play between AI players (scripts/train_selfplay.py; the generations are kept in
assets/brain.npz), and the rallies against the person at the controls (Mind.personal, kept in the save file):
what has worked against *you* counts several times over.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .styles import SITUATIONS

MENU = {
    "kill": ("smash", "half_smash", "slice", "drop"),
    "rear": ("clear", "drop", "slice", "smash", "half_smash"),
    "mid": ("drive", "push", "block", "lift"),
    "net": ("net", "lift", "push"),
    "defence": ("lift", "block", "drive"),
}
SIDES = ("straight", "cross", "body")
DOWN = ("smash", "half_smash", "kill", "slice")          # need a contact above the tape


def _actions(sit):
    out = []
    for shot in MENU[sit]:
        for side in (0, 1):
            for deep in (0, 1):
                out.append((shot, side, deep))
        if shot in ("smash", "half_smash"):
            for deep in (0, 1):
                out.append((shot, 2, deep))
    return out


ACTIONS = {sit: _actions(sit) for sit in SITUATIONS}
NA = max(len(a) for a in ACTIONS.values())
PER_SIT = 3 * 3 * 2
NS = len(SITUATIONS) * PER_SIT
GAMMA = 0.9


def state(sit, p, other, stretch):
    """(state index, c): c = +1 / -1 is the side of the court the shuttle is played from (the racket-hand
    side when it is met in the middle); `straight` targets are on that side. p, other: the player's own frame."""
    c = float(np.sign(p[0])) if abs(p[0]) > 0.4 else 1.0
    d = abs(other[1])
    od = 0 if d < 2.9 else (1 if d < 4.6 else 2)
    o = other[0] * c
    ol = 0 if o > 0.6 else (2 if o < -0.6 else 1)
    late = 1 if stretch > 0.8 else 0
    return SITUATIONS.index(sit) * PER_SIT + od * 6 + ol * 2 + late, c


def prior(style, sit, s, attack=1.0):
    """The style's own habits as a distribution over this state's actions."""
    late, ol = s % 2, (s // 2) % 3
    menu = style.shots[sit]
    w = np.zeros(NA)
    for i, (shot, side, deep) in enumerate(ACTIONS[sit]):
        ws = menu.get(shot, 0.04)                               # (a shot the style never plays can still be learned)
        if shot in ("smash", "half_smash", "kill", "slice") and len(menu) > 1:
            ws *= attack
        if late and shot in ("smash", "half_smash", "slice", "net"):
            ws *= 0.3
        body = max(style.body, 0.05) if shot in ("smash", "half_smash") else 0.0
        if side == 2:
            wd = body
        else:
            away = 0.5 if ol == 1 else (0.5 + 0.5 * style.cross if (side == 1) == (ol == 0) else 0.5 - 0.5 * style.cross)
            wd = (1.0 - body) * away
        w[i] = ws * wd * 0.5
    return w / w.sum()


class Brain:
    """Counts and summed returns per (state, action)."""

    def __init__(self, N=None, S=None, gen=0):
        self.N = np.zeros((NS, NA)) if N is None else np.array(N, float)
        self.S = np.zeros((NS, NA)) if S is None else np.array(S, float)
        self.gen = int(gen)

    def copy(self):
        return Brain(self.N, self.S, self.gen)

    def q(self, s, extra=None, weight=3.0, k_fine=6.0, k_coarse=15.0):
        """Estimated return of every action in state s. A cell with little data falls back on: how this state
        goes on average + how much better or worse the action does over all states of the situation (a rally
        met late in defence is mostly lost whatever is played; that must not make the untried shots look good).
        extra: a second Brain (what has happened against this particular player) counted `weight` times."""
        N, S = self.N, self.S
        if extra is not None:
            N, S = N + weight * extra.N, S + weight * extra.S
        k = (s // PER_SIT) * PER_SIT
        Nc, Sc = N[k:k + PER_SIT].sum(0), S[k:k + PER_SIT].sum(0)
        vc = Sc.sum() / (Nc.sum() + 1e-9)                       # the situation on average
        adv = (Sc - Nc * vc) / (Nc + k_coarse)                  # the action's advantage in this situation
        vs = (S[s].sum() + k_fine * vc) / (N[s].sum() + k_fine)  # this state on average
        return (S[s] + k_fine * (vs + adv)) / (N[s] + k_fine)

    def probs(self, s, pri, beta, eps=0.0, extra=None):
        z = beta * self.q(s, extra)
        ok = pri > 0
        z = z - z[ok].max()
        p = pri * np.exp(z)
        p = p / p.sum()
        if eps > 0:
            p = (1 - eps) * p + eps * ok / ok.sum()
        return p

    def add(self, samples):
        for s, a, g in samples:
            self.N[s, a] += 1.0
            self.S[s, a] += g

    def decay(self, f):
        self.N *= f
        self.S *= f

    def to_json(self):
        nz = np.argwhere(self.N > 0)
        return dict(gen=self.gen, cells=[[int(s), int(a), round(float(self.N[s, a]), 3), round(float(self.S[s, a]), 3)]
                                         for s, a in nz])

    @classmethod
    def from_json(cls, d):
        b = cls(gen=d.get("gen", 0))
        for s, a, n, v in d.get("cells", []):
            if 0 <= s < NS and 0 <= a < NA:
                b.N[s, a], b.S[s, a] = n, v
        return b


FILE = Path(__file__).resolve().parents[1] / "assets" / "brain.npz"
_cache = {}


def save_generations(brains, path=FILE, meta=None):
    np.savez_compressed(path, N=np.stack([b.N for b in brains]), S=np.stack([b.S for b in brains]),
                        meta=np.array(repr(meta or {})))


def generations(path=FILE):
    """The trained generations (index 0 = no data = the style itself); [] when nothing has been trained."""
    path = Path(path)
    if path not in _cache:
        out = []
        if path.exists():
            z = np.load(path, allow_pickle=False)
            out = [Brain(n, s, i) for i, (n, s) in enumerate(zip(z["N"], z["S"]))]
        _cache[path] = out
    return _cache[path]


def generation(g, path=FILE):
    """Generation g (clamped to what exists); None when nothing has been trained."""
    gens = generations(path)
    return gens[int(np.clip(g, 0, len(gens) - 1))] if gens else None


class Mind:
    """One player's use of a Brain during a match: draws the actions, remembers them, and at the end of every
    rally turns them into samples (kept in `samples`; also added to `personal` when that is given)."""

    def __init__(self, brain, beta=3.0, eps=0.0, personal=None, learn=False):
        self.brain, self.beta, self.eps = brain, beta, eps
        self.personal = personal             # Brain: what has happened against this human (None = do not adapt)
        self.learn = learn                   # add the rallies of this match to `personal` as they end
        self.rally, self.samples = [], []
        self.rallies = 0

    def order(self, s, pri, rng, n=4):
        """Up to n different actions of state s, drawn without replacement from the policy."""
        p = self.brain.probs(s, pri, self.beta, self.eps, self.personal)
        idx = np.flatnonzero(p > 0)
        n = min(n, len(idx))
        return [int(a) for a in rng.choice(idx, size=n, replace=False, p=p[idx] / p[idx].sum())]

    def took(self, s, a):
        self.rally.append((int(s), int(a)))

    def end_rally(self, won, void=False):
        """The rally is decided: every choice made in it gets its (discounted) share of the result."""
        if not void:
            k = len(self.rally)
            out = [(s, a, (1.0 if won else -1.0) * GAMMA ** (k - 1 - i)) for i, (s, a) in enumerate(self.rally)]
            self.samples += out
            if self.learn and self.personal is not None:
                self.personal.add(out)
            self.rallies += 1
        self.rally = []

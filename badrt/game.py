"""Singles match with physical shuttle feedback.

Near half: the human (vision-retargeted pose from a camera or a recorded video) or an AI.
Far half: a command-driven AI opponent (opponent.AIPlayer) with a play style (styles.Style).

Physics runs on a fixed 1 ms tick, independent of the render rate. Every shot of both sides comes
from the same swept racket-shuttle collision (shuttle.RacketImpact): the AI's racket follows its
planned swing at tick resolution, the human's racket is interpolated between retargeted frames.
The shuttle also interacts with the net (blocked by the mesh, or clipping the tape and tumbling
over) and the floor (in / out, service court on the serve).

Help for camera players (Assist) is applied to the response of a human hit only: retargeted
racket speeds are a fraction of the real ones and the racket twist is unobservable, so the face is
squared to the swing, the swing speed is scaled, and the outgoing shot is steered towards a
legal target. Assist.realistic() turns all of it off.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from .character import KP, RACKET, CharacterIndex
from .court import (LENGTH_HALF, LINE_W, SHORT_SERVICE, WIDTH_DOUBLES_HALF, WIDTH_SINGLES_HALF, build_match_model,
                    in_singles, net_height)
from dataclasses import replace

from .footwork import Carry, Loco, travel_dist
from .opponent import TICK, AIPlayer, PadPlayer, lerp_pose
from .shuttle import (RacketImpact, RacketPose, ShuttleParams, ShuttlePhysics, ShuttleState, aim_shot, hitting_speed,
                      inverse_impact, racket_point_velocity)
from .arcade import LONG_RALLY, REACT, SECONDS, Stamina, Umpire
from .styles import PLAYER_STYLE, SHOT_CN, Style

POINT_PAUSE = 2.4
DIFFICULTY_ECONOMY = {"easy": 1.25, "normal": 1.1, "pro": 1.0, "legend": 0.8}     # how tiring the same work is for the opponent


@dataclass
class Assist:
    """Help for camera players. Assist.realistic() = pure physics."""
    hit_radius: float = 0.45         # a fast racket passing this close to the shuttle still hits it
    min_swing_speed: float = 2.5     # m/s racket-head speed that counts as a swing
    square_face: float = 0.8         # 0..1 string-bed normal turned towards the swing direction
    power: float = 2.6               # retargeted racket speed -> effective racket speed
    max_racket_speed: float = 45.0
    aim: float = 1.0                 # 0..1 outgoing shot steered towards a legal target of the same intent
    aim_scatter: float = 0.3         # m, std of the steered target (a clean hit; grows for a scrappy one)
    move_speed: float = 0.0          # m/s the avatar is carried towards the shuttle (living rooms); 0 = own legs only
    reaction: float = 0.12           # s before the avatar starts moving
    soft_swing: float = 4.6          # m/s retargeted racket speed below which the shot is a touch shot
    hard_swing: float = 7.0          # m/s ... above which it is a power shot (camera tracking tops out near 10)
    placement: float = 0.75          # 0..1 target pulled from "where the swing pointed" to the corner on that side

    @classmethod
    def realistic(cls):
        return cls(0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 0.0, 0.0)

    @classmethod
    def camera(cls, move_speed=6.0):
        """Webcam in a living room: the avatar also covers the court for you."""
        return cls(move_speed=move_speed)


class Match:
    CARRY_VMAX = 6.5                 # m/s: fastest a camera / recorded player's pelvis moves over the court

    def __init__(self, segments, opponent: Style, player="human", player_style: Style | None = None,
                 player_handed="right", assist: Assist | None = None, seed=0, points_to_win=21, difficulty="pro",
                 director=None, width=1280, height=720, arena=True, pad_auto=True, pad_bot=None, arcade=None, drill=None):
        """player: "human" (camera / recorded poses passed to update), "ai" (a second command-driven player)
        or "pad" (stick + buttons: pad_input, or the scripted pad_bot).
        arcade (default: on for a controller match): stamina for both players, the umpire's clock between
        rallies with cards for delay, gestures. drill: a training exercise (modes.DRILLS) instead of a match --
        the far player feeds shuttles, every feed is one repetition."""
        self.opp_style = opponent
        self.player_style = player_style or PLAYER_STYLE
        if player == "pad":
            self.player_style = replace(self.player_style, handed=player_handed)
        self.player_handed = player_handed if player == "human" else self.player_style.handed
        self.m = build_match_model(width, height, self.player_style, opponent, self.player_handed, arena=arena)
        self.d = mujoco.MjData(self.m)
        self.pi = CharacterIndex(self.m, "p_")
        self.oi = CharacterIndex(self.m, "o_")
        self.rng = np.random.default_rng(seed)
        self.sp = ShuttleParams()
        self.phys = ShuttlePhysics(self.sp)
        self.impact = RacketImpact(RACKET["head_half_width"], RACKET["head_half_length"], shuttle=self.sp, rng=self.rng)
        self.assist = assist or Assist()
        self.director = director
        self.opp = AIPlayer(segments, +1, opponent, self.rng, self.phys, difficulty)
        self.ai = AIPlayer(segments, -1, self.player_style, self.rng, self.phys, difficulty) if player == "ai" else None
        self.pad = self.pad_bot = None
        if player == "pad":
            self.ai = self.pad = PadPlayer(segments, -1, self.player_style, self.rng, self.phys, auto_move=pad_auto)
            self.pad_bot = pad_bot
        self._pad_in = dict(stick=np.zeros(2), press=None, jump=False, dive=False)
        self._pad_serve = None
        # what the controller did last (HUD)
        self.pad_view = dict(stick=np.zeros(2), press=None, t=-9.0, ok=True, jump_t=-9.0)
        self.player_mode = player
        self.difficulty = difficulty
        self.drill = drill
        self.has_arena = bool(arena)
        if drill is not None and drill.get("passive", True):     # the feeder stays where it feeds from
            f = drill["feeder"]
            self.opp.base = self.opp.body.base = np.array([self.opp.mx * f[0], f[1]])
        self.drill_log = []                    # one entry per repetition: dict(ok, note, shot, kmh, land)
        self.rep_void = 0
        self.arcade = (player == "pad" and drill is None) if arcade is None else bool(arcade and self.pad is not None)
        self.umpire = None
        self.bubbles = {}                      # who -> (text, from, until): what a player is saying
        self._quip_rng = np.random.default_rng(seed + 23)
        self.taunts, self._taunt_until = 0, -1.0
        self.gesture_log = []                  # (gesture, who, time)
        self.forfeit = None
        self._interval = False
        if self.arcade:
            self.opp.stam = Stamina(DIFFICULTY_ECONOMY.get(difficulty, 1.0))
            self.pad.stam = Stamina()
            self.umpire = Umpire(np.random.default_rng(seed + 11))
        if self.pad is not None:
            self.pad.body.bounds = (WIDTH_DOUBLES_HALF + 1.6, -(LENGTH_HALF + 1.8), -0.3)
        self.point_pause = 1.3 if drill is not None else POINT_PAUSE
        self.shuttle = ShuttleState(pos=np.array([0.0, 0.0, -1.0]))
        self.score = [0, 0]                    # [player, opponent]
        self.points_to_win = points_to_win
        self.server = "p" if self.rng.random() < 0.5 else "o"
        if drill is not None:
            self.server = "o"
        self.state, self.t_state = "serve", None
        self.flying = False                    # shuttle in free flight (physics on)
        self.dead = False                      # rally decided (net / floor): rackets no longer count
        self.net_hit = False
        self.last_hitter = None
        self.last_hit_t = -1.0
        self.rally_shots = 0
        self.serve_shot = False
        self.serve_started = False
        self.serve_plan = None
        self.serve_right = True
        self._placed = False
        self._serve_t0 = 0.0
        self.trail = deque(maxlen=110)
        self.events = []
        self.fx = deque(maxlen=24)             # transient effects for the presentation layer
        self.last_event = None                 # last hit / point, for the HUD
        self.rk_prev = {"p": None, "o": None}
        self.rk_now = {"p": None, "o": None}
        self.t_prev = None
        self.tick0, self.tick_n = None, 0
        self.player_q = None
        self.held = None                       # where the shuttle is held before a serve
        self.winner = None
        # camera / recorded player: where the avatar stands = the person's own position + an offset that
        # travels like a body does (acceleration limits), on legs that step while it travels
        sp = self.assist.move_speed
        self._off = Loco(np.zeros(2), acc=2.9 * max(sp, 1.0), dec=2.6 * max(sp, 1.0))
        # (pure-physics play: the racket is exactly where the person's body puts it, so the pelvis is not lowered)
        self.carry = Carry(self.m, self.d, self.pi, crouch=0.10 if self.assist.hit_radius > 0 else 0.0) if player == "human" else None
        self._raw_last = self._world_last = None
        self.move_target = self.move_plan = None
        self.racket_rel = np.array([0.45, 0.25])
        self._near_d = {"p": None, "o": None}
        self._tick_rk = {"p": None, "o": None}
        self._d_prev = {"p": np.inf, "o": np.inf}
        self._swing = (-1.0, np.zeros(3))
        self._root_last, self._root_vel = None, np.zeros(3)
        self._release_t = -1.0
        self._frame_t0, self._frame_dt = 0.0, 0.0
        self.sim_ticks = 0

    # ------------------------------------------------------------------ helpers
    def _fk_racket(self, ci):
        s = ci.racket_site
        return RacketPose(self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy())

    def _player_xy(self):
        return self.d.qpos[self.pi.qadr:self.pi.qadr + 2].copy()

    def _opp_xy(self):
        return self.d.qpos[self.oi.qadr:self.oi.qadr + 2].copy()

    def name(self, who):
        return self.player_style.name if who == "p" else self.opp_style.name

    def _racket(self, who, t):
        """Racket pose of a player at tick time t."""
        ai = self.opp if who == "o" else self.ai
        if ai is not None:
            return ai.racket_at(t)
        a, b = self.rk_prev["p"], self.rk_now["p"]
        if self._frame_dt <= 0:
            return b
        f = (t - self._frame_t0) / self._frame_dt
        return lerp_pose(a, b, min(max(f, 0.0), 1.0))

    @property
    def move_offset(self):
        return self._off.pos

    def _carry_reach(self, T):
        """Metres the footwork assist can move the avatar in T seconds (from rest, coming to rest)."""
        return travel_dist(T, self.assist.move_speed, self._off.acc, self._off.dec)

    def _footwork_assist(self, raw_root_xy, dt, now):
        """Court coverage for a player who cannot run six metres in a living room: the avatar is carried
        (an offset on top of the person's own steps, moved like a body: it accelerates, runs and brakes)
        to where the shuttle can be met, to the service position before a serve, and back afterwards.
        The swing itself stays the person's."""
        sp = self.assist.move_speed
        off = self._off
        # the person's own position cannot jump: a cut in a recording (reported by the director) or a
        # tracking glitch of the camera is taken out of the offset, so the avatar stays where it is and
        # travels the difference
        jump = self.director.pop_jump() if self.director is not None else np.zeros(2)
        if self._raw_last is not None and dt > 0:
            step = raw_root_xy - self._raw_last - jump
            n = np.linalg.norm(step)
            if n > 9.0 * dt:
                jump = jump + step * (1.0 - 9.0 * dt / n)
        self._raw_last = raw_root_xy.copy()
        off.pos = off.pos - jump
        if sp <= 0:
            if dt > 0:
                off.seek(np.zeros(2), 3.0, dt)                     # only ever non-zero after such a jump
            return off.pos
        rel = self.d.site_xpos[self.pi.racket_site][:2] - self.d.qpos[self.pi.qadr:self.pi.qadr + 2]
        self.racket_rel += (1 - np.exp(-dt / 0.4)) * (rel - self.racket_rel)
        incoming = self.flying and not self.dead and self.last_hitter != "p"
        if self.state == "serve" and not self.flying and self.serve_started:
            sx = 0.9 if self.serve_right else -0.9
            want, sp = np.array([sx, -2.7 if self.server == "p" else -3.1]) - raw_root_xy, 0.6 * sp
        elif incoming and self.move_plan is not None:
            if self.move_plan["offset"] is not None:
                want = self.move_plan["offset"].copy()             # recorded stroke: its racket onto the meeting point
            else:
                want = self.move_plan["pos"][:2] - self.racket_rel - raw_root_xy   # live: where the racket usually is
            if now < self.move_plan["t0"] + self.assist.reaction:
                sp = 0.0
        else:
            want, sp = np.zeros(2), 0.5 * sp
        want[1] = min(want[1], -0.35 - raw_root_xy[1])             # never step over the net
        if dt > 0:
            off.seek(want, sp, dt)
            # the person's own steps and the carrying add up: together no faster than a player runs
            world = raw_root_xy + off.pos
            if self._world_last is not None:
                step = world - self._world_last
                n = np.linalg.norm(step)
                if n > self.CARRY_VMAX * dt:
                    off.pos = off.pos - step * (1.0 - self.CARRY_VMAX * dt / n)
            self._world_last = raw_root_xy + off.pos
        return off.pos

    def _plan_player_intercept(self, now):
        """The opponent has hit: where and when the player can meet the shuttle (target of the footwork
        assist; with a recording, the director also times the next recorded swing to it)."""
        self.move_target = self.move_plan = None
        if self.player_mode != "human" or self.assist.move_speed <= 0:
            return
        T, P, V = self.phys.predict(self.shuttle, 3.6, dt=0.004)
        if self.director is not None and self.director.mode == "react":
            end = P[-1]
            if end[1] > 0:
                return                                             # not coming over
            if end[2] <= 0.0:
                out = max(abs(end[0]) - WIDTH_SINGLES_HALF, abs(end[1]) - LENGTH_HALF) > 0.1
                if self.rally_shots == 1 and not self.serve_in(end[0], end[1]):
                    out = True
                if out:
                    self.events.append(dict(t=float(now), leave="p"))
                    return                                         # going out: leave it
            plan = self.director.react(now, T, P, self._player_xy(), self._carry_reach, self.assist.reaction)
            if plan is not None:
                plan["t0"] = now
                self.move_plan = plan
            return
        # live player: meet it overhead if it comes down that high in reach, else at waist, else low
        root = self._player_xy()
        for z in (2.25, 1.3, 0.75):
            idx = np.flatnonzero((P[:-1, 2] >= z) & (P[1:, 2] < z) & (P[1:, 1] < -0.3))
            if idx.size == 0:
                continue
            i = int(idx[0])
            lead = float(T[i])
            dist = np.linalg.norm(P[i, :2] - self.racket_rel - root)
            if lead >= 0.4 and dist <= self._carry_reach(lead - self.assist.reaction) + 0.3:
                self.move_plan = dict(t=now + lead, pos=P[i].copy(), offset=None, t0=now)
                return

    def pad_input(self, stick, press=None, jump=False, dive=False):
        """Controller state for the next update: stick (x right, y towards the net, |.| <= 1), the shot
        button that went down since the last call ("soft" | "high" | "power" | None), and whether the
        jump / dive key went down."""
        prev = self._pad_in
        self._pad_in = dict(stick=np.array(stick, float), press=press or prev["press"],
                            jump=bool(jump) or prev["jump"], dive=bool(dive) or prev["dive"])

    def _pad_step(self, now):
        pad = self.pad
        if self.pad_bot is not None:
            out = self.pad_bot.act(self, now)
            stick, press, extra = out[0], out[1], (out[2] if len(out) > 2 else {})
        else:
            extra = self._pad_in
            stick, press = extra["stick"], extra["press"]
        self._pad_in = dict(self._pad_in, press=None, jump=False, dive=False)      # a press counts once
        live = self.state == "rally" and not self.dead
        pad.roam = self.umpire is not None and (self.state == "point" or (self.state == "serve" and self.serve_plan is None))
        pad.control(stick, live)
        view = self.pad_view
        view["stick"] = np.array(stick, float)
        t_sim = self.tick0 + self.tick_n * TICK                    # where the physics stands
        if extra.get("jump"):
            if live:
                if pad.arm_jump(t_sim):
                    view["jump_t"] = float(now)
            elif self.state != "over" and pad.swing is None and not pad.body.airborne(now):
                pad.body.hop(now)                                  # between rallies: just a jump
                view["jump_t"] = float(now)
        if extra.get("dive") and live and self.flying:
            before = pad.dives
            pad.dive(self.shuttle if self.last_hitter == "o" else None, t_sim, self._opp_xy(),
                     "soft" if press == "soft" else None)
            if pad.dives > before:
                ok = pad.plan is not None and pad.plan.t_plan == t_sim
                view.update(press="dive", t=float(now), ok=ok)
                self.events.append(dict(t=float(t_sim), pad="dive", ok=ok))
                self._tick_rk = {"p": None, "o": None}
                press = None
        if press is None:
            return
        view.update(press=press, t=float(now), ok=True)
        if self.state == "serve":
            if self.server == "p" and self.serve_plan is None and self.serve_started and pad.in_position():
                self._pad_serve = press                            # (a press while still strolling about is not a serve)
        elif live and self.flying:
            incoming = self.last_hitter == "o"
            pad.press(press, self.shuttle if incoming else None, t_sim, self._opp_xy())
            ok = pad.plan is not None and pad.plan.t_plan == t_sim
            view["ok"] = ok
            self.events.append(dict(t=float(t_sim), pad=press, ok=ok))
            self._tick_rk = {"p": None, "o": None}

    # ------------------------------------------------------------------ the arcade layer
    def won(self):
        """Did the near player win (a forfeit counts)?"""
        return self.forfeit != "p" and self.score[0] > self.score[1]

    def _vel(self, who):
        """Pelvis velocity of a player (world)."""
        ai = self.opp if who == "o" else self.ai
        if ai is None:
            return self._root_vel[:2].copy()
        if ai.t is None or ai.t_prev is None or ai.t <= ai.t_prev:
            return np.zeros(2)
        return (ai.q[:2] - ai.q_prev[:2]) / (ai.t - ai.t_prev) * [ai.mx, 1.0]

    def _emote(self, who, name, text, t, seconds=None):
        ai = self.opp if who == "o" else self.ai
        if ai is None:
            return
        seconds = SECONDS.get(name, 2.4) if seconds is None else seconds
        ai.body.emote(name, t, seconds)
        self.gesture_log.append((name, who, float(t)))
        if text:
            self.bubbles[who] = (text, float(t), float(t + seconds))

    TAUNTS = ("来呀", "就这？", "略略略", "你过来呀")

    def taunt(self, t):
        """Between rallies the player shows off (the T key): a gesture and a line -- their own catch phrases when
        they have some. The umpire has an opinion after a few. Changes nothing about the play."""
        if self.pad is None or self.umpire is None or t < self._taunt_until:
            return False
        if self.state != "point" and not (self.state == "serve" and self.serve_plan is None):
            return False
        lines = self.player_style.lines.get("won_big") or self.TAUNTS
        self._emote("p", ("roar", "fist")[self.taunts % 2], lines[int(self._quip_rng.integers(len(lines)))], t)
        self.taunts += 1
        self._taunt_until = t + 1.8
        if self.taunts in (3, 7):
            self.umpire.say(t, "注意体育道德，别太嚣张" if self.taunts == 3 else "再嘚瑟我可真掏牌了（规则里没有这条）", "warn")
        return True

    def _umpire_step(self, now, dt):
        """Between rallies: the umpire's clock runs while the player keeps the game waiting."""
        u = self.umpire
        if not u.running:
            return
        if self.state == "serve" and self.serve_plan is not None:
            u.stop()
            return
        late = not self.pad.in_position() or (self.server == "p" and self.opp.in_position())
        if not late:
            u.t0 += dt
        card, gesture = u.step(now)
        if gesture is not None and (card is not None or self.opp.in_position()):
            self._emote("o", gesture[0], gesture[1], now, 2.8)
        if card is None:
            return
        self.events.append(dict(t=float(now), card=card, to="p", reason="delay", score=list(self.score)))
        if card == "red":
            self._end_point(now, "p", "card")
        elif card == "black":
            self.forfeit, self.winner = "p", "o"
            self.state, self.t_state = "over", now

    def _react(self, t, winner, why):
        """What the two players make of the point that has just ended."""
        rng = self.umpire.rng
        err = why in ("out", "net", "short", "serve_out", "serve_fault")
        for who in ("o", "p"):
            ai = self.opp if who == "o" else self.ai
            if who == winner:
                if self.rally_shots >= LONG_RALLY or self._game_over():
                    key = "won_long"                             # only these are worth a fist above the head
                elif err:
                    key = "won"
                    if rng.random() < 0.5:
                        continue
                else:
                    key = "won_big"
            else:
                key = "lost_error" if err else "lost_winner"
                if ai.stam is not None and ai.stam.winded:
                    key = "tired"
                elif rng.random() < 0.25:
                    continue
            name, lines = REACT[key]
            if who == "o":
                text = lines[int(rng.integers(len(lines)))]      # (drawn whatever is said: how many numbers this takes
                own = self.opp_style.lines.get(key)              # depends on the number of lines, and the match's own
                if own:                                          # random numbers must not depend on the character)
                    text = own[int(self._quip_rng.integers(len(own)))]
            else:                                                # the player's own catch phrases (wardrobe); their
                mine = self.player_style.lines.get(key)          # own random numbers: the match stays the same
                text = mine[int(self._quip_rng.integers(len(mine)))] if mine else None
            self._emote(who, name, text, t + 0.45)

    # ------------------------------------------------------------------ training drills
    def drill_feed(self):
        f = self.drill["feeds"]
        return f[(len(self.drill_log) + self.rep_void) % len(f)]

    def drill_zone(self):
        """Target rectangle (x0, x1, y0, y1) of the current repetition in the far half, or None."""
        z = self.drill.get("zones")
        return None if not z else z[len(self.drill_log) % len(z)]

    def _end_rep(self, t, loser, why):
        """A feed has been dealt with: was it what the drill asks for?"""
        d = self.drill
        e = self.last_event if self.last_event is not None and self.last_event.get("who") == "p" else None
        land = self.shuttle.pos[:2].copy()
        ok, note, void = False, "", False
        if d.get("rally"):
            ok = self.rally_shots >= d["rally"] or (loser == "o" and why == "in" and self.last_hitter == "p")
            void = not ok and loser == "o"                       # the coach's own mistake
            note = f"{self.rally_shots} 拍" + ("，打死了" if ok and self.rally_shots < d["rally"] else "")
        elif self.last_hitter != "p":
            void = self.last_hitter is None or why in ("out", "serve_out", "net", "short")    # the feed itself went wrong
            note = "没接到"
        elif why in ("net", "short"):
            note = "下网" if why == "net" else "没过网"
        elif why == "out":
            note = "出界"
        elif e is not None and loser == "o":
            z = self.drill_zone()
            if d.get("shots") and e["shot"] not in d["shots"]:
                note = "界内，但球种不对（" + SHOT_CN.get(e["shot"], e["shot"]) + "）"
            elif z is not None and not (z[0] <= land[0] <= z[1] and z[2] <= land[1] <= z[3]):
                note = "界内，但没到目标区"
            elif d.get("min_kmh") and e["speed"] * 3.6 < d["min_kmh"]:
                note = f"界内，但球速不够（{e['speed'] * 3.6:.0f} km/h）"
            else:
                ok, note = True, "到位"
        else:
            note = "没打过去"
        if void:
            self.rep_void += 1
        else:
            self.drill_log.append(dict(ok=bool(ok), note=note, shot=None if e is None else e["shot"],
                                       kmh=None if e is None else round(e["speed"] * 3.6), land=land.round(2).tolist()))
        self.score = [sum(r["ok"] for r in self.drill_log), sum(not r["ok"] for r in self.drill_log)]
        self.state, self.t_state = "point", t
        self.dead = True
        self.winner = "p" if ok else "o"
        ev = dict(t=float(t), point=self.winner, why=why, land=self.shuttle.pos.round(3).tolist(), shots=self.rally_shots,
                  score=list(self.score), last_hitter=self.last_hitter, drill=dict(ok=bool(ok), note=note, void=void))
        self.events.append(ev)
        self.last_event = ev
        self.move_target = self.move_plan = None
        self.pad.clear()
        if not self._game_over():
            _, po, pp = self._stations()
            self.opp.walk_to(po)
            self.ai.walk_to(pp)

    def _feed(self, t):
        d = self.director
        return d.upcoming(t) if d is not None and d.mode == "feed" else None

    # ------------------------------------------------------------------ main update
    def update(self, now, player_qpos=None):
        """Advance the match to time `now` (seconds). player_qpos: newest retargeted pose of the
        human (for the handed character), None to keep the previous one."""
        if self.t_prev is None:
            self.t_prev = self.tick0 = self.t_state = now
        dt_frame = max(now - self.t_prev, 0.0)
        rest = self.state != "rally"
        self.opp.resting = rest
        if self.ai is not None:
            self.ai.resting = rest
        if self.pad is not None:
            self._pad_step(now)
        self.opp.update(now)
        if self.ai is not None:
            self.ai.update(now)
        elif player_qpos is not None:
            self.player_q = np.asarray(player_qpos, float)
        if self.ai is None and self.player_q is not None:
            q = self.player_q.copy()
            q[:2] += self._footwork_assist(q[:2].copy(), dt_frame, now)
            q = self.carry.step(q, now, dt_frame)
            if self._root_last is not None and dt_frame > 0:
                self._root_vel = (q[:3] - self._root_last) / dt_frame
            self._root_last = q[:3].copy()
            self.d.qpos[self.pi.qslice] = q
            mujoco.mj_kinematics(self.m, self.d)
            pose = self._fk_racket(self.pi)
            self.rk_prev["p"] = self.rk_now["p"] if self.rk_now["p"] is not None else pose
            self.rk_now["p"] = pose
        self._frame_t0, self._frame_dt = self.t_prev, dt_frame
        for who, ai in (("o", self.opp), ("p", self.ai)):
            if ai is not None:
                c0, c1 = ai.rk_prev[1].center, ai.rk_now[1].center
                self._near_d[who] = (ai.racket_center(), np.linalg.norm(c1 - c0) + 0.9)
            elif self.rk_now[who] is not None:
                c0, c1 = self.rk_prev[who].center, self.rk_now[who].center
                self._near_d[who] = (c1, np.linalg.norm(c1 - c0) + max(self.assist.hit_radius, 0.3) + 0.15)
        self._tick_rk = {"p": None, "o": None}

        if self.state == "serve":
            self._serve_logic(now, dt_frame)
        if self.umpire is not None and self.state in ("point", "serve"):
            self._umpire_step(now, dt_frame)
        if self.opp.pending and self.flying and not self.dead and self.last_hitter == "p":
            t_sim = self.tick0 + self.tick_n * TICK                        # the opponent was on the floor when this
            if t_sim >= self.opp.locked:                                   # shuttle was hit: back on its feet now
                self._replan("p", t_sim)
        while self.tick0 + (self.tick_n + 1) * TICK <= now + 1e-9:         # fixed 1 kHz grid, integer counter
            if self.flying:
                self._tick(self.tick0 + self.tick_n * TICK, self.tick0 + (self.tick_n + 1) * TICK)
            self.tick_n += 1
        if self.state == "point" and now - self.t_state > self.point_pause and not self.flying:
            if self._game_over():
                self.state, self.t_state = "over", now
            else:
                self.state, self.t_state = "serve", now
                self.serve_started = False
        # poses for rendering (a swing planned during the ticks starts inside this frame)
        self.d.qpos[self.oi.qslice] = self.opp.update(now)
        if self.ai is not None:
            self.d.qpos[self.pi.qslice] = self.ai.update(now)
        mujoco.mj_kinematics(self.m, self.d)
        self.t_prev = now
        self._place_shuttle_body()

    # ------------------------------------------------------------------ serve
    def _serve_logic(self, now, dt_frame):
        if not self.serve_started:
            self.serve_started = True
            self.flying = self.dead = self.net_hit = False
            self.last_hitter = None
            self.rally_shots = 0
            self.trail.clear()
            self.move_target = self.move_plan = None
            self.serve_right, po, pp = self._stations()
            self.serve_plan = None
            self.feed_serve = False
            self._serve_retry = now
            self._serve_t0 = now
            if self.director is not None:
                self.director.skip_idle(now)             # fast-forward the recording to its next stroke
            for ai, xy in ((self.opp, po), (self.ai, pp)):
                if ai is None:
                    continue
                if self._placed:
                    ai.walk_to(xy)                       # (already on the way since the point ended)
                else:
                    ai.place(xy)
            if self.umpire is not None and not self._placed:
                self.umpire.reset(now)                   # the clock of the first serve starts with the match
            self._placed = True
            self.events.append(dict(t=float(now), serve=self.server, score=list(self.score)))
        server = self.opp if self.server == "o" else self.ai
        if server is not None:
            if self.serve_plan is None and now >= self._serve_retry:
                ais = [a for a in (self.opp, self.ai) if a is not None]
                if not all(a.in_position() for a in ais) and (now - self._serve_t0 < 6.0 or (
                        self.umpire is not None and not self.pad.in_position())):
                    return                               # still walking to their positions (a player with a
                                                         # controller can stay away as long as the umpire lets them)
                for a in ais:
                    a.set_ready()
                if server is self.pad:                   # the person serves when they press a button
                    if self._pad_serve is not None:
                        self.serve_plan = self.pad.serve(self._pad_serve, now, self.serve_right, tick0=self.tick0)
                        self._pad_serve = None
                    if self.serve_plan is None:
                        return
                if self.serve_plan is None:
                    self.serve_plan = server.plan_serve(now, self.serve_right, tick0=self.tick0,
                                                        feed=self._feed(now) if self.server == "o" else None,
                                                        drill=None if self.drill is None else self.drill_feed())
                    self._serve_retry = now + 0.5
            if self.serve_plan is None:
                return
            t_rel, st = self.serve_plan.release
            self.held = st.pos.copy()
            self.shuttle.axis = st.axis.copy()
            if now >= t_rel:
                # the ticks of this frame after t_rel are simulated right below (tick_t <= t_rel <= now)
                self.feed_serve = bool(self.serve_plan.feed) or self.drill is not None
                self.shuttle = st.copy()
                self._release_t = t_rel
                self._start_rally(t_rel)
            return
        # human serve: the shuttle is tossed into the path of the first real swing
        wrist = KP["r_wrist"] if self.player_handed == "left" else KP["l_wrist"]
        self.held = self.d.site_xpos[self.pi.kp_sites[wrist]] + np.array([0.0, 0.1, -0.06])
        self.shuttle.axis = np.array([0.0, 0.5, -0.86])
        a, b = self.rk_prev["p"], self.rk_now["p"]
        root = self.d.qpos[self.pi.qadr:self.pi.qadr + 3].copy()
        root_prev, self._root_prev = getattr(self, "_root_prev", root), root
        if a is None or dt_frame <= 0 or now - self.t_state < 1.0:
            return
        v = (b.center - a.center) / dt_frame
        v_swing = v - (root - root_prev) / dt_frame          # a lunge alone is not a swing
        if hitting_speed(v_swing) > max(self.assist.min_swing_speed, 3.5) and b.center[2] > 0.3:
            pos = b.center + v / np.linalg.norm(v) * 0.08          # right in front of the strings: met this frame
            pos[2] = max(pos[2], 0.25)
            self._swing = (now, v.copy())
            self.shuttle = ShuttleState(pos=pos, vel=np.zeros(3), axis=np.array([0.0, 0.5, -0.86]))
            self._release_t = now
            self._start_rally(now)

    def _stations(self):
        """(serving from the right court?, far player's position, near player's position) for the next serve."""
        if self.drill is not None:
            return True, np.array(self.drill["feeder"], float), np.array(self.drill["player"], float)
        right = (self.score[0] if self.server == "p" else self.score[1]) % 2 == 0
        sx = 0.9 if right else -0.9                         # near side: the right court is +x
        if self.server == "o":
            return right, np.array([-sx, 2.6]), np.array([sx, -3.6])
        return right, np.array([-sx, 3.6]), np.array([sx, -2.6])

    def _start_rally(self, t):
        self.state, self.t_state = "rally", t
        self.flying = True
        self.dead = self.net_hit = False
        self.serve_shot = True
        self.last_hitter = None
        self.last_hit_t = -1.0
        self.held = None
        self._d_prev = {"p": np.inf, "o": np.inf}
        self.bubbles.clear()
        for ai in (self.opp, self.ai):
            if ai is not None:
                ai.body.emote_now = None

    # ------------------------------------------------------------------ physics tick
    def _tick(self, t0, t1):
        if self.last_hitter is None and t1 < self._release_t + 0.5 * TICK:
            return                                       # serve not released yet at this tick
        s0 = self.shuttle.copy()
        self.phys.step(self.shuttle, TICK)
        s1 = self.shuttle
        self.sim_ticks += 1
        if self.sim_ticks % 4 == 0:
            self.trail.append(s1.pos.copy())
        if not self.dead:
            for who in ("o", "p"):
                near = self._near_d[who]
                if near is None or np.linalg.norm(s1.pos - near[0]) > near[1]:
                    self._tick_rk[who] = None
                    self._d_prev[who] = np.inf
                    continue
                if self.last_hitter == who and t1 - self.last_hit_t < 0.3:
                    self._d_prev[who] = np.inf
                    continue
                if self.last_hitter is None and who != self.server:
                    continue                             # nobody but the server touches a serve toss
                if who == "o" and self.drill is not None and self.last_hitter is not None and self.drill.get("passive", True):
                    continue                             # a feeder only feeds: the answer flies past its racket
                cached = self._tick_rk[who]
                r0 = cached[1] if cached is not None and abs(cached[0] - t0) < 1e-9 else self._racket(who, t0)
                r1 = self._racket(who, t1)
                self._tick_rk[who] = (t1, r1)
                human = who == "p" and self.ai is None
                if self._contact(who, human, s0, s1, r0, r1, t1):
                    break
        if self._check_net(s0, s1, t1) or self._check_floor(s1, t1):
            return

    def _contact(self, who, human, s0, s1, r0, r1, t):
        hit = self.impact.check(s0, s1, r0, r1)
        a = self.assist
        if not human:
            if hit is None:
                return False
            kind, point, normal = hit
            info = self.impact.respond(self.shuttle, kind, point, normal, racket_point_velocity(r0, r1, TICK, point))
            if info is None:
                return False
            ai = self.opp if who == "o" else self.ai
            if ai.plan is not None and abs(ai.plan.t_hit - t) < 0.05:
                shot = ai.plan.shot
            else:
                # not the contact of a planned stroke. With no swing in progress the shuttle simply struck
                # the racket held in the ready position (a smash at the body): a touch, by the rules, but
                # nothing the player did. Anything else (a swing meeting the shuttle off its plan) is "return".
                shot = "deflect" if ai.swing is None else "return"
            self._register_hit(who, t, shot, info)
            return True
        # human: frame-rate racket (velocity from the two retargeted frames around this tick)
        fdt = max(self._frame_dt, 1e-4)
        rp0, rp1 = self.rk_prev["p"], self.rk_now["p"]
        v_c = (rp1.center - rp0.center) / fdt
        speed_r = float(np.linalg.norm(v_c))
        d = float(np.linalg.norm(s1.pos - r1.center))
        d_prev, self._d_prev["p"] = self._d_prev["p"], d
        # a swing is a racket moving towards the net or across the body; the backswing hits nothing.
        # Measured relative to the pelvis (as the stroke detector does): stepping back under a clear
        # while swinging is still a swing, being carried forward with a still arm is not.
        # It stays "live" for a moment after its peak
        if a.hit_radius > 0 and hitting_speed(v_c - self._root_vel) > a.min_swing_speed:
            if t - self._swing[0] > 0.1 or speed_r >= np.linalg.norm(self._swing[1]):
                self._swing = (t, v_c)
            else:
                self._swing = (t, self._swing[1])
        swinging = t - self._swing[0] < 0.1
        quality = 1.0
        if hit is not None:
            kind, point, n_geo = hit
            v_r = racket_point_velocity(rp0, rp1, fdt, point)
            if a.hit_radius > 0 and swinging and np.linalg.norm(v_r) < a.min_swing_speed:
                v_r = self._swing[1]                     # met just off the peak of the swing
        elif (a.hit_radius > 0 and swinging and d < a.hit_radius and (d > d_prev or d < 0.12)
              and s1.pos[1] < 0.3):
            # assisted: the swing passed within reach of the shuttle (closest approach just went by)
            quality = 1.0 - 0.6 * d / a.hit_radius
            kind, v_r = "strings", self._swing[1]
            n_geo = r1.R[:, 0] * (1.0 if (s1.pos - r1.center) @ r1.R[:, 0] >= 0 else -1.0)
            point = s1.pos - n_geo * (self.sp.radius + 1e-3)
        else:
            return False
        n = n_geo
        v_eff = v_r
        sp_r = float(np.linalg.norm(v_r))
        if sp_r > max(a.min_swing_speed, 1e-6) and a.power != 1.0:
            v_eff = v_r * min(a.power, a.max_racket_speed / sp_r)
        if a.square_face > 0 and sp_r > a.min_swing_speed:
            vh = v_r / sp_r
            g = n_geo if n_geo @ vh >= 0 else -n_geo
            n = (1 - a.square_face) * g + a.square_face * vh
            n /= np.linalg.norm(n)
            point = s1.pos - n * (self.sp.radius + 1e-3)
        info = self.impact.respond(self.shuttle, kind, point, n, v_eff)
        if info is None and n is not n_geo:
            info = self.impact.respond(self.shuttle, kind, point, n_geo, v_eff)
        if info is None:
            return False
        info["assisted"] = hit is None
        shot = "return"
        if a.aim > 0:
            shot = self._aim_assist(self.shuttle.pos.copy(), self.last_hitter is None, quality, v_r)
        else:
            shot = _classify_shot(self.shuttle.pos, self.shuttle.vel, self.last_hitter is None)
        info["speed_out"] = float(np.linalg.norm(self.shuttle.vel))
        self._register_hit("p", t, shot, info)
        return True

    def _aim_assist(self, contact, serve=False, quality=1.0, v_racket=None):
        """Turn a camera swing into a badminton shot. What a webcam sees reliably is where the racket
        met the shuttle (overhead / side / underhand), how hard the swing was and which way it went
        sideways; the string-bed angle it does not. So the stroke type comes from the first two, the
        side from the third, and the launch velocity is then solved so that this shot lands in.
        quality (0..1): how cleanly the shuttle was met; a scrappy hit scatters more."""
        s, a = self.shuttle, self.assist
        v = s.vel.copy()
        speed = float(np.linalg.norm(v))
        if speed < 1e-3:
            return "return"
        vr = v if v_racket is None else np.asarray(v_racket, float)
        swing = float(np.linalg.norm(vr))
        hard, soft = swing >= a.hard_swing, swing < a.soft_swing
        rising = vr[2] > 0.45 * max(swing, 1e-6)
        fwd = max(v[1], 0.35 * speed, 1.0)
        x_t = contact[0] + float(np.clip(v[0] / fwd, -0.7, 0.7)) * 5.0
        z, from_net = contact[2], -contact[1]
        if serve:
            shot, depth, sp_t, high = "serve", 6.0, 37.0, True
            x_t = -1.3 if self.serve_right else 1.3          # diagonal: the far receiver's right court is -x
        elif z >= 1.85:                                      # overhead
            if hard and not rising:
                shot, depth, sp_t, high = "smash", 4.6, float(np.clip(58.0 + 4.0 * (swing - a.hard_swing), 58.0, 74.0)), False
            elif soft:
                shot, depth, sp_t, high = "drop", 1.7, 20.0, False
            else:
                shot, depth, sp_t, high = "clear", 6.1, 52.0, True
        elif z < 1.1:                                        # underhand
            if soft and from_net < 3.0:
                shot, depth, sp_t, high = "net", 1.0, 6.5, False
            else:
                shot, depth, sp_t, high = "lift", 6.1, 40.0, True
        elif from_net > 3.8:                                 # side-arm from the back: nothing flat gets through
            if soft:
                shot, depth, sp_t, high = "drop", 1.8, 22.0, False
            else:
                shot, depth, sp_t, high = "clear", 6.0, 50.0, True
        elif hard:
            shot, depth, sp_t, high = "drive", 5.4, 34.0, False
        elif soft:
            shot, depth, sp_t, high = "push", 2.6, 15.0, False
        else:
            shot, depth, sp_t, high = "lift", 6.0, 40.0, True
        # what is tried, in order: (name, depth, speed, lob). A smash needs height: from a standing reach of
        # 2 m a 60 m/s shuttle cannot both clear the tape and come down inside, so it is taken slower and
        # deeper until it can (a "half smash"); whatever the shot, the last resort is the safe lob.
        if shot == "smash":
            tries = [("smash", depth, sp_t, False), ("smash", 5.5, sp_t, False), ("smash", 5.2, 0.85 * sp_t, False),
                     ("half_smash", 5.0, 0.72 * sp_t, False), ("half_smash", 5.7, 0.6 * sp_t, False)]
        else:
            tries = [(shot, depth, sp_t * (1 + 0.12 * k), high) for k in range(3)]
        if not serve:
            tries.append(("clear", 6.0, 50.0, True) if z >= 1.4 else ("lift", 6.0, 42.0, True))
            # side: where the swing pointed; pulled to the corner on that side (the open one when it pointed
            # at the opponent), because "to the left" is all a camera can tell
            ox = float(self._opp_xy()[0])
            side = np.sign(x_t - ox) if abs(x_t - ox) > 0.6 else (-np.sign(ox) if abs(ox) > 0.3 else np.sign(x_t + 1e-6))
            x_t = (1 - a.placement) * x_t + a.placement * side * (WIDTH_SINGLES_HALF - 0.55)
        sc = 0.0 if serve else a.aim_scatter * (1.0 + 2.0 * (1.0 - quality))
        dx, dy = self.rng.normal(0, sc, 2) if sc > 0 else (0.0, 0.0)
        axis, omega = s.axis.copy(), s.omega.copy()

        def after(v0):
            return ShuttleState(contact.copy(), np.array(v0, float), axis.copy(), omega.copy())

        vd = None
        for shot, depth, sp_t, high in tries:
            if serve:
                tx, ty = x_t, depth
            else:
                tx = float(np.clip(x_t + dx, -WIDTH_SINGLES_HALF + 0.12, WIDTH_SINGLES_HALF - 0.12))
                ty = float(np.clip(depth + dy * (1.0 if depth > 3.0 else 0.4), 0.7, LENGTH_HALF - 0.15))
            vd = aim_shot(self.phys, after, contact, (tx, ty), sp_t, high=high, net_margin=0.1)
            if vd is not None:
                break
        self._aim_dbg = dict(raw=v.round(1).tolist(), swing=round(swing, 1), shot=shot, target=[round(tx, 2), round(ty, 2)],
                             ok=vd is not None, vd=None if vd is None else vd.round(1).tolist(), q=round(quality, 2))
        if vd is None:
            return shot
        f = 1.0 if serve else self.assist.aim
        if f >= 1.0:
            s.vel = np.array(vd, float)
        else:
            u = (1 - f) * v / speed + f * vd / np.linalg.norm(vd)
            s.vel = u / np.linalg.norm(u) * ((1 - f) * speed + f * np.linalg.norm(vd))
        return shot

    def _register_hit(self, who, t, shot, info):
        first = self.last_hitter is None
        self.last_hitter = who
        self.last_hit_t = t
        self.rally_shots += 1
        self.serve_shot = first
        speed = float(np.linalg.norm(self.shuttle.vel))
        ev = dict(t=float(t), who=who, shot=str(shot), speed=speed, pos=self.shuttle.pos.round(3).tolist(),
                  kind=info.get("kind"), cork_first=info.get("cork_first"), racket_speed=info.get("racket_speed"),
                  assisted=bool(info.get("assisted", False)), serve=first)
        if shot == "deflect":
            ev["deflect"] = True
        ai = self.opp if who == "o" else self.ai
        if ai is not None and ai.plan is not None and abs(ai.plan.t_hit - t) < 0.05:
            ev.update(feed=ai.plan.feed, plan_err=float(np.linalg.norm(ai.plan.v_out - self.shuttle.vel)),
                      ik_pos=ai.plan.info.get("ik_pos"), situation=ai.plan.situation, intent=ai.plan.info.get("intent"),
                      cmd_err=ai.plan.info.get("rel_err"), land_pred=ai.plan.land[:2].round(2).tolist(),
                      jump=round(float(ai.plan.info.get("jump", 0.0)), 3))
        self.events.append(ev)
        self.last_event = ev
        self.fx.append(dict(type="hit", t=float(t), pos=self.shuttle.pos.copy(), who=who, speed=speed, shot=str(shot)))
        self._replan(who, t)

    def _replan(self, hitter, t):
        """The other side reacts to the new flight."""
        if hitter == "o":
            if self.pad is not None:
                self.pad.incoming(self.shuttle, t)
            elif self.ai is not None:
                self.ai.plan_return(self.shuttle, t, self._opp_xy(), rally=self.rally_shots, other_vel=self._vel("o"))
            else:
                self._plan_player_intercept(t)
        else:
            if self.pad is not None:
                self.pad.clear()
            if self.drill is None or not self.drill.get("passive", True):      # (a feeder does not play on)
                self.opp.plan_return(self.shuttle, t, self._player_xy(), feed=self._feed(t), rally=self.rally_shots,
                                     other_vel=self._vel("p"))
        self._tick_rk = {"p": None, "o": None}

    # ------------------------------------------------------------------ net and floor
    def _check_net(self, s0, s1, t):
        if (s0.pos[1] > 0) == (s1.pos[1] > 0):
            return False
        f = s0.pos[1] / (s0.pos[1] - s1.pos[1])
        p = s0.pos + f * (s1.pos - s0.pos)
        top = net_height(p[0])
        if abs(p[0]) > WIDTH_DOUBLES_HALF + 0.02 or p[2] >= top + self.sp.radius * 0.5:
            return False
        s = self.shuttle
        side0 = 1.0 if s0.pos[1] > 0 else -1.0
        if p[2] > top - 0.03:
            # net cord: clips the tape, loses most of its pace and tumbles over
            s.pos = np.array([p[0], -side0 * 0.02, top + self.sp.radius])
            s.vel = np.array([0.5 * s1.vel[0], 0.22 * s1.vel[1], max(0.3 * s1.vel[2], 0.0) + 0.6])
            s.omega = s.omega + self.rng.normal(0, 25.0, 3)
            self.fx.append(dict(type="cord", t=float(t), pos=p.copy()))
            self.events.append(dict(t=float(t), net_cord=True, pos=p.round(3).tolist()))
            if not self.dead and self.last_hitter is not None:
                self._replan(self.last_hitter, t)
            return False
        # caught by the mesh: it drops on the hitter's side
        s.pos = np.array([p[0], side0 * 0.035, p[2]])
        s.vel = np.array([0.15 * s1.vel[0], -0.06 * s1.vel[1], 0.25 * s1.vel[2]])
        s.omega = s.omega + self.rng.normal(0, 15.0, 3)
        self.net_hit = True
        self.dead = True
        self.fx.append(dict(type="net", t=float(t), pos=p.copy()))
        return False

    def serve_in(self, x, y):
        """Landing point inside the diagonal service court of the receiver?"""
        recv = -1.0 if self.server == "o" else 1.0               # y sign of the receiver's half
        xs = (1.0 if self.serve_right else -1.0) * (-recv)        # near receiver: right court is +x
        half = LINE_W / 2
        return (y * recv > 0 and SHORT_SERVICE - half <= abs(y) <= LENGTH_HALF + half
                and x * xs >= -half and abs(x) <= WIDTH_SINGLES_HALF + half)

    def _check_floor(self, s1, t):
        if s1.pos[2] > 0.0:
            return False
        x, y = float(s1.pos[0]), float(s1.pos[1])
        self.flying = False
        self.shuttle.vel = np.zeros(3)
        self.shuttle.omega = np.zeros(3)
        self.shuttle.pos[2] = 0.012
        if self.state != "rally":
            return True
        side = "p" if y < 0 else "o"
        inside = in_singles(x, y)
        if self.last_hitter is None:
            loser, why = self.server, "serve_fault"
        elif self.net_hit:
            loser, why = self.last_hitter, "net"
        elif self.serve_shot and not self.serve_in(x, y) and not (self.feed_serve and inside):
            loser, why = self.last_hitter, "serve_out"       # (a feed to a recording is only held to the court lines)
        elif inside:
            loser, why = side, ("in" if side != self.last_hitter else "short")
        else:
            loser, why = self.last_hitter, "out"
        self.fx.append(dict(type="land", t=float(t), pos=np.array([x, y, 0.0]), inside=bool(inside and why != "serve_out")))
        self._end_point(t, loser, why)
        return True

    def _end_point(self, t, loser, why):
        if self.drill is not None:
            return self._end_rep(t, loser, why)
        winner = "o" if loser == "p" else "p"
        self.score[0 if winner == "p" else 1] += 1
        change = self.server != winner
        self.server = winner
        self.state, self.t_state = "point", t
        self.dead = True
        self.winner = winner
        ev = dict(t=float(t), point=winner, why=why, land=self.shuttle.pos.round(3).tolist(),
                  shots=self.rally_shots, score=list(self.score), last_hitter=self.last_hitter)
        self.events.append(ev)
        self.last_event = ev
        self.move_target = self.move_plan = None
        for who, a in (("o", self.opp), ("p", self.ai)):    # learned tactics: the choices of this rally get their result
            if a is not None and a.mind is not None:
                a.mind.end_rally(winner == who, void=why in ("card", "serve_fault"))
        if self.pad is not None:
            self.pad.clear()
        if not self._game_over():                           # walk to where the next point starts
            _, po, pp = self._stations()
            self.opp.walk_to(po)
            if self.ai is not None:
                self.ai.walk_to(pp)
        if self.umpire is not None:
            u = self.umpire
            a, b = (self.score[0], self.score[1]) if winner == "p" else (self.score[1], self.score[0])
            u.say(t, ("换发球，" if change else "") + f"{a} 比 {b}")
            u.reset(t)
            if why != "card":
                self._react(t, winner, why)
            if not self._interval and max(self.score) == (self.points_to_win + 1) // 2 and not self._game_over():
                self._interval = True
                self.opp.stam.interval()
                self.pad.stam.interval()
                u.say(t + 0.01, "技术暂停，体力恢复一些")

    def _game_over(self):
        if self.drill is not None:
            return len(self.drill_log) >= self.drill["reps"] or self.rep_void >= 12
        a, b = self.score
        top = max(a, b)
        return (top >= self.points_to_win and abs(a - b) >= 2) or top >= self.points_to_win + 9

    # ------------------------------------------------------------------ statistics
    def stats(self):
        hits = [e for e in self.events if "who" in e]
        pts = [e for e in self.events if "point" in e]
        out = dict(score=list(self.score), points=len(pts), hits=len(hits))
        if self.umpire is not None:
            out.update(cards=list(self.umpire.cards), forfeit=self.forfeit,
                       stamina={w: dict(burst=round(a.stam.burst, 2), tank=round(a.stam.tank, 2), spent=round(a.stam.spent, 1))
                                for w, a in (("p", self.pad), ("o", self.opp))},
                       dives={"p": self.pad.dives, "o": self.opp.dives})
        if self.drill is not None:
            out["drill"] = dict(key=self.drill["key"], ok=self.score[0], reps=len(self.drill_log), log=self.drill_log)
        if pts:
            shots = np.array([e["shots"] for e in pts])
            out.update(rally_mean=float(shots.mean()), rally_max=int(shots.max()),
                       why={k: int(sum(e["why"] == k for e in pts)) for k in sorted({e["why"] for e in pts})})
        for who in "po":
            h = [e for e in hits if e["who"] == who]
            won = [e for e in pts if e["point"] == who]
            lost = [e for e in pts if e["point"] != who]
            kinds = {}
            for e in h:
                kinds[e["shot"]] = kinds.get(e["shot"], 0) + 1
            out[who] = dict(
                hits=len(h), max_speed_kmh=float(max([e["speed"] for e in h], default=0.0) * 3.6),
                winners=int(sum(e["why"] == "in" for e in won)),
                errors=int(sum(e["why"] in ("out", "net", "short", "serve_out", "serve_fault") for e in lost)),
                shots=kinds, assisted=int(sum(e.get("assisted", False) for e in h)),
                frame_hits=int(sum(e.get("kind") == "frame" for e in h)))
        return out

    # ------------------------------------------------------------------ rendering state
    def _place_shuttle_body(self):
        s = self.shuttle
        pos = self.held if self.held is not None else s.pos
        z = s.axis / max(np.linalg.norm(s.axis), 1e-9)
        if not self.flying and self.held is None and pos[2] < 0.05:
            z = np.array([z[0], z[1], 0.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])   # lying on its side
            z = z / max(np.linalg.norm(z), 1e-9)
        x = np.cross([0.0, 0.0, 1.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c, sn = np.cos(s.spin_angle), np.sin(s.spin_angle)
        x, y = c * x + sn * y, -sn * x + c * y
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
        self.d.mocap_pos[0] = pos
        self.d.mocap_quat[0] = q


def _classify_shot(pos, vel, serve=False):
    """Name of a shot from its launch (used when no assist decides it)."""
    if serve:
        return "serve"
    speed = float(np.linalg.norm(vel))
    elev = np.degrees(np.arcsin(np.clip(vel[2] / max(speed, 1e-6), -1, 1)))
    z = pos[2]
    if z > 1.9:
        return "smash" if (elev < 0 and speed > 26) else ("drop" if elev < 4 else "clear")
    if z < 1.1:
        return "net" if speed < 13 else "lift"
    return "lift" if elev > 15 else ("drive" if speed > 20 else "push")

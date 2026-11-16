"""Checks of the game layer: racket impact inversion, swing detection, the director that stands a
recording in for a live player, complete matches (AI vs AI, recorded player right- and left-handed, the
controller played by scripted thumbs), the arcade layer of controller matches (stamina, jump, dive,
the umpire's cards), the training drills, the campaign and scripted runs through the menus.
Prints the numbers used in the report.

    python tests/test_game.py            # all (about 6 minutes)
    python tests/test_game.py quick      # without the full 21-point games
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import modes  # noqa: E402
from badrt.app import App, Settings  # noqa: E402
from badrt.arcade import Stamina  # noqa: E402
from badrt.footwork import travel_dist  # noqa: E402
from badrt.game import Assist, Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.live import DEMO_TRACKS  # noqa: E402
from badrt.opponent import PadBot, PadPlayer, aim_shot  # noqa: E402
from badrt.recorded import Director, PlayerTrack  # noqa: E402
from badrt.shuttle import (ImpactParams, RacketImpact, ShuttleParams, ShuttlePhysics, ShuttleState,  # noqa: E402
                           hitting_speed, inverse_impact)
from badrt.styles import DIFFICULTY, ROSTER, get_style  # noqa: E402

SEGS = None


def segs():
    global SEGS
    if SEGS is None:
        SEGS = load_library(ROOT / "assets" / "stroke_library.npz")
    return SEGS


def track(handed):
    path, start, dur = DEMO_TRACKS[handed]
    return PlayerTrack(ROOT / path, start=start, duration=dur) if (ROOT / path).exists() else None


def test_inverse_impact(res):
    """inverse_impact gives the string-bed normal and racket speed; the forward impact model must
    then produce the requested outgoing velocity."""
    rng = np.random.default_rng(0)
    ip, sp = ImpactParams(), ShuttleParams()
    imp = RacketImpact(params=ip, shuttle=sp)
    errs = []
    for _ in range(200):
        v_in = np.array([rng.normal(0, 3), -rng.uniform(3, 40), rng.normal(0, 6)])
        v_out = np.array([rng.normal(0, 8), rng.uniform(5, 70), rng.normal(0, 15)])
        axis = v_in / np.linalg.norm(v_in)
        n, s, e = inverse_impact(v_in, v_out, axis, ip, sp)
        st = ShuttleState(pos=np.zeros(3), vel=v_in.copy(), axis=axis.copy())
        info = imp.respond(st, "strings", st.pos - n * (sp.radius + 1e-3), n, s * n)
        assert info is not None
        errs.append(float(np.linalg.norm(st.vel - v_out)))
    res["inverse_impact_roundtrip_max_err_mps"] = max(errs)
    assert max(errs) < 1e-6


def test_hitting_speed(res):
    fwd = hitting_speed(np.array([0.0, 8.0, -2.0]))            # towards the net (the player hits to +y)
    across = hitting_speed(np.array([6.0, 0.5, 0.0]))          # across the body
    back = hitting_speed(np.array([1.0, -8.0, 3.0]))           # backswing
    up = hitting_speed(np.array([0.0, 0.0, 9.0]))              # raising the racket
    res["hitting_speed"] = dict(forward=float(fwd), across=float(across), backswing=float(back), raise_=float(up))
    assert fwd > 7.5 and across > 3.0 and back == 0.0 and up < 1.0


def test_director(res):
    """A clear to the back court: the director must find a recorded stroke, cut the recording so
    that the stroke happens at the arrival, and place the racket on the meeting point."""
    tr = track("right")
    kinds = {}
    for s in tr.strokes:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    res["track_right"] = dict(seconds=round(tr.duration, 1), strokes=len(tr.strokes), kinds=kinds)
    assert len(tr.strokes) >= 30
    phys = ShuttlePhysics()
    out = []
    for k, (v0, now) in enumerate((((0.5, -17.0, 22.0), 3.0), ((-1.0, -15.0, 20.0), 40.0), ((0.2, -9.0, 16.0), 111.0))):
        d = Director(tr)
        d.qpos(now)
        s = ShuttleState(pos=np.array([0.3, 4.5, 2.4]), vel=np.array(v0, float), axis=np.array(v0, float) / np.linalg.norm(v0))
        T, P, V = phys.predict(s, 3.6, dt=0.004)
        plan = d.react(now, T, P, d.qpos(now)[:2], lambda sec: travel_dist(sec, 6.0, 17.4, 15.6), 0.12)
        assert plan is not None, d.reactions
        st = plan["stroke"]
        tt = d.track_time(plan["t"]) % tr.duration                 # recording time shown at the arrival
        dt = (tt - st.t + tr.duration / 2) % tr.duration - tr.duration / 2
        racket = st.pos[:2] + plan["offset"]
        out.append(dict(lead_s=round(plan["t"] - now, 2), stroke_kind=st.kind, timing_err_ms=round(abs(dt) * 1e3, 1),
                        racket_to_shuttle_xy_m=round(float(np.linalg.norm(racket - plan["pos"][:2])), 4),
                        height_diff_m=round(float(plan["pos"][2] - st.pos[2]), 2)))
        assert abs(dt) < 0.04 and np.linalg.norm(racket - plan["pos"][:2]) < 1e-6
    res["director_react"] = out


def run_match(m, director=None, fps=30.0, max_min=40):
    k = 0
    while m.state != "over" and k < fps * 60 * max_min:
        now = k / fps
        m.update(now, director.qpos(now) if director is not None else None)
        k += 1
    return k / fps


def summary(m, seconds):
    st = m.stats()
    return dict(score=m.score, finished=m.state == "over", minutes=round(seconds / 60, 1), points=st["points"],
                rally_mean=round(st.get("rally_mean", 0), 1), rally_max=st.get("rally_max", 0),
                p=dict(hits=st["p"]["hits"], winners=st["p"]["winners"], errors=st["p"]["errors"], shots=st["p"]["shots"],
                       max_kmh=round(st["p"]["max_speed_kmh"])),
                o=dict(hits=st["o"]["hits"], winners=st["o"]["winners"], errors=st["o"]["errors"], shots=st["o"]["shots"],
                       max_kmh=round(st["o"]["max_speed_kmh"])))


def test_ai_match(res, points=7):
    """Two command-driven players: every stroke is a real racket-shuttle collision that was planned.
    The one contact that is not a stroke is counted apart: a shuttle smashed at a player who is not swinging
    can strike the racket held in the ready position ("deflect"). A swing that meets the shuttle at any other
    moment than its planned contact is still a failure here."""
    m = Match(segs(), get_style("lin_dan"), player="ai", player_style=get_style("lee_chong_wei"), points_to_win=points,
              seed=0, arena=False)
    tic = time.perf_counter()
    sec = run_match(m, fps=60.0)
    out = summary(m, sec)
    touches = [e for e in m.events if "who" in e]
    hits = [e for e in touches if not e.get("deflect")]
    out["deflections_off_idle_racket"] = len(touches) - len(hits)
    assert len(touches) - len(hits) <= max(1, 0.03 * len(touches))
    errs = np.array([e["plan_err"] for e in hits if "plan_err" in e])
    cmd = np.array([e["cmd_err"] for e in hits if e.get("cmd_err") is not None])
    ms = np.array(m.opp.stats["plan_ms"] + m.ai.stats["plan_ms"])
    out.update(planned_vs_actual_velocity_err_max_mps=float(errs.max()), unplanned_contacts=len(hits) - len(errs),
               command_err_median_pct=round(float(100 * np.median(cmd)), 2), plan_ms_median=round(float(np.median(ms))),
               plan_ms_p90=round(float(np.percentile(ms, 90))), realtime_factor=round(sec / (time.perf_counter() - tic), 1))
    res["ai_vs_ai"] = out
    assert m.state == "over" and errs.max() < 0.05 and len(errs) == len(hits) and out["rally_max"] >= 4


def test_recorded_match(res, handed, opponent, difficulty, points, seed=0):
    """The front-facing recording as the player (the stand-in for a camera)."""
    tr = track(handed)
    if tr is None:
        res[f"recorded_{handed}"] = "no track"
        return
    assert tr.handed == handed
    d = Director(tr)
    m = Match(segs(), get_style(opponent), player="human", player_handed=tr.handed, points_to_win=points, seed=seed,
              director=d, difficulty=difficulty, arena=False, assist=Assist.camera())
    sec = run_match(m, d, fps=30.0)
    out = summary(m, sec)
    out.update(reactions=d.reactions, cuts=len(d.cuts), loops=d._loop,
               assisted_hits=m.stats()["p"]["assisted"], opponent_plan_ms_median=round(float(np.median(m.opp.stats["plan_ms"]))))
    lands = [e["land"] for e in m.events if "point" in e and e.get("last_hitter") == "p" and e["why"] == "out"]
    out["player_shots_out_by_more_than_1m"] = int(sum(max(abs(p[0]) - 2.59, abs(p[1]) - 6.7) > 1.0 for p in lands))
    res[f"recorded_{handed}_{opponent}_{difficulty}_{points}"] = out
    assert m.state == "over" and out["p"]["hits"] >= points and d.reactions["found"] > 0.6 * d.reactions["asked"]
    assert out["player_shots_out_by_more_than_1m"] == 0
    return out


def test_pad_match(res, opponent, handed, auto, points=7, seed=0):
    """Controller mode, played by scripted thumbs through the same stick + three buttons a person has.
    Every stroke is still a planned, real collision; nothing is hit without a button."""
    m = Match(segs(), get_style(opponent), player="pad", player_handed=handed, points_to_win=points, seed=seed,
              difficulty="normal", arena=False, pad_auto=auto, pad_bot=PadBot(np.random.default_rng(seed)))
    sec = run_match(m, fps=60.0, max_min=15)
    out = summary(m, sec)
    touches = [e for e in m.events if "who" in e]
    mine = [e for e in touches if e["who"] == "p" and not e.get("deflect")]
    errs = np.array([e["plan_err"] for e in mine if "plan_err" in e])
    presses = [e for e in m.events if "pad" in e]
    body = np.array(m.pad.stats["plan_ms"])
    out.update(presses=len(presses), presses_that_became_a_swing=int(sum(e["ok"] for e in presses)),
               strokes=len(mine), unplanned_contacts=len(mine) - len(errs),
               planned_vs_actual_velocity_err_max_mps=float(errs.max()), press_ms_median=round(float(np.median(body))),
               press_ms_p90=round(float(np.percentile(body, 90))), move="auto" if auto else "stick", handed=handed)
    st = m.stats()
    out.update(jump_strokes=int(sum(e.get("jump", 0) > 0.08 for e in mine)), dives=st["dives"], cards=st["cards"],
               stamina=st["stamina"])
    res[f"pad_{opponent}_{handed}_{'auto' if auto else 'stick'}"] = out
    assert m.state == "over" and len(errs) == len(mine) and errs.max() < 0.05
    assert out["presses_that_became_a_swing"] >= 0.6 * len(presses) and out["rally_max"] >= 4
    assert m.player_handed == handed and not st["cards"]
    assert st["stamina"]["p"]["tank"] < 1.0 and st["stamina"]["o"]["tank"] < 1.0
    return out


def test_stamina(res):
    """The two reserves on their own: running flat out empties the rally bar and takes its share out of the match
    reserve; the break between points refills the bar only as far as the match reserve allows."""
    st = Stamina()
    t = 0.0
    while not st.winded and t < 60:
        st.tick(1 / 60, PadPlayer.RUN, False)               # at the controller player's top speed without a stop
        t += 1 / 60
    tank = st.tank
    out = dict(run_seconds_to_winded=round(t, 1), match_reserve_after=round(tank, 3),
               speed_winded=round(st.speed(), 3), can_jump_winded=st.can("jump"))
    assert 6.0 < t < 20.0 and st.winded and not st.can("jump") and not st.can("dive") and 0.85 < tank < 0.95
    t = 0.0
    while st.burst < st.cap - 1e-6 and t < 60:
        st.tick(1 / 60, 0.0, True)                          # the break after the point
        t += 1 / 60
    out.update(rest_seconds_to_full=round(t, 1), refilled_to=round(st.burst, 3), cap=round(st.cap, 3))
    assert not st.winded and abs(st.burst - st.cap) < 1e-6 and st.cap < 1.0 and st.tank == tank
    st.tick(5.0, 0.5, False)                                # jogging below 75 % of top speed costs nothing
    assert st.burst == st.cap
    for _ in range(200):                                    # a whole match of hard work
        st.take("smash"), st.take("jump"), st.tick(8.0, 0.0, True)
    out.update(after_200_jump_smashes=dict(tank=round(st.tank, 3), cap=round(st.cap, 3), error_x=round(st.error(), 2),
                                           power_x=round(st.power(), 3), speed_x=round(st.speed(), 3)))
    assert st.tank < 0.05 and 0.35 <= st.cap < 0.4 and st.error() > 1.4
    res["stamina"] = out


def test_jump(res):
    """One running speed (there is no sprint key). Space: the armed stroke is met higher, off the floor, and a
    smash is faster."""
    m = Match(segs(), get_style("chen_long"), player="pad", points_to_win=5, seed=0, arena=False)
    pad, b = m.pad, m.pad.body
    m.update(0.0)
    pad.place(np.array([-2.0, -4.0]))
    pad.control(np.zeros(2), False)
    b.set_speed(pad._speed_k())
    assert b.k <= pad.RUN + 1e-9 and not hasattr(pad, "sprint") and not hasattr(pad, "SPRINT")
    out = dict(seconds_for_4m=round(b.travel_time(4.0), 2), top_speed=round(pad.RUN * b.v0, 2))
    # a clear coming down in the back court, the power button 0.62 s before it could be met standing
    rows = []
    for k, x in enumerate((-1.2, 0.0, 1.2, 0.6)):
        s0 = ShuttleState(pos=np.array([0.3, 4.6, 2.2]), vel=np.zeros(3))
        v = aim_shot(m.phys, lambda v0: ShuttleState(s0.pos.copy(), np.array(v0, float), np.array(v0, float) / np.linalg.norm(v0)),
                     s0.pos, (x, -5.6), 50.0, high=True, net_margin=0.3)
        s0.vel, s0.axis = v, v / np.linalg.norm(v)
        row = {}
        for name, jump in (("standing", False), ("jump", True)):
            pad.place(np.array([x, -5.2]))
            pad.swing, pad.locked, pad._jump_until, pad.stam = None, -1.0, None, None
            pad.control(np.zeros(2), True)
            pad.incoming(s0, 0.0)
            T, P = pad.track[1][0], pad.track[1][1]
            pad.update(0.0)
            pad._refresh_hint(0.0)
            t_press = pad.hint["t"] - 0.62
            sh = ShuttleState(pos=P[int(np.searchsorted(T, t_press))].copy(), vel=pad.track[1][2][int(np.searchsorted(T, t_press))].copy())
            sh.axis = sh.vel / np.linalg.norm(sh.vel)
            pad.t = pad.t_prev = None
            pad.update(t_press)
            if jump:
                assert pad.arm_jump(t_press)
            plan = pad.press("power", sh if pad.mx > 0 else sh, t_press)
            assert plan is not None, (name, pad.message, pad.fail)
            row[name] = dict(shot=plan.shot, contact_z=round(float(plan.contact[2]), 2), jump_m=round(plan.info["jump"], 2),
                             kmh=round(float(np.linalg.norm(plan.v_out)) * 3.6))
        rows.append(row)
    out["power_button_on_a_clear"] = rows
    res["jump"] = out
    assert all(r["jump"]["jump_m"] > 0.1 and r["standing"]["jump_m"] == 0.0 for r in rows)
    assert all(r["jump"]["contact_z"] > r["standing"]["contact_z"] + 0.1 for r in rows)
    assert np.mean([r["jump"]["kmh"] for r in rows]) > np.mean([r["standing"]["kmh"] for r in rows])


def test_dive(res, n=40):
    """Smashes to the side lines, the dive key 0.18 s after they are hit: how many the feet reach, how many
    only a dive does, and that the dive's lift comes back into the court."""
    m = Match(segs(), get_style("chen_long"), player="pad", points_to_win=5, seed=0, arena=False)
    pad = m.pad
    m.update(0.0)
    rng = np.random.default_rng(4)
    stand = dive = both = lands_in = 0
    for k in range(n):
        side = 1.0 if k % 2 else -1.0
        s0 = ShuttleState(pos=np.array([rng.uniform(-1.0, 1.0), rng.uniform(3.6, 5.2), 2.75]), vel=np.zeros(3))
        tgt = (side * rng.uniform(1.9, 2.45), -rng.uniform(3.6, 5.4))
        v = aim_shot(m.phys, lambda v0: ShuttleState(s0.pos.copy(), np.array(v0, float), np.array(v0, float) / np.linalg.norm(v0)),
                     s0.pos, tgt, rng.uniform(52.0, 66.0), high=False, net_margin=0.12)
        if v is None:
            continue
        s0.vel, s0.axis = v, v / np.linalg.norm(v)
        T, P, V = m.phys.predict(s0, 0.18, dt=0.002)
        sh = ShuttleState(pos=P[-1].copy(), vel=V[-1].copy(), axis=V[-1] / np.linalg.norm(V[-1]))
        got = {}
        for name in ("stand", "dive"):
            pad.place(np.array([0.0, -4.2]))
            pad.swing, pad.locked, pad._jump_until, pad.stam, pad.plan = None, -1.0, None, None, None
            pad.t = pad.t_prev = None
            pad.control(np.zeros(2), True)
            pad.update(0.18)
            plan = pad.press("high", sh, 0.18) if name == "stand" else pad.dive(sh, 0.18)
            got[name] = plan
        both += 1
        stand += got["stand"] is not None
        if got["stand"] is None and got["dive"] is not None:
            dive += 1
            x, y = got["dive"].land[:2]
            lands_in += abs(x) < 2.59 and 0.0 < y < 6.7
            assert got["dive"].shot == "dive" and pad.locked > 0.18 + 0.8
    res["dive"] = dict(smashes=both, reached_on_the_feet=stand, reached_only_by_a_dive=dive, dive_lifts_planned_in=lands_in)
    assert dive >= 0.3 * (both - stand) and dive >= 5 and lands_in >= 0.9 * dive


def test_cards(res):
    """Delay of game: scripted thumbs that keep strolling about. 15 s: yellow; every further 10 s: red and a
    point to the opponent; after the second red: black, the match is forfeited. And 8 s of strolling: nothing."""
    m = Match(segs(), get_style("lin_dan"), player="pad", points_to_win=5, seed=1, arena=False,
              pad_bot=PadBot(np.random.default_rng(1), dawdle=1e9))
    sec = run_match(m, fps=60.0, max_min=3)
    cards = [(round(e["t"], 1), e["card"]) for e in m.events if "card" in e]
    said = [txt for _, txt, kind in m.umpire.lines if kind != "info"]
    gestures = sorted({e[0] for e in getattr(m, "gesture_log", [])})
    pts = [e for e in m.events if "point" in e]
    played = [e for e in pts if e["why"] != "card"]              # (the opponent may serve before the strolling starts)
    t0 = max([e["t"] for e in played if e["t"] < cards[0][0]], default=0.0) if cards else 0.0
    res["cards_dawdling"] = dict(cards=cards, clock_started=round(t0, 1), score=m.score, points_by_red_card=len(pts) - len(played),
                                 forfeit=m.forfeit, seconds=round(sec, 1), won=m.won(), opponent_gestures=gestures,
                                 umpire_said_last=said[-3:])
    assert [c for _, c in cards] == ["yellow", "red", "red", "black"] and m.forfeit == "p" and m.state == "over"
    assert len(pts) - len(played) == 2 and m.score[1] >= 2 and not m.won()
    assert abs(cards[0][0] - t0 - 15.0) < 0.2 and all(abs(b[0] - a[0] - 10.0) < 0.2 for a, b in zip(cards[:-1], cards[1:]))
    assert {"impatient", "clock", "appeal", "fist"} <= set(gestures)
    m = Match(segs(), get_style("lin_dan"), player="pad", points_to_win=5, seed=1, difficulty="easy", arena=False,
              pad_bot=PadBot(np.random.default_rng(1), dawdle=8.0))
    sec = run_match(m, fps=60.0, max_min=15)
    pts = [e for e in m.events if "point" in e]
    served = [h["t"] for h in m.events if "who" in h and h.get("serve")]
    gaps = [min(t for t in served if t > p["t"]) - p["t"] for p in pts[:-1]]
    res["cards_strolling_8s"] = dict(cards=list(m.umpire.cards), score=m.score, points=len(pts),
                                     seconds_from_point_to_next_serve=dict(min=round(min(gaps), 1), max=round(max(gaps), 1)))
    assert m.state == "over" and not m.umpire.cards and m.forfeit is None and min(gaps) > 8.0 and max(gaps) < 15.0


class DrillBot(PadBot):
    """Scripted thumbs that know what the drill asks for: always this button, aimed at the target zone."""

    def __init__(self, button, rng, jump=False):
        super().__init__(rng, extras=False)
        self.button, self.jump = button, jump

    def act(self, match, now):
        stick, press, extra = super().act(match, now)
        if press is not None and match.state == "rally":
            z = match.drill_zone()
            sx = 0.0 if z is None or (z[0] < 0 < z[1]) else float(np.sign(z[0] + z[1]))
            stick, press = np.array([sx, 0.0]), self.button
        return stick, press, extra


def test_drills(res):
    """Every drill once with thumbs that press at random (it has to run to its end and judge every feed), and
    the ones with a set task again with thumbs that do what is asked."""
    out = {}
    for d in modes.DRILLS:
        m = Match(segs(), modes.COACH, player="pad", seed=3, difficulty="easy", arena=False,
                  pad_bot=PadBot(np.random.default_rng(3)), drill=d)
        run_match(m, fps=60.0, max_min=8)
        feeds = [e for e in m.events if e.get("who") == "o" and e.get("serve")]
        out[d["key"]] = dict(random_thumbs=f"{m.score[0]}/{len(m.drill_log)}", feeds=len(feeds), void=m.rep_void)
        assert m.state == "over" and len(m.drill_log) == d["reps"] and m.rep_void <= 2, (d["key"], m.state, len(m.drill_log))
        assert m.umpire is None and m.pad.stam is None
        if d.get("passive", True):
            assert all(e["who"] == "p" or e.get("serve") for e in m.events if "who" in e)      # the feeder only feeds
    for key, button, need in (("clear", "high", 8), ("drop", "soft", 7), ("smash", "power", 7), ("net", "soft", 6)):
        m = Match(segs(), modes.COACH, player="pad", seed=5, difficulty="easy", arena=False,
                  pad_bot=DrillBot(button, np.random.default_rng(5)), drill=modes.drill(key))
        run_match(m, fps=60.0, max_min=8)
        notes = {}
        for r in m.drill_log:
            notes[r["note"]] = notes.get(r["note"], 0) + 1
        out[key].update(right_button=f"{m.score[0]}/{len(m.drill_log)}", notes=notes)
        assert m.state == "over" and m.score[0] >= need, (key, m.score, notes)
    res["drills"] = out


def test_campaign(res):
    """The ladder is playable data, the save file round-trips, and a stage played through the app is written to it."""
    keys = [s.key for s in ROSTER]
    assert len(modes.CAMPAIGN) == 12 and sorted(k for k, _, _ in modes.CAMPAIGN) == sorted(keys)
    assert all(d in DIFFICULTY and p in (11, 15, 21) for _, d, p in modes.CAMPAIGN)
    order = list(DIFFICULTY)
    assert [order.index(d) for _, d, _ in modes.CAMPAIGN] == sorted(order.index(d) for _, d, _ in modes.CAMPAIGN)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "save" / "progress.json"
        pr = modes.Progress(path)
        assert pr.next_open() == 0 and pr.unlocked(0) and not pr.unlocked(1)
        app = App(size=(640, 360), settings=Settings(quality=0), headless=True, render=False, start="match", mode="campaign",
                  stage=0, pad_bot=True, save=path, seed=0, log=Path(tmp) / "log.json")
        r = app.run()[0]
        again = modes.Progress(path)
        res["campaign_stage_1"] = dict(opponent=r["opponent"], difficulty=r["difficulty"], score=r["score"], stars=r["stars"],
                                       cards=r["stats"]["cards"], saved=again.stars[:3], minutes=round(r["game_seconds"] / 60, 1))
        assert r["mode"] == "campaign" and r["stage"] == 1 and r["opponent"] == modes.CAMPAIGN[0][0]
        assert r["stars"] == (0 if r["score"][0] < r["score"][1] else 1 + (not r["stats"]["cards"]) * (1 + (2 * r["score"][1] <= r["score"][0])))
        assert again.stars[0] == r["stars"] and again.unlocked(1) == (r["stars"] > 0)
        again.stage_done(0, 1)                                  # a worse result never overwrites a better one
        assert modes.Progress(path).stars[0] == r["stars"]
        path.write_text("not json", encoding="utf-8")
        assert modes.Progress(path).stars == [0] * 12            # an unreadable save is a new game, not a crash


def flow(app, seconds):
    seen = []
    while app.clock < seconds:
        img = app.frame(app.fixed_dt)
        assert img is not None and img.shape == (app.H, app.W, 3)
        if not seen or seen[-1] != app.state:
            seen.append(app.state)
    return seen


def test_pad_idle(res, seconds=40.0):
    """Hands off the controller: the character recovers and follows the play but never serves or swings."""
    m = Match(segs(), get_style("lin_dan"), player="pad", points_to_win=5, seed=1, arena=False)
    k = 0
    while m.state != "over" and k < 60 * seconds:
        m.update(k / 60.0)
        k += 1
    strokes = [e for e in m.events if e.get("who") == "p" and not e.get("deflect")]
    res["pad_idle"] = dict(score=m.score, seconds=round(k / 60.0, 1), player_strokes=len(strokes), state=m.state,
                           cards=list(m.umpire.cards))
    assert not strokes and m.score[0] == 0


def test_app_flow(res):
    """Title -> mode -> opponent -> settings -> match, driven by the key script, rendered headless; the same
    into a campaign stage, a ladder rung and a drill; and the menus by hand: pages of the roster, Tab settings,
    Esc, the wardrobe."""
    target = Settings(opponent=8, handed=1, difficulty=3, points=0, source=1, quality=0, tactics=1)
    app = App(size=(640, 360), settings=Settings(), auto=target, headless=True, duration=20.0, save=None,
              log=ROOT / "results" / "_app_flow.json")
    seen = flow(app, 20.0)
    s = app.sess
    res["app_flow"] = dict(states=seen, opponent=s.opp.key, settings=vars(app.settings), frames=app.frames)
    assert seen == ["title", "mode", "select", "settings", "loading", "match"]
    assert app.settings == target and s.opp.key == ROSTER[8].key and s.match.points_to_win == 11
    assert s.match.difficulty == "legend" and s.match.umpire is None           # (an AI exhibition has no arcade layer)
    assert s.mind is not None and s.match.opp.mind is s.mind and s.mind.personal is None and s.match.ai.mind is None
    assert [r[0] for r in app.settings.rows()] == ["handed", "difficulty", "points", "tactics", "source", "move", "quality"]
    s.close()
    out = {}
    for mode, kw in (("campaign", {}), ("ladder", {}), ("training", dict(drill=4))):
        app = App(size=(640, 360), settings=Settings(quality=0), auto=Settings(quality=0), headless=True, save=None, mode=mode,
                  pad_bot=True, **kw)
        seen = flow(app, 13.0)
        s = app.sess
        out[mode] = dict(states=seen, opponent=s.opp.key, difficulty=s.match.difficulty, points=s.match.points_to_win,
                         drill=None if s.drill is None else s.drill["key"])
        assert seen == ["title", "mode", {"campaign": "stages", "ladder": "ladder", "training": "drills"}[mode], "loading", "match"]
        if mode == "campaign":
            assert (s.opp.key, s.match.difficulty, s.match.points_to_win) == modes.CAMPAIGN[0] and s.match.umpire is not None
            assert s.mind is None and s.match.opp.mind is None
        elif mode == "ladder":
            r = modes.LADDER[0]
            assert (s.opp.key, s.match.difficulty, s.match.points_to_win) == (r["style"].key, r["diff"], r["points"])
            assert s.match.opp.mind is s.mind and s.mind.personal is app.progress.personal and s.mind.learn
            assert s.match.umpire is not None and s.match.player_style.outfit == app.progress.outfit
        else:
            assert s.drill["key"] == modes.DRILLS[4]["key"] and s.match.drill is s.drill and s.opp.key == "coach"
        s.close()
    res["app_flow_modes"] = out
    # by hand
    app = App(size=(640, 360), settings=Settings(quality=0), headless=True, save=None)

    def keys(*ks):
        app.inputs += list(ks)
        assert app.frame(app.fixed_dt) is not None
        return app.state

    assert keys("enter") == "mode" and keys("right") == "mode" and app.mode == "campaign"
    assert keys("enter") == "stages" and app.stage == 0
    assert keys("right", "enter") == "stages" and app.stage == 1              # locked: stays
    assert keys("tab") == "settings" and len(app.settings.rows(app.mode)) == 3
    assert keys("right") == "settings" and app.settings.handed == 1
    assert keys("enter") == "stages" and keys("esc") == "mode"
    assert keys("left", "enter") == "select"
    for _ in range(7):
        keys("right")
    assert app.settings.opponent == 7 and "opp7" in app.ui.rects and "opp0" not in app.ui.rects   # second page
    assert keys("@pageprev") == "select" and app.settings.opponent == 1 and "opp0" in app.ui.rects
    assert keys("esc") == "mode" and keys("esc") == "title"
    # the wardrobe: Tab from the title, a free item is put on, a dear one is refused, Esc leads back
    from badrt import wardrobe
    pr = app.progress
    coins = pr.coins
    assert keys("tab") == "wardrobe" and app.w_focus == 0 and wardrobe.SLOTS[app.w_slot][0] == "suit"
    assert keys("right") == "wardrobe" and app.w_focus == 1
    assert keys("down", "enter") == "wardrobe" and pr.outfit["suit"] == "tank" and pr.coins == coins      # (free)
    dear = list(wardrobe.SUITS).index("gold") - list(wardrobe.SUITS).index("tank")
    assert keys(*["down"] * dear, "enter") == "wardrobe" and pr.outfit["suit"] == "tank" and pr.coins == coins
    cheap = list(wardrobe.SUITS).index("school")
    app.w_item = cheap
    assert keys("enter") == "wardrobe" and pr.outfit["suit"] == "school" and pr.coins == coins - wardrobe.price("suit", "school")
    assert pr.has("suit", "school") and keys("esc") == "wardrobe" and app.w_focus == 0
    assert keys("r") == "wardrobe" and all(pr.has(sl, pr.outfit[sl]) for sl in wardrobe.CATALOG)
    assert keys("esc") == "title"
    app.preview.close()


def test_pad_keys(res):
    """The keyboard mapping of PadInput.poll with a faked key state (nobody can hold a real Ctrl in a
    test): stick from the arrows / WASD, shot buttons on their edges, Shift = nothing, Space edge = jump,
    Ctrl = dive towards the direction held or pressed within DIVE_WAIT, otherwise at the shuttle."""
    from badrt.app import PadInput

    class FakeUser32:
        keys = set()

        def GetAsyncKeyState(self, vk):
            return 0x8000 if vk in self.keys else 0

    class FakeUI:
        def pad_geometry(self):
            return (100, 100, 50), {}

    def fresh():
        p = PadInput(FakeUI(), None)
        p.user32, p._focused = FakeUser32(), (lambda: True)
        p.user32.keys = set()
        return p

    def poll(p, t, *vks):
        p.user32.keys = set(vks)
        stick, press, extra = p.poll(t)
        return [[round(float(v), 2) for v in stick], press, extra]

    LEFT, RIGHT, UP, A, S, J, L, SHIFT, SPACE, CTRL, I = 0x25, 0x27, 0x26, 0x41, 0x53, 0x4A, 0x4C, 0x10, 0x20, 0x11, 0x49
    out = {}
    p = fresh()
    out["nothing"] = poll(p, 0.00)
    out["shift_held"] = [poll(p, 0.01, SHIFT), poll(p, 0.02, SHIFT, 0x55)]                   # (0x55: U, its old alias)
    out["space"] = [poll(p, 0.10, SPACE)[2]["jump"], poll(p, 0.11, SPACE)[2]["jump"], poll(p, 0.12)[2]["jump"]]
    out["up_then_J"] = [poll(p, 0.20, UP), poll(p, 0.21, UP, J), poll(p, 0.22, UP, J)]
    out["space_then_L_with_left"] = [poll(p, 0.30, SPACE), poll(p, 0.31, SPACE, LEFT, L)]
    p = fresh()
    out["ctrl_with_left_held"] = [poll(p, 0.00, LEFT), poll(p, 0.01, LEFT, CTRL), poll(p, 0.02, LEFT, CTRL)]
    p = fresh()
    out["ctrl_then_right"] = [poll(p, 0.00, CTRL), poll(p, 0.05, CTRL, RIGHT), poll(p, 0.06, CTRL, RIGHT)]
    p = fresh()
    out["ctrl_alone"] = [poll(p, 0.00, CTRL), poll(p, 0.06, CTRL), poll(p, 0.13, CTRL), poll(p, 0.14, CTRL)]
    p = fresh()
    out["ctrl_S_is_a_dive_backwards"] = [poll(p, 0.00, CTRL, S)]
    p = fresh()
    out["I_with_A"] = [poll(p, 0.00, A), poll(p, 0.01, A, I)]
    res["pad_keys"] = out
    off = dict(jump=False, dive=False)
    assert out["nothing"] == [[0.0, 0.0], None, off]
    assert out["shift_held"] == [[[0.0, 0.0], None, off]] * 2 and out["space"] == [True, False, False]
    from badrt.ui import UI
    assert "sprint" not in PadInput.VK and "sprint" not in UI.PAD_BUTTONS and 0x10 not in sum(PadInput.VK.values(), ())
    assert out["up_then_J"] == [[[0.0, 1.0], None, off], [[0.0, 1.0], "soft", off], [[0.0, 1.0], None, off]]
    assert out["space_then_L_with_left"][0][2]["jump"] and out["space_then_L_with_left"][1][:2] == [[-1.0, 0.0], "power"]
    a = out["ctrl_with_left_held"]
    assert not a[0][2]["dive"] and a[1][0] == [-1.0, 0.0] and a[1][2]["dive"] and not a[2][2]["dive"]
    a = out["ctrl_then_right"]
    assert not a[0][2]["dive"] and a[1][0] == [1.0, 0.0] and a[1][2]["dive"] and not a[2][2]["dive"]
    a = out["ctrl_alone"]
    assert [x[2]["dive"] for x in a] == [False, False, True, False] and a[2][0] == [0.0, 0.0]
    a = out["ctrl_S_is_a_dive_backwards"]
    assert a[0][0] == [0.0, -1.0] and a[0][2]["dive"]
    a = out["I_with_A"]
    assert a[1][0] == [-1.0, 0.0] and a[1][2]["dive"]


if __name__ == "__main__":
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    res = {}
    tic = time.perf_counter()
    test_inverse_impact(res)
    test_hitting_speed(res)
    test_director(res)
    test_ai_match(res)
    test_recorded_match(res, "right", "lin_dan", "easy", 5)
    test_recorded_match(res, "left", "momota", "easy", 5)
    test_pad_match(res, "axelsen", "right", True)
    test_pad_match(res, "lin_dan", "left", False)
    test_pad_idle(res)
    test_stamina(res)
    test_jump(res)
    test_dive(res)
    test_cards(res)
    test_drills(res)
    test_campaign(res)
    test_pad_keys(res)
    test_app_flow(res)
    if not quick:
        for handed, opp, diff in (("right", "lin_dan", "easy"), ("left", "lee_chong_wei", "easy"), ("right", "axelsen", "normal"),
                                  ("right", "taufik", "pro")):
            test_recorded_match(res, handed, opp, diff, 21)
    res["seconds"] = round(time.perf_counter() - tic)
    res["difficulty"] = DIFFICULTY
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "game_tests.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("all game tests passed")

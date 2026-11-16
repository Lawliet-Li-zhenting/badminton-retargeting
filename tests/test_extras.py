"""Checks of what sits around the match: the learned tactics (brain.py), the wardrobe (every item builds and
the default character is unchanged), the ladder, coins and the verdict, and the save file.
Prints the numbers used in the report.

    python tests/test_extras.py          # about 4 minutes
    python tests/test_extras.py wardrobe quips      # only the tests with these words in their names
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import brain as B  # noqa: E402
from badrt import modes, wardrobe as W  # noqa: E402
from badrt import ui as U  # noqa: E402
from badrt.app import App, Settings  # noqa: E402
from badrt.arcade import REACT  # noqa: E402
from badrt.game import Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.styles import PLAYER_STYLE, ROSTER, SITUATIONS, get_style  # noqa: E402

SEGS = None


def segs():
    global SEGS
    if SEGS is None:
        SEGS = load_library(ROOT / "assets" / "stroke_library.npz")
    return SEGS


def test_brain_tables(res):
    """State indexing, the prior, the value estimate with little data, the policy, the bookkeeping of a rally."""
    seen = set()
    for sit in SITUATIONS:
        for ox, oy in ((1.5, 1.0), (0.0, 3.5), (-1.5, 6.0), (1.5, 6.0), (-1.5, 1.0), (0.0, 1.0), (0.0, 6.0), (1.5, 3.5), (-1.5, 3.5)):
            for stretch in (0.2, 0.95):
                s, c = B.state(sit, np.array([1.0, 4.0, 2.0]), np.array([ox, oy]), stretch)
                assert 0 <= s < B.NS and c == 1.0
                seen.add(s)
                s2, c2 = B.state(sit, np.array([-1.0, 4.0, 2.0]), np.array([-ox, oy]), stretch)
                assert s2 == s and c2 == -1.0                    # mirrored court, same state
    assert len(seen) == B.NS == 90
    st = get_style("lin_dan")
    for sit in SITUATIONS:
        for s in range(SITUATIONS.index(sit) * B.PER_SIT, (SITUATIONS.index(sit) + 1) * B.PER_SIT):
            pri = B.prior(st, sit, s)
            assert abs(pri.sum() - 1) < 1e-9 and (pri[len(B.ACTIONS[sit]):] == 0).all() and (pri[:len(B.ACTIONS[sit])] > 0).all()
    # no data: the policy is the prior
    b = B.Brain()
    pri = B.prior(st, "rear", 20)
    assert np.allclose(b.probs(20, pri, 4.0), pri)
    # one action wins, another loses, in one state: the policy leans, a neighbouring state of the same situation follows
    # weakly, a state of another situation does not move
    b.add([(20, 0, 1.0)] * 30 + [(20, 1, -1.0)] * 30)
    q = b.q(20)
    p = b.probs(20, pri, 4.0)
    assert q[0] > 0.7 and q[1] < -0.7 and p[0] / pri[0] > 5 * p[1] / pri[1]
    q_n = b.q(22)
    assert 0 < q_n[0] - q_n[2] < q[0] - q[2] and q_n[1] < q_n[2]
    assert np.allclose(b.probs(60, B.prior(st, "net", 60), 4.0), B.prior(st, "net", 60))
    # an untried action in a state that is mostly lost does not look better than it is
    b2 = B.Brain()
    b2.add([(21, 0, -1.0)] * 40)
    assert b2.q(21)[3] < -0.5
    # a rally: discounted returns, added to the personal brain when learning; a void rally leaves no trace
    me = B.Brain()
    m = B.Mind(b, beta=3.0, personal=me, learn=True)
    rng = np.random.default_rng(0)
    order = m.order(20, pri, rng)
    assert len(order) == 4 == len(set(order)) and all(pri[a] > 0 for a in order)
    m.took(20, 0), m.took(21, 5), m.took(20, 2)
    m.end_rally(True)
    assert [round(g, 3) for _, _, g in m.samples] == [0.81, 0.9, 1.0] and me.N.sum() == 3 and m.rally == []
    m.took(20, 1)
    m.end_rally(False, void=True)
    assert len(m.samples) == 3 and me.N.sum() == 3
    # what has happened against this player counts several times over
    me2 = B.Brain()
    me2.add([(20, 1, 1.0)] * 30)
    assert b.probs(20, pri, 4.0, extra=me2)[1] > 10 * b.probs(20, pri, 4.0)[1]
    back = B.Brain.from_json(json.loads(json.dumps(me2.to_json())))
    assert np.array_equal(back.N, me2.N) and np.array_equal(back.S, me2.S)
    gens = B.generations()
    res["brain"] = dict(states=B.NS, actions_max=B.NA, generations=len(gens),
                        samples_last=None if not gens else int(round(gens[-1].N.sum())))
    if gens:
        assert gens[0].N.sum() == 0 and gens[-1].N.sum() > 0 and [g.gen for g in gens] == list(range(len(gens)))


def test_brain_match(res, points=7):
    """A match in which the far player uses the last generation: its choices are logged and credited; and with
    no Mind the match is exactly the match it was before (same seed -> same events)."""
    def play(mind):
        m = Match(segs(), get_style("axelsen"), player="ai", player_style=get_style("momota"), points_to_win=points,
                  seed=3, arena=False)
        m.opp.mind = mind
        k = 0
        while m.state != "over" and k < 60 * 600:
            m.update(k / 60.0)
            k += 1
        return m

    a, b = play(None), play(None)
    assert a.score == b.score and len(a.events) == len(b.events)
    mind = B.Mind(B.generation(1 << 20) or B.Brain(), beta=4.0)
    c = play(mind)
    strokes = sum(1 for e in c.events if e.get("who") == "o" and "shot" in e and not e.get("serve"))
    res["brain_match"] = dict(score_plain=a.score, score_brain=c.score, choices=len(mind.samples), rallies=mind.rallies,
                              opponent_strokes=strokes)
    assert c.state == "over" and mind.rallies >= points and len(mind.samples) > 0 and mind.rally == []
    assert all(0 <= s < B.NS and 0 <= a_ < B.NA and -1 <= g <= 1 for s, a_, g in mind.samples)


def _model(outfit, handed="right"):
    st = replace(PLAYER_STYLE, outfit=outfit, handed=handed)
    return U._figure(st, (160, 200))


def test_wardrobe(res):
    """Every item of every slot builds into a model that compiles (right- and left-handed for what is held);
    nothing that is worn changes the skeleton; and without an outfit the character is what it was."""
    m0, d0 = U._figure(PLAYER_STYLE, (160, 200))
    n = 0
    most = (None, 0)
    for slot, title, table in W.SLOTS:
        for key, it in table.items():
            assert len(it) >= 4 and isinstance(it[0], str) and it[1] >= 0, (slot, key)
            for handed in (("right", "left") if slot in ("racket", "suit", "back") else ("right",)):
                m, d = _model({slot: key}, handed)
                n += 1
                assert m.nq == m0.nq and m.nbody == m0.nbody and m.njnt == m0.njnt, (slot, key)
                if m.ngeom - m0.ngeom > most[1]:
                    most = (f"{slot}:{key}", m.ngeom - m0.ngeom)
                if handed == "right" and slot != "body":         # the same joints in the same places
                    assert np.allclose(m.jnt_pos, m0.jnt_pos) and np.allclose(m.body_pos, m0.body_pos), (slot, key)
    full = dict(suit="astro", head="astro", face="shades", back="jet", racket="fish", body="mega", skin="green", hair="h5")
    m, d = _model(full)
    assert m.nq == m0.nq
    # the racket's string bed (what the physics uses) is where it was, whatever is held
    sid0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    if sid0 >= 0:
        assert sid >= 0 and np.allclose(m.site_pos[sid], m0.site_pos[sid0])
    assert W.clean({"suit": "nonsense", "bogus": 1, "head": "crown"}) == dict(W.DEFAULT, head="crown")
    assert W.score(W.DEFAULT) == (0, "路人甲") and W.score(full)[0] > 1400
    assert all(set(it[3]) <= set(REACT) for it in W.QUIPS.values())
    items = sum(len(t) for _, _, t in W.SLOTS)
    res["wardrobe"] = dict(slots=len(W.SLOTS), items=items, models_built=n, most_pieces=most,
                           priced=sum(1 for _, _, t in W.SLOTS for it in t.values() if it[1] > 0),
                           total_price=sum(it[1] for _, _, t in W.SLOTS for it in t.values()))


def test_default_look_unchanged(res):
    """What is worn is only looked at: the same opponent, the same seed and the same scripted thumbs play the
    same match event for event, whether the two characters are dressed (suits, things on the back, a pan instead
    of a racket, a different build) or not."""
    base = get_style("chen_long")
    m = Match(segs(), base, player="pad", points_to_win=5, seed=1, arena=False)
    worn = Match(segs(), replace(base, outfit=dict(modes.rung_style(4).outfit)), player="pad",
                 player_style=replace(PLAYER_STYLE, outfit=dict(W.DEFAULT, suit="bee", back="jet", racket="pan", body="mega",
                                                                head="crown")),
                 points_to_win=5, seed=1, arena=False)
    assert worn.m.nq == m.m.nq and worn.m.ngeom > m.m.ngeom
    from badrt.opponent import PadBot
    out = []
    for mm in (m, worn):
        mm.pad_bot = PadBot(np.random.default_rng(8))
        k = 0
        while mm.state != "over" and k < 60 * 400:
            mm.update(k / 60.0)
            k += 1
        out.append((list(mm.score), len(mm.events), round(float(mm.events[-1]["t"]), 3),
                    [(tuple(sorted(e)), round(float(e["t"]), 4)) for e in mm.events]))
    res["worn_vs_plain_match"] = dict(plain=out[0][:3], worn=out[1][:3], geoms=[int(m.m.ngeom), int(worn.m.ngeom)],
                                      identical=out[0] == out[1])
    assert out[0][0] != [0, 0]
    assert out[0] == out[1], (out[0][:3], out[1][:3])


def test_ladder_and_save(res):
    """The ladder is playable data; coins, purchases, the outfit and the personal brain round-trip through the
    save file; a rung played through the app pays coins, is written down and has taught the opponent something."""
    assert len(modes.LADDER) == 10 and len({r["key"] for r in modes.LADDER}) == 10
    gens = [r["gen"] for r in modes.LADDER]
    assert gens == sorted(gens) and gens[0] == 0.0 and gens[-1] == 1.0
    for i, r in enumerate(modes.LADDER):
        st = modes.rung_style(i, dict(W.DEFAULT, suit="bee"), "left")
        assert all(k in W.CATALOG and v in W.CATALOG[k] for k, v in st.outfit.items()), r["key"]
        assert set(st.lines) <= set(REACT) and all(st.lines.values()), r["key"]
        assert r["diff"] in ("easy", "normal", "pro", "legend") and r["points"] in (11, 15, 21)
        b = modes.rung_brain(i)
        assert b is None or 0 <= b.gen < len(B.generations())
    mirror = modes.rung_style(9, dict(W.DEFAULT, suit="bee"), "left")
    assert mirror.outfit["suit"] == "bee" and mirror.handed == "left"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "save" / "progress.json"
        pr = modes.Progress(path)
        assert pr.coins == modes.START_COINS and pr.outfit == W.DEFAULT and not pr.owned
        assert pr.has("suit", "tank") and not pr.has("suit", "gold")
        assert not pr.buy("suit", "gold") and pr.coins == modes.START_COINS          # too dear
        assert pr.buy("head", "crown") and pr.coins == modes.START_COINS - 400 and pr.has("head", "crown")
        assert pr.buy("head", "crown") and pr.coins == modes.START_COINS - 400       # not paid for twice
        pr.wear("head", "crown")
        pr.personal.add([(3, 2, 0.9), (3, 2, -1.0)])
        pr.earn(25)
        again = modes.Progress(path)
        assert again.coins == modes.START_COINS - 375 and again.outfit["head"] == "crown" and again.owned == ["head:crown"]
        assert again.personal.N[3, 2] == 2 and abs(again.personal.S[3, 2] + 0.1) < 1e-6 and again.matches == 1
        assert again.rung_unlocked(0) and not again.rung_unlocked(1) and again.next_rung() == 0
        coins0, learned0 = again.coins, again.personal.N.sum()
        app = App(size=(640, 360), settings=Settings(quality=0), headless=True, render=False, start="match", mode="ladder",
                  stage=0, pad_bot=True, save=path, seed=0, log=Path(tmp) / "log.json")
        r = app.run()[0]
        after = modes.Progress(path)
        paid = sum(n for _, n in r["coins"])
        res["ladder_rung_1"] = dict(opponent=r["opponent"], score=r["score"], stars=r["stars"], coins=r["coins"], verdict=r["verdict"],
                                    brain=r["brain"], learned=int(after.personal.N.sum() - learned0),
                                    minutes=round(r["game_seconds"] / 60, 1))
        assert r["mode"] == "ladder" and r["rung"] == 1 and r["opponent"] == "ladder_uncle" and r["outfit"]["head"] == "crown"
        assert after.coins == coins0 + paid and paid >= 2 * r["score"][0]
        assert after.ladder[0] == r["stars"] and after.rung_unlocked(1) == (r["stars"] > 0)
        assert after.personal.N.sum() - learned0 == r["brain"]["choices"] > 0
        assert isinstance(r["verdict"][0], str) and r["verdict"][0]
        if r["stars"]:
            assert ("首次击败", modes.LADDER[0]["coins"]) in [tuple(c) for c in r["coins"]]
        path.write_text('{"coins": "many", "outfit": 3}', encoding="utf-8")
        broken = modes.Progress(path)
        assert broken.coins == modes.START_COINS and broken.outfit == W.DEFAULT


def test_quips(res):
    """The player's catch phrases come up in the bubbles, the opponent's own lines replace the stock ones, and
    neither changes the match (the phrases have their own random numbers)."""
    from badrt.opponent import PadBot

    def play(me, opp):
        m = Match(segs(), opp, player="pad", player_style=me, points_to_win=7, seed=5, arena=False, pad_bot=PadBot(np.random.default_rng(2)))
        said = {"p": set(), "o": set()}
        k = 0
        while m.state != "over" and k < 60 * 500:
            m.update(k / 60.0)
            k += 1
            for who, (txt, _, _) in m.bubbles.items():
                said[who].add(txt)
        return m, said

    base = get_style("ginting")
    m0, said0 = play(PLAYER_STYLE, base)
    m1, said1 = play(replace(PLAYER_STYLE, lines=W.quips(dict(quip="meme"))), replace(base, lines=modes.LADDER[5]["style"].lines))
    mine = {t for v in W.QUIPS["meme"][3].values() for t in v}
    theirs = {t for v in modes.LADDER[5]["style"].lines.values() for t in v}
    res["quips"] = dict(player=sorted(said1["p"]), opponent=sorted(said1["o"] & theirs), score=m1.score)
    assert not said0["p"] and said1["p"] and said1["p"] <= mine
    assert said1["o"] & theirs
    assert m0.score == m1.score and len(m0.events) == len(m1.events)


class _Played:
    """What modes.verdict reads of a finished match, with the numbers given."""

    def __init__(self, score, points=11, forfeit=None, taunts=0, cards=0, dives=0, speed=200.0, hits=20, shots=None,
                 errors=0, winners=3, rally_max=8):
        self.score, self.points_to_win, self.forfeit, self.taunts = list(score), points, forfeit, taunts
        self._st = dict(p=dict(shots=shots or dict(clear=hits), max_speed_kmh=speed, hits=hits, errors=errors, winners=winners),
                        o={}, dives=dict(p=dives), cards=[dict(card="yellow")] * cards, rally_max=rally_max)

    def stats(self):
        return self._st

    def won(self):
        return self.forfeit != "p" and self.score[0] > self.score[1]


def test_verdict(res):
    """Each of the sixteen rules of the verdict is reached by a match with the numbers it asks for (in the order
    in which they are tried); a real match in which nothing is pressed gets the one for a heavy defeat."""
    cases = [("法外狂徒", _Played((3, 11), forfeit="p")), ("时间管理大师", _Played((11, 5), cards=1)),
             ("嘴强王者", _Played((5, 11), taunts=3)), ("下手太重", _Played((11, 0))),
             ("地板清洁工", _Played((11, 6), dives=5)), ("人形炮台", _Played((11, 6), speed=331.0)),
             ("杀球上瘾", _Played((11, 6), shots=dict(smash=6, half_smash=2, clear=12))),
             ("猥琐流宗师", _Played((11, 6), shots=dict(drop=6, net=4, clear=10))),
             ("慈善家", _Played((6, 11), errors=8, winners=3)), ("马拉松选手", _Played((11, 6), rally_max=25)),
             ("大心脏", _Played((13, 11))), ("砍瓜切菜", _Played((11, 3))), ("稳稳的幸福", _Played((11, 6))),
             ("重在参与", _Played((2, 11))), ("就差一点", _Played((9, 11))), ("越挫越勇", _Played((6, 11)))]
    for want, played in cases:
        name, remark = modes.verdict(played)
        assert name == want and remark, (want, name)
    assert len({w for w, _ in cases}) == 16
    res["verdict_rules"] = len(cases)
    m = Match(segs(), get_style("taufik"), player="pad", points_to_win=5, seed=1, arena=False)
    k = 0
    while m.state != "over" and k < 60 * 300:
        m.update(k / 60.0)
        k += 1
    v = modes.verdict(m)
    res["verdict_idle"] = dict(score=m.score, verdict=v, coins=modes.reward("quick", m))
    assert m.score[0] == 0 and v[0] == "重在参与" and modes.reward("quick", m) == []
    m.forfeit = "p"
    assert modes.verdict(m)[0] == "法外狂徒"


if __name__ == "__main__":
    res = {}
    tic = time.perf_counter()
    tests = [test_brain_tables, test_brain_match, test_wardrobe, test_default_look_unchanged, test_ladder_and_save,
             test_quips, test_verdict]
    only = sys.argv[1:]
    for t in tests:
        if not only or any(w in t.__name__ for w in only):
            t0 = time.perf_counter()
            t(res)
            print(f"{t.__name__}: ok ({time.perf_counter() - t0:.0f} s)", flush=True)
    res["seconds"] = round(time.perf_counter() - tic)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    if not only:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "extras_tests.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print("all extras tests passed")

"""Game modes on top of a single match: the training drills, the campaign, the ladder of odd characters
(who play with the tactics learned in self-play, brain.py), coins, the verdict after a match, and the save file.

A drill is data for game.Match(drill=...): where the feeder and the player stand, what is fed (a cycle of
feeds for opponent.AIPlayer.plan_serve(drill=...)) and what counts as a good repetition.

    feeds     dict(shot, speed [m/s], high (lob branch), x (lo, hi) [m, world], y (lo, hi) depth in the
              player's half [m from the net], over: hit from above the head, margin: net clearance)
    shots     the reply has to be one of these strokes (None: any)
    zones     the reply has to land in this rectangle of the far half (x0, x1, y0, y1); a cycle
    min_kmh   ... and leave the strings at least this fast
    rally     instead of all that: the rally has to reach this many strokes (the feeder plays on)
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from . import brain, wardrobe
from .court import LENGTH_HALF, SHORT_SERVICE, WIDTH_SINGLES_HALF
from .styles import ROSTER, get_style

W, L = WIDTH_SINGLES_HALF, LENGTH_HALF
FULL = (-W, W, 0.0, L)

DRILLS = [
    dict(key="clear", name="高远球", goal="把球打到对方后场（底线前 2 米以内）",
         tip="球在头顶时按 K（高）。按得早一点，击球点才高",
         feeder=(0.0, 3.2), player=(0.0, -4.2), reps=10, shots=("clear",), zones=[(-W, W, L - 2.0, L)],
         feeds=[dict(shot="lift", speed=38.0, high=True, x=(-1.4, 1.4), y=(5.0, 6.1))]),
    dict(key="smash", name="杀球 · 指哪打哪", goal="杀到亮起的半边：按住 ← 或 → 再按 L",
         tip="空格 + L 是跳杀（更快，但更费体力）。方向键决定杀左边还是右边",
         feeder=(0.0, 3.0), player=(0.0, -4.2), reps=10, shots=("smash", "half_smash", "kill"),
         zones=[(-W, 0.0, 0.0, L), (0.0, W, 0.0, L)],
         feeds=[dict(shot="lift", speed=34.0, high=True, x=(-1.0, 1.0), y=(4.3, 5.3))]),
    dict(key="drop", name="吊球", goal="从后场把球吊到对方发球线以内",
         tip="球在头顶时按 J（轻）。同时按 ↓ 吊得更贴网",
         feeder=(0.0, 3.2), player=(0.0, -4.2), reps=10, shots=("drop", "slice"), zones=[(-W, W, 0.0, 2.6)],
         feeds=[dict(shot="lift", speed=38.0, high=True, x=(-1.4, 1.4), y=(4.8, 6.0))]),
    dict(key="net", name="网前搓放", goal="把网前球搓回对方发球线以内",
         tip="球落到腰以下再按 J（轻）；K（高）是挑后场，这里不算",
         feeder=(0.0, 1.5), player=(0.0, -2.4), reps=10, shots=("net", "block"), zones=[(-W, W, 0.0, SHORT_SERVICE)],
         feeds=[dict(shot="net", speed=7.0, high=False, x=(-1.5, 1.5), y=(0.6, 1.4), margin=0.15)]),
    dict(key="defence", name="接杀", goal="把杀球接回界内就算成功",
         tip="K 挑高、J 挡网；够不着时 Ctrl + 方向键鱼跃",
         feeder=(0.0, 3.6), player=(0.0, -4.0), reps=10, shots=None, zones=None,
         feeds=[dict(shot="half_smash", speed=40.0, high=False, x=(-1.3, 1.3), y=(3.6, 5.0), over=True, margin=0.15),
                dict(shot="smash", speed=52.0, high=False, x=(-1.7, 1.7), y=(3.8, 5.2), over=True, margin=0.15)]),
    dict(key="corners", name="四角跑动", goal="前后左右四个角轮流来球，全部回到界内",
         tip="够不着的球用 Ctrl + 方向键鱼跃（训练里不计体力，放心扑）",
         feeder=(0.0, 2.8), player=(0.0, -3.4), reps=12, shots=None, zones=None,
         feeds=[dict(shot="drop", speed=10.0, high=False, x=(-1.9, -1.5), y=(1.0, 1.6), margin=0.25),
                dict(shot="lift", speed=38.0, high=True, x=(1.5, 1.9), y=(5.4, 6.2)),
                dict(shot="drop", speed=10.0, high=False, x=(1.5, 1.9), y=(1.0, 1.6), margin=0.25),
                dict(shot="lift", speed=38.0, high=True, x=(-1.9, -1.5), y=(5.4, 6.2))]),
    dict(key="rally", name="多拍对抗", goal="和教练打到 10 拍以上（这一项教练会回球）",
         tip="不求打死，先求不失误：高球按 K，低球按 K 或 J",
         feeder=(0.0, 3.0), player=(0.0, -3.6), reps=5, rally=10, passive=False, shots=None, zones=None,
         feeds=[dict(shot="lift", speed=37.0, high=True, x=(-1.0, 1.0), y=(5.2, 6.1))]),
]
DRILL_KEYS = [d["key"] for d in DRILLS]
COACH = replace(get_style("chen_long"), key="coach", name="教练", name_en="Coach", title="多球训练",
                shirt=(0.16, 0.18, 0.22, 1), shorts=(0.16, 0.18, 0.22, 1), accent=(0.98, 0.85, 0.15, 1))


def drill(key):
    return DRILLS[DRILL_KEYS.index(key)]


# the ladder: (opponent, difficulty, points). Easier styles first; the last two are the new top tier
CAMPAIGN = [
    ("taufik", "easy", 11), ("tai_tzu_ying", "easy", 11), ("ginting", "easy", 11),
    ("marin", "normal", 11), ("chen_long", "normal", 11), ("lee_zii_jia", "normal", 15),
    ("momota", "normal", 15), ("shi_yuqi", "pro", 15), ("axelsen", "pro", 15),
    ("lee_chong_wei", "pro", 21), ("an_se_young", "legend", 21), ("lin_dan", "legend", 21),
]
_KEYS = [s.key for s in ROSTER]


def stage(i):
    """(roster index, difficulty key, points) of campaign stage i (0-based)."""
    key, diff, pts = CAMPAIGN[i]
    return _KEYS.index(key), diff, pts


def stars(match):
    """0 lost; 1 won; 2 won without a card; 3 won without a card and gave away at most half as many points."""
    if not match.won():
        return 0
    cards = len(match.umpire.cards) if match.umpire is not None else 0
    a, b = match.score
    return 1 + (cards == 0) + (cards == 0 and 2 * b <= a)


# ---------------------------------------------------------------------------- the ladder
# Odd characters, each a costume (wardrobe.py) on one of the roster's playing styles, with its own lines.
# gen: which self-play generation of brain.py plays its tactics, as a fraction of what has been trained
# (0 = the style's own menu); beta: how strictly it follows what it learned. The held-out evaluation
# (scripts/eval_brain.py) found beta to matter more than the generation -- the last generation takes 60 / 66 / 73 /
# 74 % of the points from the hand-written tactics at beta 2 / 4 / 8 / 16 -- so the top two rungs use 8 and 16.
def _rung(key, base, name, en, title, blurb, outfit, lines, diff, points, gen, beta, coins):
    st = replace(get_style(base), key="ladder_" + key, name=name, name_en=en, title=title, blurb=blurb, outfit=outfit, lines=lines)
    return dict(key=key, style=st, diff=diff, points=points, gen=gen, beta=beta, coins=coins)


LADDER = [
    _rung("uncle", "taufik", "老头乐大爷", "Park Uncle", "公园球王", "在公园打了四十年，只用三成力。背心是战袍，蒲扇是球拍。别被外表骗了，他的吊球比你的人生规划还准。",
          dict(suit="tank", head="straw", racket="fan", hair="h3"),
          dict(won=("年轻人，还得练", "就这？"), won_big=("我像你这么大的时候……", "这球叫太极"), won_long=("腰不行了，但球还行",),
               lost_error=("老花眼，没看清",), lost_winner=("后生可畏",), tired=("歇会儿，喝口茶",)), "easy", 11, 0.0, 2.0, 60),
    _rung("auntie", "tai_tzu_ying", "广场舞大妈", "Square Auntie", "C 位领舞", "音响一开，全场都是她的节奏。手腕花活多得像在扇扇子，你永远猜不到她下一拍往哪儿打。",
          dict(suit="square", head="afro", face="blush", racket="fan"),
          dict(won=("跟上节奏啊小伙子", "最炫民族风！"), won_big=("这叫凤凰展翅", "音乐响起来"), won_long=("跳了三小时都不累，这算啥",),
               lost_error=("刚才那拍不算，没踩上点",), lost_winner=("哎呦，可以啊",), tired=("等等，换首歌",)), "easy", 11, 0.15, 2.5, 80),
    _rung("pupil", "ginting", "课代表", "Class Rep", "作业写完了", "书包里装着五年高考三年模拟。打球是他唯一的课间活动，所以每一分都当成期末考。",
          dict(suit="school", back="bag", face="monocle"),
          dict(won=("这题我会", "送分题"), won_big=("标准答案", "老师讲过"), won_long=("这道大题值十五分",),
               lost_error=("粗心了，扣两分",), lost_winner=("超纲了吧",), tired=("能不能课间休息",)), "normal", 11, 0.3, 3.0, 110),
    _rung("rider", "lee_zii_jia", "外卖小哥", "Rider", "准时达", "全场跑动距离第一名，因为超时要扣钱。杀球又快又直，口头禅是“您的外卖到了”。",
          dict(suit="rider", head="rider", back="box"),
          dict(won=("您的杀球已送达", "记得五星好评"), won_big=("准时达！", "这单不超时"), won_long=("这单跑了三公里",),
               lost_error=("导航带错路了",), lost_winner=("差评别给啊",), tired=("电瓶车没电了",)), "normal", 15, 0.45, 3.0, 140),
    _rung("panda", "chen_long", "滚滚", "Gun Gun", "国宝 · 一级保护", "看着憨，其实是防守大师：你杀十拍，它接十拍，然后坐下吃竹子。打它算不算违法，裁判也说不清。",
          dict(suit="panda", head="panda", body="bighead", racket="leek"),
          dict(won=("嗯嘛", "竹子真香"), won_big=("功夫熊猫！", "嗷——"), won_long=("（翻了个身）",),
               lost_error=("（打了个滚）",), lost_winner=("你敢打国宝？",), tired=("困了，要睡觉",)), "normal", 15, 0.6, 3.5, 180),
    _rung("boss", "axelsen", "霸道总裁", "The CEO", "身价过亿", "一米九四，西装革履。输一分就要收购你的公司。杀球的时候会说“天凉了”。",
          dict(suit="boss", face="shades", racket="brick"),
          dict(won=("天凉了，该让你破产了", "女人，你在玩火"), won_big=("这个球场，我承包了", "很好，你引起了我的注意"), won_long=("给他加个鸡腿",),
               lost_error=("这个季度的 KPI……",), lost_winner=("有意思，收购他",), tired=("让我的助理替我打",)), "pro", 15, 0.7, 4.0, 240),
    _rung("bones", "momota", "骨感先生", "Mr. Bones", "减肥成功", "体脂率 0%。不出汗，不抽筋，不喘气——因为这些他都没有。多拍相持能把你耗到也只剩骨头。",
          dict(suit="bones", head="bald", back="sword"),
          dict(won=("我没有心，所以不会手软", "骨头硬"), won_big=("入骨三分", "这球透心凉"), won_long=("我没有肺，不会喘",),
               lost_error=("手骨滑了",), lost_winner=("刻骨铭心",), tired=("散架了，等我拼一下",)), "pro", 21, 0.8, 4.0, 300),
    _rung("astro", "lee_chong_wei", "太空人", "Astronaut", "来自空间站", "在失重环境里练了三年步法，回到地球快得不讲道理。头盔里有空调，所以从来不慌。",
          dict(suit="astro", head="astro", back="jet"),
          dict(won=("休斯顿，我们得分了", "个人的一小步"), won_big=("发射成功", "三、二、一，点火"), won_long=("这一分绕了地球一圈",),
               lost_error=("重力还没适应",), lost_winner=("地球人不简单",), tired=("氧气不足",)), "pro", 21, 0.9, 4.5, 380),
    _rung("robot", "an_se_young", "阿尔法羽", "AlphaShuttle", "自对弈出身", "它的战术不是人教的：是自己跟自己打了八百多局打出来的（最新一代）。它还在边打边记你的习惯。",
          dict(suit="robot", head="robot", racket="glow"),
          dict(won=("哔——胜率上升", "已记录你的弱点"), won_big=("计算结果：你接不到", "最优解"), won_long=("电量充足，再来一百拍",),
               lost_error=("哔——出现误差",), lost_winner=("数据异常，正在学习",), tired=("电量低，请充电",)), "legend", 21, 1.0, 8.0, 500),
    _rung("mirror", "lin_dan", "另一个你", "The Other You", "你的衣服真好看", "穿得和你一模一样，还学会了你最常吃亏的球路。打败自己，听起来很励志，打起来很崩溃。",
          dict(), dict(won=("我就是你", "你打不过你自己"), won_big=("这招是跟你学的", "熟悉吗？"), won_long=("我比你更懂你",),
                       lost_error=("这也是跟你学的",), lost_winner=("不愧是我",), tired=("你累我也累",)), "legend", 21, 1.0, 16.0, 800),
]


def rung_brain(i):
    """The Brain a ladder opponent plays with (None when nothing has been trained: the style's own tactics)."""
    gens = brain.generations()
    return gens[int(round(LADDER[i]["gen"] * (len(gens) - 1)))] if gens else None


def rung_style(i, outfit=None, handed=None):
    """Style of ladder opponent i; the last one wears what the player wears."""
    st = LADDER[i]["style"]
    if LADDER[i]["key"] == "mirror":
        st = replace(st, outfit=dict(wardrobe.clean(outfit)), handed=handed or st.handed)
    return st


# ---------------------------------------------------------------------------- coins and the verdict
WIN_COINS = {"easy": 30, "normal": 50, "pro": 90, "legend": 160}


def reward(mode, match, stage=0, first=False):
    """Coins for a finished match: [(what for, coins)]."""
    st = match.stats()
    if mode == "training":
        return [("训练到位的球", 3 * match.score[0])]
    out = [("得分", 2 * match.score[0])]
    if match.won():
        if mode == "campaign":
            out.append(("闯关获胜", 40 + 10 * stage))
        elif mode == "ladder":
            out.append(("天梯获胜", LADDER[stage]["coins"]))
            if first:
                out.append(("首次击败", LADDER[stage]["coins"]))
        else:
            out.append(("获胜", WIN_COINS.get(match.difficulty, 30)))
    if st["p"]["max_speed_kmh"] >= 300:
        out.append(("杀球破 300 km/h", 15))
    if st.get("rally_max", 0) >= 20:
        out.append(("二十拍以上的回合", 15))
    return [(a, int(b)) for a, b in out if b > 0]


def verdict(match):
    """(nickname, remark) for the result screen: what kind of match the player made of it."""
    st = match.stats()
    p, o = st["p"], st["o"]
    a, b = match.score
    won = match.won()
    sm = p["shots"].get("smash", 0) + p["shots"].get("half_smash", 0) + p["shots"].get("kill", 0)
    soft = p["shots"].get("drop", 0) + p["shots"].get("net", 0) + p["shots"].get("slice", 0) + p["shots"].get("block", 0)
    dives = st.get("dives", {}).get("p", 0)
    cards = len(st.get("cards") or [])
    if match.forfeit == "p":
        return "法外狂徒", "黑牌都吃到了。裁判建议你改行打拳击"
    if won and cards:
        return "时间管理大师", "拖到裁判掏牌还能赢，对手的心态已经崩了"
    if not won and getattr(match, "taunts", 0) >= 3:
        return "嘴强王者", f"嘚瑟了 {match.taunts} 次，然后输了。全身上下嘴最硬"
    if won and b == 0:
        return "下手太重", "一分没给。对手已经在打电话报警了"
    if dives >= 5:
        return "地板清洁工", f"鱼跃 {dives} 次，场馆今天不用拖地了"
    if p["max_speed_kmh"] >= 330:
        return "人形炮台", f"{p['max_speed_kmh']:.0f} km/h。羽毛球表示想换个工作"
    if p["hits"] >= 12 and sm >= 0.4 * p["hits"]:
        return "杀球上瘾", f"{p['hits']} 拍里杀了 {sm} 拍。建议检查一下 L 键还在不在"
    if p["hits"] >= 12 and soft >= 0.5 * p["hits"]:
        return "猥琐流宗师", "轻吊搓放一条龙。对手的膝盖正在哭泣"
    if p["errors"] >= 8 and p["errors"] >= 2 * max(p["winners"], 1):
        return "慈善家", f"主动失误送了 {p['errors']} 分。对手说：谢谢老板"
    if st.get("rally_max", 0) >= 25:
        return "马拉松选手", f"最长一个回合打了 {st['rally_max']} 拍，建议下次带个板凳"
    if won and a - b == 2 and a > match.points_to_win:
        return "大心脏", "加分赛拿下。心率 180，表情很淡定"
    if won and b <= a // 3:
        return "砍瓜切菜", "赢得太轻松，建议去天梯找点苦头吃"
    if won:
        return "稳稳的幸福", "赢了。不多说，低调"
    if a <= b // 3:
        return "重在参与", "比分不重要，重要的是运动量达标了"
    if b - a <= 3:
        return "就差一点", "虽败犹荣。下次一定"
    return "越挫越勇", "输球不输人，换身衣服再战"


START_COINS = 500


class Progress:
    """The save file (a small json): campaign stars, best drill scores, ladder wins, coins, the wardrobe
    (what is owned, what is worn) and what the ladder opponents have learned about this player."""

    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self.stars, self.drills = [0] * len(CAMPAIGN), {}
        self.ladder = [0] * len(LADDER)                # stars per rung
        self.coins = START_COINS
        self.owned = []                                # "slot:key" of what has been bought
        self.outfit = dict(wardrobe.DEFAULT)
        self.personal = brain.Brain()                  # rallies against this player (brain.Mind(personal=...))
        self.matches = 0
        if self.path is not None and self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                st = [int(v) for v in d.get("stars", [])][:len(CAMPAIGN)]
                self.stars = st + [0] * (len(CAMPAIGN) - len(st))
                self.drills = {k: int(v) for k, v in d.get("drills", {}).items()}
                ld = [int(v) for v in d.get("ladder", [])][:len(LADDER)]
                self.ladder = ld + [0] * (len(LADDER) - len(ld))
                self.coins = max(int(d.get("coins", START_COINS)), 0)
                self.owned = [str(v) for v in d.get("owned", [])]
                self.outfit = wardrobe.clean(d.get("outfit"))
                self.personal = brain.Brain.from_json(d.get("personal", {}))
                self.matches = int(d.get("matches", 0))
            except (ValueError, OSError, AttributeError, TypeError):      # an unreadable save is a new game
                pass

    def unlocked(self, i):
        return i == 0 or self.stars[i - 1] > 0

    def rung_unlocked(self, i):
        return i == 0 or self.ladder[i - 1] > 0

    def next_rung(self):
        for i, s in enumerate(self.ladder):
            if s == 0:
                return i
        return len(self.ladder) - 1

    def has(self, slot, key):
        return wardrobe.price(slot, key) == 0 or f"{slot}:{key}" in self.owned

    def buy(self, slot, key):
        """True when the item is (now) owned."""
        if self.has(slot, key):
            return True
        cost = wardrobe.price(slot, key)
        if cost > self.coins:
            return False
        self.coins -= cost
        self.owned.append(f"{slot}:{key}")
        self.save()
        return True

    def wear(self, slot, key):
        self.outfit[slot] = key
        self.save()

    def earn(self, n):
        self.coins += int(n)
        self.matches += 1
        self.save()

    def rung_done(self, i, n):
        """True when this is the first win on rung i."""
        first = n > 0 and self.ladder[i] == 0
        if n > self.ladder[i]:
            self.ladder[i] = n
        self.save()
        return first

    def next_open(self):
        """The first stage that has not been won yet (the last one when all have)."""
        for i, s in enumerate(self.stars):
            if s == 0:
                return i
        return len(self.stars) - 1

    def save(self):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            d = dict(stars=self.stars, drills=self.drills, ladder=self.ladder, coins=self.coins, owned=self.owned,
                     outfit=self.outfit, matches=self.matches, personal=self.personal.to_json())
            self.path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            print("progress not saved:", e)

    def stage_done(self, i, n):
        if n > self.stars[i]:
            self.stars[i] = n
            self.save()

    def drill_done(self, key, ok):
        if ok > self.drills.get(key, -1):
            self.drills[key] = int(ok)
            self.save()

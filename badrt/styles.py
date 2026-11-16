"""Opponent roster: play styles as data.

A Style is everything that distinguishes one opponent from another: body (height, racket hand,
outfit), athletic limits (court speed, reaction, reach, smash speed), consistency, and tactics
(which shot from which situation, where to place it, when to go for the kill).

Situations (the opponent's own half, d = distance from the net, z = contact height):
    kill     z >= 1.9 and d < 3.8     high shuttle in the front/mid court
    rear     z >= 1.9 and d >= 3.8    overhead from the back
    mid      1.0 <= z < 1.9           flat exchanges
    net      z < 1.0 and d < 2.6      at the net
    defence  z < 1.0 and d >= 2.6     digging out smashes and drops

The names are those of real players because the brief asks for them; everything else in the
game refers to the `key`, so a release without name / likeness rights only has to edit this
table (or load another one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# shot -> (launch speed m/s, lob branch, landing depth range |y| in the receiver's half)
SHOTS = {
    "smash": (70.0, False, (3.4, 6.0)),
    "half_smash": (46.0, False, (2.6, 4.4)),
    "clear": (50.0, True, (5.7, 6.45)),
    "drop": (20.0, False, (1.0, 2.4)),
    "slice": (30.0, False, (1.9, 3.1)),
    "drive": (30.0, False, (4.0, 6.2)),
    "push": (17.0, False, (3.6, 5.6)),
    "block": (9.0, False, (0.6, 1.7)),
    "lift": (38.0, True, (5.5, 6.4)),
    "net": (6.5, False, (0.4, 1.3)),
    "kill": (40.0, False, (1.8, 4.2)),
    "serve_high": (37.0, True, (5.9, 6.5)),
    "serve_low": (8.5, False, (2.05, 2.5)),
}
SHOT_CN = {
    "smash": "杀球", "half_smash": "点杀", "clear": "高远球", "drop": "吊球", "slice": "劈吊", "drive": "平抽",
    "push": "推球", "block": "挡网", "lift": "挑球", "net": "搓放", "kill": "扑杀", "serve_high": "发高远球",
    "serve_low": "发网前球", "serve": "发球", "return": "回球", "deflect": "球砸拍", "dive": "鱼跃救球",
}
SITUATIONS = ("kill", "rear", "mid", "net", "defence")


@dataclass
class Style:
    key: str
    name: str                     # display name (Chinese)
    name_en: str
    title: str                    # play style in a few characters
    blurb: str                    # one line for the select screen
    handed: str = "right"
    height_m: float = 1.78
    # athletic limits
    move_speed: float = 5.6       # m/s court speed
    reaction: float = 0.12        # s before moving
    reach: float = 0.9            # m covered by the lunge / arm without moving the feet
    smash_speed: float = 72.0     # m/s shuttle speed off the strings on a full smash
    # consistency: std of the aiming error is error * 4 deg (grows when stretched)
    error: float = 0.20
    judgement: float = 0.85       # probability of leaving a shuttle that is going out
    # tactics
    shots: dict = field(default_factory=dict)        # situation -> {shot: weight}
    corner: float = 0.45          # m kept from the side line when aiming at a corner (smaller = bolder)
    cross: float = 0.5            # probability of aiming away from where the other player stands
    body: float = 0.1             # probability of smashing at the body
    aggression: float = 0.5       # willingness to end the rally rather than keep it going
    serve_low: float = 0.3        # probability of a short serve
    base_depth: float = 3.5       # where they wait, metres from the net
    # looks (rgba)
    shirt: tuple = (0.85, 0.15, 0.15, 1)
    shorts: tuple = (0.95, 0.95, 0.95, 1)
    accent: tuple = (0.1, 0.5, 0.95, 1)
    skin: tuple = (0.93, 0.75, 0.6, 1)
    hair: tuple = (0.08, 0.06, 0.05, 1)
    ratings: dict = field(default_factory=dict)      # 进攻 防守 速度 网前 稳定 (0-100, select screen)
    outfit: dict = field(default_factory=dict)       # wardrobe.py: what is worn on top of the kit colours
    lines: dict = field(default_factory=dict)        # reaction (arcade.REACT key) -> what this player says instead

    @property
    def dims(self):
        """Segment scale factors of the character (reference build is 1.76 m)."""
        f = min(max(self.height_m / 1.76, 0.85), 1.15)
        g = 1.0 + 0.5 * (f - 1.0)
        return dict(thigh=f, shin=f, upperarm=f, forearm=f, torso=f, shoulder=g, hip=g, girth=g)

    @property
    def leg_scale(self):
        f = self.dims["thigh"]
        return (0.03 + 0.44 * f + 0.43 * f + 0.08) / 0.98


def _w(**kw):
    return dict(kw)


ROSTER = [
    Style(
        key="lin_dan", name="林丹", name_en="Lin Dan", title="拉吊突击",
        blurb="左手持拍。四方球调动，等你回球不到位就起跳重杀。",
        handed="left", height_m=1.78, move_speed=5.9, reaction=0.11, reach=0.95, smash_speed=75.0,
        error=0.18, judgement=0.9, corner=0.38, cross=0.7, body=0.1, aggression=0.55, serve_low=0.35,
        shots=dict(kill=_w(smash=0.7, half_smash=0.15, slice=0.15),
                   rear=_w(clear=0.38, drop=0.2, slice=0.2, smash=0.22),
                   mid=_w(drive=0.3, push=0.3, block=0.2, lift=0.2),
                   net=_w(net=0.5, lift=0.35, push=0.15),
                   defence=_w(lift=0.55, block=0.35, drive=0.1)),
        shirt=(0.93, 0.74, 0.1, 1), shorts=(0.75, 0.08, 0.1, 1), accent=(0.85, 0.1, 0.1, 1),
        ratings={"进攻": 95, "防守": 90, "速度": 90, "网前": 88, "稳定": 90}),
    Style(
        key="lee_chong_wei", name="李宗伟", name_en="Lee Chong Wei", title="快攻 · 防守反击",
        blurb="全场最快的脚步，平抽快挡提速，守中带攻。",
        handed="right", height_m=1.72, move_speed=6.4, reaction=0.10, reach=0.9, smash_speed=76.0,
        error=0.18, judgement=0.88, corner=0.42, cross=0.65, body=0.2, aggression=0.6, serve_low=0.45,
        shots=dict(kill=_w(smash=0.75, half_smash=0.2, slice=0.05),
                   rear=_w(smash=0.4, clear=0.25, slice=0.2, drop=0.15),
                   mid=_w(drive=0.6, push=0.25, block=0.15),
                   net=_w(push=0.3, net=0.4, lift=0.3),
                   defence=_w(drive=0.35, block=0.35, lift=0.3)),
        shirt=(0.1, 0.1, 0.12, 1), shorts=(0.96, 0.8, 0.1, 1), accent=(0.96, 0.8, 0.1, 1),
        ratings={"进攻": 93, "防守": 94, "速度": 98, "网前": 85, "稳定": 90}),
    Style(
        key="taufik", name="陶菲克", name_en="Taufik Hidayat", title="技术流 · 反手天王",
        blurb="手腕细腻，网前搓放和假动作劈吊让你原地起不来。",
        handed="right", height_m=1.76, move_speed=5.2, reaction=0.13, reach=0.9, smash_speed=72.0,
        error=0.2, judgement=0.85, corner=0.3, cross=0.6, body=0.05, aggression=0.4, serve_low=0.5,
        shots=dict(kill=_w(smash=0.5, slice=0.3, half_smash=0.2),
                   rear=_w(slice=0.34, drop=0.26, clear=0.25, smash=0.15),
                   mid=_w(block=0.35, push=0.35, drive=0.2, lift=0.1),
                   net=_w(net=0.7, lift=0.2, push=0.1),
                   defence=_w(block=0.5, lift=0.4, drive=0.1)),
        shirt=(0.96, 0.96, 0.96, 1), shorts=(0.8, 0.1, 0.12, 1), accent=(0.8, 0.1, 0.12, 1),
        ratings={"进攻": 86, "防守": 82, "速度": 80, "网前": 97, "稳定": 84}),
    Style(
        key="axelsen", name="安赛龙", name_en="Viktor Axelsen", title="高点下压 · 重杀",
        blurb="1.94 米的击球点，后场一拍接一拍往下砸。",
        handed="right", height_m=1.94, move_speed=5.5, reaction=0.12, reach=1.05, smash_speed=79.0,
        error=0.2, judgement=0.85, corner=0.45, cross=0.55, body=0.25, aggression=0.75, serve_low=0.2,
        shots=dict(kill=_w(smash=0.85, half_smash=0.15),
                   rear=_w(smash=0.55, half_smash=0.15, clear=0.18, slice=0.12),
                   mid=_w(drive=0.45, push=0.3, lift=0.25),
                   net=_w(lift=0.5, net=0.3, push=0.2),
                   defence=_w(lift=0.6, block=0.25, drive=0.15)),
        shirt=(0.78, 0.06, 0.12, 1), shorts=(0.96, 0.96, 0.96, 1), accent=(0.96, 0.96, 0.96, 1),
        hair=(0.55, 0.42, 0.25, 1), skin=(0.96, 0.8, 0.68, 1),
        ratings={"进攻": 98, "防守": 86, "速度": 84, "网前": 80, "稳定": 86}),
    Style(
        key="momota", name="桃田贤斗", name_en="Kento Momota", title="控网 · 滴水不漏",
        blurb="左手持拍。贴网球逼你起高，自己几乎不失误，拖到你先犯错。",
        handed="left", height_m=1.75, move_speed=5.9, reaction=0.11, reach=0.9, smash_speed=67.0,
        error=0.13, judgement=0.93, corner=0.5, cross=0.6, body=0.05, aggression=0.3, serve_low=0.7,
        shots=dict(kill=_w(half_smash=0.45, smash=0.35, slice=0.2),
                   rear=_w(clear=0.35, drop=0.3, slice=0.2, half_smash=0.15),
                   mid=_w(block=0.45, push=0.3, drive=0.15, lift=0.1),
                   net=_w(net=0.75, lift=0.2, push=0.05),
                   defence=_w(block=0.45, lift=0.5, drive=0.05)),
        shirt=(0.1, 0.16, 0.42, 1), shorts=(0.96, 0.96, 0.96, 1), accent=(0.9, 0.15, 0.2, 1),
        ratings={"进攻": 80, "防守": 97, "速度": 90, "网前": 96, "稳定": 98}),
    Style(
        key="chen_long", name="谌龙", name_en="Chen Long", title="控网强攻",
        blurb="先用网前小球拿到主动，再凭身高臂展连续下压。",
        handed="right", height_m=1.87, move_speed=5.5, reaction=0.12, reach=1.0, smash_speed=73.0,
        error=0.16, judgement=0.9, corner=0.45, cross=0.6, body=0.15, aggression=0.55, serve_low=0.6,
        shots=dict(kill=_w(smash=0.75, half_smash=0.25),
                   rear=_w(smash=0.35, clear=0.3, slice=0.2, drop=0.15),
                   mid=_w(push=0.4, drive=0.3, block=0.3),
                   net=_w(net=0.65, push=0.2, lift=0.15),
                   defence=_w(lift=0.5, block=0.4, drive=0.1)),
        shirt=(0.05, 0.5, 0.5, 1), shorts=(0.08, 0.08, 0.1, 1), accent=(0.95, 0.8, 0.2, 1),
        ratings={"进攻": 90, "防守": 95, "速度": 84, "网前": 90, "稳定": 94}),
    # (appended: the indices of the six above are used by tests and recorded matches)
    Style(
        key="shi_yuqi", name="石宇奇", name_en="Shi Yuqi", title="全面 · 变速突击",
        blurb="没有明显短板，慢节奏里突然加速，一拍劈杀结束。",
        handed="right", height_m=1.84, move_speed=5.8, reaction=0.11, reach=0.98, smash_speed=75.0,
        error=0.16, judgement=0.9, corner=0.4, cross=0.65, body=0.12, aggression=0.55, serve_low=0.55,
        shots=dict(kill=_w(smash=0.7, half_smash=0.15, slice=0.15),
                   rear=_w(slice=0.28, smash=0.3, clear=0.27, drop=0.15),
                   mid=_w(push=0.35, drive=0.35, block=0.2, lift=0.1),
                   net=_w(net=0.6, push=0.2, lift=0.2),
                   defence=_w(lift=0.45, block=0.4, drive=0.15)),
        shirt=(0.86, 0.1, 0.1, 1), shorts=(0.86, 0.1, 0.1, 1), accent=(0.98, 0.84, 0.2, 1),
        ratings={"进攻": 93, "防守": 91, "速度": 90, "网前": 91, "稳定": 92}),
    Style(
        key="ginting", name="金廷", name_en="Anthony Ginting", title="闪电突击",
        blurb="启动快得看不清，平抽和扑杀一拍比一拍快，但也容易自己打丢。",
        handed="right", height_m=1.71, move_speed=6.5, reaction=0.09, reach=0.88, smash_speed=74.0,
        error=0.24, judgement=0.8, corner=0.36, cross=0.6, body=0.2, aggression=0.8, serve_low=0.6,
        shots=dict(kill=_w(smash=0.8, half_smash=0.2),
                   rear=_w(smash=0.5, slice=0.2, half_smash=0.15, clear=0.15),
                   mid=_w(drive=0.6, push=0.3, block=0.1),
                   net=_w(push=0.4, net=0.4, lift=0.2),
                   defence=_w(drive=0.4, block=0.3, lift=0.3)),
        shirt=(0.96, 0.96, 0.96, 1), shorts=(0.82, 0.1, 0.1, 1), accent=(0.82, 0.1, 0.1, 1),
        skin=(0.8, 0.62, 0.48, 1),
        ratings={"进攻": 94, "防守": 82, "速度": 99, "网前": 88, "稳定": 78}),
    Style(
        key="lee_zii_jia", name="李梓嘉", name_en="Lee Zii Jia", title="暴力重杀",
        blurb="反手也能杀球。给他一个半场高球，这一分就结束了。",
        handed="right", height_m=1.86, move_speed=5.6, reaction=0.12, reach=1.0, smash_speed=80.0,
        error=0.22, judgement=0.82, corner=0.42, cross=0.5, body=0.3, aggression=0.85, serve_low=0.3,
        shots=dict(kill=_w(smash=0.9, half_smash=0.1),
                   rear=_w(smash=0.62, half_smash=0.13, clear=0.15, slice=0.1),
                   mid=_w(drive=0.5, push=0.25, lift=0.25),
                   net=_w(lift=0.45, net=0.35, push=0.2),
                   defence=_w(lift=0.55, drive=0.25, block=0.2)),
        shirt=(0.1, 0.12, 0.3, 1), shorts=(0.1, 0.12, 0.3, 1), accent=(0.98, 0.82, 0.1, 1),
        skin=(0.86, 0.68, 0.52, 1),
        ratings={"进攻": 99, "防守": 82, "速度": 86, "网前": 78, "稳定": 80}),
    Style(
        key="an_se_young", name="安洗莹", name_en="An Se-young", title="铜墙铁壁",
        blurb="什么球都能救起来，而且不失误。想赢她只能比她更有耐心。",
        handed="right", height_m=1.70, move_speed=6.2, reaction=0.10, reach=0.92, smash_speed=62.0,
        error=0.11, judgement=0.95, corner=0.5, cross=0.65, body=0.05, aggression=0.3, serve_low=0.75,
        shots=dict(kill=_w(half_smash=0.5, smash=0.3, slice=0.2),
                   rear=_w(clear=0.4, drop=0.3, slice=0.2, half_smash=0.1),
                   mid=_w(block=0.4, push=0.35, lift=0.15, drive=0.1),
                   net=_w(net=0.6, lift=0.3, push=0.1),
                   defence=_w(lift=0.55, block=0.4, drive=0.05)),
        shirt=(0.96, 0.96, 0.98, 1), shorts=(0.1, 0.2, 0.55, 1), accent=(0.85, 0.12, 0.2, 1),
        hair=(0.06, 0.05, 0.05, 1),
        ratings={"进攻": 78, "防守": 99, "速度": 95, "网前": 92, "稳定": 99}),
    Style(
        key="tai_tzu_ying", name="戴资颖", name_en="Tai Tzu-ying", title="假动作 · 魔术手",
        blurb="同一个动作能打出四种球。别猜，等球出手再动。",
        handed="right", height_m=1.63, move_speed=5.9, reaction=0.11, reach=0.88, smash_speed=64.0,
        error=0.21, judgement=0.84, corner=0.26, cross=0.8, body=0.05, aggression=0.5, serve_low=0.7,
        shots=dict(kill=_w(slice=0.4, smash=0.35, half_smash=0.25),
                   rear=_w(slice=0.38, drop=0.27, clear=0.22, half_smash=0.13),
                   mid=_w(push=0.4, block=0.3, drive=0.2, lift=0.1),
                   net=_w(net=0.6, push=0.25, lift=0.15),
                   defence=_w(block=0.45, lift=0.4, drive=0.15)),
        shirt=(0.1, 0.55, 0.85, 1), shorts=(0.08, 0.08, 0.12, 1), accent=(0.96, 0.96, 0.96, 1),
        ratings={"进攻": 88, "防守": 84, "速度": 90, "网前": 98, "稳定": 80}),
    Style(
        key="marin", name="马林", name_en="Carolina Marin", title="左手快攻 · 压迫",
        blurb="左手持拍。每一拍都往前压，不给你喘气的时间。",
        handed="left", height_m=1.72, move_speed=6.1, reaction=0.10, reach=0.92, smash_speed=68.0,
        error=0.2, judgement=0.84, corner=0.4, cross=0.6, body=0.25, aggression=0.8, serve_low=0.6,
        shots=dict(kill=_w(smash=0.8, half_smash=0.2),
                   rear=_w(smash=0.4, half_smash=0.2, slice=0.2, clear=0.2),
                   mid=_w(drive=0.5, push=0.35, block=0.15),
                   net=_w(push=0.4, net=0.4, lift=0.2),
                   defence=_w(drive=0.35, lift=0.35, block=0.3)),
        shirt=(0.9, 0.1, 0.1, 1), shorts=(0.98, 0.82, 0.1, 1), accent=(0.98, 0.82, 0.1, 1),
        hair=(0.2, 0.13, 0.08, 1), skin=(0.9, 0.72, 0.58, 1),
        ratings={"进攻": 92, "防守": 86, "速度": 94, "网前": 84, "稳定": 84}),
]
BY_KEY = {s.key: s for s in ROSTER}

# the user's own avatar (and the stand-in "AI player" of the exhibition mode)
PLAYER_STYLE = Style(
    key="player", name="你", name_en="You", title="挑战者", blurb="", handed="right", height_m=1.76,
    move_speed=5.6, reaction=0.12, error=0.2, aggression=0.5,
    shots=dict(kill=_w(smash=0.7, half_smash=0.3), rear=_w(clear=0.35, smash=0.3, drop=0.2, slice=0.15),
               mid=_w(drive=0.4, push=0.3, block=0.3), net=_w(net=0.5, lift=0.4, push=0.1),
               defence=_w(lift=0.6, block=0.3, drive=0.1)),
    shirt=(0.12, 0.38, 0.9, 1), shorts=(0.08, 0.09, 0.14, 1), accent=(0.98, 0.85, 0.15, 1),
    ratings={})

DIFFICULTY = {
    # error: aiming-error multiplier; speed: court-speed multiplier; reaction: added [s];
    # attack: weight of the finishing shots in the style's menu; power: smash-speed multiplier;
    # width: how far into the corners the shots are placed (1 = the style's own margin);
    # reach: multiplier of the lunge / arm reach; beaten: probability that a hard smash is not read
    # in time; aggression: (practice-partner mode) how often the feed is dropped for a winner
    # (easy was speed 0.62 / attack 0.12 while the opponent's body could change velocity instantly; with
    # the accelerating body that speed left it late on every shuttle, so it never got a ball to attack)
    "easy": dict(error=2.5, speed=0.75, reaction=0.26, attack=0.3, power=0.58, width=0.7, reach=0.7, beaten=0.4,
                 aggression=0.5, label="休闲"),
    "normal": dict(error=2.0, speed=0.8, reaction=0.1, attack=0.45, power=0.76, width=0.85, reach=0.88, beaten=0.15,
                   aggression=0.9, label="标准"),
    "pro": dict(error=1.0, speed=1.0, reaction=0.0, attack=1.0, power=1.0, width=1.0, reach=1.0, beaten=0.0,
                aggression=1.2, label="职业"),
    # smart: how often the target is chosen from where the other player stands and is moving (wrong-footing,
    # the far corner) instead of the style's own habit; dive: how often a shuttle out of reach is dived for
    "legend": dict(error=0.8, speed=1.04, reaction=-0.01, attack=1.1, power=1.03, width=1.1, reach=1.03, beaten=0.0,
                   aggression=1.3, smart=0.75, dive=0.8, label="传奇"),
}

ALIASES = {"安塞龙": "axelsen", "林丹": "lin_dan", "李宗伟": "lee_chong_wei", "陶菲克": "taufik", "桃田": "momota",
           "桃田贤斗": "momota", "谌龙": "chen_long", "石宇奇": "shi_yuqi", "金廷": "ginting", "李梓嘉": "lee_zii_jia",
           "安洗莹": "an_se_young", "戴资颖": "tai_tzu_ying", "马林": "marin"}


def get_style(key):
    key = ALIASES.get(key, key)
    if key in BY_KEY:
        return BY_KEY[key]
    for s in ROSTER:
        if key in (s.name, s.name_en):
            return s
    raise KeyError(f"unknown opponent {key!r}; choices: {', '.join(BY_KEY)}")

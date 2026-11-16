"""羽你无瓜 (REAL2SIM badminton) -- the game. Played with the keyboard (or the on-screen stick and buttons).

    python scripts/play.py                                   # the app: menus (quick match / campaign / ladder / training,
                                                             #          Tab: the wardrobe)
    python scripts/play.py --opponent 林丹 --hand left --difficulty normal --skip-menu     # straight into a match
    python scripts/play.py --opponent 安赛龙 --tactics brain --skip-menu   # the opponent plays the self-play tactics
    python scripts/play.py --mode campaign --skip-menu       # the next campaign stage that has not been won
    python scripts/play.py --mode ladder --stage 9 --skip-menu             # ladder rung 9 (the self-play robot)
    python scripts/play.py --mode training --drill smash --skip-menu
    python scripts/play.py --source ai --skip-menu           # exhibition: two AIs

  Recording a whole game without a window (fixed time step, scripted thumbs, menus driven by a key script):
    python scripts/play.py --pad-bot --opponent lin_dan --difficulty easy --seed 1 --auto --headless --no-save --record results/match.mp4

Keys   menus: arrows / WASD, Enter, Esc, Tab (the wardrobe; settings in campaign / ladder / training), mouse
       match: P pause (exhibition: Space too), R restart, C camera, H performance read-out, F full screen, Q quit
       pad:   arrows / WASD aim (and run, with --move stick), J soft (drop / net), K high (clear / lift),
              L power (smash / drive), Space jump (then J / K / L: a jump smash is Space + L),
              Ctrl + direction dive.  Between points the arrows walk the player about -- the umpire allows 15 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.app import DIFF_KEYS, MODES, MOVES, POINTS, QUALITIES, SOURCES, TACTICS, App, Settings  # noqa: E402
from badrt.modes import CAMPAIGN, DRILL_KEYS, LADDER  # noqa: E402
from badrt.styles import ROSTER, get_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="lin_dan", help="roster key or name: " + ", ".join(s.key for s in ROSTER))
    ap.add_argument("--hand", default="right", choices=["right", "left"], help="your racket hand")
    ap.add_argument("--difficulty", default="easy", choices=DIFF_KEYS)
    ap.add_argument("--points", type=int, default=21, choices=POINTS)
    ap.add_argument("--source", default="pad", choices=SOURCES,
                    help="pad: you play (keyboard / on-screen buttons); ai: exhibition, an AI plays in your place")
    ap.add_argument("--tactics", default="style", choices=TACTICS,
                    help="quick match: the opponent's own hand-written tactics, or the latest self-play generation "
                         "(assets/brain.npz, scripts/train_selfplay.py)")
    ap.add_argument("--mode", default="quick", choices=MODES,
                    help="quick: one match with the settings given here; campaign: %d stages; ladder: %d odd characters "
                         "who play the self-play tactics; training: drills fed by a coach" % (len(CAMPAIGN), len(LADDER)))
    ap.add_argument("--stage", type=int, default=0,
                    help="campaign stage 1-%d / ladder rung 1-%d (default: the first one not won yet)" % (len(CAMPAIGN), len(LADDER)))
    ap.add_argument("--drill", default=DRILL_KEYS[0], choices=DRILL_KEYS, help="training drill")
    ap.add_argument("--no-save", action="store_true", help="do not read or write save/progress.json")
    ap.add_argument("--move", default="auto", choices=MOVES,
                    help="pad: automatic footwork (the arrows only aim) or run with the stick")
    ap.add_argument("--pad-bot", action="store_true", help="pad: scripted thumbs play for you (demo videos, tests)")
    ap.add_argument("--quality", default="high", choices=QUALITIES)
    ap.add_argument("--wear", nargs="*", default=None, metavar="SLOT=ITEM",
                    help="what your character wears in this run instead of the saved outfit, e.g. suit=panda head=panda "
                         "racket=fish (badrt/wardrobe.py lists the items)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--skip-menu", action="store_true", help="start the match straight away")
    ap.add_argument("--auto", action="store_true", help="drive the menus with a key script to the given settings")
    ap.add_argument("--headless", action="store_true", help="no window; fixed time step")
    ap.add_argument("--record", help="write the frames to this .mp4 (fixed time step)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of --record / --headless")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds")
    ap.add_argument("--log", help="match log json (default: next to --record, else results/last_match.json)")
    ap.add_argument("--perf", action="store_true", help="show the performance read-out")
    ap.add_argument("--no-render", action="store_true", help="simulate only (statistics); implies --headless --skip-menu")
    ap.add_argument("--realtime", action="store_true", help="wall clock even when --headless / --record (latency tests)")
    args = ap.parse_args()

    opp = get_style(args.opponent)
    target = Settings(opponent=[s.key for s in ROSTER].index(opp.key), handed=int(args.hand == "left"),
                      difficulty=DIFF_KEYS.index(args.difficulty), points=POINTS.index(args.points),
                      source=SOURCES.index(args.source), tactics=TACTICS.index(args.tactics),
                      quality=QUALITIES.index(args.quality), move=MOVES.index(args.move))
    outfit = None
    if args.wear is not None:
        from badrt import wardrobe
        outfit = {}
        for w in args.wear:
            slot, _, key = w.partition("=")
            if slot not in wardrobe.CATALOG:
                ap.error(f"--wear {w}: no such slot; slots: " + ", ".join(wardrobe.CATALOG))
            if key not in wardrobe.CATALOG[slot]:
                ap.error(f"--wear {w}: no such item; {slot} has: " + ", ".join(wardrobe.CATALOG[slot]))
            outfit[slot] = key
    app = App(size=tuple(args.size), settings=Settings() if args.auto else target, auto=target if args.auto else None,
              seed=args.seed, headless=args.headless or args.no_render, record=args.record, fps=args.fps,
              start="match" if (args.skip_menu or args.no_render) else "title", duration=args.duration, log=args.log,
              show_perf=args.perf, render=not args.no_render, realtime=args.realtime,
              pad_bot=args.pad_bot, mode=args.mode, stage=args.stage - 1 if args.stage else None,
              drill=DRILL_KEYS.index(args.drill), save=None if args.no_save else "default", outfit=outfit)
    for r in app.run():
        st = r["stats"]
        if "drill" in st:
            print(f"training {st['drill']['key']}: {st['drill']['ok']} / {st['drill']['reps']} good")
            continue
        extra = f", stage {r['stage']}: {r['stars']} star(s)" if "stage" in r else ""
        if "rung" in r:
            extra += f", ladder rung {r['rung']}: {r['stars']} star(s)"
        if "brain" in r:
            extra += f", opponent tactics: self-play generation {r['brain']['generation']} ({r['brain']['choices']} choices)"
        if r.get("coins"):
            extra += f", coins +{sum(n for _, n in r['coins'])}"
        if st.get("cards"):
            extra += ", cards: " + " ".join(st["cards"])
        print(f"{r['opponent']} ({r['difficulty']}, {r['source']}, {r['handed']}-handed): score {r['score'][0]}-{r['score'][1]}{extra}, "
              f"{r['game_seconds'] / 60:.1f} min, {st['points']} points, rally mean {st.get('rally_mean', 0):.1f} max {st.get('rally_max', 0)}")


if __name__ == "__main__":
    main()

"""Controller-mode balance (report section 4.12): scripted thumbs (perfect / human-like) against each difficulty.
The arcade layer is on (stamina, jump, dive, umpire), as in a real controller match.
Per game: score, shot buttons that became a swing / pressed, the player's dives (thrown > found a contact point >
touched the shuttle > the lift stayed in play), the opponent's dives, endurance left at the end (player / opponent).

    python scripts/pad_balance.py [points = 11] [seeds = 3] [opponent = axelsen]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.game import Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.opponent import PadBot  # noqa: E402
from badrt.styles import get_style  # noqa: E402

BOTS = dict(perfect=dict(), human=dict(jitter=0.14, miss=0.06))


def main():
    points = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    opp = sys.argv[3] if len(sys.argv) > 3 else "axelsen"
    lib = load_library(str(ROOT / "assets" / "stroke_library.npz"))
    for diff in ("easy", "normal", "pro", "legend"):
        for name, kw in BOTS.items():
            tot, rows = np.zeros(2), []
            for seed in range(seeds):
                m = Match(lib, get_style(opp), player="pad", points_to_win=points, seed=seed, difficulty=diff, arena=False,
                          pad_bot=PadBot(np.random.default_rng(100 + seed), **kw))
                k = 0
                while m.state != "over" and k < 60 * 60 * 12:
                    m.update(k / 60.0)
                    k += 1
                shots = [e for e in m.events if "pad" in e and e["pad"] != "dive"]
                dives = [e for e in m.events if e.get("pad") == "dive"]
                rest = [e for e in m.events if "who" in e or "point" in e]
                touched = [i for i, e in enumerate(rest) if e.get("who") == "p" and e.get("shot") == "dive"]
                back = sum(1 for i in touched if i + 1 < len(rest) and (                 # the opponent played it, or it landed in
                    rest[i + 1].get("who") == "o" or (rest[i + 1].get("point") == "p" and rest[i + 1].get("why") == "in")))
                rows.append("%d:%d (swings %d/%d, dives %d>%d>%d>%d opp %d, tank %.0f/%.0f%%)" % (
                    m.score[0], m.score[1], sum(e["ok"] for e in shots), len(shots), len(dives), sum(e["ok"] for e in dives),
                    len(touched), back, m.opp.dives, 100 * m.pad.stam.tank, 100 * m.opp.stam.tank))
                tot += m.score
            print("%-6s %-7s  points %3d:%-3d (%.0f%%)  | %s" % (diff, name, tot[0], tot[1], 100 * tot[0] / max(tot.sum(), 1),
                                                                 "  ".join(rows)), flush=True)


if __name__ == "__main__":
    main()

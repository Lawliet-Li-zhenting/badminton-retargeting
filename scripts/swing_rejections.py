"""Why swings are given up (report section 3.12): for every stroke clip tried by the planner, does the
arm get the string bed onto the contact (the one-solve precheck of AIPlayer._realise), against how
far the clip's string-bed normal is from the one the shot needs.

    python scripts/swing_rejections.py [seconds per match = 70] [seeds = 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import opponent as O  # noqa: E402
from badrt import swing as S  # noqa: E402
from badrt.game import Match  # noqa: E402
from badrt.library import load_library  # noqa: E402
from badrt.styles import get_style  # noqa: E402

rows = []                                        # (face mismatch [deg], failed, overhead, arm xy [m], arm z [m])
_contact_error = S.contact_error


def contact_error(ik, base, t_hit, p, n, r=0.03):
    ep, et = _contact_error(ik, base, t_hit, p, n, r)
    pc, Rc = ik.fk(base(t_hit))
    dp = np.asarray(p) - r * np.asarray(n) - pc
    rows.append((np.degrees(np.arccos(min(abs(float(Rc[:, 0] @ n)), 1.0))), ep > 0.03, p[2] >= 1.9,
                 float(np.linalg.norm(dp[:2])), float(dp[2])))
    return ep, et


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    O.contact_error = contact_error
    lib = load_library(str(ROOT / "assets" / "stroke_library.npz"))
    for seed in range(seeds):
        for mode in ("pad", "ai"):
            if mode == "pad":
                m = Match(lib, get_style("axelsen"), player="pad", arena=False, seed=seed, difficulty="normal",
                          pad_bot=O.PadBot(np.random.default_rng(seed + 5)))
            else:
                m = Match(lib, get_style("lin_dan"), player="ai", player_style=get_style("lee_chong_wei"), arena=False,
                          seed=seed)
            t = 0.0
            while t < seconds:
                t += 1 / 60
                m.update(t)
    R = np.array(rows, float)
    bad, oh = R[:, 1] > 0, R[:, 2] > 0
    print(f"clips tried {len(R)}, precheck failed {int(bad.sum())} (overhead {int((bad & oh).sum())})")
    for name, sel in (("overhead", oh), ("other", ~oh)):
        print(name)
        for lo, hi in ((0, 15), (15, 30), (30, 40), (40, 50), (50, 60), (60, 91)):
            k = sel & (R[:, 0] >= lo) & (R[:, 0] < hi)
            print("  face mismatch %2d-%2d deg: %3d tried, %3d failed" % (lo, hi, k.sum(), (k & bad).sum()))
    print("overheads, what the arm has to absorb (median): horizontal ok %.3f / failed %.3f m, vertical ok %+.3f / failed %+.3f m" % (
        np.median(R[oh & ~bad, 3]), np.median(R[oh & bad, 3]), np.median(R[oh & ~bad, 4]), np.median(R[oh & bad, 4])))


if __name__ == "__main__":
    main()

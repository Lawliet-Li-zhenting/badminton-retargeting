"""Collect results/*/summary.json, match logs and benchmarks into results/summary_tables.md."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FEET = [12, 13, 21, 22]         # heel->toe and ankle->heel of both feet in retarget.BONES


def retarget_rows(tags):
    out = ["| run | frames | keypoint err mm (mean/p95) | bone err deg (mean/p95) | foot bones deg | "
           "min sole mm | foot skating m/s (char / source) | hinge jitter rad/s^2 | retarget ms (mean/p99) | "
           "locked sole pts |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        f = RES / tag / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        pb = s["bone_error_per_bone_deg"]
        out.append(
            f"| {tag} ({s['preset']}, {s['source'].get('translation_mode', 'scale')}) | {s['frames']} | "
            f"{s['keypoint_error_mm']['mean']:.0f} / {s['keypoint_error_mm']['p95']:.0f} | "
            f"{s['bone_direction_error_deg']['mean']:.2f} / {s['bone_direction_error_deg']['p95']:.2f} | "
            f"{np.mean([pb[i] for i in FEET]):.1f} | {s['min_sole_height_mm']:.1f} | "
            f"{s['foot_skating_mps']:.2f} / {s['source_foot_skating_mps']:.2f} | {s['hinge_jitter_rad_s2']:.1f} | "
            f"{s['retarget_latency_ms']['mean']:.2f} / {s['retarget_latency_ms']['p99']:.2f} | "
            f"{s.get('locked_sole_points_mean', float('nan')):.2f} |")
    return out


def match_rows(paths):
    out = ["| log | duration s | points | rally shots (mean / median / max) | point endings | hits by kind |",
           "|---|---|---|---|---|---|"]
    for p in paths:
        f = RES / p
        if not f.exists():
            continue
        L = json.loads(f.read_text())
        E = L["events"]
        P = [e for e in E if "point" in e]
        H = [e for e in E if "kind" in e]
        shots = [e["shots"] for e in P] or [0]
        ends = collections.Counter(f"{e['point']}:{e['why']}" for e in P)
        kinds = collections.Counter(f"{e['who']}:{e['kind']}" for e in H)
        dur = E[-1]["t"] if E else 0
        why = collections.Counter(e["why"] for e in P)
        ends = {k: f"{v} ({100 * v / max(len(P), 1):.0f}%)" for k, v in why.most_common()}
        out.append(f"| {p} | {dur:.0f} | {len(P)} (score {L['score'][0]}:{L['score'][1]}) | "
                   f"{np.mean(shots):.1f} / {np.median(shots):.0f} / {max(shots)} | {ends} | "
                   f"{dict(sorted(kinds.items()))} |")
    return out


def main():
    tags = sys.argv[1:] or sorted(p.name for p in RES.iterdir() if (p / "summary.json").exists())
    lines = ["## Retargeting", ""] + retarget_rows(tags)
    stats = sorted(str(p.relative_to(RES)).replace("\\", "/") for p in (RES / "stats").glob("*.json"))
    lines += ["", "## Matches", ""] + match_rows(["demo/demo.json", "human_video/match.json",
                                                  "human_video/match_realistic.json"] + stats)
    for b in ("bench_realtime.json", "bench_realtime_threaded.json"):
        if (RES / b).exists():
            lines += ["", f"## {b}", "", "```", (RES / b).read_text(), "```"]
    txt = "\n".join(lines)
    (RES / "summary_tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

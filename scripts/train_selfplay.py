"""Self-play training of the opponents' tactics (badrt/brain.py).

Every generation: AI players with the current Brain (exploring) play each other, the hand-written styles and
earlier generations; every tactical choice gets the result of its rally; the Brain is updated; the new
generation is then measured (no exploration) against the hand-written tactics with the same bodies, on fixed
seeds and style pairs, both ways round. Bodies, speeds, errors: those of the roster at "pro" -- only the choice
of shot, side and depth is learned.

    python scripts/train_selfplay.py [--gens 10] [--matches 60] [--eval 12] [--points 11] [--workers 12] [--resume]

Writes assets/brain.npz (all generations) and results/selfplay_log.json after every generation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt import brain as B  # noqa: E402
from badrt.styles import ROSTER  # noqa: E402

KEYS = [s.key for s in ROSTER]
_lib = None


def _init():
    global _lib
    from badrt.library import load_library
    _lib = load_library(str(ROOT / "assets" / "stroke_library.npz"))


def play(task):
    """One AI-vs-AI game. task: seed, near, far (style keys), points, minds = {"p": spec, "o": spec} with
    spec = None (hand-written tactics) or dict(N, S, beta, eps), smart = {"p": x, "o": x} (the hand-written
    court reading of the legend tier). Returns the score and both sides' samples."""
    from badrt.game import Match
    from badrt.styles import get_style
    tic = time.perf_counter()
    m = Match(_lib, get_style(task["far"]), player="ai", player_style=get_style(task["near"]), seed=task["seed"],
              points_to_win=task["points"], difficulty="pro", arena=False)
    who = {"p": m.ai, "o": m.opp}
    for k, a in who.items():
        spec = task["minds"].get(k)
        if spec is not None:
            a.mind = B.Mind(B.Brain(spec["N"], spec["S"]), beta=spec["beta"], eps=spec["eps"])
        a.smart = task.get("smart", {}).get(k, 0.0)
    k = 0
    while m.state != "over" and k < 60 * 60 * 10:
        m.update(k / 60.0)
        k += 1
    strokes = {w: sum(1 for e in m.events if e.get("who") == w and not e.get("serve") and not e.get("deflect")) for w in who}
    return dict(tag=task.get("tag"), score=list(m.score), over=m.state == "over",
                samples={w: (a.mind.samples if a.mind is not None else []) for w, a in who.items()},
                strokes=strokes, game_s=k / 60.0, wall_s=time.perf_counter() - tic)


def spec(brain, beta, eps):
    return dict(N=brain.N, S=brain.S, beta=beta, eps=eps)


def eval_tasks(brain, pairs, points, beta, smart=0.0, tag="eval"):
    """Every pair twice: the Brain in the near player against plain tactics in the far one, and the other way round."""
    out = []
    for i, (a, b, seed) in enumerate(pairs):
        out.append(dict(tag=(tag, "p"), seed=seed, near=a, far=b, points=points, minds={"p": spec(brain, beta, 0.0)},
                        smart={"o": smart}))
        out.append(dict(tag=(tag, "o"), seed=seed, near=a, far=b, points=points, minds={"o": spec(brain, beta, 0.0)},
                        smart={"p": smart}))
    return out


def share(results, tag="eval"):
    """Points of the Brain's side / all points, with a standard error from the per-game shares."""
    mine = total = 0
    per = []
    for r in results:
        if r["tag"] is None or r["tag"][0] != tag:
            continue
        w = r["tag"][1]
        a, b = (r["score"][0], r["score"][1]) if w == "p" else (r["score"][1], r["score"][0])
        mine += a
        total += a + b
        per.append((a / max(a + b, 1), a > b))
    per = np.array(per, float)
    return dict(points=[int(mine), int(total - mine)], share=round(mine / max(total, 1), 4),
                se=round(float(per[:, 0].std(ddof=1) / np.sqrt(len(per))), 4), games_won=int(per[:, 1].sum()), games=len(per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--matches", type=int, default=60, help="training games per generation")
    ap.add_argument("--eval", type=int, default=12, help="evaluation pairs per generation (two games each)")
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--beta", type=float, default=4.0, help="how sharply the trained player follows its estimates")
    ap.add_argument("--beta-train", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.15, help="exploration while training")
    ap.add_argument("--decay", type=float, default=0.8, help="weight of older generations' data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from assets/brain.npz")
    ap.add_argument("--out", default=str(B.FILE))
    ap.add_argument("--log", default=str(ROOT / "results" / "selfplay_log.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    gens = [B.Brain()]
    log = dict(args=vars(a), generations=[])
    if a.resume and Path(a.out).exists():
        gens = B.generations(a.out)
        if Path(a.log).exists():
            log = json.loads(Path(a.log).read_text(encoding="utf-8"))
            log["args_resumed"] = vars(a)
        rng = np.random.default_rng(a.seed + 1000 * len(gens))
    cur = gens[-1].copy()
    # the same style pairs and seeds for every generation (seeds far from the training ones)
    erng = np.random.default_rng(12345)
    pairs = [(KEYS[int(erng.integers(len(KEYS)))], KEYS[int(erng.integers(len(KEYS)))], 900000 + i) for i in range(a.eval)]
    with mp.Pool(a.workers, initializer=_init) as pool:
        if len(gens) == 1:                    # generation 0: no data, i.e. the styles' habits in the Brain's action set
            ev = pool.map(play, eval_tasks(cur, pairs, a.points, a.beta), chunksize=1)
            log["generations"].append(dict(gen=0, samples=0, vs_plain=share(ev)))
            print(json.dumps(log["generations"][-1]), flush=True)
        for g in range(len(gens), len(gens) + a.gens):
            tic = time.perf_counter()
            tasks = []
            for i in range(a.matches):
                near, far = KEYS[int(rng.integers(len(KEYS)))], KEYS[int(rng.integers(len(KEYS)))]
                seed = int(rng.integers(1, 800000))
                kind = rng.random()
                me = spec(cur, a.beta_train, a.eps)
                if kind < 0.5:
                    minds = {"p": me, "o": me}                                        # itself
                else:
                    side = "p" if rng.random() < 0.5 else "o"
                    other = "o" if side == "p" else "p"
                    minds = {side: me}
                    if kind > 0.75 and len(gens) > 1:                                 # an earlier generation, playing to win
                        minds[other] = spec(gens[int(rng.integers(1, len(gens)))], a.beta, 0.0)
                tasks.append(dict(tag=("train", tuple(sorted(k for k, v in minds.items() if v is me))), seed=seed, near=near,
                                  far=far, points=a.points, minds=minds))
            res = pool.map(play, tasks, chunksize=1)
            samples = []
            for r in res:
                for w in r["tag"][1]:
                    samples += r["samples"][w]
            cur.decay(a.decay)
            cur.add(samples)
            cur.gen = g
            gens.append(cur.copy())
            B.save_generations(gens, a.out, meta=dict(generations=len(gens) - 1))
            t_train = time.perf_counter() - tic
            ev = pool.map(play, eval_tasks(cur, pairs, a.points, a.beta), chunksize=1)
            row = dict(gen=g, samples=len(samples), cells_seen=int((cur.N > 0.5).sum()), vs_plain=share(ev),
                       train_minutes=round(t_train / 60, 1), eval_minutes=round((time.perf_counter() - tic - t_train) / 60, 1),
                       game_wall_s=round(float(np.mean([r["wall_s"] for r in res])), 1),
                       brain_choices_per_stroke=round(sum(len(r["samples"][w]) for r in res for w in r["tag"][1]) / max(
                           sum(r["strokes"][w] for r in res for w in r["tag"][1]), 1), 3),
                       unfinished=int(sum(not r["over"] for r in res + ev)))
            log["generations"].append(row)
            Path(a.log).write_text(json.dumps(log, indent=1, default=float), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

"""Held-out evaluation of the self-play generations (assets/brain.npz), on style pairs and seeds that neither the
training games nor the per-generation check of train_selfplay.py used.

    python scripts/eval_brain.py                       # 576 games, about 25 minutes with 14 workers
    python scripts/eval_brain.py --pairs 12 --workers 8

Every comparison is "side A against side B" over the same pairs, each pair played twice with the sides swapped
(near / far end), 11-point games between AI players at the "pro" tier:

    generations   generation g (beta 4) against the styles' own hand-written tactics
    beta          the last generation at other values of beta against the hand-written tactics
    smart         generation 0 and the last generation against the hand-written tactics with the court reading of
                  the "legend" tier (smart = 0.75)
    head_to_head  the last generation against generation 1
    what          what was learned: per situation, the actions that the data rates best and worst over all states

Writes results/brain_eval.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# One BLAS thread per worker: with the default (one per core) twenty workers each reserve buffers for 24 threads,
# some fail to allocate them and die while starting, and the pool then waits for their games for ever.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from badrt import brain as B  # noqa: E402
from badrt.styles import SHOT_CN, SITUATIONS  # noqa: E402
from train_selfplay import KEYS, _init, play, share, spec  # noqa: E402


def versus(tag, pairs, points, a, b=None, smart=0.0):
    """Tasks: side A = spec a, side B = spec b (None: hand-written tactics, with `smart`), both ways round."""
    out = []
    for near, far, seed in pairs:
        for side, other in (("p", "o"), ("o", "p")):
            minds = {side: a}
            if b is not None:
                minds[other] = b
            out.append(dict(tag=(tag, side), seed=seed, near=near, far=far, points=points, minds=minds, smart={other: smart}))
    return out


def learned(brain, top=3):
    """Per situation: the actions with the largest and smallest advantage over the situation's average."""
    out = {}
    for i, sit in enumerate(SITUATIONS):
        k = i * B.PER_SIT
        N, S = brain.N[k:k + B.PER_SIT].sum(0), brain.S[k:k + B.PER_SIT].sum(0)
        v = S.sum() / max(N.sum(), 1e-9)
        rows = []
        for j, (shot, side, deep) in enumerate(B.ACTIONS[sit]):
            if N[j] >= 8:
                rows.append(dict(action=f"{SHOT_CN.get(shot, shot)} · {('直线', '斜线', '追身')[side]} · {('短', '长')[deep]}",
                                 n=round(float(N[j]), 1), value=round(float(S[j] / N[j]), 3), advantage=round(float(S[j] / N[j] - v), 3)))
        rows.sort(key=lambda r: -r["advantage"])
        out[sit] = dict(samples=round(float(N.sum()), 1), mean=round(float(v), 3), best=rows[:top], worst=rows[-top:][::-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=900.0, help="give up when no game finishes for this many seconds")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "brain_eval.json"))
    a = ap.parse_args()
    gens = B.generations()
    if len(gens) < 2:
        raise SystemExit("nothing trained yet: run scripts/train_selfplay.py first")
    last = len(gens) - 1
    rng = np.random.default_rng(777)
    pairs = [(KEYS[int(rng.integers(len(KEYS)))], KEYS[int(rng.integers(len(KEYS)))], 950000 + i) for i in range(a.pairs)]
    tasks = []
    picks = sorted({0, 1, 2, min(4, last), min(8, last), last})
    for g in picks:
        tasks += versus(f"gen{g}", pairs, a.points, spec(gens[g], a.beta, 0.0))
    for beta in (2.0, 8.0, 16.0):
        tasks += versus(f"beta{beta:g}", pairs, a.points, spec(gens[last], beta, 0.0))
    tasks += versus("smart_gen0", pairs, a.points, spec(gens[0], a.beta, 0.0), smart=0.75)
    tasks += versus("smart_last", pairs, a.points, spec(gens[last], a.beta, 0.0), smart=0.75)
    tasks += versus("last_vs_gen1", pairs, a.points, spec(gens[last], a.beta, 0.0), spec(gens[1], a.beta, 0.0))
    tic = time.perf_counter()
    res = []
    with mp.Pool(a.workers, initializer=_init) as pool:
        it = pool.imap_unordered(play, tasks, chunksize=1)
        for k in range(len(tasks)):
            try:
                res.append(it.next(timeout=a.timeout))
            except mp.TimeoutError:
                raise SystemExit(f"no game finished in {a.timeout:.0f} s after {k} of {len(tasks)}: a worker died "
                                 f"(see stderr above); try fewer --workers")
            if (k + 1) % 24 == 0 or k + 1 == len(tasks):
                print(f"{k + 1}/{len(tasks)} games, {(time.perf_counter() - tic) / 60:.1f} min", file=sys.stderr, flush=True)
    out = dict(pairs=a.pairs, games_per_comparison=2 * a.pairs, points=a.points, beta=a.beta, generations_trained=last,
               samples_last=int(round(gens[last].N.sum())), unfinished=int(sum(not r["over"] for r in res)),
               minutes=round((time.perf_counter() - tic) / 60, 1),
               generations={str(g): share(res, f"gen{g}") for g in picks},
               beta_last={f"{b:g}": share(res, f"beta{b:g}") for b in (2.0, 8.0, 16.0)},
               smart=dict(gen0=share(res, "smart_gen0"), last=share(res, "smart_last")),
               head_to_head=dict(last_vs_gen1=share(res, "last_vs_gen1")),
               what=learned(gens[last]))
    out["beta_last"][f"{a.beta:g}"] = out["generations"][str(last)]
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "what"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

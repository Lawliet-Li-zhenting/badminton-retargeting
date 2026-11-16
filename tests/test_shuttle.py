"""Physics checks for the shuttlecock model. Prints the numbers used in the report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badrt.shuttle import (RacketImpact, RacketPose, ShuttleParams, ShuttlePhysics,  # noqa: E402
                           ShuttleState, aim_shot, launch_batch, solve_launch)

DT = 1e-3


def fly(phys, s, t_end, dt=DT):
    t, out = 0.0, []
    while t < t_end - 1e-12:
        phys.step(s, dt); t += dt
        out.append((t, s.pos.copy(), s.vel.copy(), s.axis.copy()))
    return out


def test_free_fall(phys, res):
    U = phys.p.terminal_velocity
    s = ShuttleState(pos=np.array([0, 0, 50.0]), vel=np.zeros(3), axis=np.array([0, 0, -1.0]))
    tr = fly(phys, s, 3.0)
    t = np.array([x[0] for x in tr]); vz = np.array([-x[2][2] for x in tr])
    ref = U * np.tanh(9.81 * t / U)
    err = np.max(np.abs(vz - ref)) / U
    res["free_fall_max_rel_err"] = float(err)
    res["free_fall_v_at_1s"] = float(vz[np.argmin(np.abs(t - 1.0))])
    assert err < 0.01


def test_smash_decay(phys, res):
    s = ShuttleState(pos=np.array([0, -3.0, 2.8]), vel=np.array([0, 90.0 * np.cos(0.2), -90.0 * np.sin(0.2)]),
                     axis=np.array([0, np.cos(0.2), -np.sin(0.2)]))
    tr = fly(phys, s, 1.5)
    t = np.array([x[0] for x in tr]); p = np.array([x[1] for x in tr]); v = np.array([x[2] for x in tr])
    land = np.argmax(p[:, 2] <= 0)
    sp = np.linalg.norm(v, axis=1)
    res["smash_90mps"] = dict(speed_at_0p1s=float(sp[np.argmin(np.abs(t - 0.1))]),
                              speed_at_landing=float(sp[land]), flight_time=float(t[land]),
                              distance=float(p[land, 1] - (-3.0)))
    assert sp[land] < 0.4 * 90


def test_turnover(phys, res):
    out = {}
    for v0 in (20.0, 50.0, 80.0):
        s = ShuttleState(pos=np.array([0, 0, 10.0]), vel=np.array([0, v0, 0]),
                         axis=np.array([0, -1.0, 0.02]) / np.linalg.norm([0, -1.0, 0.02]))
        tr = fly(phys, s, 0.3, dt=2e-4)
        cosang = np.array([x[3] @ x[2] / np.linalg.norm(x[2]) for x in tr])
        k = int(np.argmax(cosang > 0.9))
        dist = np.linalg.norm(tr[k][1] - np.array([0, 0, 10.0]))
        # speed lost vs a shuttle that is aligned from the start
        s2 = ShuttleState(pos=np.array([0, 0, 10.0]), vel=np.array([0, v0, 0]), axis=np.array([0, 1.0, 0]))
        tr2 = fly(phys, s2, tr[k][0], dt=2e-4)
        out[f"{int(v0)}mps"] = dict(flip_time_ms=tr[k][0] * 1e3, flip_distance_m=float(dist),
                                    speed_after_flip=float(np.linalg.norm(tr[k][2])),
                                    speed_if_aligned=float(np.linalg.norm(tr2[-1][2])))
    res["turnover"] = out


def test_impact(res):
    imp = RacketImpact()
    out = {}
    R = np.eye(3)                                # face normal +x
    for vr in (20.0, 40.0, 60.0):
        for name, axis in (("cork_first", [-1.0, 0, 0]), ("skirt_first", [1.0, 0, 0])):
            s = ShuttleState(pos=np.array([0.01, 0, 0]), vel=np.array([-5.0, 0, 0]), axis=np.array(axis))
            info = imp.respond(s, "strings", np.zeros(3), np.array([1.0, 0, 0]), np.array([vr, 0, 0]))
            out[f"{name}_{int(vr)}"] = round(float(np.linalg.norm(s.vel)) / vr, 3)
    res["impact_speed_ratio_out_over_racket"] = out
    # swept detection of a fast racket crossing a slow shuttle in one 1 ms sub-step
    r0 = RacketPose(np.array([-0.03, 0, 0]), R); r1 = RacketPose(np.array([0.03, 0, 0]), R)
    s0 = ShuttleState(pos=np.array([0.0, 0.02, 0.03])); s1 = s0.copy()
    res["swept_detection_60mps"] = imp.check(s0, s1, r0, r1) is not None
    assert res["swept_detection_60mps"]
    assert 1.3 < out["cork_first_40"] < 1.7


def test_clear_range(phys, res):
    out = {}
    for v0, deg in ((50.0, 40.0), (60.0, 45.0), (35.0, 50.0)):
        a = np.radians(deg)
        s = ShuttleState(pos=np.array([0, 0, 2.5]), vel=np.array([0, v0 * np.cos(a), v0 * np.sin(a)]),
                         axis=np.array([0, np.cos(a), np.sin(a)]))
        tr = fly(phys, s, 5.0, dt=2e-3)
        p = np.array([x[1] for x in tr]); t = np.array([x[0] for x in tr])
        k = int(np.argmax(p[:, 2] <= 0))
        out[f"{int(v0)}mps_{int(deg)}deg"] = dict(range_m=float(p[k, 1]), apex_m=float(p[:, 2].max()),
                                                  flight_time_s=float(t[k]))
    res["clear"] = out


def test_aiming(res):
    sp = ShuttleParams()
    errs = []
    rng = np.random.default_rng(1)
    for _ in range(20):
        p0 = np.array([rng.uniform(-2, 2), rng.uniform(-6, -1.5), rng.uniform(0.5, 2.8)])
        tgt = np.array([rng.uniform(-2.3, 2.3), rng.uniform(1.0, 6.3)])
        sol = solve_launch(p0, tgt, rng.choice([25.0, 40.0, 60.0]), sp, high=bool(rng.integers(2)))
        if sol is not None:
            errs.append(np.linalg.norm(sol[1][:2] - tgt))
    res["aiming"] = dict(reachable_solved=len(errs), of=20, mean_err_m=float(np.mean(errs)),
                         max_err_m=float(np.max(errs)))
    assert np.max(errs) < 0.3


def test_turnover_aiming(phys, res):
    """Shots hit back the way the shuttle came (cork pointing at the racket): solve_launch assumes
    an aligned shuttle, aim_shot corrects with the full model including turnover drag."""
    from badrt.court import net_height
    rng = np.random.default_rng(3)
    out = {"solve_launch": [], "aim_shot": []}
    net = {"solve_launch": 0, "aim_shot": 0}
    solved = {"solve_launch": 0, "aim_shot": 0}
    n = 30
    for _ in range(n):
        p0 = np.array([rng.uniform(-2, 2), rng.uniform(-6, -3), rng.uniform(2.2, 2.8)])
        tgt = np.array([rng.uniform(-2, 2), rng.uniform(1.2, 2.8)])          # drops from the back court
        speed = 20.0
        axis_in = np.array([0, -1.0, -0.4]) / np.linalg.norm([0, -1.0, -0.4])  # incoming, cork first
        mk = lambda v0: ShuttleState(p0.copy(), v0.copy(), axis_in.copy(), np.cross(axis_in, v0) * 0.5)  # noqa: E731
        sol = None
        for k in range(6):                                   # old planner: raise speed until solvable
            sol = solve_launch(p0, tgt, speed * 1.12 ** k, phys.p)
            if sol is not None:
                break
        v_aim = aim_shot(phys, mk, p0, tgt, speed, iters=6)
        for name, v0 in (("solve_launch", None if sol is None else sol[0]), ("aim_shot", v_aim)):
            if v0 is None:
                continue
            solved[name] += 1
            T, P, V = phys.predict(mk(v0), 4.0, dt=1e-3)
            c = np.flatnonzero(P[:, 1] > 0)
            if len(c) == 0 or P[c[0], 2] < net_height(P[c[0], 0]):
                net[name] += 1
            out[name].append(float(np.linalg.norm(P[-1, :2] - tgt)))
    res["turnover_aiming_drop_20mps"] = dict(
        shots=n, solved=solved, net_faults=net,
        landing_err_m={k: dict(mean=float(np.mean(v)), max=float(np.max(v))) for k, v in out.items()})
    assert net["aim_shot"] <= net["solve_launch"]


def test_fast_predict(phys, res):
    """predict() (scalar code path used for planning) must match step() exactly."""
    errs = []
    for axis in ([0, -1.0, 0.0], [0, 0.7, 0.7], [0.3, 0.2, -0.9]):
        s = ShuttleState(pos=np.array([0, -5.0, 2.5]), vel=np.array([1.0, 60.0, -5.0]),
                         axis=np.array(axis) / np.linalg.norm(axis))
        T, P, V = phys.predict(s, 2.0, dt=2e-3)
        r = s.copy(); Q = [r.pos.copy()]
        for _ in range(len(T) - 1):
            phys.step(r, 2e-3); Q.append(r.pos.copy())
        errs.append(float(np.abs(np.array(Q) - P).max()))
    # aiming integrator step (5 ms) vs a 0.5 ms reference
    v0s = np.array([[0, 30.0, 20.0], [0, 60.0, -10.0], [1.0, 10.0, 8.0]])
    a, _, _ = launch_batch(np.array([0, -5.0, 2.5]), v0s, phys.p, dt=0.005)
    b, _, _ = launch_batch(np.array([0, -5.0, 2.5]), v0s, phys.p, dt=0.0005)
    res["fast_predict_max_dev_m"] = max(errs)
    res["launch_batch_5ms_vs_0p5ms_landing_dev_m"] = float(np.abs(a - b).max())
    assert max(errs) < 1e-9 and np.abs(a - b).max() < 0.05


if __name__ == "__main__":
    phys = ShuttlePhysics()
    res = {}
    test_free_fall(phys, res)
    test_smash_decay(phys, res)
    test_turnover(phys, res)
    test_impact(res)
    test_clear_range(phys, res)
    test_aiming(res)
    test_turnover_aiming(phys, res)
    test_fast_predict(phys, res)
    print(json.dumps(res, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "physics_tests.json").write_text(json.dumps(res, indent=1))

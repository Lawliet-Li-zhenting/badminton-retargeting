"""Fixed-structure Clarabel QP for one SQP step (condensed costs, native update()).

    x = [dq (n) ; s (ns)]                    s: slacks of the soft (self-collision) rows
    min 0.5 x'Px + c'x   s.t.  A x + s_cone = b
    rows: dq <= ub-q | -dq <= -(lb-q) | hard rows  -a'dq <= -rhs  (cap_h)
          soft rows  -a'dq - s <= -rhs (cap_s) | -s <= 0 | SOC (step, dq)
Unused rows are padded as 0*x <= 1, so the CSC pattern never changes and the
Clarabel workspace is reused through solver.update(P=, q=, A=, b=).
"""
from __future__ import annotations

import clarabel
import numpy as np
from scipy import sparse as sp


def bucket(n):
    return 0 if n == 0 else max(4, 1 << int(np.ceil(np.log2(n))))


class ParametricQP:
    def __init__(self, n, cap_hard, cap_soft, soft_weight, step_size):
        self.n, self.ch, self.cs = n, cap_hard, cap_soft
        self.nx = n + cap_soft
        self.step = step_size
        pr, pc = np.triu_indices(n)
        pr = np.r_[pr, np.arange(n, self.nx)]; pc = np.r_[pc, np.arange(n, self.nx)]
        self._P = sp.coo_matrix((np.ones(len(pr)), (pr, pc)), shape=(self.nx, self.nx)).tocsc()
        self._P.data[:] = 0.0
        ip = self._index(self._P)
        self._pr, self._pc = np.triu_indices(n)
        self._p_idx = np.array([ip[r, c] for r, c in zip(self._pr, self._pc)])
        for i in range(n, self.nx):
            self._P.data[ip[i, i]] = 2.0 * soft_weight

        rows, cols, vals = [], [], []

        def add(r, c, v):
            r = np.asarray(r).ravel(); c = np.asarray(c).ravel()
            rows.extend(r); cols.extend(c); vals.extend(np.broadcast_to(v, r.shape))

        j = np.arange(n)
        add(j, j, 1.0); add(n + j, j, 1.0)
        cur = 2 * n
        self._hard = slice(cur, cur + cap_hard)
        rr, cc = np.indices((cap_hard, n)); add(cur + rr, cc, 1.0); cur += cap_hard
        self._soft = slice(cur, cur + cap_soft)
        rr, cc = np.indices((cap_soft, n)); add(cur + rr, cc, 1.0)
        add(cur + np.arange(cap_soft), n + np.arange(cap_soft), -1.0); cur += cap_soft
        add(cur + np.arange(cap_soft), n + np.arange(cap_soft), -1.0); cur += cap_soft
        self.n_lin = cur
        add(cur + 1 + j, j, -1.0)
        self._A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(cur + n + 1, self.nx)).tocsc()
        ia = self._index(self._A)
        # restore the fixed values (coo->csc summed duplicates of ones; rebuild explicitly)
        self._A.data[:] = 0.0
        for r, c, v in zip(rows, cols, vals):
            self._A.data[ia[int(r), int(c)]] = v
        self._ub_idx = np.array([ia[i, i] for i in j])
        self._lb_idx = np.array([ia[n + i, i] for i in j])
        self._h_idx = np.array([[ia[r, c] for c in j] for r in range(self._hard.start, self._hard.stop)], int).reshape(cap_hard, n)
        self._s_idx = np.array([[ia[r, c] for c in j] for r in range(self._soft.start, self._soft.stop)], int).reshape(cap_soft, n)
        self._b = np.zeros(self._A.shape[0]); self._b[cur] = step_size
        self._c = np.zeros(self.nx)
        self._cones = [clarabel.NonnegativeConeT(cur), clarabel.SecondOrderConeT(n + 1)]
        st = clarabel.DefaultSettings(); st.verbose = False; st.presolve_enable = False
        self._settings = st
        self._solver = None
        self.status = None

    @staticmethod
    def _index(M):
        return {(int(M.indices[k]), c): k for c in range(M.shape[1])
                for k in range(M.indptr[c], M.indptr[c + 1])}

    def _rows(self, rows, idx, sl):
        self._A.data[idx] = 0.0
        self._b[sl] = 1.0
        if len(rows):
            a = np.array([r[0] for r in rows]); rhs = np.array([r[1] for r in rows])
            self._A.data[idx[:len(rows)]] = -a
            self._b[sl.start:sl.start + len(rows)] = -rhs

    def solve(self, H, g, lower, upper, hard_rows, soft_rows):
        """H, g: condensed cost dq'H dq + 2 g'dq. lower/upper: lb-q, ub-q."""
        n = self.n
        self._P.data[self._p_idx] = 2.0 * H[self._pr, self._pc]
        self._c[:n] = 2.0 * g
        au = upper < self.step; al = lower > -self.step
        self._A.data[self._ub_idx] = au.astype(float)
        self._A.data[self._lb_idx] = -al.astype(float)
        self._b[:n] = np.where(au, upper, 1.0)
        self._b[n:2 * n] = np.where(al, -lower, 1.0)
        self._rows(hard_rows, self._h_idx, self._hard)
        self._rows(soft_rows, self._s_idx, self._soft)
        if self._solver is None:
            self._solver = clarabel.DefaultSolver(self._P, self._c, self._A, self._b, self._cones, self._settings)
        else:
            self._solver.update(P=self._P.data, q=self._c, A=self._A.data, b=self._b)
        res = self._solver.solve()
        st = str(res.status)
        if st in ("InsufficientProgress", "NumericalError"):
            self._solver = clarabel.DefaultSolver(self._P, self._c, self._A, self._b, self._cones, self._settings)
            res = self._solver.solve(); st = str(res.status)
        self.status = st
        if st not in ("Solved", "AlmostSolved"):
            return None, None
        x = np.asarray(res.x)
        return x[:n], x[n:n + len(soft_rows)]

    def solve_cvxpy(self, H, g, lower, upper, hard_rows, soft_rows, soft_weight):
        """Slow fallback with the same formulation (built from scratch every call)."""
        import cvxpy as cp
        n = self.n
        dq = cp.Variable(n)
        Hs = 0.5 * (H + H.T) + 1e-9 * np.eye(n)
        cost = cp.quad_form(dq, cp.psd_wrap(Hs)) + 2 * g @ dq
        cons = [cp.norm(dq, 2) <= self.step]
        fu, fl = np.isfinite(upper), np.isfinite(lower)
        if fu.any(): cons.append(dq[fu] <= upper[fu])
        if fl.any(): cons.append(dq[fl] >= lower[fl])
        for a, rhs in hard_rows:
            cons.append(a @ dq >= rhs)
        s = None
        if soft_rows:
            s = cp.Variable(len(soft_rows), nonneg=True)
            Am = np.array([r[0] for r in soft_rows]); bm = np.array([r[1] for r in soft_rows])
            cons.append(Am @ dq + s >= bm)
            cost = cost + soft_weight * cp.sum_squares(s)
        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return None, None
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None
        return np.asarray(dq.value), (np.asarray(s.value) if s is not None else np.zeros(0))

"""
algorithm2.py
-------------
A2: Exact constrained clustering via MIQP (Gruppo 66, A.Y. 2025/26).

Formulation: Mixed-Integer Quadratic Program with auxiliary variables z_ijl
to linearise the bilinear terms a_ij * c_jl (Section 1.3.6 of the report).
Solved with Gurobi via branch-and-bound.

References
----------
[4] Frangioni – Optimization & Learning Lecture Notes, Univ. Pisa (2025)
[7] Gurobi Optimization – Gurobi Optimizer Reference Manual (2023)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Gurobi environment — created lazily on first call to run_a2()
# This avoids the WLS network validation overhead at import time.
# ---------------------------------------------------------------------------

_GRB_ENV = None

def _get_env():
    """Return the shared Gurobi environment, creating it once if needed."""
    global _GRB_ENV
    if _GRB_ENV is None:
        _GRB_ENV = gp.Env(empty=True)
        _GRB_ENV.setParam("OutputFlag", 0)
        _GRB_ENV.start()   # WLS network call happens here, once only
    return _GRB_ENV


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class A2Result:
    """
    Output of one A2 run.

    Attributes
    ----------
    centroids  : (k, d) array  – optimal centroids
    labels     : (n,)   array  – optimal point-to-cluster assignment
    cost       : float         – optimal objective value (sum of squared L2)
    gap        : float         – Gurobi MIP gap (0.0 = proven optimal)
    solve_time : float         – Gurobi solver time in seconds
    status     : str           – 'OPTIMAL', 'TIME_LIMIT', etc.
    """
    centroids:  np.ndarray
    labels:     np.ndarray
    cost:       float
    gap:        float
    solve_time: float
    status:     str


# ---------------------------------------------------------------------------
# Algorithm A2
# ---------------------------------------------------------------------------

def run_a2(X, k, m, time_limit=None, verbose=False):
    """
    Exact constrained clustering via MIQP (Algorithm A2).

    Solves the constrained clustering problem (P) exactly using the MIQP
    reformulation of Section 1.3.6. The bilinear term a_ij * c^l_j (degree 3)
    is replaced by auxiliary variable z^l_ij via linearisation constraints
    V1 and V2 (eq. 15-16), reducing the objective to degree 2 (eq. 18).

    Decision variables
    ------------------
    a[i,j]   in {0,1}         : 1 if point i is assigned to cluster j
    c[j,l]   in [L_l, U_l]   : coordinate l of centroid j
    z[i,j,l] in [min(0,L_l),  : auxiliary variable = a[i,j] * c[j,l]
                  max(0,U_l)]

    Objective (eq. 18)
    ------------------
    min  sum_i sum_j [ a[i,j]*||xi||^2 + sum_l (c[j,l] - 2*xi_l) * z[i,j,l] ]

    Parameters
    ----------
    X          : (n, d) float array – data points
    k          : int   – number of clusters
    m          : int   – capacity limit per centroid (>= ceil(n/k))
    time_limit : float – optional Gurobi time limit in seconds
    verbose    : bool  – if True, show Gurobi solver log

    Returns
    -------
    A2Result with optimal centroids, labels, cost, gap and solve time.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Bounds on centroids: optimal centroids are weighted means of data points,
    # so they necessarily lie in the data hyper-rectangle (Section 1.1).
    L = X.min(axis=0)   # (d,) lower bound per dimension
    U = X.max(axis=0)   # (d,) upper bound per dimension

    # Precompute ||xi||^2 for all points (used in objective)
    xi_sq = np.sum(X ** 2, axis=1)   # (n,)

    # ------------------------------------------------------------------
    # Build and solve Gurobi model
    # ------------------------------------------------------------------
    with gp.Model(env=_get_env()) as model:

        if verbose:
            model.setParam("OutputFlag", 1)
        if time_limit is not None:
            model.setParam("TimeLimit", time_limit)

        # --------------------------------------------------------------
        # Decision variables
        # --------------------------------------------------------------

        # a[i,j] in {0,1}: assignment
        a = model.addVars(n, k, vtype=GRB.BINARY, name="a")

        # c[j,l] in [L_l, U_l]: centroid coordinates
        c = model.addVars(
            k, d,
            lb={(j, l): float(L[l]) for j in range(k) for l in range(d)},
            ub={(j, l): float(U[l]) for j in range(k) for l in range(d)},
            name="c",
        )

        # z[i,j,l]: auxiliary variable representing a[i,j] * c[j,l]
        z = model.addVars(
            n, k, d,
            lb={(i, j, l): float(min(0.0, L[l]))
                for i in range(n) for j in range(k) for l in range(d)},
            ub={(i, j, l): float(max(0.0, U[l]))
                for i in range(n) for j in range(k) for l in range(d)},
            name="z",
        )

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------

        # (2) Each point assigned to exactly one centroid
        model.addConstrs(
            (gp.quicksum(a[i, j] for j in range(k)) == 1
             for i in range(n)),
            name="assign",
        )

        # (3) Capacity: each centroid receives at most m points
        model.addConstrs(
            (gp.quicksum(a[i, j] for i in range(n)) <= m
             for j in range(k)),
            name="capacity",
        )

        # Linearisation V1 and V2 (eq. 15-16):
        # V1: z = 0 when a = 0  (point not assigned to cluster j)
        # V2: z = c when a = 1  (point assigned to cluster j)
        for i in range(n):
            for j in range(k):
                for l in range(d):
                    Ll, Ul = float(L[l]), float(U[l])
                    # V1
                    model.addConstr(z[i, j, l] >= Ll * a[i, j])
                    model.addConstr(z[i, j, l] <= Ul * a[i, j])
                    # V2
                    model.addConstr(z[i, j, l] >= c[j, l] - Ul * (1 - a[i, j]))
                    model.addConstr(z[i, j, l] <= c[j, l] - Ll * (1 - a[i, j]))

        # --------------------------------------------------------------
        # Objective (eq. 18)
        # --------------------------------------------------------------
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(k):
                obj += xi_sq[i] * a[i, j]
                for l in range(d):
                    obj += (c[j, l] - 2.0 * X[i, l]) * z[i, j, l]

        model.setObjective(obj, GRB.MINIMIZE)

        # --------------------------------------------------------------
        # Solve
        # --------------------------------------------------------------
        model.optimize()

        STATUS_MAP = {
            GRB.OPTIMAL:    "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
        }
        status_str = STATUS_MAP.get(model.Status, str(model.Status))

        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi found no feasible solution (status={status_str})."
            )

        # --------------------------------------------------------------
        # Extract solution
        # --------------------------------------------------------------
        centroids = np.array([
            [c[j, l].X for l in range(d)]
            for j in range(k)
        ])

        labels = np.full(n, -1, dtype=int)
        for i in range(n):
            for j in range(k):
                if a[i, j].X > 0.5:
                    labels[i] = j
                    break

        cost       = float(model.ObjVal)
        gap        = float(model.MIPGap)
        solve_time = float(model.Runtime)

    return A2Result(
        centroids=centroids,
        labels=labels,
        cost=cost,
        gap=gap,
        solve_time=solve_time,
        status=status_str,
    )
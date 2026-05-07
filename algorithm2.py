"""
a2_algorithm.py
---------------
A2: Exact constrained clustering via MIQP (Gruppo 66, A.Y. 2025/26).

Formulation: Mixed-Integer Quadratic Program with auxiliary variables z_ijl
to linearise the bilinear terms a_ij * c_jl (Section 1.3.6).
Solved with Gurobi (branch-and-bound).

"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def run_a2(X, k, m, time_limit=None, verbose=False):
    """
    Exact constrained clustering via MIQP (Algorithm A2).

    Solves the MIQP formulation (eq. 18) with Gurobi:
      - a[i,j]   in {0,1}          assignment variables
      - c[j,l]   in [L_l, U_l]     centroid coordinates
      - z[i,j,l] in [min(0,L_l),   auxiliary variables: z = a*c
                      max(0,U_l)]

    Parameters
    ----------
    X          : (n, d) float array – data points
    k          : int   – number of clusters
    m          : int   – capacity limit per centroid (>= ceil(n/k))
    time_limit : float – optional Gurobi time limit in seconds
    verbose    : bool  – if True, show Gurobi log

    Returns
    -------
    centroids : (k, d) float array – optimal centroids
    labels    : (n,)   int array   – optimal assignments
    cost      : float              – optimal objective value
    gap       : float              – optimality gap (0.0 if solved to optimality)
    elapsed   : float              – wall-clock time in seconds
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Per-dimension bounds (centroids lie in the data hyper-rectangle)
    L = X.min(axis=0)   # shape (d,)
    U = X.max(axis=0)   # shape (d,)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model()
    model.setParam("OutputFlag", 1 if verbose else 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # --- Decision variables ---

    # a[i,j] in {0,1}: 1 if point i is assigned to cluster j
    a = model.addVars(n, k, vtype=GRB.BINARY, name="a")

    # c[j,l] in [L_l, U_l]: coordinate l of centroid j
    c = model.addVars(
        k, d,
        lb={(j, l): float(L[l]) for j in range(k) for l in range(d)},
        ub={(j, l): float(U[l]) for j in range(k) for l in range(d)},
        name="c"
    )

    # z[i,j,l]: auxiliary variable representing a[i,j] * c[j,l]
    z = model.addVars(
        n, k, d,
        lb={(i, j, l): min(0.0, float(L[l]))
            for i in range(n) for j in range(k) for l in range(d)},
        ub={(i, j, l): max(0.0, float(U[l]))
            for i in range(n) for j in range(k) for l in range(d)},
        name="z"
    )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (2) Each point assigned to exactly one centroid
    for i in range(n):
        model.addConstr(
            gp.quicksum(a[i, j] for j in range(k)) == 1,
            name=f"assign_{i}"
        )

    # (3) Capacity: each centroid gets at most m points
    for j in range(k):
        model.addConstr(
            gp.quicksum(a[i, j] for i in range(n)) <= m,
            name=f"cap_{j}"
        )

    # Linearisation constraints (V1 and V2, eq. 15-16)
    # V1: z = 0 when a = 0
    # V2: z = c when a = 1
    for i in range(n):
        for j in range(k):
            for l in range(d):
                Ll = float(L[l])
                Ul = float(U[l])
                # V1
                model.addConstr(z[i, j, l] >= Ll * a[i, j])
                model.addConstr(z[i, j, l] <= Ul * a[i, j])
                # V2
                model.addConstr(z[i, j, l] >= c[j, l] - Ul * (1 - a[i, j]))
                model.addConstr(z[i, j, l] <= c[j, l] - Ll * (1 - a[i, j]))

    # ------------------------------------------------------------------
    # Objective: eq. (18)
    # sum_i sum_j [ a[i,j]*||xi||^2 + sum_l (c[j,l] - 2*xi_l) * z[i,j,l] ]
    # ------------------------------------------------------------------
    obj = gp.QuadExpr()
    for i in range(n):
        norm_sq_xi = float(np.dot(X[i], X[i]))
        for j in range(k):
            obj += norm_sq_xi * a[i, j]
            for l in range(d):
                obj += (c[j, l] - 2.0 * X[i, l]) * z[i, j, l]

    model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    elapsed = time.perf_counter() - t0

    if model.SolCount == 0:
        raise RuntimeError("Gurobi found no feasible solution.")

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
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

    cost = float(model.ObjVal)
    gap  = float(model.MIPGap)   # 0.0 if solved to proven optimality

    return centroids, labels, cost, gap, elapsed
"""
algorithm1.py
-------------
A1: Modified k-means with capacity constraint (Gruppo 66, A.Y. 2025/26).

Constrained assignment solved via Minimum Cost Flow (OR-Tools cost-scaling
push-relabel, Goldberg & Tarjan [10]).  Centroid update: arithmetic mean of
assigned points — exact minimiser of the L2 objective at fixed assignments
(Theorems 3.1 & 3.3, see Section 1.2.1 of the report).

"""

import time
import numpy as np
from dataclasses import dataclass
from ortools.graph.python import min_cost_flow


# OR-Tools requires integer arc costs → scale float distances by this factor
COST_SCALE = 1_000_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class A1Result:
    """
    Output of one A1 run.

    Attributes
    ----------
    centroids  : (k, d) array  – best centroids found across all restarts
    labels     : (n,)   array  – best point-to-cluster assignment
    cost       : float         – best total cost (sum of squared L2 distances)
    n_iters    : list[int]     – number of iterations per restart
    solve_time : float         – total wall-clock time in seconds
    """
    centroids:  np.ndarray
    labels:     np.ndarray
    cost:       float
    n_iters:    list
    solve_time: float


# ---------------------------------------------------------------------------
# Step 1 – Constrained assignment via Minimum Cost Flow
# ---------------------------------------------------------------------------

def _constrained_assignment(X, centroids, m):
    """
    Solve the constrained assignment sub-problem (eq. 11) as a MCF.

    Graph structure (Section 1.2.2 of the report)
    -----------------------------------------------
    Nodes  0 .. n-1   : data points      supply = +1
    Nodes  n .. n+k-1 : centroids        supply = -m
    Node   n+k        : artificial node  supply = km - n

    Arcs (i,    n+j)  capacity 1, cost = int(||xi - cj||^2 * COST_SCALE)
    Arcs (n+k,  n+j)  capacity m, cost = 0   (absorb unused slots)

    The artificial node is necessary to balance the graph: total point supply
    is n while total centroid demand is km >= n, so km-n fictitious
    zero-cost assignments fill the unused slots.

    Thanks to total unimodularity of the MCF constraint matrix (Proposition
    3.1 of [1]), the LP relaxation automatically yields integer solutions
    aij in {0,1} — no MIP solver needed.

    Parameters
    ----------
    X         : (n, d) float array – data points
    centroids : (k, d) float array – current centroid positions
    m         : int – capacity limit per centroid (>= ceil(n/k))

    Returns
    -------
    labels : (n,) int array  –  labels[i] = index of cluster assigned to xi
    """
    n, d = X.shape
    k    = len(centroids)
    art  = n + k          # index of the artificial node

    solver = min_cost_flow.SimpleMinCostFlow()

    # --- Arcs: data point i  →  centroid j ---
    arc_ids = {}          # (i, j) -> arc index, needed to read back the flow
    for i in range(n):
        for j in range(k):
            dist_sq = float(np.sum((X[i] - centroids[j]) ** 2))
            cost    = int(dist_sq * COST_SCALE)
            arc_id  = solver.add_arc_with_capacity_and_unit_cost(
                          i, n + j, 1, cost)
            arc_ids[(i, j)] = arc_id

    # --- Arcs: artificial node  →  centroid j  (cost 0) ---
    for j in range(k):
        solver.add_arc_with_capacity_and_unit_cost(art, n + j, m, 0)

    # --- Node supplies ---
    for i in range(n):
        solver.set_node_supply(i, 1)           # each point sends 1 unit
    for j in range(k):
        solver.set_node_supply(n + j, -m)      # each centroid absorbs m units
    solver.set_node_supply(art, k * m - n)     # artificial covers the slack

    # --- Solve ---
    status = solver.solve()
    if status != solver.OPTIMAL:
        raise RuntimeError(
            f"MCF did not reach optimality (status={status}). "
            "Verify that m >= ceil(n/k) and all supplies are correct."
        )

    # --- Extract labels from the integer solution ---
    labels = np.full(n, -1, dtype=int)
    for (i, j), arc_id in arc_ids.items():
        if solver.flow(arc_id) == 1:
            labels[i] = j

    assert (labels >= 0).all(), "Unreachable: some points were not assigned."
    return labels


# ---------------------------------------------------------------------------
# Step 2 – Centroid update
# ---------------------------------------------------------------------------

def _update_centroids(X, labels, k, centroids_prev):
    """
    Update each centroid as the mean of its assigned points.

    This is the exact minimiser of sum_i ||xi - cj||^2 at fixed assignments:
    the sub-problem is strictly convex with a unique global minimum (eq. 9/10,
    Theorems 3.1 & 3.3).  If a cluster is empty the centroid is left unchanged
    (centroids_prev[j] is kept), so the update step never increases the cost.

    Parameters
    ----------
    X              : (n, d) float array
    labels         : (n,)   int array
    k              : int
    centroids_prev : (k, d) float array – centroids from the previous iteration
                     (used as fallback for empty clusters)

    Returns
    -------
    new_centroids : (k, d) float array
    """
    _, d = X.shape
    new_centroids = centroids_prev.copy()   # keeps empty-cluster centroids

    for j in range(k):
        mask = (labels == j)
        if mask.any():
            new_centroids[j] = X[mask].mean(axis=0)

    return new_centroids


# ---------------------------------------------------------------------------
# Step 3 – Full A1 algorithm with r random restarts
# ---------------------------------------------------------------------------

def run_a1(X, k, m, r=10, eps=1e-6):
    """
    Modified k-means with capacity constraint (Algorithm A1).

    Alternates between:
      1. Constrained assignment via MCF   (optimal at fixed centroids, eq. 11)
      2. Centroid update via mean         (optimal at fixed assignments, eq. 9)
    until max centroid displacement < eps (convergence criterion, eq. 23).
    Repeated r times from independent random initialisations; the best
    solution (lowest total cost) is returned.

    Both steps guarantee a monotone decrease of the objective, and since the
    number of feasible assignments is finite, convergence to a local minimum
    is guaranteed [1, 8].

    Parameters
    ----------
    X   : (n, d) float array
    k   : int   – number of clusters
    m   : int   – capacity limit per centroid (>= ceil(n/k))
    r   : int   – number of random restarts  (default: 10)
    eps : float – convergence threshold on max centroid displacement (eq. 23)

    Returns
    -------
    A1Result with best centroids, labels, cost, per-restart iter counts
    and total wall-clock time.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    best_cost      = np.inf
    best_centroids = None
    best_labels    = None
    n_iters        = []

    t_start = time.perf_counter()

    for _ in range(r):

        # Initialisation: k distinct random data points as centroids
        init_idx  = np.random.choice(n, k, replace=False)
        centroids = X[init_idx].copy()
        iters     = 0

        while True:
            centroids_old = centroids.copy()

            # Step 1 – constrained assignment (MCF)
            labels = _constrained_assignment(X, centroids, m)

            # Step 2 – centroid update (mean; empty clusters keep old centroid)
            centroids = _update_centroids(X, labels, k, centroids_old)

            iters += 1

            # Convergence: max displacement across all centroids (eq. 23)
            max_shift = float(
                np.max(np.linalg.norm(centroids - centroids_old, axis=1))
            )
            if max_shift < eps:
                break

        n_iters.append(iters)

        # Total cost: sum of squared L2 distances (eq. 22) — vectorised
        cost = float(
            np.sum((X - centroids[labels]) ** 2)
        )

        if cost < best_cost:
            best_cost      = cost
            best_centroids = centroids.copy()
            best_labels    = labels.copy()

    solve_time = time.perf_counter() - t_start

    return A1Result(
        centroids  = best_centroids,
        labels     = best_labels,
        cost       = best_cost,
        n_iters    = n_iters,
        solve_time = solve_time,
    )
"""
a1_algorithm.py
---------------
A1: Modified k-means with capacity constraint.

Constrained assignment solved via Minimum Cost Flow (OR-Tools).
Centroid update: arithmetic mean of assigned points (exact minimiser
of the L2 objective at fixed assignments, see Theorems 3.1 & 3.3).

"""

import math
import numpy as np
from ortools.graph.python import min_cost_flow

# OR-Tools requires integer arc costs → scale float distances by this factor
COST_SCALE = 1_000_000


# ---------------------------------------------------------------------------
# Step 1 – Constrained assignment via Minimum Cost Flow
# ---------------------------------------------------------------------------

def _constrained_assignment(X, centroids, m):
    """
    Solve the constrained assignment sub-problem (eq. 11) as a MCF.

    Graph structure
    ---------------
    Nodes  0 .. n-1   : data points       supply = +1
    Nodes  n .. n+k-1 : centroids         supply = -m
    Node   n+k        : artificial node   supply = km - n

    Arcs (i,  n+j)  capacity 1, cost = int(||xi - cj||^2 * COST_SCALE)
    Arcs (n+k, n+j) capacity m, cost = 0

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

    # --- Arcs: artificial node  →  centroid j  (cost 0, absorb unused slots) ---
    for j in range(k):
        solver.add_arc_with_capacity_and_unit_cost(art, n + j, m, 0)

    # --- Node supplies ---
    for i in range(n):
        solver.set_node_supply(i, 1)          # each point sends 1 unit
    for j in range(k):
        solver.set_node_supply(n + j, -m)     # each centroid absorbs m units
    solver.set_node_supply(art, k * m - n)    # artificial covers the slack

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

def _update_centroids(X, labels, k):
    """
    Update each centroid as the mean of its assigned points.

    This is the exact minimiser of sum_i ||xi - cj||^2 at fixed assignments
    (strictly convex sub-problem, unique global minimum – eq. 9/10).
    If a cluster is empty the centroid is left unchanged (returned as zeros;
    the caller must restore the previous centroid).

    Parameters
    ----------
    X      : (n, d) float array
    labels : (n,)   int array
    k      : int

    Returns
    -------
    new_centroids : (k, d) float array
    """
    n, d = X.shape
    new_centroids = np.zeros((k, d), dtype=float)

    for j in range(k):
        mask = (labels == j)
        if mask.any():
            new_centroids[j] = X[mask].mean(axis=0)

    return new_centroids


# ---------------------------------------------------------------------------
# Step 3 – Full A1 algorithm with r random restarts
# ---------------------------------------------------------------------------

def run_a1(X, k, m, r=10, eps=1e-6, return_iters=False):
    """
    Modified k-means with capacity constraint (Algorithm A1).

    Alternates between:
      - constrained assignment via MCF  (optimal at fixed centroids)
      - centroid update via mean        (optimal at fixed assignments)
    until max centroid displacement < eps.  Repeated r times from different
    random initialisations; the best solution (lowest total cost) is returned.

    Parameters
    ----------
    X            : (n, d) float array
    k            : int   – number of clusters
    m            : int   – capacity limit per centroid (>= ceil(n/k))
    r            : int   – number of random restarts
    eps          : float – convergence threshold on max centroid displacement
    return_iters : bool  – if True, also return per-restart iteration counts

    Returns
    -------
    best_centroids : (k, d) float array
    best_labels    : (n,)   int array
    best_cost      : float
    iters_list     : list[int]  (only if return_iters=True)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    best_cost      = np.inf
    best_centroids = None
    best_labels    = None
    iters_list     = []

    for _ in range(r):

        # Initialisation: k distinct random data points as centroids
        init_idx  = np.random.choice(n, k, replace=False)
        centroids = X[init_idx].copy()
        n_iters   = 0

        while True:
            centroids_old = centroids.copy()

            # Step 1 – constrained assignment (MCF)
            labels = _constrained_assignment(X, centroids, m)

            # Step 2 – centroid update (mean); keep old if cluster is empty
            new_centroids = _update_centroids(X, labels, k)
            for j in range(k):
                if (labels == j).any():
                    centroids[j] = new_centroids[j]

            n_iters += 1

            # Convergence: max displacement across all centroids (eq. 23)
            max_shift = float(
                np.max(np.linalg.norm(centroids - centroids_old, axis=1))
            )
            if max_shift < eps:
                break

        iters_list.append(n_iters)

        # Total cost: sum of squared L2 distances (eq. 22)
        cost = float(sum(
            np.sum((X[i] - centroids[labels[i]]) ** 2)
            for i in range(n)
        ))

        if cost < best_cost:
            best_cost      = cost
            best_centroids = centroids.copy()
            best_labels    = labels.copy()

    if return_iters:
        return best_centroids, best_labels, best_cost, iters_list
    return best_centroids, best_labels, best_cost
if __name__ == "__main__":
    import math
    np.random.seed(0)
    X = np.random.randn(20, 2)
    k = 3
    m = math.ceil(20 / k)
    print(f"Test: n=20, k={k}, m={m}")
    centroids, labels, cost = run_a1(X, k, m, r=5)
    print("Cost:", round(cost, 4))
    print("Labels:", labels)
    # Verifica vincolo capacità: nessun cluster deve avere più di m punti
    for j in range(k):
        count = (labels == j).sum()
        print(f"  Cluster {j}: {count} punti (max consentito: {m})")
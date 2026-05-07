import numpy as np
import math
from algorithm2 import run_a2

np.random.seed(0)
X = np.random.randn(10, 2)
k = 2
m = math.ceil(10 / k)

centroids, labels, cost, gap, elapsed = run_a2(X, k, m)
print(f"Cost: {cost:.4f}")
print(f"Gap:  {gap:.6f}")
print(f"Time: {elapsed:.2f}s")
print(f"Labels: {labels}")
for j in range(k):
    print(f"  Cluster {j}: {(labels==j).sum()} punti (max {m})")
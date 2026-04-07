# Optimization
Project for Optimization course
# Constrained Clustering in the L2 Norm: Heuristic and Exact Methods

**Course:** Optimization and Learning  
**University:** Università di Pisa — Master's in Data Science & Business Informatics  
**Academic Year:** 2025/26  
**Authors:** Federica Braghin (698156), Nicholas Vannucci (691095)  
**Supervisor:** Prof. Antonio Frangioni  

---

## Problem Description

**(P)** is the Constrained Clustering Problem in the L2 norm: find k centroids and assign 
each of the n input points to exactly one centroid, minimising the total sum of squared 
L2 distances, with the constraint that no centroid is assigned more than m ≥ ⌈n/k⌉ points.

---

## Approaches

### A1 — Heuristic: Modified K-Means with Capacity Constraint
A modified version of the classic k-means algorithm that handles the capacity constraint 
via a **Minimum Cost Flow (MCF)** problem solved at each iteration using OR-Tools.  
Multiple random initializations are used to reduce sensitivity to the starting centroids.

### A2 — Exact: Mixed-Integer Quadratic Program (MIQP)
The problem is formulated as a MIQP and solved with **Gurobi**.  
Guarantees the global optimum by exploiting the convexity of the quadratic objective, 
but becomes computationally expensive as n and k grow.

---

## Technologies
- Python
- OR-Tools (MCF solver)
- Gurobi (MIQP solver)
- Jupyter Notebook

---

## Repository Structure
- `teoria_progetto66.docx` — theoretical notes
- Code and experiments (coming soon)

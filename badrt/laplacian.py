"""Interaction mesh: Delaunay adjacency + weighted Laplacian operator."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def delaunay_adjacency(vertices):
    """Undirected neighbour lists from the 3D Delaunay tetrahedra (6 edges per tet)."""
    tets = Delaunay(vertices, qhull_options="QJ").simplices
    n = len(vertices)
    pairs = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[a].append(int(b)); adj[b].append(int(a))
    return adj


def laplacian_matrix(vertices, adj, kappa=30.0, uniform=False, eps=1e-6):
    """L with L_ii = 1, L_ij = -w_ij, w_ij = softmax_j(-kappa * ||v_i - v_j||)."""
    n = len(vertices)
    L = np.eye(n)
    for i, nb in enumerate(adj):
        if not nb:
            continue
        if uniform:
            w = np.full(len(nb), 1.0 / len(nb))
        else:
            w = np.exp(-kappa * np.linalg.norm(vertices[nb] - vertices[i], axis=1))
            s = w.sum()
            if s < eps:
                continue
            w = w / s
        L[i, nb] = -w
    return L

"""
length.py
=========

Compute the crack length from a 1-pixel-wide skeleton.

The skeleton is treated as an undirected graph whose nodes are foreground
pixels. Length is measured by walking the graph and summing the distance
between connected neighbours:

    horizontal / vertical step  = 1
    diagonal step               = sqrt(2)

If the skeleton has multiple disconnected components (e.g. a fragmented
crack), each component is measured separately and the values are summed /
reported.
"""

import numpy as np
from collections import deque

# Step cost for 8-connected neighbours
_STEP_COST = {
    (1, 0): 1.0,
    (-1, 0): 1.0,
    (0, 1): 1.0,
    (0, -1): 1.0,
    (1, 1): np.sqrt(2),
    (1, -1): np.sqrt(2),
    (-1, 1): np.sqrt(2),
    (-1, -1): np.sqrt(2),
}


def _neighbours(p: tuple) -> list:
    """Return the 8-connected neighbour offsets."""
    return [
        (p[0] + dx, p[1] + dy)
        for (dx, dy) in _STEP_COST.keys()
    ]


def _connected_components(skeleton: np.ndarray) -> list:
    """
    Split skeleton foreground pixels into connected components (8-connectivity).

    Returns
    -------
    list of arrays of (y, x) coordinates for each component.
    """
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if skeleton[y, x] and not visited[y, x]:
                # BFS from this pixel
                comp = []
                queue = deque([(y, x)])
                visited[y, x] = True
                while queue:
                    cy, cx = queue.popleft()
                    comp.append((cy, cx))
                    for ny, nx in _neighbours((cy, cx)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if skeleton[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                components.append(comp)

    return components


def _component_length(comp: list) -> float:
    """
    Compute the total path length of a single skeleton component by summing
    all unique edges in its graph.

    Each edge connects two 8-connected foreground pixels with a step cost of
    1 (horizontal/vertical) or sqrt(2) (diagonal). Summing all edges gives the
    full crack extent, which correctly accounts for branches compared to a
    single longest-path walk.
    """
    points = set(comp)
    if not points:
        return 0.0

    # Sum each undirected edge exactly once.
    total = 0.0
    seen_edges = set()
    for p in points:
        for q in _neighbours(p):
            if q in points:
                # canonicalize undirected edge
                key = (p, q) if p < q else (q, p)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                dx = abs(p[0] - q[0])
                dy = abs(p[1] - q[1])
                total += _STEP_COST[(dx, dy)]
    return total


def compute_length(skeleton: np.ndarray) -> dict:
    """
    Compute the total crack length from a skeleton.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean skeleton array, shape (H, W).

    Returns
    -------
    dict
        {
            'length_px': total length in pixels,
            'components': number of connected components,
            'lengths_px': list of per-component lengths,
        }
    """
    skeleton = np.asarray(skeleton)

    if skeleton.ndim != 2:
        raise ValueError(f"Expected a 2D skeleton, got {skeleton.ndim}D")

    skeleton = skeleton.astype(bool)

    if not skeleton.any():
        return {
            "length_px": 0.0,
            "components": 0,
            "lengths_px": [],
        }

    components = _connected_components(skeleton)
    lengths = [_component_length(c) for c in components]

    return {
        "length_px": float(sum(lengths)),
        "components": len(lengths),
        "lengths_px": [float(l) for l in lengths],
    }

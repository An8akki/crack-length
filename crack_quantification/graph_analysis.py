"""Graph measurements for a thinned crack centreline.

The skeleton is a network, not necessarily a single curve. This module keeps
that distinction explicit: network length counts each centreline edge once,
main path is the longest geodesic in a component, and branch lengths report
compressed endpoint/junction segments.
"""

import heapq
import math

import numpy as np

from .length import _STEP_COST, _connected_components, _neighbours


def _pixel_graph(skeleton):
    points = {tuple(p) for p in np.argwhere(np.asarray(skeleton, dtype=bool))}
    graph = {p: {} for p in points}
    for point in points:
        for neighbour in _neighbours(point):
            if neighbour in points:
                graph[point][neighbour] = _STEP_COST[
                    (neighbour[0] - point[0], neighbour[1] - point[1])
                ]
    return graph


def _dijkstra(graph, start):
    distances = {start: 0.0}
    parents = {start: None}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in graph[node].items():
            candidate = distance + weight
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distances, parents


def _path(graph, start, end):
    distances, parents = _dijkstra(graph, start)
    if end not in distances:
        return [], 0.0
    nodes = []
    current = end
    while current is not None:
        nodes.append(current)
        current = parents[current]
    nodes.reverse()
    return nodes, distances[end]


def _component_graph(graph, component):
    component = set(component)
    return {point: {node: weight for node, weight in graph[point].items()
                    if node in component} for point in component}


def _mst_length(graph):
    """Return MST length and edges using Kruskal's algorithm."""
    parent = {node: node for node in graph}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edges = sorted(
        (weight, node, neighbour)
        for node, neighbours in graph.items()
        for neighbour, weight in neighbours.items()
        if node < neighbour
    )
    total = 0.0
    selected_edges = []
    for weight, node, neighbour in edges:
        root_a, root_b = find(node), find(neighbour)
        if root_a != root_b:
            parent[root_a] = root_b
            total += weight
            selected_edges.append({"start": node, "end": neighbour,
                                   "length_px": float(weight)})
    return float(total), selected_edges


def _compress_segments(graph):
    junctions = {node for node, neighbours in graph.items()
                 if len(neighbours) != 2}
    visited = set()
    segments = []
    for start in junctions:
        for neighbour, first_weight in graph[start].items():
            edge = tuple(sorted((start, neighbour)))
            if edge in visited:
                continue
            visited.add(edge)
            previous, current = start, neighbour
            length = first_weight
            path = [start, current]
            while current not in junctions:
                candidates = [node for node in graph[current] if node != previous]
                if not candidates:
                    break
                next_node = candidates[0]
                visited.add(tuple(sorted((current, next_node))))
                length += graph[current][next_node]
                previous, current = current, next_node
                path.append(current)
            segments.append({"start": start, "end": current,
                             "length_px": float(length), "path": path})
    return segments


def analyze_skeleton(skeleton):
    """Return topology-aware measurements for a boolean skeleton."""
    skeleton = np.asarray(skeleton, dtype=bool)
    graph = _pixel_graph(skeleton)
    if not graph:
        return {"network_length_px": 0.0, "main_path_px": 0.0,
                "mst_length_px": 0.0, "components": [],
                "branch_lengths_px": [], "endpoint_count": 0,
            "junction_count": 0, "main_path": [], "mst_edges": []}

    components = _connected_components(skeleton)
    component_reports = []
    all_segments = []
    main_path = []
    main_length = 0.0
    network_length = 0.0
    mst_length = 0.0

    for component in components:
        local = _component_graph(graph, component)
        network = sum(sum(neighbours.values()) for neighbours in local.values()) / 2
        endpoints = [node for node, neighbours in local.items()
                     if len(neighbours) == 1]
        junctions = [node for node, neighbours in local.items()
                     if len(neighbours) > 2]
        segments = _compress_segments(local)
        all_segments.extend(segments)
        network_length += network
        component_mst, mst_edges = _mst_length(local)
        mst_length += component_mst

        candidates = endpoints or [next(iter(local))]
        best = (0.0, [])
        for start in candidates:
            distances, _ = _dijkstra(local, start)
            for end, distance in distances.items():
                if distance > best[0]:
                    best = (distance, _path(local, start, end)[0])
        if best[0] > main_length:
            main_length, main_path = best
        component_reports.append({
            "pixel_count": len(component),
            "network_length_px": float(network),
            "main_path_px": float(best[0]),
            "mst_length_px": float(component_mst),
            "mst_edges": mst_edges,
            "endpoint_count": len(endpoints),
            "junction_count": len(junctions),
            "segments": segments,
        })

    return {
        "network_length_px": float(network_length),
        "main_path_px": float(main_length),
        "mst_length_px": float(mst_length),
        "components": component_reports,
        "branch_lengths_px": [segment["length_px"] for segment in all_segments],
        "endpoint_count": sum(c["endpoint_count"] for c in component_reports),
        "junction_count": sum(c["junction_count"] for c in component_reports),
        "main_path": main_path,
        "mst_edges": [edge for component in component_reports
                       for edge in component["mst_edges"]],
    }
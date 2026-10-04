"""Unabhängiges Orakel für Bellman-Ford und das Dijkstra-Gegenstück: Floyd-Warshall über alle Paare (Entfernungen und "von der Quelle erreichbarer negativer Zyklus" = ein erreichbarer Knoten x mit D[x][x] < 0)
und ein Dijkstra mit linearer Suche statt Heap (jeder Knoten wird einmal festgelegt, spätere Verbesserungen sind Alarme). Zufallsgraphen mit Gleichständen, Nullkosten, negativen Kanten ohne und mit Zyklus."""

import numpy as np
import pytest

import bf_algorithm as alg
from bf_graph import from_arcs, route_cost

INF = float("inf")


def floyd(g):
    D = np.full((g.n, g.n), INF)
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            D[u, v] = min(D[u, v], w)
    np.fill_diagonal(D, np.minimum(np.diag(D), 0.0))
    for k in range(g.n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def linear_dijkstra(g, s):
    dist, done, alarms = [INF] * g.n, [False] * g.n, 0
    dist[s] = 0.0
    while True:
        open_ = [(dist[x], x) for x in range(g.n) if not done[x] and dist[x] < INF]
        if not open_:
            return dist, alarms
        d, u = min(open_)
        done[u] = True
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            if d + w < dist[v]:
                if done[v]:
                    alarms += 1
                else:
                    dist[v] = d + w


def random_graph(rng, it):
    n = int(rng.integers(1, 11))
    pot = rng.integers(0, 8, n)
    arcs = []
    for _ in range(int(rng.integers(0, 3 * n + 1))):
        u, v = (int(x) for x in rng.integers(0, n, 2))
        c = [int(rng.integers(0, 5)), int(rng.integers(1, 10)) + int(pot[u]) - int(pot[v]), int(rng.integers(-3, 8)), int(rng.integers(-1, 2)), float(rng.integers(-4, 9)) / 2][it % 5]
        arcs.append((u, v, float(c)))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True, clean=bool(it % 2))


@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_distances_and_cycle_flag_equal_floyd_warshall_for_every_variant_and_order(variant):
    rng = np.random.default_rng(3)
    cycles = 0
    for it in range(150):
        g = random_graph(rng, it)
        s = int(rng.integers(0, g.n))
        D = floyd(g)
        cyc = any(D[x, x] < 0 for x in range(g.n) if np.isfinite(D[s, x]))
        cycles += cyc
        for order in (alg.EDGE_ORDERS if variant != "queue" else ("as_given",)):
            for check in ((False, True) if variant != "queue" else (False,)):
                r = alg.bellman_ford(g, s, variant, order, seed=it, cycle_check=check)
                assert r.negative_cycle == cyc
                if cyc:
                    c = r.cycle
                    cost = sum(float(g.weight[g.arc(a, b)]) for a, b in zip(c[:-1], c[1:]))
                    assert c[0] == c[-1] and cost < 0 and cost == pytest.approx(r.cycle_cost) and all(np.isfinite(D[s, x]) for x in c)
                else:
                    assert [float(x) for x in r.dist] == pytest.approx([float(x) for x in D[s]])
                    for v in range(g.n):
                        route = r.route(v)
                        assert (route == []) if not np.isfinite(D[s, v]) else (route[0] == s and route[-1] == v and route_cost(g, route) == pytest.approx(D[s, v]))
    assert cycles > 10


def test_dijkstra_distances_and_alarms_equal_a_linear_scan_dijkstra():
    rng = np.random.default_rng(8)
    for it in range(200):
        g = random_graph(rng, it)
        s = int(rng.integers(0, g.n))
        dist, alarms = linear_dijkstra(g, s)
        dj = alg.dijkstra(g, s)
        assert list(dj.dist) == dist and len(dj.alarms) == alarms
        if not (g.weight < 0).any():
            assert alarms == 0 and list(dj.dist) == pytest.approx(list(floyd(g)[s]))


def test_hop_limited_equals_layered_brute_force():
    rng = np.random.default_rng(1)
    for it in range(60):
        g = random_graph(rng, it)
        s = int(rng.integers(0, g.n))
        cur = [INF] * g.n
        cur[s] = 0.0
        for k in range(g.n + 1):
            assert list(alg.hop_limited(g, s, k)) == pytest.approx(cur)
            nxt = list(cur)
            for u in range(g.n):
                if cur[u] < INF:
                    for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
                        nxt[v] = min(nxt[v], cur[u] + w)
            cur = nxt

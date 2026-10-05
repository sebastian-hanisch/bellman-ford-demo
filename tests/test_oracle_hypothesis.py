"""Property-basierte Ergänzung zum festen Orakeltest (test_oracle_bellman_ford.py): dasselbe Orakel (Floyd-Warshall über alle Paare, Dijkstra mit linearer Suche, Schicht für Schicht gerechnete
Hop-Grenze), aber mit Hypothesis erzeugten gerichteten Graphen (2-8 Knoten, negative Kanten mit und ohne Zyklus, Nullkosten, Parallelkanten, Schleifen) und automatisch verkleinerten Gegenbeispielen.
Deterministisch für die CI (derandomize, keine Beispieldatenbank)."""

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import bf_algorithm as alg  # noqa: E402
from bf_graph import from_arcs, route_cost  # noqa: E402

CI = settings(max_examples=100, deadline=None, derandomize=True, database=None, suppress_health_check=[HealthCheck.too_slow])

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


@st.composite
def graph_and_source(draw, allow_negative=True, floats=False):
    """(g, s): ganzzahlige oder halbzahlige Kosten (exakt in Gleitkomma, viele Gleichstände); `potential` liefert negative Kanten OHNE Zyklus, `free` meist mit Zyklus."""
    n = draw(st.integers(2, 8))
    node = st.integers(0, n - 1)
    kind = draw(st.sampled_from(("free", "potential", "nonneg"))) if allow_negative else "nonneg"
    pot = draw(st.lists(st.integers(0, 8), min_size=n, max_size=n))
    arcs = []
    for u, v in draw(st.lists(st.tuples(node, node), max_size=3 * n)):
        if kind == "free":
            w = draw(st.integers(-6, 14)) / 2
        elif kind == "potential":
            w = float(draw(st.integers(1, 9)) + pot[u] - pot[v])
        elif floats:
            w = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        else:
            w = float(draw(st.integers(0, 6)))
        arcs.append((u, v, w))
    g = from_arcs(n, arcs, np.zeros((n, 2)), directed=True, clean=draw(st.booleans()))
    return g, draw(node)


@CI
@given(inst=graph_and_source(), variant=st.sampled_from(alg.VARIANTS), seed=st.integers(0, 10 ** 6))
def test_distances_and_cycle_flag_equal_floyd_warshall_for_every_variant_and_order(inst, variant, seed):
    g, s = inst
    D = floyd(g)
    cyc = any(D[x, x] < 0 for x in range(g.n) if np.isfinite(D[s, x]))
    for order in (alg.EDGE_ORDERS if variant != "queue" else ("as_given",)):
        for check in ((False, True) if variant != "queue" else (False,)):
            r = alg.bellman_ford(g, s, variant, order, seed=seed, cycle_check=check)
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


@CI
@given(inst=graph_and_source(allow_negative=False, floats=True), variant=st.sampled_from(alg.VARIANTS))
def test_distances_equal_floyd_warshall_for_nonnegative_float_costs(inst, variant):
    g, s = inst
    D = floyd(g)
    r = alg.bellman_ford(g, s, variant)
    assert not r.negative_cycle
    assert [float(x) for x in r.dist] == pytest.approx([float(x) for x in D[s]], rel=1e-9, abs=1e-9)


@CI
@given(inst=graph_and_source())
def test_dijkstra_distances_and_alarms_equal_a_linear_scan_dijkstra(inst):
    g, s = inst
    dist, alarms = linear_dijkstra(g, s)
    dj = alg.dijkstra(g, s)
    assert list(dj.dist) == dist and len(dj.alarms) == alarms
    if not (g.weight < 0).any():
        assert alarms == 0 and list(dj.dist) == pytest.approx(list(floyd(g)[s]))


@CI
@given(inst=graph_and_source())
def test_hop_limited_equals_layered_brute_force(inst):
    g, s = inst
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

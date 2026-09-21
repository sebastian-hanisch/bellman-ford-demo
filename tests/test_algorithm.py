"""Bellman-Ford (alle Varianten und Kantenreihenfolgen) und das Dijkstra-Gegenstück gegen networkx und eine unabhängige Rechnung."""

import networkx as nx
import numpy as np
import pytest

import bf_algorithm as alg
from bf_graph import from_arcs, route_cost

INF = float("inf")


def _random_arcs(n, m, seed, kind):
    """kind: 'positive', 'potential' (negative Kanten, aber nie ein Zyklus: Kosten = c + p(u) - p(v)), 'free' (beliebige Kosten, meist mit Zyklen)."""
    rng = np.random.default_rng(seed)
    pot = rng.integers(0, 15, n)
    arcs = {}
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u == v or (u, v) in arcs:
            continue
        c = int(rng.integers(1, 10))
        if kind == "potential":
            c = c + int(pot[u]) - int(pot[v])
        elif kind == "free":
            c = int(rng.integers(-3, 10))
        arcs[(u, v)] = c
    return [(u, v, float(c)) for (u, v), c in arcs.items()]


def _graph(n, arcs):
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True)


def _nx(n, arcs):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v, c in arcs:
        G.add_edge(u, v, weight=c)
    return G


def _reference(n, arcs, s):
    """networkx: Entfernungen oder None bei erreichbarem negativem Zyklus."""
    try:
        return nx.single_source_bellman_ford_path_length(_nx(n, arcs), s)
    except nx.NetworkXUnbounded:
        return None


CASES = [(n, m, seed, kind) for kind in ("positive", "potential", "free") for n, m, seed in ((8, 20, 1), (15, 40, 2), (25, 60, 3), (30, 45, 4), (12, 12, 5))]


@pytest.mark.parametrize("variant", alg.VARIANTS)
@pytest.mark.parametrize("n,m,seed,kind", CASES)
def test_matches_networkx_for_every_variant(variant, n, m, seed, kind):
    arcs = _random_arcs(n, m, seed, kind)
    g = _graph(n, arcs)
    for s in range(0, n, max(1, n // 4)):
        ref = _reference(n, arcs, s)
        r = alg.bellman_ford(g, s, variant)
        assert r.negative_cycle == (ref is None), (variant, s)
        if ref is not None:
            for v in range(n):
                assert (r.dist[v] == INF) if v not in ref else r.dist[v] == pytest.approx(ref[v]), (variant, s, v)
                if v in ref:
                    route = r.route(v)
                    assert route[0] == s and route[-1] == v and route_cost(g, route) == pytest.approx(ref[v])


@pytest.mark.parametrize("variant", alg.VARIANTS)
@pytest.mark.parametrize("edge_order", alg.EDGE_ORDERS)
def test_every_edge_order_gives_the_same_answer(variant, edge_order):
    arcs = _random_arcs(20, 50, 11, "potential")
    g = _graph(20, arcs)
    ref = _reference(20, arcs, 0)
    r = alg.bellman_ford(g, 0, variant, edge_order, seed=3)
    assert not r.negative_cycle and all(r.dist[v] == pytest.approx(ref[v]) for v in ref)


@pytest.mark.parametrize("variant", alg.VARIANTS)
@pytest.mark.parametrize("seed", range(6))
def test_reported_cycle_is_a_real_negative_cycle(variant, seed):
    arcs = _random_arcs(15, 35, seed, "free")
    g = _graph(15, arcs)
    r = alg.bellman_ford(g, 0, variant)
    if not r.negative_cycle:
        assert _reference(15, arcs, 0) is not None and r.cycle == []
        return
    assert r.cycle[0] == r.cycle[-1] and len(r.cycle) >= 3                                        # geschlossen, mindestens zwei Kanten
    assert route_cost(g, r.cycle) == pytest.approx(r.cycle_cost) and r.cycle_cost < 0
    assert r.route(3) == []                                                                       # mit erreichbarem negativem Zyklus gibt es keine kürzeste Route
    # der Zyklus ist von der Quelle aus erreichbar
    reach = nx.descendants(_nx(15, arcs), 0) | {0}
    assert set(r.cycle) <= reach


def test_unreachable_negative_cycle_does_not_disturb():
    # 0 -> 1 -> 2 ist harmlos; der negative Zyklus 3 -> 4 -> 3 ist von 0 aus nicht erreichbar
    arcs = [(0, 1, 2.0), (1, 2, 3.0), (3, 4, -2.0), (4, 3, 1.0), (4, 0, 1.0)]
    g = _graph(5, arcs)
    for variant in alg.VARIANTS:
        r = alg.bellman_ford(g, 0, variant)
        assert not r.negative_cycle and list(r.dist[:3]) == [0.0, 2.0, 5.0] and r.dist[3] == INF
    assert alg.bellman_ford(g, 3, "early_stop").negative_cycle                               # von 3 aus schon


def test_zero_weight_cycles_and_single_node_and_unreachable_target():
    arcs = [(0, 1, 0.0), (1, 0, 0.0), (1, 2, 4.0)]
    g = _graph(4, arcs)
    for variant in alg.VARIANTS:
        r = alg.bellman_ford(g, 0, variant)
        assert not r.negative_cycle and list(r.dist[:3]) == [0.0, 0.0, 4.0] and r.route(3) == [] and r.route(0) == [0]
    solo = _graph(1, [])
    assert not alg.bellman_ford(solo, 0).negative_cycle


def test_negative_two_cycle_is_reported_with_its_cost():
    g = _graph(3, [(0, 1, 1.0), (1, 2, -3.0), (2, 1, 1.0)])
    r = alg.bellman_ford(g, 0, "early_stop")
    assert r.negative_cycle and sorted(set(r.cycle)) == [1, 2] and r.cycle_cost == pytest.approx(-2.0)


@pytest.mark.parametrize("kind", ("positive", "potential"))
def test_after_round_k_synchronous_dist_is_the_best_route_with_at_most_k_edges(kind):
    arcs = _random_arcs(14, 36, 8, kind)
    g = _graph(14, arcs)
    r = alg.bellman_ford(g, 0, "synchronous", trace=True)
    assert not r.negative_cycle
    for k, (dist, _, _) in enumerate(r.history, start=1):
        assert np.allclose(np.array(dist), alg.hop_limited(g, 0, k), equal_nan=False)
        # unabhängige Rechnung: alle Routen bis k Kanten durchprobieren (kleines n)
    best = {v: INF for v in range(14)}
    best[0] = 0.0
    for _ in range(3):
        nxt = dict(best)
        for u, v, c in arcs:
            if best[u] + c < nxt[v]:
                nxt[v] = best[u] + c
        best = nxt
    assert np.allclose(alg.hop_limited(g, 0, 3), [best[v] for v in range(14)])


def test_hop_limited_stops_improving_after_n_minus_1_rounds_and_early_stop_has_fewer_rounds_than_textbook():
    arcs = _random_arcs(12, 30, 5, "potential")
    g = _graph(12, arcs)
    ref = _reference(12, arcs, 0)
    assert np.allclose(alg.hop_limited(g, 0, 11), [ref.get(v, INF) for v in range(12)])
    textbook, early = alg.bellman_ford(g, 0, "textbook"), alg.bellman_ford(g, 0, "early_stop")
    assert textbook.rounds == 12 and early.rounds < textbook.rounds and textbook.counters["checks"] == 12 * g.m
    assert np.array_equal(textbook.dist, early.dist)
    assert early.counters["checks"] == early.rounds * g.m and early.improved_per_round[-1] == 0


def test_a_path_needs_as_many_rounds_as_edges_when_edges_come_against_the_direction():
    n = 8
    arcs = [(i, i + 1, 1.0) for i in range(n - 1)]
    g = _graph(n, arcs)
    far = alg.bellman_ford(g, 0, "early_stop", "far_first")
    near = alg.bellman_ford(g, 0, "early_stop", "near_first")
    assert near.rounds == 2 and far.rounds == n                                              # 1 Runde findet alles, +1 zur Bestätigung / eine Runde je Kante + Bestätigung
    assert np.array_equal(near.dist, far.dist)


def test_queue_variant_counts_and_detects_cycles_by_requeue():
    arcs = _random_arcs(20, 50, 2, "potential")
    r = alg.bellman_ford(_graph(20, arcs), 0, "queue")
    assert r.counters["pops"] >= 1 and r.counters["checks"] >= r.counters["improvements"] and not r.negative_cycle
    g = _graph(3, [(0, 1, 1.0), (1, 2, -3.0), (2, 1, 1.0)])
    q = alg.bellman_ford(g, 0, "queue")
    assert q.negative_cycle and q.cycle_cost < 0


# --- Dijkstra als Gegenstück -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_dijkstra_is_right_without_negative_edges_and_raises_no_alarm(seed):
    arcs = _random_arcs(25, 70, seed, "positive")
    g = _graph(25, arcs)
    d, b = alg.dijkstra(g, 0), alg.bellman_ford(g, 0)
    assert np.allclose(d.dist, b.dist) and d.alarms == []


def test_dijkstra_can_be_wrong_with_a_negative_edge_and_says_so():
    g = _graph(4, [(0, 1, 1.0), (0, 2, 4.0), (1, 3, 5.0), (2, 1, -4.0)])
    d, b = alg.dijkstra(g, 0), alg.bellman_ford(g, 0)
    assert d.dist[1] == 1.0 and b.dist[1] == 0.0 and d.dist[3] == 6.0 and b.dist[3] == 5.0
    assert d.alarms and d.alarms[0][0] == 1


def test_dijkstra_is_wrong_only_when_it_raised_an_alarm_and_is_wrong_sometimes():
    wrong = 0
    for seed in range(40):
        arcs = _random_arcs(12, 30, seed, "potential")
        g = _graph(12, arcs)
        d, b = alg.dijkstra(g, 0), alg.bellman_ford(g, 0)
        if not np.allclose(d.dist, b.dist):
            wrong += 1
            assert d.alarms, seed                                                                   # falsch ohne Alarm darf nicht vorkommen
    assert wrong > 0                                                                                # der Test ist nicht leer: mit negativen Kanten liegt Dijkstra manchmal falsch


@pytest.mark.parametrize("variant", ("textbook", "early_stop", "synchronous"))
@pytest.mark.parametrize("seed", range(8))
def test_predecessor_cycle_check_agrees_with_networkx_and_finds_the_cycle_early(variant, seed):
    arcs = _random_arcs(20, 50, seed, "free")
    g = _graph(20, arcs)
    ref = _reference(20, arcs, 0)
    r = alg.bellman_ford(g, 0, variant, cycle_check=True)
    assert r.negative_cycle == (ref is None)
    if r.negative_cycle:
        assert r.cycle[0] == r.cycle[-1] and route_cost(g, r.cycle) == pytest.approx(r.cycle_cost) and r.cycle_cost < 0
        assert r.counters["detected_round"] <= 20 and r.rounds <= 20
    else:
        assert all(r.dist[v] == pytest.approx(ref[v]) for v in ref)


def test_predecessor_check_is_much_cheaper_than_waiting_for_round_n_on_a_big_net():
    base = _random_arcs(200, 600, 3, "positive")
    arcs = {(u, v): c for u, v, c in base}
    arcs[(0, 7)], arcs[(7, 9)], arcs[(9, 0)] = 1.0, 1.0, -3.0
    g = _graph(200, [(u, v, c) for (u, v), c in arcs.items()])
    slow, fast = alg.bellman_ford(g, 0, "early_stop"), alg.bellman_ford(g, 0, "early_stop", cycle_check=True)
    assert slow.negative_cycle and fast.negative_cycle and slow.rounds == 200 and fast.rounds < 20 and fast.counters["checks"] * 10 < slow.counters["checks"]

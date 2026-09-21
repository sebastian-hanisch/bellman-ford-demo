"""Netze (Frachtnetz, Devisen, E-Lieferwagen, Zufallsnetz, Stadtnetz), Zielwahl, Kennzahlen, Verteilungen, Experimente."""

import math

import networkx as nx
import numpy as np
import pytest

import bf_algorithm as alg
import bf_constants as C
import bf_evaluation as ev
import bf_scenario as sc


def _nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


def _strong(g):
    return nx.number_strongly_connected_components(_nx(g))


# --- Netze -----------------------------------------------------------------------------------------------------------------------------------

def test_delivery_network_matches_its_definition():
    net = sc.delivery_network()
    g = net.graph
    assert g.n == 8 and g.m == len(sc.DELIVERY_LEGS) == 8 and g.directed and net.unit == "Euro"
    assert (g.names[net.source], g.names[net.target]) == ("Lager", "Kunde")
    negative = [(g.names[u], g.names[int(v)], float(w)) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u)) if w < 0]
    assert negative == [("Zentrum", "Umschlag Ost", -6.0)]


def test_fx_networks_differ_only_in_the_mispriced_rate_and_have_integer_costs():
    fair, bad = sc.fx_network(False), sc.fx_network(True)
    assert fair.graph.n == bad.graph.n == 6 and fair.graph.m == bad.graph.m == 30 and (fair.graph.weight == np.rint(fair.graph.weight)).all()
    diff = [(fair.graph.names[u], fair.graph.names[int(v)]) for u in range(6) for v, a, b in zip(fair.graph.out(u), fair.graph.out_weights(u), bad.graph.out_weights(u)) if a != b]
    assert diff == [("GBP", "CHF")]
    assert sc.fx_cost(1.0) == 0 and sc.fx_cost(2.0) < 0 < sc.fx_cost(0.5) and sc.fx_cost(2.0) == -sc.fx_cost(0.5)


def test_fx_cycle_only_with_the_mispriced_rate_and_it_is_a_real_arbitrage():
    assert not alg.bellman_ford(sc.fx_network(False).graph, 0).negative_cycle
    net = sc.fx_network(True)
    r = alg.bellman_ford(net.graph, 0)
    assert r.negative_cycle and set(net.graph.names[i] for i in r.cycle) == {"GBP", "CHF"}
    prod = 1.0
    for u, v in zip(r.cycle[:-1], r.cycle[1:]):
        prod *= net.rates[u][v]
    assert prod > 1.0 and math.log(prod) == pytest.approx(-r.cycle_cost / 1e4, abs=1e-3)      # negativer Zyklus = Kurse multiplizieren sich zu mehr als 1


@pytest.mark.parametrize("eta", (0, 30, 60, 90))
@pytest.mark.parametrize("seed", range(3))
def test_ev_network_never_has_a_negative_cycle_and_has_integer_costs(eta, seed):
    net = sc.ev_network(12, 40, eta, seed)
    g = net.graph
    assert (g.weight == np.rint(g.weight)).all() and _strong(g) == 1
    assert not alg.bellman_ford(g, 0).negative_cycle
    assert not alg.bellman_ford(g, g.n - 1, "queue").negative_cycle


def test_ev_network_heights_and_recuperation_behave_physically():
    flat, hilly = sc.ev_network(10, 0, 60, 1), sc.ev_network(10, 30, 60, 1)
    assert (flat.graph.weight >= 0).all() and (hilly.graph.weight < 0).any()
    h = np.array(hilly.heights)
    assert h.min() == pytest.approx(0.0) and h.max() == pytest.approx(30.0)
    lossless, lossy = sc.ev_network(10, 30, 90, 1).graph, sc.ev_network(10, 30, 0, 1).graph
    assert (lossless.weight < 0).sum() > (sc.ev_network(10, 30, 60, 1).graph.weight < 0).sum() > (lossy.weight < 0).sum() == 0


def test_random_network_is_strongly_connected_and_its_structure_does_not_depend_on_the_potential_span():
    graphs = [sc.build_random(300, 3.0, p, 4) for p in (0, 4, 20)]
    assert all(_strong(g) == 1 for g in graphs)
    assert all(np.array_equal(graphs[0].indices, g.indices) and np.array_equal(graphs[0].indptr, g.indptr) for g in graphs)
    assert (graphs[0].weight >= 1).all() and (graphs[0].weight <= 9).all() and (graphs[2].weight < 0).any()
    assert abs(graphs[0].m / graphs[0].n - 3.0) < 0.05


def test_bellman_ford_behaves_identically_for_every_potential_span():
    runs = []
    for pot in (0, 3, 8, 20):
        g = sc.build_random(300, 3.0, pot, 5)
        r = alg.bellman_ford(g, 0)
        assert not r.negative_cycle
        runs.append((r.rounds, r.counters["checks"], r.counters["improvements"], tuple(r.parent.tolist())))
    assert len(set(runs)) == 1                                                                   # die Potenziale heben sich in jedem Vergleich auf


def test_city_network_is_connected_with_positive_whole_number_costs():
    g = sc.city_network(10, 1.5, 1.0, 2).graph
    assert _strong(g) == 1 and (g.weight >= 1).all() and (g.weight == np.rint(g.weight)).all()
    net = sc.make_network("city", 10, seed=2)
    assert np.array_equal(net.graph.indices, g.indices)


def test_make_network_rejects_unknown_nets():
    with pytest.raises(ValueError):
        sc.make_network("ring")


# --- Zielwahl und Kennzahlen -----------------------------------------------------------------------------------------------------------------

def test_pick_target_uses_the_fixed_task_for_named_nets_and_the_worst_dijkstra_error_otherwise():
    small = sc.delivery_network()
    assert ev.pick_target(small) == small.target
    net = sc.make_network("ev")
    t = ev.pick_target(net)
    bf, dj = alg.bellman_ford(net.graph, net.source), alg.dijkstra(net.graph, net.source)
    err = dj.dist - bf.dist
    assert err[t] == pytest.approx(err.max()) and err[t] > 0
    city = sc.make_network("city")
    assert ev.pick_target(city) == city.target


@pytest.mark.parametrize("key", C.NETS)
@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_analysis_invariants_and_exactness_for_every_net_and_variant(key, variant):
    net = sc.make_network(key, side=8, nodes=150)
    a = ev.analyse(net, variant)
    m, g = a.metrics, net.graph
    ref = None
    try:
        ref = nx.single_source_bellman_ford_path_length(_nx(g), net.source)
    except nx.NetworkXUnbounded:
        pass
    assert m["cycle"] == (ref is None) == (key == "fx_arbitrage")
    if ref is not None:
        assert all(a.bf.dist[v] == pytest.approx(ref[v]) for v in ref)
        assert m["cost_bf"] == pytest.approx(ref[a.target]) and m["dj_wrong_nodes"] == int(sum(abs(a.dj.dist[v] - ref[v]) > 1e-9 for v in ref))
    assert m["textbook_checks"] == g.n * g.m and m["checks_dj"] == a.dj.counters["checks"] and ev.verdict(a) in ("cycle", "dijkstra_wrong", "dijkstra_ok_negative", "no_negative")
    if variant == "textbook":
        assert m["checks_bf"] == g.n * g.m                                                        # das Lehrbuch prüft immer n · m Kanten
    if key in ("city",):
        assert m["neg_edges"] == 0 and ev.verdict(a) == "no_negative" and m["cost_bf"] == m["cost_dj"]


def test_cycle_analysis_carries_the_early_detection_numbers_only_for_round_variants():
    net = sc.fx_network(True)
    a = ev.analyse(net, "early_stop")
    assert a.metrics["detected_round_check"] < a.metrics["rounds"] and a.metrics["checks_check"] < a.metrics["checks_bf"]
    assert "detected_round_check" not in ev.analyse(net, "queue").metrics


def test_unreachable_target_is_reported():
    g = sc.build_random(50, 2.0, 0, 1)
    from bf_graph import from_arcs
    g2 = from_arcs(4, [(0, 1, 1.0), (2, 3, 1.0)], np.zeros((4, 2)), directed=True)
    net = type(sc.random_network(50, 2.0, 0, 1))("random", g2, "Einheiten", "x", "y", 0, 3, False)
    a = ev.analyse(net, "early_stop", target=3)
    assert ev.verdict(a) == "unreachable"
    assert g.n == 50


def test_source_stats_fields_and_determinism():
    net = sc.make_network("random", nodes=200)
    a, b = ev.source_stats(net, k=10, seed=3), ev.source_stats(net, k=10, seed=3)
    assert a == b and a["n_sources"] == 10 and a["cycles"] == 0 and 0 <= a["share_sources_wrong"] <= 1 and a["factor"] > 1
    assert ev.source_stats(sc.make_network("city", side=8), k=5)["share_sources_wrong"] == 0.0


# --- Experimente -----------------------------------------------------------------------------------------------------------------------------

def test_effort_rows_order_the_variants_as_expected():
    rows = ev.effort_vs_size(sides=(8, 12), seeds=C.SWEEP_SEEDS[:2])
    for r in rows:
        assert r["textbook"] == r["n"] * r["m"] and r["textbook"] > r["early_random"] > r["early_stop"] > r["dijkstra"] and r["queue"] < 1.5 * r["dijkstra"]
        assert r["hop_depth"] >= r["rounds_random"] - 1 > 1


def test_edge_order_table_rows():
    rows = {r["order"]: r for r in ev.edge_order_table(side=10, seeds=C.SWEEP_SEEDS[:2], nodes=150)}
    assert list(rows) == list(alg.EDGE_ORDERS) and rows["near_first"]["city_rounds"] == 2.0 and rows["far_first"]["city_rounds"] > rows["random"]["city_rounds"]


def test_recuperation_and_potential_sweeps():
    rows = ev.recuperation_sweep(etas=(0, 90), hill=30, side=10, seeds=C.SWEEP_SEEDS[:2])
    assert rows[0]["neg_share"] == 0 and rows[0]["wrong_share"] == 0 and rows[1]["neg_share"] > 0 and all(r["cycles"] == 0 for r in rows)
    pots = ev.potential_sweep(pots=(0, 10), nodes=150, seeds=C.SWEEP_SEEDS[:2], starts=(0, 20))
    assert pots[0]["neg_share"] == 0 and pots[1]["neg_share"] > 0 and pots[0]["rounds"] == pots[1]["rounds"]


def test_cycle_detection_rows_and_injected_cycle():
    rows = ev.cycle_detection(ns=(100,), seeds=C.SWEEP_SEEDS[:2])
    r = rows[0]
    assert r["final"] == r["n"] * r["m"] and r["check"] * 10 < r["final"] and r["queue"] <= r["final"]
    g = ev.inject_cycle(sc.build_random(100, 3.0, 0, 1), 1)
    cyc = alg.bellman_ford(g, 0, "queue")
    assert cyc.negative_cycle and len(cyc.cycle) == 4 and cyc.cycle_cost == -1.0


def test_net_sweep_rows():
    rows = ev.net_sweep("hill", (0, 40), dict(side=10, hill=30, eta=60), seeds=C.SWEEP_SEEDS[:2])
    assert [r["value"] for r in rows] == [0, 40] and rows[0]["neg_share"] == 0 and rows[1]["neg_share"] > 0
    assert ev.random_factor(100, 3.0, seeds=C.SWEEP_SEEDS[:2]) > 1

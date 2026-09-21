"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Kantenprüfungen, Runden, Verbesserungen und ganzzahlige Kosten sind plattformfest;
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import numpy as np
import pytest

import bf_algorithm as alg
import bf_constants as C
import bf_evaluation as ev
import bf_scenario as sc

PRESET = {"delivery": "🚚 Frachtnetz", "fx_arbitrage": "💱 Devisen mit Kursfehler", "fx_fair": "💱 Devisen ohne Kursfehler", "ev": "🔋 E-Lieferwagen", "random": "🕸️ Zufallsnetz", "city": "🏙️ Stadtnetz"}


def _preset(key):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["hill"], p["eta"], p["reach"], p["spread"], p["nodes"], p["degree"], p["pot"], p["seed"])
    return p, net, ev.analyse(net, p["variant"], p["order"], p["seed"])


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_delivery_preset_numbers():
    _, net, a = _preset("delivery")
    m = a.metrics
    assert (m["cost_bf"], m["cost_dj"], m["rounds"], m["checks_bf"], m["checks_dj"], m["textbook_checks"]) == (16.0, 17.0, 3, 24, 8, 64) and ev.verdict(a) == "dijkstra_wrong"
    assert alg.bellman_ford(net.graph, net.source, "queue").counters["checks"] == 12 and alg.bellman_ford(net.graph, net.source, "synchronous").rounds == 8
    _has("delivery", "17 Euro", "kostet 16", "3 Runden", "24 Kantenprüfungen", "Dijkstra 8", "Lehrbuch 64")


def test_fx_arbitrage_preset_numbers():
    _, net, a = _preset("fx_arbitrage")
    m = a.metrics
    assert ev.verdict(a) == "cycle" and (m["rounds"], m["checks_bf"], m["detected_round_check"], m["checks_check"]) == (6, 180, 1, 30) and m["cycle_cost"] == -79.0
    prod = 1.0
    for u, v in zip(a.bf.cycle[:-1], a.bf.cycle[1:]):
        prod *= net.rates[u][v]
    assert prod - 1 == pytest.approx(0.008, abs=0.0005)                                          # "etwa 0.8 % Gewinn"
    _has("fx_arbitrage", "etwa 0.8 %", "nach 6 Runden (180 Kantenprüfungen)", "nach Runde 1 (30 Prüfungen)")


def test_fx_fair_preset_numbers():
    _, net, a = _preset("fx_fair")
    m = a.metrics
    assert m["neg_edges"] == 15 and m["m"] == 30 and ev.verdict(a) == "dijkstra_ok_negative" and m["cost_bf"] == m["cost_dj"]
    _has("fx_fair", "15 der 30 Umtausche")


def test_ev_preset_numbers():
    _, net, a = _preset("ev")
    m = a.metrics
    assert (m["neg_edges"], m["m"], m["n"], m["cost_bf"], m["cost_dj"], m["dj_wrong_nodes"]) == (69, 1488, 256, 97.0, 105.0, 62) and ev.verdict(a) == "dijkstra_wrong"
    _has("ev", "69 der 1 488", "97 Wh", "Dijkstra 105", "62 von 256")


def test_random_preset_numbers():
    _, net, a = _preset("random")
    m = a.metrics
    assert (m["neg_edges"], m["m"], m["cost_bf"], m["cost_dj"], m["dj_wrong_nodes"], m["n"], m["checks_bf"], m["checks_dj"]) == (142, 1200, 15.0, 20.0, 9, 400, 8400, 1200)
    _has("random", "142 von 1 200", "15, Dijkstra 20", "9 von 400", "8 400 Kantenprüfungen", "Dijkstra 1 200")


def test_city_preset_numbers():
    p, net, a = _preset("city")
    m = a.metrics
    assert (m["cost_bf"], m["cost_dj"], m["checks_bf"], m["checks_dj"], m["textbook_checks"], m["factor"]) == (3040.0, 3040.0, 4464, 1488, 380928, 3.0) and ev.verdict(a) == "no_negative"
    s = ev.source_stats(net, p["variant"], p["order"])
    assert s["factor"] == pytest.approx(12.6, abs=0.1) and s["share_sources_wrong"] == 0.0
    _has("city", "3 040 m", "4 464 Kantenprüfungen (3.0-fach)", "380 928", "20 Startknoten das 12.6-Fache")


# --- Sidebar-Hilfen ----------------------------------------------------------------------------------------------------------------------------

def test_hill_and_recuperation_help_numbers():
    hill = ev.net_sweep("hill", (0, 10, 20, 30, 40), dict(side=16, hill=30, eta=60))
    assert [r["wrong_share"] * 100 for r in hill] == pytest.approx([0, 0, 1.3, 12.8, 15.8], abs=0.1)
    rec = ev.recuperation_sweep()
    assert [r["wrong_share"] * 100 for r in rec] == pytest.approx([0, 0, 0.2, 12.8, 20.4, 24.2], abs=0.1)
    assert all(r["cycles"] == 0 for r in rec) and [r["seeds_with_alarm"] for r in rec] == [0, 0, 1, 4, 5, 5]


def test_potential_help_numbers_and_the_rounds_do_not_change():
    rows = {r["pot"]: r for r in ev.potential_sweep()}
    assert [rows[p]["neg_share"] * 100 for p in (2, 8, 20)] == pytest.approx([1.1, 11.1, 27.1], abs=0.1)
    assert [rows[p]["wrong_share"] * 100 for p in (2, 8, 20)] == pytest.approx([0.2, 6.3, 23.3], abs=0.1)
    assert len({rows[p]["rounds"] for p in rows}) == 1 and rows[0]["neg_share"] == 0 and rows[0]["wrong_share"] == 0            # "Runden von Bellman-Ford bleiben bei jeder Spanne dieselben"


def test_random_net_size_and_degree_help_numbers():
    assert [ev.random_factor(n, 3.0) for n in (100, 400, 1000)] == pytest.approx([6.4, 8.2, 8.8], abs=0.1)
    assert [ev.random_factor(400, d) for d in (2.0, 4.0, 6.0)] == pytest.approx([10.4, 6.8, 6.0], abs=0.1)


# --- Experimente und die Tabelle "Wo die Annahmen enden" ----------------------------------------------------------------------------------------

def test_effort_claims():
    rows = ev.effort_vs_size()
    big = rows[-1]
    assert big["n"] == 1600 and big["textbook"] / big["dijkstra"] == pytest.approx(1600.0)                     # "im 40 × 40-Stadtnetz das 1 600-Fache"
    assert 9.0 <= big["early_stop"] / big["dijkstra"] <= 10.0                                                  # "gut das 9-Fache"
    assert big["early_stop"] / big["dijkstra"] == pytest.approx(9.4, abs=0.05) and big["queue"] / big["dijkstra"] == pytest.approx(1.07, abs=0.02)      # README: 9.4-fach, 1.07-fach
    assert 1.05 <= big["queue"] / big["dijkstra"] <= 1.10                                                      # "knapp das 1.1-Fache"
    assert big["early_random"] / big["dijkstra"] > 2.5 * big["early_stop"] / big["dijkstra"] and big["hop_depth"] == pytest.approx(50.8, abs=0.1)
    assert [r["rounds_early"] for r in rows] == pytest.approx([3.0, 4.8, 6.8, 9.4], abs=0.05) and [r["rounds_random"] for r in rows] == pytest.approx([8.0, 15.0, 22.2, 28.6], abs=0.05)
    assert all(r["textbook"] == r["n"] * r["m"] for r in rows)


def test_edge_order_claims():
    rows = {r["order"]: r for r in ev.edge_order_table()}
    assert rows["near_first"]["city_rounds"] == 2.0 and rows["near_first"]["random_rounds"] == 2.0
    assert rows["as_given"]["city_rounds"] == pytest.approx(4.8, abs=0.05) and rows["random"]["city_rounds"] == pytest.approx(15.0, abs=0.05) and rows["far_first"]["city_rounds"] == pytest.approx(25.4, abs=0.05)
    assert rows["as_given"]["random_rounds"] == pytest.approx(8.2, abs=0.05) and rows["random"]["random_rounds"] == pytest.approx(8.4, abs=0.05)
    assert rows["as_given"]["random_rounds"] >= 0.9 * rows["random"]["random_rounds"]                          # im Zufallsnetz kaum besser als zufällig
    assert rows["as_given"]["city_rounds"] < 0.4 * rows["random"]["city_rounds"]                               # im Stadtnetz deutlich besser (Knotennummern folgen dem Raster)


def test_cycle_detection_claims():
    rows = {int(r["n"]): r for r in ev.cycle_detection()}
    big = rows[400]
    assert big["final"] == 400 * big["m"] == 481200.0 and big["final"] / big["dijkstra"] == pytest.approx(400.0)     # "immer n · m Prüfungen"
    assert big["check"] == pytest.approx(6015.0, abs=1) and big["check_rounds"] == pytest.approx(5.0, abs=0.05) and 75 <= big["final"] / big["check"] <= 85        # "80-mal weniger"
    assert big["queue"] / big["final"] > 0.9                                                                   # "die Warteschlange spart kaum etwas"
    assert all(r["check"] * 20 < r["final"] and r["check_rounds"] <= 6 for r in rows.values())


def test_lehrbuch_costs_exactly_n_times_m_everywhere():
    for key in ("ev", "random", "city"):
        net = sc.make_network(key, side=8, nodes=150)
        assert alg.bellman_ford(net.graph, net.source, "textbook").counters["checks"] == net.graph.n * net.graph.m


def test_dijkstra_is_wrong_exactly_where_it_raised_an_alarm():
    for key in ("delivery", "ev", "random"):
        net = sc.make_network(key)
        bf, dj = alg.bellman_ford(net.graph, net.source), alg.dijkstra(net.graph, net.source)
        assert (not np.allclose(bf.dist, dj.dist)) == bool(dj.alarms)

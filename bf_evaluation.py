"""Bellman-Ford und Dijkstra im Vergleich: Kennzahlen für ein Netz, Verteilung über Startknoten, Experimente (Aufwand gegen Größe, Kantenreihenfolge, wann Dijkstra danebenliegt, Kosten der Zyklus-Erkennung), Urteil für die App.

Der Aufwand wird in Kantenprüfungen gezählt (plattformfest); Laufzeiten stehen nur als Messwert in der App."""

import time
from dataclasses import dataclass

import numpy as np

import bf_algorithm as alg
import bf_constants as C
from bf_graph import from_arcs
from bf_scenario import build_random, city_network, ev_network, make_network, random_network


@dataclass(frozen=True)
class Analysis:
    net: object
    target: int
    bf: alg.BellmanFord
    dj: alg.Dijkstra
    metrics: dict
    seconds: dict


def pick_target(net):
    """Ziel: bei den kleinen Netzen die feste Aufgabe; sonst der Knoten, an dem Dijkstra am meisten danebenliegt (gibt es einen), sonst das vorgegebene Ziel."""
    if net.graph.names:
        return net.target
    bf = alg.bellman_ford(net.graph, net.source)
    if bf.negative_cycle:
        return net.target
    dj = alg.dijkstra(net.graph, net.source)
    diff = np.where(np.isfinite(dj.dist) & np.isfinite(bf.dist), dj.dist - bf.dist, 0.0)
    return int(np.argmax(diff)) if diff.max() > 1e-9 else net.target


def analyse(net, variant=C.DEFAULT_VARIANT, edge_order=C.DEFAULT_EDGE_ORDER, seed=C.DEFAULT_SEED, target=None):
    g = net.graph
    t = pick_target(net) if target is None else int(target)
    t0 = time.perf_counter()
    bf = alg.bellman_ford(g, net.source, variant, edge_order, trace=True, seed=seed)
    t_bf = time.perf_counter() - t0
    t0 = time.perf_counter()
    dj = alg.dijkstra(g, net.source)
    t_dj = time.perf_counter() - t0
    cyc = bf.negative_cycle
    reach = np.isfinite(bf.dist) if not cyc else np.isfinite(dj.dist)
    both = np.isfinite(bf.dist) & np.isfinite(dj.dist)
    wrong = int((both & ~np.isclose(bf.dist, dj.dist)).sum()) if not cyc else -1
    m = {"n": g.n, "m": g.m, "target": t, "cycle": cyc, "cycle_cost": bf.cycle_cost, "reachable": bool(np.isfinite(bf.dist[t])) if not cyc else True,
         "cost_bf": float(bf.dist[t]) if not cyc else float("nan"), "cost_dj": float(dj.dist[t]),
         "dj_wrong_target": bool(not cyc and np.isfinite(bf.dist[t]) and abs(bf.dist[t] - dj.dist[t]) > 1e-9),
         "dj_wrong_nodes": wrong, "dj_wrong_share": wrong / g.n if wrong >= 0 else float("nan"), "dj_alarms": len(dj.alarms),
         "rounds": bf.rounds, "productive_rounds": bf.counters.get("productive_rounds", 0), "checks_bf": bf.counters["checks"], "checks_dj": dj.counters["checks"],
         "factor": bf.counters["checks"] / max(dj.counters["checks"], 1), "textbook_checks": g.n * g.m, "improvements": bf.counters["improvements"],
         "neg_edges": int((g.weight < 0).sum()), "neg_share": float((g.weight < 0).mean()) if g.m else 0.0,
         "hops_bf": len(bf.route(t)) - 1 if bf.route(t) else 0, "hops_dj": len(dj.route(t)) - 1 if dj.route(t) else 0}
    if cyc and variant != "queue":
        quick = alg.bellman_ford(g, net.source, variant, edge_order, seed=seed, cycle_check=True)
        m["detected_round_check"], m["checks_check"] = quick.counters.get("detected_round", quick.rounds), quick.counters["checks"]
    return Analysis(net, t, bf, dj, m, {"bf": t_bf, "dj": t_dj})


def verdict(a):
    """Code für die App: cycle / unreachable / dijkstra_wrong / dijkstra_ok_negative / no_negative."""
    m = a.metrics
    if m["cycle"]:
        return "cycle"
    if not m["reachable"]:
        return "unreachable"
    if m["neg_edges"] == 0:
        return "no_negative"
    return "dijkstra_wrong" if m["dj_wrong_target"] else "dijkstra_ok_negative"


def source_stats(net, variant=C.DEFAULT_VARIANT, edge_order=C.DEFAULT_EDGE_ORDER, k=C.SOURCES, seed=0):
    """Über zufällige Startknoten: Anteil der Starts, bei denen Dijkstra mindestens einen Knoten falsch hat, Mittel des Anteils falscher Knoten, mittleres Verhältnis der Kantenprüfungen Bellman-Ford / Dijkstra, Runden."""
    g = net.graph
    rng = np.random.default_rng([int(seed), 1212])
    starts = [int(x) for x in rng.choice(g.n, min(k, g.n), replace=False)]
    any_wrong, wrong_share, factor, rounds, cycles = [], [], [], [], 0
    for s in starts:
        bf = alg.bellman_ford(g, s, variant, edge_order, seed=seed)
        if bf.negative_cycle:
            cycles += 1
            continue
        dj = alg.dijkstra(g, s)
        both = np.isfinite(bf.dist) & np.isfinite(dj.dist)
        bad = int((both & ~np.isclose(bf.dist, dj.dist)).sum())
        any_wrong.append(bad > 0)
        wrong_share.append(bad / g.n)
        factor.append(bf.counters["checks"] / max(dj.counters["checks"], 1))
        rounds.append(bf.rounds)
    nan = float("nan")
    return {"n_sources": len(starts), "cycles": cycles, "share_sources_wrong": float(np.mean(any_wrong)) if any_wrong else nan, "wrong_share": float(np.mean(wrong_share)) if wrong_share else nan,
            "factor": float(np.mean(factor)) if factor else nan, "rounds": float(np.mean(rounds)) if rounds else nan}


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def effort_vs_size(sides=(10, 20, 30, 40), seeds=C.SWEEP_SEEDS, reach=1.5):
    """Kantenprüfungen gegen die Netzgröße (Stadtnetz, Start unten links): Lehrbuch (immer n · m, siehe Test), früher Abbruch (in place, Kanten wie im Netz und zufällig), Warteschlange, Dijkstra; dazu die Zahl der Kanten je Route."""
    rows = []
    for side in sides:
        acc = {k: [] for k in ("n", "m", "textbook", "early_stop", "early_random", "queue", "dijkstra", "rounds_early", "rounds_random", "hop_depth")}
        for sd in seeds:
            g = city_network(side, reach, C.DEFAULT_SPREAD, sd).graph
            dj = alg.dijkstra(g, 0)
            early = alg.bellman_ford(g, 0, "early_stop")
            rnd = alg.bellman_ford(g, 0, "early_stop", "random", seed=sd)
            que = alg.bellman_ford(g, 0, "queue")
            syn = alg.bellman_ford(g, 0, "synchronous")
            for k, v in (("n", g.n), ("m", g.m), ("textbook", g.n * g.m), ("early_stop", early.counters["checks"]), ("early_random", rnd.counters["checks"]), ("queue", que.counters["checks"]),
                         ("dijkstra", dj.counters["checks"]), ("rounds_early", early.rounds), ("rounds_random", rnd.rounds), ("hop_depth", syn.rounds - 1)):
                acc[k].append(v)
        rows.append({"side": side, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def edge_order_table(side=20, seeds=C.SWEEP_SEEDS, nodes=400):
    """Runden und Kantenprüfungen bei frühem Abbruch je Kantenreihenfolge, auf dem Stadtnetz und dem Zufallsnetz (dessen Knotennummern sind zufällig, die des Stadtnetzes folgen dem Raster)."""
    rows = []
    for order in alg.EDGE_ORDERS:
        row = {"order": order}
        for key, make in (("city", lambda sd: city_network(side, 1.5, C.DEFAULT_SPREAD, sd)), ("random", lambda sd: random_network(nodes, C.DEFAULT_DEGREE, 0, sd))):
            rounds, checks = [], []
            for sd in seeds:
                net = make(sd)
                r = alg.bellman_ford(net.graph, net.source, "early_stop", order, seed=sd)
                rounds.append(r.rounds)
                checks.append(r.counters["checks"])
            row[key + "_rounds"], row[key + "_checks"] = float(np.mean(rounds)), float(np.mean(checks))
        rows.append(row)
    return rows


def recuperation_sweep(etas=(0, 20, 40, 60, 80, 90), hill=30, side=16, seeds=C.SWEEP_SEEDS):
    """E-Lieferwagen: Anteil negativer Kanten und Anteil der Knoten, bei denen Dijkstra danebenliegt, gegen die Rückgewinnung (Start unten links, fünf Netze)."""
    rows = []
    for eta in etas:
        neg, wrong, alarm, cyc = [], [], 0, 0
        for sd in seeds:
            g = ev_network(side, hill, eta, sd).graph
            bf, dj = alg.bellman_ford(g, 0), alg.dijkstra(g, 0)
            cyc += bf.negative_cycle
            neg.append(float((g.weight < 0).mean()))
            wrong.append(float((~np.isclose(bf.dist, dj.dist)).mean()))
            alarm += bool(dj.alarms)
        rows.append({"eta": eta, "neg_share": float(np.mean(neg)), "wrong_share": float(np.mean(wrong)), "seeds_with_alarm": alarm, "seeds": len(seeds), "cycles": cyc})
    return rows


def potential_sweep(pots=(0, 2, 4, 8, 12, 20), nodes=400, seeds=C.SWEEP_SEEDS, starts=(0, 50, 100, 150)):
    """Zufallsnetz mit Potenzialen: Anteil negativer Kanten, Anteil falscher Knoten bei Dijkstra, Anteil der Starts mit falschem Ergebnis, Runden (bleiben gleich: die Potenziale ändern nicht, welche Routen kürzeste sind)."""
    rows = []
    for pot in pots:
        neg, wrong, bad_start, rounds = [], [], [], []
        for sd in seeds:
            g = build_random(nodes, C.DEFAULT_DEGREE, pot, sd)
            neg.append(float((g.weight < 0).mean()))
            for s in starts:
                bf, dj = alg.bellman_ford(g, s), alg.dijkstra(g, s)
                bad = int((~np.isclose(bf.dist, dj.dist)).sum())
                wrong.append(bad / g.n)
                bad_start.append(bad > 0)
                rounds.append(bf.rounds)
        rows.append({"pot": pot, "neg_share": float(np.mean(neg)), "wrong_share": float(np.mean(wrong)), "starts_wrong": float(np.mean(bad_start)), "rounds": float(np.mean(rounds))})
    return rows


def inject_cycle(g, seed):
    """Ein negativer Dreieckszyklus (Summe -1) durch drei zufällige Knoten, zu den Kanten des Netzes hinzugefügt."""
    a, b, c = (int(x) for x in np.random.default_rng(seed).choice(g.n, 3, replace=False))
    arcs = {(u, int(v)): float(w) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u))}
    arcs[(a, b)], arcs[(b, c)], arcs[(c, a)] = 1.0, 1.0, -3.0
    return from_arcs(g.n, [(u, v, w) for (u, v), w in arcs.items()], g.xy, directed=True)


def cycle_detection(ns=(100, 200, 400, 800), seeds=C.SWEEP_SEEDS):
    """Kosten, einen negativen Zyklus zu finden (Zufallsnetz ohne negative Kanten, ein negativer Dreieckszyklus eingebaut): Kantenprüfungen bis zur Meldung bei "früher Abbruch" ohne und mit Vorgänger-Prüfung und bei der Warteschlange;
    dazu die Runden bis zur Meldung mit Prüfung."""
    rows = []
    for n in ns:
        acc = {k: [] for k in ("m", "final", "check", "check_rounds", "queue", "dijkstra")}
        for sd in seeds:
            g = inject_cycle(build_random(n, C.DEFAULT_DEGREE, 0, sd), sd)
            a, b, q = alg.bellman_ford(g, 0), alg.bellman_ford(g, 0, cycle_check=True), alg.bellman_ford(g, 0, "queue")
            assert a.negative_cycle and b.negative_cycle and q.negative_cycle
            for k, v in (("m", g.m), ("final", a.counters["checks"]), ("check", b.counters["checks"]), ("check_rounds", b.rounds), ("queue", q.counters["checks"]),
                         ("dijkstra", alg.dijkstra(g, 0).counters["checks"])):
                acc[k].append(v)
        rows.append({"n": n, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def net_sweep(parameter, values, base, seeds=C.SWEEP_SEEDS):
    """Ein Regler des E-Lieferwagen-Netzes durchgefahren (side, hill, eta wie in `base`): Anteil negativer Kanten und Anteil falscher Knoten bei Dijkstra (Start unten links, Mittel über die Sweep-Netze)."""
    rows = []
    for v in values:
        kw = {**base, parameter: v}
        neg, wrong = [], []
        for sd in seeds:
            g = ev_network(kw["side"], kw["hill"], kw["eta"], sd).graph
            bf, dj = alg.bellman_ford(g, 0), alg.dijkstra(g, 0)
            neg.append(float((g.weight < 0).mean()))
            wrong.append(float((~np.isclose(bf.dist, dj.dist)).mean()))
        rows.append({"value": v, "neg_share": float(np.mean(neg)), "wrong_share": float(np.mean(wrong))})
    return rows


def random_factor(nodes, degree, pot=C.DEFAULT_POT, seeds=C.SWEEP_SEEDS):
    """Zufallsnetz: Kantenprüfungen von Bellman-Ford (früher Abbruch, Kanten wie im Netz) geteilt durch die von Dijkstra, Start 0, Mittel über die Sweep-Netze."""
    f = []
    for sd in seeds:
        g = build_random(nodes, degree, pot, sd)
        f.append(alg.bellman_ford(g, 0).counters["checks"] / max(alg.dijkstra(g, 0).counters["checks"], 1))
    return float(np.mean(f))

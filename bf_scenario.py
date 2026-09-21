"""Die Netze der Demo: kleines Frachtnetz mit Rückvergütung, zwei Devisennetze (mit und ohne Kursfehler), E-Lieferwagen mit Rekuperation im hügeligen Stadtnetz, Zufallsnetz mit negativen Kanten
(über Potenziale, nie ein Zyklus) und das Stadtnetz ohne negative Kanten. Alle Graphen sind eigene Konstruktionen (Währungskurse erfunden), alle Kosten ganze Zahlen."""

import math
from dataclasses import dataclass

import numpy as np

import bf_constants as C
from bf_graph import Graph, from_arcs


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    unit: str                      # Einheit der Kosten
    title: str
    note: str
    source: int
    target: int
    geometric: bool = True         # Lage der Knoten ist eine Karte (Kanten zeichnen), sonst Knoten auf einem Kreis oder zufällig verteilt
    heights: tuple = ()            # E-Lieferwagen: Höhe je Kreuzung in Metern
    rates: tuple = ()              # Devisen: Kursmatrix (Zeile = von, Spalte = nach)


# --- Kleines Frachtnetz ------------------------------------------------------------------------------------------------------------------------

DELIVERY_STOPS = [("Lager", 0.0, 2.0), ("Umschlag Ost", 2.5, 4.0), ("Umschlag West", 2.5, 0.0), ("Zentrum", 5.0, 0.0), ("Grenzlager", 5.0, 4.0), ("Hafen", 7.5, 4.0), ("Terminal", 10.0, 4.0), ("Kunde", 10.0, 0.0)]
DELIVERY_LEGS = [("Lager", "Umschlag Ost", 2), ("Lager", "Umschlag West", 3), ("Umschlag Ost", "Grenzlager", 9), ("Umschlag West", "Zentrum", 4), ("Zentrum", "Umschlag Ost", -6),
                 ("Grenzlager", "Hafen", 2), ("Hafen", "Terminal", 3), ("Terminal", "Kunde", 1)]


def delivery_network():
    names = [s[0] for s in DELIVERY_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in DELIVERY_LEGS], [(s[1], s[2]) for s in DELIVERY_STOPS], names, directed=True)
    return Network("delivery", g, "Euro", "Frachtnetz mit Rückvergütung",
                   "Ein eigenes kleines Frachtnetz (Kosten in Euro je Strecke): die Strecke Zentrum → Umschlag Ost hat eine Rückvergütung von 6 Euro (negative Kosten), weil dort Rückfracht mitgenommen wird.",
                   idx["Lager"], idx["Kunde"])


# --- Devisen ------------------------------------------------------------------------------------------------------------------------------------

CURRENCIES = ("EUR", "USD", "GBP", "CHF", "JPY", "CAD")
_BASE_VALUE = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "CHF": 1.04, "JPY": 0.0062, "CAD": 0.68}    # erfundene Werte in Euro
SPREAD = 0.002                                                                                     # Aufschlag je Umtausch (0.2 %): darum gibt es ohne Kursfehler keinen Gewinnzyklus
MISPRICED = ("GBP", "CHF", 1.012)                                                                  # ein Kursfehler: dieser Umtausch bringt 1.2 % zu viel


def _fx(mispriced):
    k = len(CURRENCIES)
    rates = [[0.0] * k for _ in range(k)]
    for i, a in enumerate(CURRENCIES):
        for j, b in enumerate(CURRENCIES):
            if i != j:
                rates[i][j] = _BASE_VALUE[a] / _BASE_VALUE[b] * (1.0 - SPREAD)
    if mispriced:
        a, b, f = MISPRICED
        rates[CURRENCIES.index(a)][CURRENCIES.index(b)] *= f
    return rates


def fx_cost(rate):
    """Kosten eines Umtauschs in Zehntausendsteln der Log-Rate: -10^4 * ln(Kurs). Ein Kreis aus Umtauschen mit negativer Summe multipliziert die Kurse zu mehr als 1 - ein Gewinnzyklus."""
    return -int(round(1e4 * math.log(rate)))


def fx_network(mispriced):
    k = len(CURRENCIES)
    rates = _fx(mispriced)
    ang = 2 * math.pi * np.arange(k) / k
    xy = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 3.0
    arcs = [(i, j, fx_cost(rates[i][j])) for i in range(k) for j in range(k) if i != j]
    g = from_arcs(k, arcs, xy, CURRENCIES, directed=True)
    key = "fx_arbitrage" if mispriced else "fx_fair"
    note = ("Erfundene Devisenkurse (kein Marktdatensatz) mit einem Aufschlag von 0.2 % je Umtausch; Kosten = −10 000 · ln(Kurs). " +
            ("Ein Kursfehler beim Umtausch GBP → CHF (1.2 % zu viel) macht einen Kreis aus Umtauschen zum Gewinn: das ist ein negativer Zyklus." if mispriced
             else "Ohne Kursfehler kostet jeder Kreis aus Umtauschen etwas (die Aufschläge): es gibt negative Kanten, aber keinen negativen Zyklus."))
    return Network(key, g, "Log-Punkte", "Devisen mit Kursfehler" if mispriced else "Devisen ohne Kursfehler", note, 0, CURRENCIES.index("JPY"), False, (), tuple(tuple(r) for r in rates))


# --- Raster-Topologie (Stadtnetz und E-Lieferwagen) -------------------------------------------------------------------------------------------

def _primitive_offsets(reach):
    r = int(math.floor(reach + 1e-9))
    out = []
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if (dy == 0 and dx <= 0) or dx * dx + dy * dy > reach * reach + 1e-9 or math.gcd(abs(dx), dy) != 1:
                continue
            out.append((dy, dx))
    return out


def _grid(side, reach, blocked_pct, seed):
    """Gestörtes Raster mit Lage der Kreuzungen und Liste der Straßen (ungerichtete Paare); ein Teil der Straßen ist gesperrt, das Netz bleibt zusammenhängend."""
    rng = np.random.default_rng([int(seed), 101])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    pairs = []
    for dy, dx in _primitive_offsets(reach):
        for i in range(side):
            for j in range(side):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < side and 0 <= j2 < side:
                    pairs.append((i * side + j, i2 * side + j2))
    pairs = np.array(pairs)
    order = rng.permutation(len(pairs))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    in_tree = np.zeros(len(pairs), dtype=bool)
    for k in order:
        a, b = find(pairs[k, 0]), find(pairs[k, 1])
        if a != b:
            parent[a] = b
            in_tree[k] = True
    removable = [k for k in order if not in_tree[k]]
    drop = set(removable[: int(round(len(pairs) * blocked_pct / 100.0))])
    keep = np.array([k not in drop for k in range(len(pairs))])
    return xy, pairs[keep], rng


def build_city(side, reach, spread, seed, blocked_pct=20):
    xy, pairs, rng = _grid(side, reach, blocked_pct, seed)
    length = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    cost = np.maximum(1.0, np.rint(length * (1.0 + spread * rng.random(len(pairs)))))
    return from_arcs(side * side, [(pairs[k, 0], pairs[k, 1], cost[k]) for k in range(len(pairs))], xy)


def city_network(side, reach, spread, seed):
    g = build_city(side, reach, spread, seed)
    return Network("city", g, "m", "Stadtnetz", "Erzeugtes Stadtnetz: Kreuzungen auf einem gestörten Raster, alle Kosten (Meter mal Streuung) positiv - hier ist Dijkstra richtig, und es geht um den Aufwand.", 0, g.n - 1)


# --- E-Lieferwagen mit Rekuperation ------------------------------------------------------------------------------------------------------------

WH_PER_M = 0.15                    # Verbrauch auf ebener Strecke: 150 Wh je km
WH_PER_M_CLIMB = 5.45              # Lageenergie eines 2-Tonnen-Wagens: 5.45 Wh je Meter Höhe (m·g·h)


def build_hills(side, hill_m, seed):
    """Glatte Hügellandschaft: drei Wellen mit zufälliger Richtung und Phase, Wellenlänge 8 bis 14 Blocklängen; `hill_m` = größter Höhenunterschied (Spitze zu Tal) in Metern."""
    rng = np.random.default_rng([int(seed), 505])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1).astype(float)
    h = np.zeros(n)
    for _ in range(3):
        ang, phase = rng.uniform(0, 2 * math.pi, 2)
        wave = rng.uniform(8.0, 14.0)
        h += np.sin(2 * math.pi * (ij[:, 1] * math.cos(ang) + ij[:, 0] * math.sin(ang)) / wave + phase)
    if h.max() > h.min():
        h = (h - h.min()) / (h.max() - h.min())
    return h * hill_m


def ev_network(side, hill_m, eta_pct, seed, reach=1.5, blocked_pct=20):
    """Energiekosten in Wh je Strecke: Reibung (0.15 Wh je Meter) plus Lageenergie beim Bergauffahren (5.45 Wh je Meter Höhe); bergab bekommt der Wagen `eta_pct` Prozent der Lageenergie zurück -
    das kann größer sein als die Reibung: die Strecke hat dann negative Kosten. Ein Kreis kostet nie etwas ab: Auf- und Abstieg sind gleich, und mit Rückgewinnung unter 100 % geht Energie verloren."""
    xy, pairs, _ = _grid(side, reach, blocked_pct, seed)
    heights = build_hills(side, hill_m, seed)
    arcs = []
    for a, b in pairs:
        length = float(np.hypot(*(xy[a] - xy[b])))
        dh = float(heights[b] - heights[a])
        for u, v, d in ((int(a), int(b), dh), (int(b), int(a), -dh)):
            lift = WH_PER_M_CLIMB * (d if d > 0 else d * eta_pct / 100.0)
            arcs.append((u, v, float(round(WH_PER_M * length + lift))))
    g = from_arcs(side * side, arcs, xy, directed=True)
    note = ("Erzeugtes hügeliges Stadtnetz für einen E-Lieferwagen: Kosten in Wh (Reibung plus Lageenergie); bergab wird Energie zurückgewonnen, manche Strecken haben deshalb negative Kosten. "
            "Es gibt nie einen negativen Zyklus: was man bergab gewinnt, hat man bergauf bezahlt.")
    return Network("ev", g, "Wh", "E-Lieferwagen mit Rekuperation", note, 0, g.n - 1, True, tuple(float(x) for x in heights))


# --- Zufallsnetz mit negativen Kanten ------------------------------------------------------------------------------------------------------------

def build_random(n, degree, pot_span, seed):
    """Gerichteter, stark zusammenhängender Zufallsgraph (ein Rundweg durch alle Knoten plus zufällige Kanten bis zum mittleren Ausgangsgrad `degree`), Kosten 1 bis 9 plus Potenzial(u) - Potenzial(v).
    Jeder Kreis hat dieselbe Summe wie ohne Potenziale (die Potenziale heben sich auf): es entstehen negative Kanten, aber nie ein negativer Zyklus. Das ist die Idee hinter Johnsons Umgewichtung."""
    rng = np.random.default_rng([int(seed), 707])
    perm = rng.permutation(n)
    pot = np.random.default_rng([int(seed), 606]).integers(0, pot_span + 1, n) if pot_span > 0 else np.zeros(n, dtype=int)     # eigener Zufallsstrom: die Struktur des Netzes hängt nicht von der Spanne ab
    arcs = {(int(perm[i]), int(perm[(i + 1) % n])): int(rng.integers(1, 10)) for i in range(n)}
    target = int(round(n * degree))
    while len(arcs) < target:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs:
            arcs[(u, v)] = int(rng.integers(1, 10))
    xy = rng.random((n, 2)) * 1000.0
    return from_arcs(n, [(u, v, float(c + pot[u] - pot[v])) for (u, v), c in arcs.items()], xy, directed=True)


def random_network(n, degree, pot_span, seed):
    g = build_random(int(n), float(degree), int(pot_span), int(seed))
    return Network("random", g, "Einheiten", "Zufallsnetz mit negativen Kanten",
                   "Erzeugter gerichteter Zufallsgraph, Kosten 1 bis 9 plus Potenzial(Start) − Potenzial(Ziel): je größer die Potenzialspanne, desto mehr negative Kanten - aber nie ein negativer Zyklus. Keine Karte, nur Punkte.", 0, n // 2, False)


# --- Zusammenbau -------------------------------------------------------------------------------------------------------------------------------

def make_network(net, side=C.DEFAULT_SIDE, hill=C.DEFAULT_HILL, eta=C.DEFAULT_ETA, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, pot=C.DEFAULT_POT, seed=C.DEFAULT_SEED):
    if net == "delivery":
        return delivery_network()
    if net == "fx_arbitrage":
        return fx_network(True)
    if net == "fx_fair":
        return fx_network(False)
    if net == "ev":
        return ev_network(int(side), float(hill), float(eta), int(seed))
    if net == "random":
        return random_network(int(nodes), float(degree), int(pot), int(seed))
    if net == "city":
        return city_network(int(side), float(reach), float(spread), int(seed))
    raise ValueError(net)

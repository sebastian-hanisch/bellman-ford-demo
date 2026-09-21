"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("delivery", "fx_arbitrage", "fx_fair", "ev", "random", "city")
NET_LABELS = {
    "delivery": "🚚 Frachtnetz mit Rückvergütung (klein)",
    "fx_arbitrage": "💱 Devisen mit Kursfehler (negativer Zyklus)",
    "fx_fair": "💱 Devisen ohne Kursfehler",
    "ev": "🔋 E-Lieferwagen mit Rekuperation (erzeugt)",
    "random": "🕸️ Zufallsnetz mit negativen Kanten (erzeugt)",
    "city": "🏙️ Stadtnetz ohne negative Kanten (erzeugt)",
}
SMALL_NETS = ("delivery", "fx_arbitrage", "fx_fair")       # eigene kleine Graphen mit Namen, fester Aufgabe und Tabelle
SIZED_NETS = ("ev", "city")                                # Netze mit Größenregler (Raster)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 6, 40, 16
HILL_MIN, HILL_MAX, DEFAULT_HILL = 0, 40, 30              # Höhenunterschied der Hügel in Metern (E-Lieferwagen)
ETA_MIN, ETA_MAX, DEFAULT_ETA = 0, 90, 60                 # Rückgewinnung beim Bergabfahren in Prozent
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 1.5       # Reichweite der Straßen in Blocklängen (Stadtnetz)
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
NODES_MIN, NODES_MAX, DEFAULT_NODES = 100, 1000, 400
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.0, 6.0, 3.0
POT_MIN, POT_MAX, DEFAULT_POT = 0, 20, 8                  # Spanne der Potenziale (Zufallsnetz): je größer, desto mehr negative Kanten
DEFAULT_SEED = 7
DEFAULT_NET = "delivery"

DEFAULT_VARIANT = "early_stop"
DEFAULT_EDGE_ORDER = "as_given"
VARIANT_LABELS = {"early_stop": "früher Abbruch (Standard)", "textbook": "Lehrbuch (immer n − 1 Runden)", "synchronous": "früher Abbruch, gleichzeitig (Runde k = höchstens k Kanten)", "queue": "Warteschlange (nur geänderte Knoten)"}
ORDER_LABELS = {"as_given": "wie im Netz (nach Startknoten)", "random": "zufällig", "near_first": "nahe Knoten zuerst (nicht vorab wissbar)", "far_first": "ferne Knoten zuerst"}

SWEEP_SEEDS = tuple(range(100000, 100005))
SOURCES = 20                       # zufällige Startknoten je Netz für die Verteilungen

COLORS = {"bf": "#d62728", "dijkstra": "#1f77b4", "negative": "#2ca02c", "cycle": "#d62728", "start": "#111111", "goal": "#ff7f0e", "changed": "#ffd54f"}

_BASE = dict(side=DEFAULT_SIDE, hill=DEFAULT_HILL, eta=DEFAULT_ETA, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, pot=DEFAULT_POT,
             variant=DEFAULT_VARIANT, order=DEFAULT_EDGE_ORDER, seed=DEFAULT_SEED)
PRESETS = {
    "🚚 Frachtnetz": {**_BASE, "net": "delivery"},
    "💱 Devisen mit Kursfehler": {**_BASE, "net": "fx_arbitrage"},
    "💱 Devisen ohne Kursfehler": {**_BASE, "net": "fx_fair"},
    "🔋 E-Lieferwagen": {**_BASE, "net": "ev"},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
}
PRESET_HELP = {
    "🚚 Frachtnetz": "Kleines eigenes Frachtnetz mit einer Rückvergütung (negative Kante): Dijkstra findet 17 Euro, die billigste Route über die Rückvergütung kostet 16. Bellman-Ford braucht dafür 3 Runden und 24 Kantenprüfungen (Dijkstra 8; das Lehrbuch 64).",
    "💱 Devisen mit Kursfehler": "Sechs Währungen mit erfundenen Kursen und einem Kursfehler bei GBP → CHF: ein Rundlauf GBP → CHF → GBP bringt etwa 0.8 % Gewinn, das ist ein negativer Zyklus. Bellman-Ford meldet ihn nach 6 Runden (180 Kantenprüfungen), mit Vorgänger-Prüfung schon nach Runde 1 (30 Prüfungen).",
    "💱 Devisen ohne Kursfehler": "Dieselben sechs Währungen ohne Kursfehler: 15 der 30 Umtausche haben negative Kosten (Kurs über 1), aber es gibt keinen negativen Zyklus - jeder Rundlauf kostet etwas. Bellman-Ford und Dijkstra finden beide denselben Weg.",
    "🔋 E-Lieferwagen": "Hügeliges Stadtnetz (16 × 16 Kreuzungen, 30 m Hügel, 60 % Rückgewinnung): 69 der 1 488 Kanten sind negativ. Für das gezeigte Ziel findet Bellman-Ford 97 Wh, Dijkstra 105; Dijkstra liegt an 62 von 256 Knoten falsch.",
    "🕸️ Zufallsnetz": "Zufallsnetz (400 Knoten, mittlerer Grad 3, Potenzialspanne 8): 142 von 1 200 Kanten sind negativ, nie ein negativer Zyklus. Für das gezeigte Ziel findet Bellman-Ford 15, Dijkstra 20; Dijkstra liegt an 9 von 400 Knoten falsch. Bellman-Ford braucht 8 400 Kantenprüfungen, Dijkstra 1 200.",
    "🏙️ Stadtnetz": "Stadtnetz ohne negative Kanten (16 × 16): beide finden dieselben Kosten, 3 040 m. Bellman-Ford braucht 4 464 Kantenprüfungen (3.0-fach), das Lehrbuch 380 928; im Mittel über 20 Startknoten das 12.6-Fache - hier ist Dijkstra die bessere Wahl.",
}

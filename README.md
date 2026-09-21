# Bellman-Ford – negative Kanten erlaubt, Runde für Runde – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bellman-ford-demo.streamlit.app/)**

Fünftes Stück der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung der [Dijkstra-Demo](../dijkstra-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Bellman-Ford** – an einem wachsenden Beispiel.
Dijkstra legt jeden Knoten einmal endgültig fest und geht davon aus, dass eine Route nie billiger wird, wenn man Kanten anhängt. Mit einer **negativen Kante** (Rückvergütung, Umtausch mit Gewinn, Bergabfahrt mit Rückgewinnung) stimmt das nicht mehr.
Bellman-Ford verzichtet auf das endgültige Festlegen: es **prüft in jeder Runde alle Kanten**; nach Runde *k* ist jede Route mit höchstens *k* Kanten berücksichtigt. Verbessert sich in Runde *n* noch etwas, gibt es einen **negativen Zyklus** – und damit keine kürzeste Route.
Der Preis: viel mehr Arbeit als Dijkstra.

**Einordnung in die Reihe (die Kanten des Graphen):** Bellman-Ford setzt an Dijkstras Grenze "keine negativen Kanten" an. Die Dijkstra-Demo nutzt es nur als Messwerkzeug, hier ist es das Thema.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       ├─ bellman-ford-demo ─┐                                                       [dieses Stück]
       │   Floyd-Warshall ───┴→ Johnson (Konvergenz: Umgewichtung mit Potenzialen)   [nicht gebaut]
       └─ Mehrkriterien-Routing (Zeit gegen CO₂, Pareto)                             [nicht gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Runden, Lockerung, Zyklus-Erkennung) | Ford (1956) und Bellman (1958); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) erwähnen es nur in einem Nebensatz, ein Buchbeispiel gibt es nicht zu spiegeln |
| Vier Varianten (Lehrbuch, früher Abbruch, gleichzeitig, Warteschlange), Kantenreihenfolgen, Vorgänger-Prüfung, Dijkstra mit Alarm | eigene Umsetzung |
| Alle Netze | **eigene Graphen und Erzeuger**: kleines Frachtnetz, zwei Devisennetze (die Kurse sind **erfunden**, keine Marktdaten), E-Lieferwagen mit Rekuperation im erzeugten hügeligen Stadtnetz, Zufallsnetz mit negativen Kanten über Potenziale, Stadtnetz ohne negative Kanten |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug** in diesem Stück, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Frachtnetz mit einer Rückvergütung | ✅ Dijkstra findet **17 Euro**, die billigste Route über die negative Kante kostet **16**; Bellman-Ford braucht 3 Runden und **24** Kantenprüfungen (Dijkstra 8, Lehrbuch 64) |
| E-Lieferwagen (16 × 16, 30 m Hügel, 60 % Rückgewinnung) | ✅ **69** von 1 488 Kanten negativ; für das gezeigte Ziel **97 Wh** gegen **105** bei Dijkstra, der an **62** von 256 Knoten falsch liegt. Anteil falscher Knoten bei 0 / 20 / 40 / 60 / 80 / 90 % Rückgewinnung: 0 / 0 / 0.2 / 12.8 / 20.4 / 24.2 % (nie ein negativer Zyklus) |
| Zufallsnetz mit Potenzialen (400 Knoten) | ✅ Potenzialspanne 2 / 8 / 20 → **1.1 / 11.1 / 27.1 %** negative Kanten, Dijkstra falsch an 0.2 / 6.3 / 23.3 % der Knoten; die **Runden von Bellman-Ford bleiben bei jeder Spanne gleich** (die Potenziale ändern nicht, welche Routen kürzeste sind) – der Vorgriff auf Johnson |
| Devisen mit Kursfehler | 🔁 ein Kursfehler bei GBP → CHF macht GBP → CHF → GBP zum **Gewinnzyklus (etwa 0.8 %)**; Bellman-Ford meldet ihn nach **6 Runden (180 Kantenprüfungen)**, mit Vorgänger-Prüfung schon nach **Runde 1 (30)** |
| Devisen ohne Kursfehler | ℹ️ 15 der 30 Umtausche haben negative Kosten, aber es gibt keinen negativen Zyklus; beide Verfahren finden denselben Weg |
| Aufwand im Stadtnetz (ohne negative Kanten, 40 × 40 = 1 600 Knoten) | ⚠️ Lehrbuch **1 600-fach** Dijkstra (immer n · m), früher Abbruch **9.4-fach** (28.6 Runden bei zufälliger Kantenreihenfolge, 9.4 bei der günstigen "wie im Netz"), Warteschlange **1.07-fach** |
| Kantenreihenfolge | ⚠️ Stadtnetz (20 × 20): 2 Runden bei "nahe Knoten zuerst" (nur mit den wahren Entfernungen möglich), 4.8 "wie im Netz" (günstig, weil die Knotennummern dem Raster folgen), 15.0 zufällig, 25.4 "ferne zuerst"; im Zufallsnetz (400 Knoten) ist "wie im Netz" mit 8.2 kaum besser als zufällig (8.4) |
| Erkennung eines negativen Zyklus (Zufallsnetz, 400 Knoten) | ❌ Lehrbuch-Regel und Warteschlange brauchen fast **n · m = 481 200** Kantenprüfungen (400-mal Dijkstra); die **Vorgänger-Prüfung** nach jeder Runde meldet ihn nach **5 Runden mit 6 015 Prüfungen**, etwa 80-mal weniger |
| Korrektheit | ✅ jede Variante und jede Kantenreihenfolge liefert auf jedem geprüften Netz exakt die Entfernungen von networkx und meldet genau dann einen Zyklus, wenn networkx einen findet; jeder gemeldete Zyklus ist ein echter, von der Quelle erreichbarer negativer Zyklus |

Die Zahl der Kantenprüfungen ist der Aufwand und plattformfest. Laufzeiten stehen in der App nur als Messwerte und werden nirgends behauptet oder getestet. Das Lehrbuch wird nicht ausgeführt, sondern als n · m gerechnet (ein Test belegt, dass es immer genau so viel prüft).
Die Kantenreihenfolge "wie im Netz" ist im Stadtnetz auffällig günstig – ein Artefakt der Knotennummerierung, das die App ausdrücklich zeigt, statt es als Ergebnis von Bellman-Ford auszugeben.

## Was die Demo zeigt

1. **Bellman-Ford in Aktion** (Schritt-Regler + Abspielen): Karte mit Entfernungen als Farbe, den in der Runde gelockerten Kanten (gelb), den negativen Kanten (grün) und – nach der letzten Runde – der Route (rot) gegen die von Dijkstra (blau gestrichelt). Bei den kleinen Netzen die **Tabelle** mit Entfernung und Vorgänger je Knoten (Änderungen gelb) und die Lockerungen der Runde; bei den großen Netzen der Verlauf der Verbesserungen je Runde. Bei einem negativen Zyklus wird er in Rot eingezeichnet.
2. **Negative Kanten – Runde für Runde:** Kennzahlen des gezeigten Ziels (Kosten beider Verfahren, Runden, Kantenprüfungen) mit Urteil, dazu die Werte über 20 zufällige Startknoten (Anteil der Starts, bei denen Dijkstra danebenliegt, falsche Knoten, Kantenprüfungen im Verhältnis zu Dijkstra).
3. **Vergleich** (Expander): Kantenprüfungen und Messwerte; **Experimente auf Knopfdruck**: Aufwand gegen Netzgröße, Kantenreihenfolge, wann Dijkstra danebenliegt (Rekuperation und Potenziale), die Kosten der Zyklus-Erkennung.
4. **Wo die Annahmen enden** (Tabelle mit den Ansatzpunkten der nächsten Stücke) und **Mathematische Formulierung** (Lockerung, Invariante "höchstens *k* Kanten", Beweis der n − 1 Runden, negative Zyklen, Vorgänger-Prüfung, Potenziale).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, Variante (Lehrbuch / früher Abbruch / gleichzeitig / Warteschlange) und – wo sie wirkt – Kantenreihenfolge wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz oder zur Variante nicht gehören, sind ausgeblendet.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `bf_graph.py`, `bf_queues.py` | gerichteter Graph in CSR-Form, Warteschlange für das Dijkstra-Gegenstück |
| `bf_algorithm.py` | Bellman-Ford (vier Varianten, Kantenreihenfolgen, Vorgänger-Prüfung), Dijkstra mit Alarm, "höchstens k Kanten" |
| `bf_scenario.py` | Netze: Frachtnetz, Devisen, E-Lieferwagen, Zufallsnetz, Stadtnetz |
| `bf_evaluation.py` | Zielwahl, Kennzahlen, Verteilung über Startknoten, Experimente |
| `bf_visualization.py`, `bf_presets.py`, `bf_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx (`single_source_bellman_ford_path_length`) für alle Varianten und Kantenreihenfolgen auf Netzen mit und ohne negative Kanten und Zyklen, mit unerreichbaren Zyklen, Nullkanten und unerreichbaren Zielen.

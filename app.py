"""Bellman-Ford - negative Kanten erlaubt, Runde für Runde - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Bellman-Ford - und lässt stattdessen das Beispiel wachsen.
Fünftes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Fortsetzung der Dijkstra-Demo: Dijkstra darf keine negativen Kanten haben, Bellman-Ford schon - und erkennt, wenn es gar keinen kürzesten Weg gibt.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import bf_constants as C
import bf_evaluation as ev
from bf_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bf_scenario import make_network
from bf_visualization import build_cycle, build_effort, build_network, build_order, build_progress, build_recuperation, state_at, table_state

st.set_page_config(page_title="Bellman-Ford – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


def _cost(net, x):
    return f"{x:,.0f} {net.unit}".replace(",", ".") if float(x).is_integer() else f"{x:g} {net.unit}"


def _pct(x, digits=0):
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _analysis(params, variant, order):
    return ev.analyse(_network(params), variant, order, params[-1])


@st.cache_data(show_spinner=False)
def _source_stats(params, variant, order):
    return ev.source_stats(_network(params), variant, order, seed=params[-1])


@st.cache_data(show_spinner=False)
def _effort():
    return ev.effort_vs_size()


@st.cache_data(show_spinner=False)
def _order_table():
    return ev.edge_order_table()


@st.cache_data(show_spinner=False)
def _recuperation():
    return ev.recuperation_sweep()


@st.cache_data(show_spinner=False)
def _potentials():
    return ev.potential_sweep()


@st.cache_data(show_spinner=False)
def _cycles():
    return ev.cycle_detection()


st.title("🔁 Bellman-Ford – negative Kanten erlaubt, Runde für Runde")
st.markdown(
    """
Dijkstra legt jeden Knoten einmal endgültig fest - und geht davon aus, dass eine Route nie billiger wird, wenn man Kanten anhängt. Mit einer **negativen Kante** (eine Rückvergütung, ein Umtausch mit Gewinn, eine Bergabfahrt, die mehr Energie zurückgibt, als sie kostet) stimmt das nicht mehr.
**Bellman-Ford** verzichtet auf das endgültige Festlegen: es **prüft in jeder Runde alle Kanten** und verbessert jede Entfernung, die sich über eine Kante verbessern lässt. Nach Runde $k$ ist jede Route mit höchstens $k$ Kanten berücksichtigt, nach spätestens $n-1$ Runden ist Schluss.
Verbessert sich in Runde $n$ noch etwas, gibt es einen **negativen Zyklus** - und damit keine kürzeste Route. Der Preis: viel mehr Arbeit als Dijkstra.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - fünftes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Fortsetzung der Dijkstra-Demo - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwächen von Bellman-Ford sind der Aufwand (Größenordnung $n \\cdot m$ im Lehrbuch) und dass es nur **einen** Start bedient. Das Verfahren geht auf Ford (1956) und Bellman (1958) zurück; "
    "alle Netze und Zahlen dieser Demo sind eigene Graphen und Messungen (die Devisenkurse sind erfunden, keine Marktdaten)."
)

with st.expander("So funktioniert Bellman-Ford", expanded=True):
    st.markdown(
        """
1. **Start:** der Start hat Entfernung 0, alle anderen $\\infty$.
2. **Runde:** für **jede Kante** $(u,v)$ mit Kosten $c$: ist $d(u)+c < d(v)$, wird $d(v)$ auf $d(u)+c$ gesenkt und $u$ als Vorgänger von $v$ gemerkt (eine **Lockerung**).
3. **Wiederholen:** eine Route hat höchstens $n-1$ Kanten, wenn es keinen negativen Zyklus gibt. Nach Runde $k$ ist deshalb jede Route mit höchstens $k$ Kanten berücksichtigt - nach $n-1$ Runden stehen alle Entfernungen fest.
   Bricht man ab, sobald eine Runde nichts mehr ändert, ist man meist viel früher fertig (**früher Abbruch**). Eine **Warteschlange** prüft nur noch Kanten von Knoten, deren Wert sich gerade verbessert hat.
4. **Negativer Zyklus:** ändert sich in Runde $n$ noch etwas, kann man die Kosten endlos senken, indem man den Zyklus wieder und wieder durchläuft: es gibt keine kürzeste Route. Den Zyklus selbst findet man über die Vorgänger.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (Frachtnetz mit Rückvergütung, zwei Devisennetze) oder erzeugt (E-Lieferwagen mit Rekuperation, Zufallsnetz mit negativen Kanten, Stadtnetz ohne negative Kanten). Alle Kosten sind ganze Zahlen; "
             "kein Netz braucht Kartendaten.",
    )
    sized = net_key in C.SIZED_NETS
    if sized:
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider", help="Größe des Rasters: Kreuzungen je Seite (das Netz hat das Quadrat davon).")
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "ev":
        hill = st.slider("Hügel [m Höhenunterschied]", *bounds("hill_slider"), key="hill_slider",
                         help="Höhenunterschied zwischen Tal und Spitze. Anteil der Knoten, an denen Dijkstra falsche Kosten liefert (Mittel über fünf Netze, 16 × 16 Kreuzungen, 60 % Rückgewinnung) bei 0 / 10 / 20 / 30 / 40 m: 0 % / 0 % / 1.3 % / 12.8 % / 15.8 %.")
        st.session_state[KEPT["hill_slider"]] = hill
        eta = st.slider("Rückgewinnung bergab [%]", *bounds("eta_slider"), key="eta_slider",
                        help="Wie viel Lageenergie der Wagen beim Bergabfahren zurückgewinnt. Anteil falscher Knoten bei Dijkstra (30 m Hügel) bei 0 / 20 / 40 / 60 / 80 / 90 %: 0 % / 0 % / 0.2 % / 12.8 % / 20.4 % / 24.2 %. "
                             "Unter 100 % entsteht nie ein negativer Zyklus.")
        st.session_state[KEPT["eta_slider"]] = eta
    else:
        hill = int(st.session_state.get(KEPT["hill_slider"], C.DEFAULT_HILL))
        eta = int(st.session_state.get(KEPT["eta_slider"], C.DEFAULT_ETA))
    if net_key == "city":
        reach = st.slider("Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1, help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster).")
        st.session_state[KEPT["reach_slider"]] = reach
        spread = st.slider("Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25, help="Kosten einer Straße = Länge × (1 + Streuung × Zufall), gerundet auf ganze Meter.")
        st.session_state[KEPT["spread_slider"]] = spread
    else:
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
    if net_key == "random":
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=100,
                          help="Anzahl der Knoten. Kantenprüfungen von Bellman-Ford (früher Abbruch) im Verhältnis zu Dijkstra bei 100 / 400 / 1 000 Knoten (mittlerer Grad 3): 6.4-fach / 8.2-fach / 8.8-fach.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5,
                           help="Kanten je Knoten (ausgehend). Verhältnis der Kantenprüfungen Bellman-Ford / Dijkstra bei Grad 2 / 4 / 6 (400 Knoten): 10.4-fach / 6.8-fach / 6.0-fach.")
        st.session_state[KEPT["degree_slider"]] = degree
        pot = st.slider("Potenzialspanne", *bounds("pot_slider"), key="pot_slider",
                        help="Kosten = 1 bis 9 plus Potenzial(Start) − Potenzial(Ziel), Potenziale zufällig zwischen 0 und dieser Spanne. Je größer, desto mehr negative Kanten - nie ein negativer Zyklus. "
                             "Anteil negativer Kanten bei Spanne 2 / 8 / 20: 1.1 % / 11.1 % / 27.1 %; Anteil falscher Knoten bei Dijkstra 0.2 % / 6.3 % / 23.3 %.")
        st.session_state[KEPT["pot_slider"]] = pot
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
        pot = int(st.session_state.get(KEPT["pot_slider"], C.DEFAULT_POT))
    variant = st.selectbox("Variante von Bellman-Ford", list(C.VARIANT_LABELS), key="variant_select", format_func=lambda k: C.VARIANT_LABELS[k],
                           help="Lehrbuch: immer n − 1 Runden, dann eine Prüfrunde. Früher Abbruch: hört nach der ersten Runde ohne Änderung auf. Gleichzeitig: jede Runde liest nur die Werte der vorigen Runde (Runde k = höchstens k Kanten). "
                                "Warteschlange: prüft nur Kanten von Knoten, die sich gerade verbessert haben. Das Ergebnis ist bei allen gleich, nur der Aufwand nicht.")
    if variant in ("textbook", "early_stop"):
        order = st.selectbox("Reihenfolge der Kanten", list(C.ORDER_LABELS), key="order_select", format_func=lambda k: C.ORDER_LABELS[k],
                             help="In welcher Reihenfolge eine Runde die Kanten prüft; Verbesserungen wirken sofort. Kanten \"wie im Netz\" laufen nach Startknoten - im Stadtnetz folgen die Knotennummern dem Raster, das ist eine günstige Reihenfolge. "
                                  "\"Nahe zuerst\" braucht die wahren Entfernungen und ist nur eine Grenze, keine Möglichkeit. Am Ergebnis ändert nichts etwas.")
        st.session_state[KEPT["order_select"]] = order
    else:
        order = st.session_state.get(KEPT["order_select"], C.DEFAULT_EDGE_ORDER)
    if net_key in ("ev", "random", "city"):
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "hill_slider": int(hill), "eta_slider": int(eta), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2),
                   "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "pot_slider": int(pot), "variant_select": variant, "order_select": order, "seed_input": int(seed)})

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, hill=C.DEFAULT_HILL, eta=C.DEFAULT_ETA, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, pot=C.DEFAULT_POT, seed=C.DEFAULT_SEED)
if net_key == "ev":
    d.update(side=int(side), hill=int(hill), eta=int(eta), seed=int(seed))
elif net_key == "city":
    d.update(side=int(side), reach=round(float(reach), 1), spread=round(float(spread), 2), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), pot=int(pot), seed=int(seed))
params = (net_key, d["side"], d["hill"], d["eta"], d["reach"], d["spread"], d["nodes"], d["degree"], d["pot"], d["seed"])
order_used = order if variant in ("textbook", "early_stop") else C.DEFAULT_EDGE_ORDER
with st.spinner("Rechne ..."):
    a = _analysis(params, variant, order_used)
net, m, bf, g = a.net, a.metrics, a.bf, a.net.graph
small = bool(g.names)
view_key = (params, variant, order_used)
last_step = len(bf.history)
queue_var = variant == "queue"
unit_word = "Zwischenstand" if queue_var else "Runde"

# --- Bellman-Ford in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Bellman-Ford in Aktion")
if st.session_state.get("bf_step_owner") != view_key:
    st.session_state["bf_step"] = last_step
    st.session_state["bf_step_owner"] = view_key
step_col, play_col = st.columns([5, 2])
with step_col:
    if last_step > 1:
        step = st.slider("Zwischenstände der Warteschlange" if queue_var else "Runden", 0, last_step, key="bf_step",
                         help=("Wie viele Zwischenstände (Gruppen von Entnahmen aus der Warteschlange) schon gerechnet sind." if queue_var else
                               "Wie viele Runden über alle Kanten schon gerechnet sind: 0 = nur der Start ist bekannt, ganz rechts = fertig. " +
                               ("Bei der Variante \"gleichzeitig\" ist die Entfernung nach Runde k genau die billigste Route mit höchstens k Kanten." if variant == "synchronous" else "")))
    else:
        step = last_step
        st.caption("Es gibt nur einen Schritt.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
show_dj = False
if not m["cycle"]:
    show_dj = st.checkbox("Route von Dijkstra einblenden (blau gestrichelt)", value=True, key="show_dj",
                          help="Nach der letzten Runde: die Route, die Dijkstra gewählt hätte. Sie liegt über der von Bellman-Ford, wenn ein festgelegter Knoten über eine negative Kante noch billiger zu erreichen war.")
view_slot = st.empty()


def _render(current):
    with view_slot.container():
        if small:
            st.plotly_chart(build_network(net, a, current, show_dj, height=420), width="stretch", key="net_chart")
            rows = table_state(a, current)
            df = pd.DataFrame(rows)
            changed = df.pop("geändert")

            def highlight(row):
                return ["background-color: #ffe08a" if (changed.loc[row.name] and col in ("Entfernung", "Vorgänger")) else "" for col in row.index]

            t1, t2 = st.columns([3, 2])
            t1.markdown(f"**Entfernungen und Vorgänger nach {unit_word} {current}** (gelb: in diesem Schritt geändert)")
            t1.dataframe(df.style.apply(highlight, axis=1).format({"Entfernung": lambda x: "∞" if x is None or pd.isna(x) else f"{x:g}"}), hide_index=True, width="stretch")
            _, _, ch = state_at(a, current)
            t2.markdown(f"**{len(ch)} Lockerung(en) in diesem Schritt:**" if ch else "**Keine Lockerung in diesem Schritt.**")
            for u, v, old, new in ch[:12]:
                t2.markdown(f"- {g.names[v]}: {'∞' if not np.isfinite(old) else f'{old:g}'} → {new:g} (über {g.names[u]})")
        else:
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_network(net, a, current, show_dj), width="stretch", key="net_chart")
            c2.markdown(f"**Verbesserungen je {unit_word}**")
            c2.plotly_chart(build_progress(a, current), width="stretch", key="progress_chart")
            _, _, ch = state_at(a, current)
            c2.caption(f"Nach {unit_word} {current} von {last_step}; {len(ch)} Verbesserungen in diesem Schritt. Grün eingezeichnet sind die negativen Kanten, gelb die in diesem Schritt gelockerten." if net.geometric else
                       f"Nach {unit_word} {current} von {last_step}; {len(ch)} Verbesserungen in diesem Schritt. Gefärbt sind die bisher erreichten Knoten nach ihrer Entfernung; das Zufallsnetz hat keine Karte, die Kanten sind nicht gezeichnet.")


if auto_play:
    n_frames = min(max(last_step, 1), 60)
    for k in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames + 1)}):
        _render(k)
        time.sleep(min(0.6, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)
if m["cycle"]:
    st.caption("Mit einem negativen Zyklus sinken die Werte in jeder Runde weiter: was in der Tabelle steht, sind keine Entfernungen, sondern der Stand beim Abbruch. Eine kürzeste Route gibt es nicht.")

if net.rates and m["cycle"] and bf.cycle:
    cur = list(g.names)
    prod = 1.0
    for u, v in zip(bf.cycle[:-1], bf.cycle[1:]):
        prod *= net.rates[u][v]
    st.markdown(f"**Der Gewinnzyklus:** {' → '.join(cur[i] for i in bf.cycle)}. Ein Rundlauf macht aus 1 000 {cur[bf.cycle[0]]} genau {1000 * prod:,.1f} {cur[bf.cycle[0]]} ({prod - 1:.2%} Gewinn) - und das beliebig oft wiederholt.".replace(",", "X").replace(".", ",").replace("X", "."))

st.markdown("---")

# --- Negative Kanten - Runde für Runde ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Negative Kanten – Runde für Runde")
st.caption("**Kantenprüfung** = eine Kante darauf ansehen, ob sie eine Entfernung verbessert - der Aufwand einer Suche, plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten.")
code = ev.verdict(a)
if code == "unreachable":
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar.")
else:
    m1, m2, m3, m4 = st.columns(4)
    if m["cycle"]:
        m1.metric("Bellman-Ford", "kein kürzester Weg", delta=f"Zyklus mit {m['cycle_cost']:g} {net.unit}", delta_color="off", help="Bei einem negativen Zyklus kann man die Kosten beliebig senken.")
        m2.metric("Dijkstra", f"{m['cost_dj']:g} {net.unit}", delta=f"{m['dj_alarms']} Alarm(e)", delta_color="off", help="Dijkstra merkt den Zyklus nicht; sein Ergebnis ist nichts wert.")
    else:
        m1.metric("Bellman-Ford", _cost(net, m["cost_bf"]), delta=f"{m['hops_bf']} Kanten", delta_color="off", help="Kosten der billigsten Route zum Ziel.")
        m2.metric("Dijkstra", _cost(net, m["cost_dj"]), delta=(f"{m['cost_dj'] - m['cost_bf']:+g} {net.unit}" if m["dj_wrong_target"] else "gleich"), delta_color="off", help="Ergebnis von Dijkstra; mit negativen Kanten nicht mehr sicher richtig.")
    m3.metric(f"{'Entnahmen' if queue_var else 'Runden'}", _num(m["rounds"]), delta=("" if queue_var else f"Lehrbuch: {_num(g.n)}"), delta_color="off",
              help="Zahl der Runden, die Bellman-Ford gebraucht hat (in der letzten fand es nichts mehr). Das Lehrbuch rechnet immer n Runden." if not queue_var else "Knoten, die aus der Warteschlange genommen wurden.")
    m4.metric("Kantenprüfungen", _num(m["checks_bf"]), delta=f"{m['factor']:.1f}× Dijkstra ({_num(m['checks_dj'])})", delta_color="off", help=f"Lehrbuch (immer n · m): {_num(m['textbook_checks'])}.")
    if code == "cycle":
        extra = (f" Mit der Vorgänger-Prüfung nach jeder Runde wäre er schon in Runde {m['detected_round_check']} gefunden worden ({_num(m['checks_check'])} statt {_num(m['checks_bf'])} Kantenprüfungen)." if "detected_round_check" in m else "")
        st.warning(f"🔁 **Negativer Zyklus:** {' → '.join(g.names[i] if small else str(i) for i in bf.cycle)} kostet zusammen {m['cycle_cost']:g} {net.unit} pro Umlauf. Es gibt keine kürzeste Route - Bellman-Ford meldet das nach {_num(m['rounds'])} "
                   f"{'Entnahmen' if queue_var else 'Runden'} ({_num(m['checks_bf'])} Kantenprüfungen).{extra}")
    elif code == "dijkstra_wrong":
        extra = f" Dijkstra liefert an {m['dj_wrong_nodes']} von {m['n']} Knoten falsche Kosten und schlägt {m['dj_alarms']}-mal Alarm."
        st.success(f"✅ Bellman-Ford findet {_cost(net, m['cost_bf'])} ({m['hops_bf']} Kanten), Dijkstra nur {_cost(net, m['cost_dj'])} ({m['hops_dj']} Kanten) - {m['cost_dj'] / m['cost_bf'] - 1:.1%} mehr.{extra} "
                   f"Der Preis: {_num(m['checks_bf'])} Kantenprüfungen statt {_num(m['checks_dj'])} ({m['factor']:.1f}-fach); das Lehrbuch bräuchte {_num(m['textbook_checks'])}.")
    elif code == "dijkstra_ok_negative":
        st.info(f"ℹ️ Das Netz hat {m['neg_edges']} negative Kanten ({m['neg_share']:.0%}), Dijkstra liegt hier aber richtig: keine Garantie, nur kein Fehler in diesem Netz. Bellman-Ford braucht {_num(m['checks_bf'])} Kantenprüfungen, Dijkstra {_num(m['checks_dj'])}.")
    else:
        st.info(f"ℹ️ Keine negativen Kanten: beide liefern dieselben Kosten ({_cost(net, m['cost_bf'])}). Bellman-Ford braucht dafür {_num(m['checks_bf'])} Kantenprüfungen, Dijkstra {_num(m['checks_dj'])} ({m['factor']:.1f}-fach) - "
                f"das Lehrbuch sogar {_num(m['textbook_checks'])}. Ohne negative Kanten ist Dijkstra die bessere Wahl.")

    if not small and code != "cycle":
        st.markdown("**Nicht nur dieser eine Start**")
        ss = _source_stats(params, variant, order_used)
        p1, p2, p3 = st.columns(3)
        p1.metric("Starts, bei denen Dijkstra danebenliegt", _pct(ss["share_sources_wrong"]), help=f"Anteil von {ss['n_sources']} zufälligen Startknoten, bei denen Dijkstra an mindestens einem Knoten falsche Kosten liefert.")
        p2.metric("Falsche Knoten bei Dijkstra (Mittel)", _pct(ss["wrong_share"], 1), help="Mittlerer Anteil der Knoten mit falschen Kosten.")
        p3.metric("Kantenprüfungen gegenüber Dijkstra", f"{ss['factor']:.1f}×", help=f"Mittel über {ss['n_sources']} zufällige Startknoten.")
        st.caption(f"{ss['n_sources']} zufällige Startknoten; Bellman-Ford ist bei jedem exakt. Beim Start unten links (oben) sind es weniger Prüfungen als im Mittel, weil die Knotennummern der Rasterlage folgen (siehe Reihenfolge der Kanten unten)." if net.geometric else
                   f"{ss['n_sources']} zufällige Startknoten; Bellman-Ford ist bei jedem exakt.")

st.markdown("---")

# --- Vergleich ------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Bellman-Ford und Dijkstra im Vergleich"):
    st.table({"Verfahren": [f"Bellman-Ford ({C.VARIANT_LABELS[variant].split(' (')[0]})", "Dijkstra", "Lehrbuch n · m (gerechnet)"],
              "Kantenprüfungen": [_num(m["checks_bf"]), _num(m["checks_dj"]), _num(m["textbook_checks"])],
              "Laufzeit [ms]": [f"{a.seconds['bf'] * 1000:.2f}", f"{a.seconds['dj'] * 1000:.2f}", "–"]})
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf, die Bellman-Ford-Zeit enthält das Protokoll für die Ansicht oben) und schwanken. Die Kantenprüfungen sind plattformfest. Das Lehrbuch wird nicht ausgeführt: es prüft in jeder der n Runden alle m Kanten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie viel Arbeit ist das?")
if st.button("Aufwand gegen Netzgröße messen (dauert einen Moment)", key="effort_start"):
    st.session_state["effort_on"] = True
if st.session_state.get("effort_on"):
    with st.spinner("Rechne vier Netzgrößen × 5 Netze ..."):
        rows = _effort()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_effort(rows), width="stretch", key="effort_chart")
    big = rows[-1]
    c2.table({"Verfahren": ["Lehrbuch", "früher Abbruch (Netz)", "früher Abbruch (zufällig)", "Warteschlange", "Dijkstra"],
              f"Prüfungen bei {_num(big['n'])} Knoten": [_num(big["textbook"]), _num(big["early_stop"]), _num(big["early_random"]), _num(big["queue"]), _num(big["dijkstra"])]})
    st.caption(f"Stadtnetz ohne negative Kanten, Start unten links, Mittel über 5 feste Netze. Das **Lehrbuch** prüft immer n · m Kanten: bei {_num(big['n'])} Knoten {big['textbook'] / big['dijkstra']:.0f}-mal so viel wie Dijkstra. "
               f"**Früher Abbruch** braucht nur so viele Runden, wie die kürzeste Route zum entferntesten Knoten Kanten hat, wenn die Reihenfolge ungünstig ist (hier {big['hop_depth']:.0f} Kanten); bei günstiger Reihenfolge {big['rounds_early']:.1f}, bei zufälliger {big['rounds_random']:.1f} Runden: "
               f"{big['early_stop'] / big['dijkstra']:.1f}-fach bzw. {big['early_random'] / big['dijkstra']:.1f}-fach so viele Prüfungen wie Dijkstra. Die **Warteschlange** liegt bei {big['queue'] / big['dijkstra']:.2f}-fach.")

st.markdown("---")

st.subheader("🔬 Die Reihenfolge der Kanten")
if st.button("Vier Kantenreihenfolgen vergleichen (dauert einen Moment)", key="order_start"):
    st.session_state["order_on"] = True
if st.session_state.get("order_on"):
    with st.spinner("Rechne vier Reihenfolgen × 2 Netztypen × 5 Netze ..."):
        orows = _order_table()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_order(orows), width="stretch", key="order_chart")
    names = {"as_given": "wie im Netz", "random": "zufällig", "near_first": "nahe zuerst", "far_first": "ferne zuerst"}
    c2.table({"Reihenfolge": [names[r["order"]] for r in orows], "Runden (Stadtnetz)": [f"{r['city_rounds']:.1f}" for r in orows], "Runden (Zufallsnetz)": [f"{r['random_rounds']:.1f}" for r in orows]})
    ob = {r["order"]: r for r in orows}
    st.caption(f"Früher Abbruch, Stadtnetz 20 × 20 und Zufallsnetz mit 400 Knoten, Mittel über 5 Netze. Das Ergebnis ist immer dasselbe, die Zahl der Runden nicht: {ob['near_first']['city_rounds']:.0f} Runden bei nahen Knoten zuerst, "
               f"{ob['far_first']['city_rounds']:.1f} bei fernen zuerst (Stadtnetz). Die Reihenfolge \"wie im Netz\" ist im **Stadtnetz** günstig ({ob['as_given']['city_rounds']:.1f} Runden, gegen {ob['random']['city_rounds']:.1f} bei zufälliger Reihenfolge), weil die Knotennummern dem Raster folgen; "
               f"im **Zufallsnetz** mit zufälligen Knotennummern ist sie kaum besser als zufällig ({ob['as_given']['random_rounds']:.1f} gegen {ob['random']['random_rounds']:.1f}). \"Nahe zuerst\" setzt die wahren Entfernungen voraus - das ist die Grenze, keine Möglichkeit.")

st.markdown("---")

st.subheader("🔬 Wann Dijkstra danebenliegt")
if st.button("Rekuperation und Potenziale durchfahren (dauert einen Moment)", key="wrong_start"):
    st.session_state["wrong_on"] = True
if st.session_state.get("wrong_on"):
    with st.spinner("Rechne 6 Stufen der Rückgewinnung und 6 Potenzialspannen × 5 Netze ..."):
        rrows, prows = _recuperation(), _potentials()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_recuperation(rrows), width="stretch", key="recuperation_chart")
    c2.table({"Potenzialspanne": [str(r["pot"]) for r in prows], "negative Kanten": [_pct(r["neg_share"], 1) for r in prows], "Dijkstra falsch (Knoten)": [_pct(r["wrong_share"], 1) for r in prows]})
    r0, r1 = rrows[3], prows[-1]
    same_rounds = len({round(r["rounds"], 6) for r in prows}) == 1
    st.caption(f"Links: E-Lieferwagen im 16 × 16-Netz mit 30 m Hügeln, Start unten links, Mittel über 5 Netze; bei 60 % Rückgewinnung sind {r0['neg_share']:.1%} der Kanten negativ und Dijkstra hat an {r0['wrong_share']:.1%} der Knoten falsche Kosten "
               f"(in {r0['seeds_with_alarm']} von {r0['seeds']} Netzen schlägt er Alarm). Es gab in keinem Netz einen negativen Zyklus. Rechts: Zufallsnetz mit 400 Knoten, vier Startknoten je Netz; bei Spanne {r1['pot']} sind {r1['neg_share']:.1%} der Kanten negativ und Dijkstra liegt an {r1['wrong_share']:.1%} der Knoten falsch. "
               + ("Die **Runden von Bellman-Ford bleiben bei jeder Spanne dieselben**: die Potenziale ändern nicht, welche Routen kürzeste sind, und heben sich in jedem Vergleich auf. Genau das nutzt **Johnson** (nächstes Stück): ein Mal Bellman-Ford rechnen, damit alle Kanten nichtnegativ machen, dann Dijkstra." if same_rounds else ""))

st.markdown("---")

st.subheader("🔬 Ein negativer Zyklus kostet Zeit")
if st.button("Erkennung des Zyklus vergleichen (dauert einen Moment)", key="cycle_start"):
    st.session_state["cycle_on"] = True
if st.session_state.get("cycle_on"):
    with st.spinner("Baue in 4 Netzgrößen × 5 Netze einen negativen Zyklus ein ..."):
        crows = _cycles()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_cycle(crows), width="stretch", key="cycle_chart")
    c2.table({"Knoten": [str(int(r["n"])) for r in crows], "Lehrbuch-Abbruch": [_num(r["final"]) for r in crows], "Vorgänger-Prüfung": [_num(r["check"]) for r in crows]})
    big = crows[2]
    st.caption(f"Zufallsnetz ohne negative Kanten, dazu ein negativer Dreieckszyklus; Mittel über 5 Netze; Kantenprüfungen bis zur Meldung. Die Lehrbuch-Regel bemerkt den Zyklus erst in Runde n - immer n · m Prüfungen ({_num(big['final'])} bei {int(big['n'])} Knoten, {big['final'] / big['dijkstra']:.0f}-mal Dijkstra), "
               f"die Warteschlange spart kaum etwas ({_num(big['queue'])}). Wer nach jeder Runde in den Vorgängern nach einem Kreis sucht (jeder Kreis dort ist ein negativer Zyklus), findet ihn nach {big['check_rounds']:.0f} Runden: {_num(big['check'])} Prüfungen, {big['final'] / big['check']:.0f}-mal weniger.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt keinen negativen Zyklus** | Dann gibt es keine kürzeste Route. Bellman-Ford meldet den Zyklus, aber nach Lehrbuch erst nach n · m Kantenprüfungen; die Vorgänger-Prüfung findet ihn in wenigen Runden. Dijkstra merkt nichts. | (nichts zu tun: das Problem hat keine Lösung) |
| **Der Aufwand ist tragbar** | Lehrbuch: n · m Prüfungen (im 40 × 40-Stadtnetz das 1 600-Fache von Dijkstra). Früher Abbruch: gut das 9-Fache, Warteschlange: knapp das 1.1-Fache - aber keine dieser Zahlen ist eine Garantie, die Reihenfolge entscheidet. | **Johnson**: einmal Bellman-Ford, dann Dijkstra |
| **Nur ein Start** | Bellman-Ford beantwortet die Entfernungen von **einem** Knoten zu allen. Für alle Paare müsste man n-mal rechnen. | **Floyd-Warshall**, **Johnson** |
| **Es gibt keine Nebenbedingungen** | Bellman-Ford kennt nur eine Kostenart; Zeit gegen Energie gleichzeitig ist ein anderes Problem. | **Mehrkriterien-Routing** |
"""
)
st.caption("Die Nachbarn der Kürzeste-Wege-Linie: Floyd-Warshall, Johnson und Mehrkriterien-Routing (noch nicht gebaut). Bereits gebaut: Breitensuche, Dijkstra, bidirektionale Suche und Contraction Hierarchies.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \in \mathbb{R}$ (negativ erlaubt), Start $s$, $n=|V|$, $m=|E|$. Gesucht: $\delta(v)=\min$ über alle Routen $s \leadsto v$ von der Summe der Kosten - oder die Feststellung, dass es keinen kleinsten Wert gibt.

**Lockerung.** $d(v) \leftarrow \min\bigl(d(v),\, d(u)+c_{uv}\bigr)$. Anfangswerte $d(s)=0$, sonst $\infty$. Es gilt immer $d(v) \ge \delta(v)$, denn jeder Wert $d(v)$ ist die Länge einer wirklichen Route.

**Invariante.** Nach Runde $k$ gilt (bei gleichzeitigem Lesen der Vorrundenwerte) $d_k(v)=\min\{\,c(P) : P \text{ Route } s \leadsto v \text{ mit höchstens } k \text{ Kanten}\,\}$. Beweis durch Induktion: eine Route mit $k$ Kanten besteht aus einer Route mit $k-1$ Kanten zu einem Vorgänger $u$ und der Kante $(u,v)$; die Runde $k$ probiert genau diese Kante.
Bei sofortiger Wirkung der Lockerungen ("in place") gilt nur $d_k(v) \le$ dieser Wert - schneller, nie schlechter.

**Abbruch nach $n-1$ Runden.** Ohne negativen Zyklus hat eine kürzeste Route keinen Knoten doppelt (ein Kreis kostet $\ge 0$ und ließe sich weglassen), also höchstens $n-1$ Kanten. Nach Runde $n-1$ ist $d=\delta$, und keine Kante lässt sich mehr lockern.

**Negativer Zyklus.** Gibt es einen von $s$ erreichbaren Zyklus $Z$ mit $c(Z)<0$, lässt sich in jeder Runde noch eine Kante von $Z$ lockern: wäre für alle $(u,v)\in Z$ schon $d(v)\le d(u)+c_{uv}$, ergäbe Summieren über $Z$ die Ungleichung $0 \le c(Z)$, ein Widerspruch.
Die Prüfrunde $n$ findet also eine Verbesserung. Den Zyklus liefern die Vorgänger-Zeiger: **jeder Kreis in den Zeigern ist ein negativer Zyklus** (Summe der Ungleichungen $d(v) \ge d(\pi(v))+c_{\pi(v)v}$ um den Kreis, mit strenger Ungleichung bei der zuletzt gesetzten Kante) - das ist die Vorgänger-Prüfung.

**Aufwand.** Lehrbuch $\Theta(n\cdot m)$, früher Abbruch $O(m\cdot r)$ mit der Rundenzahl $r \le n$ (bei ungünstiger Reihenfolge $r$ = größte Kantenzahl einer kürzesten Route plus 1), Warteschlange im schlechtesten Fall wieder $O(n\cdot m)$.

**Potenziale (Vorgriff auf Johnson).** Für beliebige $p:V\to\mathbb{R}$ ersetze $c'_{uv}=c_{uv}+p(u)-p(v)$. Jede Route $P: s\leadsto v$ ändert ihre Kosten um $p(s)-p(v)$: **die kürzesten Routen bleiben dieselben**, jeder Kreis behält seine Kosten (kein neuer negativer Zyklus).
Bellman-Ford verhält sich deshalb bei jedem $p$ gleich (jede Lockerung $d(u)+c_{uv}<d(v)$ ist zu $d'(u)+c'_{uv}<d'(v)$ äquivalent) - Dijkstra nicht. Wählt man $p=\delta$ (aus einem Bellman-Ford-Lauf ab einem Hilfsknoten), sind alle $c'\ge 0$.

Implementiert in `bf_graph.py` (CSR-Graph), `bf_queues.py` (Warteschlange für Dijkstra), `bf_algorithm.py` (vier Varianten, Kantenreihenfolgen, Vorgänger-Prüfung, Dijkstra mit Alarm), `bf_scenario.py` (Netze), `bf_evaluation.py` (Kennzahlen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)

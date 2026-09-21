"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Varianten und Kantenreihenfolgen, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bf_constants as C
from bf_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
# Urteil je Preset: success = Bellman-Ford findet, was Dijkstra verfehlt; warning = negativer Zyklus; info = kein Unterschied (gemessen, siehe test_claims)
EXPECTED_KIND = {"🚚 Frachtnetz": "success", "💱 Devisen mit Kursfehler": "warning", "💱 Devisen ohne Kursfehler": "info", "🔋 E-Lieferwagen": "success", "🕸️ Zufallsnetz": "success", "🏙️ Stadtnetz": "info"}


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _kinds(at):
    return {"success": len(at.success), "warning": len(at.warning), "info": len(at.info)}


def test_default_renders_without_exception():
    at = _run()
    assert any("Bellman-Ford in Aktion" in m.value for m in at.markdown)
    assert _kinds(at) == {"success": 1, "warning": 0, "info": 0}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    want = {"success": 0, "warning": 0, "info": 0}
    want[EXPECTED_KIND[name]] = 1
    assert _kinds(at) == want


@pytest.mark.parametrize("variant", list(C.VARIANT_LABELS))
@pytest.mark.parametrize("order", list(C.ORDER_LABELS))
def test_every_variant_and_edge_order_renders_on_the_small_and_the_ev_net(variant, order):
    def small(at):
        at.session_state["variant_select"] = variant
        at.session_state["order_select"] = order

    def ev(at):
        small(at)
        at.session_state["net_select"] = "ev"
        at.session_state["side_slider"] = 10
    for setup in (small, ev):
        at = _run(setup)
        assert len(at.success) + len(at.warning) + len(at.info) == 1


@pytest.mark.parametrize("variant", list(C.VARIANT_LABELS))
def test_a_negative_cycle_is_reported_by_every_variant(variant):
    def setup(at):
        at.session_state["net_select"] = "fx_arbitrage"
        at.session_state["variant_select"] = variant
    at = _run(setup)
    assert len(at.warning) == 1 and "Negativer Zyklus" in at.warning[0].value


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big(at):
        at.session_state["net_select"] = "ev"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["variant_select"] = "textbook"
        at.session_state["hill_slider"] = C.HILL_MAX
        at.session_state["eta_slider"] = C.ETA_MAX

    def flat(at):
        at.session_state["net_select"] = "ev"
        at.session_state["hill_slider"] = C.HILL_MIN

    def rnd(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = C.NODES_MAX
        at.session_state["pot_slider"] = C.POT_MAX
        at.session_state["variant_select"] = "queue"
    for setup in (small, big, flat, rnd):
        at = _run(setup)
        assert at.slider(key="bf_step").value == at.slider(key="bf_step").max


def test_hidden_controls_follow_the_net_and_the_variant():
    def labels_for(net, variant=None):
        def setup(at):
            at.session_state["variant_select"] = variant
        return _labels(_run(setup, net=net) if variant else _run(net=net))
    small, ev, rnd, city = (labels_for(n) for n in ("delivery", "ev", "random", "city"))
    assert small == {"Netz", "Variante von Bellman-Ford", "Reihenfolge der Kanten"}
    assert ev == small | {"Kreuzungen je Seite", "Hügel [m Höhenunterschied]", "Rückgewinnung bergab [%]", "Zufalls-Seed"}
    assert rnd == small | {"Knoten", "Mittlerer Grad", "Potenzialspanne", "Zufalls-Seed"}
    assert city == small | {"Kreuzungen je Seite", "Reichweite der Straßen [Blocklängen]", "Streuung der Kosten", "Zufalls-Seed"}
    assert "Reihenfolge der Kanten" not in labels_for("delivery", "queue") and "Reihenfolge der Kanten" not in labels_for("delivery", "synchronous")


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="ev")
    at.session_state["hill_slider"] = 12
    at.run()
    at.session_state["net_select"] = "delivery"
    at.run()
    at.session_state["net_select"] = "ev"
    at.run()
    assert not at.exception and at.slider(key="hill_slider").value == 12


def test_hidden_order_value_survives_a_variant_without_an_order():
    at = _run(net="ev")
    at.session_state["order_select"] = "random"
    at.run()
    at.session_state["variant_select"] = "queue"
    at.run()
    at.session_state["variant_select"] = "early_stop"
    at.run()
    assert not at.exception and at.selectbox(key="order_select").value == "random"


def test_step_slider_returns_to_the_last_step_when_anything_changes():
    at = _run(net="ev")
    at.slider(key="bf_step").set_value(1)
    at.run()
    assert at.slider(key="bf_step").value == 1
    at.session_state["variant_select"] = "synchronous"
    at.run()
    assert not at.exception and at.slider(key="bf_step").value == at.slider(key="bf_step").max


def test_every_step_of_the_small_net_renders_for_every_variant():
    for variant in C.VARIANT_LABELS:
        at = _run(lambda a: a.session_state.__setitem__("variant_select", variant))
        for k in range(0, int(at.slider(key="bf_step").max) + 1):
            at.slider(key="bf_step").set_value(k)
            at.run()
            assert not at.exception, (variant, k)


def test_dijkstra_route_checkbox_is_only_shown_without_a_cycle():
    assert _run().checkbox(key="show_dj").value
    at = _run(net="fx_arbitrage")
    assert not at.checkbox
    at = _run(net="ev")
    at.checkbox(key="show_dj").set_value(False)
    at.run()
    assert not at.exception


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "random"
    at.query_params["nodes"] = "999999"
    at.query_params["var"] = "never"
    at.query_params["order"] = "random"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "random"
    assert at.slider(key="nodes_slider").value == C.NODES_MAX and at.selectbox(key="variant_select").value == C.DEFAULT_VARIANT and at.selectbox(key="order_select").value == "random"


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 feste Netze" in c.value for c in at.caption)
    for key in ("effort_start", "order_start", "wrong_start", "cycle_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Das **Lehrbuch** prüft immer n · m Kanten", "Das Ergebnis ist immer dasselbe, die Zahl der Runden nicht", "bei 60 % Rückgewinnung sind", "Die Lehrbuch-Regel bemerkt den Zyklus erst in Runde n"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # das Netz wird je nach Netz an einer von zwei Stellen gezeichnet (kleine Netze mit Tabelle, große mit Verlauf), nie an beiden
    assert len(calls) == 7 and sorted(set(keys)) == sorted(["net_chart", "progress_chart", "effort_chart", "order_chart", "recuperation_chart", "cycle_chart"]) and keys.count("net_chart") == 2, keys
    viz = (ROOT / "bf_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]

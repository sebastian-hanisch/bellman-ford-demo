"""Plotly-Abbildungen: Netz nach k Runden (Entfernung als Farbe, gelockerte Kanten gelb, negative Kanten grün, Zyklus rot), Verbesserungen je Runde, Experimente. Achsen sind gesperrt (fixedrange),
damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import bf_constants as C

ORDER_SHORT = {"as_given": "wie im Netz", "random": "zufällig", "near_first": "nahe zuerst", "far_first": "ferne zuerst"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def state_at(a, k):
    """Zustand nach Schritt k (Runde bzw. Zwischenstand der Warteschlange): Entfernungen, Vorgänger, in diesem Schritt gelockerte Kanten (u, v, alt, neu)."""
    g, bf = a.net.graph, a.bf
    if k <= 0 or not bf.history:
        dist = np.full(g.n, np.inf)
        dist[bf.source] = 0.0
        return dist, np.full(g.n, -1), []
    dist, parent, changed = bf.history[min(k, len(bf.history)) - 1]
    return np.array(dist), np.array(parent), changed


def table_state(a, k):
    """Kleine Netze: Tabellenzeilen (Knoten, Entfernung, Vorgänger, geändert) nach Schritt k."""
    g = a.net.graph
    dist, parent, changed = state_at(a, k)
    touched = {v for _, v, _, _ in changed}
    return [{"Knoten": g.names[v], "Entfernung": None if not np.isfinite(dist[v]) else float(dist[v]), "Vorgänger": g.names[parent[v]] if parent[v] >= 0 else "–", "geändert": v in touched} for v in range(g.n)]


def _segments(g, mask=None):
    """Kanten als eine Linienspur (None trennt die Segmente); bei ungerichteten Netzen Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    keep = np.ones(len(src), dtype=bool) if mask is None else mask.copy()
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _shifted(g, u, v, amount=0.11):
    """Endpunkte der Kante u -> v, ein Stück nach rechts der Fahrtrichtung versetzt, damit Hin- und Rückrichtung getrennt zu sehen sind."""
    p, q = g.xy[u], g.xy[v]
    d = q - p
    norm = float(np.hypot(*d)) or 1.0
    off = np.array([d[1], -d[0]]) / norm * amount
    return p + off, q + off


def _fmt(x):
    return f"{x:g}"


def build_network(net, a, k, show_dijkstra=False, height=520):
    """Das Netz nach `k` Runden: Knoten nach Entfernung gefärbt (grau = noch nicht erreicht), in dieser Runde gelockerte Kanten gelb, negative Kanten grün, bei einem negativen Zyklus der Zyklus in Rot;
    nach der letzten Runde erscheint die Route zum Ziel (schwarz, bei Bedarf die von Dijkstra blau gestrichelt daneben)."""
    g, bf = net.graph, a.bf
    small = bool(g.names)
    labels = small and g.m <= 20                                     # Kosten an den Kanten nur bei wenigen Kanten (die Devisennetze haben 30)
    fig = go.Figure()
    dist, parent, changed = state_at(a, k)
    last = k >= len(bf.history)
    neg = g.weight < 0
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip", showlegend=False))
    if small and g.directed:
        src = np.repeat(np.arange(g.n), g.degree())
        both = {(int(u), int(v)) for u, v in zip(src, g.indices)}
        for u, v, w in zip(src.tolist(), g.indices.tolist(), g.weight.tolist()):
            p, q = _shifted(g, u, v) if (v, u) in both else (g.xy[u], g.xy[v])
            col = "rgba(44,160,44,0.9)" if w < 0 else "rgba(120,120,120,0.55)"
            fig.add_annotation(x=q[0], y=q[1], ax=p[0], ay=p[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.3 if w >= 0 else 2.2,
                               arrowcolor=col, standoff=13, startstandoff=13)
            if labels:
                mid = (p + q) / 2
                fig.add_annotation(x=mid[0], y=mid[1], text=_fmt(w), showarrow=False, font=dict(size=12, color="#1b7f1b" if w < 0 else "#555"), bgcolor="rgba(255,255,255,0.8)")
    elif neg.any() and net.geometric:
        nx_, ny_ = _segments(g, neg)
        fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="lines", line=dict(color="rgba(44,160,44,0.9)", width=2), name="negative Kanten", hoverinfo="skip"))
    if changed and not small:
        seg = [(u, v) for u, v, _, _ in changed[:400]]
        x = np.full(3 * len(seg), None, dtype=object)
        y = np.full(3 * len(seg), None, dtype=object)
        for i, (u, v) in enumerate(seg):
            x[3 * i], x[3 * i + 1], y[3 * i], y[3 * i + 1] = g.xy[u, 0], g.xy[v, 0], g.xy[u, 1], g.xy[v, 1]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C.COLORS["changed"], width=3), name="in dieser Runde gelockert", hoverinfo="skip"))
    elif changed and small:
        for u, v, _, _ in changed:
            p, q = _shifted(g, u, v) if g.arc(v, u) >= 0 else (g.xy[u], g.xy[v])
            fig.add_trace(go.Scatter(x=[p[0], q[0]], y=[p[1], q[1]], mode="lines", line=dict(color=C.COLORS["changed"], width=6), hoverinfo="skip", showlegend=False, opacity=0.8))
    finite = np.isfinite(dist)
    if bf.negative_cycle and last and bf.cycle:
        pts = g.xy[bf.cycle]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["cycle"], width=6), name=f"negativer Zyklus ({_fmt(bf.cycle_cost)} {net.unit})", hoverinfo="skip"))
    if small:
        text = [f"{g.names[v]}<br>{_fmt(dist[v]) if finite[v] else '∞'}" for v in range(g.n)]
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers+text", showlegend=False, text=text, textposition="top center", hoverinfo="skip",
                                 marker=dict(size=15, color=np.where(finite, dist, np.nan) if finite.any() else "white", colorscale="Viridis", cmin=float(np.nanmin(dist[finite])) if finite.any() else 0,
                                             cmax=float(np.nanmax(dist[finite])) if finite.any() and np.nanmax(dist[finite]) > np.nanmin(dist[finite]) else 1, line=dict(color="gray", width=1.5),
                                             showscale=False)))
    else:
        size = 3 if g.n > 1500 else (5 if g.n > 600 else 7)
        rest = np.where(~finite)[0]
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(size=size, color="rgba(200,200,200,0.9)")))
        got = np.where(finite)[0]
        fig.add_trace(go.Scatter(x=g.xy[got, 0], y=g.xy[got, 1], mode="markers", showlegend=False, customdata=dist[got], hovertemplate=f"Kosten vom Start: %{{customdata:g}} {net.unit}<extra></extra>",
                                 marker=dict(size=size, color=dist[got], colorscale="Viridis", cmin=float(dist[got].min()), cmax=max(float(dist[got].max()), float(dist[got].min()) + 1),
                                             colorbar=dict(title=f"Kosten<br>[{net.unit}]", thickness=12, len=0.6))))
    t = a.target
    if last and not bf.negative_cycle:
        if show_dijkstra and a.dj.route(t):
            pts = g.xy[a.dj.route(t)]
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["dijkstra"], width=4, dash="dash"), name="Route von Dijkstra", hoverinfo="skip"))
        route = bf.route(t)
        if route:
            pts = g.xy[route]
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["bf"], width=5), name="Route von Bellman-Ford", hoverinfo="skip"))
    for node, name, color, symbol in ((bf.source, "Start", C.COLORS["start"], "diamond"), (t, "Ziel", C.COLORS["goal"], "star")):
        label = g.names[node] if small else f"{node}"
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {label}", hoverinfo="skip", marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if not small or not net.geometric:                               # Karten und der Devisenkreis bleiben maßstabsgetreu; das kleine Frachtnetz ist ein Schema und füllt die Fläche
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]])
        fig.update_yaxes(range=[lo[1] - pad[1], hi[1] + pad[1] + 0.05 * (hi[1] - lo[1])])
    return _base(fig, height)


def build_progress(a, k, height=260):
    """Verbesserungen je Runde (bzw. je Zwischenstand der Warteschlange): am Anfang viele, dann klingen sie ab - die letzte Runde ohne Verbesserung ist der Beweis, dass es fertig ist."""
    bf = a.bf
    if a.bf.improved_per_round:
        y, xlabel = bf.improved_per_round, "Runde"
    else:
        y, xlabel = [len(ch) for _, _, ch in bf.history], "Zwischenstand (Warteschlange)"
    x = np.arange(1, len(y) + 1)
    colors = ["#d62728" if i <= k else "#f0c4c4" for i in x]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors, hovertemplate=f"{xlabel} %{{x}}: %{{y}} Verbesserungen<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Verbesserungen")
    return _base(fig, height)


def build_effort(rows, height=340):
    """Kantenprüfungen gegen die Zahl der Kanten des Netzes (log-log): Lehrbuch wächst wie n · m, früher Abbruch wie m mal Runden, Warteschlange und Dijkstra fast wie m."""
    x = [r["m"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("textbook", "Lehrbuch (n − 1 Runden)", "#7f7f7f", "dot"), ("early_random", "früher Abbruch, zufällige Reihenfolge", "#d62728", "solid"),
                                   ("early_stop", "früher Abbruch, wie im Netz", "#ff7f0e", "solid"), ("queue", "Warteschlange", "#2ca02c", "solid"), ("dijkstra", "Dijkstra", "#1f77b4", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="Kanten im Netz", type="log"), yaxis=dict(title="Kantenprüfungen", type="log"))
    return _base(fig, height)


def build_order(rows, height=300):
    fig = go.Figure()
    for key, name, color in (("city_rounds", "Stadtnetz (Knotennummern folgen dem Raster)", "#1f77b4"), ("random_rounds", "Zufallsnetz (Knotennummern zufällig)", "#ff7f0e")):
        fig.add_trace(go.Bar(x=[ORDER_SHORT[r["order"]] for r in rows], y=[r[key] for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="group", xaxis_title="Kantenreihenfolge", yaxis_title="Runden bis zum Abbruch")
    return _base(fig, height)


def build_recuperation(rows, height=320):
    x = [f"{r['eta']} %" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[r["neg_share"] * 100 for r in rows], name="negative Kanten [%]", marker_color=C.COLORS["negative"]))
    fig.add_trace(go.Bar(x=x, y=[r["wrong_share"] * 100 for r in rows], name="Knoten mit falschen Kosten bei Dijkstra [%]", marker_color=C.COLORS["dijkstra"]))
    fig.update_layout(barmode="group", xaxis_title="Rückgewinnung beim Bergabfahren", yaxis_title="Anteil [%]")
    return _base(fig, height)


def build_cycle(rows, height=340):
    x = [r["n"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("final", "Lehrbuch-Abbruch (Runde n)", "#7f7f7f", "dot"), ("queue", "Warteschlange", "#2ca02c", "solid"), ("check", "Vorgänger-Prüfung nach jeder Runde", "#d62728", "solid"), ("dijkstra", "Dijkstra (nur zum Vergleich)", "#1f77b4", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Kantenprüfungen bis der Zyklus gemeldet wird", type="log"))
    return _base(fig, height)

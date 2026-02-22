"""Before/after what-if comparison plot."""

from __future__ import annotations

from typing import Any, Literal, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.schemas import LineFlow, PowerFlowResult
from viz.network_plot import Theme, build_graph, bus_display_id, compute_layout
from viz.voltage_heatmap import _voltage_colorscale


def _bus_vm_map(result: PowerFlowResult) -> dict[int, float]:
    return {int(b.bus_id): float(b.vm_pu) for b in result.bus_voltages}


def _bus_violation_set(result: PowerFlowResult) -> set[int]:
    return {int(b.bus_id) for b in result.voltage_violations}


def _line_flow_map(result: PowerFlowResult) -> dict[int, LineFlow]:
    return {int(l.line_id): l for l in result.line_flows}


def _line_id(kind: str, element_id: int) -> int:
    return int(element_id) if kind == "line" else 100000 + int(element_id)


def make_comparison(
    net: Any,
    before: PowerFlowResult,
    after: PowerFlowResult,
    *,
    positions: Optional[dict[int, tuple[float, float]]] = None,
    theme: Theme = "light",
    lang: Literal["zh", "en"] = "zh",
    topk: int = 5,
    vmin: float = 0.90,
    vmax: float = 1.10,
) -> go.Figure:
    """Side-by-side comparison with explicit change highlighting."""

    g = build_graph(net)
    if positions is None:
        positions = compute_layout(net, g)

    bus_idxs = list(net.bus.index.tolist())
    bus_ids = [int(bus_display_id(net, int(i))) for i in bus_idxs]
    bus_pos = {int(bid): (positions[int(i)][0], positions[int(i)][1]) for i, bid in zip(bus_idxs, bus_ids)}

    vm_before = _bus_vm_map(before)
    vm_after = _bus_vm_map(after)
    viol_before = _bus_violation_set(before)
    viol_after = _bus_violation_set(after)

    xs = [bus_pos[bid][0] for bid in bus_ids]
    ys = [bus_pos[bid][1] for bid in bus_ids]
    vb = [float(vm_before.get(bid, 1.0)) for bid in bus_ids]
    va = [float(vm_after.get(bid, 1.0)) for bid in bus_ids]

    bus_delta_v: dict[int, float] = {bid: float(vm_after.get(bid, 1.0) - vm_before.get(bid, 1.0)) for bid in bus_ids}
    changed_bus_ids = {
        bid
        for bid in bus_ids
        if abs(bus_delta_v[bid]) >= 0.005 or ((bid in viol_before) != (bid in viol_after))
    }

    line_before = _line_flow_map(before)
    line_after = _line_flow_map(after)

    changed_edges: list[dict[str, Any]] = []
    edge_delta_candidates: list[dict[str, Any]] = []
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v, data in g.edges(data=True):
        x0, y0 = positions[int(u)]
        x1, y1 = positions[int(v)]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        kind = str(data.get("kind", "line"))
        eid = int(data.get("element_id", -1))
        lid = _line_id(kind, eid)
        bf = line_before.get(lid)
        af = line_after.get(lid)
        if bf is None and af is None:
            continue

        b_p = float(bf.p_from_mw) if bf is not None else 0.0
        a_p = float(af.p_from_mw) if af is not None else 0.0
        b_loading = float(bf.loading_percent) if bf is not None else 0.0
        a_loading = float(af.loading_percent) if af is not None else 0.0
        b_vi = bool(bf.is_violation) if bf is not None else False
        a_vi = bool(af.is_violation) if af is not None else False

        delta_p = a_p - b_p
        delta_loading = a_loading - b_loading
        edge_info = {
            "kind": kind,
            "eid": eid,
            "lid": lid,
            "from_bus": int(bus_display_id(net, int(u))),
            "to_bus": int(bus_display_id(net, int(v))),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "before_p": b_p,
            "after_p": a_p,
            "delta_p": delta_p,
            "before_loading": b_loading,
            "after_loading": a_loading,
            "delta_loading": delta_loading,
            "before_violation": b_vi,
            "after_violation": a_vi,
        }
        edge_delta_candidates.append(edge_info)

        changed = abs(delta_p) >= 1.0 or abs(delta_loading) >= 2.0 or (a_vi != b_vi)
        if not changed:
            continue

        changed_edges.append(edge_info)

    changed_edges.sort(
        key=lambda e: max(abs(float(e["delta_loading"])), abs(float(e["delta_p"]))),
        reverse=True,
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=(
            "Before" if lang == "en" else "修改前",
            "After" if lang == "en" else "修改后",
            "Top Changes" if lang == "en" else "关键变化",
        ),
        row_heights=[0.42, 0.42, 0.16],
        vertical_spacing=0.08,
    )

    # Base network context in both panels.
    base_edge = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(160,160,160,0.45)", width=1.8),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_trace(base_edge, row=1, col=1)
    fig.add_trace(base_edge, row=2, col=1)

    # Highlight changed branches: subtle in "before", strong in "after".
    highlight_edges = list(changed_edges)
    if not highlight_edges:
        fallback_edges = [e for e in edge_delta_candidates if abs(float(e["delta_p"])) > 1e-9]
        fallback_edges.sort(key=lambda e: abs(float(e["delta_p"])), reverse=True)
        highlight_edges = fallback_edges[: max(1, int(topk))]

    for e in highlight_edges:
        fig.add_trace(
            go.Scatter(
                x=[e["x0"], e["x1"]],
                y=[e["y0"], e["y1"]],
                mode="lines",
                line=dict(color="#DD6B20", width=3, dash="dot"),
                hovertemplate=(
                    f"{e['kind']}#{e['eid']} {e['from_bus']}->{e['to_bus']}<br>"
                    f"P={e['before_p']:.2f} MW, loading={e['before_loading']:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[e["x0"], e["x1"]],
                y=[e["y0"], e["y1"]],
                mode="lines",
                line=dict(color="#d62728", width=4, dash="dash"),
                hovertemplate=(
                    f"{e['kind']}#{e['eid']} {e['from_bus']}->{e['to_bus']}<br>"
                    f"P={e['after_p']:.2f} MW, loading={e['after_loading']:.1f}%<br>"
                    f"Delta P={e['delta_p']:+.2f} MW, Delta loading={e['delta_loading']:+.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Node traces (voltage color) + change highlight.
    node_line_before = ["#d62728" if bid in changed_bus_ids else "rgba(0,0,0,0.3)" for bid in bus_ids]
    node_line_after = ["#d62728" if bid in changed_bus_ids else "rgba(0,0,0,0.3)" for bid in bus_ids]
    node_line_w_before = [3 if bid in changed_bus_ids else 1 for bid in bus_ids]
    node_line_w_after = [3 if bid in changed_bus_ids else 1 for bid in bus_ids]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[str(bid) for bid in bus_ids],
            textposition="top center",
            marker=dict(
                color=vb,
                colorscale=_voltage_colorscale(),
                cmin=vmin,
                cmax=vmax,
                size=14,
                line=dict(color=node_line_before, width=node_line_w_before),
            ),
            hovertemplate="bus %{text}<br>V(before)=%{marker.color:.4f} pu<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[str(bid) for bid in bus_ids],
            textposition="top center",
            customdata=[[float(bus_delta_v.get(bid, 0.0))] for bid in bus_ids],
            marker=dict(
                color=va,
                colorscale=_voltage_colorscale(),
                cmin=vmin,
                cmax=vmax,
                size=14,
                line=dict(color=node_line_after, width=node_line_w_after),
                colorbar=dict(title="V (p.u.)"),
            ),
            hovertemplate=(
                "bus %{text}<br>"
                "V(after)=%{marker.color:.4f} pu<br>"
                "Delta V=%{customdata[0]:+.4f} pu"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # In-plot numeric annotations for key changes.
    n_bus = int(len(bus_ids))
    bus_anno_n = max(3, int(topk))
    edge_anno_n = max(4, int(topk))
    if n_bus > 20:
        bus_anno_n = min(bus_anno_n, 4)
        edge_anno_n = min(edge_anno_n, 2)
    if n_bus > 60:
        bus_anno_n = min(bus_anno_n, 2)
        edge_anno_n = 0

    bus_rank = sorted(changed_bus_ids, key=lambda bid: abs(bus_delta_v[bid]), reverse=True)
    for idx, bid in enumerate(bus_rank[:bus_anno_n]):
        x, y = bus_pos[bid]
        dv = bus_delta_v[bid]
        fig.add_annotation(
            x=x,
            y=y,
            row=2,
            col=1,
            text=f"Delta V={dv:+.3f}" if lang == "en" else f"变化 ΔV={dv:+.3f}",
            showarrow=True,
            arrowhead=0,
            ay=(-18 if idx % 2 == 0 else -28),
            font=dict(size=10, color="#d62728"),
            bgcolor="rgba(255,245,245,0.85)",
            bordercolor="#d62728",
            borderwidth=1,
        )

    edge_anno_source = changed_edges
    if not edge_anno_source:
        edge_anno_source = [e for e in edge_delta_candidates if abs(float(e["delta_p"])) > 1e-9]
        edge_anno_source.sort(key=lambda e: abs(float(e["delta_p"])), reverse=True)

    for idx, e in enumerate(edge_anno_source[:edge_anno_n]):
        fig.add_annotation(
            x=(float(e["x0"]) + float(e["x1"])) / 2.0,
            y=(float(e["y0"]) + float(e["y1"])) / 2.0,
            row=2,
            col=1,
            text=f"Delta P {float(e['delta_p']):+.1f} MW" if lang == "en" else f"变化 ΔP {float(e['delta_p']):+.1f} MW",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=(-12 if idx % 2 == 0 else 12),
            font=dict(size=9, color="#d62728"),
            bgcolor="rgba(255,245,245,0.8)",
            borderpad=1,
        )

    table_rows: list[list[str]] = []
    top_n = max(1, int(topk))
    for bid in bus_rank[:top_n]:
        v_before = float(vm_before.get(bid, 1.0))
        v_after = float(vm_after.get(bid, 1.0))
        vchg = (
            f"{'Y' if bid in viol_before else 'N'}→{'Y' if bid in viol_after else 'N'}"
            if lang == "en"
            else f"{'是' if bid in viol_before else '否'}→{'是' if bid in viol_after else '否'}"
        )
        table_rows.append(
            [
                "Bus",
                str(bid),
                f"{v_before:.4f} p.u.",
                f"{v_after:.4f} p.u.",
                f"{(v_after - v_before):+.4f}",
                vchg,
            ]
        )

    for e in changed_edges[:top_n]:
        vchg = (
            f"{'Y' if e['before_violation'] else 'N'}→{'Y' if e['after_violation'] else 'N'}"
            if lang == "en"
            else f"{'是' if e['before_violation'] else '否'}→{'是' if e['after_violation'] else '否'}"
        )
        table_rows.append(
            [
                "Branch",
                f"{e['from_bus']}->{e['to_bus']}",
                f"P={float(e['before_p']):.2f}MW / {float(e['before_loading']):.1f}%",
                f"P={float(e['after_p']):.2f}MW / {float(e['after_loading']):.1f}%",
                f"ΔP={float(e['delta_p']):+.2f}MW, ΔL={float(e['delta_loading']):+.1f}%",
                vchg,
            ]
        )

    if not table_rows:
        table_rows = [["-", "-", "-", "-", "-", "-"]]

    headers = (
        ["Type", "Element", "Before", "After", "Delta", "Violation"]
        if lang == "en"
        else ["类型", "元件", "修改前", "修改后", "变化量", "越限变化"]
    )
    fig.add_trace(
        go.Table(
            header=dict(values=headers, align="left"),
            cells=dict(values=[[r[i] for r in table_rows] for i in range(len(headers))], align="left"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=(f"What-if Comparison - {after.case_name}" if lang == "en" else f"修改前后对比 - {after.case_name}"),
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=1600,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    for r, c in [(1, 1), (2, 1)]:
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=r, col=c)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=r, col=c)

    return fig

"""viz/voltage_heatmap.py

电压热力图（节点颜色按电压幅值编码）。

PRD 颜色语义：
- 深绿 (0.98-1.02): 非常健康
- 浅绿 (0.97-0.98, 1.02-1.03): 健康
- 黄色 (0.95-0.97, 1.03-1.05): 注意
- 红色 (<0.95 或 >1.05): 越限

实现策略：
- 使用自定义连续 colorscale，并在节点 marker 的 line 上高亮越限。
- 支持传入 positions，保证与其它图共享布局。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import plotly.graph_objects as go

from models.schemas import PowerFlowResult
from viz.network_plot import STYLE, Theme, bus_display_id, make_base_network_figure


def _voltage_colorscale() -> list[list[float | str]]:
    """连续色标：低电压红 -> 黄 -> 绿 -> 黄 -> 红。"""
    # 约束在 [0.90, 1.10] 展示窗口内
    # 0.90 red
    # 0.95 yellow
    # 0.98 deep green
    # 1.02 deep green
    # 1.05 yellow
    # 1.10 red
    return [
        [0.00, "#d62728"],  # red
        [0.25, "#ffdd57"],  # yellow
        [0.40, "#1a9850"],  # deep green
        [0.60, "#1a9850"],
        [0.75, "#ffdd57"],
        [1.00, "#d62728"],
    ]


def make_voltage_heatmap(
    net: Any,
    result: PowerFlowResult,
    *,
    positions: Optional[dict[int, tuple[float, float]]] = None,
    theme: Theme = "light",
    lang: Literal["zh", "en"] = "en",
    vmin: float = 0.90,
    vmax: float = 1.10,
) -> go.Figure:
    """生成电压热力图 Figure。"""

    title = (
        f"Voltage Heatmap - {result.case_name}"
        if str(lang).lower().startswith("en")
        else f"电压热力图 - {result.case_name}"
    )
    fig, positions = make_base_network_figure(
        net,
        result,
        positions=positions,
        theme=theme,
        title=title,
        show_bus_labels=True,
    )

    # node trace 是第 3 个 trace
    node_trace = fig.data[2]

    # 取 bus_idx 顺序对应 node_trace 的点顺序
    bus_idxs = list(net.bus.index.tolist())

    # 构建 bus_id -> vm_pu
    vm_map = {int(b.bus_id): float(b.vm_pu) for b in result.bus_voltages}
    # 按 node_trace 点顺序生成电压值
    vm_vals: list[float] = []
    is_violation: list[bool] = []
    for bi in bus_idxs:
        bid = int(bus_display_id(net, int(bi)))
        vm = float(vm_map.get(bid, 1.0))
        vm_vals.append(vm)
        # 用 result 中标注的越限
        bv = next((x for x in result.bus_voltages if int(x.bus_id) == int(bid)), None)
        is_violation.append(bool(bv.is_violation) if bv is not None else False)

    # 更新 marker：颜色为电压值；越限节点红色边框
    node_trace.marker.color = vm_vals
    node_trace.marker.colorscale = _voltage_colorscale()
    node_trace.marker.cmin = vmin
    node_trace.marker.cmax = vmax
    node_trace.marker.colorbar = dict(title="V (p.u.)")
    node_trace.marker.line.color = [STYLE.violation_color if v else "rgba(0,0,0,0.35)" for v in is_violation]
    node_trace.marker.line.width = [3 if v else 1 for v in is_violation]

    fig.update_layout(
        title=title,
    )

    return fig

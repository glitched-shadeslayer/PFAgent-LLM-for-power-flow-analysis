import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from solver import case_loader
from solver.power_flow import run_power_flow, modify_bus_load

from viz import (
    make_comparison,
    make_flow_diagram,
    make_voltage_heatmap,
    make_violation_overview,
)

from llm.tools import ToolContext, build_default_dispatcher


@pytest.mark.parametrize("case_name", ["case14", "case30", "case57"])
def test_voltage_heatmap_returns_figure(case_name: str):
    net, _ = case_loader.load(case_name)
    res = run_power_flow(net)
    fig = make_voltage_heatmap(net, res)
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) >= 1
    assert case_name in fig.layout.title.text


@pytest.mark.parametrize("case_name", ["case14", "case30"])
def test_flow_diagram_returns_figure(case_name: str):
    net, _ = case_loader.load(case_name)
    res = run_power_flow(net)
    fig = make_flow_diagram(net, res)
    assert fig is not None
    assert len(fig.data) >= 1
    assert case_name in (fig.layout.title.text or "")


def test_violation_overview_returns_figure():
    net, _ = case_loader.load("case14")
    res = run_power_flow(net)
    fig = make_violation_overview(net, res)
    assert fig is not None
    assert "case14" in (fig.layout.title.text or "")


def test_comparison_returns_figure_with_table():
    net, _ = case_loader.load("case14")
    before = run_power_flow(net)

    # make a change
    load_idx = int(net.load.index[0])
    bus_internal = int(net.load.at[load_idx, "bus"])
    bus_name = int(net.bus.at[bus_internal, "name"])
    old_p = float(net.load.loc[net.load["bus"] == bus_internal, "p_mw"].sum())
    after = modify_bus_load(net, bus_id=bus_name, p_mw=old_p + 5.0)

    fig = make_comparison(net, before, after)
    assert fig is not None
    assert "case14" in (fig.layout.title.text or "")
    # last trace should be table
    assert any(t.type == "table" for t in fig.data)


def test_generate_plot_tool_outputs_json():
    ctx = ToolContext()
    dispatcher = build_default_dispatcher(ctx)

    # load and run
    out1 = dispatcher.dispatch("load_case", {"case_name": "case14"})
    assert "case14" in out1
    out2 = dispatcher.dispatch("run_powerflow", {})
    assert "\"converged\"" in out2

    out3 = dispatcher.dispatch("generate_plot", {"plot_type": "voltage_heatmap"})
    assert "figure_json" in out3
    assert "voltage_heatmap" in out3

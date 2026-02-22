import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm_only import (
    BaselineParsed,
    build_baseline_prompt,
    evaluate_against_truth,
    export_case_tables,
    parse_llm_baseline_json,
    run_baseline_case,
)
from solver import case_loader
from solver.power_flow import run_power_flow


def test_export_tables_non_empty_case14():
    net, _ = case_loader.load("case14")
    tables = export_case_tables(net)
    assert "bus" in tables and len(tables["bus"]) > 20
    assert "line" in tables and len(tables["line"]) > 20


def test_build_prompt_contains_key_sections():
    net, _ = case_loader.load("case14")
    prompt = build_baseline_prompt("case14", net)
    assert "Bus Table" in prompt
    assert "Line Table" in prompt
    assert "line_id" in prompt


@pytest.mark.parametrize(
    "text",
    [
        "{\"converged\": true, \"bus_voltages\": [], \"line_flows\": [], \"total_generation_mw\": 0, \"total_load_mw\": 0, \"total_loss_mw\": 0}",
        "```json\n{\"converged\": true, \"bus_voltages\": [], \"line_flows\": [], \"total_generation_mw\": 0, \"total_load_mw\": 0, \"total_loss_mw\": 0}\n```",
        "{'converged': True, 'bus_voltages': [], 'line_flows': [], 'total_generation_mw': 0, 'total_load_mw': 0, 'total_loss_mw': 0}",
    ],
)
def test_parse_llm_json_variants(text: str):
    obj, err = parse_llm_baseline_json(text)
    assert err is None
    assert isinstance(obj, dict)
    assert "converged" in obj


def test_parse_llm_json_fenced_nested_object():
    text = (
        "结果如下：\n"
        "```json\n"
        "{\n"
        '  "converged": true,\n'
        '  "bus_voltages": [{"bus_id": 1, "vm_pu": 1.0, "va_deg": 0.0}],\n'
        '  "line_flows": [{"line_id": 0, "p_from_mw": 10.0, "loading_percent": 20.0}],\n'
        '  "total_generation_mw": 100.0,\n'
        '  "total_load_mw": 95.0,\n'
        '  "total_loss_mw": 5.0\n'
        "}\n"
        "```\n"
        "结束。"
    )
    obj, err = parse_llm_baseline_json(text)
    assert err is None
    assert isinstance(obj, dict)
    assert obj.get("total_loss_mw") == pytest.approx(5.0)


def test_baseline_parsed_converged_string_false():
    obj = {
        "converged": "false",
        "bus_voltages": [],
        "line_flows": [],
        "total_generation_mw": 0.0,
        "total_load_mw": 0.0,
        "total_loss_mw": 0.0,
    }
    parsed, err = BaselineParsed.from_json(obj)
    assert err is None
    assert parsed is not None
    assert parsed.converged is False


def test_run_baseline_case_rejects_invalid_runs():
    with pytest.raises(ValueError, match="n_runs"):
        run_baseline_case("case14", n_runs=0)


def test_metrics_zero_error_when_matching_truth():
    net, _ = case_loader.load("case14")
    truth = run_power_flow(net)
    assert truth.converged

    # 构造一个“完美 baseline 输出”
    obj = {
        "converged": True,
        "bus_voltages": [
            {"bus_id": int(b.bus_id), "vm_pu": float(b.vm_pu), "va_deg": float(b.va_deg)}
            for b in truth.bus_voltages
        ],
        "line_flows": [
            {"line_id": int(l.line_id), "p_from_mw": float(l.p_from_mw), "loading_percent": float(l.loading_percent)}
            for l in truth.line_flows
        ],
        "total_generation_mw": float(truth.total_generation_mw),
        "total_load_mw": float(truth.total_load_mw),
        "total_loss_mw": float(truth.total_loss_mw),
    }

    parsed, err = BaselineParsed.from_json(obj)
    assert err is None and parsed is not None

    metrics = evaluate_against_truth(parsed, truth)
    assert metrics["voltage_mae"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["flow_mae"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["voltage_violation_precision"] == 1.0
    assert metrics["voltage_violation_recall"] == 1.0
    assert metrics["thermal_violation_precision"] == 1.0
    assert metrics["thermal_violation_recall"] == 1.0

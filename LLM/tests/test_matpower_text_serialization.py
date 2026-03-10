import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from solver.llm_pf import solve_with_llm
from solver.matpower_meta import MatpowerMeta, get_matpower_meta_from_net
from solver.matpower_text import read_case_m_text, serialize_net_to_matpower_text


def _fake_net():
    return SimpleNamespace(
        name="case14",
        _case_name="case14",
        sn_mva=100.0,
        bus=pd.DataFrame(
            {
                "name": ["1", "2", "3"],
                "vn_kv": [135.0, 135.0, 135.0],
                "zone": [1, 1, 1],
                "max_vm_pu": [1.06, 1.06, 1.06],
                "min_vm_pu": [0.94, 0.94, 0.94],
                "in_service": [True, True, True],
            },
            index=[0, 1, 2],
        ),
        load=pd.DataFrame(
            {
                "bus": [1, 1],
                "p_mw": [77.7, 999.0],
                "q_mvar": [33.3, 999.0],
                "scaling": [1.0, 1.0],
                "in_service": [True, False],
            },
            index=[0, 1],
        ),
        ext_grid=pd.DataFrame(
            {
                "bus": [0],
                "vm_pu": [1.04],
                "in_service": [True],
            },
            index=[0],
        ),
        gen=pd.DataFrame(
            {
                "bus": [2],
                "p_mw": [40.0],
                "vm_pu": [1.01],
                "max_p_mw": [80.0],
                "min_p_mw": [0.0],
                "max_q_mvar": [50.0],
                "min_q_mvar": [-40.0],
                "in_service": [True],
            },
            index=[0],
        ),
        sgen=pd.DataFrame(columns=["bus", "p_mw", "in_service"]),
        line=pd.DataFrame(
            {
                "from_bus": [0],
                "to_bus": [1],
                "length_km": [1.0],
                "r_ohm_per_km": [0.1],
                "x_ohm_per_km": [0.2],
                "c_nf_per_km": [10.0],
                "max_i_ka": [0.0],
                "df": [1.0],
                "parallel": [1],
                "in_service": [False],
            },
            index=[0],
        ),
        trafo=pd.DataFrame(
            {
                "hv_bus": [1],
                "lv_bus": [2],
                "sn_mva": [50.0],
                "vk_percent": [10.0],
                "vkr_percent": [1.0],
                "tap_neutral": [0.0],
                "tap_pos": [0.0],
                "tap_step_percent": [0.0],
                "shift_degree": [0.0],
                "in_service": [True],
            },
            index=[0],
        ),
    )


def _matrix_rows(text: str, matrix_name: str) -> list[list[str]]:
    match = re.search(rf"mpc\\.{matrix_name}\\s*=\\s*\\[(.*?)\\];", text, re.DOTALL)
    assert match, f"{matrix_name} matrix not found"
    rows: list[list[str]] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(stripped.rstrip(";").split())
    return rows


def test_read_case_m_text_applies_flat_start_sanitization(tmp_path):
    root = tmp_path / "case14" / "2017-01-01"
    root.mkdir(parents=True)
    (root / "case14.m").write_text(
        "\n".join(
            [
                "% comment should be dropped",
                "function mpc = case14;",
                "mpc.version = '2';",
                "mpc.bus = [",
                "1 3 0 0 0 0 1 1.0600 -4.98 135 1 1.06 0.94;",
                "2 1 21.7 12.7 0 0 1 1.0450 -5.12 135 1 1.06 0.94;",
                "];",
            ]
        ),
        encoding="utf-8",
    )

    text = read_case_m_text("case14", date="2017-01-01", root=str(tmp_path), flat_start=True)

    assert "% comment should be dropped" not in text
    assert "1.0600 -4.98" not in text
    assert "1.0450 -5.12" not in text
    assert re.search(r"1\s+3\s+0\s+0\s+0\s+0\s+1\s+1\.0\s+0\.0", text)
    assert re.search(r"2\s+1\s+21\.7\s+12\.7\s+0\s+0\s+1\s+1\.0\s+0\.0", text)


def test_serialize_net_to_matpower_text_reflects_current_net_and_flat_start():
    text = serialize_net_to_matpower_text(_fake_net(), case_name="case14", flat_start=True)

    bus_rows = {int(float(row[0])): row for row in _matrix_rows(text, "bus")}
    branch_rows = _matrix_rows(text, "branch")

    assert float(bus_rows[2][2]) == pytest.approx(77.7)
    assert float(bus_rows[2][3]) == pytest.approx(33.3)
    assert float(bus_rows[2][7]) == pytest.approx(1.0)
    assert float(bus_rows[2][8]) == pytest.approx(0.0)
    assert float(branch_rows[0][5]) == pytest.approx(0.0)
    assert float(branch_rows[0][10]) == pytest.approx(0.0)
    assert float(branch_rows[1][5]) == pytest.approx(50.0)


def test_get_matpower_meta_from_net_uses_serialized_branch_order():
    meta = get_matpower_meta_from_net(_fake_net())

    assert meta.total_load_mw_ref == pytest.approx(77.7)
    assert meta.zero_rateA_line_ids == {1}
    assert meta.slack_bus_id_ref == 1


def test_solve_with_llm_prefers_net_serialization(monkeypatch):
    net = _fake_net()
    captured = {}
    meta = MatpowerMeta(total_load_mw_ref=77.7, zero_rateA_line_ids={1}, slack_bus_id_ref=1)

    monkeypatch.setattr("solver.llm_pf.serialize_net_to_matpower_text", lambda *args, **kwargs: "SERIALIZED_TEXT")
    monkeypatch.setattr("solver.llm_pf.get_matpower_meta_from_net", lambda *_: meta)

    def fake_solve_from_matpower_text(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr("solver.llm_pf.solve_from_matpower_text", fake_solve_from_matpower_text)

    result = solve_with_llm(
        net,
        api_key="dummy-key",
        model="gemini-2.5-flash",
        llm_provider="gemini",
        debug_mode=False,
    )

    assert result == "sentinel"
    assert captured["matpower_text"] == "SERIALIZED_TEXT"
    assert captured["m_file_path"] == "<serialized:case14>"
    assert captured["matpower_meta"] == meta

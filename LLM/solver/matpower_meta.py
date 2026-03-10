"""Metadata extraction for MATPOWER .m files (no power-flow solve)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

from solver.matpower_text import build_branch_rows_from_net, bus_display_id_from_net, total_load_from_net


@dataclass(frozen=True)
class MatpowerMeta:
    total_load_mw_ref: float
    zero_rateA_line_ids: set[int]
    slack_bus_id_ref: int


def _read_zero_ratea_and_slack(m_file_path: str) -> tuple[set[int], int]:
    try:
        from matpowercaseframes import CaseFrames
    except Exception as e:  # pragma: no cover - dependency/runtime specific
        raise RuntimeError(
            "matpowercaseframes is required for MATPOWER metadata parsing. "
            "Install it via `pip install matpowercaseframes`."
        ) from e

    cf = CaseFrames(str(m_file_path))
    branch = getattr(cf, "branch", None)
    bus = getattr(cf, "bus", None)
    if branch is None or bus is None:
        raise ValueError(f"Invalid MATPOWER case format: {m_file_path}")

    if "RATE_A" not in branch.columns:
        raise ValueError(f"MATPOWER branch matrix missing RATE_A column: {m_file_path}")
    if "BUS_TYPE" not in bus.columns or "BUS_I" not in bus.columns:
        raise ValueError(f"MATPOWER bus matrix missing BUS_TYPE/BUS_I columns: {m_file_path}")

    rate_a = branch["RATE_A"].astype(float).tolist()
    zero_rate = {i + 1 for i, v in enumerate(rate_a) if float(v) == 0.0}

    slack_rows = bus[bus["BUS_TYPE"].astype(float) == 3.0]
    if len(slack_rows) == 0:
        raise ValueError(f"No slack bus (BUS_TYPE==3) found in MATPOWER file: {m_file_path}")
    slack_bus = int(float(slack_rows.iloc[0]["BUS_I"]))

    return zero_rate, slack_bus


def _read_total_load_from_mpc(m_file_path: str) -> float:
    from pandapower.converter.matpower import from_mpc

    net = from_mpc(str(m_file_path), f_hz=50, validate_conversion=False)
    if hasattr(net, "load") and len(net.load) > 0 and "p_mw" in net.load.columns:
        return float(net.load["p_mw"].astype(float).sum())
    return 0.0


@lru_cache(maxsize=64)
def get_matpower_meta(m_file_path: str) -> MatpowerMeta:
    path = Path(m_file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MATPOWER file does not exist: {path}")

    total_load = _read_total_load_from_mpc(str(path))
    zero_rate, slack_bus = _read_zero_ratea_and_slack(str(path))
    return MatpowerMeta(
        total_load_mw_ref=float(total_load),
        zero_rateA_line_ids=set(zero_rate),
        slack_bus_id_ref=int(slack_bus),
    )


def get_matpower_meta_from_net(net: Any) -> MatpowerMeta:
    total_load = float(total_load_from_net(net))

    zero_rate: set[int] = set()
    for row in build_branch_rows_from_net(net):
        if math.isclose(float(row.rate_a), 0.0, rel_tol=0.0, abs_tol=1e-12):
            zero_rate.add(int(row.line_id))

    ext_grid_df = getattr(net, "ext_grid", None)
    slack_bus: int | None = None
    if ext_grid_df is not None and len(ext_grid_df) > 0:
        for _, row in ext_grid_df.iterrows():
            if bool(row.get("in_service", True)):
                slack_bus = int(bus_display_id_from_net(net, int(row.get("bus"))))
                break

    if slack_bus is None:
        raise ValueError("Current net has no in-service ext_grid; cannot infer slack bus for LLM-only mode.")

    return MatpowerMeta(
        total_load_mw_ref=float(total_load),
        zero_rateA_line_ids=set(zero_rate),
        slack_bus_id_ref=int(slack_bus),
    )

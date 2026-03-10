"""Utilities for locating, sanitizing, and re-serializing MATPOWER case text."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any


SUPPORTED_CASES = {"case14", "case30", "case57", "case118", "case300"}
FETCH_CMD = "python scripts/fetch_matpower_cases.py --ref master --date 2017-01-01"

_BUS_BLOCK_RE = re.compile(r"(mpc\.bus\s*=\s*\[\s*)(.*?)(\s*\];)", re.DOTALL)


@dataclass(frozen=True)
class MatpowerBranchRow:
    line_id: int
    from_bus: int
    to_bus: int
    br_r: float
    br_x: float
    br_b: float
    rate_a: float
    rate_b: float
    rate_c: float
    tap: float
    shift: float
    br_status: int
    angmin: float
    angmax: float


def _normalize_case_key(case_key: str) -> str:
    s = str(case_key or "").strip().lower()
    if s in SUPPORTED_CASES:
        return s
    raise ValueError(f"Unsupported MATPOWER case: {case_key}. Supported: {sorted(SUPPORTED_CASES)}")


def _parse_case_and_date(case_name_or_dataset_id: str, default_date: str) -> tuple[str, str]:
    raw = str(case_name_or_dataset_id or "").strip()
    if not raw:
        raise ValueError("case_name_or_dataset_id is empty")

    # dataset id format: matpower/case14/2017-01-01
    parts = [p for p in raw.split("/") if p]
    if len(parts) >= 3 and parts[0].lower() == "matpower":
        case_key = _normalize_case_key(parts[1])
        date = parts[2]
        return case_key, date

    return _normalize_case_key(raw), str(default_date)


def get_case_m_path(
    case_name_or_dataset_id: str,
    date: str = "2017-01-01",
    root: str = "data/matpower",
) -> Path:
    case_key, date_key = _parse_case_and_date(case_name_or_dataset_id, date)
    path = Path(root) / case_key / date_key / f"{case_key}.m"
    if not path.exists():
        raise FileNotFoundError(
            f"MATPOWER case file not found: {path}\n"
            f"Run:\n{FETCH_CMD}"
        )
    return path.resolve()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_zero(value: float, *, abs_tol: float = 1e-12) -> float:
    x = _safe_float(value, 0.0)
    return 0.0 if math.isclose(x, 0.0, rel_tol=0.0, abs_tol=abs_tol) else x


def _format_number(value: Any) -> str:
    x = _normalize_zero(_safe_float(value, 0.0))
    text = f"{x:.10g}"
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return text


def _strip_comment_lines(raw_text: str) -> str:
    out_lines: list[str] = []
    for line in str(raw_text or "").splitlines():
        if line.strip().startswith("%"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).rstrip() + "\n"


def sanitize_flat_start_matpower_text(raw_text: str) -> str:
    """Reset MATPOWER bus Vm/Va columns to flat-start values."""

    def _rewrite(match: re.Match[str]) -> str:
        head, body, tail = match.groups()
        rows: list[str] = []
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                continue

            has_semicolon = stripped.endswith(";")
            line = stripped[:-1].strip() if has_semicolon else stripped
            if not line:
                continue

            tokens = line.split()
            if len(tokens) >= 9:
                tokens[7] = "1.0"
                tokens[8] = "0.0"
            rows.append("    " + " ".join(tokens) + ";")
        if not rows:
            return head + tail
        return head + "\n" + "\n".join(rows) + "\n" + tail

    sanitized, _ = _BUS_BLOCK_RE.subn(_rewrite, str(raw_text or ""), count=1)
    return sanitized.rstrip() + "\n"


def _parse_bus_name_to_id(name: object) -> int | None:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    return None


def bus_display_id_from_net(net: Any, bus_idx: int) -> int:
    try:
        parsed = _parse_bus_name_to_id(net.bus.at[int(bus_idx), "name"])
        if parsed is not None:
            return int(parsed)
    except Exception:
        pass
    return int(bus_idx) + 1


def _row_in_service(row: Any) -> bool:
    if hasattr(row, "get"):
        value = row.get("in_service", True)
    else:
        try:
            value = row["in_service"]
        except Exception:
            value = True
    try:
        return bool(value)
    except Exception:
        return True


def _base_mva_from_net(net: Any) -> float:
    base_mva = _safe_float(getattr(net, "sn_mva", 100.0), 100.0)
    return 100.0 if base_mva <= 0.0 else float(base_mva)


def total_load_from_net(net: Any) -> float:
    load_df = getattr(net, "load", None)
    if load_df is None or len(load_df) == 0:
        return 0.0

    total = 0.0
    for _, row in load_df.iterrows():
        if not _row_in_service(row):
            continue
        scale = _safe_float(row.get("scaling", 1.0), 1.0)
        total += _safe_float(row.get("p_mw", 0.0), 0.0) * scale
    return float(total)


def _aggregate_bus_loads(net: Any) -> dict[int, tuple[float, float]]:
    out: dict[int, list[float]] = {}
    load_df = getattr(net, "load", None)
    if load_df is None or len(load_df) == 0:
        return {}

    for _, row in load_df.iterrows():
        if not _row_in_service(row):
            continue
        bus_idx = _safe_int(row.get("bus", -1), -1)
        if bus_idx < 0:
            continue
        scale = _safe_float(row.get("scaling", 1.0), 1.0)
        p_mw = _safe_float(row.get("p_mw", 0.0), 0.0) * scale
        q_mvar = _safe_float(row.get("q_mvar", 0.0), 0.0) * scale
        slot = out.setdefault(int(bus_idx), [0.0, 0.0])
        slot[0] += p_mw
        slot[1] += q_mvar

    return {int(k): (float(v[0]), float(v[1])) for k, v in out.items()}


def _aggregate_bus_shunts(net: Any) -> dict[int, tuple[float, float]]:
    out: dict[int, list[float]] = {}
    shunt_df = getattr(net, "shunt", None)
    if shunt_df is None or len(shunt_df) == 0:
        return {}

    for _, row in shunt_df.iterrows():
        if not _row_in_service(row):
            continue
        bus_idx = _safe_int(row.get("bus", -1), -1)
        if bus_idx < 0:
            continue
        # MATPOWER bus shunts use GS/BS at V=1.0 p.u.; this is a direct table-level
        # approximation from pandapower's shunt P/Q values.
        gs = _safe_float(row.get("p_mw", 0.0), 0.0)
        bs = -_safe_float(row.get("q_mvar", 0.0), 0.0)
        slot = out.setdefault(int(bus_idx), [0.0, 0.0])
        slot[0] += gs
        slot[1] += bs

    return {int(k): (float(v[0]), float(v[1])) for k, v in out.items()}


def _render_matrix(name: str, rows: list[list[float]]) -> str:
    if not rows:
        return f"mpc.{name} = [];"
    out = [f"mpc.{name} = ["]
    for row in rows:
        out.append("    " + " ".join(_format_number(v) for v in row) + ";")
    out.append("];")
    return "\n".join(out)


def _case_symbol(case_name: str | None) -> str:
    s = re.sub(r"\W+", "_", str(case_name or "case").strip())
    if not s:
        return "case"
    if not re.match(r"[A-Za-z_]", s):
        return f"case_{s}"
    return s


def _bus_sort_key(net: Any, bus_idx: int) -> tuple[int, int]:
    return (bus_display_id_from_net(net, int(bus_idx)), int(bus_idx))


def _build_bus_rows_from_net(net: Any, *, flat_start: bool) -> list[list[float]]:
    ext_grid_buses: set[int] = set()
    ext_grid_df = getattr(net, "ext_grid", None)
    if ext_grid_df is not None and len(ext_grid_df) > 0:
        for _, row in ext_grid_df.iterrows():
            if _row_in_service(row):
                ext_grid_buses.add(_safe_int(row.get("bus", -1), -1))

    gen_buses: set[int] = set()
    gen_df = getattr(net, "gen", None)
    if gen_df is not None and len(gen_df) > 0:
        for _, row in gen_df.iterrows():
            if _row_in_service(row):
                gen_buses.add(_safe_int(row.get("bus", -1), -1))

    bus_loads = _aggregate_bus_loads(net)
    bus_shunts = _aggregate_bus_shunts(net)
    rows: list[list[float]] = []

    bus_indices = sorted([int(idx) for idx in net.bus.index.tolist()], key=lambda idx: _bus_sort_key(net, idx))
    for bus_idx in bus_indices:
        bus = net.bus.loc[bus_idx]
        if not bool(bus.get("in_service", True)):
            bus_type = 4
        elif bus_idx in ext_grid_buses:
            bus_type = 3
        elif bus_idx in gen_buses:
            bus_type = 2
        else:
            bus_type = 1

        pd_mw, qd_mvar = bus_loads.get(bus_idx, (0.0, 0.0))
        gs, bs = bus_shunts.get(bus_idx, (0.0, 0.0))
        vm_pu = 1.0 if flat_start else _safe_float(bus.get("vm_pu", 1.0), 1.0)
        va_deg = 0.0 if flat_start else _safe_float(bus.get("va_degree", 0.0), 0.0)
        rows.append(
            [
                float(bus_display_id_from_net(net, bus_idx)),
                float(bus_type),
                float(pd_mw),
                float(qd_mvar),
                float(gs),
                float(bs),
                float(_safe_int(bus.get("zone", 1), 1)),
                float(vm_pu),
                float(va_deg),
                float(_safe_float(bus.get("vn_kv", 0.0), 0.0)),
                float(_safe_int(bus.get("zone", 1), 1)),
                float(_safe_float(bus.get("max_vm_pu", 1.1), 1.1)),
                float(_safe_float(bus.get("min_vm_pu", 0.9), 0.9)),
            ]
        )
    return rows


def _build_gen_rows_from_net(net: Any, *, base_mva: float) -> list[list[float]]:
    rows: list[list[float]] = []

    ext_grid_df = getattr(net, "ext_grid", None)
    if ext_grid_df is not None and len(ext_grid_df) > 0:
        for _, row in ext_grid_df.iterrows():
            bus_idx = _safe_int(row.get("bus", -1), -1)
            if bus_idx < 0:
                continue
            rows.append(
                [
                    float(bus_display_id_from_net(net, bus_idx)),
                    0.0,
                    0.0,
                    float(_safe_float(row.get("max_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("min_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("vm_pu", 1.0), 1.0)),
                    float(base_mva),
                    1.0 if _row_in_service(row) else 0.0,
                    float(_safe_float(row.get("max_p_mw", 0.0), 0.0)),
                    float(_safe_float(row.get("min_p_mw", 0.0), 0.0)),
                ]
            )

    gen_df = getattr(net, "gen", None)
    if gen_df is not None and len(gen_df) > 0:
        for _, row in gen_df.iterrows():
            bus_idx = _safe_int(row.get("bus", -1), -1)
            if bus_idx < 0:
                continue
            rows.append(
                [
                    float(bus_display_id_from_net(net, bus_idx)),
                    float(_safe_float(row.get("p_mw", 0.0), 0.0)),
                    0.0,
                    float(_safe_float(row.get("max_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("min_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("vm_pu", 1.0), 1.0)),
                    float(base_mva),
                    1.0 if _row_in_service(row) else 0.0,
                    float(_safe_float(row.get("max_p_mw", row.get("p_mw", 0.0)), 0.0)),
                    float(_safe_float(row.get("min_p_mw", 0.0), 0.0)),
                ]
            )

    sgen_df = getattr(net, "sgen", None)
    if sgen_df is not None and len(sgen_df) > 0:
        for _, row in sgen_df.iterrows():
            bus_idx = _safe_int(row.get("bus", -1), -1)
            if bus_idx < 0:
                continue
            rows.append(
                [
                    float(bus_display_id_from_net(net, bus_idx)),
                    float(_safe_float(row.get("p_mw", 0.0), 0.0)),
                    0.0,
                    float(_safe_float(row.get("max_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("min_q_mvar", 0.0), 0.0)),
                    float(_safe_float(row.get("vm_pu", 1.0), 1.0)),
                    float(base_mva),
                    1.0 if _row_in_service(row) else 0.0,
                    float(_safe_float(row.get("max_p_mw", row.get("p_mw", 0.0)), 0.0)),
                    float(_safe_float(row.get("min_p_mw", 0.0), 0.0)),
                ]
            )

    return rows


def _line_branch_rows(net: Any, *, base_mva: float, f_hz: float) -> list[MatpowerBranchRow]:
    line_df = getattr(net, "line", None)
    if line_df is None or len(line_df) == 0:
        return []

    rows: list[MatpowerBranchRow] = []
    for idx, row in line_df.iterrows():
        fb_idx = _safe_int(row.get("from_bus", -1), -1)
        tb_idx = _safe_int(row.get("to_bus", -1), -1)
        if fb_idx < 0 or tb_idx < 0:
            continue

        parallel = max(_safe_float(row.get("parallel", 1.0), 1.0), 1.0)
        length_km = _safe_float(row.get("length_km", 0.0), 0.0)
        from_kv = _safe_float(net.bus.at[fb_idx, "vn_kv"], 0.0)
        to_kv = _safe_float(net.bus.at[tb_idx, "vn_kv"], 0.0)
        base_kv = max(from_kv, to_kv, 1e-9)
        base_z = (base_kv * base_kv) / max(base_mva, 1e-9)

        r_ohm = _safe_float(row.get("r_ohm_per_km", 0.0), 0.0) * length_km / parallel
        x_ohm = _safe_float(row.get("x_ohm_per_km", 0.0), 0.0) * length_km / parallel
        c_total_f = _safe_float(row.get("c_nf_per_km", 0.0), 0.0) * length_km * parallel * 1e-9
        br_b = 2.0 * math.pi * max(f_hz, 0.0) * c_total_f * base_z

        df = max(_safe_float(row.get("df", 1.0), 1.0), 0.0)
        max_i_ka = max(_safe_float(row.get("max_i_ka", 0.0), 0.0), 0.0)
        rate_a = math.sqrt(3.0) * base_kv * max_i_ka * parallel * df if max_i_ka > 0.0 else 0.0
        rows.append(
            MatpowerBranchRow(
                line_id=int(idx),
                from_bus=bus_display_id_from_net(net, fb_idx),
                to_bus=bus_display_id_from_net(net, tb_idx),
                br_r=r_ohm / base_z,
                br_x=x_ohm / base_z,
                br_b=br_b,
                rate_a=rate_a,
                rate_b=rate_a,
                rate_c=rate_a,
                tap=0.0,
                shift=0.0,
                br_status=1 if _row_in_service(row) else 0,
                angmin=-360.0,
                angmax=360.0,
            )
        )

    return rows


def _trafo_branch_rows(net: Any, *, base_mva: float) -> list[MatpowerBranchRow]:
    trafo_df = getattr(net, "trafo", None)
    if trafo_df is None or len(trafo_df) == 0:
        return []

    rows: list[MatpowerBranchRow] = []
    for idx, row in trafo_df.iterrows():
        fb_idx = _safe_int(row.get("hv_bus", -1), -1)
        tb_idx = _safe_int(row.get("lv_bus", -1), -1)
        if fb_idx < 0 or tb_idx < 0:
            continue

        sn_mva = max(_safe_float(row.get("sn_mva", 0.0), 0.0), 0.0)
        z_pu = _safe_float(row.get("vk_percent", 0.0), 0.0) / 100.0
        r_pu = _safe_float(row.get("vkr_percent", 0.0), 0.0) / 100.0
        x_sq = max(z_pu * z_pu - r_pu * r_pu, 0.0)
        base_scale = (base_mva / sn_mva) if sn_mva > 0.0 else 0.0
        tap_neutral = _safe_float(row.get("tap_neutral", 0.0), 0.0)
        tap_pos = _safe_float(row.get("tap_pos", tap_neutral), tap_neutral)
        tap_step_pct = _safe_float(row.get("tap_step_percent", 0.0), 0.0)
        tap_ratio = 1.0 + ((tap_pos - tap_neutral) * tap_step_pct / 100.0)
        rows.append(
            MatpowerBranchRow(
                line_id=100000 + int(idx),
                from_bus=bus_display_id_from_net(net, fb_idx),
                to_bus=bus_display_id_from_net(net, tb_idx),
                br_r=r_pu * base_scale,
                br_x=math.sqrt(x_sq) * base_scale,
                br_b=0.0,
                rate_a=sn_mva,
                rate_b=sn_mva,
                rate_c=sn_mva,
                tap=tap_ratio,
                shift=_safe_float(row.get("shift_degree", 0.0), 0.0),
                br_status=1 if _row_in_service(row) else 0,
                angmin=-360.0,
                angmax=360.0,
            )
        )

    return rows


def build_branch_rows_from_net(
    net: Any,
    *,
    base_mva: float | None = None,
    f_hz: float = 50.0,
) -> list[MatpowerBranchRow]:
    base = _base_mva_from_net(net) if base_mva is None else float(base_mva)
    rows: list[MatpowerBranchRow] = []
    next_row_id = 1
    for branch in _line_branch_rows(net, base_mva=base, f_hz=f_hz):
        rows.append(
            MatpowerBranchRow(
                line_id=next_row_id,
                from_bus=int(branch.from_bus),
                to_bus=int(branch.to_bus),
                br_r=float(branch.br_r),
                br_x=float(branch.br_x),
                br_b=float(branch.br_b),
                rate_a=float(branch.rate_a),
                rate_b=float(branch.rate_b),
                rate_c=float(branch.rate_c),
                tap=float(branch.tap),
                shift=float(branch.shift),
                br_status=int(branch.br_status),
                angmin=float(branch.angmin),
                angmax=float(branch.angmax),
            )
        )
        next_row_id += 1
    for branch in _trafo_branch_rows(net, base_mva=base):
        rows.append(
            MatpowerBranchRow(
                line_id=next_row_id,
                from_bus=int(branch.from_bus),
                to_bus=int(branch.to_bus),
                br_r=float(branch.br_r),
                br_x=float(branch.br_x),
                br_b=float(branch.br_b),
                rate_a=float(branch.rate_a),
                rate_b=float(branch.rate_b),
                rate_c=float(branch.rate_c),
                tap=float(branch.tap),
                shift=float(branch.shift),
                br_status=int(branch.br_status),
                angmin=float(branch.angmin),
                angmax=float(branch.angmax),
            )
        )
        next_row_id += 1
    return rows


def serialize_net_to_matpower_text(
    net: Any,
    *,
    case_name: str | None = None,
    flat_start: bool = True,
    f_hz: float = 50.0,
) -> str:
    """Serialize the current pandapower net into a MATPOWER-like text payload."""

    symbol = _case_symbol(case_name or getattr(net, "_case_name", None) or getattr(net, "name", None) or "case")
    base_mva = _base_mva_from_net(net)
    bus_rows = _build_bus_rows_from_net(net, flat_start=bool(flat_start))
    gen_rows = _build_gen_rows_from_net(net, base_mva=base_mva)
    branch_rows = build_branch_rows_from_net(net, base_mva=base_mva, f_hz=f_hz)

    branch_matrix = [
        [
            float(row.from_bus),
            float(row.to_bus),
            float(row.br_r),
            float(row.br_x),
            float(row.br_b),
            float(row.rate_a),
            float(row.rate_b),
            float(row.rate_c),
            float(row.tap),
            float(row.shift),
            float(row.br_status),
            float(row.angmin),
            float(row.angmax),
        ]
        for row in branch_rows
    ]

    parts = [
        f"function mpc = {symbol};",
        "mpc.version = '2';",
        f"mpc.baseMVA = {_format_number(base_mva)};",
        _render_matrix("bus", bus_rows),
        _render_matrix("gen", gen_rows),
        _render_matrix("branch", branch_matrix),
    ]
    return "\n\n".join(parts).rstrip() + "\n"


def read_case_m_text(
    case_name_or_dataset_id: str,
    date: str = "2017-01-01",
    root: str = "data/matpower",
    flat_start: bool = True,
) -> str:
    path = get_case_m_path(case_name_or_dataset_id, date=date, root=root)
    text = _strip_comment_lines(path.read_text(encoding="utf-8"))
    if flat_start:
        text = sanitize_flat_start_matpower_text(text)
    return text

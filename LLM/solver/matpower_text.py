"""Utilities for locating and reading MATPOWER case text files."""

from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_CASES = {"case14", "case30", "case57", "case118", "case300"}
FETCH_CMD = "python scripts/fetch_matpower_cases.py --ref master --date 2017-01-01"
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)


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


def read_case_m_text(
    case_name_or_dataset_id: str,
    date: str = "2017-01-01",
    root: str = "data/matpower",
) -> str:
    path = get_case_m_path(case_name_or_dataset_id, date=date, root=root)
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("%"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _replace_number_by_index(text: str, replacements: dict[int, str]) -> str:
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) < max(replacements.keys(), default=-1) + 1:
        return text

    out: list[str] = []
    pos = 0
    for idx, match in enumerate(matches):
        out.append(text[pos : match.start()])
        out.append(replacements.get(idx, match.group(0)))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _scrub_bus_row_flat_start(line: str) -> str:
    body = line.rstrip("\r\n")
    newline = line[len(body) :]

    comment_start = body.find("%")
    if comment_start >= 0:
        code = body[:comment_start]
        comment = body[comment_start:]
    else:
        code = body
        comment = ""

    # MATPOWER bus columns are 1-based: 8 = Vm, 9 = Va.
    scrubbed = _replace_number_by_index(code, {7: "1.0", 8: "0"})
    return f"{scrubbed}{comment}{newline}"


def scrub_matpower_bus_initial_conditions(matpower_text: str) -> str:
    """Apply flat-start protocol to MATPOWER bus Vm/Va initialization fields.

    Only rows inside ``mpc.bus = [ ... ];`` are changed. Column 8 (Vm) becomes
    1.0 p.u. and column 9 (Va) becomes 0 degrees for every bus row.
    """

    lines = str(matpower_text or "").splitlines(keepends=True)
    out: list[str] = []
    in_bus_matrix = False

    for line in lines:
        stripped = line.strip()
        if not in_bus_matrix:
            out.append(line)
            if re.match(r"^mpc\.bus\s*=\s*\[", stripped):
                in_bus_matrix = True
            continue

        if stripped.startswith("];"):
            in_bus_matrix = False
            out.append(line)
            continue

        out.append(_scrub_bus_row_flat_start(line))

    return "".join(out)

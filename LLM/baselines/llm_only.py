"""baselines/llm_only.py

LLM-Only 基线（不调用工具/求解器）。

做法（PRD Step 5）：
1) 将 pandapower 网络表（bus/load/gen/ext_grid/line/trafo）序列化为文本，塞进 prompt。
2) 让 LLM 直接输出“潮流结果 JSON”。
3) 解析 LLM 输出，提取其声称的数值。
4) 与 pandapower 求解器真值对比，得到 MAE 与越限识别精度。

注意：
- 该基线几乎必然产生幻觉与大误差，其作用是提供“下界”。
- 为了可重复评估，建议固定 temperature 较低（如 0.0~0.2）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from llm.engine import LLMClient, OpenAIChatClient
from solver import case_loader
from solver.power_flow import SolverConfig, run_power_flow
from models.schemas import PowerFlowResult

from baselines.prompts_baseline import (
    BASELINE_PROMPT_TEMPLATE,
    BASELINE_SYSTEM_PROMPT,
)


# ---------------------------
# Prompt data export
# ---------------------------


def _df_to_text(df: Any, *, columns: Optional[List[str]] = None, max_rows: int = 200) -> str:
    """将 DataFrame 转为紧凑文本。

    不使用 DataFrame.to_markdown()（避免 tabulate 依赖）。
    """

    if df is None:
        return "<empty>"

    try:
        if columns:
            cols = [c for c in columns if c in df.columns]
            df2 = df[cols]
        else:
            df2 = df

        if len(df2) > max_rows:
            df2 = df2.head(max_rows)
            suffix = f"\n... (truncated, showing first {max_rows} rows)"
        else:
            suffix = ""

        # 用 to_string 保留索引（line_id/trafo_id 需要 index）
        txt = df2.to_string(index=True)
        return txt + suffix
    except Exception as e:
        return f"<failed to serialize table: {type(e).__name__}: {e}>"


def export_case_tables(net: Any) -> Dict[str, str]:
    """导出用例关键表为文本，用于 baseline prompt。"""

    bus_cols = ["name", "vn_kv", "type", "zone", "in_service"]
    load_cols = ["bus", "p_mw", "q_mvar", "in_service"]
    gen_cols = [
        "bus",
        "p_mw",
        "vm_pu",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    ]
    ext_cols = ["bus", "vm_pu", "va_degree", "in_service"]
    line_cols = [
        "from_bus",
        "to_bus",
        "length_km",
        "r_ohm_per_km",
        "x_ohm_per_km",
        "c_nf_per_km",
        "max_i_ka",
        "df",
        "parallel",
        "in_service",
    ]
    trafo_cols = [
        "hv_bus",
        "lv_bus",
        "sn_mva",
        "vn_hv_kv",
        "vn_lv_kv",
        "vk_percent",
        "vkr_percent",
        "pfe_kw",
        "i0_percent",
        "shift_degree",
        "tap_side",
        "tap_neutral",
        "tap_min",
        "tap_max",
        "tap_step_percent",
        "in_service",
    ]

    return {
        "bus": _df_to_text(net.bus, columns=bus_cols),
        "load": _df_to_text(getattr(net, "load", None), columns=load_cols),
        "gen": _df_to_text(getattr(net, "gen", None), columns=gen_cols),
        "ext_grid": _df_to_text(getattr(net, "ext_grid", None), columns=ext_cols),
        "line": _df_to_text(getattr(net, "line", None), columns=line_cols),
        "trafo": _df_to_text(getattr(net, "trafo", None), columns=trafo_cols),
    }


def build_baseline_prompt(case_name: str, net: Any) -> str:
    """构造 baseline 的 user prompt。"""

    tables = export_case_tables(net)
    # 额外说明 bus_id 语义，避免“0-based/1-based”混乱
    header = (
        "\n\n[重要说明] bus_id 必须使用 net.bus 表中的 name 列（即 IEEE/MATPOWER 常见的 1..N 编号），"
        "不要使用 net.bus 的行索引。\n"
    )
    return (
        BASELINE_PROMPT_TEMPLATE.format(
            case_name=case_name,
            bus_data_table=tables["bus"],
            load_data_table=tables["load"],
            gen_data_table=tables["gen"],
            ext_grid_data_table=tables["ext_grid"],
            line_data_table=tables["line"],
            trafo_data_table=tables["trafo"],
        )
        + header
    )


# ---------------------------
# Robust JSON parsing
# ---------------------------


_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_balanced_json_object(text: str) -> Optional[str]:
    """提取首个花括号平衡的对象，支持嵌套结构。"""

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    quote = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
            if depth < 0:
                break

    return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """尽量从混杂文本中提取第一个 JSON 对象字符串。"""

    if not text:
        return None

    for m in _FENCE_BLOCK_RE.finditer(text):
        obj = _extract_balanced_json_object(m.group(1))
        if obj:
            return obj

    # 退化：找第一个 '{' 到最后一个 '}'，并尝试平衡
    return _extract_balanced_json_object(text)


def _repair_json_like(s: str) -> str:
    """对常见的“准 JSON”做轻量修复。"""

    s = s.strip()

    # 把单引号替换为双引号（非常粗糙，但 baseline 常见）
    # 注意：这可能破坏包含缩写的文本字段；但 baseline 输出应基本为数值。
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    # 去掉尾逗号
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 规范布尔与 null
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    return s


def parse_llm_baseline_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解析 baseline LLM 输出。

    返回 (obj, error)。成功时 error=None。
    """

    raw = _extract_first_json_object(text)
    if not raw:
        return None, "no_json_object_found"

    raw2 = _repair_json_like(raw)

    try:
        obj = json.loads(raw2)
        if not isinstance(obj, dict):
            return None, "json_not_object"
        return obj, None
    except Exception:
        return None, "json_parse_failed"


def _parse_bool_like(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v) == 1.0:
            return True
        if float(v) == 0.0:
            return False
        return None

    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False

    return None


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


@dataclass
class BaselineParsed:
    converged: bool
    bus_vm: Dict[int, float]
    bus_va: Dict[int, float]
    line_p: Dict[int, float]
    line_loading: Dict[int, float]
    total_generation_mw: float
    total_load_mw: float
    total_loss_mw: float

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> Tuple[Optional["BaselineParsed"], Optional[str]]:
        if not isinstance(obj, dict):
            return None, "json_not_object"

        required = [
            "converged",
            "bus_voltages",
            "line_flows",
            "total_generation_mw",
            "total_load_mw",
            "total_loss_mw",
        ]
        for k in required:
            if k not in obj:
                return None, f"missing_field:{k}"

        converged = _parse_bool_like(obj.get("converged"))
        if converged is None:
            return None, "converged_parse_failed"

        bus_items = obj.get("bus_voltages")
        if not isinstance(bus_items, list):
            return None, "bus_voltages_not_list"

        bus_vm: Dict[int, float] = {}
        bus_va: Dict[int, float] = {}
        for item in bus_items:
            try:
                bid = int(item.get("bus_id"))
                vm = float(item.get("vm_pu"))
                va = float(item.get("va_deg"))
                if _finite(vm) and _finite(va):
                    bus_vm[bid] = vm
                    bus_va[bid] = va
            except Exception:
                continue

        line_items = obj.get("line_flows")
        if not isinstance(line_items, list):
            return None, "line_flows_not_list"

        line_p: Dict[int, float] = {}
        line_loading: Dict[int, float] = {}
        for item in line_items:
            try:
                lid = int(item.get("line_id"))
                p = float(item.get("p_from_mw"))
                ld = float(item.get("loading_percent"))
                if _finite(p) and _finite(ld):
                    line_p[lid] = p
                    line_loading[lid] = ld
            except Exception:
                continue

        totals = (obj.get("total_generation_mw"), obj.get("total_load_mw"), obj.get("total_loss_mw"))
        try:
            tg, tl, tlo = (float(totals[0]), float(totals[1]), float(totals[2]))
        except Exception:
            return None, "totals_parse_failed"

        if not (_finite(tg) and _finite(tl) and _finite(tlo)):
            return None, "totals_not_finite"

        return (
            BaselineParsed(
                converged=converged,
                bus_vm=bus_vm,
                bus_va=bus_va,
                line_p=line_p,
                line_loading=line_loading,
                total_generation_mw=tg,
                total_load_mw=tl,
                total_loss_mw=tlo,
            ),
            None,
        )


# ---------------------------
# Metrics
# ---------------------------


def _mae(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    vals = [abs(a - b) for a, b in pairs]
    if not vals:
        return None
    return float(np.mean(vals))


def _set_precision_recall(pred: set[int], truth: set[int]) -> Tuple[Optional[float], Optional[float]]:
    if not pred and not truth:
        return 1.0, 1.0
    if not pred and truth:
        return 1.0, 0.0
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else None
    rec = tp / len(truth) if truth else None
    return (float(prec) if prec is not None else None, float(rec) if rec is not None else None)


def evaluate_against_truth(
    parsed: BaselineParsed,
    truth: PowerFlowResult,
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    max_loading: float = 100.0,
) -> Dict[str, Any]:
    """计算 baseline 输出与求解器真值的误差与越限识别指标。"""

    truth_vm = {int(b.bus_id): float(b.vm_pu) for b in truth.bus_voltages}
    truth_p = {int(l.line_id): float(l.p_from_mw) for l in truth.line_flows}

    vm_pairs = [(parsed.bus_vm[i], truth_vm[i]) for i in parsed.bus_vm.keys() & truth_vm.keys()]
    p_pairs = [(parsed.line_p[i], truth_p[i]) for i in parsed.line_p.keys() & truth_p.keys()]

    voltage_mae = _mae(vm_pairs)
    flow_mae = _mae(p_pairs)

    # violations by threshold
    pred_v = {i for i, v in parsed.bus_vm.items() if v < v_min or v > v_max}
    truth_v = {int(b.bus_id) for b in truth.voltage_violations}

    pred_t = {i for i, ld in parsed.line_loading.items() if ld > max_loading}
    truth_t = {int(l.line_id) for l in truth.thermal_violations}

    v_prec, v_rec = _set_precision_recall(pred_v, truth_v)
    t_prec, t_rec = _set_precision_recall(pred_t, truth_t)

    return {
        "voltage_mae": voltage_mae,
        "flow_mae": flow_mae,
        "bus_coverage": {
            "pred": len(parsed.bus_vm),
            "truth": len(truth_vm),
            "intersection": len(parsed.bus_vm.keys() & truth_vm.keys()),
        },
        "line_coverage": {
            "pred": len(parsed.line_p),
            "truth": len(truth_p),
            "intersection": len(parsed.line_p.keys() & truth_p.keys()),
        },
        "voltage_violation_precision": v_prec,
        "voltage_violation_recall": v_rec,
        "thermal_violation_precision": t_prec,
        "thermal_violation_recall": t_rec,
        "pred_voltage_violations": sorted(pred_v),
        "truth_voltage_violations": sorted(truth_v),
        "pred_thermal_violations": sorted(pred_t),
        "truth_thermal_violations": sorted(truth_t),
    }


# ---------------------------
# Runner
# ---------------------------


@dataclass
class BaselineRunConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_s: float = 90.0


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                txt = part.get("text")
                if isinstance(txt, dict):
                    txt = txt.get("value")
                if txt is None:
                    txt = part.get("content")
                if txt is not None:
                    chunks.append(str(txt))
                continue
            txt = getattr(part, "text", None)
            if txt is not None:
                chunks.append(str(txt))
        return "\n".join(c for c in chunks if c)

    return str(content or "")


def _extract_response_text(resp: Any) -> str:
    if isinstance(resp, dict):
        try:
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _content_to_text(content)
        except Exception:
            return str(resp)

    try:
        content = resp.choices[0].message.content
    except Exception:
        return str(resp)
    return _content_to_text(content)


def run_llm_only_once(
    client: LLMClient,
    *,
    model: str,
    temperature: float,
    timeout_s: float,
    prompt: str,
) -> Tuple[str, float]:
    """调用一次 LLM（无 tools），返回 (raw_text, llm_time_s)。"""

    t0 = time.time()
    resp = client.create(
        model=model,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=timeout_s,
    )
    dt = time.time() - t0

    text = _extract_response_text(resp)
    return text, float(dt)


def run_baseline_case(
    case_name: str,
    *,
    n_runs: int = 5,
    client: Optional[LLMClient] = None,
    llm_config: Optional[BaselineRunConfig] = None,
    solver_config: Optional[SolverConfig] = None,
) -> Dict[str, Any]:
    """对一个 case 执行 n_runs 次 baseline，并汇总指标。"""

    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("n_runs must be a positive integer")

    if llm_config is None:
        llm_config = BaselineRunConfig()
    if solver_config is None:
        solver_config = SolverConfig()

    net, _ = case_loader.load(case_name)
    truth = run_power_flow(net, config=solver_config)
    if not truth.converged:
        raise RuntimeError(f"真值求解未收敛：{case_name}")

    prompt = build_baseline_prompt(case_name, net)

    if client is None:
        client = OpenAIChatClient(api_key=os.getenv("OPENAI_API_KEY"))

    per_run: List[Dict[str, Any]] = []
    fail_modes: Dict[str, int] = {}

    for k in range(n_runs):
        raw, llm_time_s = run_llm_only_once(
            client,
            model=llm_config.model,
            temperature=llm_config.temperature,
            timeout_s=llm_config.timeout_s,
            prompt=prompt,
        )

        obj, err = parse_llm_baseline_json(raw)
        if err:
            fail_modes[err] = fail_modes.get(err, 0) + 1
            per_run.append({"run": k, "ok": False, "error": err, "llm_time_s": llm_time_s})
            continue

        parsed, err2 = BaselineParsed.from_json(obj or {})
        if err2:
            fail_modes[err2] = fail_modes.get(err2, 0) + 1
            per_run.append({"run": k, "ok": False, "error": err2, "llm_time_s": llm_time_s})
            continue

        metrics = evaluate_against_truth(
            parsed,
            truth,
            v_min=solver_config.v_min,
            v_max=solver_config.v_max,
            max_loading=solver_config.max_loading,
        )

        per_run.append(
            {
                "run": k,
                "ok": True,
                "llm_time_s": llm_time_s,
                "converged": bool(parsed.converged),
                "metrics": metrics,
            }
        )

    # aggregate
    ok_runs = [r for r in per_run if r.get("ok")]
    voltage_mae_vals = [r["metrics"]["voltage_mae"] for r in ok_runs if r["metrics"]["voltage_mae"] is not None]
    flow_mae_vals = [r["metrics"]["flow_mae"] for r in ok_runs if r["metrics"]["flow_mae"] is not None]

    agg = {
        "success_rate": float(len(ok_runs) / n_runs) if n_runs else 0.0,
        "voltage_mae_mean": float(np.mean(voltage_mae_vals)) if voltage_mae_vals else None,
        "voltage_mae_std": float(np.std(voltage_mae_vals)) if voltage_mae_vals else None,
        "flow_mae_mean": float(np.mean(flow_mae_vals)) if flow_mae_vals else None,
        "flow_mae_std": float(np.std(flow_mae_vals)) if flow_mae_vals else None,
        "fail_modes": fail_modes,
        "avg_llm_time_s": float(np.mean([r["llm_time_s"] for r in per_run])) if per_run else None,
    }

    return {
        "case_name": case_name,
        "n_runs": n_runs,
        "llm": {
            "model": llm_config.model,
            "temperature": llm_config.temperature,
        },
        "thresholds": {
            "v_min": solver_config.v_min,
            "v_max": solver_config.v_max,
            "max_loading": solver_config.max_loading,
        },
        "aggregate": agg,
        "runs": per_run,
    }


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown_summary(path: Path, report: Dict[str, Any]) -> None:
    agg = report.get("aggregate", {})
    lines = []
    lines.append(f"# LLM-Only Baseline Report: {report.get('case_name')}\n")
    lines.append(f"- Runs: {report.get('n_runs')}\n")
    lines.append(f"- Model: {report.get('llm', {}).get('model')}\n")
    lines.append(f"- Temperature: {report.get('llm', {}).get('temperature')}\n")
    lines.append("\n## Aggregate\n")
    lines.append(f"- success_rate: {agg.get('success_rate')}\n")
    lines.append(f"- voltage_mae_mean: {agg.get('voltage_mae_mean')}\n")
    lines.append(f"- flow_mae_mean: {agg.get('flow_mae_mean')}\n")
    lines.append(f"- avg_llm_time_s: {agg.get('avg_llm_time_s')}\n")
    lines.append("\n## Failure Modes\n")
    fm = agg.get("fail_modes", {}) or {}
    if not fm:
        lines.append("- (none)\n")
    else:
        for k, v in sorted(fm.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {k}: {v}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM-only baseline for IEEE cases.")
    parser.add_argument("--case", dest="case_name", default="case14", help="case14/case30/case57")
    parser.add_argument("--runs", dest="runs", type=int, default=5)
    parser.add_argument("--model", dest="model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.2)
    parser.add_argument("--out", dest="out_dir", default="baselines/results")

    args = parser.parse_args(argv)

    report = run_baseline_case(
        args.case_name,
        n_runs=args.runs,
        llm_config=BaselineRunConfig(model=args.model, temperature=args.temperature),
    )

    out_dir = Path(args.out_dir)
    _write_json(out_dir / f"baseline_{args.case_name}.json", report)
    _write_markdown_summary(out_dir / f"baseline_{args.case_name}.md", report)
    print(f"Saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

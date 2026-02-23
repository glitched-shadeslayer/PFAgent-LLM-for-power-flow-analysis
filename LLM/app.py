"""Main Streamlit app entrypoint."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from app_utils import extract_tool_artifacts, pick_last_n1_report, pick_last_plot, safe_json_loads
from config import (
    DEFAULT_MAX_LOADING,
    DEFAULT_V_MAX,
    DEFAULT_V_MIN,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TIMEOUT_S,
    LLM_ONLY_DEBUG_MODE,
    MATPOWER_CASE_DATE,
    MATPOWER_DATA_ROOT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_TIMEOUT_S,
)
from llm.engine import EngineConfig, LLMEngine, OpenAIChatClient
from llm.tools import ToolContext, build_default_dispatcher
from models.schemas import Modification, PowerFlowResult, RemedialAction, SessionState
from solver.contingency import run_n1_contingency
from solver.power_flow import SolverConfig
from solver.power_flow import run_power_flow
from solver.remedial import apply_remedial_action_inplace, recommend_remedial_actions
from solver.validators import validate_result
from viz import make_flow_diagram, make_voltage_heatmap, make_violation_overview
from viz import make_comparison, make_n1_ranking, make_remedial_ranking
from viz.comparison import compute_comparison_summary, make_quantitative_comparison
from viz.benchmark import make_benchmark_figure, compute_benchmark_cards
from baselines.llm_only import baseline_parsed_from_result, evaluate_against_truth_extended
from viz.flow_diagram import resolve_flow_positions
from viz.flow_particles import build_particle_segments, make_flow_particles_html
from viz.network_plot import build_graph, compute_layout


# -----------------------------
# UI strings
# -----------------------------


ZH = {
    "title": "电力系统潮流分析平台",
    "sidebar_case": "选择测试系统",
    "sidebar_load": "加载系统",
    "sidebar_settings": "分析设置",
    "sidebar_vmin": "电压下限 (p.u.)",
    "sidebar_vmax": "电压上限 (p.u.)",
    "sidebar_loading": "线路负载率上限 (%)",
    "sidebar_backend": "求解后端",
    "backend_pp": "PandaPower（真值）",
    "backend_llm": "LLM-only（估算/研究用）",
    "backend_active": "当前后端",
    "sidebar_history": "修改历史",
    "sidebar_undo": "撤销上一步",
    "sidebar_clear": "清空对话",
    "no_api": "未配置 API Key。你仍可在侧边栏直接运行求解和作图。",
    "welcome_h": "### 欢迎使用",
    "welcome_t": "试试下面这些快捷指令：",
    "btn_run14": "运行 IEEE 14 潮流",
    "btn_v30": "IEEE 30 电压图",
    "btn_57": "IEEE 57 What-if",
    "chat_placeholder": "输入分析指令...",
    "btn_run": "运行潮流",
    "btn_v": "电压图",
    "btn_violation": "越限概览",
    "btn_disconnect": "断开线路",
    "btn_report": "导出报告",
    "btn_n1": "N-1",
    "btn_remedial": "缓解建议",
    "sidebar_n1": "N-1 故障分析",
    "sidebar_n1_topk": "Top-K",
    "sidebar_n1_go": "运行 N-1",
    "sidebar_remedial": "缓解建议",
    "disconnect_fb": "起始节点 from_bus",
    "disconnect_tb": "终止节点 to_bus",
    "disconnect_go": "确认断开并分析",
    "status": "正在分析...",
    "status_done": "完成",
    "status_parse": "解析意图并规划工具调用...",
    "status_tools": "执行求解/校验/可视化工具...",
    "status_answer": "生成解读...",
    "case_loaded": "已加载系统：{case}",
    "need_case": "请先加载一个测试系统。",
    "need_result": "请先运行潮流计算后再作图。",
    "report_todo": "报告导出功能后续实现。",
    "undo_done": "已撤销上一步修改。",
    "remedial_apply": "应用",
    "remedial_confirm_title": "确认应用缓解建议",
    "remedial_confirm_warn": "该操作会修改当前网络并重算潮流，是否继续？",
    "remedial_cancel": "取消",
    "remedial_confirm": "确认",
    "remedial_applied": "已应用缓解建议 #{idx}: {desc}",
    "remedial_apply_failed": "应用缓解建议失败：{err}",
    "remedial_stale": "提示：应用/撤销后建议可能已过期，请重新生成。",
    "llm_raw_expander": "🔍 查看 LLM 原始输入/输出",
    "llm_prompt": "LLM 输入 Prompt",
    "llm_response": "LLM 原始输出",
    "external_import_title": "外挂 LLM 结果导入",
    "external_import_input": "粘贴外挂 LLM 输出 JSON",
    "external_import_btn": "导入并作图",
    "external_import_ok": "已导入外挂 LLM 结果并完成作图。",
    "external_import_err": "导入外挂 LLM 结果失败：{err}",
    "benchmark_btn": "Benchmark vs 真值",
    "benchmark_ok": "已完成 LLM vs 真值 Benchmark 对比。",
    "benchmark_err": "Benchmark 对比失败：{err}",
    "benchmark_need_llm": "请先导入外挂 LLM 结果。",
    "benchmark_title": "LLM Benchmark 对比报告",
    "remedial_none": "暂无可应用的缓解建议：请先生成建议。",
}

EN = {
    "title": "Power Flow Analysis Platform",
    "sidebar_case": "Select test case",
    "sidebar_load": "Load case",
    "sidebar_settings": "Settings",
    "sidebar_vmin": "Voltage min (p.u.)",
    "sidebar_vmax": "Voltage max (p.u.)",
    "sidebar_loading": "Max loading (%)",
    "sidebar_backend": "Solver Backend",
    "backend_pp": "PandaPower (ground truth)",
    "backend_llm": "LLM-only (estimation/research)",
    "backend_active": "Active backend",
    "sidebar_history": "Modification log",
    "sidebar_undo": "Undo",
    "sidebar_clear": "Clear chat",
    "no_api": "API key is not set. You can still solve and plot from sidebar.",
    "welcome_h": "### Welcome",
    "welcome_t": "Try these examples:",
    "btn_run14": "Run IEEE 14 PF",
    "btn_v30": "IEEE 30 voltage map",
    "btn_57": "IEEE 57 what-if",
    "chat_placeholder": "Type your request...",
    "btn_run": "Run PF",
    "btn_v": "Voltage map",
    "btn_violation": "Violation overview",
    "btn_disconnect": "Disconnect",
    "btn_report": "Export report",
    "btn_n1": "N-1",
    "btn_remedial": "Remedial",
    "sidebar_n1": "N-1 Contingency",
    "sidebar_n1_topk": "Top-K",
    "sidebar_n1_go": "Run N-1",
    "sidebar_remedial": "Remedial actions",
    "disconnect_fb": "from_bus",
    "disconnect_tb": "to_bus",
    "disconnect_go": "Disconnect & analyze",
    "status": "Analyzing...",
    "status_done": "Done",
    "status_parse": "Parsing intent and planning tool calls...",
    "status_tools": "Running solver/validation/viz tools...",
    "status_answer": "Writing explanation...",
    "case_loaded": "Loaded case: {case}",
    "need_case": "Please load a test case first.",
    "need_result": "Please run power flow first before plotting.",
    "report_todo": "Report export will be implemented later.",
    "undo_done": "Undone the last change.",
    "remedial_apply": "Apply",
    "remedial_confirm_title": "Confirm applying remedial action",
    "remedial_confirm_warn": "This will modify the network and re-run power flow. Proceed?",
    "remedial_cancel": "Cancel",
    "remedial_confirm": "Confirm",
    "remedial_applied": "Applied remedial action #{idx}: {desc}",
    "remedial_apply_failed": "Failed to apply remedial action: {err}",
    "remedial_stale": "Note: suggestions may be stale after apply/undo; regenerate if needed.",
    "llm_raw_expander": "🔍 View LLM Raw Input/Output",
    "llm_prompt": "LLM Prompt",
    "llm_response": "LLM Raw Response",
    "external_import_title": "External LLM Result Import",
    "external_import_input": "Paste external LLM output JSON",
    "external_import_btn": "Import and Plot",
    "external_import_ok": "Imported external LLM result and rendered plots.",
    "external_import_err": "Failed to import external LLM result: {err}",
    "benchmark_btn": "Benchmark vs Ground Truth",
    "benchmark_ok": "LLM vs Ground Truth benchmark completed.",
    "benchmark_err": "Benchmark failed: {err}",
    "benchmark_need_llm": "Import an external LLM result first.",
    "benchmark_title": "LLM Benchmark Report",
    "remedial_none": "No remedial action available to apply. Generate suggestions first.",
}
# -----------------------------
# LLM provider options
# -----------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

LLM_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "gemini": "Gemini",
}

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]
OPENAI_MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1": "GPT-4.1",
}

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemini-3-flash",
]
GEMINI_MODEL_LABELS = {
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}


# -----------------------------
# Session init
# -----------------------------


def _init_state() -> None:
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # UI-only
    if "tool_ctx" not in st.session_state:
        st.session_state.tool_ctx = ToolContext(
            net=None,
            session=st.session_state.session,
            solver_config=SolverConfig(
                v_min=DEFAULT_V_MIN,
                v_max=DEFAULT_V_MAX,
                max_loading=DEFAULT_MAX_LOADING,
            ),
        )
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "net_history" not in st.session_state:
        st.session_state.net_history = []  # snapshots for undo
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "en"
    if "pending_remedial_apply" not in st.session_state:
        st.session_state.pending_remedial_apply = None  # {"index": int, "source": str}
    if "llm_provider" not in st.session_state:
        if GEMINI_API_KEY:
            st.session_state.llm_provider = "gemini"
        elif OPENAI_API_KEY:
            st.session_state.llm_provider = "openai"
        else:
            st.session_state.llm_provider = "gemini"
    if "llm_api_key" not in st.session_state:
        if st.session_state.llm_provider == "gemini":
            st.session_state.llm_api_key = GEMINI_API_KEY or ""
        else:
            st.session_state.llm_api_key = OPENAI_API_KEY or ""
    elif not str(st.session_state.get("llm_api_key", "")).strip():
        default_key = GEMINI_API_KEY if st.session_state.llm_provider == "gemini" else OPENAI_API_KEY
        if default_key:
            st.session_state.llm_api_key = default_key
    if "llm_model" not in st.session_state:
        if st.session_state.llm_provider == "gemini":
            st.session_state.llm_model = GEMINI_MODEL
        else:
            st.session_state.llm_model = OPENAI_MODEL
    if "llm_last_model_by_provider" not in st.session_state:
        current_provider = str(st.session_state.get("llm_provider", "gemini"))
        current_model = str(st.session_state.get("llm_model", GEMINI_MODEL if current_provider == "gemini" else OPENAI_MODEL))
        st.session_state.llm_last_model_by_provider = {
            "gemini": current_model if current_provider == "gemini" else GEMINI_MODEL,
            "openai": current_model if current_provider == "openai" else OPENAI_MODEL,
        }
    if "llm_temperature" not in st.session_state:
        if st.session_state.llm_provider == "gemini":
            st.session_state.llm_temperature = float(GEMINI_TEMPERATURE)
        else:
            st.session_state.llm_temperature = float(OPENAI_TEMPERATURE)
    if "llm_timeout_s" not in st.session_state:
        if st.session_state.llm_provider == "gemini":
            st.session_state.llm_timeout_s = float(GEMINI_TIMEOUT_S)
        else:
            st.session_state.llm_timeout_s = float(OPENAI_TIMEOUT_S)
    if "llm_only_debug_mode" not in st.session_state:
        st.session_state.llm_only_debug_mode = bool(LLM_ONLY_DEBUG_MODE)
    if "solver_backend" not in st.session_state:
        st.session_state.solver_backend = "pandapower"
    if "flow_heavy_threshold" not in st.session_state:
        st.session_state.flow_heavy_threshold = 60
    if "flow_over_threshold" not in st.session_state:
        st.session_state.flow_over_threshold = 100
    if "flow_show_labels" not in st.session_state:
        st.session_state.flow_show_labels = True
    if "flow_show_arrows" not in st.session_state:
        st.session_state.flow_show_arrows = True
    if "flow_show_vm_overlay" not in st.session_state:
        st.session_state.flow_show_vm_overlay = True
    if "flow_show_particles" not in st.session_state:
        st.session_state.flow_show_particles = True
    if "flow_particles_autoset_case" not in st.session_state:
        st.session_state.flow_particles_autoset_case = ""
    if "flow_color_scheme" not in st.session_state:
        st.session_state.flow_color_scheme = "light"
    if "flow_branch_override_df" not in st.session_state:
        st.session_state.flow_branch_override_df = None


def _get_theme_base() -> str:
    base = st.get_option("theme.base")
    if isinstance(base, str) and base.lower().startswith("dark"):
        return "dark"
    return "light"


def _get_plot_theme() -> str:
    """Global plot theme controlled by sidebar Color scheme."""
    scheme = str(st.session_state.get("flow_color_scheme", "light")).strip().lower()
    if scheme == "dark":
        return "dark"
    # "light" and "print" both render on light template.
    return "light"


def _models_for_provider(provider: str) -> list[str]:
    if provider == "gemini":
        return GEMINI_MODELS
    return OPENAI_MODELS


def _normalize_gemini_model_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    if s.startswith("models/"):
        return s.split("/", 1)[1]
    if "/models/" in s:
        return s.rsplit("/models/", 1)[1]
    return s


def _list_gemini_models_dynamic(api_key: str) -> tuple[list[str], Optional[str]]:
    key = str(api_key or "").strip()
    if not key:
        return GEMINI_MODELS, None

    cache = st.session_state.setdefault("_gemini_models_cache", {})
    if key in cache:
        cached = cache.get(key) or {}
        return list(cached.get("models") or GEMINI_MODELS), cached.get("error")

    try:
        from google import genai
    except Exception as e:
        err_text = str(e)
        if "API key expired" in err_text or "API_KEY_INVALID" in err_text:
            msg = "Gemini API key is invalid or expired. Please update API key."
        else:
            msg = f"dynamic model list unavailable: {type(e).__name__}"
        cache[key] = {"models": GEMINI_MODELS, "error": msg}
        return GEMINI_MODELS, msg

    try:
        client = genai.Client(api_key=key)
        mids: list[str] = []
        for m in client.models.list():
            actions = [str(a).lower() for a in (getattr(m, "supported_actions", None) or [])]
            if not any(("generate" in a and "content" in a) for a in actions):
                continue
            mid = _normalize_gemini_model_name(getattr(m, "name", ""))
            if not mid:
                continue
            mids.append(mid)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        model_list = []
        for mid in mids:
            if mid in seen:
                continue
            seen.add(mid)
            model_list.append(mid)
        # Keep a sensible fallback default visible.
        if GEMINI_MODEL and GEMINI_MODEL not in seen:
            model_list.append(GEMINI_MODEL)
        if not model_list:
            model_list = list(GEMINI_MODELS)
        cache[key] = {"models": model_list, "error": None}
        return model_list, None
    except Exception as e:
        err_text = str(e)
        if "API key expired" in err_text or "API_KEY_INVALID" in err_text:
            msg = "Gemini API key is invalid or expired. Please update API key."
        elif "PERMISSION_DENIED" in err_text:
            msg = "Gemini API key has no permission to list models."
        else:
            msg = f"dynamic model list failed: {type(e).__name__}"
        cache[key] = {"models": GEMINI_MODELS, "error": msg}
        return GEMINI_MODELS, msg


def _model_label(provider: str, model_id: str) -> str:
    if provider == "gemini":
        return GEMINI_MODEL_LABELS.get(model_id, model_id)
    return OPENAI_MODEL_LABELS.get(model_id, model_id)


def _provider_default_model(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_MODEL
    return OPENAI_MODEL


def _provider_default_key(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_API_KEY or ""
    return OPENAI_API_KEY or ""


def _resolve_llm_settings() -> tuple[str, str, str]:
    provider = str(st.session_state.get("llm_provider", "openai"))
    if provider not in LLM_PROVIDER_LABELS:
        provider = "openai"

    api_key = str(st.session_state.get("llm_api_key", "") or "").strip()
    model = str(st.session_state.get("llm_model", "") or "").strip()

    if not model:
        model = _provider_default_model(provider)
    return provider, api_key, model


def _sync_tool_context() -> None:
    """Sync UI/backend settings into tool context before dispatch."""

    ctx: ToolContext = st.session_state.tool_ctx
    provider, api_key, model = _resolve_llm_settings()
    ctx.theme = _get_plot_theme()
    ctx.solver_backend = str(st.session_state.get("solver_backend", "pandapower"))
    ctx.llm_provider = provider
    ctx.llm_api_key = api_key
    ctx.llm_model = model
    ctx.llm_base_url = GEMINI_BASE_URL if provider == "gemini" else None
    default_temp = GEMINI_TEMPERATURE if provider == "gemini" else OPENAI_TEMPERATURE
    default_timeout = GEMINI_TIMEOUT_S if provider == "gemini" else OPENAI_TIMEOUT_S
    ctx.llm_temperature = float(st.session_state.get("llm_temperature", default_temp))
    ctx.llm_timeout_s = float(st.session_state.get("llm_timeout_s", default_timeout))
    ctx.llm_only_debug_mode = bool(st.session_state.get("llm_only_debug_mode", LLM_ONLY_DEBUG_MODE))
    ctx.matpower_data_root = str(st.session_state.get("matpower_data_root", MATPOWER_DATA_ROOT))
    ctx.matpower_case_date = str(st.session_state.get("matpower_case_date", MATPOWER_CASE_DATE))
    ctx.ui_lang = str(st.session_state.get("ui_lang", "en"))


def _no_api_message(T: Dict[str, str], provider: str) -> str:
    if provider == "gemini":
        return (
            "未检测到 Gemini API Key（可设置 GEMINI_API_KEY/GOOGLE_API_KEY 或在侧边栏输入）。"
            if T is ZH
            else "Gemini API key is not set (configure GEMINI_API_KEY/GOOGLE_API_KEY or enter it in the sidebar)."
        )
    return (
        "未检测到 OpenAI API Key（可设置 OPENAI_API_KEY 或在侧边栏输入）。"
        if T is ZH
        else "OpenAI API key is not set (configure OPENAI_API_KEY or enter it in the sidebar)."
    )


def _llm_only_disabled_message() -> str:
    return "This feature is disabled in LLM-only mode due to high token cost and instability. Please switch to PandaPower backend."


def _format_llm_runtime_error(err: Exception, T: Dict[str, str]) -> str:
    s = str(err or "")
    if "API key expired" in s or "API_KEY_INVALID" in s:
        return (
            "Gemini API key is invalid or expired. Please update API key."
            if T is EN
            else "Gemini API Key 无效或已过期，请更新后重试。"
        )
    if "timeout" in s.lower():
        return "LLM request timed out." if T is EN else "LLM 请求超时。"
    return f"LLM request failed: {type(err).__name__}: {err}"


def _render_runtime_status_bar(T: Dict[str, str]) -> None:
    backend = str(st.session_state.get("solver_backend", "pandapower"))
    provider = str(st.session_state.get("llm_provider", "openai"))
    model = str(st.session_state.get("llm_model", "") or "").strip() or _provider_default_model(provider)
    temp = float(st.session_state.get("llm_temperature", GEMINI_TEMPERATURE if provider == "gemini" else OPENAI_TEMPERATURE))
    timeout_s = float(st.session_state.get("llm_timeout_s", GEMINI_TIMEOUT_S if provider == "gemini" else OPENAI_TIMEOUT_S))
    debug_mode = bool(st.session_state.get("llm_only_debug_mode", LLM_ONLY_DEBUG_MODE))
    api_key_set = bool(str(st.session_state.get("llm_api_key", "") or "").strip())

    backend_label = T["backend_llm"] if backend == "llm_only" else T["backend_pp"]
    provider_label = LLM_PROVIDER_LABELS.get(provider, provider)
    key_label = "set" if api_key_set else "missing"
    if T is not EN:
        key_label = "已设置" if api_key_set else "未设置"

    status_text = (
        f"{T['backend_active']}: {backend_label} | "
        f"Provider: {provider_label} | "
        f"Model: {model} | "
        f"Temperature: {temp:.2f} | "
        f"Timeout: {timeout_s:.0f}s | "
        f"Debug: {'on' if debug_mode else 'off'} | "
        f"API Key: {key_label}"
    )
    st.info(status_text)


def _extract_fetch_command(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(python scripts/fetch_matpower_cases\.py[^\n\r`]*)", str(text))
    if m:
        return m.group(1).strip()
    return None


def _format_tool_error(out: Dict[str, Any]) -> str:
    msg = str(out.get("error") or "Unknown tool error")
    cmd = out.get("fetch_command") or _extract_fetch_command(msg)
    if cmd:
        return f"{msg}\n\n{cmd}"
    return msg


def _parse_env_local(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if key:
            out[key] = v.strip()
    return out


def _persist_insecure_api_key(provider: str, api_key: str, model: Optional[str] = None) -> None:
    """
    Intentionally insecure: store API key in plain text `.env.local`.
    Silently skips on read-only filesystems (e.g. Streamlit Cloud).
    """
    env_path = Path(__file__).resolve().parent / ".env.local"
    try:
        env_map = _parse_env_local(env_path)
        p = str(provider or "").strip().lower()
        k = str(api_key or "").strip()
        if p == "gemini":
            if k:
                env_map["GEMINI_API_KEY"] = k
            else:
                env_map.pop("GEMINI_API_KEY", None)
            m = str(model or "").strip()
            if m:
                env_map["GEMINI_MODEL"] = m
        elif p == "openai":
            if k:
                env_map["OPENAI_API_KEY"] = k
            else:
                env_map.pop("OPENAI_API_KEY", None)
            m = str(model or "").strip()
            if m:
                env_map["OPENAI_MODEL"] = m
        lines = [f"{key}={value}" for key, value in sorted(env_map.items())]
        env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except OSError:
        pass


def _extract_json_block(raw_text: str) -> str:
    s = (raw_text or "").strip()
    if not s:
        raise ValueError("empty input")

    if "```" in s:
        parts = s.split("```")
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                return p

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    raise ValueError("no JSON object found")


def _normalize_external_result_payload(payload: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """Accept either direct PowerFlowResult dict or llm_pf schema with `totals`."""

    if isinstance(payload.get("result"), dict):
        payload = payload["result"]

    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    case_name = payload.get("case_name") or st.session_state.session.active_case or "external_case"
    def _bus_display_id_for_import(net_obj: Any, bus_idx: int) -> int:
        try:
            name = str(net_obj.bus.at[int(bus_idx), "name"]).strip()
            if name and re.fullmatch(r"[-+]?\d+", name):
                return int(name)
        except Exception:
            pass
        return int(bus_idx) + 1

    def _line_id_by_endpoints(net_obj: Any, fb: int, tb: int) -> Optional[int]:
        if net_obj is None or not hasattr(net_obj, "line"):
            return None
        try:
            for idx, row in net_obj.line.iterrows():
                a = _bus_display_id_for_import(net_obj, int(row["from_bus"]))
                b = _bus_display_id_for_import(net_obj, int(row["to_bus"]))
                if (a == int(fb) and b == int(tb)) or (a == int(tb) and b == int(fb)):
                    return int(idx)
        except Exception:
            return None
        return None

    raw_lines = payload.get("line_flows") or []
    norm_lines: list[Dict[str, Any]] = []
    net_obj = None
    try:
        net_obj = st.session_state.tool_ctx.net
    except Exception:
        net_obj = None
    if isinstance(raw_lines, list):
        for i, row in enumerate(raw_lines):
            if not isinstance(row, dict):
                continue
            nr = dict(row)
            # External JSONs often use null when RATE_A == 0; UI schema requires float.
            if nr.get("loading_percent") is None:
                nr["loading_percent"] = 0.0
            mapped_id = None
            try:
                fb = int(nr.get("from_bus"))
                tb = int(nr.get("to_bus"))
                mapped_id = _line_id_by_endpoints(net_obj, fb, tb)
            except Exception:
                mapped_id = None

            if mapped_id is not None:
                nr["line_id"] = int(mapped_id)
            else:
                # Fallback keeps caller-provided id.
                try:
                    nr["line_id"] = int(nr.get("line_id", i + 1))
                except Exception:
                    nr["line_id"] = i + 1
            norm_lines.append(nr)

    out: Dict[str, Any] = {
        "case_name": case_name,
        "converged": bool(payload.get("converged", True)),
        "bus_voltages": payload.get("bus_voltages") or [],
        "line_flows": norm_lines,
        "total_generation_mw": payload.get("total_generation_mw", totals.get("total_generation_mw", 0.0)),
        "total_load_mw": payload.get("total_load_mw", totals.get("total_load_mw", 0.0)),
        "total_loss_mw": payload.get("total_loss_mw", totals.get("total_loss_mw", 0.0)),
        "summary_text": payload.get("summary_text") or "Imported external LLM output.",
        "solver_backend": payload.get("solver_backend") or "llm_only_external",
        "llm_prompt": payload.get("llm_prompt"),
        "llm_response": payload.get("llm_response") or raw_text,
    }
    return out


def _import_external_llm_result(raw_text: str, T: Dict[str, str]) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return

    try:
        json_text = _extract_json_block(raw_text)
        payload = safe_json_loads(json_text)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("parsed JSON is empty or not an object")
        normalized = _normalize_external_result_payload(payload, raw_text)
        result = PowerFlowResult.model_validate(normalized)
        result = validate_result(
            ctx.net,
            result,
            v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
            v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
            max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
        )
    except Exception as e:
        _append_ui_message("assistant", T["external_import_err"].format(err=f"{type(e).__name__}: {e}"))
        st.rerun()
        return

    ctx.prev_result = session.last_result
    session.last_result = result

    plot_json, extra_plots, cmp_note = _build_all_plots_payload(T)
    msg = T["external_import_ok"]
    if cmp_note:
        msg = f"{msg}\n\n{cmp_note}"
    _append_ui_message("assistant", msg, plot_json=plot_json, extra_plots=extra_plots, result=result.model_dump())
    st.rerun()


def _benchmark_external_vs_truth(T: Dict[str, str]) -> None:
    """Run PandaPower ground truth, compare with imported LLM result, render benchmark."""
    import copy

    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session

    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return

    if session.last_result is None or "llm" not in str(getattr(session.last_result, "solver_backend", "")).lower():
        _append_ui_message("assistant", T["benchmark_need_llm"])
        st.rerun()
        return

    try:
        net_copy = copy.deepcopy(ctx.net)
        solver_cfg = SolverConfig(
            v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
            v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
            max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
        )
        truth = run_power_flow(net_copy, config=solver_cfg)
        if not truth.converged:
            raise RuntimeError("Ground truth power flow did not converge")

        parsed = baseline_parsed_from_result(session.last_result)
        metrics = evaluate_against_truth_extended(
            parsed, truth,
            v_min=solver_cfg.v_min,
            v_max=solver_cfg.v_max,
            max_loading=solver_cfg.max_loading,
        )

        lang = st.session_state.get("ui_lang", "en")
        bench_fig = make_benchmark_figure(metrics, lang=lang)

        extra_plots = [
            {
                "plot_type": "benchmark",
                "figure_json": pio.to_json(bench_fig, validate=False),
                "title": T["benchmark_title"],
                "metrics": metrics,
            }
        ]

        _append_ui_message(
            "assistant",
            T["benchmark_ok"],
            extra_plots=extra_plots,
            result=session.last_result.model_dump(),
        )
    except Exception as e:
        _append_ui_message("assistant", T["benchmark_err"].format(err=f"{type(e).__name__}: {e}"))

    st.rerun()


def _build_engine() -> Optional[LLMEngine]:
    provider, api_key, model = _resolve_llm_settings()
    if not api_key:
        return None

    ctx: ToolContext = st.session_state.tool_ctx
    _sync_tool_context()

    dispatcher = build_default_dispatcher(ctx)
    base_url = GEMINI_BASE_URL if provider == "gemini" else None
    client = OpenAIChatClient(api_key=api_key, base_url=base_url)
    cfg = EngineConfig(
        model=model,
        temperature=float(st.session_state.get("llm_temperature", OPENAI_TEMPERATURE)),
        timeout_s=float(st.session_state.get("llm_timeout_s", OPENAI_TIMEOUT_S)),
    )
    return LLMEngine(client=client, dispatcher=dispatcher, config=cfg)


def _push_snapshot(label: str = "") -> None:
    """Save one undo snapshot (net + result + modification_log)."""

    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None:
        return

    st.session_state.net_history.append(
        {
            "label": label,
            "net": copy.deepcopy(ctx.net),
            "active_case": session.active_case,
            "last_result": copy.deepcopy(session.last_result),
            "last_n1_report": copy.deepcopy(session.last_n1_report),
            "last_remedial_plan": copy.deepcopy(session.last_remedial_plan),
            "modification_log": copy.deepcopy(session.modification_log),
        }
    )


def _undo_last() -> None:
    if len(st.session_state.net_history) <= 1:
        return

    st.session_state.net_history.pop()
    snap = st.session_state.net_history[-1]

    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session

    ctx.net = snap["net"]
    ctx.cached_positions = None
    session.active_case = snap["active_case"]
    session.last_result = snap["last_result"]
    session.last_n1_report = snap.get("last_n1_report")
    session.last_remedial_plan = snap.get("last_remedial_plan")
    session.modification_log = snap["modification_log"]


def _ensure_positions() -> Dict[int, tuple[float, float]]:
    ctx: ToolContext = st.session_state.tool_ctx
    if ctx.net is None:
        return {}
    if ctx.cached_positions is None:
        g = build_graph(ctx.net)
        ctx.cached_positions = compute_layout(ctx.net, g)
    return ctx.cached_positions


def _flow_plot_kwargs() -> Dict[str, Any]:
    heavy = float(st.session_state.get("flow_heavy_threshold", 60))
    over = float(st.session_state.get("flow_over_threshold", 100))
    if over <= heavy:
        over = heavy + 1.0
    return {
        "heavy_threshold": heavy,
        "over_threshold": over,
        "show_power_labels": bool(st.session_state.get("flow_show_labels", True)),
        "show_flow_arrows": bool(st.session_state.get("flow_show_arrows", True)),
        "show_voltage_overlay": bool(st.session_state.get("flow_show_vm_overlay", True)),
        "color_scheme": str(st.session_state.get("flow_color_scheme", "light")),
        "branch_override_df": st.session_state.get("flow_branch_override_df"),
    }


def _flow_particles_enabled() -> bool:
    return bool(st.session_state.get("flow_show_particles", True))


def _build_flow_particle_html(fig_json: str, positions: Optional[Dict[int, tuple[float, float]]] = None) -> Optional[str]:
    if not _flow_particles_enabled():
        return None
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None or session.last_result is None:
        return None

    base_pos = positions if positions is not None else _ensure_positions()
    pos = resolve_flow_positions(
        ctx.net,
        session.last_result,
        positions=base_pos,
        use_ieee14_fixed_layout=True,
    )
    if not pos:
        return None

    heavy = float(st.session_state.get("flow_heavy_threshold", 60))
    over = float(st.session_state.get("flow_over_threshold", 100))
    if over <= heavy:
        over = heavy + 1.0

    try:
        particle_theme = str(st.session_state.get("flow_color_scheme", _get_theme_base()))
        backend = str(getattr(session.last_result, "solver_backend", "") or "")
        use_line_id_mapping = backend in {"llm_only", "llm_only_external"}
        segs = build_particle_segments(
            ctx.net,
            session.last_result,
            pos,
            theme=particle_theme,
            heavy_threshold=heavy,
            over_threshold=over,
            max_particles=800,
            use_line_id_mapping=use_line_id_mapping,
        )
        if not segs:
            return None
        return make_flow_particles_html(fig_json, segs, height_px=700)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("Flow particle build failed: %s", exc, exc_info=True)
        return None


def _build_flow_plot_artifacts(
    *,
    positions: Optional[Dict[int, tuple[float, float]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None or session.last_result is None:
        return None, None

    base_pos = positions if positions is not None else _ensure_positions()
    pos = resolve_flow_positions(
        ctx.net,
        session.last_result,
        positions=base_pos,
        use_ieee14_fixed_layout=True,
    )
    fig = make_flow_diagram(
        ctx.net,
        session.last_result,
        positions=pos,
        use_ieee14_fixed_layout=False,
        theme=_get_plot_theme(),
        lang=st.session_state.get("ui_lang", "en"),
        **_flow_plot_kwargs(),
    )
    fig_json = pio.to_json(fig, validate=False)
    flow_html = _build_flow_particle_html(fig_json, positions=pos)
    return fig_json, flow_html


def _auto_default_plot() -> tuple[Optional[str], Optional[str]]:
    """Generate a default plot when no explicit plot tool is invoked."""

    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None or session.last_result is None:
        return None, None
    if not bool(getattr(session.last_result, "converged", False)):
        return None, None

    return _build_flow_plot_artifacts(positions=_ensure_positions())


def _append_ui_message(
    role: str,
    content: str,
    *,
    plot_json: Optional[str] = None,
    plot_type: Optional[str] = None,
    plot_html: Optional[str] = None,
    n1_report: Optional[Dict[str, Any]] = None,
    remedial_plan: Optional[Dict[str, Any]] = None,
    extra_plots: Optional[list[Dict[str, Any]]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    msg: Dict[str, Any] = {"role": role, "content": content}
    if plot_json:
        msg["plot_json"] = plot_json
    if plot_type:
        msg["plot_type"] = plot_type
    if plot_html:
        msg["plot_html"] = plot_html
    if n1_report:
        msg["n1_report"] = n1_report
    if remedial_plan:
        msg["remedial_plan"] = remedial_plan
    if result:
        msg["result"] = result
    if extra_plots:
        msg["extra_plots"] = extra_plots
    # Attach latest result details to assistant messages for richer UI rendering.
    if role == "assistant" and st.session_state.session.last_result is not None:
        msg["result"] = st.session_state.session.last_result.model_dump()
    st.session_state.messages.append(msg)


def _split_comparison_figure(fig: go.Figure, T: Dict[str, str]) -> list[go.Figure]:
    """Split legacy comparison subplot into Before / After / Top-Changes-Table figures."""
    has_second = hasattr(fig.layout, "xaxis2") and hasattr(fig.layout, "yaxis2")
    if not has_second:
        return [fig]

    out: list[go.Figure] = []
    panel_specs = [
        ("x", "y", "Before" if T is EN else "修改前"),
        ("x2", "y2", "After" if T is EN else "修改后"),
    ]
    for xkey, ykey, title in panel_specs:
        f = go.Figure()
        for tr in fig.data:
            if str(getattr(tr, "type", "") or "") == "table":
                continue
            tx = getattr(tr, "xaxis", None) or "x"
            ty = getattr(tr, "yaxis", None) or "y"
            if str(tx) == xkey and str(ty) == ykey:
                tr_copy = copy.deepcopy(tr)
                if hasattr(tr_copy, "xaxis"):
                    tr_copy.xaxis = "x"
                if hasattr(tr_copy, "yaxis"):
                    tr_copy.yaxis = "y"
                f.add_trace(tr_copy)

        # Preserve panel-local annotations (e.g., red Delta P / Delta V labels on "After").
        panel_annotations = []
        for ann in list(getattr(fig.layout, "annotations", []) or []):
            axref = str(getattr(ann, "xref", "") or "")
            ayref = str(getattr(ann, "yref", "") or "")
            if axref == xkey and ayref == ykey:
                ann_copy = copy.deepcopy(ann)
                ann_copy.xref = "x"
                ann_copy.yref = "y"
                panel_annotations.append(ann_copy)

        f.update_layout(
            title=title,
            template=fig.layout.template,
            height=680,
            margin=dict(l=10, r=10, t=56, b=10),
            annotations=panel_annotations,
        )
        f.update_xaxes(showgrid=False, zeroline=False, visible=False)
        f.update_yaxes(showgrid=False, zeroline=False, visible=False)
        out.append(f)

    table_traces = [tr for tr in fig.data if str(getattr(tr, "type", "") or "") == "table"]
    if table_traces:
        table_fig = go.Figure()
        for tr in table_traces:
            td = copy.deepcopy(tr.to_plotly_json())
            td.pop("domain", None)
            table_fig.add_trace(go.Table(**td))

        full_h = int(float(getattr(fig.layout, "height", 1600) or 1600))
        # Original table row was 0.16 of full height; render standalone at ~2x.
        table_h = max(420, int(round(full_h * 0.32)))
        table_fig.update_layout(
            title=("Top Changes Table" if T is EN else "关键变化表"),
            template=fig.layout.template,
            height=table_h,
            margin=dict(l=10, r=10, t=56, b=10),
        )
        out.append(table_fig)
    return out


def _render_qc_summary_row(
    ep: Dict[str, Any], mi: int, epi: int, T: Dict[str, str]
) -> None:
    """Render the summary metrics row below the quantitative comparison chart."""
    before_dict = ep.get("before_result")
    after_dict = ep.get("after_result")
    if not before_dict or not after_dict:
        return
    try:
        before_r = PowerFlowResult(**before_dict)
        after_r = PowerFlowResult(**after_dict)
    except Exception:
        return

    summaries = compute_comparison_summary(before_r, after_r)
    lang = st.session_state.get("ui_lang", "en")
    cols = st.columns(len(summaries))
    for ci, (col, s) in enumerate(zip(cols, summaries)):
        label = s["label_zh"] if lang == "zh" else s["label_en"]
        text = s["fmt"](s["before"], s["after"], s["delta"])
        if s["improved"]:
            color = "#38A169"
        elif s["worsened"]:
            color = "#E53E3E"
        else:
            color = "#718096"
        col.markdown(
            f"<div style='text-align:center; padding:8px 4px; border-radius:8px; "
            f"background:#fff; box-shadow:0 1px 2px rgba(0,0,0,0.06);'>"
            f"<span style='font-family:Source Sans 3,sans-serif; font-size:12px; "
            f"color:#718096;'>{label}</span><br>"
            f"<span style='font-family:Source Sans 3,sans-serif; font-size:15px; "
            f"font-weight:600; color:{color};'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_benchmark_cards(ep: Dict[str, Any], mi: int, epi: int, T: Dict[str, str]) -> None:
    """Render 14 benchmark metric cards in tiered rows."""
    metrics = ep.get("metrics")
    if not metrics:
        return

    lang = st.session_state.get("ui_lang", "en")
    cards = compute_benchmark_cards(metrics, lang=lang)

    tier_labels = {
        1: ("State Estimation Accuracy", "状态估计精度"),
        2: ("Flow Estimation Accuracy", "潮流估计精度"),
        3: ("Physical Consistency", "物理一致性"),
        4: ("Advanced Metrics", "高级指标"),
    }

    tiers: Dict[int, list] = {}
    for card in cards:
        tiers.setdefault(card.get("tier", 0), []).append(card)

    for tier_num in sorted(tiers.keys()):
        tier_cards = tiers[tier_num]
        label_en, label_zh = tier_labels.get(tier_num, (f"Tier {tier_num}", f"层级 {tier_num}"))
        tier_label = label_zh if lang == "zh" else label_en

        st.markdown(
            f"<div style='margin-top:8px; margin-bottom:4px;'>"
            f"<span style='font-family:Source Sans 3,sans-serif; font-size:13px; "
            f"color:#4A5568; font-weight:600;'>{tier_label}</span></div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(tier_cards))
        for ci, (col, card) in enumerate(zip(cols, tier_cards)):
            label = card["label_zh"] if lang == "zh" else card["label_en"]
            value = card["value"]
            unit = card.get("unit", "")
            color = card["color"]
            fmt = card.get("fmt", ".4f")

            if isinstance(value, bool):
                display_val = "Yes" if value else "No"
            elif value is None:
                display_val = "N/A"
            elif isinstance(value, float) and fmt:
                display_val = f"{value:{fmt}}"
            else:
                display_val = str(value)

            unit_str = f" {unit}" if unit else ""
            col.markdown(
                f"<div style='text-align:center; padding:8px 4px; border-radius:8px; "
                f"background:#fff; box-shadow:0 1px 2px rgba(0,0,0,0.06);'>"
                f"<span style='font-family:Source Sans 3,sans-serif; font-size:11px; "
                f"color:#718096;'>{label}</span><br>"
                f"<span style='font-family:Source Sans 3,sans-serif; font-size:15px; "
                f"font-weight:600; color:{color};'>{display_val}{unit_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_result_details(result_dict: Dict[str, Any], T: Dict[str, str]) -> None:
    """Render structured result details in an expandable panel."""
    backend = str(result_dict.get("solver_backend") or st.session_state.get("solver_backend", "pandapower"))
    st.info(f"{T['backend_active']}: `{backend}`")

    with st.expander("Key metrics", expanded=False):
        st.write(
            {
                "case": result_dict.get("case_name"),
                "converged": result_dict.get("converged"),
                "total_load_mw": result_dict.get("total_load_mw"),
                "total_generation_mw": result_dict.get("total_generation_mw"),
                "total_loss_mw": result_dict.get("total_loss_mw"),
                "solver_backend": backend,
                "n_voltage_violations": len(result_dict.get("voltage_violations") or []),
                "n_thermal_violations": len(result_dict.get("thermal_violations") or []),
            }
        )
    if not bool(result_dict.get("converged", True)):
        summary = str(result_dict.get("summary_text") or "").strip()
        if not summary:
            summary = "Power flow did not converge." if T is EN else "潮流计算未收敛。"
        st.error(summary)

    if backend == "llm_only":
        prompt = result_dict.get("llm_prompt")
        response = result_dict.get("llm_response")
        if prompt or response:
            with st.expander(T["llm_raw_expander"], expanded=False):
                st.markdown(f"**{T['llm_prompt']}**")
                prompt_text = str(prompt or "")
                st.caption(f"{len(prompt_text):,} chars")
                st.download_button(
                    "Download Prompt",
                    data=prompt_text,
                    file_name=f"{result_dict.get('case_name','case')}_llm_prompt.txt",
                    mime="text/plain",
                    key=f"dl_prompt_{id(result_dict)}",
                )
                st.text_area(
                    T["llm_prompt"],
                    value=prompt_text,
                    height=360,
                    key=f"ta_prompt_{id(result_dict)}",
                    disabled=True,
                )
                st.markdown(f"**{T['llm_response']}**")
                response_text = str(response or "")
                st.caption(f"{len(response_text):,} chars")
                st.download_button(
                    "Download Response",
                    data=response_text,
                    file_name=f"{result_dict.get('case_name','case')}_llm_response.json",
                    mime="application/json",
                    key=f"dl_resp_{id(result_dict)}",
                )
                st.text_area(
                    T["llm_response"],
                    value=response_text,
                    height=360,
                    key=f"ta_resp_{id(result_dict)}",
                    disabled=True,
                )

    vvs = result_dict.get("voltage_violations") or []
    tvs = result_dict.get("thermal_violations") or []
    if vvs:
        st.warning(
            f"⚠️ Voltage violation buses: {len(vvs)}" if T is EN else f"⚠️ 电压越限节点：{len(vvs)}"
        )
        st.dataframe(vvs, use_container_width=True)
    if tvs:
        st.error(
            f"🔥 Overloaded lines/transformers: {len(tvs)}" if T is EN else f"🔥 线路/变压器过载：{len(tvs)}"
        )
        st.dataframe(tvs, use_container_width=True)


def _render_bottom_stats(T: Dict[str, str]) -> None:
    session: SessionState = st.session_state.session
    result = session.last_result
    if result is None:
        return
    max_loading = max((float(l.loading_percent) for l in result.line_flows), default=0.0)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Generation" if T is EN else "总发电量", f"{result.total_generation_mw:.2f} MW")
    with c2:
        st.metric("Total Load" if T is EN else "总负荷", f"{result.total_load_mw:.2f} MW")
    with c3:
        st.metric("System Losses" if T is EN else "总损耗", f"{result.total_loss_mw:.2f} MW")
    with c4:
        st.metric("Max Branch Loading" if T is EN else "最大支路负载率", f"{max_loading:.1f}%")



def _is_all_plots_request(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False
    keywords = [
        "all plots",
        "plot all",
        "all-in-one",
        "一次性",
        "全部输出",
        "四张图",
        "全部图",
    ]
    return any(k in t for k in keywords)


def _build_all_plots_payload(T: Dict[str, str]) -> tuple[str, list[Dict[str, Any]], str]:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    positions = _ensure_positions()
    theme = _get_plot_theme()

    voltage_fig = make_voltage_heatmap(
        ctx.net,
        session.last_result,
        positions=positions,
        theme=theme,
        lang=st.session_state.get("ui_lang", "en"),
    )
    lang = st.session_state.get("ui_lang", "en")
    flow_json, flow_html = _build_flow_plot_artifacts(positions=positions)
    if flow_json is None:
        flow_json = pio.to_json(
            make_flow_diagram(
                ctx.net,
                session.last_result,
                positions=positions,
                theme=theme,
                lang=lang,
                **_flow_plot_kwargs(),
            ),
            validate=False,
        )
    violation_fig = make_violation_overview(ctx.net, session.last_result, positions=positions, theme=theme, lang=lang)

    cmp_before = ctx.prev_result if ctx.prev_result is not None else session.last_result
    cmp_note = ""
    if ctx.prev_result is None:
        cmp_note = (
            "（What-if 对比：未检测到上一工况，当前按 before=after 展示。）"
            if T is ZH
            else "(What-if comparison: previous scenario not found; rendering with before=after.)"
        )
    comparison_fig = make_comparison(ctx.net, cmp_before, session.last_result, positions=positions, theme=theme, lang=lang)

    plot_json = pio.to_json(voltage_fig, validate=False)
    extra_plots = [
        {
            "plot_type": "flow_diagram",
            "figure_json": flow_json,
            "html": flow_html,
            "title": ("Flow Distribution" if T is EN else "潮流分布图"),
        },
        {
            "plot_type": "violation_overview",
            "figure_json": pio.to_json(violation_fig, validate=False),
            "title": ("Violation Overview" if T is EN else "越限概览图"),
        },
        {
            "plot_type": "comparison",
            "figure_json": pio.to_json(comparison_fig, validate=False),
            "title": ("What-if Comparison" if T is EN else "What-if 对比图"),
        },
    ]
    return plot_json, extra_plots, cmp_note


def _plot_all_direct(T: Dict[str, str]) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return

    if session.last_result is None:
        ctx.solver_config = SolverConfig(
            v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
            v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
            max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
        )
        _sync_tool_context()
        dispatcher = build_default_dispatcher(ctx)
        out = safe_json_loads(dispatcher.dispatch("run_powerflow", {}))
        if out.get("error"):
            _append_ui_message("assistant", f"Error: {_format_tool_error(out)}")
            st.rerun()
            return
        _push_snapshot(label="run_pf")

    plot_json, extra_plots, cmp_note = _build_all_plots_payload(T)
    msg = "已一次性输出 4 张图：电压图、潮流图、越限概览图、What-if 对比图。"
    if T is EN:
        msg = "Rendered all 4 plots at once: voltage, flow, violation overview, and what-if comparison."
    if cmp_note:
        msg = f"{msg}\n\n{cmp_note}"

    _append_ui_message("assistant", msg, plot_json=plot_json, extra_plots=extra_plots)
    st.rerun()


def process_user_input(user_text: str, T: Dict[str, str]) -> None:
    """Core interaction: user message -> engine -> assistant message + plots."""

    st.session_state.conversation_started = True
    _append_ui_message("user", user_text)

    if _is_all_plots_request(user_text):
        _plot_all_direct(T)
        return

    engine = _build_engine()
    provider, _, _ = _resolve_llm_settings()
    no_api_message = _no_api_message(T, provider)
    session: SessionState = st.session_state.session
    ctx: ToolContext = st.session_state.tool_ctx

    # Sync sidebar thresholds into tool context.
    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )
    _sync_tool_context()

    start_len = len(session.conversation_history)
    prev_net_case = session.active_case
    prev_mod_len = len(session.modification_log)
    prev_last_result_id = id(session.last_result) if session.last_result is not None else None

    with st.status(T["status"], expanded=True) as status:
        status.write(T["status_parse"])
        if engine is None:
            status.write(no_api_message)
            assistant_text = no_api_message
        else:
            status.write(T["status_tools"])
            active_case = str(session.active_case or "").strip()
            case_guard = (
                f"\n\n[Current active case in runtime state: {active_case or 'none'}. "
                "Do not reference any other case unless you first call load_case.]"
            )
            lang_locked_text = user_text + case_guard
            if T is EN:
                lang_locked_text = (
                    user_text
                    + case_guard
                    + "\n\n[Language requirement: Reply in English only. Do not use Chinese in the final answer.]"
                )
            else:
                lang_locked_text = user_text + case_guard + "\n\n[语言要求：最终回答请仅使用中文。]"
            try:
                assistant_text = engine.run(lang_locked_text, session)
            except Exception as e:
                assistant_text = _format_llm_runtime_error(e, T)
        status.write(T["status_answer"])
        status.update(label=T["status_done"], state="complete")

    new_entries = session.conversation_history[start_len:]
    artifacts = extract_tool_artifacts(new_entries)

    # Snapshot strategy:
    # - After successful load_case: reset history and save first snapshot.
    # - After topology/modification changes: save snapshot for undo.
    loaded_case = False
    for a in artifacts:
        if a.tool_name == "load_case" and not a.payload.get("error"):
            loaded_case = True
            break

    if loaded_case:
        st.session_state.net_history = []
        # Case switch should start a fresh LLM context to avoid stale case carry-over.
        session.conversation_history = []
        _push_snapshot(label=f"load:{session.active_case}")

    changed_topology = len(session.modification_log) > prev_mod_len
    if changed_topology and ctx.net is not None:
        _push_snapshot(label=f"mod:{len(session.modification_log)}")

    # Extract N-1 report if present.
    last_n1 = pick_last_n1_report(artifacts)
    n1_report = last_n1.n1_report if last_n1 else None

    # Extract remedial plan if present.
    remedial_plan = None
    remedial_extra = []
    extra_plots = []
    result_payload = None
    for a in reversed(artifacts):
        if getattr(a, "has_remedial_plan", False):
            remedial_plan = a.remedial_plan
            remedial_extra = a.extra_figures
            break

    # Generic extra figures (e.g. from apply_remedial_action).
    for a in reversed(artifacts):
        if a.extra_figures:
            extra_plots = a.extra_figures
            break
    # Add canvas overlay payload for flow diagrams in extra plots.
    if extra_plots:
        for ep in extra_plots:
            if ep.get("plot_type") == "flow_diagram" and isinstance(ep.get("figure_json"), str):
                ep.setdefault("html", _build_flow_particle_html(ep["figure_json"]))

    # Some tools return `result` directly in payload.
    for a in reversed(artifacts):
        payload = a.payload or {}
        if isinstance(payload.get("result"), dict):
            result_payload = payload.get("result")
            break

    # Extract primary plot.
    last_plot = pick_last_plot(artifacts)
    plot_json = last_plot.figure_json if last_plot else None
    plot_type = last_plot.plot_type if last_plot else None
    plot_html = None
    if plot_json is not None and plot_type == "flow_diagram":
        plot_html = _build_flow_particle_html(plot_json)
    if plot_json is None:
        # If result changed and no explicit plot was produced, auto-render a default plot.
        last_result_id = id(session.last_result) if session.last_result is not None else None
        if last_result_id is not None and last_result_id != prev_last_result_id:
            plot_json, plot_html = _auto_default_plot()
            if plot_json:
                plot_type = "flow_diagram"

    # If user asks for report export, append a placeholder notice for now.
    if "导出" in user_text or "报告" in user_text or "report" in user_text.lower():
        if "generate_report" not in user_text:
            assistant_text = (assistant_text + "\n\n" + T["report_todo"]).strip()

    render_extra_plots = extra_plots or remedial_extra
    if render_extra_plots:
        for ep in render_extra_plots:
            if ep.get("plot_type") == "flow_diagram" and isinstance(ep.get("figure_json"), str):
                ep.setdefault("html", _build_flow_particle_html(ep["figure_json"]))

    # assistant message
    _append_ui_message(
        "assistant",
        assistant_text or "",
        plot_json=plot_json,
        plot_type=plot_type,
        plot_html=plot_html,
        n1_report=n1_report,
        remedial_plan=remedial_plan,
        extra_plots=render_extra_plots,
        result=result_payload,
    )

    # Keep sidebar case selection in sync when case is loaded through chat.
    if session.active_case and session.active_case != prev_net_case:
        st.session_state["selected_case"] = session.active_case

    st.rerun()


# -----------------------------
# Sidebar actions (direct tool call)
# -----------------------------


def _load_case_direct(case_key: str, T: Dict[str, str]) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )
    _sync_tool_context()

    dispatcher = build_default_dispatcher(ctx)
    out = safe_json_loads(dispatcher.dispatch("load_case", {"case_name": case_key}))
    if out.get("error"):
        _append_ui_message("assistant", f"Error: {_format_tool_error(out)}")
        st.rerun()
        return

    st.session_state.net_history = []
    # Reset LLM dialogue memory on case switch to avoid stale case references.
    session.conversation_history = []
    _push_snapshot(label=f"load:{case_key}")
    _append_ui_message("assistant", T["case_loaded"].format(case=case_key))
    st.rerun()


def _run_pf_direct(T: Dict[str, str]) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return

    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )
    _sync_tool_context()

    dispatcher = build_default_dispatcher(ctx)
    out = safe_json_loads(dispatcher.dispatch("run_powerflow", {}))
    if out.get("error"):
        _append_ui_message("assistant", f"Error: {_format_tool_error(out)}")
        st.rerun()
        return
    if not bool(out.get("converged", False)):
        summary = str(out.get("summary_text") or "").strip()
        if not summary:
            summary = "Power flow did not converge." if T is EN else "潮流计算未收敛。"
        _append_ui_message("assistant", f"❌ {summary}", result=out)
        _push_snapshot(label="run_pf_failed")
        st.rerun()
        return

    # Default plot after successful PF.
    positions = _ensure_positions()
    plot_json, plot_html = _build_flow_plot_artifacts(positions=positions)
    _append_ui_message(
        "assistant",
        ("✅ Power flow completed." if T is EN else "✅ 潮流计算完成。"),
        plot_json=plot_json,
        plot_type="flow_diagram",
        plot_html=plot_html,
    )
    _push_snapshot(label="run_pf")
    st.rerun()


def _plot_direct(plot_type: str, T: Dict[str, str]) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return
    if session.last_result is None:
        _append_ui_message("assistant", T["need_result"])
        st.rerun()
        return

    positions = _ensure_positions()
    theme = _get_plot_theme()
    if plot_type == "voltage_heatmap":
        fig = make_voltage_heatmap(
            ctx.net,
            session.last_result,
            positions=positions,
            theme=theme,
            lang=st.session_state.get("ui_lang", "en"),
        )
        _append_ui_message(
            "assistant",
            ("📊 Plot generated." if T is EN else "📊 图表已生成。"),
            plot_json=pio.to_json(fig, validate=False),
            plot_type="voltage_heatmap",
        )
    elif plot_type == "flow_diagram":
        plot_json, plot_html = _build_flow_plot_artifacts(positions=positions)
        _append_ui_message(
            "assistant",
            ("📊 Plot generated." if T is EN else "📊 图表已生成。"),
            plot_json=plot_json,
            plot_type="flow_diagram",
            plot_html=plot_html,
        )
    elif plot_type == "violation_overview":
        fig = make_violation_overview(
            ctx.net,
            session.last_result,
            positions=positions,
            theme=theme,
            lang=st.session_state.get("ui_lang", "en"),
        )
        _append_ui_message(
            "assistant",
            ("📊 Plot generated." if T is EN else "📊 图表已生成。"),
            plot_json=pio.to_json(fig, validate=False),
            plot_type="violation_overview",
        )
    else:
        _append_ui_message(
            "assistant",
            (f"❌ Unsupported plot_type: {plot_type}" if T is EN else f"❌ 不支持的 plot_type: {plot_type}"),
        )
        st.rerun()
        return
    st.rerun()


def _run_n1_direct(T: Dict[str, str], *, top_k: int = 5) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if str(st.session_state.get("solver_backend", "pandapower")) == "llm_only":
        _append_ui_message("assistant", f"❌ {_llm_only_disabled_message()}")
        st.rerun()
        return
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return
    if session.last_result is None:
        _append_ui_message("assistant", T["need_result"])
        st.rerun()
        return

    ctx.theme = _get_plot_theme()
    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )

    report = run_n1_contingency(ctx.net, top_k=int(top_k), criteria="max_violations", config=ctx.solver_config)
    session.last_n1_report = report
    fig = make_n1_ranking(report, theme=_get_plot_theme(), lang=st.session_state.get("ui_lang", "en"))
    _append_ui_message(
        "assistant",
        ("🧨 N-1 analysis completed." if T is EN else "🧨 N-1 分析完成。"),
        plot_json=pio.to_json(fig, validate=False),
        n1_report=report.model_dump(),
    )
    st.rerun()


def _run_remedial_direct(T: Dict[str, str], *, max_actions: int = 5) -> None:
    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if str(st.session_state.get("solver_backend", "pandapower")) == "llm_only":
        _append_ui_message("assistant", f"❌ {_llm_only_disabled_message()}")
        st.rerun()
        return
    if ctx.net is None:
        _append_ui_message("assistant", T["need_case"])
        st.rerun()
        return
    if session.last_result is None:
        # Convenience path: auto-run PF before remedial generation.
        ctx.solver_config = SolverConfig(
            v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
            v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
            max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
        )
        _sync_tool_context()
        dispatcher = build_default_dispatcher(ctx)
        out = safe_json_loads(dispatcher.dispatch("run_powerflow", {}))
        if out.get("error"):
            _append_ui_message("assistant", f"Error: {_format_tool_error(out)}")
            st.rerun()
            return
        if not bool(out.get("converged", False)):
            summary = str(out.get("summary_text") or "").strip()
            if not summary:
                summary = "Power flow did not converge." if T is EN else "潮流计算未收敛。"
            _append_ui_message("assistant", f"❌ {summary}", result=out)
            st.rerun()
            return

    ctx.theme = _get_plot_theme()
    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )

    plan = recommend_remedial_actions(
        ctx.net,
        session.last_result,
        config=ctx.solver_config,
        max_actions=int(max_actions),
        allow_load_shed=True,
        allow_voltage_control=True,
    )
    if not plan.actions:
        n_v = len(session.last_result.voltage_violations) if session.last_result is not None else 0
        n_t = len(session.last_result.thermal_violations) if session.last_result is not None else 0
        if T is EN:
            hint = (
                "No remedial action generated because there are no active violations under current thresholds. "
                f"(V violations={n_v}, thermal violations={n_t}, "
                f"V range=[{ctx.solver_config.v_min:.2f}, {ctx.solver_config.v_max:.2f}], "
                f"max loading={ctx.solver_config.max_loading:.0f}%). "
                "Try stricter limits (e.g., Vmax=1.05)."
            )
        else:
            hint = (
                "当前阈值下没有越限，因此未生成缓解建议。"
                f"（电压越限={n_v}，热越限={n_t}，"
                f"电压范围=[{ctx.solver_config.v_min:.2f}, {ctx.solver_config.v_max:.2f}]，"
                f"最大负载率={ctx.solver_config.max_loading:.0f}%）。"
                "可尝试收紧阈值（如 Vmax=1.05）后重试。"
            )
        _append_ui_message("assistant", hint, remedial_plan=plan.model_dump())
        st.rerun()
        return
    session.last_remedial_plan = plan
    fig = make_remedial_ranking(
        plan,
        theme=_get_plot_theme(),
        lang=st.session_state.get("ui_lang", "en"),
    )

    extra_plots = []
    if plan.actions and plan.actions[0].preview_result is not None:
        positions = _ensure_positions()
        cmp_fig = make_comparison(
            ctx.net,
            session.last_result,
            plan.actions[0].preview_result,
            positions=positions,
            theme=_get_plot_theme(),
            lang=st.session_state.get("ui_lang", "en"),
        )
        extra_plots.append(
            {
                "plot_type": "comparison",
                "figure_json": pio.to_json(cmp_fig, validate=False),
                "title": ("Best Remedial Before/After" if T is EN else "最佳建议前后对比"),
            }
        )
        # 4-panel quantitative comparison
        try:
            qc_fig = make_quantitative_comparison(
                session.last_result,
                plan.actions[0].preview_result,
                vmin=ctx.solver_config.v_min,
                vmax=ctx.solver_config.v_max,
                max_loading=ctx.solver_config.max_loading,
                lang=st.session_state.get("ui_lang", "en"),
            )
            extra_plots.append(
                {
                    "plot_type": "quantitative_comparison",
                    "figure_json": pio.to_json(qc_fig, validate=False),
                    "title": ("Quantitative Comparison" if T is EN else "量化对比视图"),
                    "before_result": session.last_result.model_dump(),
                    "after_result": plan.actions[0].preview_result.model_dump(),
                }
            )
        except Exception:
            pass

    _append_ui_message(
        "assistant",
        ("🛠️ Remedial actions generated." if T is EN else "🛠️ 已生成缓解建议。"),
        plot_json=pio.to_json(fig, validate=False),
        remedial_plan=plan.model_dump(),
        extra_plots=extra_plots,
    )
    st.rerun()


def _queue_remedial_apply(action_index_0b: int, source: str = "chat") -> None:
    """Queue a remedial apply action and ask for confirmation."""

    st.session_state.pending_remedial_apply = {"index": int(action_index_0b), "source": str(source)}
    st.rerun()


def _apply_remedial_action_ui(action_index_0b: int, T: Dict[str, str]) -> None:
    """Apply one remedial action in UI and update modification history."""

    ctx: ToolContext = st.session_state.tool_ctx
    session: SessionState = st.session_state.session
    if str(st.session_state.get("solver_backend", "pandapower")) == "llm_only":
        _append_ui_message("assistant", f"❌ {_llm_only_disabled_message()}")
        return

    if ctx.net is None or session.last_result is None:
        _append_ui_message("assistant", T["need_case"])
        return

    plan = session.last_remedial_plan
    if plan is None or not plan.actions:
        _append_ui_message("assistant", f"❌ {T['remedial_none']}")
        return

    if action_index_0b < 0 or action_index_0b >= len(plan.actions):
        _append_ui_message("assistant", f"❌ action_index out of range: 0..{len(plan.actions)-1}")
        return

    act = plan.actions[action_index_0b]
    before = session.last_result

    # 写入当前阈值配置
    ctx.solver_config = SolverConfig(
        v_min=float(st.session_state.get("v_min", DEFAULT_V_MIN)),
        v_max=float(st.session_state.get("v_max", DEFAULT_V_MAX)),
        max_loading=float(st.session_state.get("max_loading", DEFAULT_MAX_LOADING)),
    )
    ctx.theme = _get_plot_theme()

    # Push undo snapshot BEFORE modifying the network so we can rollback on failure.
    _push_snapshot(label=f"pre_remedial:{action_index_0b + 1}")

    try:
        after = apply_remedial_action_inplace(ctx.net, act, config=ctx.solver_config)
    except Exception as e:
        _undo_last()  # rollback the network to pre-modification state
        _append_ui_message("assistant", T["remedial_apply_failed"].format(err=f"{type(e).__name__}: {e}"))
        return

    # Update session state.
    session.last_result = after
    session.modification_log.append(
        Modification(
            action="apply_remedial_action",
            description=f"Apply remedial action #{action_index_0b+1}: {act.description}",
            parameters={
                "action_index": action_index_0b + 1,
                "action": act.action,
                **(act.parameters or {}),
            },
        )
    )
    # Plan is based on old state; clear it after applying to avoid misuse.
    session.last_remedial_plan = None

    # Auto-generate before/after comparison figure.
    positions = _ensure_positions()
    try:
        cmp_fig = make_comparison(
            ctx.net,
            before,
            after,
            positions=positions,
            theme=_get_plot_theme(),
            lang=st.session_state.get("ui_lang", "en"),
        )
        cmp_json = pio.to_json(cmp_fig, validate=False)
    except Exception:
        cmp_json = None

    # 4-panel quantitative comparison
    qc_extra: list[Dict[str, Any]] = []
    try:
        qc_fig = make_quantitative_comparison(
            before,
            after,
            vmin=ctx.solver_config.v_min,
            vmax=ctx.solver_config.v_max,
            max_loading=ctx.solver_config.max_loading,
            lang=st.session_state.get("ui_lang", "en"),
        )
        qc_extra.append(
            {
                "plot_type": "quantitative_comparison",
                "figure_json": pio.to_json(qc_fig, validate=False),
                "title": ("Quantitative Comparison" if T is EN else "量化对比视图"),
                "before_result": before.model_dump(),
                "after_result": after.model_dump(),
            }
        )
    except Exception:
        pass

    text = T["remedial_applied"].format(idx=action_index_0b + 1, desc=act.description)
    if not after.converged:
        text += (
            "\n\n⚠️ Power flow did not converge after applying this action. Consider undoing and trying another action."
            if T is EN
            else "\n\n⚠️ 应用后潮流未收敛。建议撤销并尝试其他动作。"
        )
    text += "\n\n" + T["remedial_stale"]

    _append_ui_message(
        "assistant",
        text,
        plot_json=cmp_json,
        result=after.model_dump(),
        extra_plots=qc_extra if qc_extra else None,
    )

    # 记录快照（用于撤销）
    _push_snapshot(label=f"mod:{len(session.modification_log)}")


def _render_remedial_confirm_dialog(T: Dict[str, str]) -> None:
    """Render confirmation dialog for pending remedial action."""

    pending = st.session_state.get("pending_remedial_apply")
    if not isinstance(pending, dict):
        return

    idx = int(pending.get("index", -1))
    session: SessionState = st.session_state.session
    plan = session.last_remedial_plan
    if plan is None or not plan.actions or idx < 0 or idx >= len(plan.actions):
        st.session_state.pending_remedial_apply = None
        return

    act = plan.actions[idx]

    if hasattr(st, "dialog"):

        @st.dialog(T["remedial_confirm_title"])
        def _dlg():
            st.markdown(f"**#{idx+1}** {act.description}")
            st.json({"action": act.action, "parameters": act.parameters, "risk_reduction": act.risk_reduction})
            st.warning(T["remedial_confirm_warn"])
            c1, c2 = st.columns(2)
            with c1:
                if st.button(T["remedial_cancel"], key="modal_remedial_cancel", use_container_width=True):
                    st.session_state.pending_remedial_apply = None
                    st.rerun()
            with c2:
                if st.button(T["remedial_confirm"], key="modal_remedial_confirm", use_container_width=True):
                    st.session_state.pending_remedial_apply = None
                    _apply_remedial_action_ui(idx, T)
                    st.rerun()

        _dlg()
    else:
        # fallback inline
        st.info(T["remedial_confirm_warn"])
        if st.button(T["remedial_confirm"], key="confirm_apply_inline"):
            st.session_state.pending_remedial_apply = None
            _apply_remedial_action_ui(idx, T)
            st.rerun()
        if st.button(T["remedial_cancel"], key="cancel_apply_inline"):
            st.session_state.pending_remedial_apply = None
            st.rerun()


# -----------------------------
# Main app
# -----------------------------


def main() -> None:
    st.set_page_config(page_title=EN["title"], layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Source+Sans+3:wght@400;600;700&display=swap');
        :root {
            --primary-color: #facc15;
        }
        .stButton > button[kind="primary"] {
            background-color: #facc15 !important;
            border-color: #eab308 !important;
            color: #111827 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #eab308 !important;
            border-color: #ca8a04 !important;
            color: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _init_state()

    # Language toggle controls UI; LLM reply language still follows user input.
    with st.sidebar:
        en = st.toggle("English", value=(st.session_state.ui_lang == "en"))
        st.session_state.ui_lang = "en" if en else "zh"
    T = EN if st.session_state.ui_lang == "en" else ZH
    llm_only_active = str(st.session_state.get("solver_backend", "pandapower")) == "llm_only"

    st.title(T["title"])

    # Show pending remedial confirmation dialog first.
    _render_remedial_confirm_dialog(T)

    # Sidebar controls
    case_map = {
        "IEEE 14-bus": "case14",
        "IEEE 30-bus": "case30",
        "IEEE 57-bus": "case57",
        "IEEE 118-bus": "case118",
        "IEEE 300-bus": "case300",
    }
    rev_case_map = {v: k for k, v in case_map.items()}

    with st.sidebar:
        st.markdown(f"### {T['sidebar_case']}")
        current_case = st.session_state.session.active_case
        default_label = rev_case_map.get(current_case, "IEEE 14-bus")
        selected_label = st.selectbox(
            T["sidebar_case"],
            list(case_map.keys()),
            index=list(case_map.keys()).index(default_label),
            key="selected_case_label",
        )
        selected_case = case_map[selected_label]
        if st.button(T["sidebar_load"], key="sidebar_load_case", use_container_width=True, type="primary"):
            _load_case_direct(selected_case, T)

        st.divider()
        st.markdown(f"### {T['sidebar_settings']}")
        st.slider(T["sidebar_vmin"], 0.90, 0.98, float(DEFAULT_V_MIN), 0.01, key="v_min")
        st.slider(T["sidebar_vmax"], 1.02, 1.10, float(DEFAULT_V_MAX), 0.01, key="v_max")
        st.slider(T["sidebar_loading"], 80, 120, int(DEFAULT_MAX_LOADING), 5, key="max_loading")
        backend_options = [("pandapower", T["backend_pp"]), ("llm_only", T["backend_llm"])]
        backend_labels = [x[1] for x in backend_options]
        current_backend = str(st.session_state.get("solver_backend", "pandapower"))
        backend_idx = 0 if current_backend == "pandapower" else 1
        selected_backend_label = st.selectbox(
            T["sidebar_backend"],
            backend_labels,
            index=backend_idx,
            key="solver_backend_label",
        )
        selected_backend = backend_options[backend_labels.index(selected_backend_label)][0]
        st.session_state.solver_backend = selected_backend
        llm_only_active = selected_backend == "llm_only"
        st.caption(f"{T['backend_active']}: {selected_backend_label}")

        current_n_bus = 0
        try:
            if st.session_state.tool_ctx.net is not None and hasattr(st.session_state.tool_ctx.net, "bus"):
                current_n_bus = int(len(st.session_state.tool_ctx.net.bus))
        except Exception:
            current_n_bus = 0
        active_case = str(st.session_state.session.active_case or "")
        autoset_case = str(st.session_state.get("flow_particles_autoset_case", ""))
        if autoset_case != active_case:
            st.session_state.flow_particles_autoset_case = active_case

        with st.expander("Flow Diagram", expanded=False):
            st.slider(
                "Heavy threshold (%)" if T is EN else "重载阈值 (%)",
                min_value=20,
                max_value=120,
                value=int(st.session_state.get("flow_heavy_threshold", 60)),
                step=1,
                key="flow_heavy_threshold",
            )
            st.slider(
                "Overloaded threshold (%)" if T is EN else "过载阈值 (%)",
                min_value=30,
                max_value=180,
                value=int(st.session_state.get("flow_over_threshold", 100)),
                step=1,
                key="flow_over_threshold",
            )
            st.toggle("Show power labels" if T is EN else "显示功率标签", key="flow_show_labels")
            st.toggle("Show flow arrows" if T is EN else "显示流向箭头", key="flow_show_arrows")
            st.toggle("Voltage overlay" if T is EN else "电压热力叠加", key="flow_show_vm_overlay")
            st.toggle("Particle overlay" if T is EN else "粒子叠加动画", key="flow_show_particles")
            st.caption(
                "Canvas particle overlay is enabled in Phase 2 (for bus > 200, only branches with |P| >= 50 MW are rendered)."
                if T is EN
                else "Phase 2：已启用 Canvas 粒子叠加动画（bus > 200 仅渲染 |P| >= 50 MW 的支路）。"
            )
            st.selectbox(
                "Color scheme" if T is EN else "配色方案",
                options=["light", "dark", "print"],
                key="flow_color_scheme",
                format_func=lambda x: {"light": "Light", "dark": "Dark", "print": "Print-friendly"}.get(x, x),
            )
            st.caption(
                "CSV columns: from_bus,to_bus,p_from_mw[,q_from_mvar,loading_percent]"
                if T is EN
                else "CSV 列: from_bus,to_bus,p_from_mw[,q_from_mvar,loading_percent]"
            )
            uploaded_flow_csv = st.file_uploader(
                "Upload custom branch flow CSV" if T is EN else "上传自定义支路潮流 CSV",
                type=["csv"],
                key="flow_csv_upload",
            )
            if uploaded_flow_csv is not None:
                try:
                    df = pd.read_csv(uploaded_flow_csv)
                    required = {"from_bus", "to_bus", "p_from_mw"}
                    if not required.issubset(set(df.columns)):
                        st.error(
                            "CSV missing required columns."
                            if T is EN
                            else "CSV 缺少必要列（from_bus,to_bus,p_from_mw）。"
                        )
                    else:
                        st.session_state.flow_branch_override_df = df.copy()
                        st.success("Custom branch flow loaded." if T is EN else "已加载自定义支路潮流数据。")
                except Exception as e:
                    st.error(
                        f"Failed to parse CSV: {type(e).__name__}: {e}"
                        if T is EN
                        else f"CSV 解析失败：{type(e).__name__}: {e}"
                    )
            if st.button("Clear custom flow CSV" if T is EN else "清除自定义潮流 CSV", use_container_width=True):
                st.session_state.flow_branch_override_df = None

        st.divider()
        st.markdown("### LLM")
        previous_provider = str(st.session_state.get("llm_provider", "openai"))
        provider_options = list(LLM_PROVIDER_LABELS.keys())
        if previous_provider not in provider_options:
            previous_provider = "openai"
        provider_index = provider_options.index(previous_provider)
        selected_provider = st.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            format_func=lambda p: LLM_PROVIDER_LABELS.get(p, p),
            key="llm_provider",
        )
        if selected_provider != previous_provider:
            remember = st.session_state.get("llm_last_model_by_provider", {})
            if not isinstance(remember, dict):
                remember = {}
            st.session_state.llm_model = str(remember.get(selected_provider) or _provider_default_model(selected_provider))
            provider_default_key = _provider_default_key(selected_provider)
            if provider_default_key:
                st.session_state.llm_api_key = provider_default_key

        st.text_input(
            "API Key",
            type="password",
            key="llm_api_key",
            placeholder="Enter API key for selected provider",
            help="OpenAI: sk-... | Gemini: AIza... (or set env var)",
        )
        _persist_insecure_api_key(
            selected_provider,
            str(st.session_state.get("llm_api_key", "") or ""),
            str(st.session_state.get("llm_model", "") or ""),
        )
        if selected_provider == "gemini":
            api_key_for_list = str(st.session_state.get("llm_api_key", "") or "").strip()
            model_options, model_list_error = _list_gemini_models_dynamic(api_key_for_list)
            if model_list_error:
                st.caption(f"Model list fallback: {model_list_error}")
        else:
            model_options = _models_for_provider(selected_provider)

        current_model = str(st.session_state.get("llm_model", _provider_default_model(selected_provider)))
        if current_model and current_model not in model_options:
            # Preserve explicit user choice across reruns/load-case even if
            # dynamic model listing does not include it at this moment.
            model_options = [current_model] + [m for m in model_options if m != current_model]
        elif not current_model:
            current_model = _provider_default_model(selected_provider)
            if current_model not in model_options and model_options:
                current_model = model_options[0]
            st.session_state.llm_model = current_model

        model_index = model_options.index(current_model) if current_model in model_options else 0
        st.selectbox(
            "Model",
            model_options,
            index=model_index,
            key="llm_model",
            format_func=lambda m: _model_label(selected_provider, m),
        )
        _persist_insecure_api_key(
            selected_provider,
            str(st.session_state.get("llm_api_key", "") or ""),
            str(st.session_state.get("llm_model", "") or ""),
        )
        remember = st.session_state.get("llm_last_model_by_provider", {})
        if not isinstance(remember, dict):
            remember = {}
        remember[selected_provider] = str(st.session_state.get("llm_model", current_model))
        st.session_state.llm_last_model_by_provider = remember
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.get("llm_temperature", OPENAI_TEMPERATURE)),
            step=0.05,
            key="llm_temperature",
        )
        st.number_input(
            "Timeout (s)",
            min_value=5.0,
            max_value=300.0,
            value=float(st.session_state.get("llm_timeout_s", OPENAI_TIMEOUT_S)),
            step=5.0,
            key="llm_timeout_s",
        )
        st.toggle(
            "LLM-only Debug Mode (blueprint)",
            value=bool(st.session_state.get("llm_only_debug_mode", LLM_ONLY_DEBUG_MODE)),
            key="llm_only_debug_mode",
            help="Enable debug_routing_step output with top-30 loads and top-50 branches.",
        )
        st.text_input(
            "MATPOWER data root",
            key="matpower_data_root",
            value=str(st.session_state.get("matpower_data_root", MATPOWER_DATA_ROOT)),
            help="Default: data/matpower",
        )
        st.text_input(
            "MATPOWER case date",
            key="matpower_case_date",
            value=str(st.session_state.get("matpower_case_date", MATPOWER_CASE_DATE)),
            help="Default: 2017-01-01",
        )
        with st.expander(T["external_import_title"], expanded=False):
            raw_external = st.text_area(
                T["external_import_input"],
                key="external_llm_result_raw",
                height=180,
                placeholder='{"converged": true, "bus_voltages": [...], "line_flows": [...], "totals": {...}}',
            )
            if st.button(T["external_import_btn"], key="sidebar_import_external_llm_result", use_container_width=True):
                _import_external_llm_result(raw_external, T)

            has_llm_result = (
                st.session_state.session.last_result is not None
                and "llm" in str(getattr(st.session_state.session.last_result, "solver_backend", "")).lower()
            )
            if st.button(T["benchmark_btn"], key="sidebar_benchmark_vs_truth", use_container_width=True, disabled=not has_llm_result):
                _benchmark_external_vs_truth(T)

            cur_res = st.session_state.session.last_result
            if cur_res is not None:
                case_name = str(getattr(cur_res, "case_name", "") or st.session_state.session.active_case or "case")
                export_name = f"pf_result_{case_name}.json"
                export_json = json.dumps(cur_res.model_dump(), ensure_ascii=False, indent=2)
                st.download_button(
                    "Export current PF JSON" if T is EN else "导出当前 PF 结果 JSON",
                    data=export_json,
                    file_name=export_name,
                    mime="application/json",
                    key="sidebar_export_pf_json",
                    use_container_width=True,
                )

        st.divider()
        st.markdown(f"### {T['sidebar_n1']}")
        st.slider(T["sidebar_n1_topk"], 1, 20, 5, 1, key="n1_topk")
        if st.button(T["sidebar_n1_go"], key="sidebar_run_n1", disabled=llm_only_active, use_container_width=True):
            # N-1 can run without API key (direct tool path).
            _run_n1_direct(T, top_k=int(st.session_state.get("n1_topk", 5) or 5))

        st.divider()
        st.markdown(f"### {T['sidebar_remedial']}")
        if st.button(T["btn_remedial"], key="sidebar_generate_remedial", disabled=llm_only_active, use_container_width=True):
            _run_remedial_direct(T, max_actions=5)

        # If a recent plan exists, show one-click apply actions.
        if st.session_state.session.last_remedial_plan is not None and st.session_state.session.last_remedial_plan.actions:
            plan = st.session_state.session.last_remedial_plan
            st.caption(plan.summary_text or "")
            for i, a in enumerate(plan.actions[:10]):
                c1, c2 = st.columns([0.78, 0.22])
                with c1:
                    st.write(f"#{i+1} {a.description}")
                with c2:
                    if st.button(
                        T["remedial_apply"],
                        key=f"sidebar_apply_remedial_{i}",
                        disabled=llm_only_active,
                        use_container_width=True,
                    ):
                        _queue_remedial_apply(i, source="sidebar")

        st.divider()
        st.markdown(f"### {T['sidebar_history']}")
        if st.session_state.session.modification_log:
            st.dataframe(
                [m.model_dump() for m in st.session_state.session.modification_log],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("(empty)")

        undo_disabled = len(st.session_state.net_history) <= 1
        if st.button(T["sidebar_undo"], key="sidebar_undo_last", disabled=undo_disabled, use_container_width=True):
            _undo_last()
            _append_ui_message("assistant", T["undo_done"])
            st.rerun()

        if st.button(T["sidebar_clear"], key="sidebar_clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session.conversation_history = []
            st.session_state.conversation_started = False
            st.rerun()

        provider, api_key, _ = _resolve_llm_settings()
        if not api_key:
            st.warning(_no_api_message(T, provider))

        st.divider()
        st.markdown("### Quick (no-LLM)")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Run PF", key="sidebar_quick_run_pf", use_container_width=True, type="primary"):
                _run_pf_direct(T)
        with cols[1]:
            if st.button("Voltage", key="sidebar_quick_voltage", use_container_width=True):
                _plot_direct("voltage_heatmap", T)

        cols2 = st.columns(2)
        with cols2[0]:
            if st.button("N-1", key="sidebar_quick_n1", disabled=llm_only_active, use_container_width=True):
                _run_n1_direct(T, top_k=int(st.session_state.get("n1_topk", 5) or 5))
        with cols2[1]:
            if st.button("Remedial", key="sidebar_quick_remedial", disabled=llm_only_active, use_container_width=True):
                _run_remedial_direct(T, max_actions=5)

        if st.button("All Plots", key="sidebar_quick_all_plots", use_container_width=True):
            _plot_all_direct(T)

    _render_runtime_status_bar(T)

    # Welcome card
    if not st.session_state.conversation_started and not st.session_state.messages:
        st.markdown(T["welcome_h"])
        st.markdown(T["welcome_t"])
        cols = st.columns(3)
        with cols[0]:
            if st.button(T["btn_run14"], key="welcome_run_14_pf", use_container_width=True):
                process_user_input(
                    "Run IEEE 14 power flow and summarize key results."
                    if T is EN
                    else "运行 IEEE 14 节点潮流计算，并输出关键结果。",
                    T,
                )
        with cols[1]:
            if st.button(T["btn_v30"], key="welcome_ieee30_voltage", use_container_width=True):
                process_user_input(
                    "Load IEEE 30, run power flow, and show voltage map."
                    if T is EN
                    else "加载 IEEE 30 节点系统，运行潮流并显示电压图。",
                    T,
                )
        with cols[2]:
            if st.button(T["btn_57"], key="welcome_ieee57_what_if", use_container_width=True):
                process_user_input(
                    "Load IEEE 57, run power flow, do one line-outage what-if, and show comparison."
                    if T is EN
                    else "加载 IEEE 57，运行潮流，做一次断线 what-if 并输出对比图。",
                    T,
                )

    # Chat history render
    for mi, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m.get("content") or "")
            fetch_cmd = _extract_fetch_command(m.get("content") or "")
            if fetch_cmd:
                st.error(
                    "MATPOWER case file is missing for LLM-only blueprint mode."
                    if T is EN
                    else "LLM-only 蓝图模式缺少 MATPOWER 案例文件。"
                )
                st.code(fetch_cmd, language="bash")

            if m.get("plot_json"):
                try:
                    if m.get("plot_type") == "flow_diagram" and isinstance(m.get("plot_html"), str):
                        components.html(m["plot_html"], height=720, scrolling=False)
                    elif m.get("plot_type") == "comparison":
                        fig = pio.from_json(m["plot_json"])
                        panels = _split_comparison_figure(fig, T)
                        for pi, pf in enumerate(panels):
                            st.plotly_chart(
                                pf,
                                use_container_width=True,
                                key=f"chat_plot_{mi}_cmp_{pi}",
                                config={"displaylogo": False},
                            )
                    else:
                        fig = pio.from_json(m["plot_json"])
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"chat_plot_{mi}",
                            config={"displaylogo": False},
                        )
                except Exception as e:
                    st.error(f"Plot render error: {type(e).__name__}: {e}")

            # Support rendering multiple figures in one message.
            if m.get("extra_plots"):
                for epi, ep in enumerate(m["extra_plots"]):
                    try:
                        if ep.get("title"):
                            st.markdown(f"**{ep['title']}**")
                        if ep.get("plot_type") == "flow_diagram" and isinstance(ep.get("html"), str):
                            components.html(ep["html"], height=720, scrolling=False)
                        elif ep.get("plot_type") == "comparison":
                            fig2 = pio.from_json(ep["figure_json"])
                            panels2 = _split_comparison_figure(fig2, T)
                            for pi, pf2 in enumerate(panels2):
                                st.plotly_chart(
                                    pf2,
                                    use_container_width=True,
                                    key=f"chat_extra_plot_{mi}_{epi}_cmp_{pi}",
                                    config={"displaylogo": False},
                                )
                        elif ep.get("plot_type") == "quantitative_comparison":
                            # White card container for quantitative comparison
                            with st.container():
                                st.markdown(
                                    "<div style='background:#fff; border-radius:12px; padding:8px 4px; "
                                    "box-shadow:0 1px 3px rgba(0,0,0,0.08);'>",
                                    unsafe_allow_html=True,
                                )
                                fig2 = pio.from_json(ep["figure_json"])
                                st.plotly_chart(
                                    fig2,
                                    use_container_width=True,
                                    key=f"chat_extra_plot_{mi}_{epi}_qc",
                                    config={"displaylogo": False},
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                            # Summary metrics row
                            _render_qc_summary_row(ep, mi, epi, T)
                        elif ep.get("plot_type") == "benchmark":
                            with st.container():
                                st.markdown(
                                    "<div style='background:#fff; border-radius:12px; padding:8px 4px; "
                                    "box-shadow:0 1px 3px rgba(0,0,0,0.08);'>",
                                    unsafe_allow_html=True,
                                )
                                fig2 = pio.from_json(ep["figure_json"])
                                st.plotly_chart(
                                    fig2,
                                    use_container_width=True,
                                    key=f"chat_extra_plot_{mi}_{epi}_bench",
                                    config={"displaylogo": False},
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                            _render_benchmark_cards(ep, mi, epi, T)
                        else:
                            fig2 = pio.from_json(ep["figure_json"])
                            st.plotly_chart(
                                fig2,
                                use_container_width=True,
                                key=f"chat_extra_plot_{mi}_{epi}",
                                config={"displaylogo": False},
                            )
                    except Exception as e:
                        st.error(f"Extra plot render error: {type(e).__name__}: {e}")

            if m["role"] == "assistant" and m.get("n1_report"):
                with st.expander("N-1 Top-K Scenarios", expanded=True):
                    rep = m["n1_report"]
                    st.markdown(rep.get("summary_text") or "")
                    st.dataframe(rep.get("results") or [], use_container_width=True)

            if m["role"] == "assistant" and m.get("remedial_plan"):
                with st.expander("Remedial actions", expanded=True):
                    rp = m["remedial_plan"]
                    st.markdown(rp.get("summary_text") or "")
                    st.write({"base_risk": rp.get("base_risk"), "case": rp.get("case_name")})
                    actions = rp.get("actions") or []
                    if actions:
                        # Only keep core fields to avoid an overly long table.
                        table = [
                            {
                                "action": a.get("action"),
                                "description": a.get("description"),
                                "risk_reduction": a.get("risk_reduction"),
                                "predicted_risk": a.get("predicted_risk"),
                                "parameters": a.get("parameters"),
                            }
                            for a in actions
                        ]
                        st.dataframe(table, use_container_width=True)

                        # Use only latest plan to avoid applying stale suggestions.
                        latest_plan = st.session_state.session.last_remedial_plan
                        same_plan = (
                            latest_plan is not None
                            and str(getattr(latest_plan, "case_name", "")) == str(rp.get("case_name"))
                            and float(getattr(latest_plan, "base_risk", -1.0)) == float(rp.get("base_risk") or -1.0)
                        )
                        if same_plan and getattr(latest_plan, "actions", None):
                            st.divider()
                            st.markdown("**One-click apply (modifies network)**")
                            for i, a in enumerate(latest_plan.actions[: len(actions)]):
                                cols = st.columns([0.84, 0.16])
                                with cols[0]:
                                    st.write(f"#{i+1} {a.description}")
                                with cols[1]:
                                    if st.button(
                                        T["remedial_apply"],
                                        key=f"chat_apply_remedial_{mi}_{i}",
                                        disabled=llm_only_active,
                                    ):
                                        _queue_remedial_apply(i, source="chat")

            if m["role"] == "assistant" and m.get("result"):
                _render_result_details(m["result"], T)

    # Quick buttons row
    btn_cols = st.columns(8)
    with btn_cols[0]:
        if st.button(T["btn_run"], key="quick_run_pf", use_container_width=True, type="primary"):
            _run_pf_direct(T)
    with btn_cols[1]:
        if st.button(T["btn_v"], key="quick_voltage_map", use_container_width=True):
            _plot_direct("voltage_heatmap", T)
    with btn_cols[2]:
        if st.button(T["btn_violation"], key="quick_violation_overview", use_container_width=True):
            _plot_direct("violation_overview", T)
    with btn_cols[3]:
        with st.popover(T["btn_disconnect"], use_container_width=True):
            fb = st.number_input(T["disconnect_fb"], min_value=0, value=1, step=1)
            tb = st.number_input(T["disconnect_tb"], min_value=0, value=2, step=1)
            if st.button(T["disconnect_go"], key="quick_disconnect_go", use_container_width=True):
                process_user_input(
                    f"Disconnect line from bus {int(fb)} to bus {int(tb)} and re-analyze."
                    if T is EN
                    else f"断开 bus {int(fb)} 到 bus {int(tb)} 的线路并重新分析。",
                    T,
                )
    with btn_cols[4]:
        if st.button(T["btn_n1"], key="quick_run_n1", disabled=llm_only_active, use_container_width=True):
            topk = int(st.session_state.get("n1_topk", 5) or 5)
            process_user_input(
                f"Run N-1 analysis and return top {topk} severe scenarios."
                if T is EN
                else f"执行 N-1 分析并输出前 {topk} 个最严重场景。",
                T,
            )
    with btn_cols[5]:
        if st.button(T["btn_remedial"], key="quick_generate_remedial", disabled=llm_only_active, use_container_width=True):
            _run_remedial_direct(T, max_actions=5)
    with btn_cols[6]:
        if st.button("All Plots", key="quick_all_plots", use_container_width=True):
            _plot_all_direct(T)
    with btn_cols[7]:
        if st.button(T["btn_report"], key="quick_export_report", use_container_width=True):
            process_user_input("Export current analysis report." if T is EN else "导出当前分析报告。", T)

    _render_bottom_stats(T)

    # Chat input
    user_text = st.chat_input(T["chat_placeholder"])
    if user_text:
        process_user_input(user_text, T)


if __name__ == "__main__":
    main()

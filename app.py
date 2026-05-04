import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
MANIFEST_PATH = PROJECT_ROOT / "chroma_db" / "manifest.json"
DEFAULT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://127.0.0.1:8000/api/v1")
KNOWLEDGE_FILE_TYPES = ["txt", "pdf", "doc", "docx", "xlsx", "xls", "pptx", "csv"]


st.set_page_config(
    page_title="扫地机器人智能客服",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f7f8fb;
            --surface: #ffffff;
            --surface-soft: #f2f4f7;
            --border: #d8dee8;
            --text-main: #1d2939;
            --text-muted: #667085;
            --primary: #2563eb;
            --primary-soft: #e9f0ff;
            --success: #059669;
            --warning: #b7791f;
            --danger: #dc2626;
        }

        .stApp {
            background: var(--page-bg);
            color: var(--text-main);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.3rem;
            padding-bottom: 5rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--text-main);
        }

        .topbar {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .topbar h1 {
            margin: 0;
            font-size: 1.55rem;
            line-height: 1.25;
        }

        .topbar-subtitle {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 0.35rem;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            background: var(--surface-soft);
            color: var(--text-main);
            font-size: 0.8rem;
            line-height: 1.2;
            white-space: nowrap;
        }

        .pill strong {
            color: var(--primary);
            margin-right: 0.25rem;
        }

        .panel {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            color: var(--text-main);
        }

        .muted {
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .trace-line {
            border-left: 3px solid var(--primary);
            background: #f8fafc;
            padding: 0.55rem 0.7rem;
            margin: 0.45rem 0;
            border-radius: 6px;
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--text-main);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.35rem 0.8rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--primary-soft);
            border-color: #bcd0ff;
            color: var(--primary);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #ffffff;
            color: var(--text-main);
            font-weight: 600;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--primary);
            color: var(--primary);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
            background: #ffffff;
        }

        div[data-testid="stMetricValue"] {
            color: var(--primary);
            font-size: 1.35rem;
        }

        .stChatMessage {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #ffffff;
        }

        [data-testid="stChatMessageContent"] {
            line-height: 1.75;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .topbar {
                padding: 0.9rem;
            }

            .topbar h1 {
                font-size: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    defaults = {
        "message": [],
        "conversation_id": None,
        "api_base_url": DEFAULT_API_BASE_URL,
        "last_events": [],
        "last_answer": "",
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_base_url() -> str:
    return st.session_state["api_base_url"].rstrip("/")


def api_get(path: str, **params) -> dict[str, Any]:
    response = requests.get(f"{api_base_url()}{path}", params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(f"{api_base_url()}{path}", json=payload or {}, timeout=60)
    response.raise_for_status()
    return response.json()


def normalize_tool_items(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, dict):
        normalized: list[dict[str, Any]] = []
        for name, metadata in items.items():
            if isinstance(metadata, dict):
                normalized.append({"name": name, **metadata})
            else:
                normalized.append({"name": name, "description": str(metadata)})
        return normalized
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"files": {}}


def list_knowledge_files() -> list[dict[str, Any]]:
    manifest = load_manifest()
    records = manifest.get("files", {})
    files: list[dict[str, Any]] = []
    DATA_DIR.mkdir(exist_ok=True)
    for path in sorted(DATA_DIR.glob("*")):
        if not path.is_file() or path.suffix.lower().lstrip(".") not in KNOWLEDGE_FILE_TYPES:
            continue
        record = records.get(path.name, {})
        files.append(
            {
                "file": path.name,
                "type": path.suffix.lower().lstrip("."),
                "size_kb": round(path.stat().st_size / 1024, 2),
                "chunks": record.get("chunk_count", ""),
                "synced": "yes" if record else "no",
            }
        )
    return files


def load_eval_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def eval_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0, "hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "mrr": 0}
    count = len(rows)
    return {
        "samples": count,
        "hit_at_1": sum(int(row.get("hit_at_1") or 0) for row in rows) / count,
        "hit_at_3": sum(int(row.get("hit_at_3") or 0) for row in rows) / count,
        "hit_at_5": sum(int(row.get("hit_at_5") or 0) for row in rows) / count,
        "mrr": sum(float(row.get("mrr") or 0) for row in rows) / count,
    }


def render_topbar() -> None:
    status = "未连接"
    version = "-"
    try:
        health = api_get("/health")
        status = "正常" if health.get("status") == "ok" else str(health.get("status"))
        version = str(health.get("version", "-"))
    except requests.RequestException:
        status = "离线"

    st.markdown(
        f"""
        <section class="topbar">
            <h1>扫地机器人智能客服 Agent</h1>
            <div class="topbar-subtitle">
                垂直领域 RAG、工具调用、MCP 天气查询、会话持久化和检索评估集成工作台。
            </div>
            <div class="status-row">
                <span class="pill"><strong>API</strong>{status}</span>
                <span class="pill"><strong>版本</strong>{version}</span>
                <span class="pill"><strong>RAG</strong>Hybrid + BGE</span>
                <span class="pill"><strong>知识库</strong>{len(list_knowledge_files())} 文件</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("控制台")
        api_base = st.text_input(
            "API 地址",
            value=st.session_state["api_base_url"],
            help="FastAPI 服务地址",
        )
        st.session_state["api_base_url"] = api_base.rstrip("/")

        st.divider()
        user_count = sum(1 for item in st.session_state["message"] if item["role"] == "user")
        assistant_count = sum(
            1 for item in st.session_state["message"] if item["role"] == "assistant"
        )
        st.metric("用户提问", user_count)
        st.metric("客服回复", assistant_count)
        st.metric("本轮事件", len(st.session_state["last_events"]))

        if st.session_state["conversation_id"]:
            st.caption(f"会话 ID：{st.session_state['conversation_id']}")

        if st.button("清空当前会话", use_container_width=True):
            st.session_state["message"] = []
            st.session_state["conversation_id"] = None
            st.session_state["last_events"] = []
            st.session_state["last_answer"] = ""
            st.session_state["last_error"] = ""
            st.rerun()


def format_event(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "tool_call":
        return f"[工具调用] {event.get('tool_name', '')} 参数：{event.get('args', {})}"
    if event_type == "tool_result":
        return f"[工具结果] {event.get('tool_name', 'unknown_tool')}：{event.get('content', '')}"
    if event_type == "error":
        return f"[错误] {event.get('content', '')}"
    if event_type == "answer":
        return str(event.get("content", ""))
    return str(event.get("content") or event)


def yield_text(text: str):
    if text and not text.endswith("\n"):
        text += "\n"
    for char in text:
        time.sleep(0.006)
        yield char


def parse_sse_event(event_name: str, data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    payload = "\n".join(data_lines)
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        event = {"type": event_name or "message", "content": payload}
    if "type" not in event:
        event["type"] = event_name or "message"
    return event


def stream_agent_events(query: str, final_answer_chunks: list[str]):
    url = f"{api_base_url()}/chat/stream"
    payload = {
        "query": query,
        "conversation_id": st.session_state["conversation_id"],
        "include_events": True,
    }
    event_name = "message"
    data_lines: list[str] = []
    st.session_state["last_events"] = []
    st.session_state["last_error"] = ""

    try:
        with requests.post(url, json=payload, stream=True, timeout=(5, None)) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if line == "":
                    event = parse_sse_event(event_name, data_lines)
                    event_name = "message"
                    data_lines = []
                    if event is None:
                        continue
                    if event.get("type") == "metadata":
                        st.session_state["conversation_id"] = event.get("conversation_id")
                        continue
                    st.session_state["last_events"].append(event)
                    if event.get("type") == "answer":
                        final_answer_chunks.append(str(event.get("content", "")))
                    if event.get("type") == "error":
                        st.session_state["last_error"] = str(event.get("content", ""))
                    yield from yield_text(format_event(event))
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
    except requests.RequestException as e:
        error_message = f"无法连接后端服务：{e}"
        st.session_state["last_error"] = error_message
        final_answer_chunks.append(error_message)
        yield from yield_text(error_message)


def render_chat_tab() -> None:
    left, right = st.columns([0.68, 0.32], gap="large")
    with left:
        st.subheader("对话")
        if not st.session_state["message"]:
            st.markdown(
                '<div class="panel muted">当前会话暂无消息。</div>',
                unsafe_allow_html=True,
            )
        for message in st.session_state["message"]:
            st.chat_message(message["role"]).write(message["content"])

        prompt = st.chat_input("输入售后、选购、保养或故障问题")
        if prompt:
            st.chat_message("user").write(prompt)
            st.session_state["message"].append({"role": "user", "content": prompt})
            final_answer_chunks: list[str] = []
            with st.spinner("正在分析"):
                st.chat_message("assistant").write_stream(
                    stream_agent_events(prompt, final_answer_chunks)
                )
            assistant_response = "".join(final_answer_chunks).strip()
            if not assistant_response:
                assistant_response = "当前没有获取到模型最终回答。"
            st.session_state["last_answer"] = assistant_response
            st.session_state["message"].append(
                {"role": "assistant", "content": assistant_response}
            )
            st.rerun()

    with right:
        st.subheader("快捷问题")
        examples = [
            "扫地机器人吸力下降应该先检查哪里？",
            "机器人找不到充电座如何处理？",
            "小户型应该如何选择扫地机器人？",
            "地毯区域不想拖地应该怎么设置？",
        ]
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state["message"].append({"role": "user", "content": example})
                final_answer_chunks = []
                st.session_state["last_events"] = []
                with st.spinner("正在分析"):
                    for _ in stream_agent_events(example, final_answer_chunks):
                        pass
                assistant_response = "".join(final_answer_chunks).strip()
                st.session_state["last_answer"] = assistant_response
                st.session_state["message"].append(
                    {"role": "assistant", "content": assistant_response}
                )
                st.rerun()

        st.markdown('<div class="panel-title">最近工具轨迹</div>', unsafe_allow_html=True)
        if not st.session_state["last_events"]:
            st.markdown('<div class="muted">暂无事件。</div>', unsafe_allow_html=True)
        for event in st.session_state["last_events"][-8:]:
            st.markdown(
                f'<div class="trace-line">{format_event(event)}</div>',
                unsafe_allow_html=True,
            )


def render_knowledge_tab() -> None:
    st.subheader("知识库")
    files = list_knowledge_files()
    col1, col2, col3 = st.columns(3)
    col1.metric("文件数", len(files))
    col2.metric("已同步", sum(1 for item in files if item["synced"] == "yes"))
    col3.metric("切片数", sum(int(item["chunks"] or 0) for item in files))

    uploaded = st.file_uploader(
        "上传知识文件",
        type=KNOWLEDGE_FILE_TYPES,
        accept_multiple_files=True,
    )
    if uploaded and st.button("保存上传文件", use_container_width=True):
        DATA_DIR.mkdir(exist_ok=True)
        for file in uploaded:
            target = DATA_DIR / file.name
            target.write_bytes(file.getvalue())
        st.success("文件已保存到 data 目录。")
        st.rerun()

    action_col, _ = st.columns([0.3, 0.7])
    with action_col:
        if st.button("重建知识库", use_container_width=True):
            try:
                result = api_post("/knowledge/rebuild")
                st.success(result.get("message", "知识库重建完成。"))
            except requests.RequestException as e:
                st.error(f"重建失败：{e}")

    st.dataframe(files, use_container_width=True, hide_index=True)

    manifest = load_manifest()
    with st.expander("manifest"):
        st.json(manifest)


def render_tools_tab() -> None:
    st.subheader("工具")
    try:
        data = api_get("/tools")
        tools = normalize_tool_items(data.get("items", []))
    except requests.RequestException as e:
        st.error(f"工具列表获取失败：{e}")
        tools = []

    if tools:
        st.dataframe(tools, use_container_width=True, hide_index=True)

    names = [item.get("name", "") for item in tools if item.get("name")]
    selected = st.selectbox("工具名称", names) if names else ""
    raw_input = st.text_area("工具参数 JSON", value="{}", height=120)
    if st.button("调用工具", disabled=not selected):
        try:
            payload = json.loads(raw_input or "{}")
            result = api_post("/tools/call", {"name": selected, "tool_input": payload})
            st.json(result)
        except json.JSONDecodeError as e:
            st.error(f"JSON 格式错误：{e}")
        except requests.RequestException as e:
            st.error(f"调用失败：{e}")


def render_conversations_tab() -> None:
    st.subheader("会话记录")
    try:
        data = api_get("/conversations", limit=50)
        conversations = data.get("items", [])
    except requests.RequestException as e:
        st.error(f"会话列表获取失败：{e}")
        conversations = []

    if not conversations:
        st.markdown('<div class="panel muted">暂无持久化会话。</div>', unsafe_allow_html=True)
        return

    st.dataframe(conversations, use_container_width=True, hide_index=True)
    options = [item["conversation_id"] for item in conversations]
    selected = st.selectbox("查看会话", options)
    col1, col2 = st.columns(2)
    with col1:
        try:
            messages = api_get(f"/conversations/{selected}/messages").get("items", [])
            st.markdown('<div class="panel-title">消息</div>', unsafe_allow_html=True)
            for item in messages:
                st.chat_message(item["role"]).write(item["content"])
        except requests.RequestException as e:
            st.error(f"消息获取失败：{e}")
    with col2:
        try:
            logs = api_get(f"/conversations/{selected}/tool-logs").get("items", [])
            st.markdown('<div class="panel-title">工具日志</div>', unsafe_allow_html=True)
            st.dataframe(logs, use_container_width=True, hide_index=True)
        except requests.RequestException as e:
            st.error(f"工具日志获取失败：{e}")


def render_eval_tab() -> None:
    st.subheader("RAG 评估")
    csv_files = sorted(EVAL_DIR.glob("retrieval*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        st.markdown('<div class="panel muted">暂无评估结果文件。</div>', unsafe_allow_html=True)
        st.code("py eval/evaluate_retrieval.py --output eval/retrieval_bge.csv", language="powershell")
        return

    selected = st.selectbox("评估结果", [path.name for path in csv_files])
    rows = load_eval_rows(EVAL_DIR / selected)
    summary = eval_summary(rows)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("样本数", int(summary["samples"]))
    col2.metric("Hit@1", f"{summary['hit_at_1']:.0%}")
    col3.metric("Hit@3", f"{summary['hit_at_3']:.0%}")
    col4.metric("Hit@5", f"{summary['hit_at_5']:.0%}")
    col5.metric("MRR", f"{summary['mrr']:.4f}")

    failures = [row for row in rows if row.get("hit_at_1") == "0"]
    st.markdown('<div class="panel-title">明细</div>', unsafe_allow_html=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Hit@1 未命中样本"):
        st.dataframe(failures, use_container_width=True, hide_index=True)


def main() -> None:
    inject_style()
    init_session()
    render_sidebar()
    render_topbar()

    chat_tab, kb_tab, tools_tab, history_tab, eval_tab = st.tabs(
        ["对话", "知识库", "工具", "会话", "评估"]
    )
    with chat_tab:
        render_chat_tab()
    with kb_tab:
        render_knowledge_tab()
    with tools_tab:
        render_tools_tab()
    with history_tab:
        render_conversations_tab()
    with eval_tab:
        render_eval_tab()


if __name__ == "__main__":
    main()

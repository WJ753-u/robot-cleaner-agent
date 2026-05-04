from typing import Any

from langchain_core.tools import BaseTool

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)


TOOL_META: dict[str, dict[str, str]] = {
    "rag_summarize": {
        "title": "知识库检索",
        "source": "rag",
        "category": "knowledge",
    },
    "get_weather": {
        "title": "实时天气查询",
        "source": "mcp",
        "category": "external",
    },
    "get_user_location": {
        "title": "用户位置获取",
        "source": "local",
        "category": "user_context",
    },
    "get_user_id": {
        "title": "用户ID获取",
        "source": "local",
        "category": "user_context",
    },
    "get_current_month": {
        "title": "当前月份获取",
        "source": "local",
        "category": "time",
    },
    "fetch_external_data": {
        "title": "用户使用记录查询",
        "source": "local",
        "category": "report",
    },
    "fill_context_for_report": {
        "title": "报告场景上下文标记",
        "source": "local",
        "category": "report",
    },
}

AGENT_TOOLS: list[BaseTool] = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]

TOOL_MAP: dict[str, BaseTool] = {tool.name: tool for tool in AGENT_TOOLS}


def get_agent_tools() -> list[BaseTool]:
    return list(AGENT_TOOLS)


def get_tool_by_name(name: str) -> BaseTool | None:
    return TOOL_MAP.get(name)


def list_tool_metadata() -> dict[str, dict[str, Any]]:
    return {
        name: tool_to_metadata(tool)
        for name, tool in TOOL_MAP.items()
    }


def tool_to_metadata(tool: BaseTool) -> dict[str, Any]:
    meta = TOOL_META.get(tool.name, {})
    return {
        "name": tool.name,
        "title": meta.get("title", tool.name),
        "description": (tool.description or "").strip(),
        "args": getattr(tool, "args", {}) or {},
        "source": meta.get("source", "local"),
        "category": meta.get("category", "general"),
    }


def call_registered_tool(name: str, tool_input: dict[str, Any] | None = None) -> Any:
    tool = get_tool_by_name(name)
    if tool is None:
        raise KeyError(f"工具不存在：{name}")
    return tool.invoke(tool_input or {})

import asyncio
import json
import os
from pathlib import Path
from threading import Thread
from typing import Any

from utils.logger_handler import logger

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


AMAP_API_KEY_ENV = "AMAP_MAPS_API_KEY"
AMAP_MCP_URL_TEMPLATE = "https://mcp.amap.com/mcp?key={api_key}"
AMAP_SERVER_NAME = "amap"
AMAP_WEATHER_TOOL_NAME = "maps_weather"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

_cached_amap_tools = None


def _run_async(coro):
    """
    在现有同步 Agent 链路里执行异步 MCP 调用。
    如果当前线程已有事件循环，则放到独立线程中运行，避免 asyncio.run 冲突。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:
            result["error"] = e

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def _stringify_mcp_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item for item in (_stringify_mcp_result(v) for v in value) if item
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    text = getattr(value, "text", None)
    if text:
        return str(text)

    content = getattr(value, "content", None)
    if content is not None:
        return _stringify_mcp_result(content)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), ensure_ascii=False)

    return str(value)


def _get_api_key() -> str:
    load_dotenv(dotenv_path=ENV_FILE)
    return os.getenv(AMAP_API_KEY_ENV, "").strip()


async def _load_amap_tools() -> list[Any]:
    global _cached_amap_tools

    if _cached_amap_tools is not None:
        return _cached_amap_tools

    if MultiServerMCPClient is None:
        raise RuntimeError("未安装 langchain-mcp-adapters，无法加载高德 MCP 工具")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(f"未配置环境变量 {AMAP_API_KEY_ENV}")

    client = MultiServerMCPClient(
        {
            AMAP_SERVER_NAME: {
                "transport": "http",
                "url": AMAP_MCP_URL_TEMPLATE.format(api_key=api_key),
            }
        }
    )
    _cached_amap_tools = await client.get_tools()
    return _cached_amap_tools


def _find_weather_tool(tools: list[Any]):
    for candidate in tools:
        if getattr(candidate, "name", "") == AMAP_WEATHER_TOOL_NAME:
            return candidate

    for candidate in tools:
        name = getattr(candidate, "name", "")
        description = getattr(candidate, "description", "")
        searchable_text = f"{name} {description}".lower()
        if "weather" in searchable_text or "天气" in searchable_text:
            return candidate

    tool_names = ", ".join(getattr(tool, "name", "unknown") for tool in tools)
    raise RuntimeError(f"高德 MCP 中未找到天气工具，可用工具：{tool_names}")


def _build_weather_args(tool: Any, city: str) -> dict[str, str]:
    args = getattr(tool, "args", None)
    fields = set(args.keys()) if isinstance(args, dict) else set()

    if "city" in fields:
        return {"city": city}
    if "location" in fields:
        return {"location": city}
    if "address" in fields:
        return {"address": city}

    return {"city": city}


async def _query_amap_weather(city: str) -> str:
    tools = await _load_amap_tools()
    weather_tool = _find_weather_tool(tools)
    weather_args = _build_weather_args(weather_tool, city)
    result = await weather_tool.ainvoke(weather_args)
    return _stringify_mcp_result(result)


def get_amap_weather(city: str) -> str:
    city = city.strip()
    if not city:
        return "城市名称为空，无法查询实时天气"

    try:
        return _run_async(_query_amap_weather(city)) or "高德 MCP 天气查询返回为空"
    except Exception as e:
        logger.warning(f"[amap mcp]天气查询失败：{e}", exc_info=True)
        return f"高德 MCP 天气查询失败：{e}"

from typing import Any

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    type: str
    content: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    args: dict[str, Any] | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的问题")
    conversation_id: str | None = Field(None, description="会话ID，不传则自动创建新会话")
    include_events: bool = Field(True, description="是否在非流式响应中返回完整Agent事件")


class ChatResponse(BaseModel):
    success: bool
    conversation_id: str | None = None
    answer: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    event_count: int = 0


class ReportRequest(BaseModel):
    query: str = Field("给我生成我的使用报告", min_length=1, description="报告生成请求")
    conversation_id: str | None = Field(None, description="会话ID，不传则自动创建新会话")
    include_events: bool = Field(True, description="是否返回完整Agent事件")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class KnowledgeRebuildResponse(BaseModel):
    success: bool
    message: str


class ToolCallRequest(BaseModel):
    name: str = Field(..., min_length=1, description="工具名称")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="工具入参")


class ToolCallResponse(BaseModel):
    success: bool
    name: str
    result: Any | None = None
    error: str | None = None

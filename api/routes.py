import json
from typing import Any, Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agent.react_agent import ReactAgent
from agent.tools.tool_registry import call_registered_tool, list_tool_metadata
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    KnowledgeRebuildResponse,
    ReportRequest,
    ToolCallRequest,
    ToolCallResponse,
)
from rag.vector_store import VectorStoreService
from storage.database import SessionLocal, get_db
from storage.repositories import (
    get_or_create_conversation,
    list_conversations,
    list_messages,
    list_tool_call_logs,
    save_message,
    save_report,
    save_tool_call_logs,
)
from utils.logger_handler import logger

router = APIRouter()

_agent: ReactAgent | None = None


def get_agent() -> ReactAgent:
    global _agent
    if _agent is None:
        _agent = ReactAgent()
    return _agent


def collect_agent_response(query: str) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    answer_chunks: list[str] = []

    for event in get_agent().execute_stream(query):
        events.append(event)
        if event.get("type") == "answer":
            answer_chunks.append(str(event.get("content", "")))

    answer = "".join(answer_chunks).strip()
    return answer, events


def has_error(events: list[dict[str, Any]]) -> bool:
    return any(event.get("type") == "error" for event in events)


def to_sse_event(event: dict[str, Any]) -> str:
    event_type = event.get("type", "message")
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def conversation_to_dict(conversation) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "source": conversation.source,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def message_to_dict(message) -> dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
    }


def tool_log_to_dict(log) -> dict[str, Any]:
    return {
        "id": log.id,
        "conversation_id": log.conversation_id,
        "tool_call_id": log.tool_call_id,
        "tool_name": log.tool_name,
        "tool_args": log.tool_args,
        "tool_result": log.tool_result,
        "success": log.success,
        "error_message": log.error_message,
        "latency_ms": log.latency_ms,
        "created_at": log.created_at,
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="robot-agent-api",
        version="0.1.0",
    )


@router.get("/tools")
def list_tools() -> dict[str, Any]:
    tools = list_tool_metadata()
    return {
        "items": tools,
        "count": len(tools),
    }


@router.post("/tools/call", response_model=ToolCallResponse)
def call_tool(request: ToolCallRequest) -> ToolCallResponse:
    try:
        result = call_registered_tool(request.name, request.tool_input)
    except KeyError as e:
        return ToolCallResponse(
            success=False,
            name=request.name,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"[api tool call]工具调用失败：{e}", exc_info=True)
        return ToolCallResponse(
            success=False,
            name=request.name,
            error=str(e),
        )

    return ToolCallResponse(
        success=True,
        name=request.name,
        result=result,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    conversation = get_or_create_conversation(
        db,
        conversation_id=request.conversation_id,
        query=request.query,
        source="chat",
    )
    save_message(
        db,
        conversation_id=conversation.conversation_id,
        role="user",
        content=request.query,
    )

    try:
        answer, events = collect_agent_response(request.query)
    except Exception as e:
        logger.error(f"[api chat]请求失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    success = not has_error(events)
    save_message(
        db,
        conversation_id=conversation.conversation_id,
        role="assistant",
        content=answer,
    )
    save_tool_call_logs(
        db,
        conversation_id=conversation.conversation_id,
        events=events,
    )

    return ChatResponse(
        success=success,
        conversation_id=conversation.conversation_id,
        answer=answer,
        events=events if request.include_events else [],
        event_count=len(events),
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_generator() -> Generator[str, None, None]:
        db = SessionLocal()
        events: list[dict[str, Any]] = []
        answer_chunks: list[str] = []
        conversation = None

        try:
            conversation = get_or_create_conversation(
                db,
                conversation_id=request.conversation_id,
                query=request.query,
                source="chat",
            )
            save_message(
                db,
                conversation_id=conversation.conversation_id,
                role="user",
                content=request.query,
            )
            yield to_sse_event({
                "type": "metadata",
                "conversation_id": conversation.conversation_id,
            })

            for event in get_agent().execute_stream(request.query):
                events.append(event)
                if event.get("type") == "answer":
                    answer_chunks.append(str(event.get("content", "")))
                yield to_sse_event(event)
        except Exception as e:
            logger.error(f"[api chat stream]请求失败：{e}", exc_info=True)
            error_event = {
                "type": "error",
                "content": f"流式请求失败：{e}",
                "error": str(e),
            }
            events.append(error_event)
            yield to_sse_event(error_event)
        finally:
            try:
                if conversation is not None:
                    answer = "".join(answer_chunks).strip()
                    if not answer and has_error(events):
                        answer = "请求处理失败，未生成有效回答。"
                    if answer:
                        save_message(
                            db,
                            conversation_id=conversation.conversation_id,
                            role="assistant",
                            content=answer,
                        )
                    save_tool_call_logs(
                        db,
                        conversation_id=conversation.conversation_id,
                        events=events,
                    )
            except Exception as e:
                logger.error(f"[api chat stream]流式结果落库失败：{e}", exc_info=True)
            finally:
                db.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/report/generate", response_model=ChatResponse)
def generate_report(
    request: ReportRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    conversation = get_or_create_conversation(
        db,
        conversation_id=request.conversation_id,
        query=request.query,
        source="report",
    )
    save_message(
        db,
        conversation_id=conversation.conversation_id,
        role="user",
        content=request.query,
    )

    try:
        answer, events = collect_agent_response(request.query)
    except Exception as e:
        logger.error(f"[api report]请求失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    success = not has_error(events)
    save_message(
        db,
        conversation_id=conversation.conversation_id,
        role="assistant",
        content=answer,
    )
    save_tool_call_logs(
        db,
        conversation_id=conversation.conversation_id,
        events=events,
    )
    save_report(
        db,
        conversation_id=conversation.conversation_id,
        query=request.query,
        content=answer,
        success=success,
    )

    return ChatResponse(
        success=success,
        conversation_id=conversation.conversation_id,
        answer=answer,
        events=events if request.include_events else [],
        event_count=len(events),
    )


@router.post("/knowledge/rebuild", response_model=KnowledgeRebuildResponse)
def rebuild_knowledge() -> KnowledgeRebuildResponse:
    try:
        VectorStoreService().load_document()
    except Exception as e:
        logger.error(f"[api knowledge rebuild]知识库重建失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return KnowledgeRebuildResponse(
        success=True,
        message="知识库重建任务已执行完成",
    )


@router.get("/conversations")
def get_conversations(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    limit = min(max(limit, 1), 100)
    conversations = list_conversations(db, limit=limit)
    return {
        "items": [conversation_to_dict(item) for item in conversations],
        "count": len(conversations),
    }


@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    messages = list_messages(db, conversation_id=conversation_id)
    return {
        "conversation_id": conversation_id,
        "items": [message_to_dict(item) for item in messages],
        "count": len(messages),
    }


@router.get("/conversations/{conversation_id}/tool-logs")
def get_conversation_tool_logs(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    tool_logs = list_tool_call_logs(db, conversation_id=conversation_id)
    return {
        "conversation_id": conversation_id,
        "items": [tool_log_to_dict(item) for item in tool_logs],
        "count": len(tool_logs),
    }

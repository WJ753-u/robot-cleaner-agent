from uuid import uuid4

from sqlalchemy.orm import Session

from storage.models import Conversation, Message, Report, ToolCallLog


def _build_title(query: str) -> str:
    normalized = " ".join(query.split())
    return normalized[:50] or "新会话"


def get_or_create_conversation(
    db: Session,
    *,
    conversation_id: str | None,
    query: str,
    source: str,
) -> Conversation:
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )
        if conversation:
            return conversation

    conversation = Conversation(
        conversation_id=conversation_id or str(uuid4()),
        title=_build_title(query),
        source=source,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def save_message(
    db: Session,
    *,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def _is_failed_tool_result(event: dict) -> bool:
    content = str(event.get("content", ""))
    error = event.get("error")
    return bool(error) or "失败" in content or "未配置" in content


def save_tool_call_logs(
    db: Session,
    *,
    conversation_id: str,
    events: list[dict],
) -> list[ToolCallLog]:
    pending_calls: dict[str, dict] = {}
    saved_logs: list[ToolCallLog] = []

    for event in events:
        event_type = event.get("type")
        tool_call_id = str(event.get("tool_call_id") or "")

        if event_type == "tool_call":
            pending_calls[tool_call_id] = event
            continue

        if event_type != "tool_result":
            continue

        call_event = pending_calls.pop(tool_call_id, {})
        success = not _is_failed_tool_result(event)
        log = ToolCallLog(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id or None,
            tool_name=str(
                event.get("tool_name")
                or call_event.get("tool_name")
                or "unknown_tool"
            ),
            tool_args=call_event.get("args"),
            tool_result=str(event.get("content", "")),
            success=success,
            error_message=None if success else str(event.get("content", "")),
            latency_ms=None,
        )
        db.add(log)
        saved_logs.append(log)

    for tool_call_id, call_event in pending_calls.items():
        log = ToolCallLog(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id or None,
            tool_name=str(call_event.get("tool_name") or "unknown_tool"),
            tool_args=call_event.get("args"),
            tool_result=None,
            success=False,
            error_message="工具调用未返回结果",
            latency_ms=None,
        )
        db.add(log)
        saved_logs.append(log)

    db.commit()
    for log in saved_logs:
        db.refresh(log)
    return saved_logs


def save_report(
    db: Session,
    *,
    conversation_id: str,
    query: str,
    content: str,
    success: bool,
) -> Report:
    report = Report(
        conversation_id=conversation_id,
        query=query,
        content=content,
        success=success,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def list_conversations(db: Session, *, limit: int = 20) -> list[Conversation]:
    return (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .all()
    )


def list_messages(
    db: Session,
    *,
    conversation_id: str,
) -> list[Message]:
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )


def list_tool_call_logs(
    db: Session,
    *,
    conversation_id: str,
) -> list[ToolCallLog]:
    return (
        db.query(ToolCallLog)
        .filter(ToolCallLog.conversation_id == conversation_id)
        .order_by(ToolCallLog.created_at.asc(), ToolCallLog.id.asc())
        .all()
    )

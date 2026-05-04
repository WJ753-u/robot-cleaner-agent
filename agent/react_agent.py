from langchain.agents import create_agent
from model.factory import chat_model
from model.llama_cpp_client import LlamaCppCompletionClient
from utils.prompt_loader import load_system_prompts
from agent.tools.tool_registry import get_agent_tools
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.logger_handler import logger

class ReactAgent: 
    def __init__(self):
        self.final_answer_client = LlamaCppCompletionClient()
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = get_agent_tools(),
            middleware = [monitor_tool, log_before_model, report_prompt_switch],
        )

    @staticmethod#静态方法，不需要实例化对象即可调用
    def _message_key(message):
        return (
            getattr(message, "id", None)
            or f"{getattr(message, 'type', type(message).__name__)}|"
            f"{getattr(message, 'tool_call_id', '')}|"
            f"{getattr(message, 'content', '')}|"
            f"{getattr(message, 'tool_calls', '')}"
        )

    @staticmethod
    def _tool_call_events(message):
        tool_calls = getattr(message, "tool_calls", None) or []
        events = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id", "")
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})
            else:
                tool_call_id = getattr(tool_call, "id", "")
                name = getattr(tool_call, "name", "")
                args = getattr(tool_call, "args", {})
            events.append({
                "type": "tool_call",
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "args": args,
            })
        return events

    @staticmethod
    def _tool_result_event(message):
        name = getattr(message, "name", "unknown_tool")
        tool_call_id = getattr(message, "tool_call_id", "")
        content = getattr(message, "content", "")
        if not content:
            content = "空结果"
        return {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": name,
            "content": content,
        }

    @staticmethod
    def format_event(event):
        event_type = event.get("type")
        if event_type == "tool_call":
            return f"[工具调用] {event.get('tool_name', '')} 参数：{event.get('args', {})}"
        if event_type == "tool_result":
            return f"[工具结果] {event.get('tool_name', 'unknown_tool')}：{event.get('content', '')}"
        if event_type == "error":
            return f"[错误] {event.get('content', '')}"
        return str(event.get("content", ""))

    def execute_stream(self, query):
        input_dict = {
            "messages":[
            {"role": "user", "content": query}
            ]
        }
        seen_messages = set()
        tool_result_chunks = []
        draft_answer_chunks = []
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):#流式输出，第三个context就是上下文runtime中的信息，就是提示词切换的标记。首先需要默认关闭report（False），因为report需要在中间件中处理。
                latest_message = chunk["messages"][-1]#获取最新的消息
                message_key = self._message_key(latest_message)
                if message_key in seen_messages:#如果消息已处理过，直接跳过
                    continue#continue是指满足条件后，放弃处理当前数据的循环，直接处理下一项数据
                seen_messages.add(message_key)

                message_type = getattr(latest_message, "type", "")
                if message_type in ("human", "system"):#如果是用户消息或系统消息，直接跳过
                    continue
                tool_call_events = self._tool_call_events(latest_message)
                if tool_call_events:#如果有工具调用事件
                    for event in tool_call_events:#遍历工具调用事件
                        yield event#返回工具调用事件
                    continue#继续处理下一个消息

                if message_type == "tool":
                    tool_result_event = self._tool_result_event(latest_message)
                    tool_result_chunks.append(
                        f"{tool_result_event.get('tool_name', 'unknown_tool')}："
                        f"{tool_result_event.get('content', '')}"
                    )
                    yield tool_result_event
                    continue

                if message_type == "ai" and latest_message.content:#如果是AI消息且有内容
                    draft_answer_chunks.append(latest_message.content.strip())

            draft_answer = "".join(draft_answer_chunks).strip()
            evidence = "\n".join(tool_result_chunks).strip()
            final_answer = (
                self.final_answer_client.try_complete_final_answer(
                    query,
                    evidence,
                    draft_answer,
                )
                or draft_answer
            )
            if final_answer:
                yield {
                    "type": "answer",
                    "content": final_answer,
                }
        except Exception as e:
            logger.error(f"[agent stream]执行失败：{e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"Agent执行失败：{e}",
                "error": str(e),
            }
# if __name__ == "__main__":
#     agent = ReactAgent()
#     for event in agent.execute_stream("给我生成我的使用报告"):
#         print(ReactAgent.format_event(event), flush=True)

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import (
    ToolCallRequest,
    ModelRequest,
    AgentState,
)
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    after_model,
)
from utils.prompt_loader import load_report_prompts, load_system_prompts
@wrap_tool_call#包装工具调用，添加监控功能
def monitor_tool(#工具执行的监控
    request: ToolCallRequest,#请求的数据封装（入参）
    handler: Callable[[ToolCallRequest], ToolMessage | Command],#执行的函数本身（函数）
) -> ToolMessage | Command:#返回 ToolMessage | Command
    logger.info(f"[tool monitor]执行工具： {request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数： {request.tool_call['args']}")
    try:
        res = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}执行成功，返回结果：{res}")
            
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return res
    except Exception as e:
        logger.error(f"[tool monitor]工具{request.tool_call['name']}执行失败，错误信息：{e}")
        raise e#抛出异常，让调用者处理
@before_model
def log_before_model(#在模型调用前记录日志
    state:AgentState,#整个agent智能体中的状态记录
    runtime:Runtime,#记录整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]模型即将调用，当前有：{len(state['messages'])}条消息")#info级别日志，用于记录模型调用前的状态
    logger.debug(f"[log_before_model]当前上下文类型：{type(state['messages'][-1].content)}|{state['messages'][-1].content.strip()}")#debug级别日志，用于调试时查看上下文,-1表示取最后一个消息(只取最新的)
    return None#返回None，不改变模型调用行为，结束就可以
@dynamic_prompt#每一次在生成提示词之前调用此函数
def report_prompt_switch(request: ModelRequest):#动态提示词切换
    is_report = request.runtime.context.get("report", False)#获取到report的key值，取不到默认返回false
    if is_report:#如果是报告生成场景，返回报告生成的提示词内容
        print("="*50)
        print(load_report_prompts())
        return load_report_prompts()#返回报告提示词
    return load_system_prompts()#返回系统提示词
    

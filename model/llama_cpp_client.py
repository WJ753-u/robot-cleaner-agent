import re
from typing import Any

import requests

from utils.config_handler import rag_config
from utils.logger_handler import logger


DEFAULT_SYSTEM_PROMPT = (
    "你是扫地/扫拖一体机器人领域的智能客服助手。只回答产品选购、故障排查、"
    "维护保养、耗材更换、地图建图、基站和使用设置相关问题。领域内问题要"
    "直接给出中文排查步骤或使用建议；领域外请求要礼貌拒绝，不要完成无关任务。"
)


class LlamaCppCompletionClient:
    """Call llama.cpp server /completion with the prompt format verified locally."""

    def __init__(self):
        self.enabled = bool(rag_config.get("llama_cpp_enabled", False))
        self.base_url = str(
            rag_config.get("llama_cpp_base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        self.timeout = int(rag_config.get("llama_cpp_timeout", 240))
        self.system_prompt = str(
            rag_config.get("llama_cpp_system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        self.n_predict = int(rag_config.get("llama_cpp_n_predict", 512))
        self.temperature = float(rag_config.get("llama_cpp_temperature", 0.1))
        self.top_p = float(rag_config.get("llama_cpp_top_p", 0.8))
        self.top_k = int(rag_config.get("llama_cpp_top_k", 20))
        self.repeat_penalty = float(rag_config.get("llama_cpp_repeat_penalty", 1.15))

    @staticmethod
    def clean_output(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"(?s)<think>.*?</think>\s*", "", text)
        text = text.replace("Thinking...", "")
        return text.strip()

    def build_prompt(self, query: str, context: str) -> str:
        user_prompt = (
            "请基于以下参考资料回答用户问题。回答必须围绕扫地/扫拖机器人场景，"
            "只输出最终客服回复，不输出思考过程。\n\n"
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context}"
        )
        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt} /no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def build_final_answer_prompt(
        self,
        query: str,
        evidence: str,
        draft_answer: str,
    ) -> str:
        user_prompt = (
            "请根据用户问题、工具调用结果和草稿回答，生成最终客服回复。"
            "要求：只输出最终回答；不输出思考过程；回答必须简洁、具体、可执行；"
            "如果用户问题不属于扫地/扫拖机器人相关范围，需要礼貌拒绝。\n\n"
            f"用户问题：{query}\n\n"
            f"工具调用结果：\n{evidence or '无'}\n\n"
            f"草稿回答：\n{draft_answer or '无'}"
        )
        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt} /no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def complete(self, query: str, context: str) -> str:
        payload: dict[str, Any] = {
            "prompt": self.build_prompt(query, context),
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return self.clean_output(str(data.get("content", "")))

    def complete_final_answer(
        self,
        query: str,
        evidence: str,
        draft_answer: str,
    ) -> str:
        payload: dict[str, Any] = {
            "prompt": self.build_final_answer_prompt(query, evidence, draft_answer),
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return self.clean_output(str(data.get("content", "")))

    def try_complete(self, query: str, context: str) -> str | None:
        if not self.enabled:
            return None

        try:
            answer = self.complete(query, context)
            if answer:
                return answer
        except Exception as e:
            logger.warning(
                f"[llama.cpp]领域模型生成失败，回退到默认RAG链路：{e}",
                exc_info=True,
            )
        return None

    def try_complete_final_answer(
        self,
        query: str,
        evidence: str,
        draft_answer: str,
    ) -> str | None:
        if not self.enabled:
            return None

        try:
            answer = self.complete_final_answer(query, evidence, draft_answer)
            if answer:
                return answer
        except Exception as e:
            logger.warning(
                f"[llama.cpp]最终回答生成失败，回退到Agent原始回答：{e}",
                exc_info=True,
            )
        return None

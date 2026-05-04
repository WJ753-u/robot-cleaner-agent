#RAG总结服务
from pathlib import Path

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from model.llama_cpp_client import LlamaCppCompletionClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def print_prompt(prompt):#打印提示词
    print("*"*50)
    print(prompt.to_string())#打印提示词模板对象的字符串表示
    print("*"*50)
    return prompt
class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()#原始提示词文本，只是普通字符串
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)#把字符串文本进一步包装后的提示词模板对象，是langchain能识别并参与链调用的模板对象
        self.model = chat_model
        self.llama_cpp_client = LlamaCppCompletionClient()
        self.chain = self._init_chain()
    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    def retriever_docs(self, query:str) -> list[Document]:
        return self.retriever.invoke(query)

    @staticmethod
    def format_references(context_docs: list[Document]) -> str:
        if not context_docs:
            return ""

        references = []
        seen = set()
        for doc in context_docs:
            metadata = doc.metadata or {}
            source_value = metadata.get("source_name") or metadata.get("knowledge_file") or metadata.get("source")
            source_name = Path(str(source_value)).name if source_value else "未知来源"
            chunk_id = metadata.get("chunk_id", "未知片段")
            page = metadata.get("page")
            key = (source_name, chunk_id, page)
            if key in seen:
                continue
            seen.add(key)

            page_text = f"，第{page + 1}页" if isinstance(page, int) else ""
            score = metadata.get("hybrid_score")
            vector_rank = metadata.get("vector_rank")
            bm25_rank = metadata.get("bm25_rank")
            source_intent_score = metadata.get("source_intent_score")
            bge_rerank_score = metadata.get("bge_rerank_score")
            rerank_final_score = metadata.get("rerank_final_score")
            score_text = f"，hybrid_score={score}" if score is not None else ""
            rank_text = []
            if vector_rank:
                rank_text.append(f"vector_rank={vector_rank}")
            if bm25_rank:
                rank_text.append(f"bm25_rank={bm25_rank}")
            if source_intent_score:
                rank_text.append(f"source_intent_score={source_intent_score}")
            if bge_rerank_score is not None:
                rank_text.append(f"bge_rerank_score={bge_rerank_score}")
            if rerank_final_score is not None:
                rank_text.append(f"rerank_final_score={rerank_final_score}")
            rank_text = f"，{', '.join(rank_text)}" if rank_text else ""
            preview = RagSummarizeService._format_preview(doc.page_content)
            references.append(
                f"{len(references) + 1}. {source_name} - chunk_{chunk_id}{page_text}{score_text}{rank_text}\n"
                f"   片段预览：{preview}"
            )

        return "\n".join(references)

    @staticmethod
    def _format_preview(text: str, max_length: int = 120) -> str:
        preview = " ".join((text or "").split())
        if len(preview) <= max_length:
            return preview
        return preview[:max_length] + "..."

    def rag_summarize(self, query:str) -> str:#RAG总结服务
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"第{counter}条文档： {doc.page_content} | 参考元数据：{doc.metadata}\n"
        answer = self.llama_cpp_client.try_complete(query, context)
        if not answer:
            answer = self.chain.invoke(
                {
                "input":query,
                "context":context,
                }
            )
        references = self.format_references(context_docs)
        if references:
            return f"{answer}\n\n参考来源：\n{references}"
        return answer
if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("小户型适合那种扫地机器人？"))

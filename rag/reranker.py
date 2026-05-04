from __future__ import annotations

from langchain_core.documents import Document

from utils.config_handler import chroma_config


class BGEReranker:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = chroma_config.get(
            "reranker_model_name",
            "BAAI/bge-reranker-v2-m3",
        )

        self.model_name = model_name
        self.torch = torch
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        if chroma_config.get("reranker_use_fp16", False) and self.device == "cuda":
            self.model.half()

        self.normalize = chroma_config.get("reranker_normalize", True)
        self.batch_size = chroma_config.get("reranker_batch_size", 8)
        self.max_length = chroma_config.get("reranker_max_length", 512)
        self.reranker_weight = chroma_config.get("reranker_weight", 0.85)
        self.hybrid_weight = chroma_config.get("reranker_hybrid_weight", 0.15)

    def _resolve_device(self) -> str:
        configured_device = chroma_config.get("reranker_device")
        if configured_device:
            return configured_device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        scores = self._compute_scores(query, docs)

        reranked: list[tuple[float, Document]] = []
        for doc, score in zip(docs, scores):
            bge_score = float(score)
            hybrid_score = float((doc.metadata or {}).get("hybrid_score", 0.0))
            final_score = (
                self.reranker_weight * bge_score
                + self.hybrid_weight * hybrid_score
            )
            doc.metadata = {
                **(doc.metadata or {}),
                "bge_rerank_score": round(bge_score, 6),
                "rerank_final_score": round(final_score, 6),
                "reranker_model": self.model_name,
            }
            reranked.append((final_score, doc))

        reranked.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in reranked]

    def _compute_scores(self, query: str, docs: list[Document]) -> list[float]:
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = docs[start:start + self.batch_size]
                queries = [query] * len(batch_docs)
                passages = [doc.page_content for doc in batch_docs]
                inputs = self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                if self.normalize:
                    logits = self.torch.sigmoid(logits)
                scores.extend(logits.cpu().tolist())
        return scores

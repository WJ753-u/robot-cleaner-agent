import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from utils.config_handler import chroma_config
from utils.file_handler import (
    csv_loader,
    doc_loader,
    docx_loader,
    excel_loader,
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    pptx_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


DOMAIN_TERMS = [
    "扫地机器人",
    "扫拖机器人",
    "扫拖一体",
    "吸力下降",
    "吸不干净",
    "尘盒",
    "滤网",
    "风道",
    "滚刷",
    "主刷",
    "边刷",
    "吸口",
    "集尘袋",
    "自动集尘",
    "充电座",
    "充电触点",
    "找不到充电座",
    "回充",
    "电量",
    "续航",
    "wifi",
    "地图",
    "建图",
    "多楼层",
    "水箱",
    "出水",
    "漏水",
    "拖布",
    "木地板",
    "地毯",
    "禁拖",
    "宠物",
    "毛发",
    "小户型",
    "大户型",
    "选购",
    "保养",
    "维护",
    "清洁",
    "清理",
    "故障",
    "异常",
    "排查",
    "异响",
    "卡顿",
]


SOURCE_INTENT_RULES = [
    {
        "query_terms": ["选购", "选择", "购买", "买", "适合", "关注", "续航", "户型"],
        "source_terms": ["选购指南"],
    },
    {
        "query_terms": ["故障", "异常", "无法", "不转", "不出水", "找不到", "错乱", "不完整", "怎么办", "怎么处理", "如何处理", "排查", "解决"],
        "source_terms": ["故障排除"],
    },
    {
        "query_terms": ["保养", "维护", "清理", "清洁", "每天使用后", "多久清理", "毛发"],
        "source_terms": ["维护保养"],
    },
    {
        "query_terms": ["拖地", "拖布", "水箱", "出水量", "禁拖", "木地板", "扫拖"],
        "source_terms": ["扫拖一体"],
    },
]


class SimpleBM25Retriever:
    def __init__(self, docs: list[Document], k: int = 5):
        self.docs = docs
        self.k = k
        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in docs]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized_docs:
            self.doc_freq.update(set(tokens))
        self.avgdl = (
            sum(len(tokens) for tokens in self.tokenized_docs) / len(self.tokenized_docs)
            if self.tokenized_docs
            else 0
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower()
        english_tokens = re.findall(r"[a-zA-Z0-9]+", normalized)
        domain_tokens = [
            term for term in DOMAIN_TERMS
            if term in normalized
        ]
        chinese_tokens = []
        for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
            chinese_tokens.extend(
                segment[index:index + size]
                for size in (2, 3)
                for index in range(max(len(segment) - size + 1, 0))
            )
        return english_tokens + domain_tokens + chinese_tokens

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        tokens = self.tokenized_docs[doc_index]
        if not tokens:
            return 0.0

        frequencies = Counter(tokens)
        score = 0.0
        total_docs = len(self.docs)
        doc_len = len(tokens)
        k1 = 1.5
        b = 0.75

        for token in query_tokens:
            freq = frequencies.get(token, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = freq + k1 * (1 - b + b * doc_len / (self.avgdl or 1))
            score += idf * (freq * (k1 + 1)) / denominator
        return score

    def invoke(self, query: str) -> list[Document]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = [
            (self._score(query_tokens, index), doc)
            for index, doc in enumerate(self.docs)
        ]
        scored = [(score, doc) for score, doc in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, doc in scored[: self.k]:
            cloned = Document(
                page_content=doc.page_content,
                metadata={**(doc.metadata or {}), "bm25_score": score},
            )
            results.append(cloned)
        return results


class HybridRetriever:
    QUERY_EXPANSIONS = {#查询扩展词
        "保养": ["维护", "清洁", "清理"],
        "清洁": ["保养", "维护", "清理"],
        "故障": ["问题", "异常", "排查", "修复"],
        "异常": ["故障", "问题", "排查"],
        "吸力": ["吸不干净", "堵塞", "滤网"],
        "充电": ["回充", "充电座", "电量"],
        "回充": ["充电", "充电座", "找不到充电座"],
        "水箱": ["出水", "漏水", "拖布"],
        "漏水": ["水箱", "出水", "拖布"],
        "地毯": ["禁拖", "拖布抬升", "避障"],
    }

    def __init__(self, vector_store: Chroma, bm25_docs: list[Document], k: int):#初始化混合检索器
        self.vector_store = vector_store
        self.k = k
        self.vector_weight = chroma_config.get("vector_weight", 0.75)
        self.bm25_weight = chroma_config.get("bm25_weight", 0.15)
        self.overlap_weight = chroma_config.get("overlap_weight", 0.10)
        self.source_intent_weight = chroma_config.get("source_intent_weight", 0.20)
        self.max_chunks_per_source = chroma_config.get("max_chunks_per_source", 2)
        self.reranker = self._init_reranker()
        self.vector_retriever = vector_store.as_retriever(#创建向量检索器
            search_kwargs={"k": max(k, chroma_config.get("vector_k", k))}#设置检索数量
        )
        self.bm25_retriever = SimpleBM25Retriever(#创建BM25检索器
            bm25_docs,
            k=chroma_config.get("bm25_k", k),#设置检索数量
        )

    @staticmethod
    def _init_reranker():
        if not chroma_config.get("reranker_enabled", False):
            return None
        try:
            from rag.reranker import BGEReranker

            return BGEReranker()
        except Exception as e:
            logger.warning(f"[reranker] BGE reranker unavailable, skip rerank: {e}")
            return None

    @classmethod
    def _expand_query(cls, query: str) -> str:#拓展查询词
        if not chroma_config.get("query_expansion", True):#如果查询扩展词未开启
            return query

        expanded_terms = []
        for keyword, aliases in cls.QUERY_EXPANSIONS.items():#遍历查询扩展词
            if keyword in query:
                expanded_terms.extend(aliases)

        if not expanded_terms:
            return query

        unique_terms = []
        for term in expanded_terms:
            if term not in query and term not in unique_terms:
                unique_terms.append(term)
        return f"{query} {' '.join(unique_terms)}" if unique_terms else query

    @staticmethod
    def _doc_key(doc: Document) -> str:
        metadata = doc.metadata or {}
        return str(metadata.get("chunk_uid") or metadata.get("id") or doc.page_content[:80])

    @staticmethod
    def _source_key(doc: Document) -> str:
        metadata = doc.metadata or {}
        source = (
            metadata.get("source_name")
            or metadata.get("knowledge_file")
            or metadata.get("source")
            or "unknown"
        )
        return os.path.basename(str(source))

    @staticmethod
    def _source_overlap_score(query: str, doc: Document) -> float:
        query_chars = set(re.findall(r"[\u4e00-\u9fff]", query))
        doc_chars = set(re.findall(r"[\u4e00-\u9fff]", doc.page_content[:300]))
        if not query_chars or not doc_chars:
            return 0.0
        return len(query_chars & doc_chars) / len(query_chars)

    @classmethod
    def _source_intent_score(cls, query: str, doc: Document) -> float:#计算来源意图分数
        """
        计算查询与文档的来源意图分数。
        分数范围为0到1，1表示查询与文档的来源意图高度相关。
        """
        source = cls._source_key(doc)
        for rule in SOURCE_INTENT_RULES:
            if not any(term in query for term in rule["query_terms"]):
                continue
            if any(term in source for term in rule["source_terms"]):
                return 1.0
        return 0.0

    def invoke(self, query: str) -> list[Document]:
        bm25_query = self._expand_query(query)
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(bm25_query)

        candidates: dict[str, dict] = {}
        for rank, doc in enumerate(vector_docs, start=1):
            key = self._doc_key(doc)
            candidates.setdefault(key, {"doc": doc, "vector_rank": None, "bm25_rank": None})
            candidates[key]["vector_rank"] = rank

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._doc_key(doc)
            candidates.setdefault(key, {"doc": doc, "vector_rank": None, "bm25_rank": None})
            candidates[key]["bm25_rank"] = rank

        ranked = []
        for item in candidates.values():
            doc = item["doc"]
            vector_rank = item["vector_rank"]
            bm25_rank = item["bm25_rank"]
            vector_score = 1 / vector_rank if vector_rank else 0
            bm25_score = 1 / bm25_rank if bm25_rank else 0
            overlap_score = self._source_overlap_score(query, doc)
            source_intent_score = self._source_intent_score(query, doc)
            final_score = (
                self.vector_weight * vector_score
                + self.bm25_weight * bm25_score
                + self.overlap_weight * overlap_score
                + self.source_intent_weight * source_intent_score
            )
            doc.metadata = {
                **(doc.metadata or {}),
                "vector_rank": vector_rank,
                "bm25_rank": bm25_rank,
                "hybrid_score": round(final_score, 6),
                "retrieval_mode": "hybrid",
                "source_intent_score": source_intent_score,
                "expanded_query": bm25_query if bm25_query != query else None,
            }
            ranked.append((final_score, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked_docs = [doc for _, doc in ranked]
        if self.reranker:
            candidate_limit = chroma_config.get("reranker_candidate_limit", 20)
            rerank_candidates = ranked_docs[:candidate_limit]
            reranked_candidates = self.reranker.rerank(query, rerank_candidates)
            ranked_docs = reranked_candidates + ranked_docs[candidate_limit:]

        return self._apply_source_diversity(ranked_docs)

    def _apply_source_diversity(self, ranked_docs: list[Document]) -> list[Document]:
        if not self.max_chunks_per_source or self.max_chunks_per_source <= 0:
            return ranked_docs[: self.k]

        selected = []
        overflow = []
        source_counts: Counter[str] = Counter()
        for doc in ranked_docs:
            source = self._source_key(doc)
            if source_counts[source] < self.max_chunks_per_source:
                selected.append(doc)
                source_counts[source] += 1
            else:
                overflow.append(doc)
            if len(selected) >= self.k:
                break

        if len(selected) < self.k:
            selected.extend(overflow[: self.k - len(selected)])
        return selected[: self.k]


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

        self.manifest_path = Path(
            get_abs_path(chroma_config.get("manifest_path", "chroma_db/manifest.json"))
        )
        self.bm25_docs = self._load_bm25_docs_from_vector_store()

    def get_retriever(self):
        if chroma_config.get("hybrid_search", True):
            return HybridRetriever(
                self.vector_store,
                self.bm25_docs,
                k=chroma_config["k"],
            )
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    @staticmethod
    def clean_text(text: str) -> str:#清理文本
        text = text.replace("\ufeff", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?m)^\s*[-*]{3,}\s*$", "", text)
        return text.strip()

    @staticmethod
    def normalize_for_dedup(text: str) -> str:#归一化文本，用于去重
        return re.sub(r"\s+", "", text.lower())

    @staticmethod
    def content_hash(text: str) -> str:#计算文本哈希值
        import hashlib

        return hashlib.md5(VectorStoreService.normalize_for_dedup(text).encode("utf-8")).hexdigest()

    def _load_manifest(self) -> dict:#加载manifest文件
        if not self.manifest_path.exists():#如果manifest文件不存在
            return {"files": {}}#返回空字典
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"[manifest]文件损坏，将重新创建：{self.manifest_path}")
            return {"files": {}}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _delete_file_chunks(self, source_name: str) -> None:
        collection = getattr(self.vector_store, "_collection", None)
        if collection is None:
            return
        collection.delete(where={"source_name": source_name})

    def _sync_stale_chunks(self, manifest: dict, current_source_names: set[str]) -> None:
        known_files = manifest.get("files", {})
        stale_sources = [
            source_name
            for source_name in known_files
            if source_name not in current_source_names
        ]
        for source_name in stale_sources:
            self._delete_file_chunks(source_name)
            known_files.pop(source_name, None)
            logger.info(f"[manifest]已清理陈旧切片：{source_name}")

    def _get_file_documents(self, read_path: str) -> list[Document]:
        suffix = Path(read_path).suffix.lower()
        if suffix == ".pdf":
            return pdf_loader(read_path)
        if suffix == ".txt":
            return txt_loader(read_path)
        if suffix == ".doc":
            return doc_loader(read_path)
        if suffix == ".docx":
            return docx_loader(read_path)
        if suffix in (".xlsx", ".xls"):
            return excel_loader(read_path)
        if suffix == ".pptx":
            return pptx_loader(read_path)
        if suffix == ".csv":
            return csv_loader(read_path)
        logger.error(f"[文件加载]文件{read_path}不是支持的知识库文件类型")
        return []

    def _looks_like_qa_line(self, line: str) -> bool:
        return bool(
            re.match(r"^\s*\d+[\.、]\s*\*\*.+[？?]\*\*", line)
            or re.match(r"^\s*\d+[\.、]\s*.+[？?]\s*$", line)
        )

    def _split_faq_documents(self, documents: list[Document], source_path: str) -> list[Document]:
        faq_docs: list[Document] = []
        source_name = os.path.basename(source_path)

        for page_index, doc in enumerate(documents, start=1):
            lines = self.clean_text(doc.page_content).splitlines()
            current: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if self._looks_like_qa_line(stripped) and current:
                    faq_docs.append(self._build_faq_document(current, doc, source_name, page_index))
                    current = []
                current.append(stripped)
            if current:
                faq_docs.append(self._build_faq_document(current, doc, source_name, page_index))

        return [doc for doc in faq_docs if len(doc.page_content) >= 20]

    def _build_faq_document(
        self,
        lines: list[str],
        source_doc: Document,
        source_name: str,
        page_index: int,
    ) -> Document:
        content = self.clean_text("\n".join(lines))
        metadata = {
            **(source_doc.metadata or {}),
            "source_name": source_name,
            "knowledge_file": source_name,
            "chunk_type": "faq",
            "page_index": page_index,
        }
        return Document(page_content=content, metadata=metadata)

    def _split_documents_by_type(self, documents: list[Document], source_path: str) -> list[Document]:
        source_name = os.path.basename(source_path)
        if "100问" in source_name or "问答" in source_name:
            faq_docs = self._split_faq_documents(documents, source_path)
            if faq_docs:
                return faq_docs

        cleaned_docs = [
            Document(
                page_content=self.clean_text(doc.page_content),
                metadata={**(doc.metadata or {}), "chunk_type": "recursive"},
            )
            for doc in documents
            if self.clean_text(doc.page_content)
        ]
        return self.splitter.split_documents(cleaned_docs)

    def _enrich_and_dedup_chunks(
        self,
        chunks: list[Document],
        source_path: str,
        file_md5: str,
    ) -> list[Document]:
        source_name = os.path.basename(source_path)
        enriched: list[Document] = []
        seen_hashes: set[str] = set()

        for chunk_index, doc in enumerate(chunks, start=1):
            cleaned = self.clean_text(doc.page_content)
            if len(cleaned) < chroma_config.get("min_chunk_chars", 20):
                continue

            chunk_hash = self.content_hash(cleaned)
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            metadata = doc.metadata or {}
            source = metadata.get("source", source_path)
            chunk_uid = f"{source_name}:{file_md5}:{chunk_index}:{chunk_hash[:8]}"
            enriched.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        **metadata,
                        "source": source,
                        "source_name": os.path.basename(source),
                        "chunk_id": chunk_index,
                        "chunk_uid": chunk_uid,
                        "chunk_hash": chunk_hash,
                        "file_md5": file_md5,
                        "knowledge_file": source_name,
                    },
                )
            )
        return enriched

    def _load_bm25_docs_from_vector_store(self) -> list[Document]:
        collection = getattr(self.vector_store, "_collection", None)
        if collection is None:
            return []
        try:
            data = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"[bm25]从向量库加载文本失败：{e}")
            return []

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
            if content
        ]

    def _current_knowledge_files(self) -> tuple[str, ...]:
        return listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )

    def load_document(self):
        manifest = self._load_manifest()
        current_paths = self._current_knowledge_files()
        current_source_names = {os.path.basename(path) for path in current_paths}
        self._sync_stale_chunks(manifest, current_source_names)

        for path in current_paths:
            source_name = os.path.basename(path)
            file_md5 = get_file_md5_hex(path)
            if not file_md5:
                continue

            old_record = manifest.get("files", {}).get(source_name)
            if old_record and old_record.get("md5") == file_md5:
                logger.info(f"[加载知识库]{path}内容未变化，跳过")
                continue

            try:
                if old_record:
                    self._delete_file_chunks(source_name)

                documents = self._get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}无有效内容，跳过")
                    continue

                chunks = self._split_documents_by_type(documents, path)
                chunks = self._enrich_and_dedup_chunks(chunks, path, file_md5)
                if not chunks:
                    logger.warning(f"[加载知识库]{path}切片后无有效内容，跳过")
                    continue

                self.vector_store.add_documents(chunks)
                manifest.setdefault("files", {})[source_name] = {
                    "md5": file_md5,
                    "chunk_count": len(chunks),
                    "path": path,
                }
                logger.info(f"[加载知识库]{path}已加载，切片数：{len(chunks)}")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

        self._save_manifest(manifest)
        self.bm25_docs = self._load_bm25_docs_from_vector_store()

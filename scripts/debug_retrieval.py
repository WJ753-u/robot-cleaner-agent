from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import VectorStoreService


def main():
    query = " ".join(sys.argv[1:]).strip() or "扫地机器人怎么保养"
    service = VectorStoreService()
    docs = service.get_retriever().invoke(query)

    print(f"query: {query}")
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source_name = metadata.get("source_name") or metadata.get("knowledge_file") or metadata.get("source", "未知来源")
        chunk_id = metadata.get("chunk_id", "未知片段")
        print("-" * 80)
        print(f"{index}. source: {source_name} | chunk_id: {chunk_id} | metadata: {metadata}")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()

"""
法律知识库检索器模块

功能：
- 加载 docs/ 目录下的 PDF 法律文档
- 构建 BM25 + FAISS 混合检索（FAISS 下载失败时降级为纯 BM25）
- CrossEncoder 重排序（模型不可用时跳过）
- LLM 查询改写
- 索引持久化到 storage/ 目录
"""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.common.llm import model

# ============ 常量 ============

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
STORAGE_DIR = PROJECT_ROOT / "storage"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
TOP_K = 5

_HF_UNAVAILABLE_MSG = (
    "\n[retriever] ============================================================\n"
    "[retriever] 无法从 HuggingFace 下载模型，请设置镜像后重试：\n"
    "[retriever]   在 .env 文件中添加：HF_ENDPOINT=https://hf-mirror.com\n"
    "[retriever]   然后重启 langgraph dev\n"
    "[retriever]   当前降级为纯 BM25 检索（无向量检索，无重排序）\n"
    "[retriever] ============================================================\n"
)


# ============ 查询改写检索器 ============

class RewrittenRetriever(BaseRetriever):
    """在检索前用 LLM 将口语化查询改写为更适合法律条文检索的查询"""

    retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        rewritten = _rewrite_query(query)
        return self.retriever.invoke(rewritten)


def _rewrite_query(query: str) -> str:
    """用 LLM 将用户查询改写为法律检索优化的查询"""
    system_message = SystemMessage(
        content="""你是一个法律检索查询优化专家。请将用户的口语化查询改写为更适合法律条文检索的查询。

改写规则：
1. 将口语化表达转换为法律术语（如"房东不退押金"→"押金退还 承租人权利"）
2. 补充相关法律概念关键词
3. 保留原始查询的核心意图
4. 只输出改写后的查询文本，不要解释

示例：
- "提前退租要赔多少钱" → "提前解除租赁合同 违约金 赔偿责任"
- "房子漏水谁来修" → "租赁物维修义务 出租人责任"
- "能不能转租给别人" → "转租 承租人权利 出租人同意" """
    )
    human_message = HumanMessage(content=f"请改写以下查询：{query}")
    response = model.invoke([system_message, human_message])
    return response.content.strip()


# ============ 文档加载 ============

def _load_docs(docs_dir: Path) -> List[Document]:
    """加载 docs_dir 下的所有 PDF 文件"""
    from langchain_community.document_loaders import PyMuPDFLoader

    docs = []
    if not docs_dir.exists():
        print(f"[retriever] docs 目录不存在: {docs_dir}")
        return docs

    pdf_files = list(docs_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"[retriever] docs 目录下未找到 PDF 文件: {docs_dir}")
        return docs

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_path.name
            docs.extend(pages)
            print(f"[retriever] 已加载: {pdf_path.name} ({len(pages)} 页)")
        except Exception as e:
            print(f"[retriever] 加载失败: {pdf_path.name}, 错误: {e}")

    return docs


def _compute_docs_hash(docs_dir: Path) -> str:
    """计算 docs 目录下所有 PDF 文件的哈希，用于检测变更"""
    if not docs_dir.exists():
        return ""

    pdf_files = sorted(docs_dir.glob("**/*.pdf"))
    hasher = hashlib.md5()
    for pdf_path in pdf_files:
        hasher.update(pdf_path.name.encode())
        hasher.update(str(pdf_path.stat().st_mtime_ns).encode())
        hasher.update(str(pdf_path.stat().st_size).encode())
    return hasher.hexdigest()


# ============ 构建检索器 ============

def build_law_retriever(
    docs_dir: Optional[str] = None,
    storage_dir: Optional[str] = None,
) -> BaseRetriever:
    """
    构建法律知识库检索器。

    如果 storage/ 下已有索引且 docs 未变更，直接加载；
    否则从 docs/ 重新构建索引并持久化。

    模型下载失败时自动降级：
    - 无 embedding → 纯 BM25 检索
    - 无 reranker → 跳过重排序
    """
    docs_path = Path(docs_dir) if docs_dir else DOCS_DIR
    storage_path = Path(storage_dir) if storage_dir else STORAGE_DIR
    faiss_path = storage_path / "faiss_index"
    bm25_path = storage_path / "bm25.pkl"
    hash_path = storage_path / "docs_hash.json"

    # 尝试初始化 embedding 模型（国内网络可能无法直连 HuggingFace）
    embedding_model = None
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        print(_HF_UNAVAILABLE_MSG, file=sys.stderr)

    current_hash = _compute_docs_hash(docs_path)
    saved_hash = None
    if hash_path.exists():
        saved_hash = json.loads(hash_path.read_text(encoding="utf-8")).get("hash")

    vector_store = None
    if current_hash and faiss_path.exists() and bm25_path.exists() and current_hash == saved_hash:
        print("[retriever] 检测到已有索引且文档未变更，直接加载...")
        if embedding_model is not None:
            try:
                vector_store = FAISS.load_local(
                    str(faiss_path), embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[retriever] 加载 FAISS 索引失败: {e}，降级为纯 BM25")
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("[retriever] 构建新索引...")
        docs = _load_docs(docs_path)
        if not docs:
            print("[retriever] 警告：无文档可加载，返回空检索器")
            bm25_retriever = BM25Retriever.from_documents(
                [Document(page_content="暂无法律条文", metadata={})]
            )
            bm25_retriever.k = TOP_K
            return RewrittenRetriever(retriever=bm25_retriever)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
        print(f"[retriever] 文档分块完成: {len(chunks)} 个chunk")

        if embedding_model is not None:
            try:
                vector_store = FAISS.from_documents(chunks, embedding_model)
                vector_store.save_local(str(faiss_path))
            except Exception as e:
                print(f"[retriever] FAISS 索引构建失败: {e}，降级为纯 BM25")

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = TOP_K
        try:
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
            storage_path.mkdir(parents=True, exist_ok=True)
            hash_path.write_text(json.dumps({"hash": current_hash}), encoding="utf-8")
            print("[retriever] 索引已持久化到 storage/")
        except Exception as e:
            print(f"[retriever] 索引持久化失败: {e}")

    bm25_retriever.k = TOP_K

    # 构建检索链路
    if vector_store is not None:
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )
        base_retriever = ensemble_retriever
    else:
        base_retriever = bm25_retriever

    # 尝试初始化 reranker
    reranked_retriever = base_retriever
    if embedding_model is not None:
        try:
            reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
            reranker = CrossEncoderReranker(model=reranker_model, top_n=TOP_K)
            reranked_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=base_retriever
            )
        except Exception:
            print("[retriever] reranker 模型不可用，跳过重排序")

    return RewrittenRetriever(retriever=reranked_retriever)


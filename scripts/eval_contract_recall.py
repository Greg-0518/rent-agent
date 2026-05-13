# -*- coding: utf-8 -*-
"""合同法律检索召回率评估 — 四种策略对比，30条分层测试集。
运行：python scripts/eval_contract_recall.py"""

import os; os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, hashlib
from pathlib import Path
from typing import List

import jieba

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE = PROJECT_ROOT / "storage"

# 用于 query rewriting 的 LLM
_rewrite_llm = None


def _get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = init_chat_model(model="deepseek-chat", temperature=0)
    return _rewrite_llm


class RewriteRetriever(BaseRetriever):
    """检索前用 LLM 将口语化查询改写为法律术语"""
    retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        system = SystemMessage(content="""你是法律检索查询优化专家。将口语化查询改写为法律术语检索词。

规则：
1. 口语→法律术语（"房东不退押金"→"押金退还 承租人权利 租赁物返还"）
2. 补充相关法律概念关键词
3. 保留核心意图
4. 只输出改写后的查询文本，不要解释""")
        try:
            resp = _get_rewrite_llm().invoke([system, HumanMessage(content=query)])
            rewritten = resp.content.strip()
        except Exception:
            rewritten = query
        return self.retriever.invoke(rewritten)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_retrievers():
    """从 storage 加载四种检索器"""
    faiss_path = STORAGE / "faiss_index"
    bm25_path = STORAGE / "bm25.pkl"

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(str(faiss_path), embedding, allow_dangerous_deserialization=True)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = 5
    bm25.preprocess_func = lambda text: [w for w in jieba.cut(text) if w.strip()]

    bm25_docs = [bm25.docs[i] for i in range(len(bm25.docs))]
    text_to_idx = {_md5(d.page_content): i for i, d in enumerate(bm25_docs)}

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.3, 0.7])

    reranked = None
    try:
        cmodel = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        reranker = CrossEncoderReranker(model=cmodel, top_n=5)
        reranked = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble)
    except Exception as e:
        print(f"[WARN] reranker 不可用: {e}")

    rewrite_rerank = RewriteRetriever(retriever=reranked) if reranked else None

    return {
        "vector_only": vector_retriever,
        "bm25_only": bm25,
        "hybrid": ensemble,
        "hybrid_rerank": reranked,
        "hybrid_rerank_rewrite": rewrite_rerank,
    }, text_to_idx, bm25_docs


def create_eval_dataset():
    """
    30条分层测试集，手工标注 based on 逐条阅读 storage/bm25.pkl 的 chunk 内容。

    关键 chunk 索引（民法典第十四章「租赁合同」第 703-760 条）：
      137: 第703-706条（租赁合同定义，内容要素，20年上限）
      138: 第707-714条（书面形式，交付，使用方法，损耗免责710，过错损坏711，维修义务712/713，保管义务714）
      139: 第715-720条（改善715，转租716-718，次承租人代付719，收益归属720）
      140: 第721-726条（租金支付期限721/催告解除722，第三人主张权利723，承租人解除情形724，
                      买卖不破租赁725，优先购买权726）
      141: 第727-732条（拍卖通知727，妨害优先购买728，不可归责事由729，
                      不定期租赁730，安全解约731，死亡续租732）
      142: 第733-734条（期满返还733，续租/优先承租734）

    合同通则（违约责任 & 违约金）：
       92: 第577-580条（违约责任种类，实际履行）
      104: 第563条（合同解除事由）
      105: 第563-565条（合同解除程序、期限）
      116: 第585-588条（违约金、定金）
      121: 第561-563条（债务抵充 / 合同解除一般规定）
      122: 第565-566条（解除通知、解除后责任）

    测试集设计原则：
      - 口语化查询（semantic）：测试向量/混合检索的语义理解能力
      - 法言法语查询（keyword）：测试 BM25 的精确关键词匹配能力
      - 法条编号查询（exact）：测试精确检索能力

    【重点】口语化查询对 BM25 不公平：民法典中不存在"押金""漏水""房东"等词，
    这些词全部为法律术语替代表达（如"押金"→"保证金"/"保管物"、"房东"→"出租人"）。
    """
    return [
        # ────────── A 组：口语化查询（主打向量的语义优势）──────────
        {
            "query": "房东不退押金怎么办",
            "relevant_idx": [138, 139],
            "tags": ["semantic"],
            "reason": "民法典无'押金'词→BM25不可达。714条承租人保管义务+711条过错损坏赔偿，作为退还/抵扣押金的法理基础。chunk138+139",
        },
        {
            "query": "房屋漏水谁负责维修",
            "relevant_idx": [138],
            "tags": ["semantic"],
            "reason": "712条出租人维修义务+713条自行维修报销权。chunk138",
        },
        {
            "query": "合同没到期想退租要赔多少钱",
            "relevant_idx": [92, 116, 141],
            "tags": ["semantic"],
            "reason": "577条违约一般规定(ch92)+585条违约金不超过损失30%(ch116)+730条不定期租赁随时解除(ch141)",
        },
        {
            "query": "租客拖欠房租多久房东可以把他赶走",
            "relevant_idx": [140, 139],
            "tags": ["semantic"],
            "reason": "722条催告后逾期不付可解除(ch140)+719条次承租人可代付租金避免解除(ch139)",
        },
        {
            "query": "租房期间房东把房子卖了新房东让我搬走怎么办",
            "relevant_idx": [140, 142],
            "tags": ["semantic"],
            "reason": "725条买卖不破租赁(ch140)+734条期满优先承租权(ch142)。'买卖不破租赁'为法律格言",
        },
        {
            "query": "租房必须签书面合同吗口头说好租房行不行",
            "relevant_idx": [138],
            "tags": ["semantic"],
            "reason": "707条：6个月以上须书面，否则视为不定期租赁。chunk138",
        },
        {
            "query": "租客擅自把房子转租给朋友怎么处理",
            "relevant_idx": [139],
            "tags": ["semantic"],
            "reason": "716条：未经同意转租→出租人可解除合同。718条：知转租6月未异议视为同意。chunk139",
        },
        {
            "query": "装修花了钱退租的时候能找房东要补偿吗",
            "relevant_idx": [139],
            "tags": ["semantic"],
            "reason": "715条改善增设规则：经同意可改善(ch138)。补偿需依不当得利或合同约定。chunk139",
        },
        {
            "query": "出租的房子有安全隐患甲醛超标可以退租吗",
            "relevant_idx": [141, 138],
            "tags": ["semantic"],
            "reason": "731条安全危及随时解除(ch141)+708条出租人应保持租赁物符合约定用途(ch138)",
        },
        {
            "query": "正常住着东西旧了坏了需要我赔吗",
            "relevant_idx": [138],
            "tags": ["semantic"],
            "reason": "710条正常损耗免责+714条保管不善才需赔偿。chunk138",
        },

        # ────────── B 组：法言法语查询（测试关键词匹配，BM25 应能命中）──────────
        {
            "query": "出租人的维修义务",
            "relevant_idx": [138],
            "tags": ["keyword"],
            "reason": "712条：出租人应当履行租赁物的维修义务。关键词'维修义务'直接命中。chunk138",
        },
        {
            "query": "承租人转租需要出租人同意吗 转租条件",
            "relevant_idx": [139],
            "tags": ["keyword"],
            "reason": "716条+718条均在chunk139，关键词'转租''同意'直接匹配。chunk139",
        },
        {
            "query": "租赁合同的租赁期限不得超过多少年",
            "relevant_idx": [137, 138],
            "tags": ["keyword"],
            "reason": "705条：20年(ch137)+707条：6个月以上书面形式(ch138)。关键词'租赁期限'直接匹配",
        },
        {
            "query": "承租人优先购买权 出租人出卖租赁房屋",
            "relevant_idx": [140, 141],
            "tags": ["keyword"],
            "reason": "726条优先购买权(ch140)+727条拍卖通知(ch141)。关键词'优先购买'命中",
        },
        {
            "query": "租赁合同解除的条件 承租人解除权",
            "relevant_idx": [140, 141, 142],
            "tags": ["keyword"],
            "reason": "724条承租人解除事由(ch140)+730条不定期解除(ch141)+731条安全解除(ch141)。关键词'解除'匹配",
        },
        {
            "query": "违约金 违约损害赔偿 合同法",
            "relevant_idx": [92, 116],
            "tags": ["keyword"],
            "reason": "577条违约责任(ch92)+585条违约金调整(ch116)。关键词'违约金''违约责任'直接命中",
        },
        {
            "query": "不定期租赁 租赁期限届满后的续租 优先承租权",
            "relevant_idx": [142, 141],
            "tags": ["keyword"],
            "reason": "734条优先承租权+期满续租(ch142)+730条不定期租赁(ch141)",
        },
        {
            "query": "承租人的保管义务 租赁物毁损灭失的赔偿责任",
            "relevant_idx": [138, 139],
            "tags": ["keyword"],
            "reason": "714条保管义务(ch138)+715条改善增设(ch139)+711条过错损坏赔偿(ch138)",
        },
        {
            "query": "融资租赁合同 承租人的权利义务 租赁物归属",
            "relevant_idx": [142, 143, 144, 145],
            "tags": ["keyword"],
            "reason": "第十五章融资租赁735-760条，分布在chunk142-146。关键词'融资租赁'命中",
        },
        {
            "query": "承租人经催告后在合理期限内仍不支付租金的",
            "relevant_idx": [140, 145],
            "tags": ["keyword"],
            "reason": "722条租赁合同催告解除(ch140)+752条融资租赁催告解除(ch145)。民法典原文摘录式查询",
        },

        # ────────── C 组：法条编号查询（精确匹配，两种检索器都应命中）──────────
        {
            "query": "民法典 第七百一十二条 第七百一十三条",
            "relevant_idx": [138],
            "tags": ["exact"],
            "reason": "直接检索法条号。712+713条均在chunk138（租赁物维修义务）。",
        },
        {
            "query": "民法典 第七百一十六条 转租",
            "relevant_idx": [139],
            "tags": ["exact"],
            "reason": "716条：转租需要出租人同意。chunk139",
        },
        {
            "query": "民法典 第七百二十二条 承租人支付租金",
            "relevant_idx": [140],
            "tags": ["exact"],
            "reason": "722条：租金支付催告。chunk140",
        },
        {
            "query": "民法典 第七百零三条 第七百零四条 第七百零五条",
            "relevant_idx": [137],
            "tags": ["exact"],
            "reason": "租赁合同定义+内容要素+20年上限，全部在chunk137",
        },
        {
            "query": "民法典 第五百七十七条 违约责任 第五百八十五条 违约金",
            "relevant_idx": [92, 116],
            "tags": ["exact"],
            "reason": "577条违约责任(ch92)+585条违约金(ch116)。法条号+关键词双重命中",
        },

        # ────────── D 组：综合复杂查询（2-3个chunk，法言法语）──────────
        {
            "query": "承租人违反支付租金义务 出租人解除租赁合同 违约金赔偿",
            "relevant_idx": [140, 92, 116],
            "tags": ["keyword", "complex"],
            "reason": "722条租金拖欠解除(ch140)+577条违约(ch92)+585条违约金(ch116)。三重法律关系的复杂查询",
        },
        {
            "query": "因不可归责于承租人的事由租赁物毁损灭失 租金减免",
            "relevant_idx": [141, 140],
            "tags": ["keyword", "complex"],
            "reason": "729条不可归责减租/解除(ch141)+723条第三人主张权利减租(ch140)。跨chunk的法律关系链",
        },
        {
            "query": "未经出租人同意转租的后果 出租人的解除权",
            "relevant_idx": [139, 92],
            "tags": ["keyword", "complex"],
            "reason": "716条转租解除(ch139)+563条合同解除权一般规定。租赁特殊规定+合同通则的组合",
        },
        {
            "query": "租赁物危及承租人安全健康 承租人解除合同 损害赔偿",
            "relevant_idx": [141, 138, 92],
            "tags": ["keyword", "complex"],
            "reason": "731条安全解除(ch141)+708条适租义务(ch138)+584条损害赔偿范围。三个法律层次的交叉",
        },
        {
            "query": "承租人死亡 共同居住人继续租赁权",
            "relevant_idx": [141, 142],
            "tags": ["keyword"],
            "reason": "732条共同居住人继续租赁权(ch141)+734条优先承租权(ch142)。生前共同居住人权益链",
        },
    ]


# ============ 指标 ============

def recall_at_k(retrieved, relevant, k=5):
    r, rel = set(retrieved[:k]), set(relevant)
    return len(r & rel) / len(rel) if rel else 0.0

def mrr(retrieved, relevant):
    rel = set(relevant)
    for i, rid in enumerate(retrieved, 1):
        if rid in rel:
            return 1.0 / i
    return 0.0

def hit_rate(retrieved, relevant, k=5):
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def evaluate(retriever, name, eval_data, text_to_idx, k=5):
    print(f"\n{'='*60}")
    print(f"策略：{name}")
    print(f"{'='*60}")

    results = []
    for i, item in enumerate(eval_data):
        query = item["query"]
        relevant = item["relevant_idx"]
        tags = item.get("tags", [])

        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = []

        retrieved_idx = []
        for d in docs:
            key = _md5(d.page_content)
            if key in text_to_idx:
                retrieved_idx.append(text_to_idx[key])

        r = recall_at_k(retrieved_idx, relevant, k)
        m = mrr(retrieved_idx, relevant)
        h = hit_rate(retrieved_idx, relevant, k)
        tag_str = "/".join(tags)

        mark = "✓" if h else "✗"
        print(f"  {mark} [{i+1:02d}] {query[:40]}...  R@5={r:.2f}  MRR={m:.2f}  [{tag_str}]")
        results.append({"recall": r, "mrr": m, "hit": h, "tags": tags})

    n = len(eval_data)
    avg_r = sum(x["recall"] for x in results) / n
    avg_m = sum(x["mrr"] for x in results) / n
    avg_h = sum(x["hit"] for x in results) / n

    # 按标签分组统计
    by_tag = {}
    for x in results:
        for t in x["tags"]:
            by_tag.setdefault(t, []).append(x)

    print(f"  {'─'*52}")
    print(f"  全部: Recall@5={avg_r:.2%}  MRR={avg_m:.2%}  HitRate@5={avg_h:.2%}")
    for tag in sorted(by_tag):
        items = by_tag[tag]
        tr = sum(x["recall"] for x in items) / len(items)
        tm = sum(x["mrr"] for x in items) / len(items)
        th = sum(x["hit"] for x in items) / len(items)
        print(f"    [{tag:<10}] n={len(items):<3} Recall@5={tr:.2%}  MRR={tm:.2%}  HitRate@5={th:.2%}")

    return {"name": name, "Recall@5": round(avg_r, 4), "MRR": round(avg_m, 4),
            "HitRate@5": round(avg_h, 4), "by_tag": {t: round(sum(x["recall"] for x in v)/len(v), 4) for t, v in by_tag.items()}}


def main():
    print("加载检索器...")
    retrievers, text_to_idx, _ = load_retrievers()
    eval_data = create_eval_dataset()
    n_sem = len([d for d in eval_data if 'semantic' in d['tags']])
    n_kw  = len([d for d in eval_data if 'keyword' in d['tags']])
    n_ex  = len([d for d in eval_data if 'exact' in d['tags']])
    print(f"BM25 chunk 数: {len(text_to_idx)}，测试集: {len(eval_data)} 条")
    print(f"标签分布: {n_sem} semantic, {n_kw} keyword, {n_ex} exact\n")

    all_results = []
    for name in ["vector_only", "bm25_only", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite"]:
        r = retrievers[name]
        if r is None:
            continue
        all_results.append(evaluate(r, name, eval_data, text_to_idx))

    print(f"\n{'='*60}")
    print("最终汇总对比")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'Recall@5':>10} {'MRR':>10} {'HitRate@5':>10}  semantic  keyword  exact")
    print("-" * 78)
    for r in all_results:
        bt = r.get("by_tag", {})
        print(f"{r['name']:<20} {r['Recall@5']:>10.2%} {r['MRR']:>10.2%} {r['HitRate@5']:>10.2%}  "
              f"{bt.get('semantic',0):>7.1%}  {bt.get('keyword',0):>7.1%}  {bt.get('exact',0):>7.1%}")
    print()


if __name__ == "__main__":
    main()

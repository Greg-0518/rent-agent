"""
合同审核子图

流程：START -> clause_extraction -> law_retrieval -> risk_analysis -> report_generation -> END
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.common.retriever import build_law_retriever
from src.agent.node.contract import (
    clause_extraction_node,
    make_law_retrieval_node,
    risk_analysis_node,
    report_generation_node,
)
from src.agent.state.contract import ContractState

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_law_retriever()
    return _retriever


def build_contract_graph():
    """
    构建合同审核子图。

    Returns:
        compiled 的 ContractState 子图
    """
    builder = StateGraph(ContractState)
    builder.add_node("clause_extraction", clause_extraction_node)
    builder.add_node("law_retrieval", make_law_retrieval_node(_get_retriever))
    builder.add_node("risk_analysis", risk_analysis_node)
    builder.add_node("report_generation", report_generation_node)

    builder.add_edge(START, "clause_extraction")
    builder.add_edge("clause_extraction", "law_retrieval")
    builder.add_edge("law_retrieval", "risk_analysis")
    builder.add_edge("risk_analysis", "report_generation")
    builder.add_edge("report_generation", END)

    return builder.compile()


contract_graph = build_contract_graph()

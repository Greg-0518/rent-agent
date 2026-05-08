"""
合同审核Agent状态定义
"""

from typing import List, Optional, TypedDict, Annotated
from langgraph.graph import MessagesState


class Clause(TypedDict):
    """合同条款"""
    clause_type: str  # 条款类型：租金、押金、租期等
    content: str  # 条款内容
    position: str  # 位置信息


class RiskAnalysis(TypedDict):
    """风险分析结果"""
    clause_type: str        # 条款类型
    risk_level: str         # 高/中/低
    risk_description: str   # 风险描述
    legal_basis: str        # 法律基础
    suggestion: str         # 建议


class ContractState(MessagesState):
    """合同审核状态"""
    # 合同原文
    contract_text: str = ""
    
    # 提取的条款
    extracted_clauses: List[Clause] = []
    
    # 检索到的法律条文
    retrieved_laws: List[dict] = []
    
    # 风险分析结果
    risk_analysis: List[RiskAnalysis] = []
    
    # 审核报告
    audit_report: str = ""

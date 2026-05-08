from typing import TypedDict

from langgraph.graph import MessagesState


class State(MessagesState):
    user_intent: str           # 用户意图
    user_preferences: dict     # 用户偏好
    contract_text: str         # 合同原文
    audit_report: str          # 审核报告

class NeedReserveOutput(TypedDict):
    reserve: str

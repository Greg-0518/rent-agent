from typing import TypedDict

from langgraph.graph import MessagesState


class State(MessagesState):
    user_intent: str           # 用户意图
    user_preferences: dict     # 用户偏好

class NeedReserveOutput(TypedDict):
    reserve: str

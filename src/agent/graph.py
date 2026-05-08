"""租房助手主图 — 意图识别 → 路由分发"""

from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from src.agent.common.content import ContextSchema
from src.agent.contract import contract_graph
from src.agent.extend import extend_graph
from src.agent.node.main import (
    get_store_info,
    identify_question,
    need_reserve,
    get_user_preferences,
    set_contract_text,
)
from src.agent.recommend import recommended_graph
from src.agent.reserve import reserve_graph
from src.agent.state.main import State, NeedReserveOutput

# 构建图
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(get_store_info)
builder.add_node(identify_question)
builder.add_node("recommended_graph", recommended_graph)
builder.add_node(need_reserve)
builder.add_node("reserve_graph", reserve_graph)
builder.add_node("extend_graph", extend_graph)
builder.add_node("contract_graph", contract_graph)
builder.add_node(get_user_preferences)
builder.add_node(set_contract_text)

builder.add_edge(START, "get_store_info")
builder.add_edge("get_store_info", "identify_question")


def router_message(state: State) -> Literal[
    "recommended_graph", "reserve_graph", "set_contract_text", "extend_graph", "get_user_preferences"
]:
    user_intent = state["user_intent"]
    if user_intent == "recommend_house":
        return "recommended_graph"
    elif user_intent == "reserve_house":
        return "reserve_graph"
    elif user_intent == "contract_audit":
        return "set_contract_text"
    elif user_intent == "get_info":
        return "get_user_preferences"
    else:
        return "extend_graph"


builder.add_conditional_edges(
    "identify_question",
    router_message,
    ["recommended_graph", "reserve_graph", "set_contract_text", "extend_graph", "get_user_preferences"]
)
# 合同审核：提取文本 → 审核子图
builder.add_edge("set_contract_text", "contract_graph")
# 推荐房源 → 中断询问是否预定
builder.add_edge("recommended_graph", "need_reserve")


def should_reserve(state: NeedReserveOutput) -> Literal[END, "reserve_graph"]:
    reserve = state["reserve"]
    if reserve == '需要':
        return "reserve_graph"
    else:
        return END


builder.add_conditional_edges(
    "need_reserve",
    should_reserve,
    [END, "reserve_graph"]
)
# 路由2：预定房源
builder.add_edge("reserve_graph", END)
# 路由3：合同审核
builder.add_edge("contract_graph", END)
# 路由4：查询我的信息
builder.add_edge("get_user_preferences", END)
# 路由5：其它
builder.add_edge("extend_graph", END)

graph = builder.compile()

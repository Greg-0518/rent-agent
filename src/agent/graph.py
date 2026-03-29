"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from src.agent.common.content import ContextSchema
from src.agent.extend import extend_graph
from src.agent.node.main import get_store_info, identify_question, need_reserve, get_user_preferences
from src.agent.recommend import recommended_graph
from src.agent.reserve import reserve_graph
from src.agent.state.main import State, NeedReserveOutput

# 构建图
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(get_store_info)
# 查询持久化消息
builder.add_node(identify_question)
# 识别⽤⼾输⼊的问题
builder.add_node("recommended_graph", recommended_graph)
# 推荐房源⼦图, 注意需要指定名称
builder.add_node(need_reserve)
# 是否需要预定房源
builder.add_node("reserve_graph", reserve_graph)
# 预定房源⼦图
builder.add_node("extend_graph", extend_graph)
# 待扩展⼦图
builder.add_node(get_user_preferences)
builder.add_edge(START, "get_store_info")
builder.add_edge("get_store_info", "identify_question")


# 识别问题
def router_message(state: State) -> Literal["recommended_graph", "reserve_graph", "extend_graph", "get_user_preferences"]:
    user_intent = state["user_intent"]
    if user_intent == "recommend_house":
        return "recommended_graph"
    elif user_intent == "reserve_house":
        return "reserve_graph"
    elif user_intent == "get_info":
        return "get_user_preferences"
    else:
        return "extend_graph"


builder.add_conditional_edges(
    "identify_question",
    router_message,  # 消息路由。
    ["recommended_graph", "reserve_graph", "extend_graph", "get_user_preferences"]
)
# 路由1：推荐房源-》（中断询问）预定房源
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
    # 不需要预定就结束对话
    [END, "reserve_graph"]
)
# 路由2：预定房源
builder.add_edge("reserve_graph", END)
# 路由3：查询我的信息
builder.add_edge("get_user_preferences", END)
# 路由4：其它
builder.add_edge("extend_graph", END)

graph = builder.compile()

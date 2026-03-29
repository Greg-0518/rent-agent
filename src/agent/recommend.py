from typing import Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from src.agent.common.content import ContextSchema
from src.agent.node.recommend import (
    collect_user_info,
    list_tables,
    call_get_schema,
    get_schema_node,
    run_query_node,
    generate_query,
    check_query
)
from src.agent.state.recommend import RecommendState

# 构建图
builder = StateGraph(RecommendState, context_schema=ContextSchema)
builder.add_node(collect_user_info)  # 收集⽤⼾信息节点
builder.add_node(list_tables)  # 直接调用sql_db_list_tables工具
builder.add_node(call_get_schema)  # LLM绑定sql_db_schema⼯具，强制⼯具调⽤
builder.add_node("get_schema", get_schema_node)  # sql_db_schema⼯具
builder.add_node("run_query", run_query_node)  # sql_db_query⼯具
builder.add_node(generate_query)  # LLM绑定sql_db_query⼯具，非强制⼯具调⽤
builder.add_node(check_query)  # LLM绑定sql_db_query⼯具，强制⼯具调⽤

# 添加边
builder.add_edge(START, "collect_user_info")
builder.add_edge("collect_user_info", "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")


def should_continue(state: RecommendState):
    messages = state["messages"]
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        return END
    else:
        return "check_query"


builder.add_conditional_edges(
    "generate_query",
    should_continue,
    [END, "check_query"]
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

recommended_graph = builder.compile()
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent.node.reserve import (
    get_title,
    get_phone,
    get_id,
    add_reserve_message,
    call_orders,
    generate_orders
)
from src.agent.state.reserve import ReserveState

builder = StateGraph(ReserveState)
builder.add_node("get_title", get_title)
builder.add_node("get_phone", get_phone)
builder.add_node("get_id", get_id)
builder.add_node("add_reserve_message", add_reserve_message)
builder.add_node("call_orders", call_orders)
builder.add_node("tool_node", ToolNode([generate_orders]))

builder.add_edge(START, "get_title")
builder.add_edge("get_title", "get_phone")
builder.add_edge("get_phone", "get_id")
builder.add_edge("get_id", "add_reserve_message")
builder.add_edge("add_reserve_message", "call_orders")
builder.add_conditional_edges(
    "call_orders",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)
builder.add_edge("tool_node", "call_orders")
reserve_graph = builder.compile()

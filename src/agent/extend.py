from langchain_core.messages import AIMessage, SystemMessage
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState
from src.agent.common.llm import model


def extend_node(state: MessagesState):
    response = model.invoke(
        [SystemMessage(content="你是⼀个乐于助⼈的助⼿，可以根据历史对话进⾏回复。")]
        + state["messages"]
    )
    return {
        "messages": [response]
    }


extend_graph = (
    StateGraph(MessagesState)
    .add_node(extend_node)
    .add_edge(START, "extend_node")
    .compile()
)

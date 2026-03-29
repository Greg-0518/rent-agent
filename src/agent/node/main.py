from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, filter_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agent.common.content import ContextSchema
from src.agent.common.llm import model
from src.agent.state.main import State, NeedReserveOutput


class UserMessage(BaseModel):
    """用户提问的消息摘要"""
    type: Literal["recommend_house", "reserve_house", "get_info", "others"] = Field(
        description="根据⽤⼾问题描述判断问题类型：推荐房源、预定房源、获取信息、其他内容"
    )


# 节点：识别⽤⼾问题：预定、推荐、我的
def identify_question(state: State):
    def extract_info(messages) -> UserMessage:
        system_message = SystemMessage(
            content="""你是⼀个根据描述提取信息提取专家。请从⽤⼾的描述中提取⽤⼾想要咨询的相关信息。
                    严谨根据语义推断信息，但不能猜测或编造信息。"""
        )
        # 创建结构化提取模型
        return model.with_structured_output(schema=UserMessage).invoke([system_message] + messages)

    # 最新的⽤⼾消息
    user_question = state["messages"][-1].content
    user_message = extract_info([HumanMessage(content=user_question)])
    return {"user_intent": user_message.type}


# 节点：查询持久化消息
def get_store_info(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    # 搜索⽤⼾信息
    user_id = runtime.context.get("user_id")
    namespace = (user_id, "preferences")
    pref_result = store.search(namespace)
    if pref_result and pref_result[0]:
        return {"user_preferences": pref_result[0].value}
    else:
        return {}


# 节点：中断询问是否主要帮助预定房源
def need_reserve(state: State) -> NeedReserveOutput:
    prompt = f"已经为您推荐合适的房源，是否需要帮您预订房源？\n"
    prompt += "如果不需要，请输⼊'**不需要**'。\n"
    prompt += "如果需要，请输⼊'**需要**'。\n(注意输⼊其它值⽆效)"
    # 中断，等待⽤⼾输⼊
    answer = interrupt(prompt)
    return {"reserve": str(answer).strip()}


# 节点：返回⽤⼾偏好信息
def get_user_preferences(state: State):
    prefs = state.get("user_preferences", {})
    user_messages = filter_messages(state["messages"], include_types="human")
    # 格式化已预定过的信息
    reserved_list = prefs.get('reserved_info', [])
    if reserved_list:
        reserved_str = "\n"
        for i, item in enumerate(reserved_list, 1):
            reserved_str += f"{i}. 预定⼯单ID: {item.get('order_id')}, " \
                            f"房源标题: {item.get('title')}, " \
                            f"预定电话: {item.get('phone_number')}\n"
    else:
        reserved_str = "⽆"

    response = model.invoke([
        SystemMessage(content="""你是⼀个乐于助⼈的助⼿，可以根据⽤⼾偏好信息进⾏回复。
        如果有的偏好数据为空，不要猜测或编造数据。
        不要直接回复偏好数据是什么，要结合问题进⾏⽣动回复。
        如果问题与⽤⼾偏好数据⽆关，直接回复即可。"""),
        HumanMessage(
            content="⽤⼾的历史偏好信息如下："
                    f"1. 最低预算：{prefs.get('budget_min')}"
                    f"2. 最⾼预算：{prefs.get('budget_max')}"
                    f"3. 已预定过的信息：{reserved_str}"
        ),
        user_messages[-1]
    ])

    return {"messages": [response]}

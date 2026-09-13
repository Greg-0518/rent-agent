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
    type: Literal["recommend_house", "reserve_house", "get_info", "contract_audit",
                 "image_analysis", "rent_calc", "others"] = Field(
        # 这里原先只有一行「推荐房源、预定房源、获取信息……」，等于**七个标签一个都没定义**。
        # 后果实测过：统计型问句（「有多少套一居室」「哪个区最划算」）被分给了 get_info，
        # 而 get_info 的落点 get_user_preferences 只读用户偏好、不查房源库——
        # 模型只能拿世界知识作答（实测答"几百元到一千多元"，库里最便宜是 550 元）。
        # 评测里表现为整档 12 条判「不可达」。改前/改后对比见
        # docs/项目B_评测改动说明_R5.md §6：把 12 条强制路由进 recommend_house，
        # 12/12 全部能答对，所以这是**分类器缺定义**，不是缺一条产品链路。
        description=(
            "根据用户问题描述判断问题类型。各标签的边界（按此判断，不要望文生义）：\n"
            "- recommend_house：**所有需要查房源库才能回答的问题**。既包括找房/推荐"
            "（「有没有可以养宠物的房子」「帮我推荐10套」），**也包括统计、聚合、最值类**"
            "（「有多少套一居室」「哪套最便宜」「深圳哪个区租房最划算」"
            "「房源数量超过5套的区有哪些」）。判断依据是**答案要不要从房源表里查**，"
            "而不是问句里有没有「推荐」二字。\n"
            "- get_info：**只指查询用户自己在本系统里存过的信息**——我的预算、我的预约记录、"
            "我登记过的偏好。**不包含**对房源库的任何查询。\n"
            "- reserve_house：预定/预约房源，即要产生下单动作的。\n"
            "- rent_calc：租金试算、费用计算（月供、押付、通勤成本等）。\n"
            "- contract_audit：审查合同、看条款里有没有坑。\n"
            "- image_analysis：需要看图才能回答（房源图、户型图、合同截图）。\n"
            "- others：以上都不是的闲聊、寒暄、与租房无关的问题。"
        )
    )


# 节点：识别⽤⼾问题：预定、推荐、我的
def identify_question(state: State):
    def extract_info(messages) -> UserMessage:
        system_message = SystemMessage(
            content="""你是⼀个根据描述提取信息提取专家。请从⽤⼾的描述中提取⽤⼾想要咨询的相关信息。
                    严谨根据语义推断信息，但不能猜测或编造信息。
                    判断问题类型时严格按各类型的定义与边界，特别注意：
                    要从房源库里查数据才能回答的问题（含套数、均价、最值等统计类），
                    属于 recommend_house，不属于 get_info。"""
        )
        # 创建结构化提取模型
        return model.with_structured_output(schema=UserMessage).invoke([system_message] + messages)

    # 最新的⽤⼾消息
    user_question = state["messages"][-1].content
    user_message = extract_info([HumanMessage(content=user_question)])
    return {"user_intent": user_message.type}


# 节点：查询持久化消息
def get_store_info(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    # 防御性检查：确保 context 不为 None
    if runtime.context is None:
        return {"user_preferences": None}
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


# 节点：从用户消息中提取合同文本
def set_contract_text(state: State):
    user_messages = filter_messages(state["messages"], include_types="human")
    contract_text = user_messages[-1].content if user_messages else ""
    return {"contract_text": contract_text}

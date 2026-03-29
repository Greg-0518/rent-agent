import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import filter_messages, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agent.common.content import ContextSchema
from src.agent.common.llm import model
from src.agent.common.store import UserPreferences
from src.agent.state.recommend import RecommendState, get_recommend_info


class UserInfo(BaseModel):
    """用户的租房需求信息"""

    city: Optional[str] = Field(
        default=None,
        description="用户所在或想要租房的城市，例如：北京、上海、深圳、广州等"
    )
    district: Optional[str] = Field(
        default=None,
        description="⽤⼾想要租房的具体区域或⾏政区，例如：雁塔区、碑林区、海淀区"
    )
    budget_min: Optional[float] = Field(
        default=None,
        description="⽤⼾的最低预算，单位为元/⽉"
    )
    budget_max: Optional[float] = Field(
        default=None,
        description="⽤⼾的最⾼预算，单位为元/⽉"
    )
    room_type: Optional[str] = Field(
        default=None,
        description="房屋类型，例如：整租、合租、公寓、⼀室⼀厅、两室⼀厅"
    )
    orientation: Optional[str] = Field(
        default=None,
        description="房屋朝向，例如：朝南、朝北、东南、南北通透"
    )
    room_count: Optional[int] = Field(
        default=None,
        description="需要推荐的房屋数量"
    )
    others: Optional[str] = Field(
        default=None,
        description="特殊要求，例如：带阳台、独⽴卫⽣间、近地铁、可养宠物、有电梯等"
    )


def collect_user_info(state: RecommendState, runtime: Runtime[ContextSchema], *, store: BaseStore):
    """收集用户希望的推荐信息"""

    # 1.获取需要被解析的数据，最新的用户消息 + 用户的偏好数据
    user_messages = filter_messages(state["messages"], include_types="human")
    pref = state.get("user_preferences")
    if pref and (pref["budget_min"] or pref["budget_max"]):
        # 偏好中包含最高和最低的预算
        budget_min = pref["budget_min"]
        budget_max = pref["budget_max"]
        extract_messages = [
            HumanMessage(content="用户的历史偏好消息如下："
                                 f"1. 最低预算：{budget_min}"
                                 f"1. 最高预算：{budget_max}"),
            user_messages[-1]
        ]
    else:
        extract_messages = [user_messages[-1]]

    # 2、提取信息(LLM结构返回)
    def extract_info(messages) -> UserInfo:
        system_message = SystemMessage(
            content="""
            你是⼀个租房需求信息提取专家。请从⽤⼾的描述与历史信息中提取租房相关信息。
            如果⽤⼾历史偏好信息与最新⽤⼾消息冲突，以最新的⽤⼾消息为主。
            只提取⽤⼾明确提到的信息，不要猜测或推断。
            如果某个信息⽤⼾没有提到，就返回null。
            注意预算的单位可能是元/⽉、元/天等，请统⼀转换为元/⽉。
            如果⽤⼾提到价格范围，请分别提取最低和最⾼预算。
            如果⽤⼾提到推荐⼏套房，提取room_count字段。"""
        )
        # 创建结构胡提取模型
        return model.with_structured_output(schema=UserInfo).invoke([system_message] + messages)

    # 更新状态函数
    def update_state(current_state: dict, info: UserInfo) -> dict:
        if not info:
            return current_state

        user_info_dict = info.model_dump(exclude_none=True)
        current_state.update(user_info_dict)
        return current_state

    # 根据历史偏好和用户消息提取消息
    updated_state = {}
    extracted_info = extract_info(extract_messages)
    updated_state = update_state(updated_state, extracted_info)

    # 3.中断咨询推荐的必要参数，如城市、预算范围
    missing_info = []
    if not updated_state.get("city"):
        missing_info.append("**城市**")
    if updated_state.get("budget_min") is None or updated_state.get("budget_max") is None:
        missing_info.append("**预算范围**")

    if missing_info:
        prompt = f"为了给您推荐合适的房源，请提供以下信息：{','.join(missing_info)}和其他信息。\n"
        prompt += "如果您不想提供，你输出’**不提供**‘，我会根据已有信息为您推荐房源"
        # 中断，等待用户输入
        answer = interrupt(prompt)
        if str(answer).strip() == "不提供":
            if not updated_state.get("city"):
                updated_state['city'] = "随机城市"
            if not updated_state.get("budget_min"):
                updated_state['budget_min'] = 500.0
            if not updated_state.get("budget_max"):
                updated_state['budget_max'] = 3000.0
            if not updated_state.get("room_count"):
                updated_state['room_count'] = 6
            print(f"⽤⼾选择不提供信息，使⽤默认值: 城市={updated_state.get('city')}, "
                  f"预算={updated_state.get('budget_min')}-{updated_state.get('budget_max')}")
        else:
            # 用户提供了必要的选择
            user_response_message = HumanMessage(content=str(answer))
            extracted_info = extract_info([user_response_message])
            updated_state = update_state(updated_state, extracted_info)

    # 4.持久化处理，更新跨会话参数
    if updated_state.get('budget_min') or updated_state.get('budget_max'):
        user_id = runtime.context.get("user_id")
        namespace = (user_id, "preferences")
        # 先查询，获取key进行更新
        prefs_result = store.search(namespace)
        if len(prefs_result) == 0:
            # 没有持久化信息，新增
            prefs = UserPreferences(
                budget_min=updated_state.get("budget_min"),
                budget_max=updated_state.get("budget_max")
            )
            store.put(
                namespace,
                str(uuid.uuid4()),
                prefs.model_dump(exclude_none=True)
            )
            # 更新用户偏好
            updated_state["user_preferences"] = prefs.model_dump()
        else:
            # 有持久化信息，判断更新
            prefs = prefs_result[0].value
            store_min = prefs["budget_min"]
            store_max = prefs["budget_max"]
            cur_min = updated_state.get("budget_min")
            cur_max = updated_state.get("budget_max")
            update_min = False
            update_max = False
            if store_min and cur_min and cur_min < store_min:
                update_min = True
            elif not store_min and cur_min:
                update_min = True

            if store_max and cur_max and cur_max > store_max:
                update_max = True
            elif not store_max and cur_max:
                update_max = True

            if update_min or update_max:
                if update_min:
                    prefs['budget_min'] = cur_min
                    print(f"更新⽤⼾最低预算={cur_min}")
                if update_max:
                    prefs['budget_max'] = cur_max
                    print(f"更新⽤⼾最⾼预算={cur_max}")
                store.put(
                    namespace,
                    prefs_result[0].key,
                    prefs
                )
                # 更新用户偏好
                updated_state["user_preferences"] = prefs

    # 5. 准备最终消息并更新消息，确保消息列表中包含最新消息
    updated_state["messages"] = [HumanMessage(content=get_recommend_info(updated_state))]

    # 打印⽇志
    print(f"已收集⽤⼾信息: 城市={updated_state.get('city')}, "
          f"区域={updated_state.get('district')}, "
          f"预算={updated_state.get('budget_min')}-{updated_state.get('budget_max')}, "
          f"房间数={updated_state.get('room_count')}")

    return updated_state


# 使用.env环境变量
load_dotenv()
db_user = os.getenv('DB_USER')
print(db_user)
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
# 获取数据库⼯具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)

# 节点：获取表信息
get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')
get_schema_node = ToolNode([get_schema_tool], name='get_schema')

# 节点：执行SQL查询
run_query_tool = next(tool for tool in tools if tool.name == 'sql_db_query')
run_query_node = ToolNode([run_query_tool], name='run_query')

# 节点：获取全量表
def list_tables(state: RecommendState):
    # 1.调用llm，获取AIMessage(tool_calls)
    tool_call = {
        "name" : "sql_db_list_tables",
        "args": {},
        "id": "123654",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # 2.手动调用工具 -> sql_db_list_tables
    lis_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
    tool_message = lis_tables_tool.invoke(tool_call)

    # 3.整合结果
    response = AIMessage(content=f"可用的表：{tool_message.content}")
    return {
        "messages": [tool_call_message, tool_message, response]
    }

# 节点：强制创建⼀个获取表信息的⼯具调⽤
def call_get_schema(state: RecommendState):
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])   # AIMessage
    return {"messages": [response]}

def generate_query(state: RecommendState):
    generate_query_system_prompt = """
    您是⼀个设计⽤于与SQL数据库交互的代理。
    给定⼀个输⼊问题，创建⼀个语法正确的{dialect}查询来运⾏，然后查看查询的结果并返回答案。
    需要根据rows from table的⽰例设置真实查询的值。
    除⾮⽤⼾指定了他们希望获得的特定数量的⽰例，否则始终将查询限制为最多{top_k}个结果。
    您可以按相关列对结果排序，以返回最感兴趣的结果。不要查询特定表中的所有列，只查询给定问题的
    相关列。
    不要对数据库做任何DML语句（INSERT， UPDATE， DELETE， DROP等)。
    """
    system_prompt = generate_query_system_prompt.format(
        dialect=db.dialect,
        top_k=state.get("room_count", 5)
    )
    system_message = SystemMessage(content=system_prompt)

    llm_with_tools = model.bind_tools([run_query_tool])
    # 将用户信息也加入到查询条件中
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def check_query(state: RecommendState):
    check_query_system_prompt = """
    你是⼀个⾮常注重细节的SQL专家。仔细检查{dialect}查询中的常⻅错误，包括：
    -使⽤NULL值的NOT IN
    -在应该使⽤UNION ALL时使⽤UNION
    -使⽤BETWEEN表⽰独占范围
    -谓词中的数据类型不匹配
    -正确引⽤标识符
    -使⽤正确数量的函数参数
    -转换为正确的数据类型
    -使⽤合适的列进⾏连接
    如果存在上述任何错误，请重写查询。如果没有错误，只需复制原始查询即可。
    在运⾏此检查之后，您将调⽤适当的⼯具来执⾏查询。""".format(dialect=db.dialect)
    system_message = SystemMessage(content=check_query_system_prompt)

    # ⽣成⼈⼯⽤⼾消息进⾏检查
    # 上⼀个节点是generate_query。如果⾛到这，必定调⽤了⼯具。这样获取到的SQL是准确的。
    tool_call = state["messages"][-1].tool_calls[0]
    # 将SQL当作⽤⼾消息传⼊进⾏检查
    user_message = HumanMessage(content=tool_call["args"]["query"])

    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}











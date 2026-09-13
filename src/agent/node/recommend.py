import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import filter_messages, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src.agent.common.content import ContextSchema
from src.agent.common.llm import model
from src.agent.common.sql_guard import rejection_message, validate_sql
from src.agent.common.store import UserPreferences, read_preferences
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
    print("------------------    into collect_user info    ----------------------------------------------")
    # 1.获取需要被解析的数据，最新的用户消息 + 用户的偏好数据
    user_messages = filter_messages(state["messages"], include_types="human")
    # 必须走 read_preferences 正规化：state 里的 user_preferences 是 store 的原始
    # dict，键集合随写侧（`exclude_none=True`）而变，直接下标会在"用户第一次只说
    # 了预算上限"时 KeyError——且只在第二个会话暴露。见 common/store.py 的 docstring。
    pref = read_preferences(state.get("user_preferences"))
    if pref.budget_min or pref.budget_max:
        # 偏好中包含最高和最低的预算
        budget_min = pref.budget_min
        budget_max = pref.budget_max
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

    # 诊断：看看每次进入函数时 state 里已有哪些字段
    _existing = {k: state.get(k) for k in ("city","district","budget_min","budget_max","room_type","room_count")}
    print(f"[collect_user_info] ENTER  state 已有字段: {_existing}")
    print(f"[collect_user_info] ENTER  state messages 数: {len(state.get('messages',[]))}")
    # 显示最后一条用户消息的内容（用于判断是否是新 run 还是 resume）
    if user_messages:
        last = user_messages[-1].content
        print(f"[collect_user_info] 最后一条用户消息(前80字): {last[:80]}")

    # 从已有 state 中继承字段，避免中断恢复时丢失之前提取的值
    _state_fields = ("city", "district", "budget_min", "budget_max",
                     "room_type", "orientation", "room_count", "others")
    #updated_state = {k: state.get(k) for k in _state_fields if state.get(k) is not None}
    updated_state = {}
    extracted_info = extract_info(extract_messages)
    updated_state = update_state(updated_state, extracted_info)

    print(f"[collect_user_info] 本轮提取结果: {updated_state}")
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

    # if missing_info:
    #     # 检查用户本轮是否说了"不提供"
    #     last_user_msg = user_messages[-1].content if user_messages else ""
    #     if "不提供" in str(last_user_msg):
    #         if not updated_state.get("city"):
    #             updated_state['city'] = "随机城市"
    #         if not updated_state.get("budget_min"):
    #             updated_state['budget_min'] = 500.0
    #         if not updated_state.get("budget_max"):
    #             updated_state['budget_max'] = 3000.0
    #         if not updated_state.get("room_count"):
    #             updated_state['room_count'] = 6
    #         print(f"用户选择不提供信息，使用默认值: 城市={updated_state.get('city')}, "
    #               f"预算={updated_state.get('budget_min')}-{updated_state.get('budget_max')}")
    #     else:
    #         prompt = f"为了给您推荐合适的房源，请提供以下信息：{','.join(missing_info)}和其他信息。\n"
    #         prompt += "如果您不想提供，你输出’**不提供**’，我会根据已有信息为您推荐房源"
    #         # 用 Command 中断同时把已提取的值写入 state，恢复时不会丢失
    #         return Command(resume=prompt, update=updated_state)


    # 4.持久化处理，更新跨会话参数
    if (updated_state.get('budget_min') or updated_state.get('budget_max')) and store is not None:
        if runtime.context is None:
            user_id = "default"
        else:
            user_id = runtime.context.get("user_id", "default")
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
            #
            # 这里**同时保留两个表示**，别合并：
            #   prefs  —— store 的**原始 dict**，写回路径（下面就地改 + store.put）必须用它，
            #             因为 pydantic 会把 UserPreferences 没声明的键丢掉，而那正是
            #             reserve.py 存进来的 reserved_info。
            #   stored —— 正规化后的对象，**只用来读**预算，缺键即 None，不会 KeyError。
            prefs = prefs_result[0].value
            stored = read_preferences(prefs)
            store_min = stored.budget_min
            store_max = stored.budget_max
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

_tools = []
_DB_AVAILABLE = False
_db_error_msg = "数据库不可用"
# 只读账号优先（设计文档 §4.7）：即使 prompt 注入绕过了白名单，DB 权限仍是最后一道门
_ro_user = os.getenv('DB_RO_USER')
_ro_password = os.getenv('DB_RO_PASSWORD')
if _ro_user:
    _conn_user, _conn_password = _ro_user, _ro_password
else:
    _conn_user, _conn_password = db_user, db_password
    print("[WARN] 未配置 DB_RO_USER，SQL 将使用可写账号连接——建议为评测链路建只读账号")
try:
    db = SQLDatabase.from_uri(f"mysql+pymysql://{_conn_user}:{_conn_password}@{db_host}:{db_port}/{db_name}")
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    _tools = toolkit.get_tools()
    _DB_AVAILABLE = True
    for _t in _tools:
        print(_t.name)
except Exception as e:
    print(f"[WARN] 数据库连接失败，SQL 工具不可用: {e}")
    _tools = []

_DB_ERROR_RESPONSE = {"messages": [AIMessage(content=f"⚠️ {_db_error_msg}，请检查 MySQL 连接后重试。")]}

# 节点：获取表信息 / 执行SQL查询
_get_schema = next((t for t in _tools if t.name == 'sql_db_schema'), None)
_raw_get_query = next((t for t in _tools if t.name == 'sql_db_query'), None)


class _QueryInput(BaseModel):
    """sql_db_query 的入参 schema（与原工具保持一致）"""

    query: str = Field(description="要执行的 SQL 查询语句，必须是只读的 SELECT")


def _build_guarded_query_tool(raw_tool):
    """把 sql_guard 包到 sql_db_query 外面 —— 所有执行 SQL 的路径都必须过安全层。

    包出来的工具**名字和入参 schema 与原工具完全一致**（sql_db_query / {"query": str}），
    所以 generate_query / check_query 里的 bind_tools 和 ToolNode 都不用改：
    它们拿到的已经是带安全层的版本。
    """
    if raw_tool is None:
        return None

    @tool("sql_db_query", args_schema=_QueryInput)
    def guarded_query(query: str) -> str:
        """Input to this tool is a detailed and correct SQL query, output is a result from the database.
        If the query is not correct, an error message will be returned. If an error is returned, rewrite
        the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx'
        in 'field list', use sql_db_schema to query the correct table fields.
        只接受只读的 SELECT 查询；写操作会被安全层拒绝。"""
        verdict = validate_sql(query)
        if not verdict.allowed:
            print(f"[sql_guard] 拦截 SQL: rule={verdict.rule} | {verdict.reason}")
            return rejection_message(verdict)
        if verdict.rule == "forced_limit":
            print(f"[sql_guard] 强制注入/收紧 LIMIT → {verdict.sql}")
        return raw_tool.invoke({"query": verdict.sql})

    return guarded_query


_get_query = _build_guarded_query_tool(_raw_get_query)

if _get_schema and _get_query:
    get_schema_node = ToolNode([_get_schema], name='get_schema')
    run_query_node  = ToolNode([_get_query], name='run_query')
else:
    # 无数据库时用占位函数，避免 import 阶段崩溃
    def _no_db_warning(state):
        return {"messages": [AIMessage(content="数据库不可用，请检查 MySQL 连接")]}
    get_schema_node = _no_db_warning
    run_query_node  = _no_db_warning

# 节点：获取全量表
def list_tables(state: RecommendState):
    if not _DB_AVAILABLE:
        return _DB_ERROR_RESPONSE
    tool_call = {
        "name" : "sql_db_list_tables",
        "args": {},
        "id": "123654",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    lis_tables_tool = next((t for t in _tools if t.name == 'sql_db_list_tables'), None)
    tool_message = lis_tables_tool.invoke(tool_call)
    response = AIMessage(content=f"可用的表：{tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}

# 节点：强制创建⼀个获取表信息的⼯具调⽤
def call_get_schema(state: RecommendState):
    if not _DB_AVAILABLE:
        return _DB_ERROR_RESPONSE
    llm_with_tools = model.bind_tools([_get_schema], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate_query(state: RecommendState):
    if not _DB_AVAILABLE:
        return _DB_ERROR_RESPONSE
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
    llm_with_tools = model.bind_tools([_get_query])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def check_query(state: RecommendState):
    if not _DB_AVAILABLE:
        return _DB_ERROR_RESPONSE
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
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = HumanMessage(content=tool_call["args"]["query"])
    llm_with_tools = model.bind_tools([_get_query], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}











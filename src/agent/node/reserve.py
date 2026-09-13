import uuid
from typing import Annotated, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime, InjectedStore
from langgraph.types import interrupt

from src.agent.common.llm import model
from src.agent.common.store import ReservedInfo, UserPreferences
from src.agent.state.reserve import ReserveState

# 节点：获取⽤⼾预定房源
def get_title(state: ReserveState):
    print(f"[get_title] ENTER  state.title={state.get('title', '未设置')}  msg_count={len(state.get('messages',[]))}")
    # 优先从 state 里取（中断恢复后的新 run 也能拿到之前的值）
    if state.get("title"):
        print(f"[get_title] title 已有值，直接跳过: {state['title']}")
        return {}
    # 尝试从最后一条用户消息提取 title
    user_msgs = [m for m in state["messages"] if m.type == "human"]
    last_msg = user_msgs[-1].content if user_msgs else ""
    print(f"[get_title] 最后一条用户消息(前80字): {last_msg[:80]}")
    # 如果最后一条消息是纯文本（不含"需要"/"不需要"等控制词），就当成 title
    if last_msg and "请输入" not in last_msg and last_msg.strip() not in ("需要", "不需要", ""):
        print(f"[get_title] 从消息中提取 title: '{last_msg}'")
        return {"title": last_msg.strip()}
    # 兜底：中断问询
    title = interrupt("请输入要预订房源的名称")
    print(f"[get_title] interrupt returned: '{title}' type={type(title).__name__}")
    if title:
        return {"title": title}
    return {}

def isPhoneVaild(phone_number: str):
    if len(phone_number) != 11:
        return False, "手机号需要11位"

    # 检查是否全部为数字
    if not phone_number.isdigit():
        return False, "手机号只能包含数字"

    # 检查第一位是否为1
    if phone_number[0] != '1':
        return False, "手机号必须以1开头"

    # 检查第二位是否在3-9之间（常见号段）
    if phone_number[1] not in '3456789':
        return False, "手机号第二位必须在3-9之间"

    # 通过所有检查
    return True, ""

# 节点：获取⽤⼾预定电话
def get_phone(state: ReserveState):
    print(f"[get_phone] ENTER  state.title={state.get('title','?')}  phone_number={state.get('phone_number','未设置')}")
    if state.get("phone_number"):
        print(f"[get_phone] phone_number 已有值，直接跳过")
        return {}
    user_msgs = [m for m in state["messages"] if m.type == "human"]
    last_msg = user_msgs[-1].content.strip() if user_msgs else ""
    # 如果不是控制词且有数字，当成手机号
    if last_msg and last_msg not in ("需要", "不需要", "") and any(c.isdigit() for c in last_msg):
        cleaned = ''.join(c for c in last_msg if c.isdigit())
        if isPhoneVaild(cleaned)[0]:
            print(f"[get_phone] 从消息中提取 phone: '{cleaned}'")
            return {"phone_number": cleaned}
    phone_number = interrupt("请输⼊要预定的⼿机号")
    print(f"[get_phone] interrupt returned: '{phone_number}'")
    if phone_number and isPhoneVaild(phone_number.strip())[0]:
        return {"phone_number": phone_number.strip()}
    return {}

def is_id_card_valid(id_card: str) -> bool:
    """
    验证身份证号码是否有效（支持18位）
    规则：
    1. 长度为18位
    2. 前17位为数字
    3. 最后一位为数字或大写X
    4. 校验码验证（ISO 7064:1983.MOD 11-2）
    """
    # 去除空格并转换为大写
    id_card = id_card.strip().upper()

    # 长度检查
    if len(id_card) != 18:
        return False

    # 前17位必须为数字
    if not id_card[:17].isdigit():
        return False

    # 最后一位必须是数字或X
    if not (id_card[17].isdigit() or id_card[17] == 'X'):
        return False

    # 加权因子
    factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    # 校验码对应值（模11的结果映射）
    check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

    # 计算加权和
    total = sum(int(id_card[i]) * factors[i] for i in range(17))
    # 取模
    mod = total % 11
    # 验证校验码
    return check_codes[mod] == id_card[17]

# 节点：获取⽤⼾⾝份证
def get_id(state: ReserveState):
    print(f"[get_id] ENTER  state.id_card={state.get('id_card','未设置')}")
    if state.get("id_card"):
        print(f"[get_id] id_card 已有值，直接跳过")
        return {}
    user_msgs = [m for m in state["messages"] if m.type == "human"]
    last_msg = user_msgs[-1].content.strip() if user_msgs else ""
    # 如果不是控制词且长度合适（18位或含X），当成身份证号
    if last_msg and last_msg not in ("需要", "不需要", "") and (len(last_msg) == 18 or any(c.isdigit() for c in last_msg)):
        cleaned = last_msg.replace(" ", "").upper()
        if is_id_card_valid(cleaned):
            print(f"[get_id] 从消息中提取 valid id_card")
            return {"id_card": cleaned}
    id_card = interrupt("请输⼊要预定的⾝份证号码")
    print(f"[get_id] interrupt returned: '{id_card}'")
    if id_card:
        return {"id_card": id_card.strip().upper()}
    return {}


def add_reserve_message(state: ReserveState):
    reserve_prompt = """根据提供的信息，帮我预定房源。
    - 预定的房源标题: {title}
    - ⽤⼾预定号码: {phone_number}
    - ⽤⼾⾝份证号码: {id_card}"""
    reserve_message = HumanMessage(content=reserve_prompt.format(
        title=state['title'],
        phone_number=state['phone_number'],
        id_card=state['id_card']
    ))

    return {"messages": [reserve_message]}


@tool
def generate_orders(house_title: str,
                    phone_number: str,
                    id_card: str,
                    runtime: ToolRuntime,
                    store: Annotated[Any, InjectedStore()]) -> str:
    """根据⽤⼾预定房源，电话，⾝份证，。

    Args:
        phone_number: ⽤⼾电话
        id_card: ⾝份证
        house_title: ⽤⼾要预定的房源标题
        runtime: ⼯具的运⾏时信息
        store: 注⼊⼯具的持久存储
    """

    # 1. 生成工单号
    order_id = str(uuid.uuid4())

    # 2. 构建预订信息
    reserved_house = ReservedInfo(
        order_id=order_id,
        title=house_title,
        phone_number=phone_number
    )

    #
    user_id = runtime.context.get("user_id")
    if user_id == None:
        return "用户ID获取不到"
    namespace = (user_id, "preferences")
    prefs_result = store.search(namespace)
    if len(prefs_result) == 0:
    # 没有持久化信息，新增
        prefs = UserPreferences(
            reserved_info=[reserved_house]
        )
        store.put(
            namespace,
            str(uuid.uuid4()),
            prefs.model_dump(exclude_none=True)
        )
    else:
        # 有值，更新
        prefs = prefs_result[0].value or {}
        prefs.setdefault('reserved_info', []).append(reserved_house)
        store.put(
            namespace,
            prefs_result[0].key,
            prefs
        )

    # 4. 扩展：持久化订单表
    return f"已成功预定房源：{house_title}，预定⼯单号为：{order_id}"

# 节点：⽣成⼯单结果
def call_orders(state: ReserveState):
    response = model.bind_tools([generate_orders]).invoke(
        [SystemMessage(content="你是⼀个⼯单⽣成的助⼿，⽀持调⽤⼯具进⾏房源预定⼯单⽣成。⽀持查看查询的结果并返回最终答案")]
        + state["messages"]
    )
    return {"messages": [response]}
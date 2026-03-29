from langgraph.graph import MessagesState


# 预订状态
class ReserveState(MessagesState):
    title: str           # 预订的房源
    phone_number: str    # 预订的电话
    id_card: str         # 身份证
    
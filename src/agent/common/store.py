from typing import Optional, List

from pydantic import BaseModel, Field


class ReservedInfo(BaseModel):
    """房源预订信息"""
    order_id: str = Field(description="预订id号")
    title: str = Field(description="预订房源的标题")
    phone_number: str = Field(description="预订手机号码")

    price: Optional[float] = Field(
        default=None,
        description="预订房源的价格，单位为元/月"
    )

    introduce: Optional[str] = Field(
        default=None,
        description="预定的房源介绍"
    )

    city_name: Optional[str] = Field(
        default=None,
        description="预定的房源所在城市名"
    )

    region_name: Optional[str] = Field(
        default=None,
        description="预订的房源所在区/县"
    )


class UserPreferences(BaseModel):
    """用户偏好信息"""
    budget_min: Optional[float] = Field(
        default=None,
        description="用户最低预算，单位为元/月"
    )

    budget_max: Optional[float] = Field(
        default=None,
        description="用户最高预算，单位为元/月"
    )

    reserved_info: Optional[list[ReservedInfo]] = Field(
        default=None,
        description="用户预订过房源的列表"
    )


def read_preferences(value) -> UserPreferences:
    """把 store 里读出来的原始值正规化成 UserPreferences。

    **为什么必须走这里、不能直接下标**：写侧是 `model_dump(exclude_none=True)`
    （`node/recommend.py`、`node/reserve.py`），所以落库的 dict
    **键集合随数据而变** —— 用户第一次只说了预算上限，库里就没有 `budget_min`
    这个键；预定过的用户库里甚至只有 `reserved_info`。

    后果是 `prefs["budget_min"]` 在这种情况下必然 KeyError，而且**不在当场**炸：
    那一轮走的是写侧新用户分支，它把不带 `exclude_none` 的完整键集合（值为 None）
    写进了 state，所以同一轮读不到缺键的版本。要等**下一次调用**——
    `get_store_info` 是图的**入口**（`graph.py` 的 `add_edge(START, ...)`），
    **每轮都重跑**且**无条件**用 store 的值覆盖 state（`node/main.py`），
    于是缺键版本被盖进 state。新会话会炸，**同一会话发第二条消息也会**。
    曾导致整个会话在入口就死。复现与回归检查（两种触发都有）：
    `eval/tools/probe_cross_session.py`。

    统一从这里读，读者只需要处理 None，不必知道哪些键存在：
    `UserPreferences` 的字段全是 Optional，缺键 → None；多出来的键被 pydantic
    忽略（默认 `extra='ignore'`），不会因为库里多存了什么就崩。

    刻意**不**吞 `ValidationError`：库里存的是我们自己写的，真出现形状损坏
    应当显式报出来，而不是静默降级成"没有偏好"——那会让系统再问一遍用户，
    把数据损坏伪装成正常交互。
    """
    return UserPreferences(**(value or {}))

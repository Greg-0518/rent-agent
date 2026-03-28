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

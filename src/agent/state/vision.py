"""
多模态看房Agent状态定义
"""

from typing import List, Optional, TypedDict
from langgraph.graph import MessagesState


class RoomAnalysis(TypedDict):
    """房间分析结果"""
    room_type: str  # 房间类型
    area_estimate: float  # 面积估计
    decoration_condition: str  # 装修状况
    facilities: List[str]  # 设施列表
    lighting: str  # 采光情况
    cleanliness: str  # 整洁程度
    issues: List[str]  # 潜在问题
    overall_rating: float  # 整体评分


class VisionState(MessagesState):
    """多模态看房状态"""
    # 图片URL或base64列表
    images: List[str] = []
    
    # 分析结果
    analysis_results: List[RoomAnalysis] = []
    
    # 检测到的问题
    issues_detected: List[dict] = []
    
    # 看房报告
    report: str = ""

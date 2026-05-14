"""
多模态看房Agent图构建
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.state.vision import VisionState
from src.agent.node.vision import (
    image_analysis_node,
    issue_detection_node,
    report_generation_node
)


def build_vision_graph():
    """
    构建多模态看房Agent图
    
    流程：
    START -> 图片分析 -> 问题检测 -> 报告生成 -> END
    """
    builder = StateGraph(VisionState)
    
    # 添加节点
    builder.add_node("analyze_images", image_analysis_node)
    builder.add_node("detect_issues", issue_detection_node)
    builder.add_node("generate_report", report_generation_node)
    
    # 添加边
    builder.add_edge(START, "analyze_images")
    builder.add_edge("analyze_images", "detect_issues")
    builder.add_edge("detect_issues", "generate_report")
    builder.add_edge("generate_report", END)
    
    return builder.compile()


# 导出编译后的图
vision_graph = build_vision_graph()

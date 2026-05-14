"""
租金计算Agent图构建
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.state.finance import FinanceState
from src.agent.node.finance import (
    code_generation_node,
    code_execution_node,
    error_correction_node,
    answer_generation_node,
    should_retry
)


def build_finance_graph():
    """
    构建租金计算Agent图
    
    流程：
    START -> 代码生成 -> 代码执行 -> [成功? -> 答案生成 : 错误修正 -> 代码执行]
    """
    builder = StateGraph(FinanceState)
    
    # 添加节点
    builder.add_node("generate_code", code_generation_node)
    builder.add_node("execute_code", code_execution_node)
    builder.add_node("correct_error", error_correction_node)
    builder.add_node("generate_answer", answer_generation_node)
    
    # 添加边
    builder.add_edge(START, "generate_code")
    builder.add_edge("generate_code", "execute_code")
    
    # 条件边：根据执行结果决定下一步
    builder.add_conditional_edges(
        "execute_code",
        should_retry,
        {
            "retry": "correct_error",
            "answer": "generate_answer",
            "give_up": END
        }
    )
    
    # 错误修正后重新执行
    builder.add_edge("correct_error", "execute_code")
    
    # 生成答案后结束
    builder.add_edge("generate_answer", END)
    
    return builder.compile()


# 导出编译后的图
finance_graph = build_finance_graph()

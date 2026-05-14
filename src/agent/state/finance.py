"""
租金计算Agent状态定义
"""

from typing import Optional, TypedDict
from langgraph.graph import MessagesState


class ExecutionResult(TypedDict):
    """代码执行结果"""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timed_out: bool


class FinanceState(MessagesState):
    """租金计算状态"""
    # 用户问题
    user_question: str = ""
    
    # 生成的Python代码
    generated_code: str = ""
    
    # 执行结果
    execution_result: Optional[ExecutionResult] = None
    
    # 错误信息
    error_message: str = ""
    
    # 图表base64
    chart_base64: str = ""
    
    # 最终答案
    final_answer: str = ""
    
    # 重试次数
    retry_count: int = 0

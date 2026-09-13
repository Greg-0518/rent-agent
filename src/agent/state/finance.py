"""
租金计算Agent状态定义
"""

from typing import Optional, TypedDict
from langgraph.graph import MessagesState


class ExecutionResult(TypedDict):
    """代码执行结果

    rejected / truncated 是沙箱升级时加的（见 node/finance.py 的 execute_code_sandbox）：
    `rejected=True` 表示代码被**静态白名单**挡下、解释器根本没启动；
    `truncated=True` 表示输出超过上限被截断（不截断会把父进程内存吃光）。
    两者都同时体现为 `exit_code=-1` 与非空 stderr，所以沿"报错→修代码"的既有循环
    自动走，节点侧不需要额外分支。
    """
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timed_out: bool
    rejected: bool
    truncated: bool


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

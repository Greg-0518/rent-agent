"""
租金计算Agent节点定义
"""
import re, subprocess, tempfile, os, time
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.common.llm import model
from src.agent.state.finance import FinanceState, ExecutionResult


# ============ 工具函数 ============

def generate_code(user_question: str, context: str = "") -> str:
    """根据用户问题生成 Python 代码"""
    system = SystemMessage(content="""你是 Python 数值计算专家。根据用户问题生成可直接执行的 Python 代码。

规则：
1. 只输出 Python 代码，放在 ```python ``` 代码块内
2. 用 print() 输出计算结果，每行一个结果项
3. 可以定义变量和函数，但不要用 input()
4. 代码内嵌所有数据，用注释标注数据来源
5. 计算金额时保留 2 位小数""")

    ctx = f"\n上下文数据：{context}" if context else ""
    resp = model.invoke([system, HumanMessage(content=f"{user_question}{ctx}")])
    content = resp.content

    # 提取代码块
    match = re.search(r'```(?:python)?\s*\n?(.*?)```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 没找到代码块，尝试提取 import/def/print 开头的行
    lines = [l for l in content.split('\n') if l.strip() and (
        l.strip().startswith(('import ', 'from ', 'def ', 'print(', '#', '=', 'if ', 'for ', 'result')))]
    return '\n'.join(lines) if lines else content.strip()


def execute_code_sandbox(code: str, timeout: int = 30) -> ExecutionResult:
    """在沙箱中执行 Python 代码"""
    result: ExecutionResult = {"stdout": "", "stderr": "", "exit_code": 0, "execution_time": 0.0, "timed_out": False}

    if not code.strip():
        result["stderr"] = "代码为空"
        result["exit_code"] = -1
        return result

    try:
        # 用临时文件隔离执行
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            tmp = f.name

        start = time.time()
        proc = subprocess.run(
            ['python', tmp], capture_output=True, text=True,
            timeout=timeout, encoding='utf-8', errors='replace'
        )
        result["execution_time"] = time.time() - start
        result["stdout"] = proc.stdout.strip()
        result["stderr"] = proc.stderr.strip()
        result["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["stderr"] = f"执行超时（>{timeout}秒）"
        result["exit_code"] = -1
    except Exception as e:
        result["stderr"] = str(e)
        result["exit_code"] = -1
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

    return result


def fix_code(original_code: str, error_message: str) -> str:
    """修复代码错误"""
    system = SystemMessage(content="你是 Python 调试专家。根据错误信息修复代码，只输出修复后的完整代码在 ```python ``` 块内。")
    resp = model.invoke([system, HumanMessage(content=f"原代码：\n```python\n{original_code}\n```\n\n错误信息：{error_message}\n请修复。")])
    match = re.search(r'```(?:python)?\s*\n?(.*?)```', resp.content, re.DOTALL)
    return match.group(1).strip() if match else resp.content.strip()


# ============ 节点函数 ============

def code_generation_node(state: FinanceState) -> dict:
    question = state.get("user_question", "")
    if not question:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            question = last
    code = generate_code(question)
    return {"generated_code": code, "user_question": question, "retry_count": 0}


def code_execution_node(state: FinanceState) -> dict:
    code = state.get("generated_code", "")
    result = execute_code_sandbox(code)
    error = result["stderr"] if result["exit_code"] != 0 else ""
    return {"execution_result": result, "error_message": error}


def error_correction_node(state: FinanceState) -> dict:
    code = state.get("generated_code", "")
    error = state.get("error_message", "")
    fixed = fix_code(code, error)
    count = state.get("retry_count", 0) + 1
    return {"generated_code": fixed, "retry_count": count}


def answer_generation_node(state: FinanceState) -> dict:
    question = state.get("user_question", "")
    result = state.get("execution_result", {})
    stdout = result.get("stdout", "") if result else ""
    stderr = result.get("stderr", "") if result else ""

    prompt = f"用户问题：{question}\n\n计算结果：\n{stdout}\n"
    if stderr:
        prompt += f"\n警告/错误：{stderr}\n"
    prompt += "\n请用自然语言汇总计算结果，格式清晰，每个计算项单独列出。"

    resp = model.invoke([HumanMessage(content=prompt)])
    return {"final_answer": resp.content,
            "messages": [AIMessage(content=f"[租金计算结果]\n\n{resp.content}")]}


# ============ 条件函数 ============

def should_retry(state: FinanceState) -> str:
    error = state.get("error_message", "")
    count = state.get("retry_count", 0)
    if not error:
        return "answer"
    if count >= 3:
        return "give_up"
    return "retry"

"""
租金计算Agent节点定义
"""
import ast
import re, shutil, subprocess, sys, tempfile, threading, os, time
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


# ============ 代码沙箱 ============

SANDBOX_REJECT_MARKER = "[CODE_SANDBOX_REJECTED]"

DEFAULT_TIMEOUT = 30
# 单条流最多留多少字符。**读干、丢弃多余**：只截断不读干，子进程会因管道写满而卡住，
# 那时报出来的是"超时"——把一个"输出太多"伪装成"算得太慢"。
MAX_OUTPUT_CHARS = 200_000

# 允许 import 的模块：**纯计算 / 纯数据结构**，没有文件、网络、进程、反射能力。
#
# 为什么不一禁到底？生成的代码里 `import math`（算面积/平均）和
# `from decimal import Decimal`（prompt 明确要求"金额保留 2 位小数"）都是很自然的写法。
# 一律禁掉不是"更安全"，只是把成本转移给 fix_code 重试循环——每次被拒都要多花一次
# 模型调用，而防御收益为零（这两个模块本来就够不到 IO）。
ALLOWED_MODULES = frozenset({
    "math", "statistics", "decimal", "fractions", "itertools", "functools",
    "collections", "datetime", "json",
})

# 允许调用的内置函数：纯计算 / 纯字符串，没有 IO、没有反射。
# 用白名单而不是黑名单：`generate_code` 的 prompt 已经要求"代码内嵌所有数据、
# 不许用 input()"，所以这份代码本来只需要算术能力，白名单不会误伤真实用例。
# （`from math import sqrt` 这类导入进来的名字另外放行，见 check_code_safety）
ALLOWED_CALLS = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "len", "list", "map", "max", "min",
    "pow", "print", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "isinstance",
})

# 名字黑名单：反射 / IO / 解释器开关。即使不 import 也应该拦掉，
# 因为这些是"从一个 `__builtins__` 逃出去"的常用跳板。
BANNED_NAMES = frozenset({
    "open", "input", "eval", "exec", "compile", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "dir",
    "help", "type", "object", "super", "classmethod", "staticmethod",
    "property", "memoryview", "__builtins__", "__loader__", "__spec__",
})

# 语句级黑名单。放黑名单而不是全量白名单：未知但无害的写法不该被拒，
# 真正的牙是三条——不许 import 白名单外的模块、不许魔术属性、不许调黑名单里的名字。
# （Import/ImportFrom 不在这里：它们要先过 ALLOWED_MODULES，见 check_code_safety）
DENIED_NODES = (
    (ast.Global, "禁止使用 global"),
    (ast.Nonlocal, "禁止使用 nonlocal"),
    (ast.ClassDef, "本沙箱不需要定义类"),
    (ast.AsyncFunctionDef, "不支持 async"),
    (ast.Await, "不支持 async"),
    (ast.AsyncFor, "不支持 async"),
    (ast.AsyncWith, "不支持 async"),
    (ast.Yield, "不支持生成器"),
    (ast.YieldFrom, "不支持生成器"),
    (ast.Delete, "禁止 del"),
)


def check_code_safety(code: str) -> tuple[bool, str, str]:
    """静态检查生成的代码，返回 (是否放行, 规则名, 原因)。规则名给日志用。

    这层是**闸口**：不合规的代码一次都不会被执行，连解释器都不启动。

    第一趟先收集"哪些名字可以调用"，因为 `Call` 的白名单判据依赖它：
      · import 进来的名字（`from math import sqrt` → `sqrt(...)` 放行）
      · 代码自己定义/绑定的名字（`def area(...)` → `area(3, 4)` 放行；
        赋值产生的名字同理）
    漏掉第二类会把**正常的自定义函数**判成违规——这是写完第一版就踩到的
    （`def area(w, h): ...; print(area(3,4))` 被判 `banned_call:area`）。

    放行"调用自己绑定的名字"不构成逃逸口：`x = open` / `x = eval` 里的 `open`/`eval`
    是 Load 进的名字，会在 `banned_name` 那一关被拦下，
    所以不可能把一个黑名单函数改名绕过。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, "syntax_error", f"代码语法错误：{e.msg}（第 {e.lineno} 行）"

    callable_names: set[str] = set(ALLOWED_CALLS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"banned_import:{alias.name}", f"禁止导入 `{alias.name}`"
                callable_names.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            # node.level > 0 是相对导入（`from . import x`），一律拒
            if node.level or root not in ALLOWED_MODULES:
                return False, f"banned_import:{node.module or '.'}", f"禁止导入 `{node.module}`"
            for alias in node.names:
                callable_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.FunctionDef):
            callable_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            # 赋值/循环变量：`f = lambda x: x` 之后 `f(...)` 也要放行
            callable_names.add(node.id)

    for node in ast.walk(tree):
        for cls, reason in DENIED_NODES:
            if isinstance(node, cls):
                return False, f"banned_node:{cls.__name__}", reason
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"private_attr:{node.attr}", (
                f"禁止访问下划线开头的属性 `{node.attr}`——`().__class__.__bases__` "
                "这类写法是逃出沙箱的常规路子"
            )
        if isinstance(node, ast.Name):
            name = node.id
            if name in BANNED_NAMES or (name.startswith("__") and name.endswith("__")):
                return False, f"banned_name:{name}", f"禁止使用 `{name}`"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called = node.func.id
            if called not in callable_names:
                return False, f"banned_call:{called}", (
                    f"禁止调用 `{called}`（本沙箱只放开纯计算与字符串函数、"
                    "白名单模块里的函数，以及代码自己定义/绑定的名字）"
                )
    return True, "", ""


def _sandbox_env(workdir: str) -> dict:
    """最小环境变量：**不继承 os.environ**。

    这是有意写死的：父进程环境里有 API key 与数据库口令，而"代码内嵌所有数据"
    这条生成约束意味着被测代码没有任何理由读环境变量。只留 Windows 运行时
    自身需要的少量变量。

    这里**故意不写任何 PYTHON* 变量**：命令行的 `-I` 隐含 `-E`，PYTHON* 环境变量
    在子进程里会被忽略，写了只会让下一个人以为它生效——所以 UTF-8 输出改用
    命令行上的 `-X utf8` 保证（命令行参数不受 -E 影响）。
    """
    env = {"TEMP": workdir, "TMP": workdir}
    for key in ("SystemRoot", "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _drain(stream, sink: list, cap: int) -> None:
    """把管道读干，但只保留前 cap 个字符。

    在有界读取线程里做：子进程 write 到满就会阻塞，所以必须持续读；
    而"持续读"如果不设上限，一段 `while True: print()` 就能把父进程内存吃光
    （旧版 capture_output 正是这样）。
    """
    kept = 0
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            room = cap - kept
            if room > 0:
                sink.append(chunk[:room])
                kept += min(len(chunk), room)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill(proc) -> None:
    """超时后杀掉执行进程。"""
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def execute_code_sandbox(code: str, timeout: int = DEFAULT_TIMEOUT) -> ExecutionResult:
    """在收紧过的子进程沙箱里执行生成的 Python 代码。

    三层，都能单测（`tests/unit_tests/test_code_sandbox.py`）：

      1. **静态白名单**（`check_code_safety`）：AST 检查，禁 import、
         禁魔术属性、禁 `open/eval/exec/...`，不合规的代码**根本不启动解释器**。
      2. **进程收紧**：用 `sys.executable -I -B`（隔离模式，忽略 PYTHONPATH 与
         user site）、cwd 指向一次性临时目录（相对路径写入不会落在仓库里）、
         **env 不继承父进程**。
      3. **资源收紧**：超时杀掉；stdout/stderr 各留前 200k 字符（读干后丢弃多余）。

    **残留缺口（明知而不做，不要误读成"已经安全"）**：
      · 没有内存 / CPU 配额——Windows 上没有 stdlib 的 rlimit，被放行的代码
        仍能把一个核跑满到超时为止；
      · 没有容器/文件系统隔离——进程仍能读整个文件系统（只是不允许 import
        与反射，写路径被引到临时目录）。要真隔离得上容器，那是另一个量级的改动。
    """
    result: ExecutionResult = {
        "stdout": "", "stderr": "", "exit_code": 0, "execution_time": 0.0,
        "timed_out": False, "rejected": False, "truncated": False,
    }

    if not code.strip():
        result["stderr"] = "代码为空"
        result["exit_code"] = -1
        return result

    ok, rule, reason = check_code_safety(code)
    if not ok:
        result["rejected"] = True
        result["exit_code"] = -1
        result["stderr"] = f"{SANDBOX_REJECT_MARKER} 代码被沙箱拒绝：{reason}\n规则：{rule}"
        return result

    workdir = tempfile.mkdtemp(prefix="rent_sandbox_")
    script = os.path.join(workdir, "calc.py")
    out_buf: list[str] = []
    err_buf: list[str] = []
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)

        start = time.time()
        proc = subprocess.Popen(
            # -I 隔离（忽略 PYTHONPATH/user site，隐含 -E）、-B 不写 .pyc、
            # -X utf8 保证输出编码与区域设置无关
            [sys.executable, "-I", "-B", "-X", "utf8", script],
            cwd=workdir, env=_sandbox_env(workdir), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, out_buf, MAX_OUTPUT_CHARS), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err_buf, MAX_OUTPUT_CHARS), daemon=True),
        ]
        for t in readers:
            t.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            _kill(proc)
        for t in readers:
            t.join(timeout=5)

        result["execution_time"] = time.time() - start
        stdout = "".join(out_buf)
        stderr = "".join(err_buf)
        result["truncated"] = len(stdout) >= MAX_OUTPUT_CHARS or len(stderr) >= MAX_OUTPUT_CHARS
        result["stdout"] = stdout.strip()
        if result["timed_out"]:
            result["exit_code"] = -1
            # 超时前往往已经算出一部分，别丢——回灌给模型比只回一句"超时"有用
            partial = []
            if result["stdout"]:
                partial.append(f"（超时前的输出）\n{result['stdout']}")
            if stderr.strip():
                partial.append(f"（超时前的错误输出）\n{stderr.strip()}")
            result["stderr"] = f"执行超时（>{timeout}秒）" + ("\n" + "\n".join(partial) if partial else "")
        else:
            result["stderr"] = stderr.strip()
            result["exit_code"] = proc.returncode
    except Exception as e:
        result["stderr"] = f"{type(e).__name__}: {e}"
        result["exit_code"] = -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

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

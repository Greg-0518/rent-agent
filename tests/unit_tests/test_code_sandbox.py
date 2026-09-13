"""代码执行沙箱单测 —— 不依赖数据库、不调模型。

补这个文件的原因：`execute_code_sandbox` 原来只是"写临时文件 + subprocess.run"，
生成的代码可以 `import os` 读环境变量（里面有 API key）、起子进程、往仓库里写文件。
那些行为**当时一条测试都没有**，所以收紧之后需要把"紧了什么"钉住，
同时钉住"正常算术不能被误伤"——否则下次有人收得过紧，先坏的是真用例。

三层各自的最小用例：
  1. 静态白名单（`check_code_safety`）：纯函数，逐条列规则
  2. 进程收紧（`_sandbox_env` + Popen 参数）：用假 Popen 看接线，不真起进程
  3. 资源收紧：真起进程（超时、输出截断、临时目录清理）
"""

import io
import os
import tempfile

import pytest

from src.agent.node import finance
from src.agent.node.finance import (
    SANDBOX_REJECT_MARKER,
    check_code_safety,
    execute_code_sandbox,
)


# ---------- 1. 静态白名单：放行 ----------

@pytest.mark.parametrize("code", [
    "print(round(sum([1.5, 2.25]), 2))",
    "import math\nprint(round(math.pi * 2, 2))",
    # prompt 明确要求"金额保留 2 位小数"，Decimal 是很自然的写法，不能拒
    "from decimal import Decimal\nprint(Decimal('1.005').quantize(Decimal('0.01')))",
    "import statistics\nprint(statistics.mean([1, 2, 3]))",
    "rows = [{'price': 100}, {'price': 200}]\nprint(sum(r.get('price', 0) for r in rows))",
    "print(f'总计 {sum([1, 2])} 元')",
    "def area(w, h):\n    return w * h\nprint(area(3, 4))",
    "print(sorted([3, 1, 2], key=lambda x: -x))",
])
def test_safe_code_passes_static_check(code):
    ok, rule, reason = check_code_safety(code)
    assert ok, f"{rule}: {reason}"


# ---------- 1'. 静态白名单：拦下 ----------

@pytest.mark.parametrize("code,rule_prefix", [
    ("import os", "banned_import:os"),
    ("import subprocess", "banned_import:subprocess"),
    ("import socket", "banned_import:socket"),
    ("import sys", "banned_import:sys"),
    ("from os import system", "banned_import:os"),
    ("from . import x", "banned_import"),
    ("import os.path", "banned_import:os.path"),
    ("open('x.txt', 'w')", "banned_call:open"),
    ("print(input())", "banned_call:input"),          # prompt 禁了 input()；它会挂住整张图
    ("eval('1+1')", "banned_call:eval"),
    # 规则名取决于 ast.walk 的访问顺序：Call 节点先于里面的 Name 节点被访问，
    # 所以报的是 banned_call 而不是 banned_name。两者都拦得住，这里钉住实际的那个。
    ("__import__('os')", "banned_call:__import__"),
    ("print(().__class__)", "private_attr:__class__"),
    ("print([].__class__.__bases__)", "private_attr:"),   # __bases__ 先被访问到
    ("globals()", "banned_call:globals"),
    ("x = eval\nprint(1)", "banned_name:eval"),           # 改名绕不过黑名单
    ("def f():\n    global x", "banned_node:Global"),
    ("class A:\n    pass", "banned_node:ClassDef"),
    ("del x", "banned_node:Delete"),
    ("def f(:\n    pass", "syntax_error"),
])
def test_unsafe_code_is_rejected(code, rule_prefix):
    ok, rule, _ = check_code_safety(code)
    assert not ok
    assert rule.startswith(rule_prefix), f"{code!r} → {rule}"


def test_rejected_code_never_reaches_the_interpreter(monkeypatch):
    """闸口的意义：被拒的代码连解释器都不启动（Popen 一次都不能被调用）"""
    def _boom(*a, **kw):  # pragma: no cover - 只在接线坏了才会被执行
        raise AssertionError("被拒的代码不应该启动子进程")

    monkeypatch.setattr(finance.subprocess, "Popen", _boom)
    r = execute_code_sandbox("import os\nprint(os.getcwd())")
    assert r["rejected"] is True
    assert r["exit_code"] == -1
    assert r["stdout"] == ""
    assert SANDBOX_REJECT_MARKER in r["stderr"]


def test_empty_code_short_circuits():
    r = execute_code_sandbox("   ")
    assert r["exit_code"] == -1
    assert r["rejected"] is False       # 不是"被拒"，是没东西可跑


# ---------- 2. 进程收紧：看接线，不真起进程 ----------

def test_sandbox_env_does_not_inherit_secrets(monkeypatch):
    """环境变量不继承：父进程里的 key/口令不能进子进程。

    只能在机制层测——被测代码连 `os` 都 import 不了，没法自己报告环境。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DB_PASSWORD", "should-not-leak")
    env = finance._sandbox_env("W:\\tmp")
    assert "DEEPSEEK_API_KEY" not in env
    assert "DB_PASSWORD" not in env
    assert not any("KEY" in k.upper() or "PASSWORD" in k.upper() for k in env)
    # 工作目录通过 TEMP/TMP 指给子进程
    assert env["TEMP"] == "W:\\tmp" and env["TMP"] == "W:\\tmp"


def test_popen_is_called_with_hardened_args(monkeypatch):
    """接线检查：隔离模式 + 工作目录在临时目录 + 不传父进程环境

    `cwd_existed` / `script_written` 必须在**调用当刻**记下来：临时目录在 finally
    里就被删了，跑到断言那一步再看必然不存在（第一版就是这么写错的）。
    """
    seen = {}

    class _FakeProc:
        def __init__(self, *a, **kw):
            seen["args"], seen["kwargs"] = a, kw
            cwd = kw["cwd"]
            seen["cwd_existed"] = os.path.isdir(cwd)
            seen["script_written"] = os.path.isfile(os.path.join(cwd, "calc.py"))
            self.stdout, self.stderr = io.StringIO("ok"), io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(finance.subprocess, "Popen", _FakeProc)
    r = execute_code_sandbox("print('ok')")

    args = seen["args"][0]
    assert args[0] == finance.sys.executable      # 用当前解释器，不依赖 PATH 上的 python
    assert "-I" in args and "-B" in args and "utf8" in args
    assert args[-1].endswith("calc.py")
    assert seen["cwd_existed"], "工作目录在调用当刻应当真实存在"
    assert seen["script_written"], "脚本应当写在工作目录里（相对路径不会落到仓库）"
    assert os.path.abspath(seen["kwargs"]["cwd"]) != os.getcwd(), "cwd 不能留在仓库里"
    assert "DEEPSEEK_API_KEY" not in seen["kwargs"]["env"]
    assert r["stdout"] == "ok" and r["exit_code"] == 0


# ---------- 3. 资源收紧：真起进程 ----------

def test_normal_arithmetic_runs_end_to_end():
    r = execute_code_sandbox("print(round(1.5 + 2.25, 2))")
    assert r["exit_code"] == 0
    assert r["stdout"] == "3.75"
    assert r["rejected"] is False and r["truncated"] is False


def test_chinese_output_survives_round_trip():
    r = execute_code_sandbox("print('中文输出 123')")
    assert r["stdout"] == "中文输出 123"


def test_timeout_is_killed_and_reported():
    r = execute_code_sandbox("while True:\n    pass\n", timeout=2)
    assert r["timed_out"] is True
    assert r["exit_code"] == -1
    assert r["stderr"].startswith("执行超时（>2秒）")


def test_runaway_output_is_capped_but_kept(monkeypatch):
    """输出洪水：必须既不 OOM 也不丢干净——留前 N 个字符并标 truncated。

    旧版用 capture_output，5 秒的 print 洪水能把父进程内存吃满；
    而"只截断不读干"会让子进程卡在写管道上，最后被报成"超时"。
    """
    monkeypatch.setattr(finance, "MAX_OUTPUT_CHARS", 200)
    r = execute_code_sandbox("for i in range(500):\n    print('x' * 20)\n")
    assert r["truncated"] is True
    assert len(r["stdout"]) == 200


def test_timeout_keeps_partial_output():
    """超时前的输出要留着——回灌给模型比只回一句"超时"有用。

    `flush=True` 是必须的：子进程的 stdout 接的是管道，默认块缓冲，
    没写满 4KB 就一直在缓冲里——被 kill 时那段输出根本还没进管道，
    断言也就成了随机的。顺带说明：真实用例里模型通常不写 flush，
    所以"超时前输出"多数时候是空的，这个能力只在输出量已经超过一个块时生效。
    """
    r = execute_code_sandbox("print('先算好的部分', flush=True)\nwhile True:\n    pass\n", timeout=2)
    assert r["timed_out"] is True
    assert "先算好的部分" in r["stderr"]


def test_temp_dir_is_cleaned_up(monkeypatch):
    made = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*a, **kw):
        path = real_mkdtemp(*a, **kw)
        made.append(path)
        return path

    monkeypatch.setattr(finance.tempfile, "mkdtemp", _spy)
    execute_code_sandbox("print(1)")
    assert made, "应当真的建了临时目录"
    assert all(not os.path.exists(p) for p in made), "跑完必须删掉临时目录"

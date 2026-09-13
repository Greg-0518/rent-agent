"""text2sql 评测用例 —— 一键回归

    pytest eval -m smoke              日常（20 条）
    pytest eval                       全量开发集（70 条）
    pytest eval --holdout             并入保留测试集（收官，共 89 条）

四种结果，两类都**不算通过**，但成因完全不同：

    PASS              结果集/值对上了
    FAIL              明确错了（附归因，真因在模型能力轴上）
    skip(存疑)        走到了、但断言器判不了（如列名无交集）——不计入通过，
                      会被 conftest 的存疑占比上限兜住
    skip(不可达)      压根没走到被测链路（被意图路由分走）——与模型能力无关，
                      不计入模型能力分，但会被 conftest 的不可达占比上限兜住
                      （防"路由全线失灵 → 全绿的空跑"）

刻意不把"存疑"当通过：宁可漏判，不给假通过。
刻意不把"不可达"判失败：12 条永远不可能变绿的用例会让回归常红，
而常红的门禁等于没有门禁。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.asserters.rule import FAIL, UNREACHABLE, UNVERIFIABLE  # noqa: E402
from eval.runner.selection import SMOKE_IDS, select_cases  # noqa: E402

pytestmark = pytest.mark.eval


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    cases = select_cases(holdout=metafunc.config.getoption("--holdout"))
    params = []
    for c in cases:
        marks = []
        if c.id in set(SMOKE_IDS):
            marks.append(pytest.mark.smoke)
        if c._holdout:
            marks.append(pytest.mark.holdout)
        params.append(pytest.param(c, id=c.id, marks=marks))
    metafunc.parametrize("case", params)


def _failure_report(case, trace, result) -> str:
    lines = [
        f"[{case.id}] {case.tier} · {case.assertion.mode} · {case.split}",
        f"问题：{case.question}",
        f"归因：{result.attribution or '(未分类)'}",
        f"判定：{result.reason}",
        f"期望：{result.expected!r}   实际：{result.actual!r}",
    ]
    if trace.error:
        lines.append(f"图执行异常：{trace.error}")
    if trace.sql_executed:
        lines.append(f"最终执行的 SQL：{trace.sql_executed[-1]}")
    if trace.sql_errors:
        lines.append(f"SQL 报错 {len(trace.sql_errors)} 次，最后一条：{trace.sql_errors[-1][:200]}")
    if trace.guard_rejections:
        lines.append(f"安全层拦截 {len(trace.guard_rejections)} 次："
                     f"{trace.guard_rejections[-1].get('message', '')[:120]}")
    detail = {k: v for k, v in (result.detail or {}).items() if k != "final_sql"}
    if detail:
        lines.append(f"明细：{detail}")
    lines.append(f"完整轨迹与 stdout 见 eval/results/*.jsonl 里的 {case.id}")
    return "\n".join(lines)


def test_text2sql(case, golden, run_one_case):
    trace, result = run_one_case(case, golden[case.id])

    if result.kind == FAIL:
        pytest.fail(_failure_report(case, trace, result), pytrace=False)

    if result.kind == UNVERIFIABLE:
        pytest.skip(f"[{case.id}] 存疑（不计通过）：{result.reason}")

    if result.kind == UNREACHABLE:
        # 不是通过，也不是模型失败。单独成一个 skip 原因，好让人一眼看出
        # "这轮有多少条根本没测到"，而不是把它混进存疑里。
        pytest.skip(f"[{case.id}] 不可达（未走到被测链路，不计入模型能力分）："
                    f"{result.reason}")

    assert result.kind == "pass", f"未知结果类型 {result.kind}"

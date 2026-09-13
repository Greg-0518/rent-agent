"""评测 pytest 入口的公共装置

    pytest eval -m smoke          # 日常：20 条核心子集，约 3 分钟
    pytest eval                   # 全量开发集 70 条，约 12 分钟
    pytest eval --holdout         # 阶段收官：并入 19 条保留测试集（共 89 条）
    python eval/report/gen_report.py   # 读最近一次 run 的 jsonl 出报告

**环境不对就先拦住**：数据库连不上、模型 key 失效，都会让几十条用例一起失败，
看起来像"模型能力崩了"，实际是环境问题。所以在 session 开始时就一次性探明，
不通直接 `pytest.exit`，并指路 `python eval/tools/check_env.py`。

**每条用例的完整 stdout 都会落进 jsonl**（图里那些 print 是排查问题的唯一线索，
评测跑完人不在现场，不留日志就只能重跑）。
"""

import json
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.asserters.rule import FAIL, UNREACHABLE, UNVERIFIABLE, assert_case  # noqa: E402
from eval.runner.agent_runner import build_graph, run_case, run_multi_turn_case  # noqa: E402
from eval.runner.selection import SMOKE_IDS, select_cases  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"

# 存疑占比上限：超过这个比例，说明断言器判不了的比例太高，分数不可信。
# 与其出一个"完成率 92%"的漂亮数字，不如让回归本身失败——飞轮文
# "评测可信度 > 系统复杂度"，一个可信度存疑的成绩单比没有成绩单更危险。
MAX_UNVERIFIABLE_RATIO = 0.20

# 不可达占比上限：**这道门禁是防"空跑伪装成全绿"**。
# 被路由分走的用例判 unreachable、不算失败，这是对的（见 rule.py 的说明）。
# 但代价是：如果哪天路由坏了、把所有用例都分流出去，整套回归会变成
# "0 条通过、0 条失败、89 条不可达"——一个绿色的、什么都没测的 run。
# 已知基线是 **12/70 = 17%**（L2 整档十条 + L3-015 + L4-010），留到 40% 作余量。
# 注意别把它和 `audit_intent.py` 的 24/79 ≈ 30% 搞混：那是**可达性探针**的结果，
# 里面把 L5 整档算作不可达；而记分时 `refusal` 模式第一段就短路返回、压根不看
# text2sql 通不通，所以 L5 是被计分的。两个数各有用途，不能混着引用。
MAX_UNREACHABLE_RATIO = 0.40


def pytest_addoption(parser):
    parser.addoption(
        "--holdout", action="store_true", default=False,
        help="并入保留测试集（只在阶段收官跑，跑完不要据它改 Prompt）",
    )
    parser.addoption(
        "--run-id", action="store", default=None,
        help="本次 run 的标识，用作 eval/results/{run_id}.jsonl 的文件名；默认按时间生成",
    )


def _gate_environment() -> None:
    """跑之前把环境问题拦掉，避免几十条用例一起失败被误读成模型退化"""
    from eval.runner.db import connect
    from src.agent.common.llm import get_model_config, model

    try:
        connect().close()
    except Exception as exc:
        pytest.exit(
            f"数据库连不上（{type(exc).__name__}: {str(exc)[:150]}）。\n"
            "评测依赖 MySQL，先起 VM 再跑；或用 `python eval/tools/check_env.py` 逐项自检。",
            returncode=3,
        )

    cfg = get_model_config()
    try:
        model.invoke("回两个字：可用")
    except Exception as exc:
        pytest.exit(
            f"模型调用失败（profile={cfg['profile'] or '(未设)'} model={cfg['model']}）："
            f"{type(exc).__name__}: {str(exc)[:150]}\n"
            "先跑 `python eval/tools/check_env.py` 确认 key / base_url。",
            returncode=3,
        )


def pytest_configure(config):
    config._eval_records: list[dict] = []
    config._eval_run_id = config.getoption("--run-id") or time.strftime("run-%Y%m%d-%H%M%S")
    config._eval_fh = None
    config._eval_meta: dict = {}
    config._eval_path = RESULTS_DIR / f"{config._eval_run_id}.jsonl"


def pytest_collection_modifyitems(config, items):
    """保留测试集默认不收集——它只在收官跑，日常跑它就等于让它参与迭代"""
    if config.getoption("--holdout"):
        return
    skip = pytest.mark.skip(reason="保留测试集，需显式 --holdout（阶段收官才跑）")
    for item in items:
        if "holdout" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def golden() -> dict:
    path = ROOT / "eval" / "golden" / "text2sql.json"
    if not path.exists():
        pytest.exit(
            "缺 eval/golden/text2sql.json，先跑 `python eval/tools/make_golden.py`",
            returncode=3,
        )
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="session", autouse=True)
def _environment():
    _gate_environment()


@pytest.fixture(scope="session")
def eval_cases(request) -> list:
    cases = select_cases(holdout=request.config.getoption("--holdout"))
    if not cases:
        pytest.exit("没有选中任何用例", returncode=3)
    return cases


@pytest.fixture
def run_one_case(request):
    """跑一条用例：捕获 stdout → 断言 → 落盘 jsonl。返回 (trace, result)。"""

    def _run(case, golden_entry):
        # 每条用例一份独立的 checkpointer / store：用例之间不能互相污染
        # （store 里存着 user_preferences，共用会让第 2 条用例读到第 1 条的预算）。
        #
        # 多轮用例（`case.is_multi_turn`）不用这里建的这个图：它的隔离粒度是
        # **用例，不是轮**——整条用例共用一份 store + 一份 checkpointer，新会话
        # 只换 thread_id。所以它自己建图（见 agent_runner.run_multi_turn_case）。
        graph = build_graph()
        buf = StringIO()
        started = time.time()
        with redirect_stdout(buf):
            if case.is_multi_turn:
                trace = run_multi_turn_case(case)
            else:
                trace = run_case(case, graph=graph)
        elapsed = time.time() - started

        result = assert_case(case, trace, golden_entry)

        request.config._eval_records.append({
            "kind": "case",
            "case_id": case.id,
            "tier": case.tier,
            "split": case.split,
            "mode": case.assertion.mode,
            "question": case.question,
            "room_count": case.room_count,
            # 0 = 单轮用例。报告里靠它区分"多轮的失败"与"单轮的失败"——
            # 两者的排查方式完全不同（多轮先看 turn_sql_counts 与 turn_errors）。
            "n_turns": trace.n_turns,
            "seconds": round(elapsed, 1),
            "latency_ms": round(trace.latency_ms, 1),
            "tokens_in": trace.tokens_in,
            "tokens_out": trace.tokens_out,
            "result": {
                "kind": result.kind,
                "reason": result.reason,
                "attribution": result.attribution,
                "expected": result.expected,
                "actual": result.actual,
                "detail": result.detail,
            },
            "trace": trace.to_dict(),
            "stdout": buf.getvalue(),
        })
        # 即时落盘并 flush：全量回归要跑十分钟，中途 Ctrl-C 也得留下已经跑完的
        # 那些用例（哪些过了、哪些挂了、卡在哪条），否则排查只能从头重跑。
        _write_record(request.config, request.config._eval_records[-1])
        return trace, result

    return _run


def _write_record(config, rec: dict) -> None:
    """一条记录一行，写完立即 flush —— 进程被强杀也不会丢已完成的用例。

    文件是**惰性创建**的：第一条用例落盘时才建，同时补写 run_meta。
    这样 `--collect-only`（或环境门禁提前退出）不会留下一个只有 run_meta 的
    空 jsonl —— 那种文件在 `gen_report.py --list` 里全是 0/0 的噪声，
    会让人分不清"这次没跑"和"这次跑了但全军覆没"。
    """
    if config._eval_fh is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config._eval_fh = config._eval_path.open("w", encoding="utf-8")
        config._eval_fh.write(
            json.dumps(config._eval_meta, ensure_ascii=False) + "\n"
        )
    config._eval_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    config._eval_fh.flush()


def pytest_sessionstart(session):
    """run 元信息准备好（先不落盘）：这份成绩单是哪个模型、哪次选择跑出来的，必须可追溯

    落盘时机推迟到第一条用例（见 `_write_record`）。同一 run_id 重跑会覆盖同名文件，
    这正是想要的——run_id 就是这次成绩单的身份。
    """
    config = session.config
    if not hasattr(config, "_eval_records"):
        return
    from src.agent.common.llm import get_model_config

    config._eval_meta = {
        "kind": "run_meta",
        "run_id": config._eval_run_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": "holdout" if config.getoption("--holdout") else "dev",
        "model": get_model_config(),
        "db_readonly": bool(os.getenv("DB_RO_USER")),
    }


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    records = getattr(config, "_eval_records", None)
    if records is None:
        return

    if config._eval_fh is None:
        # 没有任何用例落盘：要么被 --collect-only / 环境门禁提前结束，
        # 要么全部用例在 setup 阶段就出错了。两种情况都没东西可汇报。
        return
    config._eval_fh.close()
    config._eval_fh = None
    out = config._eval_path

    cases = [r for r in records if r["kind"] == "case"]
    if not cases:
        return

    by_kind: dict[str, int] = {}
    for r in cases:
        by_kind[r["result"]["kind"]] = by_kind.get(r["result"]["kind"], 0) + 1
    total = len(cases)
    n_pass = by_kind.get("pass", 0)
    n_fail = by_kind.get("fail", 0)
    n_unver = by_kind.get(UNVERIFIABLE, 0)
    n_unreach = by_kind.get(UNREACHABLE, 0)
    # 模型能力口径的分母：把"没走到被测链路"的摘出去。
    # 存疑的**不摘**——它可能是模型错也可能是断言器判不了，宁可算在分母里拉低分数。
    scorable = total - n_unreach

    print(f"\n{'=' * 66}")
    print(f"run_id={config._eval_run_id}  写出 {out.relative_to(ROOT)}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) + f"  共 {total} 条")
    rate_line = f"  完成率 {n_pass}/{total} = {n_pass / total:.1%}"
    if scorable:
        rate_line += f"   ·   模型能力口径 {n_pass}/{scorable} = {n_pass / scorable:.1%}"
    print(rate_line)
    if n_unreach:
        print(f"  其中 {n_unreach} 条被路由分走（unreachable），不计入模型能力分")
    print(f"  报告：python eval/report/gen_report.py --run-id {config._eval_run_id}")
    print(f"{'=' * 66}")

    if n_unver / total > MAX_UNVERIFIABLE_RATIO:
        # 用 sessionfinish 改 exitstatus 让"可信度不足"本身成为一次失败
        session.exitstatus = 1
        print(
            f"\n[!] 存疑用例 {n_unver}/{total} = {n_unver / total:.0%}，超过上限 "
            f"{MAX_UNVERIFIABLE_RATIO:.0%} —— 这份成绩单的可信度不足，不能用来下结论。\n"
            "    存疑的典型成因：Agent 返回的列与黄金列无交集（列投影差太远）。\n"
            "    先看 jsonl 里这些用例的 result.detail，判断是断言口径该放宽、\n"
            "    还是 Agent 真的没查对列。",
            file=sys.stderr,
        )

    if scorable and n_unreach / total > MAX_UNREACHABLE_RATIO:
        session.exitstatus = 1
        print(
            f"\n[!] 不可达用例 {n_unreach}/{total} = {n_unreach / total:.0%}，超过上限 "
            f"{MAX_UNREACHABLE_RATIO:.0%} —— 路由把太多用例分流出去了。\n"
            "    这多半不是模型问题：先跑 `python eval/tools/audit_intent.py` 看\n"
            "    意图分类把哪些档分到了不走 text2sql 的分支。\n"
            "    注意这道门禁是防「空跑伪装成全绿」：unreachable 不算失败，\n"
            "    若路由全线失灵，整套回归会变成绿色的、什么都没测的一轮。",
            file=sys.stderr,
        )
    elif not scorable:
        session.exitstatus = 1
        print(
            f"\n[!] 全部 {total} 条用例都被路由分走了，没有任何一条走到 text2sql —— "
            "这次跑没有测量价值。先跑 `python eval/tools/audit_intent.py`。",
            file=sys.stderr,
        )

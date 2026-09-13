"""单条问题端到端跑一遍并打印轨迹 —— 调试用例、写归因结论时用

    python eval/tools/run_one.py "深圳宝安区3000元以下的一居室，推荐6套"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.agent_runner import run_case  # noqa: E402
from eval.runner.loader import Assertion, Case  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    question = " ".join(sys.argv[1:])
    case = Case(
        id="adhoc",
        tier="L1",
        question=question,
        gold_sql="",
        assertion=Assertion(mode="id_set"),
    )
    trace = run_case(case)

    print(f"\n{'=' * 70}\n问题：{question}\n{'=' * 70}")
    print(f"ok={trace.ok} error={trace.error}")
    print(f"步数={trace.steps} 时延={trace.latency_ms:.0f}ms "
          f"token(in/out)={trace.tokens_in}/{trace.tokens_out}")

    print(f"\n--- 工具调用（{len(trace.tool_calls)} 次）---")
    for i, call in enumerate(trace.tool_calls, 1):
        arg = call.args.get("query") or call.args.get("table_names") or ""
        # SQL 打印全文——截断会看不出被拒和放行的 SQL 差在哪
        print(f"{i:>2}. [{call.result_kind}] {call.name}\n      {str(arg)}")
        if call.result_kind != "executed":
            print(f"      → {call.result_excerpt[:200]}")

    print(f"\n--- 中断（{len(trace.interrupts)} 次）---")
    for intr in trace.interrupts:
        print(f"  问：{intr['prompt'][:90]}\n  答：{intr['answer']}")

    print(f"\n--- 轨迹指标 ---")
    print(f"  先查 schema 再写 SQL：{trace.schema_queried_first}")
    print(f"  SQL 报错次数：{trace.retry_count}")
    print(f"  安全层拒绝：{trace.refused}")

    print(f"\n--- 最终回答 ---\n{trace.final_output[:1200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

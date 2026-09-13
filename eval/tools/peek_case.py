"""看单条用例的轨迹 —— jsonl 是给机器读的，这个是给人读的

    python eval/tools/peek_case.py smoke-baseline L2-012
    python eval/tools/peek_case.py smoke-baseline L3-001 L4-004

成绩单只告诉你"哪条挂了"，不告诉你"为什么挂"。要看为什么就得翻 jsonl，
而 jsonl 里一条记录含完整 trace + 整段 stdout，几百行没法用眼睛扫。
这个脚本把最该看的四样挑出来：**工具调用序列 / 中断问答 / 最终输出 / stdout 尾部**。

首轮基线靠它定的性：L2-012 打出来是"工具调用序列（0 次）"，一眼就知道
问题不在 SQL 写错，而在根本没走到 SQL 那一步。

**先用 `summary.py` 定位再看这里**：一次跑几十条，挨个 peek 是浪费。
summary 先给"哪些编号、错在哪一类"，再把你指过来的编号交给这个脚本。

    python eval/tools/summary.py full-dec2                    # 先看全貌
    python eval/tools/summary.py full-dec2 --show fail        # 再看某一类
    python eval/tools/peek_case.py full-dec2 L4-006           # 最后看单条

`render_case()` 被 summary.py 复用，所以两处打印的格式必然一致——
改这里的排版，那边跟着变。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_records(run_id: str) -> tuple[dict, list[dict]]:
    """读一个 run 的 jsonl。返回 (run_meta, 用例记录列表)。"""
    path = ROOT / "eval" / "results" / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"没有 {path.relative_to(ROOT)}。用 "
            "`python eval/report/gen_report.py --list` 看有哪些 run"
        )
    meta, cases = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "run_meta":
            meta = rec
        elif rec.get("kind") == "case":
            cases.append(rec)
    return meta, cases


def render_case(rec: dict, *, stdout_tail: int = 25) -> str:
    """把一条用例记录渲染成人读的几段。"""
    out = ["=" * 78]
    out.append(f"[{rec['case_id']}] {rec['question']}")
    out.append(f"  结果={rec['result']['kind']} 归因={rec['result']['attribution']}")
    out.append(f"  断言理由：{rec['result']['reason']}")
    t = rec["trace"]
    out.append(f"  工具调用序列（{len(t['tool_calls'])} 次）：")
    for tc in t["tool_calls"]:
        args = json.dumps(tc["args"], ensure_ascii=False)[:110]
        out.append(f"    {tc['name']:<22} {tc['result_kind']:<9} {args}")
    out.append(f"  中断（{len(t['interrupts'])} 次）：")
    for i in t["interrupts"]:
        out.append(f"    Q: {i['prompt'][:90]}")
        out.append(f"    A: {i['answer']}")
    out.append(f"  最终输出：{t['final_output'][:400]}")
    out.append("  ---- stdout ----")
    tail = rec["stdout"].splitlines()[-stdout_tail:] if stdout_tail else rec["stdout"].splitlines()
    out.append("\n".join(tail))
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    meta, cases = load_records(sys.argv[1])
    del meta
    want = set(sys.argv[2:])
    hit = 0
    for rec in cases:
        if rec["case_id"] not in want:
            continue
        hit += 1
        print(render_case(rec))
    missing = want - {r["case_id"] for r in cases}
    if missing:
        print(f"\n[!] 这些编号不在这次 run 里：{'、'.join(sorted(missing))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

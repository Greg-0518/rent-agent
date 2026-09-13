"""离线重判 —— 拿一份已存盘的 run，用当前的断言器重新判一遍，不重跑 Agent。

    python eval/tools/rescore.py full-dec2              # 只重判，打印新旧对比
    python eval/tools/rescore.py full-dec2 --write      # 另存 full-dec2-rescored.jsonl

**为什么要有这个脚本**：改了断言器/归因码之后，想知道"分数会怎么变"，
直觉做法是重跑一次全量——那是 12 分钟和一次 API 账单，而且**测的是两个变量**：
断言器改了，模型输出也换了（temperature、路由抖动）。分不清是谁导致的差异。

trace 已经存在 jsonl 里了，重判是纯本地的、秒级的、**可复现**的：
同一份 trace 进去，永远同一份判定出来。所以改断言器时先用这个把影响面看清，
确认符合预期再决定要不要花那次重跑。

注意：重判只能反映**断言器**的变化。想让模型输出变化（改 Prompt、改路由），
还是得真跑。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.asserters.rule import assert_case  # noqa: E402
from eval.runner.agent_runner import RunTrace, ToolCallRecord  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
GOLDEN_PATH = ROOT / "eval" / "golden" / "text2sql.json"


def load_golden() -> dict:
    """黄金结果集。路径与 `eval/conftest.py` 的 `golden` fixture 保持一致。"""
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError("缺 eval/golden/text2sql.json，先跑 python eval/tools/make_golden.py")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _trace_from_dict(case_id: str, d: dict) -> RunTrace:
    """把 jsonl 里的 trace 字典还原成 RunTrace。

    只还原断言器真正会读的字段；其余（steps/latency/tokens）置默认值，
    因为它们不参与判定，还原它们只会制造"看起来像但其实是假的"数据。
    """
    return RunTrace(
        case_id=case_id,
        ok=d.get("ok", True),
        error=d.get("error"),
        final_output=d.get("final_output") or "",
        user_intent=d.get("user_intent") or "",
        tool_calls=[ToolCallRecord(name=t.get("name", ""), args=t.get("args") or {},
                                   result_kind=t.get("result_kind", "unknown"),
                                   result_excerpt=t.get("result_excerpt", ""))
                    for t in (d.get("tool_calls") or [])],
        sql_executed=list(d.get("sql_executed") or []),
        sql_errors=list(d.get("sql_errors") or []),
        guard_rejections=list(d.get("guard_rejections") or []),
        interrupts=list(d.get("interrupts") or []),
    )


def _summary(rows: list[dict]) -> Counter:
    return Counter(r["result"]["kind"] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--write", action="store_true", help="另存 <run_id>-rescored.jsonl")
    args = ap.parse_args()

    path = RESULTS_DIR / f"{args.run_id}.jsonl"
    if not path.exists():
        print(f"没有 {path}")
        return 3

    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = next((l for l in lines if l.get("kind") == "run_meta"), {})
    records = [l for l in lines if l.get("kind") == "case"]

    cases = {c.id: c for c in load_cases("text2sql", include_holdout=True)}
    golden = load_golden()

    out, changed = [], []
    for rec in records:
        cid = rec["case_id"]
        case = cases.get(cid)
        gold = golden.get(cid)
        if case is None or gold is None:
            continue  # 用例已被删除/改名，跳过而不是崩

        new = assert_case(case, _trace_from_dict(cid, rec.get("trace") or {}), gold)
        old = rec["result"]
        if (new.kind, new.attribution) != (old.get("kind"), old.get("attribution")):
            changed.append((cid, old.get("kind"), old.get("attribution"),
                            new.kind, new.attribution))
        merged = dict(rec)
        merged["result"] = {"kind": new.kind, "reason": new.reason,
                            "attribution": new.attribution, "expected": new.expected,
                            "actual": new.actual, "detail": new.detail}
        out.append(merged)

    before, after = _summary(records), _summary(out)
    n = len(out)
    print(f"重判 {args.run_id}：{n} 条（Agent 未重跑，只换了断言器）")
    print()
    print(f"{'':6s}{'通过':>6}{'失败':>6}{'存疑':>6}{'不可达':>7}")
    for label, c in (("旧", before), ("新", after)):
        print(f"{label:6s}{c['pass']:>6}{c['fail']:>6}{c['unverifiable']:>6}{c['unreachable']:>7}")
    print()
    for label, c in (("旧", before), ("新", after)):
        scorable = n - c["unreachable"]
        print(f"  {label}：完成率 {c['pass']}/{n} = {c['pass']/n:.1%}"
              f"   模型能力口径 {c['pass']}/{scorable} = {c['pass']/scorable:.1%}")

    if changed:
        print(f"\n判定发生变化的 {len(changed)} 条：")
        for cid, ok, oa, nk, na in changed:
            print(f"  {cid:9s} {ok:12s} {str(oa or '-'):22s} → {nk:12s} {na or '-'}")
    else:
        print("\n没有任何一条判定发生变化。")

    if args.write:
        dest = RESULTS_DIR / f"{args.run_id}-rescored.jsonl"
        with dest.open("w", encoding="utf-8") as f:
            if meta:
                f.write(json.dumps({**meta, "run_id": dest.stem,
                                    "rescored_from": args.run_id}, ensure_ascii=False) + "\n")
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n已另存 {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

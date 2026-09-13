"""一次跑的**索引** —— 先看全貌，再挑着看日志

    python eval/tools/summary.py                          # 最近一次 run
    python eval/tools/summary.py full-dec2                # 指定 run
    python eval/tools/summary.py --list                   # 有哪些 run

看全貌（判定分布 + 每类的编号 + 失败的归因分组）之后，用 `--show` 直接
把某一类的完整日志拉出来，不必一条条敲编号：

    python eval/tools/summary.py full-dec2 --show fail
    python eval/tools/summary.py full-dec2 --show CONDITION_MAPPED_WRONG
    python eval/tools/summary.py full-dec2 --show Edge-001,Edge-004
    python eval/tools/summary.py full-dec2 --show fail --limit 3     # 默认 5，防刷屏
    python eval/tools/summary.py full-dec2 --show fail --limit 0     # 0 = 不限

`--show` 认四种写法，混着用也行：
    · 判定名      pass / fail / unverifiable / unreachable（也认中文：通过/失败/存疑/不可达）
    · 归因码      CONDITION_MAPPED_WRONG 等（见 gen_report.py 的 ATTRIBUTIONS）
    · 编号前缀    L4- 会命中 L4 全档；Edge-001 精确命中
    · all         全部用例（配 --limit 用）

**为什么单独有这么个脚本**：`gen_report.py` 回答"这轮比上轮好还是坏"，
`peek_case.py` 回答"这一条为什么挂"。中间缺一层——"这轮总共挂了哪些、
它们是不是同一个病"。缺了这层，人就只能拿着编号一条条试，
而**同一类错误一次性看完才能看出模式**（本轮 L4 那 5 条正是这么看出
"把口语词扩成一串近义词"这个共同形态的）。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tools.peek_case import load_records, render_case  # noqa: E402

# 判定名 -> (中文标签, 展示顺序)
KINDS = [
    ("pass", "通过"),
    ("fail", "失败"),
    ("unverifiable", "存疑"),
    ("unreachable", "不可达"),
]
KIND_LABEL = dict(KINDS)

# 中文/英文别名都收，敲哪个都行
KIND_ALIAS = {
    "pass": "pass", "通过": "pass", "passed": "pass",
    "fail": "fail", "失败": "fail", "failed": "fail", "failure": "fail",
    "unverifiable": "unverifiable", "存疑": "unverifiable", "unver": "unverifiable",
    "unreachable": "unreachable", "不可达": "unreachable", "unreach": "unreachable",
}

# 归因码从断言器引，不在这里重抄一遍——那边新增分类，这边自动认
from eval.asserters.rule import (  # noqa: E402
    BOUNDARY_OFF_BY_ONE, CONDITION_MAPPED_WRONG, GRAPH_CRASH, GUARD_BYPASS,
    NO_QUERY_ATTEMPT, RESULT_TRANSFORM_WRONG, ROUTED_AWAY, SCHEMA_MISUNDERSTOOD,
    SQL_SYNTAX_ERROR,
)
ATTRIBUTION_CODES = {
    SQL_SYNTAX_ERROR, SCHEMA_MISUNDERSTOOD, CONDITION_MAPPED_WRONG,
    BOUNDARY_OFF_BY_ONE, RESULT_TRANSFORM_WRONG, NO_QUERY_ATTEMPT,
    ROUTED_AWAY, GUARD_BYPASS, GRAPH_CRASH,
}

# 同一类归因下，理由文本还会带每条自己的数字/列表，没法直接分组。
# 取"（"「[」「：」之前的那截当签名——那截是人写的结论，后面的是本条的实测值。
_CUTS = ("（", "[", "：", ":")


def reason_signature(rec: dict) -> str:
    """把一条记录的理由压成一个可分组、可显示的短标签。

    归因码优先（它是断言器给的确定值）；没有归因码的（存疑档）才回去裁理由文本。
    """
    res = rec["result"]
    if res.get("attribution"):
        return res["attribution"]
    text = (res.get("reason") or "").strip()
    for cut in _CUTS:
        idx = text.find(cut)
        if idx > 0:
            return text[:idx]
    return text or "(无理由)"


def latest_run_id() -> str | None:
    d = ROOT / "eval" / "results"
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1].stem if files else None


def list_runs() -> None:
    d = ROOT / "eval" / "results"
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("还没有任何 run。先跑 `pytest eval` 或 `pytest eval -m smoke`。")
        return
    print(f"{'run_id':<22}{'用例数':>7}{'通过':>6}{'失败':>6}{'存疑':>6}{'不可达':>8}  时间")
    print("-" * 74)
    for p in files:
        try:
            meta, cases = load_records(p.stem)
        except Exception:
            continue
        n = defaultdict(int)
        for r in cases:
            n[r["result"]["kind"]] += 1
        print(f"{p.stem:<22}{len(cases):>7}{n['pass']:>6}{n['fail']:>6}"
              f"{n['unverifiable']:>6}{n['unreachable']:>8}  "
              f"{meta.get('started_at', '')}")


def print_overview(run_id: str, cases: list[dict], meta: dict) -> None:
    by_kind = defaultdict(list)
    for rec in cases:
        by_kind[rec["result"]["kind"]].append(rec)

    total = len(cases)
    n_unreach = len(by_kind["unreachable"])
    scorable = total - n_unreach
    n_pass = len(by_kind["pass"])

    print("=" * 78)
    model = (meta.get("model") or {})
    print(f"run_id={run_id}  {meta.get('started_at', '')}  "
          f"selection={meta.get('selection', '?')}  "
          f"model={model.get('model', '?')}(profile={model.get('profile') or '默认'})")
    print("=" * 78)

    print(f"\n【判定分布】共 {total} 条")
    for kind, label in KINDS:
        recs = by_kind.get(kind, [])
        share = f"{len(recs) / total:>5.1%}" if total else "  —  "
        print(f"  {label:<6}{len(recs):>4} 条  {share}")

    rate = f"  完成率 {n_pass}/{total} = {n_pass / total:.1%}" if total else ""
    if scorable:
        rate += f"   模型能力口径 {n_pass}/{scorable} = {n_pass / scorable:.1%}"
    print(rate)
    print("  （完整三维指标见 python eval/report/gen_report.py "
          f"--run-id {run_id}）")

    # 每一类都按"理由"分组列编号——这是这个脚本的主产出
    for kind, label in KINDS:
        recs = by_kind.get(kind, [])
        if not recs or kind == "pass":
            continue
        groups = defaultdict(list)
        for rec in recs:
            groups[reason_signature(rec)].append(rec["case_id"])
        print(f"\n【{label} {len(recs)} 条】按原因分组")
        for sig, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  {sig}   {len(ids)} 条")
            for i in range(0, len(ids), 6):
                print(f"    {'  '.join(ids[i:i + 6])}")

    if by_kind["pass"]:
        ids = [r["case_id"] for r in by_kind["pass"]]
        print(f"\n【通过 {len(ids)} 条】")
        for i in range(0, len(ids), 6):
            print(f"    {'  '.join(ids[i:i + 6])}")

    print("\n看某一类的完整日志：")
    for kind, label in KINDS:
        if by_kind.get(kind) and kind != "pass":
            print(f"  python eval/tools/summary.py {run_id} --show {kind}")


def select_cases(cases: list[dict], selector: str) -> tuple[list[dict], list[str]]:
    """把 --show 的选择器解析成记录列表。返回 (命中, 认不出的词)。"""
    picked: list[dict] = []
    unknown: list[str] = []
    seen: set[str] = set()

    for token in [t.strip() for t in selector.split(",") if t.strip()]:
        hit: list[dict] = []
        low = token.lower()

        if low in ("all", "*"):
            hit = list(cases)
        elif low in KIND_ALIAS:
            kind = KIND_ALIAS[low]
            hit = [r for r in cases if r["result"]["kind"] == kind]
        elif token.upper() in ATTRIBUTION_CODES:
            hit = [r for r in cases
                   if r["result"]["attribution"] == token.upper()]
        else:
            # 编号：精确命中，或前缀命中（`L4-` 要能拉出 L4 全档）
            hit = [r for r in cases if r["case_id"] == token]
            if not hit:
                hit = [r for r in cases if r["case_id"].startswith(token)]
            if not hit:
                # 最后退一步：理由签名的前缀匹配，便于按"话"找而不是按"码"找
                hit = [r for r in cases
                       if token in reason_signature(r)]

        if not hit:
            unknown.append(token)
            continue
        for rec in hit:
            if rec["case_id"] not in seen:
                seen.add(rec["case_id"])
                picked.append(rec)

    picked.sort(key=lambda r: r["case_id"])
    return picked, unknown


def main() -> int:
    ap = argparse.ArgumentParser(
        description="一次跑的索引：先看全貌与编号，再用 --show 拉出某一类的日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("run_id", nargs="?", default=None,
                    help="eval/results/{run_id}.jsonl 的名字；不填取最近一次")
    ap.add_argument("--list", action="store_true", help="列出所有 run 后退出")
    ap.add_argument("--show", default=None,
                    help="判定名 / 归因码 / 编号或编号前缀 / all，逗号分隔")
    ap.add_argument("--limit", type=int, default=5,
                    help="--show 最多渲染几条（默认 5；0 = 不限）")
    ap.add_argument("--stdout-tail", type=int, default=25,
                    help="每条日志带上 stdout 的末几行（默认 25；0 = 全带）")
    args = ap.parse_args()

    if args.list:
        list_runs()
        return 0

    run_id = args.run_id or latest_run_id()
    if not run_id:
        print("还没有任何 run。先跑 `pytest eval -m smoke`。", file=sys.stderr)
        return 3
    try:
        meta, cases = load_records(run_id)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if not cases:
        print(f"{run_id} 里没有用例记录（可能是 --collect-only 或环境门禁提前退出）。",
              file=sys.stderr)
        return 3

    print_overview(run_id, cases, meta)

    if args.show is None:
        return 0

    picked, unknown = select_cases(cases, args.show)
    print("\n" + "=" * 78)
    print(f"--show {args.show}  命中 {len(picked)} 条")
    print("=" * 78)
    if unknown:
        print(f"[!] 认不出的选择器：{'、'.join(unknown)}", file=sys.stderr)
    if not picked:
        return 1

    shown = picked if args.limit <= 0 else picked[:args.limit]
    for rec in shown:
        print()
        print(render_case(rec, stdout_tail=args.stdout_tail))
    if len(picked) > len(shown):
        print(f"\n[已省略 {len(picked) - len(shown)} 条] "
              f"看全部：--limit 0；只看第 6 条起：先 --show 精确编号")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

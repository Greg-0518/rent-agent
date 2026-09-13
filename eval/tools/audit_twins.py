"""孪生对体检 —— 配对是不是真的成立

    python eval/tools/audit_twins.py

孪生的价值全在"控制变量"四个字上：配对的 `gold_sql` 必须**恰好差一个条件**。
一旦这个前提破了，配对就退化成两条各测各的普通用例，而且**不会报错**——
分数照出，只是它不再说明它声称说明的事。这类静默失效最值得用机器盯着。

本工具查 5 条不变式：

  1. 配对完整：每条 Edge 空集用例都有孪生，反之亦然
  2. 原用例空：原用例的黄金结果必须是 0 行（否则它已经不是"空集档"了）
  3. 孪生非空：孪生的黄金结果必须 > 0 行（否则逼不出过度约束）
  4. **只差一个合取项**：两条的 WHERE 合取项集合必须恰好相差一个
  5. 划分一致：配对的 holdout 标记必须相同（否则收官跑只剩半边）
  6. **孪生行数 ≤ room_count**：见下

第 4 条是核心。数据一改（加了几行、某个区变了），第 4 条可能悄悄不成立，
而 yaml 注释里的说明不会跟着变——所以不能靠注释，要靠跑。

第 6 条是踩出来的：Agent 的 SQL 一律带 `LIMIT room_count`，而 `count` 断言
比的是**行数是否相等**。所以孪生黄金若有 22 行、模型却只能取回 10 行，
条件全写对也判 FAIL——那不是模型的问题，是断言在结果被截断时失去了区分力。
更坏的是反过来：模型若丢掉一个条件、又恰好被 LIMIT 截到同一个数，会**假通过**。
所以孪生的行数必须落在 LIMIT 之内，这样"少一行"一定意味着条件写错。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.loader import load_cases  # noqa: E402
from eval.tools.audit_cases import _split_and, _where_clause  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "golden" / "text2sql.json"

# 孪生 id = 原用例 id + "T"。这是配对关系的**唯一**依据，
# 从命名而不是从 yaml 里的自由文本推断，免得靠 notes 里的人话去解析。
TWIN_SUFFIX = "T"


def _conjuncts(sql: str) -> set[str] | None:
    """WHERE 里的合取项集合。归一化空白，免得排版差异被当成"多了一个条件"。"""
    clause, _s, _e = _where_clause(sql)
    if not clause:
        return None
    return {" ".join(c.split()) for c in _split_and(clause)}


def main() -> int:
    if not GOLDEN_PATH.exists():
        print(f"缺 {GOLDEN_PATH.relative_to(ROOT)}，先跑 python eval/tools/make_golden.py")
        return 3

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
    cases = {c.id: c for c in load_cases("text2sql", include_holdout=True)}
    originals = {i: c for i, c in cases.items()
                 if c.tier == "Edge" and not i.endswith(TWIN_SUFFIX)}
    twins = {i: c for i, c in cases.items() if i.endswith(TWIN_SUFFIX)}

    problems: list[str] = []

    orphans = [i for i in originals if i + TWIN_SUFFIX not in twins]
    extra = [i for i in twins if i[:-len(TWIN_SUFFIX)] not in originals]
    if orphans:
        problems.append(f"原用例没有孪生：{'、'.join(sorted(orphans))}")
    if extra:
        problems.append(f"孪生没有原用例：{'、'.join(sorted(extra))}")

    print(f"配对 {len(originals) - len(orphans)} 对\n")
    print(f"{'原用例':<10} {'原行数':>6} {'孪生':<10} {'孪行数':>6}  差异条件")
    print("-" * 74)

    for oid in sorted(originals):
        tid = oid + TWIN_SUFFIX
        if tid not in twins:
            continue
        oc, tc = originals[oid], twins[tid]
        o_rows = (golden.get(oid) or {}).get("row_count")
        t_rows = (golden.get(tid) or {}).get("row_count")

        if o_rows != 0:
            problems.append(f"{oid} 的黄金结果不是空集（{o_rows} 行）——已不再是空集档")
        if not t_rows:
            problems.append(f"{tid} 的黄金结果是空的——孪生逼不出过度约束")
        limit = tc.room_count
        if limit and t_rows and t_rows > limit:
            problems.append(
                f"{tid} 的黄金 {t_rows} 行 > room_count {limit}：Agent 的 LIMIT 只能取回 "
                f"{limit} 行，count 断言会因截断而误判（模型条件写对也 FAIL）"
            )

        oc_set, tc_set = _conjuncts(oc.gold_sql), _conjuncts(tc.gold_sql)
        if oc_set is None or tc_set is None:
            diff = "解析不出 WHERE"
            problems.append(f"{oid}/{tid} 的 gold_sql 缺 WHERE，无法核对控制变量")
        elif len(oc_set) - len(tc_set) != 1 or not tc_set < oc_set:
            removed = oc_set - tc_set
            added = tc_set - oc_set
            if len(removed) == 1 and not added:
                diff = f"删 [{next(iter(removed))}]"
            else:
                diff = f"**不是差一个条件** 删{removed or '{}'} 增{added or '{}'}"
                problems.append(f"{oid}/{tid} 不是控制变量：删 {removed}、增 {added}")
        else:
            diff = f"删 [{next(iter(oc_set - tc_set))}]"

        if bool(oc._holdout) != bool(tc._holdout):
            problems.append(f"{oid}/{tid} 划分不一致：holdout {oc._holdout} vs {tc._holdout}")
            diff += "  **划分不一致**"

        print(f"{oid:<10} {str(o_rows):>6} {tid:<10} {str(t_rows):>6}  {diff}")

    print()
    if problems:
        print(f"[!] {len(problems)} 处问题：")
        for p in problems:
            print(f"    - {p}")
        return 1

    print("配对全部成立：每条原用例空集、每条孪生非空、恰好差一个条件、划分一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

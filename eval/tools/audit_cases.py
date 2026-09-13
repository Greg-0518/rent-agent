"""用例体检 —— 数据够不够、哪些用例其实考不出东西

    python eval/tools/audit_cases.py

回答一个具体问题：**当前数据量下，哪些用例的断言其实没有区分力？**

误判"数据太小所以评测不可信"是错的——50 行配上"结果集精确相等"的断言，
区分力其实很强（差一个条件就是另一组行）。真正的漏洞是**条件冗余**：

    用例问"一楼的、能养宠物的房子"，黄金答案是 {id 5}。
    但全表只有 id 5 能养宠物，所以**漏写"一楼"这个条件，答案一模一样**。
    这条用例看不出来模型有没有理解"一楼"。

体检做两件事：

  1. **答案选择性**：正确答案占全表多少行。占比过高说明区分力弱。
  2. **条件冗余**：逐条删掉 `WHERE` 里的一个条件重跑，若结果集**没变**，
     说明这个条件在当前数据下是冗余的 —— 这条用例检测不出漏写它。

输出按"可修复性"分组：冗余条件如果能靠**补几行数据**变成非冗余，就标出来
（这比笼统地"再爬一万条数据"高效得多）。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.db import normalize_value, run_sql  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402

TABLE_SIZE = 50   # house 表总行数，选择性分母


def _split_and(expr: str) -> list[str]:
    """在**括号深度 0** 处按 AND 切分，并跳过 BETWEEN ... AND ... 里的那个 AND。

    括号深度是关键：`(a OR b) AND c` 必须切成两段而不是三段；
    子查询 `price > (SELECT ... WHERE x)` 里的 WHERE 也不能被切出来。
    """
    parts, buf, depth, pending_between = [], [], 0, False
    i, n = 0, len(expr)
    while i < n:
        and_match = re.match(r"\s+AND\s+", expr[i:], re.IGNORECASE)
        if and_match and depth == 0:
            if pending_between:
                # BETWEEN 的 AND：属于同一个条件，消费掉继续
                buf.append(expr[i:i + and_match.end()])
                pending_between = False
            else:
                parts.append("".join(buf))
                buf = []
            i += and_match.end()
            continue
        if re.match(r"\bBETWEEN\b", expr[i:], re.IGNORECASE):
            pending_between = True
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _where_clause(sql: str) -> tuple[str, int, int]:
    """定位顶层 WHERE 与其结束位置。返回 (子句, 起点, 终点)。"""
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if not m:
        return "", -1, -1
    start = m.end()
    end_m = re.search(
        r"\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$", sql[start:], re.IGNORECASE
    )
    end = start + (end_m.start() if end_m else len(sql[start:]))
    return sql[start:end], start, end


def audit_case(case, golden_entry) -> dict:
    sql = case.gold_sql
    base_rows = golden_entry.get("row_count") or 0
    where, start, end = _where_clause(sql)

    result = {
        "id": case.id,
        "tier": case.tier,
        "selectivity": base_rows / TABLE_SIZE,
        "base_rows": base_rows,
        "conjuncts": [],
        "redundant": [],
        "untestable": False,
    }
    if not where or start < 0:
        return result

    conjuncts = _split_and(where)
    result["conjuncts"] = conjuncts
    if len(conjuncts) < 2:
        # 只有一个条件，删掉就没条件了 —— 无法做冗余检测
        result["untestable"] = True
        return result

    base_ms = _golden_fingerprint(golden_entry)

    for idx, conj in enumerate(conjuncts):
        remaining = [c for j, c in enumerate(conjuncts) if j != idx]
        new_where = " AND ".join(remaining)
        # 拼接要点：start 正好落在 "WHERE" 的最后一个字母后、end 正好落在
        # "ORDER BY" 的第一个字母上，两侧都没有空格可借。直接拼会粘成
        # "WHEREcity" / "x'ORDER BY"。所以显式 strip 两端再补空格。
        new_sql = f"{sql[:start].rstrip()} {new_where} {sql[end:].lstrip()}".rstrip()
        try:
            cols, rows = run_sql(new_sql)
        except Exception as exc:
            result["redundant"].append((conj, f"重跑失败：{type(exc).__name__}: {exc}"))
            continue
        if _fingerprint(cols, rows) == base_ms:
            result["redundant"].append((conj, "删掉后结果集不变"))
    return result


def _fingerprint(cols, rows) -> "Counter | None":
    """结果集的指纹：列名转小写排序后，把每行归一化成元组做多重集。

    列名先排序是为了让 `SELECT a, b` 与 `SELECT b, a` 指纹一致 —— 只删了
    WHERE 条件，SELECT 列表不该变，排序纯粹是防御性写法。

    两边都必须走 `normalize_value`：黄金集落盘时是 JSON（Decimal→str、
    None→null），重跑出来的是原始 Python 类型。不归一化就会因为
    `Decimal('2000') != 2000` 这类差异误报"结果集变了"，把体检结论带偏。
    """
    from collections import Counter

    lowered = [str(c).lower() for c in cols]
    order = sorted(range(len(lowered)), key=lambda i: lowered[i])
    return Counter(
        tuple(normalize_value(r[i]) for i in order) for r in rows
    )


def _golden_fingerprint(entry) -> "object":
    return _fingerprint(entry["columns"], entry["rows"])


_CONST_CACHE: dict[str, bool] = {}


def _is_constant_column(col: str) -> bool:
    """这一列在整表里是否只有一个取值。

    `city` 就是典型：全表都是深圳，所以 `city = '深圳'` 删掉答案不变。
    这类条件下冗余**不是缺陷**——它在数据上就是个常量，抱怨它没意义
    （真要说问题，是"全表只有一个城市"这件事本身限制了城市维度的考点）。
    """
    if col in _CONST_CACHE:
        return _CONST_CACHE[col]
    try:
        _, rows = run_sql(f"SELECT COUNT(DISTINCT `{col}`) FROM house")
        value = int(rows[0][0]) <= 1
    except Exception:
        value = False
    _CONST_CACHE[col] = value
    return value


def _classify(conj: str) -> str:
    """把一条冗余条件归类：benign（数据常量）/ substantive（真的该修）"""
    m = re.match(r"^`?(\w+)`?\s*=\s*'([^']*)'\s*$", conj)
    if m and _is_constant_column(m.group(1)):
        return "benign"
    return "substantive"


def dump_stats() -> None:
    """数据本身的体检：测得了什么、测不了什么

    这些数字决定了哪些**能力维度**根本没法出题，和"行数够不够"是两件事：
      - price 有并列 → 排名类用例的 Top-N 边界会歧义（排名对不对无法判定）
      - 某列有 NULL   → 才测得了 NULL 语义（如 NOT IN 遇 NULL 返回空集）
      - 只有一张表    → JOIN 档无从谈起
    """
    def q(sql):
        return run_sql(sql)[1]

    total = q("SELECT COUNT(*) FROM house")[0][0]
    print("=" * 78)
    print(f"数据体检：house 共 {total} 行")
    print("=" * 78)

    print("\n【列的取值空间】")
    for col in ("id", "district", "bedroom", "price", "floor", "orientation", "city"):
        try:
            d, n, mn, mx = q(
                f"SELECT COUNT(DISTINCT `{col}`), SUM(`{col}` IS NULL),"
                f" MIN(`{col}`), MAX(`{col}`) FROM house"
            )[0]
        except Exception as exc:
            print(f"  {col:<12} 查询失败 {type(exc).__name__}")
            continue
        print(f"  {col:<12} 去重 {d:>4}  NULL {n:>3}  最小 {str(mn):<8} 最大 {str(mx):<8}")

    price_distinct, price_nulls = q(
        "SELECT COUNT(DISTINCT price), SUM(price IS NULL) FROM house"
    )[0]
    print(f"\n  → price 去重 {price_distinct}/{total}："
          f"{'无并列，排名类用例的 Top-N 边界是确定的' if price_distinct == total else '存在并列，Top-N 边界会歧义'}")

    print("\n【区 × 户型 的格子大小】（格子越小，格子内次级属性越容易「整齐」→ 条件冗余）")
    rows = q(
        "SELECT district, bedroom, COUNT(*) c FROM house"
        " GROUP BY district, bedroom ORDER BY c, district"
    )
    from collections import Counter

    size_hist = Counter(r[2] for r in rows)
    print(f"  共 {len(rows)} 个非空格子，大小分布：" +
          "  ".join(f"{k}行×{v}格" for k, v in sorted(size_hist.items())))
    print("  最小的格子：" + "  ".join(f"{r[0]}{r[1]}室({r[2]})" for r in rows[:8]))

    print("\n【设计文档想要的、当前数据给不了的】")
    # 枚举表走裸连接：sql_guard 只放行 SELECT/WITH（information_schema 是 deny 模式，
    # SHOW 也不是 SELECT），这是**正确的**生产约束——Agent 列表用的是 toolkit 自带的
    # sql_db_list_tables。本探测不是候选问句，所以绕开守卫，而不是给守卫开口子。
    from eval.runner.db import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            names = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    print(f"  库里的表：{names}")
    if len(names) < 2:
        print("    → 只有一张表，**JOIN 档无从出题**（设计文档 §4 里的 JOIN 分层）")
    if not price_nulls:
        print("    → price 无 NULL，**NULL 语义测不了**"
              "（项目自己的 check_query 提示里还专门警告了 NOT IN 遇 NULL）")
    print("    → floor 是 varchar（如 '3楼'），数值比较必须 CAST+REPLACE，"
          "这**是**个真考点（已用在 L3-001..004）")


def dump_table() -> None:
    """按「区 × 户型」铺开全表。

    体检发现某条件是冗余的之后，**必须看那个格子才知道补什么数据**：
    冗余的成因几乎都是"这个格子里次级属性是齐的"（整格都有电梯、
    整格楼层都 <10），补一行把这个属性打破即可，不必动其他地方。
    """
    cols, rows = run_sql(
        "SELECT id, district, bedroom, price, floor, orientation,"
        " LEFT(description, 20) FROM house ORDER BY district, bedroom, price"
    )
    print(" | ".join(str(c) for c in cols))
    print("-" * 104)
    prev = None
    for r in rows:
        key = (r[1], r[2])
        if key != prev:
            print("-" * 104)
            prev = key
        print(f"{r[0]:>3} | {r[1]:<4} | {r[2]}室 | {r[3]:>5} | {r[4]:<5} | {r[5]:<4} | {r[6]}")


def main() -> int:
    import json

    if "--table" in sys.argv:
        dump_table()
        return 0
    if "--stats" in sys.argv:
        dump_stats()
        return 0

    golden_path = ROOT / "eval" / "golden" / "text2sql.json"
    if not golden_path.exists():
        print("缺 eval/golden/text2sql.json，先跑 python eval/tools/make_golden.py")
        return 2
    golden = json.loads(golden_path.read_text(encoding="utf-8"))["cases"]

    cases = [c for c in load_cases("text2sql", include_holdout=True)
             if c.assertion.mode in ("id_set", "count")]

    reports = [audit_case(c, golden[c.id]) for c in cases]

    # 冗余要分三类看，否则 58 条里真正该修的 11 条会被噪声埋掉：
    #   结构性 —— Edge 空集档：结果集本来就是空的，删掉任何条件还是空的，
    #             冗余判定对空集天然无意义（这是空集的性质，不是用例的毛病）
    #   benign —— 条件作用在整表只取一个值的列上（city='深圳'），数据上就是常量
    #   实质   —— 剩下的，才是"这条用例考不出它想考的东西"
    substantive, benign, structural = [], [], []
    for r in reports:
        if not r["redundant"]:
            continue
        # Edge 档先短路：空集里删任何条件都还是空集，逐条件分类没有意义，
        # 否则 Edge-003/005/008 会因为"另外还有个非恒定条件"漏进实质清单
        if r["tier"] == "Edge":
            structural.append(r)
            continue
        real = [(c, w) for c, w in r["redundant"] if _classify(c) == "substantive"]
        (substantive if real else benign).append(r | {"real": real})

    untestable = [r for r in reports if r["untestable"]]
    high_sel = [r for r in reports if r["selectivity"] >= 0.5]

    print("=" * 78)
    print(f"用例体检：{len(cases)} 条（id_set + count 档，可做冗余检测的）")
    print("=" * 78)

    print(f"\n【1】实质冗余 —— {len(substantive)} 条用例「删掉某条件答案不变」，")
    print("     它检测不出模型漏写那个条件。这是当前数据量下真正该修的清单：")
    for r in substantive:
        print(f"\n  {r['id']} ({r['tier']}, 答案 {r['base_rows']} 行)")
        for conj, why in r["real"]:
            print(f"     冗余条件: {conj}   ← {why}")

    print(f"\n\n【2】被判冗余但不该修的 —— {len(benign) + len(structural)} 条")
    print(f"     数据常量档 {len(benign)} 条（条件作用在整表只有一个取值的列上，如 city）")
    if benign:
        consts = sorted({c for r in benign for c, _ in r["redundant"]})
        print(f"       涉及条件：{'  '.join(consts)}")
    print(f"     Edge 空集档 {len(structural)} 条 —— 结果集本来就是空的，")
    print("       删掉任何条件它还是空的，冗余判定对空集天然无意义（是空集的性质，不是用例的毛病）")

    print(f"\n\n【3】只有单个条件、无法检测冗余 —— {len(untestable)} 条")
    print("     " + "  ".join(r["id"] for r in untestable))

    print(f"\n\n【4】答案占全表 ≥50%（区分力偏弱）—— {len(high_sel)} 条")
    for r in high_sel:
        print(f"     {r['id']}: {r['base_rows']}/{TABLE_SIZE} = {r['selectivity']:.0%}")

    sizes = sorted(r["base_rows"] for r in reports)
    print(f"\n\n【5】答案行数分布：最小 {sizes[0]}  中位 {sizes[len(sizes)//2]}  最大 {sizes[-1]}")
    print(f"     答案 ≤2 行的用例 {sum(1 for s in sizes if s <= 2)} 条"
          "（越少越具体，但越容易因冗余条件而失灵）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

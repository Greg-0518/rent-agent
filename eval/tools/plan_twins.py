"""空集用例的孪生规划 —— 给每条 Edge 用例找出"删掉哪个条件就能出结果"

    python eval/tools/plan_twins.py

**为什么要孪生**：空集用例的通过是弱通过。`count=0` 只看"返回了几行"，
所以模型只要返回 0 行就算过——哪怕它把条件整个丢了、恰好还是空集，
甚至它根本没理解问句。一条空集用例单看是证明不了什么的。

配一条**只差一个条件、结果非空**的孪生之后，这对才成为真信号：

    原用例期望 0 行，模型返回 5 行   → 模型丢了条件（Edge-008 就是这么挂的）
    孪生期望 5 行，模型返回 0 行     → 模型过度约束（把条件写严了）

**控制变量**：孪生不是"重新出题"，而是把原 `gold_sql` 的 WHERE
**恰好删掉一个合取项**。删哪个不是随便挑的——本工具把每个合取项逐个删掉重跑，
列出哪些删除能出结果、各出几行，再由人来挑一个写成问句。

不能自动生成问句：删掉条件之后那句话得用自然语言说得通
（"深圳罗湖区的房子"可以说，"罗湖区 AND city=深圳"合成一句就绕了）。
所以工具负责枚举，人负责措辞 —— 分工在"可判定"和"需要判断"之间。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.db import run_sql  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402
from eval.tools.audit_cases import _split_and, _where_clause  # noqa: E402


def _remove_conjunct(sql: str, nth: int) -> str | None:
    """删掉 WHERE 里第 nth 个合取项，返回新 SQL。

    这是 `audit_cases.py` 里条件冗余体检的同一套手术：括号感知地切 AND，
    再拼回去。**只删一个**，删多了就不是控制变量了。
    """
    clause, start, end = _where_clause(sql)
    # `_where_clause` 的 start 落在 **WHERE 关键字之后**，所以 sql[:start]
    # 自带 "WHERE"——这里绝不能再拼一个，否则得到 `WHERE WHERE`，
    # 每条改出来的 SQL 都是语法错误（我第一版就这么写的）。
    if not clause:
        return None
    conj = _split_and(clause)
    if nth >= len(conj) or len(conj) < 2:
        return None
    kept = [c for i, c in enumerate(conj) if i != nth]
    new_where = " AND ".join(kept)
    return f"{sql[:start].rstrip()} {new_where} {sql[end:].lstrip()}".rstrip()


def _count(sql: str) -> tuple[int, str]:
    try:
        _cols, rows = run_sql(sql)
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {str(exc)[:70]}"
    return len(rows), ""


def main() -> int:
    cases = [c for c in load_cases("text2sql", include_holdout=True) if c.tier == "Edge"]
    if not cases:
        print("没有 Edge 用例")
        return 1

    print(f"Edge 用例 {len(cases)} 条，逐条枚举「删掉哪个条件能出结果」：\n")
    planned = 0
    for c in cases:
        sql = getattr(c, "gold_sql", None)
        if not sql:
            print(f"[{c.id}] 没有 gold_sql，跳过")
            continue
        base, err = _count(sql)
        print(f"[{c.id}] {c.question}")
        print(f"    现状：{base if base >= 0 else 'ERR'} 行"
              + (f"  {err}" if err else "（空集，符合预期）"))

        clause, _s, _e = _where_clause(sql)
        conj = _split_and(clause) if clause else []
        options, broken = [], []
        for nth in range(len(conj)):
            new_sql = _remove_conjunct(sql, nth)
            if not new_sql:
                continue
            n, err = _count(new_sql)
            if n > 0:
                options.append((nth, conj[nth], n, new_sql))
            elif n < 0:
                # 改出来的 SQL 跑不通 = **本工具的 bug**，不是"这条用例没孪生"。
                # 两者混在一起报会把工具缺陷说成数据性质，所以分开。
                broken.append((conj[nth], err))

        if broken:
            print(f"    [!] 本工具有 bug：改出来的 SQL 跑不通 {len(broken)} 条 "
                  f"（不是数据的问题）：{broken[0][1]}")
            print()

        if not options:
            print("    [!] 删掉任何一个条件都还是空集 —— 这条用例的孪生得换思路"
                  "（可能要放宽阈值而不是删条件）")
            print()
            continue

        # 行数最少的那个通常最贴身：它离"恰好出一点点结果"最近，
        # 模型稍有过度约束就会掉回 0 行，最能逼出问题。
        options.sort(key=lambda o: o[2])
        for nth, cond, n, _sql in options:
            star = " ←建议" if nth == options[0][0] else ""
            print(f"    删掉 [{cond.strip()}] → {n} 行{star}")
        planned += 1
        print()

    print(f"可规划孪生 {planned}/{len(cases)} 条。")
    print("下一步：挑一个删除项，把剩下的条件写成自然问句，"
          "以 assert.mode=count 加进 edge_twin.yaml（黄金用 make_golden.py 生成）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

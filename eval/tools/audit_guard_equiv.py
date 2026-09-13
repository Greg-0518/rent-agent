"""SQL 守卫的等价性审计：拿**改之前**那版守卫，跟当前工作区这版逐条对比结论。

用途：改 `src/agent/common/sql_guard.py` 之后，在提交前确认"既有判定一条都没变"。
比重新跑一遍评测便宜得多——不调模型、不连数据库，只把历史轨迹里真正过过守卫的
SQL（含黄金 SQL）全部重放一遍，逐条比 `allowed / rule / 改写后的 SQL`。

    .\\venv\\Scripts\\python.exe eval\\tools\\audit_guard_equiv.py            # 对比 HEAD 与工作区
    .\\venv\\Scripts\\python.exe eval\\tools\\audit_guard_equiv.py --old HEAD~1
    .\\venv\\Scripts\\python.exe eval\\tools\\audit_guard_equiv.py --show-diffs

**注意**：它对比的是"改前 vs 改后"，所以要在**提交之前**跑；
提交之后再跑就是拿同一版比自己，输出必然是 0（那时它只是个恒真检查）。

输出里 `结论不同` 未必是坏事——修掉误判时本来就该有差异，但每一条都要能解释；
`旧版允许 / 新版拒绝` 这种方向才是必须逐条交代的（那是收紧，可能误杀真用例）。
"""

import argparse
import glob
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agent.common import sql_guard as new_guard  # noqa: E402

OLD_MODULE_PATH = REPO / ".tmp_guard_old.py"

# 本次收紧要修的四种失效模式（两种误拒、一种篡改、一种静默改语义）。
# 放在工具里而不是只写在文档里：下次改守卫时，这几条会被立刻重跑出来。
SAMPLES = (
    ("分号在字面量里（误拒 · multi_statement）",
     "SELECT id FROM house WHERE description LIKE '%;%'"),
    ("LIMIT 在字面量里（篡改 · 把 '%LIMIT 1000%' 改成 '%LIMIT 50%'）",
     "SELECT id FROM house WHERE description LIKE '%LIMIT 1000%'"),
    ("写关键字在字面量里（误拒 · deny_keyword:DROP）",
     "SELECT id FROM house WHERE description LIKE '%DROP%'"),
    ("子查询 LIMIT 被当顶层收紧（静默改语义）",
     "SELECT id, price FROM house WHERE id IN (SELECT id FROM house LIMIT 999)"),
)


def load_old(ref: str):
    """把 ref 里的旧版守卫作为独立模块加载（不污染 sys.modules 里的当前版）。"""
    src = subprocess.run(
        ["git", "show", f"{ref}:src/agent/common/sql_guard.py"],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO,
    )
    if src.returncode != 0:
        print(f"取不到 {ref} 版本的 sql_guard.py：{src.stderr.strip()}")
        raise SystemExit(2)
    OLD_MODULE_PATH.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_guard_old", OLD_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_sql() -> list[str]:
    """历史轨迹里真正执行过的 SQL + final_sql + 黄金 SQL。"""
    sqls: list[str] = []
    for path in sorted(glob.glob(str(REPO / "eval/results/*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            trace = rec.get("trace") or {}
            sqls.extend(trace.get("sql_executed") or [])
            final = ((rec.get("result") or {}).get("detail") or {}).get("final_sql")
            if final:
                sqls.append(final)
    golden_file = REPO / "eval/golden/text2sql.json"
    if golden_file.exists():
        golden = json.loads(golden_file.read_text(encoding="utf-8"))
        for entry in golden.values():
            if isinstance(entry, dict) and entry.get("gold_sql"):
                sqls.append(entry["gold_sql"])
    return list(dict.fromkeys(sqls))


def main() -> int:
    ap = argparse.ArgumentParser(description="SQL 守卫改动前后的等价性审计")
    ap.add_argument("--old", default="HEAD", help="旧版所在的 git ref（默认 HEAD）")
    ap.add_argument("--show-diffs", action="store_true", help="打印每一条差异")
    args = ap.parse_args()

    old_guard = load_old(args.old)

    # 先跑内置样例：这四条是本次要修的失效模式，正好当"改了什么"的活文档
    print(f"对比 {args.old} 与工作区\n")
    print("— 内置样例（本次要修的四种失效模式）—")
    for label, sql in SAMPLES:
        o = old_guard.validate_sql(sql)
        n = new_guard.validate_sql(sql)
        mark = "≠" if (o.allowed, o.rule, o.sql) != (n.allowed, n.rule, n.sql) else "="
        print(f"  {mark} {label}")
        print(f"      旧: allowed={o.allowed} rule={o.rule!r} sql={o.sql[:90]!r}")
        print(f"      新: allowed={n.allowed} rule={n.rule!r} sql={n.sql[:90]!r}")

    sqls = collect_sql()
    print(f"\n— 真实数据重放 —\n重放 SQL {len(sqls)} 条（已去重）")

    loosened, tightened, other = [], [], []
    for sql in sqls:
        o = old_guard.validate_sql(sql)
        n = new_guard.validate_sql(sql)
        if (o.allowed, o.rule, o.sql) == (n.allowed, n.rule, n.sql):
            continue
        row = (sql, (o.allowed, o.rule, o.sql), (n.allowed, n.rule, n.sql))
        if not o.allowed and n.allowed:
            loosened.append(row)          # 变宽松：要确认是修误判，不是开了口子
        elif o.allowed and not n.allowed:
            tightened.append(row)         # 变严格：要确认不会误杀真用例
        else:
            other.append(row)

    print(f"  结论相同 {len(sqls) - len(loosened) - len(tightened) - len(other)} 条")
    print(f"  变宽松 {len(loosened)} 条   ← 修误判应有它；但要逐条看是不是放开了该拦的")
    print(f"  变严格 {len(tightened)} 条  ← 收紧，逐条确认不会误杀")
    print(f"  其它   {len(other)} 条   ← 只是改写结果不同（如 LIMIT 位置）")

    if args.show_diffs:
        for label, rows in (("变宽松", loosened), ("变严格", tightened), ("其它", other)):
            for sql, o, n in rows:
                print(f"\n[{label}] {sql[:160]}\n   旧: {o}\n   新: {n}")
    else:
        for label, rows in (("变宽松", loosened), ("变严格", tightened), ("其它", other)):
            for sql, o, n in rows[:5]:
                print(f"  [{label}] {sql[:100]}  → 旧 {o[0]}/{o[1]} 新 {n[0]}/{n[1]}")

    try:
        OLD_MODULE_PATH.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

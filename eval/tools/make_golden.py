"""黄金结果集生成 —— 执行用例的 gold_sql，把结果集落盘

**黄金结果集由执行产生，不手抄。** 用例 yaml 里只写人写的、可复核的 `gold_sql`；
结果集与派生断言值（scalar / count）都由本脚本执行后写入 `eval/golden/{module}.json`。
手抄结果集必然出错，且出错后无法复核。

    python eval/tools/make_golden.py              # 全部用例（含保留测试集）
    python eval/tools/make_golden.py --review     # 生成后再打印人工复核表
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.db import normalize_value, run_sql  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402

GOLDEN_DIR = ROOT / "eval" / "golden"
MODULES = ("text2sql",)


def build_golden(module: str) -> dict:
    # 保留测试集也要黄金结果集——它只是"不参与日常迭代"，不是不评测
    cases = load_cases(module, include_holdout=True)
    entries: dict[str, dict] = {}
    failures: list[str] = []

    for case in cases:
        if case.assertion.mode == "refusal":
            # 拒绝档本来就不该有可执行的结果集
            entries[case.id] = {"mode": "refusal", "gold_sql": case.gold_sql}
            continue
        try:
            cols, rows = run_sql(case.gold_sql)
        except Exception as exc:
            failures.append(f"{case.id}: {type(exc).__name__}: {exc}")
            continue

        norm_rows = [[normalize_value(v) for v in row] for row in rows]
        entry = {
            "mode": case.assertion.mode,
            "gold_sql": case.gold_sql,
            "columns": cols,
            "rows": norm_rows,
            "row_count": len(norm_rows),
            # 约定：**答案是 gold_sql 结果第一行的第一列**——写黄金 SQL 时把要断言的
            # 值放在第 1 列（如「哪个区最贵」→ SELECT district, AVG(price) ... LIMIT 1）
            "scalar": norm_rows[0][0] if norm_rows else None,
        }
        entries[case.id] = entry

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "case_count": len(entries),
        "db": {"database": _db_name(), "house_rows": _house_rows()},
        "cases": entries,
    }
    return payload | {"_failures": failures}


def _db_name() -> str:
    import os

    return f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def _house_rows() -> int | None:
    try:
        _, rows = run_sql("SELECT COUNT(*) FROM house")
        return int(rows[0][0])
    except Exception:
        return None


def review_table(module: str, payload: dict) -> None:
    """人工复核表：黄金 SQL 与它的结果集并排看，便于一眼发现写错的 SQL"""
    print(f"\n{'=' * 78}\n人工复核：{module}（{payload['case_count']} 条）\n{'=' * 78}")
    for cid, entry in payload["cases"].items():
        if entry.get("mode") == "refusal":
            print(f"\n[{cid}] 拒绝档（无结果集，断言走了安全层拒绝分支）")
            continue
        print(f"\n[{cid}] 结果集 {entry['row_count']} 行 · 列 {entry['columns']}")
        print(f"  期望值(scalar) = {entry['scalar']!r}")
        if entry["row_count"]:
            head = entry["rows"][:3]
            print(f"  前 3 行: {head}{' ...' if entry['row_count'] > 3 else ''}")


def main() -> int:
    do_review = "--review" in sys.argv
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for module in MODULES:
        payload = build_golden(module)
        failures = payload.pop("_failures")

        out = GOLDEN_DIR / f"{module}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{module}] 写出 {out.relative_to(ROOT)} —— {payload['case_count']} 条用例")

        if failures:
            exit_code = 1
            print(f"[{module}] {len(failures)} 条黄金 SQL 执行失败（这些用例没有黄金结果集）：")
            for f in failures:
                print(f"    {f}")

        if do_review:
            review_table(module, payload)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""把仓库里 sql/*.sql 的迁移语句执行到数据库上 —— 人工跑的，评测链路不碰它。

    .\venv\Scripts\python.exe eval\tools\apply_sql.py sql\001_normalize_decoration_wording.sql
    .\venv\Scripts\python.exe eval\tools\apply_sql.py sql\001_normalize_decoration_wording.sql --yes

**为什么要有这个文件**：R5 §8.11 给 id 45 归一化定的前置条件是
"把这条 UPDATE 落成仓库里的一个 sql/ 文件"。如果只有 .sql 文件、没有可复现的执行方式，
那个文件就只是一段没人能安全执行、也没法核对影响行数的文本——等于"改共享库不留记录"
换了个形式。所以执行入口和 SQL 一起进仓库。

**默认空跑（不带 --yes）**：只打印要执行的语句，不连库、不提交。
带 --yes 才真正执行，且**每步都报告 affected rows**——迁移的返回值必须是可核对的数字，
"执行成功了"和"改了 1 行"是两回事。

**连接用的是可写账号 `DB_USER`**，不是评测链路用的只读账号 `DB_RO_USER`：
迁移本来就该由人用可写凭据执行。反过来说，**这个脚本不该出现在任何自动化链路里**
（评测侧执行 SQL 一律走 eval/runner/db.py，那条路有 sql_guard 且只读）。

语句切分是最朴素的"去掉 `--` 注释行后按 `;` 切"：够用，也让 .sql 文件必须写成
一条语句一个分号。刻意**不**吞异常、不做部分提交——迁移要么整份成功，要么整份回滚。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)


def split_statements(text: str) -> list[str]:
    """去掉 `--` 注释行与空行，按 `;` 切出非空语句。"""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("--")]
    body = "\n".join(lines)
    return [s.strip() for s in re.split(r";", body) if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="执行 sql/*.sql 迁移（默认空跑）")
    ap.add_argument("sql_file", help="相对仓库根目录或绝对路径的 .sql 文件")
    ap.add_argument("--yes", action="store_true", help="真正执行并提交；缺省只打印")
    args = ap.parse_args()

    path = Path(args.sql_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"[apply_sql] 找不到文件：{path}")
        return 2

    statements = split_statements(path.read_text(encoding="utf-8"))
    if not statements:
        print(f"[apply_sql] {path.name} 里没有可执行语句")
        return 2

    print(f"[apply_sql] {path.relative_to(ROOT)} → {len(statements)} 条语句")
    for i, st in enumerate(statements, 1):
        print(f"  --- [{i}] ---")
        print("  " + st.replace("\n", "\n  "))

    if not args.yes:
        print("\n[apply_sql] 空跑结束，**未连库、未提交**。确认无误后加 --yes 重跑。")
        return 0

    user = os.getenv("DB_USER")
    if not user:
        print("[apply_sql] .env 里没有 DB_USER（可写账号），拒绝执行")
        return 2

    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=user,
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        autocommit=False,
    )
    try:
        with conn.cursor() as cur:
            for i, st in enumerate(statements, 1):
                cur.execute(st)
                print(f"[apply_sql] 语句 {i}: affected rows = {cur.rowcount}")
        conn.commit()
        print("[apply_sql] 已提交。")
    except Exception as e:  # noqa: BLE001 —— 迁移失败要原样抛出，不做部分提交
        conn.rollback()
        print(f"[apply_sql] 失败，已回滚：{type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

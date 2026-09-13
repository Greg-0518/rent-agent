"""评测侧只读 DB 访问 —— 黄金结果集生成与结果集比对共用

评测自己执行 SQL 时也过一遍 sql_guard：既复用同一套安全规则，
也保证"评测要跑的东西"和"生产允许跑的东西"是同一个集合。
"""

import os
import pymysql
from dotenv import load_dotenv

from src.agent.common.sql_guard import validate_sql

load_dotenv(override=True)  # 与 common/llm.py 一致：.env 权威


class SQLRejected(RuntimeError):
    """被安全层拦下的 SQL —— 调用方据此判定 L5 拒绝分支"""

    def __init__(self, reason: str, rule: str):
        super().__init__(f"{rule}: {reason}")
        self.reason = reason
        self.rule = rule


def connect():
    """建连接。优先用只读账号（DB_RO_USER），缺省回落到主账号。"""
    user = os.getenv("DB_RO_USER") or os.getenv("DB_USER")
    password = os.getenv("DB_RO_PASSWORD") or os.getenv("DB_PASSWORD")
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=user,
        password=password,
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )


def run_sql(sql: str, *, enforce_guard: bool = True) -> tuple[list[str], list[tuple]]:
    """执行一条 SQL，返回 (列名, 行列表)。

    enforce_guard=False 只用于黄金结果集生成——黄金 SQL 是人写的、可信的，
    但仍然会被记录，方便人工复核时看到实际执行的是什么。
    """
    if enforce_guard:
        verdict = validate_sql(sql)
        if not verdict.allowed:
            raise SQLRejected(verdict.reason, verdict.rule)
        sql = verdict.sql

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = list(cur.fetchall())
    finally:
        conn.close()
    return cols, rows


def normalize_value(v):
    """把 Decimal / date 等归一成可比较可哈希的值（多重集比对要 hashable）。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    try:
        from decimal import Decimal

        if isinstance(v, Decimal):
            return round(float(v), 2)
    except ImportError:  # pragma: no cover
        pass
    return str(v).strip()

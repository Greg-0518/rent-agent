"""SQL 安全层单测 —— 不需要数据库，纯字符串规则验证"""

import pytest

from src.agent.common.sql_guard import (
    DEFAULT_MAX_ROWS,
    REJECT_MARKER,
    rejection_message,
    validate_sql,
)


# ---------- 放行：合法只读查询 ----------

def test_simple_select_gets_limit_injected():
    v = validate_sql("SELECT id, price FROM house WHERE city = '西安'")
    assert v.allowed
    assert v.sql.endswith(f"LIMIT {DEFAULT_MAX_ROWS}")
    assert v.rule == "forced_limit"


def test_existing_small_limit_untouched():
    v = validate_sql("SELECT * FROM house LIMIT 10")
    assert v.allowed
    assert v.sql == "SELECT * FROM house LIMIT 10"
    assert v.rule == ""


def test_oversized_limit_clamped():
    v = validate_sql("SELECT * FROM house LIMIT 1000")
    assert v.allowed
    assert v.sql == f"SELECT * FROM house LIMIT {DEFAULT_MAX_ROWS}"


def test_offset_limit_clamped_keeps_offset():
    v = validate_sql("SELECT * FROM house LIMIT 20, 1000")
    assert v.allowed
    assert v.sql == f"SELECT * FROM house LIMIT 20, {DEFAULT_MAX_ROWS}"


def test_offset_form_beyond_regex_group():
    v = validate_sql("SELECT * FROM house LIMIT 1000 OFFSET 5")
    assert v.allowed
    assert v.sql == f"SELECT * FROM house LIMIT {DEFAULT_MAX_ROWS} OFFSET 5"


def test_cte_is_allowed():
    v = validate_sql("WITH t AS (SELECT * FROM house) SELECT * FROM t")
    assert v.allowed


def test_trailing_semicolon_and_comment_ok():
    v = validate_sql("SELECT * FROM house; -- 查一下")
    assert v.allowed


def test_column_names_containing_keywords_not_flagged():
    """update_time / deleted 这类列名不能被词边界规则误伤"""
    v = validate_sql("SELECT update_time, deleted FROM house WHERE deleted = 0")
    assert v.allowed


def test_order_by_not_mistaken_for_orders_table():
    """回归：ORDER BY 里的 ORDER 曾被当成 orders 表，导致任何排序查询都被拒"""
    v = validate_sql("SELECT id, price FROM house ORDER BY price ASC LIMIT 6")
    assert v.allowed


def test_group_by_and_other_keyword_collisions_survive():
    for sql in (
        "SELECT district, COUNT(*) FROM house GROUP BY district ORDER BY COUNT(*) DESC",
        "SELECT * FROM house ORDER BY price DESC, bedroom ASC",
        "SELECT id FROM house WHERE description LIKE '%电梯%' ORDER BY id",
    ):
        assert validate_sql(sql).allowed, sql


def test_user_function_not_mistaken_for_user_table():
    v = validate_sql("SELECT USER() AS current_user FROM house")
    assert v.allowed


def test_replace_function_not_mistaken_for_write_statement():
    """回归：REPLACE 既是写语句又是字符串函数。

    `floor` 是 varchar（值形如 "3楼"），做楼层数值比较必须用 REPLACE 去掉"楼"字。
    曾经的一刀切让**所有涉及楼层的问题在生产链路上都答不了**——
    黄金 SQL 生成时才暴露（deny_keyword:REPLACE）。
    """
    for sql in (
        "SELECT id, floor FROM house WHERE CAST(REPLACE(floor, '楼', '') AS UNSIGNED) >= 30",
        "SELECT REPLACE(description, '无电梯', '') FROM house",
        "SELECT id FROM house WHERE REPLACE (floor, '楼', '') = '3'",
        "SELECT id FROM house WHERE CAST(REPLACE/*去楼*/ (floor, '楼', '') AS UNSIGNED) > 5",
    ):
        assert validate_sql(sql).allowed, sql


def test_replace_into_still_rejected():
    """豁免函数形式不能把 REPLACE INTO 放进来——写语句必须照拦"""
    for sql in (
        "REPLACE INTO house (id, price) VALUES (1, 100)",
        "REPLACE house SET price = 1",
        "REPLACE/*c*/INTO house (id) VALUES (1)",
    ):
        v = validate_sql(sql)
        assert not v.allowed, sql
        assert v.rule == "deny_keyword:REPLACE", f"{sql} → {v.rule}"


# ---------- 拒绝：写操作与多语句 ----------

@pytest.mark.parametrize("sql,rule_prefix", [
    ("INSERT INTO house (price) VALUES (1)", "deny_keyword:INSERT"),
    ("UPDATE house SET price = 1", "deny_keyword:UPDATE"),
    ("DELETE FROM house", "deny_keyword:DELETE"),
    ("DROP TABLE house", "deny_keyword:DROP"),
    ("TRUNCATE TABLE house", "deny_keyword:TRUNCATE"),
    ("SELECT 1; DROP TABLE house", "multi_statement"),
])
def test_write_operations_rejected(sql, rule_prefix):
    v = validate_sql(sql)
    assert not v.allowed
    assert v.rule.startswith(rule_prefix)


def test_empty_rejected():
    assert not validate_sql("   ").allowed
    assert not validate_sql("-- 只有注释").allowed


# ---------- 拒绝：越权与注入 ----------

def test_sensitive_table_rejected():
    v = validate_sql("SELECT phone FROM users")
    assert not v.allowed
    assert v.rule == "deny_table:users"


def test_sensitive_table_in_join_rejected():
    v = validate_sql("SELECT h.id FROM house h JOIN users u ON h.id = u.house_id")
    assert not v.allowed
    assert v.rule == "deny_table:users"


def test_sensitive_table_in_comma_join_rejected():
    v = validate_sql("SELECT h.id FROM house h, orders o WHERE h.id = o.house_id")
    assert not v.allowed
    assert v.rule == "deny_table:orders"


def test_schema_qualified_table_name_rejected():
    v = validate_sql("SELECT * FROM house_prd.users")
    assert not v.allowed
    assert v.rule == "deny_table:users"


def test_information_schema_rejected():
    v = validate_sql("SELECT table_name FROM information_schema.tables")
    assert not v.allowed
    assert v.rule.startswith("deny_pattern:")


def test_outfile_rejected():
    v = validate_sql("SELECT * INTO OUTFILE '/tmp/x' FROM house")
    assert not v.allowed


def test_sleep_injection_rejected():
    v = validate_sql("SELECT SLEEP(10)")
    assert not v.allowed


def test_comment_obfuscation_fails_safe():
    """关键字被注释拆开时会落在 not_select 分支——失败方向是拒绝，不是放行"""
    assert not validate_sql("SEL/**/ECT * FROM house").allowed


def test_deny_tables_env_override(monkeypatch):
    monkeypatch.setenv("SQL_GUARD_DENY_TABLES", "house_secret")
    assert not validate_sql("SELECT * FROM house_secret").allowed
    # 覆盖后 users 不再被拦
    assert validate_sql("SELECT phone FROM users").allowed


def test_max_rows_env_override(monkeypatch):
    monkeypatch.setenv("SQL_GUARD_MAX_ROWS", "5")
    v = validate_sql("SELECT * FROM house")
    assert v.sql.endswith("LIMIT 5")


# ---------- 拒绝消息：评测侧靠标记判定拒绝分支 ----------

def test_rejection_message_carries_marker():
    msg = rejection_message(validate_sql("DELETE FROM house"))
    assert REJECT_MARKER in msg
    assert "deny_keyword:DELETE" in msg

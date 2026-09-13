"""验证 sql_guard 真的挂在 sql_db_query 的执行路径上

用假工具代替 SqlDatabaseToolkit，**不需要数据库**。
补这个文件的原因：`node/recommend.py` 的接线原先只靠端到端实跑 + 肉眼看 stdout 验证，
没有任何自动化测试——安全层一旦被改回裸工具，单测应该立刻报警。
"""

import pytest
from langchain_core.tools import tool  # noqa: F401  （保留 import 以便复现历史坑）

from src.agent.common.sql_guard import REJECT_MARKER
from src.agent.node.recommend import _build_guarded_query_tool


class _FakeRawTool:
    """冒充 SQLDatabaseToolkit 里的 sql_db_query"""

    def __init__(self):
        self.calls: list[dict] = []

    def invoke(self, payload):
        self.calls.append(payload)
        return "[(1, '温馨单间', 550)]"


@pytest.fixture
def raw():
    return _FakeRawTool()


def test_wrapper_keeps_name_and_schema(raw):
    """包装后名字与入参 schema 必须与原工具一致，否则 bind_tools / ToolNode 会错位"""
    t = _build_guarded_query_tool(raw)
    assert t.name == "sql_db_query"
    assert "query" in t.args


def test_legit_sql_passes_through_and_gets_limit(raw):
    t = _build_guarded_query_tool(raw)
    out = t.invoke({"query": "SELECT id FROM house"})
    assert out == "[(1, '温馨单间', 550)]"
    # 真正落到 DB 的是注入 LIMIT 之后的 SQL，不是原句
    assert raw.calls[0]["query"].endswith("LIMIT 50")


def test_dangerous_sql_never_reaches_the_database(raw):
    """安全层是真正的闸口：拒绝时底层工具一次都不能被调用"""
    t = _build_guarded_query_tool(raw)
    out = t.invoke({"query": "DELETE FROM house"})
    assert REJECT_MARKER in out
    assert raw.calls == []


def test_sensitive_table_never_reaches_the_database(raw):
    t = _build_guarded_query_tool(raw)
    out = t.invoke({"query": "SELECT phone FROM users"})
    assert REJECT_MARKER in out
    assert "deny_table:users" in out
    assert raw.calls == []


@pytest.mark.parametrize("sql", [
    "SELECT id, price FROM house ORDER BY price ASC LIMIT 6",
    "SELECT district, COUNT(*) FROM house GROUP BY district ORDER BY COUNT(*) DESC",
])
def test_regression_order_by_reaches_database(raw, sql):
    """回归：ORDER BY 曾被误判成 orders 表，导致任何排序查询都被拒"""
    t = _build_guarded_query_tool(raw)
    out = t.invoke({"query": sql})
    assert REJECT_MARKER not in out
    assert len(raw.calls) == 1


def test_none_raw_tool_yields_none():
    """DB 不可用时返回 None，让上层走占位节点（不能在这里崩掉 import）"""
    assert _build_guarded_query_tool(None) is None

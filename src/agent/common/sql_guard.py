"""SQL 执行安全层 —— 只读白名单 + 强制 LIMIT + 敏感表黑名单

挂载点在**工具层**（`node/recommend.py` 包住 sql_db_query），不是连接串层：
LLM 生成的 SQL 不管从哪条路径进来，执行前都要过这里。

对应设计文档 §4.7（前置必改项），也是 L5 档「越界/恶意」用例成立的前提——
没有这一层，L5 的十条危险语句会真的被执行。

环境变量：
    SQL_GUARD_MAX_ROWS      单条查询最大返回行数，默认 50
    SQL_GUARD_DENY_TABLES   敏感表黑名单（逗号分隔），设置后**覆盖**默认名单
"""

import os
import re
from dataclasses import dataclass

DEFAULT_MAX_ROWS = 50

# 允许的起始关键字。WITH 也放行：CTE 最终仍落到 SELECT，且是 text2sql 的常见写法，
# 一刀切掉会系统性误杀 L2/L3 档用例。（此处相对设计文档 §4.7 的"必须 SELECT 开头"有放宽，
# 真正的防线是下面的关键字/模式黑名单。）
ALLOWED_LEADING = ("SELECT", "WITH")

# 写操作 / DDL / DCL —— 出现在任何位置都拒绝。
# 用 \b 词边界，`update_time`、`deleted` 这类列名不会误伤。
DENY_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "RENAME", "GRANT", "REVOKE", "MERGE", "CALL", "EXECUTE",
    "HANDLER", "LOCK", "UNLOCK", "SAVEPOINT", "PREPARE", "DEALLOCATE",
)

# 双关关键字：既是写语句、又是内置函数名。**紧跟 `(` 时按函数调用放行**。
#
# REPLACE 是唯一的这类关键字：`REPLACE INTO t ...` 是写语句，而
# `REPLACE(floor, '楼', '')` 是字符串函数。一刀切会误杀，而且误杀得很隐蔽——
# 本项目实际踩过：楼层比较必须用 REPLACE 去掉"楼"字，于是**所有涉及楼层的问题
# 在生产链路上全部答不了**（黄金 SQL 生成时才发现，`deny_keyword:REPLACE`）。
# 与 ORDER BY 命中 orders 表属同一类：关键字扫描没区分语法位置。
#
# 只对这一个关键字开豁免，不做成通用规则：其余关键字后面跟 `(` 在只读查询里
# 本就不可能合法出现（`CALL(`、`EXECUTE(` 也不是合法 MySQL 写法），
# 通用豁免只会白白放宽被接受的语言。
FUNCTION_LIKE_KEYWORDS = ("REPLACE",)

# 文件读写 / 拖库 / 延时注入
DENY_PATTERNS = (
    r"INTO\s+OUTFILE", r"INTO\s+DUMPFILE", r"LOAD_FILE", r"LOAD\s+DATA",
    r"SLEEP\s*\(", r"BENCHMARK\s*\(", r"GET_LOCK",
    r"INFORMATION_SCHEMA", r"PERFORMANCE_SCHEMA",
    r"MYSQL\.", r"SYS\.",
)

# 敏感表黑名单：用户/订单/支付相关表默认不可查
DEFAULT_DENY_TABLES = (
    "users", "user", "orders", "order", "payment", "payments",
    "account", "accounts", "admin", "contract", "contracts",
)

# 拒绝标记：评测侧靠它判定"走了拒绝分支"（设计文档 §4.3 的 followed_path）
REJECT_MARKER = "[SQL_GUARD_REJECTED]"


@dataclass
class SQLVerdict:
    """校验结论。allowed=False 时 rule 给出触发的规则名，供评测归因四分类使用。"""

    allowed: bool
    sql: str                 # 通过校验后真正要执行的 SQL（LIMIT 已被注入/收紧）
    reason: str = ""         # 拒绝原因，会作为工具消息回灌给 LLM
    rule: str = ""           # 规则名，如 not_select / deny_keyword:DELETE / forced_limit


def _max_rows() -> int:
    try:
        return int(os.getenv("SQL_GUARD_MAX_ROWS", str(DEFAULT_MAX_ROWS)))
    except ValueError:
        return DEFAULT_MAX_ROWS


def _deny_table_set() -> set[str]:
    """敏感表黑名单集合。SQL_GUARD_DENY_TABLES 设置后覆盖默认名单。"""
    raw = os.getenv("SQL_GUARD_DENY_TABLES")
    if raw:
        return {t.strip().lower() for t in raw.split(",") if t.strip()}
    return {t.lower() for t in DEFAULT_DENY_TABLES}


def deny_tables() -> tuple[str, ...]:
    """敏感表黑名单（排序后的元组，便于打印与测试断言）。"""
    return tuple(sorted(_deny_table_set()))


def _strip_comments(sql: str) -> str:
    """去掉注释再扫描 —— 否则 /* */ 能被用来把关键字拆开绕过检测。"""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"#[^\n]*", " ", sql)
    return sql


def _norm_ident(raw: str) -> str:
    """标识符归一：去反引号/双引号/方括号、去库名前缀、转小写。"""
    ident = raw.strip().strip("`\"[]")
    return ident.split(".")[-1].lower()


def _referenced_tables(sql: str) -> set[str]:
    """抽取 FROM / JOIN 引用的表名。

    **只认表位置**，不做全句关键字扫描——否则 `ORDER BY` 里的 ORDER 会被当成
    orders 表、`USER()` 函数会被当成 user 表（本项目实际踩过这个坑：
    任何带 ORDER BY 的查询都被误拒，模型只能反复重试同一句 SQL）。
    """
    tables: set[str] = set()

    # FROM/JOIN 后紧跟的标识符
    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+([`\"\[]?[A-Za-z_][\w.]*[`\"\]]?)", sql, re.IGNORECASE
    ):
        tables.add(_norm_ident(m.group(1)))

    # 逗号连接的多表：FROM a, b, c
    for m in re.finditer(
        r"\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|$)",
        sql,
        re.IGNORECASE | re.DOTALL,
    ):
        for part in m.group(1).split(","):
            head = part.strip().split()[0] if part.strip() else ""
            if re.fullmatch(r"[`\"\[]?[A-Za-z_][\w.]*[`\"\]]?", head):
                tables.add(_norm_ident(head))

    return tables


def _enforce_limit(sql: str, max_rows: int) -> tuple[str, bool]:
    """没有 LIMIT 就注入，有但超限就收紧。返回 (SQL, 是否改动过)。

    只处理**最后一个** LIMIT，即当作顶层 LIMIT——子查询里带 LIMIT 的写法
    （如 `WHERE id IN (SELECT id FROM t LIMIT 10)`）会被误收紧，属已知简化。
    """
    # 注意 \s* 只在逗号组内出现：否则 `LIMIT 1000 OFFSET 5` 里的空格会被吞掉，
    # 替换后变成 `LIMIT 50OFFSET 5`
    pattern = re.compile(r"\bLIMIT\s+(\d+)(?:\s*,\s*(\d+))?", re.IGNORECASE)
    matches = list(pattern.finditer(sql))

    if not matches:
        return f"{sql} LIMIT {max_rows}", True

    last = matches[-1]
    # `LIMIT offset, n` 取第二个数（行数）；`LIMIT n` 取第一个
    rows = int(last.group(2) if last.group(2) is not None else last.group(1))
    if rows <= max_rows:
        return sql, False

    replacement = f"LIMIT {last.group(1)}, {max_rows}" if last.group(2) is not None else f"LIMIT {max_rows}"
    return f"{sql[:last.start()]}{replacement}{sql[last.end():]}", True


def validate_sql(sql: str, max_rows: int | None = None) -> SQLVerdict:
    """校验一条准备执行的 SQL。只读语句返回改写后的 SQL，其余一律拒绝。"""
    raw = (sql or "").strip()
    if not raw:
        return SQLVerdict(False, raw, "SQL 为空", "empty")

    # 语句本体：去注释 → 去末尾分号
    body = _strip_comments(raw).strip().rstrip(";").strip()
    if not body:
        return SQLVerdict(False, raw, "SQL 去注释后为空", "empty")

    # 多语句：分号只允许出现在末尾（`SELECT 1; DROP TABLE x` 走这条）
    if ";" in body:
        return SQLVerdict(False, raw, "检测到多条语句，已拒绝执行", "multi_statement")

    upper = body.upper()

    # 先扫具体规则再判起始关键字：`DELETE FROM house` 这种整句写操作要归到
    # deny_keyword:DELETE 而不是笼统的 not_select——评测归因靠这个粒度。
    for kw in DENY_KEYWORDS:
        if kw in FUNCTION_LIKE_KEYWORDS:
            # 允许 `REPLACE(a, b, c)` 这种函数调用，但 `REPLACE INTO t` 仍要拦：
            # 判据是关键字后面**紧跟**左括号（先去注释，`REPLACE/*x*/(...)` 也躲不掉）
            hit = re.search(rf"\b{kw}\b(?!\s*\()", upper)
        else:
            hit = re.search(rf"\b{kw}\b", upper)
        if hit:
            return SQLVerdict(False, raw, f"检测到写操作关键字 {kw}，已拒绝执行", f"deny_keyword:{kw}")

    for pat in DENY_PATTERNS:
        if re.search(pat, upper):
            return SQLVerdict(False, raw, f"检测到危险模式 {pat}，已拒绝执行", f"deny_pattern:{pat}")

    for tbl in _referenced_tables(body):
        if tbl in _deny_table_set():
            return SQLVerdict(False, raw, f"表 {tbl} 在访问黑名单中，已拒绝执行", f"deny_table:{tbl}")

    head = re.match(r"[A-Z_]+", upper)
    head_kw = head.group(0) if head else "?"
    if head_kw not in ALLOWED_LEADING:
        return SQLVerdict(
            False, raw, f"只允许只读查询（SELECT/WITH 开头），实际以 {head_kw} 开头", "not_select"
        )

    limited, changed = _enforce_limit(body, max_rows or _max_rows())
    return SQLVerdict(
        True, limited, "已强制注入/收紧 LIMIT" if changed else "", "forced_limit" if changed else ""
    )


def rejection_message(verdict: SQLVerdict) -> str:
    """把拒绝结论包成回灌给 LLM 的工具消息（带标记，评测可 grep）。"""
    return (
        f"{REJECT_MARKER} 查询被安全层拦截：{verdict.reason}\n"
        f"规则：{verdict.rule}\n"
        "只允许生成只读的 SELECT 查询；请改写后重试，或告知用户该操作不被允许。"
    )

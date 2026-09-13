"""SQL 执行安全层 —— 只读白名单 + 强制 LIMIT + 敏感表黑名单

挂载点在**工具层**（`node/recommend.py` 包住 sql_db_query），不是连接串层：
LLM 生成的 SQL 不管从哪条路径进来，执行前都要过这里。

对应设计文档 §4.7（前置必改项），也是 L5 档「越界/恶意」用例成立的前提——
没有这一层，L5 的十条危险语句会真的被执行。

**扫描方式**：所有关键字/模式/表名/分号/LIMIT 的判断都在一份"掩码"上做——
把字符串字面量与注释按**等长**替换成空格后的副本（反引号见 `_mask` 的说明：
标识符不掩码，否则表黑名单可被绕过）。
原来直接在原文上跑正则，会同时产生两个方向的错：

| 现象 | 后果 |
|---|---|
| `LIKE '%;%'` 里的分号被当成多语句 | **误拒**——描述里带分号就查不了 |
| `LIKE '%DROP%'` 里的关键字被当成写操作 | 误拒 |
| `LIKE '%LIMIT 1000%'` 里的 LIMIT 被当成真的 LIMIT | **改写进了字面量**，等于篡改 SQL 语义 |
| `WHERE id IN (SELECT id FROM t LIMIT 999)` | 子查询的 LIMIT 被当顶层收紧（旧版已注明的已知简化） |

等长是关键：LIMIT 的收紧仍要按 offset 改**原文**，掩码只用来回答
"这个匹配在不在字面量里"。

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


def _mask(sql: str) -> str:
    """把**字符串字面量**与注释的内部字符替换成空格，长度不变。

    手写单趟扫描而不是正则：正则无法正确处理"引号里的引号"——
    `'%--%'` 会被 `--[^\\n]*` 当成行注释，`'it''s'`（SQL 里的转义单引号）也判不准。
    扫描时保留引号本身（`'abc'` → `'   '`），保持可读性，且引号不是关键字。

    长度不变是硬约束：LIMIT 收紧要按 offset 改原文，掩码只负责回答
    "这个匹配在不在字面量里"。

    **两处刻意不掩码**（都实测过代价，别顺手"补全"）：

    · **反引号**：它是标识符而不是字面量。掩掉内容会让
      `SELECT * FROM \\`users\\`` 里的表名消失、从而绕过表黑名单——
      这是我在实现过程中真的踩出来的绕过（`FROM \\`users\\`` 一度变成放行）。
      代价是列名正好叫 `update` 时仍会被 `deny_keyword` 误拒，
      但方向是**误拒**（安全），且本库没有这种列名。
    · **`#` 行注释**：`#` 出现在反引号标识符里（如 \\`a#b\\`）会把**本行剩下的一切**
      掩成空格，包括后面的 `FROM users`——那是 fail-open。而本项目的 SQL 里
      从来没出现过 `#` 注释（336 条历史轨迹里 0 次），所以直接不认它：
      真出现 `#` 会被当成普通字符参与扫描，最坏是误拒，不会漏拦。
    """
    out = list(sql)
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"":
            quote = ch
            i += 1
            while i < n:
                if sql[i] == "\\":
                    # 反斜杠转义（MySQL 默认开启），把转义对一起吃掉，避免 \' 被当成收尾
                    out[i] = " "
                    if i + 1 < n:
                        out[i + 1] = " "
                    i += 2
                    continue
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        # SQL 标准的 '' 转义：跳过一对，不算收尾
                        out[i] = out[i + 1] = " "
                        i += 2
                        continue
                    i += 1
                    break
                out[i] = " "
                i += 1
            continue
        if ch == "-" and sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end < 0 else end + 2
            for k in range(i, end):
                out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


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


def _top_level_limit_spans(masked: str) -> list[re.Match]:
    """找出**括号深度为 0** 的 LIMIT 位置。

    旧版取"最后一个 LIMIT"当顶层，于是 `WHERE id IN (SELECT id FROM t LIMIT 999)`
    这种写法里的子查询 LIMIT 会被当成顶层收紧——改的是子查询的行数上限，
    和"最多返回多少行给用户"根本不是一回事，属静默改语义。
    """
    depth = 0
    out: list[re.Match] = []
    for m in re.finditer(r"[()]|\bLIMIT\b", masked, re.IGNORECASE):
        token = m.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(m)
    return out


def _enforce_limit(sql: str, masked: str, max_rows: int) -> tuple[str, bool]:
    """没有顶层 LIMIT 就注入，有但超限就收紧。返回 (SQL, 是否改动过)。

    `sql` 是原文（改写落在它身上），`masked` 是等长掩码（只用来定位）。
    只认**顶层** LIMIT：子查询里的不动，`LIKE '%LIMIT 1000%'` 这类字面量更不动。
    """
    # 注意 \s* 只在逗号组内出现：否则 `LIMIT 1000 OFFSET 5` 里的空格会被吞掉，
    # 替换后变成 `LIMIT 50OFFSET 5`
    pattern = re.compile(r"LIMIT\s+(\d+)(?:\s*,\s*(\d+))?", re.IGNORECASE)

    spans = _top_level_limit_spans(masked)
    if not spans:
        return f"{sql} LIMIT {max_rows}", True

    parsed = []
    for span in spans:
        m = pattern.match(masked, span.start())
        if m:
            parsed.append(m)

    if not parsed:
        # 有 LIMIT 关键字但后面不是字面数字（如 `LIMIT ?`）：注入会造出两条 LIMIT，
        # 交给 DB 报错比静默放行好——这里选择不注入、原样执行。
        return sql, False

    last = parsed[-1]
    # `LIMIT offset, n` 取第二个数（行数）；`LIMIT n` 取第一个
    rows = int(last.group(2) if last.group(2) is not None else last.group(1))
    if rows <= max_rows:
        return sql, False

    replacement = f"LIMIT {last.group(1)}, {max_rows}" if last.group(2) is not None else f"LIMIT {max_rows}"
    return f"{sql[:last.start()]}{replacement}{sql[last.end():]}", True


def _trim_trailing_comments(raw: str, masked: str) -> str:
    """去掉末尾的注释与空白，返回**原文**的这段前缀。

    注入 LIMIT 必须落在这之后：`SELECT * FROM house; -- 查一下` 若直接追加，
    会变成 `... -- 查一下 LIMIT 50`——LIMIT 被注释吃掉，等于没加。
    判据只看掩码是否为空（掩码里只有注释/字面量内部会是空格），
    遇到换行就停：`-- 行注释\\nLIMIT 50` 是合法的。
    """
    end = len(raw.rstrip())
    while end > 0 and raw[end - 1] != "\n" and masked[end - 1] == " ":
        end -= 1
    body = raw[:end].rstrip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    return body


def validate_sql(sql: str, max_rows: int | None = None) -> SQLVerdict:
    """校验一条准备执行的 SQL。只读语句返回改写后的 SQL，其余一律拒绝。

    所有"找关键字/找表名/找分号/找 LIMIT"都在掩码上做（见模块 docstring），
    只有 LIMIT 的**改写**落在原文上。
    """
    raw = (sql or "").strip()
    if not raw:
        return SQLVerdict(False, raw, "SQL 为空", "empty")

    masked = _mask(raw)
    body_masked = masked.strip().rstrip(";").strip()
    if not body_masked:
        return SQLVerdict(False, raw, "SQL 去注释后为空", "empty")

    # 多语句：分号只允许出现在末尾（`SELECT 1; DROP TABLE x` 走这条）。
    # 掩码里的分号一定是真的分号——`LIKE '%;%'` 那种现在不再被误判。
    if ";" in body_masked:
        return SQLVerdict(False, raw, "检测到多条语句，已拒绝执行", "multi_statement")

    upper = body_masked.upper()

    # 先扫具体规则再判起始关键字：`DELETE FROM house` 这种整句写操作要归到
    # deny_keyword:DELETE 而不是笼统的 not_select——评测归因靠这个粒度。
    for kw in DENY_KEYWORDS:
        if kw in FUNCTION_LIKE_KEYWORDS:
            # 允许 `REPLACE(a, b, c)` 这种函数调用，但 `REPLACE INTO t` 仍要拦：
            # 判据是关键字后面**紧跟**左括号（注释在掩码里已是空格，
            # `REPLACE/*x*/(...)` 与 `REPLACE (floor, ...)` 都躲不掉也误伤不了）
            hit = re.search(rf"\b{kw}\b(?!\s*\()", upper)
        else:
            hit = re.search(rf"\b{kw}\b", upper)
        if hit:
            return SQLVerdict(False, raw, f"检测到写操作关键字 {kw}，已拒绝执行", f"deny_keyword:{kw}")

    for pat in DENY_PATTERNS:
        if re.search(pat, upper):
            return SQLVerdict(False, raw, f"检测到危险模式 {pat}，已拒绝执行", f"deny_pattern:{pat}")

    for tbl in _referenced_tables(body_masked):
        if tbl in _deny_table_set():
            return SQLVerdict(False, raw, f"表 {tbl} 在访问黑名单中，已拒绝执行", f"deny_table:{tbl}")

    head = re.match(r"[A-Z_]+", upper)
    head_kw = head.group(0) if head else "?"
    if head_kw not in ALLOWED_LEADING:
        return SQLVerdict(
            False, raw, f"只允许只读查询（SELECT/WITH 开头），实际以 {head_kw} 开头", "not_select"
        )

    body = _trim_trailing_comments(raw, masked)
    # 掩码与原文等长，但 body 是去掉尾部注释后的前缀——把掩码截到同样长度，
    # LIMIT 的位置才和 body 对得上（尾部注释里不可能有顶层 LIMIT）
    limited, changed = _enforce_limit(body, masked[:len(body)], max_rows or _max_rows())
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

"""SQL 比较符的定位与取反 —— 边界审计与断言器**共用同一份**。

这里是唯一一处"数 WHERE 里第几个比较符"的实现。审计工具和断言器各自数一遍
迟早就错位（起止索引对不上，翻转出来的 SQL 是坏的，而且坏得很安静：
跑得通、只是答的不是同一句话），所以两边都必须调这里的函数。

`<>`（不等于）和 `<=>`（NULL 安全等于）**不是边界比较**，只是恰好含尖括号。
把 `<>` 里的 `<` 换成 `<=` 会得到合法的 `<=>`，于是"取反后结果变了"——
一条凭空捏造的边界敏感用例。所以先整词识别，再过滤掉非边界符号。
"""

import re

# 比较符及其"取反"写法。顺序要紧：先长后短，否则 `>=` 会被 `>` 先吃掉。
FLIPS = [(">=", ">"), ("<=", "<"), (">", ">="), ("<", "<=")]

# 所有含 `<` / `>` 的符号，**必须先按长到短识别**（`<=>` 在最前）
OP_TOKEN = re.compile(r"<=>|<>|!=|>=|<=|>|<")

# 只有这些才谈得上"边界"
BOUNDARY_OPS = frozenset({">=", ">", "<=", "<"})

# 问句里出现这些词 = 边界含义**明确**，不需要改问句
UNAMBIGUOUS = ("高于", "超过", "不低于", "不少于", "大于", "低于", "小于", "不多于")
# 出现这些词 = 边界含义**可有两种读法**
AMBIGUOUS = ("以上", "以下", "以内", "左右")


def where_span(sql: str) -> tuple[int, int]:
    """WHERE 子句的范围 —— 只在这个范围里动比较符，
    免得改坏 `LIMIT 10`（没有比较符）或 SELECT 列表里的表达式。

    没有 WHERE 时返回 (-1, -1)。
    """
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if not m:
        return -1, -1
    tail = re.search(r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT|HAVING)\b", sql[m.end():], re.IGNORECASE)
    end = m.end() + tail.start() if tail else len(sql)
    return m.end(), end


def boundary_ops(sql: str) -> list[tuple[int, int, str]]:
    """WHERE 子句里真正的边界比较符：(起, 止, 符号)。"""
    start, end = where_span(sql)
    if start < 0:
        return []
    return [(m.start(), m.end(), m.group())
            for m in OP_TOKEN.finditer(sql, start, end)
            if m.group() in BOUNDARY_OPS]


def flip_nth(sql: str, nth: int, dst: str) -> str | None:
    """把 WHERE 里第 nth 个边界比较符换成 dst，其余不动。

    nth 越界返回 None（调用方据此判断"没有第 nth 个"）。
    """
    ops = boundary_ops(sql)
    if nth >= len(ops):
        return None
    s, e, _ = ops[nth]
    return sql[:s] + dst + sql[e:]


def flip_variants(sql: str) -> list[tuple[str, str]]:
    """黄金 SQL 的每个"取反一个比较符"变体：[(说明, 变体 SQL), ...]。

    说明形如 `>= → >`，用于日志与归因理由。
    """
    out = []
    for nth, (_s, _e, original) in enumerate(boundary_ops(sql)):
        for src, dst in FLIPS:
            if src != original:
                continue
            flipped = flip_nth(sql, nth, dst)
            if flipped:
                out.append((f"{original} → {dst}", flipped))
            break
    return out


def boundary_verdict(question: str) -> str:
    """问句措辞是否明确了边界口径：明确 / 歧义 / 无边界词。"""
    if any(w in question for w in UNAMBIGUOUS):
        return "明确"
    if any(w in question for w in AMBIGUOUS):
        return "歧义"
    return "无边界词"

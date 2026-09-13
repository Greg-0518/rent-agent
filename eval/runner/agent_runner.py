"""Agent 执行器 —— 跑一条用例，产出执行轨迹（trace）

两个关键设计：

1. **不改生产代码就能拿到 checkpointer**：`src/agent/graph.py` 里的 `builder` 是模块级
   对象，harness 直接重新 `compile(checkpointer=..., store=...)`。`interrupt()` 没有
   checkpointer 会直接报错，而生产侧的 `graph` 是靠 `langgraph dev` 运行时注入的。
   （直接复用 builder 而不是重写一套节点连线，避免 harness 与生产图漂移。）

2. **轨迹从 checkpointer 的最终 state 里读**，不靠解析流式分片：子图的 messages 会合并
   进父图 state，所以一次 `get_state` 就能拿到全链路消息（工具调用、执行过的 SQL、token）。
"""

import time
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.agent.common.sql_guard import REJECT_MARKER

# **每一轮**允许的最大步数 / 最大中断次数——防止图跑飞把整轮回归挂住。
# 注意"每一轮"这个限定：多轮用例的轮数会让总步数成倍增长，拿总步数做上限等于
# 用轮数稀释保护阈值。实测单轮就是 6 步（推荐链路）或 3 步（不走工具），留了足够余量。
MAX_STEPS = 40
MAX_RESUMES = 6

INTERRUPT_KEY = "__interrupt__"


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result_kind: str = "unknown"   # executed | rejected | error
    result_excerpt: str = ""


@dataclass
class RunTrace:
    case_id: str
    ok: bool = True
    error: str | None = None
    final_output: str = ""
    # 图里的意图分类结果（`node/main.py:identify_question` 写入 state）。
    # 断言器靠它区分两件长得很像但归因完全相反的事：
    #   意图 != recommend_house 且没执行 SQL → **路由分走了**，与模型能力无关
    #   意图 == recommend_house 且没执行 SQL → 模型该查库却没查，是模型失败
    # 只靠"零工具调用"分辨不出这两种，必须拿图里的真值。
    user_intent: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sql_executed: list[str] = field(default_factory=list)
    sql_errors: list[str] = field(default_factory=list)
    guard_rejections: list[dict] = field(default_factory=list)
    interrupts: list[dict] = field(default_factory=list)
    steps: int = 0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    # ---- 多轮用例专用（单轮一律为 0 / 空）----
    # 已经**尝试过**的轮数。0 = 单轮用例。
    n_turns: int = 0
    # 崩在哪一轮。**任何一轮崩都算失败**，哪怕后面几轮照跑完了——D6 就是崩在
    # 入口节点（第一轮）上的，只看最后一轮的轨迹会把它漏掉。
    turn_errors: list[str] = field(default_factory=list)
    # 每一轮各自执行成功的 SQL 条数。这不只是诊断信息，它让"只看最后一轮"这件事
    # **可核对**：同会话第 2 轮应当只算第 2 轮那条 SQL，而不是第 1+2 轮的和。
    # （若不切片，第 2 轮的 sql_executed 会带上第 1 轮的 SQL，
    #   而 _judge_executed 会回头认前面任意一条 —— 于是"最后一轮答错"可能被
    #   前面某轮答对的 SQL 判成 PASS。切片就是为了堵这个假通过。）
    turn_sql_counts: list[int] = field(default_factory=list)

    @property
    def schema_queried_first(self) -> bool:
        """是否先查了 schema 再写 SQL（设计文档 §4.3 的过程正确性信号）"""
        names = [c.name for c in self.tool_calls]
        if "sql_db_query" not in names:
            return False
        return names.index("sql_db_schema") < names.index("sql_db_query")

    @property
    def retry_count(self) -> int:
        """SQL 报错后的重试次数"""
        return len(self.sql_errors)

    @property
    def refused(self) -> bool:
        """是否走了安全层拒绝分支（L5 档靠这个判定）"""
        return bool(self.guard_rejections)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            schema_queried_first=self.schema_queried_first,
            retry_count=self.retry_count,
            refused=self.refused,
        )
        return d


def build_graph(checkpointer=None, store=None):
    """把生产图的 builder 重新编译，挂上 checkpointer（interrupt 必需）与 store。"""
    from src.agent.graph import builder

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        store=store or InMemoryStore(),
    )


def _answer_for_interrupt(prompt: str, case) -> str:
    """脚本化应答。人不在环里，但中断是生产行为，必须如实走完。"""
    text = str(prompt)
    # 注意 prompt 里用的是「预订」（node/main.py:55），别只匹配「预定」
    if "预订" in text or "预定" in text:
        return "不需要"

    params = case.params or {}
    parts: list[str] = []
    if params.get("city"):
        parts.append(str(params["city"]))
    if params.get("district"):
        parts.append(str(params["district"]))

    bmin, bmax = params.get("budget_min"), params.get("budget_max")
    if bmin is not None and bmax is not None:
        parts.append(f"预算{bmin}到{bmax}元每月")
    elif bmax is not None:
        parts.append(f"预算{bmax}元以下")
    elif bmin is not None:
        parts.append(f"预算{bmin}元以上")

    if params.get("room_type"):
        parts.append(str(params["room_type"]))
    if params.get("others"):
        parts.append(str(params["others"]))

    return "，".join(parts) if parts else "不提供"


def _collect_usage(messages) -> tuple[int, int]:
    tin = tout = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if isinstance(usage, dict):
            tin += int(usage.get("input_tokens") or 0)
            tout += int(usage.get("output_tokens") or 0)
    return tin, tout


def _pair_tool_results(messages) -> list[ToolCallRecord]:
    """把 AIMessage 的 tool_calls 与随后的 ToolMessage 结果配对。"""
    from langchain_core.messages import ToolMessage

    results: dict[str, object] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            results[str(m.tool_call_id)] = m

    records: list[ToolCallRecord] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            content = ""
            tm = results.get(str(tc.get("id")))
            if tm is not None:
                content = str(tm.content)
            if REJECT_MARKER in content:
                kind = "rejected"
            elif content.lstrip().lower().startswith("error") or "Error:" in content[:200]:
                kind = "error"
            else:
                kind = "executed"
            records.append(
                ToolCallRecord(
                    name=tc.get("name", "?"),
                    args=tc.get("args") or {},
                    result_kind=kind,
                    result_excerpt=content[:300],
                )
            )
    return records


def run_case(case, graph=None) -> RunTrace:
    """跑一条用例直到图自然结束（含中断-恢复循环）。"""
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": f"eval-{case.id}"}}
    context = {"user_id": f"eval-{case.id}"}

    trace = RunTrace(case_id=case.id)
    started = time.time()

    try:
        _run_turn(graph, config, context, case.question, case, trace)

        snapshot = graph.get_state(config)
        messages = list(snapshot.values.get("messages") or [])
        records = _pair_tool_results(messages)
        trace.tokens_in, trace.tokens_out = _collect_usage(messages)
        _fill_outcome(trace, messages, records, str(snapshot.values.get("user_intent") or ""))

    except Exception as exc:  # 图跑挂也算一种结果，不能让整轮回归中断
        trace.ok = False
        trace.error = f"{type(exc).__name__}: {exc}"

    trace.latency_ms = (time.time() - started) * 1000
    return trace


def _run_turn(graph, config, context, question, answer_source, trace) -> None:
    """把**一轮**跑到底（含该轮自己的中断-恢复循环）。

    `answer_source` 只要有 `.params` 即可（`Case` 与 `Turn` 都满足）——
    中断追问必须拿**这一轮**的参数回答，不能全用最后一轮的。
    """
    payload: dict | Command = {"messages": [HumanMessage(content=question)]}
    steps = 0        # 本轮步数（上限按轮算，见 MAX_STEPS 的说明）
    resumes = 0
    while True:
        pending: Command | None = None
        for _chunk in graph.stream(
            payload, config=config, context=context, stream_mode="updates"
        ):
            steps += 1
            trace.steps += 1
            if steps > MAX_STEPS:
                raise RuntimeError(f"单轮超过最大步数 {MAX_STEPS}，疑似图跑飞")
            if isinstance(_chunk, dict) and INTERRUPT_KEY in _chunk:
                intr = _chunk[INTERRUPT_KEY][0]
                answer = _answer_for_interrupt(intr.value, answer_source)
                trace.interrupts.append({"prompt": str(intr.value)[:200], "answer": answer})
                pending = Command(resume=answer)
        if pending is None:
            break
        resumes += 1
        if resumes > MAX_RESUMES:
            raise RuntimeError(f"单轮超过最大中断次数 {MAX_RESUMES}，疑似中断循环")
        payload = pending


def _fill_outcome(trace, messages, records, user_intent: str) -> None:
    """把某**一轮**的消息解析成判定用的结果字段。

    只对**最后一轮**调用（多轮时前面几轮只统计 token 与 SQL 条数）——
    否则判定会因为"前面某轮答对过"而给整个用例记 PASS，见 turn_sql_counts 的说明。
    """
    trace.user_intent = user_intent
    trace.tool_calls = records

    for rec in records:
        if rec.name != "sql_db_query":
            continue
        sql = str(rec.args.get("query", ""))
        if rec.result_kind == "executed":
            trace.sql_executed.append(sql)
        elif rec.result_kind == "rejected":
            trace.guard_rejections.append({"sql": sql, "message": rec.result_excerpt})
        else:
            trace.sql_errors.append(rec.result_excerpt)

    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            trace.final_output = str(m.content)
            break


def _executed_sql_count(records) -> int:
    return sum(1 for r in records if r.name == "sql_db_query" and r.result_kind == "executed")


def _msg_count(graph, config) -> int:
    """这个 thread 现在有多少条消息。新 thread（或读不到）返回 0。"""
    try:
        return len(graph.get_state(config).values.get("messages") or [])
    except Exception:
        return 0


def run_multi_turn_case(case) -> RunTrace:
    """跑一条多轮用例：**一个用例一份 store + 一份 checkpointer**，轮与轮之间延续。

    与单轮的差别只有隔离粒度，但正是这个差别让下面这些场景第一次可测：

      · `store` 与 `checkpointer` 在轮之间存活。单轮用例每条一份全新的，于是
        "store 里已有该用户偏好"这条**读**方向的分支从来没被走到过（R4 的 D6 就这么漏的）。
      · 新会话**只换 thread_id，不换 store / user_id** —— 同一用户的第二次会话，
        正是跨会话记忆与"读回自己写的数据"的现场。
      · 判定只看**最后一轮**，但**任何一轮崩都算失败**（turn_errors）。
    """
    graph = build_graph(checkpointer=MemorySaver(), store=InMemoryStore())
    context = {"user_id": f"eval-{case.id}"}
    trace = RunTrace(case_id=case.id)
    started = time.time()

    session = 0
    thread_id = ""
    try:
        for i, turn in enumerate(case.turns, 1):
            if i == 1 or turn.new_session:
                session += 1
                thread_id = f"eval-{case.id}-s{session}"
            config = {"configurable": {"thread_id": thread_id}}
            before = _msg_count(graph, config)
            trace.n_turns = i
            try:
                _run_turn(graph, config, context, turn.question, turn, trace)
            except Exception as exc:
                trace.turn_errors.append(f"第 {i} 轮（会话 {session}）: {type(exc).__name__}: {exc}")
                # 崩了就不再往下跑：后面的轮次会踩在一个残缺的状态上，
                # 跑出来的东西既不是"模型答错了"也不是"产品对了"，没有解释价值。
                break

            snapshot = graph.get_state(config)
            messages = list(snapshot.values.get("messages") or [])
            # 只取本轮新增的消息。同会话的后续轮次里，checkpointer 里的 messages
            # 是**累积**的（前面几轮的都在），不切片就会把前面轮次的 SQL 也算进来。
            # 切片为空说明"消息只增不减"这个假设不成立了，退回整份（宁可多算，不可漏判）。
            turn_messages = messages[before:] or messages
            records = _pair_tool_results(turn_messages)
            tin, tout = _collect_usage(turn_messages)
            trace.tokens_in += tin
            trace.tokens_out += tout
            trace.turn_sql_counts.append(_executed_sql_count(records))

            if i == len(case.turns):
                _fill_outcome(trace, turn_messages, records,
                              str(snapshot.values.get("user_intent") or ""))

    except Exception as exc:  # 图之外的问题（构造 config、读 state）也算结果
        trace.ok = False
        trace.error = f"{type(exc).__name__}: {exc}"

    if trace.turn_errors:
        trace.ok = False
        trace.error = trace.turn_errors[0]

    trace.latency_ms = (time.time() - started) * 1000
    return trace

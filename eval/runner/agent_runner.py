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

# 单条用例允许的最大步数 / 最大中断次数——防止图跑飞把整轮回归挂住
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

    payload = {"messages": [HumanMessage(content=case.question)]}
    try:
        while True:
            pending: Command | None = None
            for _chunk in graph.stream(
                payload, config=config, context=context, stream_mode="updates"
            ):
                trace.steps += 1
                if trace.steps > MAX_STEPS:
                    raise RuntimeError(f"超过最大步数 {MAX_STEPS}，疑似图跑飞")
                if isinstance(_chunk, dict) and INTERRUPT_KEY in _chunk:
                    intr = _chunk[INTERRUPT_KEY][0]
                    answer = _answer_for_interrupt(intr.value, case)
                    trace.interrupts.append({"prompt": str(intr.value)[:200], "answer": answer})
                    pending = Command(resume=answer)
            if pending is None:
                break
            if len(trace.interrupts) > MAX_RESUMES:
                raise RuntimeError(f"超过最大中断次数 {MAX_RESUMES}，疑似中断循环")
            payload = pending

        snapshot = graph.get_state(config)
        messages = list(snapshot.values.get("messages") or [])
        trace.user_intent = str(snapshot.values.get("user_intent") or "")

        trace.tool_calls = _pair_tool_results(messages)
        trace.tokens_in, trace.tokens_out = _collect_usage(messages)

        for rec in trace.tool_calls:
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

    except Exception as exc:  # 图跑挂也算一种结果，不能让整轮回归中断
        trace.ok = False
        trace.error = f"{type(exc).__name__}: {exc}"

    trace.latency_ms = (time.time() - started) * 1000
    return trace

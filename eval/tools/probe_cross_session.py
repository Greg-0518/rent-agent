"""跨会话探针 —— 同一个用户、两次会话，跨会话记忆这条链路走得通吗

    python eval/tools/probe_cross_session.py

现有 89 条用例**全是单轮的**：`run_case` 只发一条 HumanMessage，之后的中断-恢复
仍在同一轮里。而每条用例又都拿**全新的 store + 独立的 user_id**（`eval-<case_id>`），
于是"store 里已经有这个用户的偏好"这个状态**在评测里压根构造不出来**——
写方向的代码每轮都跑（`recommend.py` 的 `store.put`），读方向的分支一次都没进过。

这个探针就是为了把那个状态造出来。

═══ 它的历史：先当复现脚本，现在是回归检查 ═══

**修之前**（R4 的 D6，挂了很久没人发现）：

    会话一（store 空） → OK，写入 {'budget_max': 3000.0}
    会话二（同一用户） → KeyError: 'budget_min'

真因是写侧与读侧对「键集合」的假设不一致：

    写  `prefs.model_dump(exclude_none=True)`  ← 只设了 budget_max 时，
                                                落库的 dict 里**没有 budget_min 这个键**
    读  `store_min = prefs["budget_min"]`      ← 直接下标取值，不是 .get()

**为什么不是当场就炸**：那一轮走的是**新用户分支**（`recommend.py` 的
`updated_state["user_preferences"] = prefs.model_dump()`，**不带** `exclude_none`），
写进 state 的键集合是完整的，所以同一轮读不到缺键的版本。

**什么时候炸**：下一次调用。`get_store_info` 是图的**入口**
（`graph.py:34` 的 `add_edge(START, "get_store_info")`），**每一轮都会重跑**，
并且**无条件**用 store 里的值覆盖 state（`main.py:70-72` 没有条件判断）。
于是：

    新会话                    → 入口读 store → 覆盖 → 缺键 → KeyError
    同一会话发第二条消息      → 入口重跑，同样覆盖      → KeyError

两种都会。第二点容易漏（"同一会话内 state 里不是有完整版本吗"——是，
但入口每轮都会把它盖掉）。本脚本把两种都走一遍。所以不是"某条链路答得不好"，
是**整个会话在入口就死**。

**修法（已定，2026-09-13）**：读侧统一走 `src/agent/common/store.py:read_preferences`，
缺键当 None。**没有**改写侧的 `exclude_none=True` —— 读侧正规化能同时容下
**修之前就已落库的旧数据**，写侧改法只管未来写入、历史行照样崩。

**当前结果**（`eval/results/probe_cross_session-fixed.txt`）：

    会话一（store 空） → OK，写入 {'budget_max': 3000.0}
    会话二（同一用户） → OK，读到会话一的 budget_max=3000，store 条数=1

`read_preferences` 的键集合不变式由 `tests/unit_tests/test_store_preferences.py`
零成本守着（不依赖库、不调模型）；这个脚本负责端到端那一半。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langgraph.types import Command  # noqa: E402

from eval.runner.agent_runner import _answer_for_interrupt, build_graph  # noqa: E402

USER_ID = "probe-user"
STORE = InMemoryStore()


def run(question: str, params: dict, thread: str, checkpointer=None) -> str:
    """一个会话 = 一个新 checkpointer + 新 thread_id；store 与 user_id 沿用。

    传 `checkpointer` 就是**同一会话的下一轮**（thread 状态延续），不传则是一次
    全新会话。这两种都要测：`get_store_info` 是图的入口（`graph.py:34` 的
    `add_edge(START, "get_store_info")`），**每轮都会重跑**并用 store 里的值覆盖
    state，所以"新会话"与"同会话第二轮"都会命中缺键分支。
    """
    graph = build_graph(checkpointer=checkpointer or MemorySaver(), store=STORE)
    case = SimpleNamespace(params=params)
    config = {"configurable": {"thread_id": thread}}
    context = {"user_id": USER_ID}
    payload = {"messages": [HumanMessage(content=question)]}
    try:
        while True:
            pending = None
            for chunk in graph.stream(payload, config=config, context=context,
                                      stream_mode="updates"):
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    intr = chunk["__interrupt__"][0]
                    pending = Command(resume=_answer_for_interrupt(intr.value, case))
            if pending is None:
                break
            payload = pending
        snap = graph.get_state(config)
        prefs = snap.values.get("user_preferences")
        n = len(STORE.search((USER_ID, "preferences")))
        return f"OK   user_preferences={prefs}  store 条数={n}"
    except Exception as exc:
        n = len(STORE.search((USER_ID, "preferences")))
        return f"!!   {type(exc).__name__}: {exc}   store 条数={n}"


ck1 = MemorySaver()
print("会话一·第一轮（store 为空）")
print("  ", run("深圳预算3000元以下的房子，帮我推荐10套",
               {"city": "深圳", "budget_max": 3000}, "thread-1", ck1))
print("会话一·第二轮（**同一 thread**：验证同会话下一轮也命中）")
print("  ", run("深圳2500元以下的房子还有哪些",
               {"city": "深圳", "budget_max": 2500}, "thread-1", ck1))
print("会话二（同一用户，**新会话**）")
print("  ", run("深圳预算2000元以下的房子，帮我推荐10套",
               {"city": "深圳", "budget_max": 2000}, "thread-2"))
print("\nstore 里的原始内容：")
for item in STORE.search((USER_ID, "preferences")):
    print("  key=", item.key, " value=", item.value)

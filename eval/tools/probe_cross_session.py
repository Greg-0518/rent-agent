"""跨会话探针 —— 同一个用户、两次会话，跨会话记忆这条链路走得通吗

    python eval/tools/probe_cross_session.py

现有 89 条用例**全是单轮的**：`run_case` 只发一条 HumanMessage，之后的中断-恢复
仍在同一轮里。而每条用例又都拿**全新的 store + 独立的 user_id**（`eval-<case_id>`），
于是"store 里已经有这个用户的偏好"这个状态**在评测里压根构造不出来**——
写方向的代码每轮都跑（`recommend.py:186` 的 `store.put`），读方向的分支一次都没进过。

这个探针就是为了把那个状态造出来。当前结果：**第二个会话直接 KeyError 崩掉**。

    会话一（store 空） → OK，写入 {'budget_max': 3000.0}
    会话二（同一用户） → KeyError: 'budget_min'

真因在 `src/agent/node/recommend.py`：

    189 行   prefs.model_dump(exclude_none=True)   ← 只设了 budget_max 时，
                                                     落库的 dict 里**没有 budget_min 这个键**
    196 行   store_min = prefs["budget_min"]       ← 直接下标取值，不是 .get()

所以只要用户第一次只说了上限、第二次再来，"有持久化信息"这条分支一进去就炸。
这两行都在 HEAD 里（不是工作区未提交的改动），是**既有缺陷**。

修法（任选，改前先想清楚语义）：
  - 读侧用 `prefs.get("budget_min")`，缺键当 None —— 最小改动，与 `UserPreferences`
    两个字段都是 `Optional` 的定义一致
  - 或写侧去掉 `exclude_none=True`，让 None 也落库
    （代价：store 里开始存 None，"没设过"和"设过然后是 None"就分不开了）

fix 之后这个脚本就是它的回归检查：两个会话都该 OK，且会话二能读到会话一写的偏好。
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


def run(question: str, params: dict, thread: str) -> str:
    """一个会话 = 一个新 thread_id，但 store 与 user_id 沿用。"""
    graph = build_graph(checkpointer=MemorySaver(), store=STORE)
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


print("会话一（首轮，store 为空）")
print("  ", run("深圳预算3000元以下的房子，帮我推荐10套",
               {"city": "深圳", "budget_max": 3000}, "thread-1"))
print("会话二（同一用户，新会话）")
print("  ", run("深圳预算2000元以下的房子，帮我推荐10套",
               {"city": "深圳", "budget_max": 2000}, "thread-2"))
print("\nstore 里的原始内容：")
for item in STORE.search((USER_ID, "preferences")):
    print("  key=", item.key, " value=", item.value)

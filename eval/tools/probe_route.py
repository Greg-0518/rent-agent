"""路由探针 —— 把一条用例**强制**塞进指定的子图，看它到底答不答得出来。

    python eval/tools/probe_route.py                    # 默认：那 12 条被判"不可达"的
    python eval/tools/probe_route.py L2-009 L4-010      # 指定用例
    python eval/tools/probe_route.py --intent get_info L4-002

**为什么需要它**：一条用例判 FAIL 或 UNREACHABLE 时，"是模型不行"还是
"它压根没走到被测链路"是两件事，而且**修法完全相反**——前者改 Prompt/schema，
后者改路由或补产品链路。光看基线结果分不出来，因为两种情况下模型都"答错了"。

做法：照抄生产图 src/agent/graph.py 的节点与连边，只把 identify_question
换成写死 user_intent 的节点。其余（Prompt、子图、工具）全是生产原件，
所以测出来的差异**只来自"路由"这一个变量**。

实测（2026-09）：把基线里 12 条判 UNREACHABLE 的统计型问句强制路由进
`recommend_house`，12/12 全部通过——SQL 写对了、答也答对了。
结论：那 12 条是**纯路由问题，不是能力缺口**，改意图 Prompt 即可，不必动代码。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from eval.asserters.rule import assert_case  # noqa: E402
from eval.runner.agent_runner import run_case  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402
from eval.tools.rescore import load_golden  # noqa: E402

# 生产图 router_message 认识的意图 → 它落到的节点。要强制进哪个分支，就换哪个名字。
INTENT_NODES = {
    "recommend_house": ("recommended_graph", "need_reserve"),
    "get_info": ("get_user_preferences",),
    "reserve_house": ("reserve_graph",),
    "rent_calc": ("finance_graph",),
    "image_analysis": ("image_analysis",),
    "contract_audit": ("set_contract_text", "contract_graph"),
    "others": ("extend_graph",),
}

# 基线里被判「不可达」（意图分到 get_info）的那 12 条
IDS = ["L2-001", "L2-003", "L2-004", "L2-005", "L2-006", "L2-007",
       "L2-008", "L2-009", "L2-011", "L2-012", "L3-015", "L4-010"]


def build_forced_graph(intent: str):
    """生产图的节点与连边，只把 identify_question 换成写死 user_intent 的节点。

    进入目标节点之后的下游仍按生产连线走（如 recommended_graph → need_reserve），
    否则中断-恢复那一环会被跳掉，trace 就和真实跑法不是一回事了。
    """
    from src.agent.common.content import ContextSchema
    from src.agent.graph import builder as prod_builder
    from src.agent.node.main import get_store_info
    from src.agent.state.main import State

    if intent not in INTENT_NODES:
        raise SystemExit(f"不认识的意图 {intent!r}；可选：{'、'.join(INTENT_NODES)}")
    chain = INTENT_NODES[intent]

    b = StateGraph(State, context_schema=ContextSchema)
    b.add_node(get_store_info)
    b.add_node("force", lambda _s: {"user_intent": intent})
    # 直接复用生产 builder 里已注册的节点/子图，不重抄一遍连线。
    # need_reserve 单独加：它不在 intent→落点 的表里，但 recommended_graph 的下游需要它。
    for name in {n for nodes in INTENT_NODES.values() for n in nodes}:
        if name != "need_reserve":
            b.add_node(name, prod_builder.nodes[name].runnable)
    b.add_node("need_reserve", prod_builder.nodes["need_reserve"].runnable)
    b.add_edge("__start__", "get_store_info")
    b.add_edge("get_store_info", "force")
    b.add_edge("force", chain[0])
    for src, dst in zip(chain, chain[1:]):
        b.add_edge(src, dst)
    b.add_edge(chain[-1], END)
    return b.compile(checkpointer=MemorySaver(), store=InMemoryStore())


def main() -> int:
    argv = sys.argv[1:]
    intent = "recommend_house"
    if "--intent" in argv:
        i = argv.index("--intent")
        intent = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ids = argv or IDS

    cases = {c.id: c for c in load_cases("text2sql", include_holdout=True)}
    golden = load_golden()
    g = build_forced_graph(intent)

    print(f"强制 user_intent={intent!r}（落点 {INTENT_NODES[intent][0]}）\n")
    print(f"{'用例':<9}{'强制路由后':<12}{'SQL数':>6}  说明")
    for cid in ids:
        case = cases[cid]
        tr = run_case(case, graph=g)
        r = assert_case(case, tr, golden[cid])
        note = (r.reason or tr.error or "")[:56]
        print(f"{cid:<9}{r.kind:<12}{len(tr.sql_executed):>6}  {note}")
        if tr.sql_executed:
            print(f"          SQL: {tr.sql_executed[-1].strip()[:200]}")
        print(f"          答: {(tr.final_output or '')[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

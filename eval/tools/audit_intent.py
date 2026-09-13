"""用例可达性体检 —— 这条用例会不会被路由到 text2sql？

    python eval/tools/audit_intent.py              # 全量 79 条
    python eval/tools/audit_intent.py L2-001 L5-010
    python eval/tools/audit_intent.py --tier L2

**为什么需要它**：text2sql 不是图的入口，它挂在 `recommended_graph` 下面，
而进不进 `recommended_graph` 由 `identify_question`（`node/main.py:23`）的
分类结果决定。意图枚举里没有"事实查询"这一类，所以：

    统计型问句（"哪个区最贵""一共有多少套"） → get_info → get_user_preferences
                                              ↑ 这个节点拿**用户偏好数据**回答，不碰数据库
    写操作请求（"把价格都改成1000"）         → others   → extend_graph（纯 LLM）

后果是这些用例**永远不可能通过**，而且成绩与模型的 text2sql 能力无关。
首轮 smoke 基线就是这么踩到的：L2 整档 12 条全挂、L5 整档 10 条"空过"，
分母被污染了 22 条（79 条里的 28%）。

**所以写新用例之前先跑这个**：一条用例若不可达，要么改写问句让它落到
`recommend_house`，要么把它记为产品缺口而不是模型能力分。别让它悄悄待在分母里。
"""

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from src.agent.common.llm import model  # noqa: E402
from src.agent.node.main import UserMessage  # noqa: E402

# 与 node/main.py 里 identify_question 逐字一致。这里**故意抄一份而不是 import**：
# 这段 prompt 是分类行为的一部分，改了它分类结果就会变；抄一份能让这份体检报告
# 清楚显示"我当时是按哪版 prompt 判的"。漂移了就在这里改，别偷偷引用。
#
# 2026-09-13（R5 §6）：给七个标签补了定义，核心是把"要查房源库才能回答的问题"
# （含统计/聚合/最值）明确划给 recommend_house，get_info 收窄成"我的"。
# 注意 UserMessage 本身是从 node/main.py import 的，所以**字段描述会自动跟上**；
# 这里只有 system prompt 需要手抄，改 main.py 时别忘了同步这一段。
_SYSTEM = """你是⼀个根据描述提取信息提取专家。请从⽤⼾的描述中提取⽤⼾想要咨询的相关信息。
                    严谨根据语义推断信息，但不能猜测或编造信息。
                    判断问题类型时严格按各类型的定义与边界，特别注意：
                    要从房源库里查数据才能回答的问题（含套数、均价、最值等统计类），
                    属于 recommend_house，不属于 get_info。"""

# 唯一能走到 text2sql 的意图
REACHABLE_INTENT = "recommend_house"

TIER_ORDER = ["L1", "L2", "L3", "L4", "L5", "Edge"]


def load_cases() -> list[dict]:
    cases = []
    for f in sorted((ROOT / "eval" / "cases" / "text2sql").glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        cases.extend(doc["cases"] if isinstance(doc, dict) else doc)
    return cases


def classify(question: str) -> str:
    got = model.with_structured_output(schema=UserMessage).invoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=question)]
    )
    return got.type


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tier_filter = None
    if "--tier" in sys.argv:
        tier_filter = sys.argv[sys.argv.index("--tier") + 1]
        args = [a for a in args if a != tier_filter]

    cases = load_cases()
    if args:
        want = set(args)
        cases = [c for c in cases if c["id"] in want]
    if tier_filter:
        cases = [c for c in cases if c.get("tier") == tier_filter]
    if not cases:
        print("没选中任何用例")
        return 2

    print("=" * 78)
    print(f"可达性体检：{len(cases)} 条（判据：意图 == {REACHABLE_INTENT} 才走得到 text2sql）")
    print("=" * 78)

    results = []
    for c in cases:
        intent = classify(c["question"])
        results.append((c, intent))

    unreachable = [(c, i) for c, i in results if i != REACHABLE_INTENT]
    by_tier = defaultdict(lambda: [0, 0])
    for c, intent in results:
        t = by_tier[c.get("tier") or "?"]
        t[1] += 1
        if intent == REACHABLE_INTENT:
            t[0] += 1

    print("\n【分档可达率】")
    for tier in sorted(by_tier, key=lambda x: (TIER_ORDER.index(x) if x in TIER_ORDER else 99, x)):
        ok, tot = by_tier[tier]
        flag = "   ← 整档不可达" if ok == 0 else ""
        print(f"    {tier:<6} {ok:>2}/{tot:<2} {ok / tot:>6.0%}{flag}")

    print(f"\n【不可达用例】{len(unreachable)}/{len(results)} 条"
          f"（{len(unreachable) / len(results):.0%} —— 这些用例的成绩与模型 text2sql 能力无关）")
    for c, intent in unreachable:
        print(f"    {c['id']:<9} [{c.get('tier')}] → {intent:<16} 「{c['question']}」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""成绩单 —— 把一次 run 的 jsonl 汇成三维指标 + 归因分布 + 过程信号

    python eval/report/gen_report.py                    # 最近一次 run
    python eval/report/gen_report.py --run-id smoke-baseline
    python eval/report/gen_report.py --list             # 列出所有 run
    python eval/report/gen_report.py --md out.md        # 另存 Markdown

**为什么要有这个脚本**：jsonl 里躺着几千条记录，人眼看不出一轮改了 Prompt
之后是变好还是变坏。飞轮文的那句"评测可信度 > 系统复杂度"落到工程上就是这个：
先把**可比**的数字固定下来（同一个 run_id 口径、同一套分档），再谈优化。

三维指标（设计文档 §5）：
  1. **完成率**——分档看，不能只看总分（L1 全过、L5 全挂，总分也是"85%"）
  2. **时延**——P50 / P95
  3. **token 成本**——输入/输出分别统计（输入涨往往是 schema 反复查）

外加三项**比完成率更值钱**的：
  4. **不可达清单**——被意图路由分走的用例（首轮基线实测 24/79）。它们不是失败，
     但如果混在完成率里，24 条产品缺口的账会算到模型头上。
  5. **失败归因分布**——七分类，前五类属模型能力、后两类不是。看"错在哪一类"
     才知道该改 Prompt、改路由还是改安全层。单看完成率只能告诉你"变差了"。
  6. **过程信号**——是否先查 schema 再写 SQL、SQL 报错重试次数、守卫拦截次数、
     中断次数。这些在 trace 里是白拿的，而且**先于完成率退化**（模型开始瞎猜 schema 时，
     完成率还没掉，`schema_queried_first` 已经掉了）。

**三个完成率，别混用**（`UNVERIFIABLE` 绝不能算通过，否则等于给假通过开后门）：

    完成率       = PASS / 全部                  ← 最保守，含"判不了"和"没测到"
    模型能力口径 = PASS / (全部 − 不可达)        ← **跨轮比较用这个**
    可判完成率   = PASS / (PASS + FAIL)          ← 再剔除判不了的

口径的分母为什么这么切：
  - **不可达的摘掉**：路由分走的用例与模型能力无关，留着会持续拉低模型分数，
    看起来像模型退化，实际是产品范围问题。
  - **存疑的留下**：它可能真是模型错，只是断言器判不了。把"判不了"都摘掉
    等于给自己发奖金——这个数字会越修越漂亮。

后两个口径都打印，是因为单独引用任何一个都有粉饰空间。存疑占比超过 20% 直接标红
（同一条线在 `eval/conftest.py` 里也是让整个 session 判失败的门槛）。
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "eval" / "results"

# 归因分类，顺序即报告里的展示顺序：
# 前六类是**模型能力**失败（该改 Prompt / 该改 schema 说明），
# 后三类**不属于模型能力问题**（该改安全层 / 该改路由 / 该改代码），必须分开看。
ATTRIBUTIONS = [
    ("SQL_SYNTAX_ERROR", "SQL 语法/执行错误"),
    ("SCHEMA_MISUNDERSTOOD", "理解错表结构"),
    ("CONDITION_MAPPED_WRONG", "条件映射错"),
    ("BOUNDARY_OFF_BY_ONE", "边界开闭取反（>= 写成 > 之类）"),
    ("RESULT_TRANSFORM_WRONG", "结果转换错"),
    ("NO_QUERY_ATTEMPT", "该查库却没查"),
    ("ROUTED_AWAY", "被路由分走（非模型能力问题）"),
    ("GUARD_BYPASS", "安全层被绕过（非模型能力问题）"),
    ("GRAPH_CRASH", "图执行异常（非模型能力问题，是要修的代码 bug）"),
]

MODEL_ATTRIBUTIONS = {code for code, _ in ATTRIBUTIONS[:6]}

# 与 conftest.MAX_UNVERIFIABLE_RATIO 同一个数：那边是让回归失败的门槛，
# 这边是标红提示的门槛。两处必须一致，否则会出现"报告说没事、CI 却红了"。
MAX_UNVERIFIABLE_RATIO = 0.20

# 单价（元 / 百万 token）。**留空不填就是不算金额**——报一个凭记忆写的价格
# 比不报更糟。要算就照官方定价页填，两个数都要填。
PRICE_PER_MTOK: dict[str, float] | None = None
# 例：PRICE_PER_MTOK = {"in": 2.0, "out": 8.0}


def load_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.jsonl"))


def read_run(path: Path) -> tuple[dict, list[dict]]:
    meta, cases = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "run_meta":
            meta = rec
        elif rec.get("kind") == "case":
            cases.append(rec)
    return meta, cases


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数。样本少时它没有统计意义，调用方负责标注。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# 展示顺序：按档位难度递增、断言模式按"从具体到抽象"排。
# 不按字母序——`Edge` 排到 `L1` 前面会让人以为 Edge 是最简单的一档，
# 而它其实是另一个维度（边界情形，不参与难度比较）。
TIER_ORDER = ["L1", "L2", "L3", "L4", "L5", "Edge"]
MODE_ORDER = ["id_set", "scalar", "count", "refusal"]


def _sort_key(order: list[str]):
    def key(name: str) -> tuple[int, str]:
        return (order.index(name) if name in order else len(order), name)
    return key


def _rate(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "—"


def _bar(ratio: float, width: int = 20) -> str:
    filled = int(round(ratio * width))
    return "█" * filled + "·" * (width - filled)


def build(meta: dict, cases: list[dict]) -> str:
    out: list[str] = []
    w = out.append

    total = len(cases)
    by_kind = defaultdict(int)
    for c in cases:
        by_kind[c["result"]["kind"]] += 1
    n_pass = by_kind.get("pass", 0)
    n_fail = by_kind.get("fail", 0)
    n_unver = by_kind.get("unverifiable", 0)
    n_unreach = by_kind.get("unreachable", 0)
    # 模型能力口径的分母 = 走到了被测链路的那些。**存疑的留在分母里**：
    # 它可能真是模型错，只是断言器判不了；把判不了的都摘掉，等于给自己发奖金。
    # 不可达的摘掉：路由分走的用例与模型能力无关，留着会持续拉低模型分数。
    scorable = total - n_unreach
    decidable = n_pass + n_fail

    model = meta.get("model") or {}
    w("=" * 78)
    w(f"成绩单  run_id={meta.get('run_id', '?')}    {meta.get('started_at', '?')}")
    w(f"        selection={meta.get('selection', '?')}   "
      f"model={model.get('model', '?')}(profile={model.get('profile') or '默认'})   "
      f"只读账号={'是' if meta.get('db_readonly') else '否'}")
    w("=" * 78)

    # ---- 1. 完成率 ----
    w("\n【1】完成率")
    w(f"    完成率        {n_pass}/{total} = {_rate(n_pass, total):>6}   "
      f"{_bar(n_pass / total if total else 0)}   ← 全量口径")
    w(f"    模型能力口径  {n_pass}/{scorable} = {_rate(n_pass, scorable):>6}   "
      f"← 剔除「没走到被测链路」的，**跨轮比较用这个**")
    w(f"    可判完成率    {n_pass}/{decidable} = {_rate(n_pass, decidable):>6}   "
      "← 再剔除断言器判不了的（别单独引用，有粉饰空间）")
    w(f"    分布：pass={n_pass}  fail={n_fail}  存疑={n_unver}  不可达={n_unreach}")
    ratio_unver = n_unver / total if total else 0.0
    if ratio_unver > MAX_UNVERIFIABLE_RATIO:
        w(f"    [!] 存疑占比 {ratio_unver:.0%} 超过上限 {MAX_UNVERIFIABLE_RATIO:.0%}"
          " —— 这份成绩单可信度不足，别用来下结论")

    # ---- 2. 分档 / 分断言模式 ----
    for title, key, order in (
        ("【2】分档完成率（看清是哪一档在拖后腿）", "tier", TIER_ORDER),
        ("【3】分断言模式（聚合/拒绝那些路径的问题只看这里）", "mode", MODE_ORDER),
    ):
        w(f"\n{title}")
        groups = defaultdict(lambda: [0, 0, 0, 0])   # pass, fail, unver, unreach
        for c in cases:
            g = groups[c.get(key) or "?"]
            g[["pass", "fail", "unverifiable", "unreachable"]
              .index(c["result"]["kind"])] += 1
        for name in sorted(groups, key=_sort_key(order)):
            p, f, u, x = groups[name]
            t = p + f + u + x
            scored = t - x          # 该档的模型能力分母
            notes = []
            if u:
                notes.append(f"存疑 {u}")
            if x:
                notes.append(f"不可达 {x}")
            w(f"    {name:<8} {p:>2}/{scored:<2} {_rate(p, scored):>6}  "
              f"{_bar(p / scored if scored else 0, 16)}"
              f"{'   ' + '  '.join(notes) if notes else ''}")

    # ---- 4. 诊断：不可达 ----
    # 这一节是首轮基线之后加的。它必须**独立于失败清单**存在：
    # 不可达的用例不是失败（判失败会让回归常红），但它是"这次没测到"的硬信息，
    # 混在完成率里会让 24 条产品缺口的账算到模型头上。
    unreach = [c for c in cases if c["result"]["kind"] == "unreachable"]
    w("\n【4】不可达用例（没走到被测链路，与模型 text2sql 能力无关）")
    if not unreach:
        w("    无")
    else:
        w(f"    共 {n_unreach}/{total} = {n_unreach / total:.0%}"
          "  —— 这些用例的成绩不计入模型能力分")
        by_tier = defaultdict(list)
        for c in unreach:
            by_tier[c.get("tier") or "?"].append(c["case_id"])
        for tier in sorted(by_tier, key=_sort_key(TIER_ORDER)):
            ids = by_tier[tier]
            w(f"    {tier:<6} {len(ids):>2} 条  {' '.join(sorted(ids))}")
        w("    真因走意图路由（`identify_question`）：text2sql 只挂在 recommended_graph 下，")
        w("    统计型问句被判成 get_info（拿用户偏好作答，不查库）、写操作请求被判成 others。")
        w("    用 `python eval/tools/audit_intent.py` 复查整档可达性。")

    # ---- 5. 失败归因分布 ----
    w("\n【5】失败归因分布（决定下一步改什么）")
    attr = defaultdict(list)
    for c in cases:
        if c["result"]["kind"] == "fail":
            attr[c["result"].get("attribution") or "(未分类)"].append(c["case_id"])
    if not attr:
        w("    无失败用例")
    else:
        n_model_attr = sum(len(ids) for code, ids in attr.items()
                           if code in MODEL_ATTRIBUTIONS)
        if n_model_attr:
            w(f"    其中属于模型能力的 {n_model_attr} 条 —— 这才是该改 Prompt / schema 说明的：")
        else:
            w("    没有一条属于模型能力——先别改 Prompt，看上面标了「非模型问题」的那些：")
        for code, label in ATTRIBUTIONS:
            ids = attr.pop(code, [])
            if ids:
                tag = "" if code in MODEL_ATTRIBUTIONS else "  ← 非模型问题"
                w(f"      {label:<26} {len(ids):>2} 条  {' '.join(sorted(ids))}{tag}")
        for code, ids in sorted(attr.items()):      # 兜底：断言器新增了分类别漏掉
            w(f"      {code:<26} {len(ids):>2} 条  {' '.join(sorted(ids))}")

    # ---- 5. 时延 ----
    lat = [c.get("latency_ms") or 0.0 for c in cases]
    w("\n【6】时延")
    if lat:
        p50, p95 = percentile(lat, 0.50) / 1000, percentile(lat, 0.95) / 1000
        w(f"    P50 {p50:.1f}s   P95 {p95:.1f}s   max {max(lat) / 1000:.1f}s   "
          f"min {min(lat) / 1000:.1f}s")
        if len(lat) < 20:
            w(f"    [!] 只有 {len(lat)} 个样本，P95 实际接近最大值，不具有统计意义")
        slow = sorted(cases, key=lambda c: -(c.get("latency_ms") or 0))[:3]
        w("    最慢 3 条：" + "  ".join(
            f"{c['case_id']}({c.get('latency_ms', 0) / 1000:.0f}s)" for c in slow))

    # ---- 6. token 成本 ----
    tin = sum(c.get("tokens_in") or 0 for c in cases)
    tout = sum(c.get("tokens_out") or 0 for c in cases)
    w("\n【7】token 成本")
    if cases:
        w(f"    输入 {tin:,}  输出 {tout:,}  合计 {tin + tout:,}   "
          f"单条均值 输入 {tin // len(cases):,} / 输出 {tout // len(cases):,}")
    if PRICE_PER_MTOK:
        cost = tin / 1e6 * PRICE_PER_MTOK["in"] + tout / 1e6 * PRICE_PER_MTOK["out"]
        w(f"    折算金额 ≈ {cost:.3f} 元/轮（单价 {PRICE_PER_MTOK['in']}/"
          f"{PRICE_PER_MTOK['out']} 元每百万 token）")
    else:
        w("    （未配置单价，不显示金额——在 gen_report.py 的 PRICE_PER_MTOK 填入）")

    # ---- 7. 过程信号 ----
    w("\n【8】过程信号（比完成率更早退化：模型开始瞎猜表结构时，先掉的是这个）")
    traces = [c.get("trace") or {} for c in cases]
    if traces:
        first = sum(1 for t in traces if t.get("schema_queried_first"))
        w(f"    先查 schema 再写 SQL   {first}/{len(traces)} = {_rate(first, len(traces))}")
        retries = [t.get("retry_count") or 0 for t in traces]
        w(f"    SQL 报错重试          合计 {sum(retries)} 次，"
          f"发生过的用例 {sum(1 for r in retries if r)}/{len(retries)}，"
          f"单条最多 {max(retries)} 次")
        n_rej = sum(len(t.get("guard_rejections") or []) for t in traces)
        n_intr = sum(len(t.get("interrupts") or []) for t in traces)
        n_sql = sum(len(t.get("sql_executed") or []) for t in traces)
        steps = [t.get("steps") or 0 for t in traces]
        w(f"    执行的 SQL            合计 {n_sql} 条，单条均值 {n_sql / len(traces):.1f}")
        w(f"    守卫拦截 {n_rej} 次   中断-恢复 {n_intr} 次   "
          f"图步数 中位 {statistics.median(steps):.0f} 最大 {max(steps)}")

        # 零工具调用是"没走到被测链路"的**原始信号**；判 unreachable 是断言器
        # 拿图里的意图真值做的**定性**。正常情况下两者应当重合（零调用 ⇒ 不可达）。
        # 所以这里不再重复报一遍，而是做**交叉校验**：列出"零工具调用、却没被判
        # 不可达"的用例——那才是真异常（要么路由对了而模型不查库，要么意图没写进
        # state）。这样这条线既能发现路由问题，也能发现断言器的判定漏了。
        dead = [c for c, t in zip(cases, traces) if not (t.get("tool_calls") or [])]
        # 拒绝档要排除掉：那里的**正确答案就是"一次工具都别调"**——
        # 模型自己用自然语言拒绝，正是安全的表现。不排除的话
        # L5 整档会被误报成"零工具调用却没判不可达"（首轮就报了两条假警报）。
        dead_check = [c for c in dead if c.get("mode") != "refusal"]
        mismatch = [c for c in dead_check if c["result"]["kind"] != "unreachable"]
        w(f"    零工具调用        {len(dead)}/{len(cases)} 条"
          f"（其中 {len(dead_check) - len(mismatch)} 条已判不可达）")
        if mismatch:
            w(f"    ⚠ 零工具调用但未判不可达 {len(mismatch)} 条 —— 这不符合预期：")
            for c in mismatch:
                w(f"        {c['case_id']:<9} 判为 {c['result']['kind']}"
                  f"（意图={((c.get('trace') or {}).get('user_intent') or '未记录')}）"
                  f"  {c['result'].get('reason', '')[:46]}")
            w("        两种可能：意图已路由到 text2sql 而模型就是不查库（模型失败，正常），"
              "或意图没写进 state（harness 漏记，要修）")
        n_refusal = len(dead) - len(dead_check)
        if n_refusal:
            w(f"    （另有 {n_refusal} 条拒绝档零工具调用，属预期，不计入上面这条校验）")

    # ---- 8. 需要人看的清单 ----
    fails = [c for c in cases if c["result"]["kind"] == "fail"]
    unvers = [c for c in cases if c["result"]["kind"] == "unverifiable"]
    w("\n【9】需要人看的清单")
    if not fails and not unvers:
        w("    无")
    for c in fails:
        w(f"    FAIL  {c['case_id']:<9} [{c.get('tier')}/{c.get('mode')}] "
          f"{c['result'].get('attribution') or '?'}  {c['result'].get('reason', '')[:60]}")
    for c in unvers:
        w(f"    存疑  {c['case_id']:<9} [{c.get('tier')}/{c.get('mode')}] "
          f"{c['result'].get('reason', '')[:60]}")
    if fails or unvers:
        w(f"\n    完整轨迹（含每条用例的 stdout）见 "
          f"eval/results/{meta.get('run_id', '?')}.jsonl")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="把一次 run 的 jsonl 汇成成绩单")
    ap.add_argument("--run-id", help="run 标识（文件名去掉 .jsonl）；缺省用最近一次")
    ap.add_argument("--list", action="store_true", help="列出所有 run 后退出")
    ap.add_argument("--md", help="另存一份 Markdown 到指定路径")
    args = ap.parse_args()

    runs = load_runs()
    if args.list:
        if not runs:
            print(f"{RESULTS_DIR} 下还没有 run。先跑 pytest eval -m smoke")
            return 0
        for p in runs:
            meta, cases = read_run(p)
            n_pass = sum(1 for c in cases if c["result"]["kind"] == "pass")
            print(f"  {p.stem:<28} {meta.get('started_at', '?'):<20} "
                  f"selection={meta.get('selection', '?'):<8} {n_pass}/{len(cases)}")
        return 0

    if args.run_id:
        path = RESULTS_DIR / f"{args.run_id}.jsonl"
        if not path.exists():
            print(f"找不到 {path}。用 --list 看有哪些 run。", file=sys.stderr)
            return 2
    else:
        if not runs:
            print(f"{RESULTS_DIR} 下还没有 run。先跑 pytest eval -m smoke", file=sys.stderr)
            return 2
        path = runs[-1]

    meta, cases = read_run(path)
    if not cases:
        print(f"{path.name} 里没有用例记录（只有 run_meta）——"
              "多半是那次 run 被 collect-only / 环境门禁提前结束了。", file=sys.stderr)
        return 2

    report = build(meta, cases)
    print(report)

    if args.md:
        dest = Path(args.md)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"```\n{report}\n```\n", encoding="utf-8")
        print(f"\n[写出 Markdown] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""边界体检 —— 哪些用例在考"边界语义"，而不是"会不会写 SQL"

    python eval/tools/audit_boundary.py

问句里的"30楼以上""3000元以下"是**歧义措辞**：中文口语里
"以上"按规范含本数、"以下"按规范含本数，但日常使用中大量人不这么理解。
一条用例如果在边界值上恰好有一行数据，那么：

    黄金按"含30楼"取到 4 行，模型按"高于30楼"取到 3 行 → 判模型错

可模型没错——它只是猜边界猜反了。**这条用例量的是猜测，不是能力**，
它的分差是噪声，会跟着模型换一版就随机抖动。

体检方式：把黄金 SQL 里的比较符逐个取反（`>` ↔ `>=`、`<` ↔ `<=`）
重跑，若结果集**变了**，说明这个边界上有真实数据，用例对边界敏感。

敏感 ≠ 一定要改。分两种情况处置：

    问句用「以上 / 以下 / 以内」这类歧义措辞  → 改问句（要么写明"及以上"，
                                                  要么改用"高于/超过"）
    问句用「高于 / 超过 / 不低于」等明确措辞  → 用例没问题，是模型真错

所以工具同时读问句措辞，把两者并排打出来——只看 SQL 是判断不了的。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner.db import run_sql  # noqa: E402
from eval.runner.loader import load_cases  # noqa: E402
# 比较符的定位与取反在 sqlops 里，与断言器的 BOUNDARY_OFF_BY_ONE 判定**共用**。
# 各自数一遍索引迟早会错位，而且错得很安静（翻转出来的 SQL 跑得通、只是答的不是同一句话）。
from eval.runner.sqlops import (  # noqa: E402
    AMBIGUOUS, FLIPS, boundary_ops as _boundary_ops, boundary_verdict as _verdict,
    flip_nth as _flip_nth,
)

GOLDEN_PATH = ROOT / "eval" / "golden" / "text2sql.json"


def _fingerprint(sql: str):
    try:
        cols, rows = run_sql(sql)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"
    cols = sorted(str(c).lower() for c in cols)
    return (tuple(cols), frozenset(map(tuple, ([str(v) for v in r] for r in rows)))), ""


def main() -> int:
    import json

    if not GOLDEN_PATH.exists():
        print(f"缺 {GOLDEN_PATH.relative_to(ROOT)}，先跑 python eval/tools/make_golden.py")
        return 3

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
    # 问句在用例 yaml 里，不在 golden 里 —— 合并进来，否则没法判断
    # 那条敏感用例到底该改问句还是该留（纯看 SQL 判断不了歧义）。
    questions = {c.id: c.question for c in load_cases("text2sql", include_holdout=True)}
    sensitive, checked, skipped = [], 0, []

    for cid, entry in sorted(golden.items()):
        sql = entry.get("gold_sql")
        if not sql:
            skipped.append((cid, "无黄金 SQL（拒绝档）"))
            continue
        base_fp, err = _fingerprint(sql)
        if err:
            skipped.append((cid, f"黄金 SQL 跑不通：{err}"))
            continue
        checked += 1

        for nth, (_s, _e, original) in enumerate(_boundary_ops(sql)):
            for f_src, f_dst in FLIPS:
                if f_src != original:
                    continue
                flip = _flip_nth(sql, nth, f_dst)
                if not flip:
                    continue
                flip_fp, err = _fingerprint(flip)
                if err or flip_fp == base_fp:
                    continue  # 取反后结果没变 → 这个边界上没有数据，无害
                q = questions.get(cid, "")
                n_base, n_flip = len(base_fp[1]), len(flip_fp[1])
                sensitive.append({
                    "case_id": cid,
                    "question": q,
                    "flip": f"{original} → {f_dst}",
                    "rows": f"{n_base} → {n_flip}",
                    # 行数没变、内容变了：**典型边界特征**（边界上那行被换成另一行）。
                    # 行数变了反而可能只是过滤范围变了，不一定真是边界问题。
                    "row_count_same": n_base == n_flip,
                    "verdict": _verdict(q),
                })
                break  # 一条用例只报一次，免得同一个条件报两遍

    print(f"检查 {checked} 条（{len(skipped)} 条跳过）")
    if skipped:
        print("  跳过：" + "；".join(f"{c}({r})" for c, r in skipped[:6])
              + ("…" if len(skipped) > 6 else ""))

    if not sensitive:
        print("\n没有对边界敏感的用例 —— 所有比较条件在边界上都没有数据，"
              "问句怎么写都不影响答案。")
        return 0

    print(f"\n边界敏感 {len(sensitive)} 条（flip 左边是黄金实际用的比较符）：\n")
    for s in sensitive:
        mark = "!" if s["verdict"] == "歧义" else "·"
        # 行数是否变化本身不重要（取反就是把边界上那行去掉/换回来，变才是正常的），
        # 但它顺手说明了一件事：边界行有没有被 LIMIT 截掉。
        note = "" if not s["row_count_same"] else "  （边界行被 LIMIT 挡在窗口外）"
        print(f"  {mark} {s['case_id']:<9} {s['flip']:<10} 行数 {s['rows']:<10} "
              f"[{s['verdict']}]{note}")
        print(f"      {s['question']}")

    # ---- 最强的一类缺陷：同一个词在不同用例里被映射成了不同口径 ----
    # 这不是"某条用例问得含糊"，而是**用例集自相矛盾**：
    # 模型无论采用哪种读法，都必然踩中其中一条。这种分差 100% 是噪声。
    by_phrase: dict[str, set[str]] = {}
    for s in sensitive:
        phrase = next((w for w in AMBIGUOUS if w in s["question"]), None)
        if not phrase:
            continue
        gold_op = s["flip"].split("→")[0].strip()
        # 规范中文里「以上/以下」含本数 → >= / <=；用严格符就是非规范口径
        standard = (phrase in ("以上", "以内") and gold_op == ">=") or \
                   (phrase == "以下" and gold_op == "<=")
        by_phrase.setdefault(phrase, set()).add("含本数" if standard else "不含本数")

    conflict = {p: c for p, c in by_phrase.items() if len(c) > 1}
    if conflict:
        print("\n[!] 用例集自相矛盾 —— 同一个词被映射成了两种口径：")
        for phrase, conventions in conflict.items():
            users = [f"{s['case_id']}({s['flip'].split('→')[0].strip()})"
                     for s in sensitive if phrase in s["question"]]
            print(f"    「{phrase}」同时存在 {' 和 '.join(sorted(conventions))}：{'、'.join(users)}")
        print("    模型采用任何一种读法都必然答错其中一条。这不是模型的问题，")
        print("    也不是改一条问句能解决的 —— 得统一口径（见下方建议）。")

    amb = [s for s in sensitive if s["verdict"] == "歧义"]
    clear = [s for s in sensitive if s["verdict"] == "明确"]
    other = [s for s in sensitive if s["verdict"] == "无边界词"]
    if other:
        print(f"\n[!] {len(other)} 条问句里根本没有边界词，黄金却对比较符敏感："
              f"{'、'.join(s['case_id'] for s in other)}")
        print("    问句省略了边界语义，而黄金按某一种口径取数 —— 模型无从判断。")

    if amb and not conflict:
        # 敏感但不矛盾 = 好用例。这段必须写清楚，否则很容易被误读成
        # 「敏感就是有问题，把问句改到不敏感」—— 那恰好是把真实信号改没了。
        print(f"\n以上 {len(amb)} 条**没有自相矛盾**，黄金一律用规范中文口径"
              "（「以上/以内」含本数）。这是**可学的信号，不是噪声**：")
        print("    模型若把「30楼以上」读成「高于30楼」，会在这些用例上稳定丢分——")
        print("    这正是要测的东西，不要为了让模型好看而改问句。")
        print("    只有同一个词在不同用例里口径不一致（见上）时，才是用例集的错。")

    if clear:
        print(f"\n{len(clear)} 条措辞本来明确（{'、'.join(s['case_id'] for s in clear)}）—— "
              "边界敏感是好事，用例真的在考边界，答错就是真错。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

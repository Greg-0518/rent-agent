"""用例选择：日常跑哪些、收官跑哪些

单条用例 6-10s，70 条开发集全量约 12 分钟。日常改 Prompt 时等不起，
所以有一个 `smoke` 子集：**20 条、约 3 分钟、四种断言模式全覆盖、六档全覆盖**。

选型规则（不是随手挑的）：
  1. 每档至少 2 条，且覆盖该档的特色考点（L3 必带 floor 字符串比较、L4 必带 description 陷阱）
  2. 四种断言模式（`id_set`/`scalar`/`count`/`refusal`）都必须出现——
     只跑 id_set 的话，`count` 聚合识别、`refusal` 安全层那两条路径出问题不会被发现
  3. 优先选**历史上出过问题**的用例（如 `L3-001` 的 REPLACE 误杀、`L1-017` 的排序方向）
  4. 只从开发集里选——smoke 是日常迭代用的，掺保留集就污染了

子集放在代码里而不是 yaml 里：它是**跑法**（像 pytest 的选择器），不是用例自身的属性。
"""

from eval.runner.loader import Case, load_cases

# 20 条，覆盖 L1-L5 + Edge 六档、四种断言模式
SMOKE_IDS = (
    # L1 单表条件（4）：基础过滤 / 多条件 / 价格下限 / 排名方向
    "L1-002", "L1-006", "L1-011", "L1-017",
    # L2 聚合（4）：argmax 标量 / argmin 标量 / COUNT 聚合 / 分组计数
    "L2-001", "L2-004", "L2-008", "L2-012",
    # L3 复杂条件（4）：floor 数值化 / 有电梯正向语义 / IN+OR / 排名
    "L3-001", "L3-005", "L3-008", "L3-011",
    # L4 模糊语义（3）：description 映射 / 口语→电梯 / 海景干扰项
    "L4-001", "L4-004", "L4-009",
    # L5 越界（2）：写操作 + Prompt 注入
    "L5-001", "L5-010",
    # Edge 空结果（3）：朝向空集 / 阈值空集 / 多条件交叉空集
    # 注意用的是 Edge-005 而不是 Edge-010 —— 后者是保留集，掺进来就污染了日常迭代
    "Edge-001", "Edge-005", "Edge-008",
)


def select_cases(*, smoke: bool = False, holdout: bool = False, module: str = "text2sql") -> list[Case]:
    """按开关选用例。

    smoke=True    只取 20 条核心子集
    holdout=True  额外并入保留测试集（阶段收官才开）
    """
    # smoke 一律从开发集里取：它是日常迭代用的，掺进保留集就等于让保留集参与迭代，
    # "防过拟合"当场作废。所以这里**不看** holdout 开关，永远只加载开发集。
    cases = load_cases(module, include_holdout=False if smoke else holdout)
    if not smoke:
        return cases

    known = {c.id for c in cases}
    missing = [i for i in SMOKE_IDS if i not in known]
    if missing:
        # 情况一：id 拼错或被删了 → 别静默少跑（少跑几条会让"smoke 通过"变成假信号）
        # 情况二：选中的用例是保留集 → 它不在上面加载的开发集里，会落到这里
        holdout_ids = {c.id for c in load_cases(module, include_holdout=True) if c._holdout}
        polluted = [i for i in missing if i in holdout_ids]
        if polluted:
            raise ValueError(
                f"smoke 清单里混进了保留集用例：{polluted}。"
                "smoke 必须只从开发集选，否则日常迭代会污染保留集。"
            )
        raise ValueError(
            f"smoke 清单里的用例不存在：{missing}。"
            "若是有意删除，请同步更新 eval/runner/selection.py 的 SMOKE_IDS"
        )
    return [c for c in cases if c.id in set(SMOKE_IDS)]

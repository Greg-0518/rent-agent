"""用例加载 —— yaml → Case 对象，字段缺失/拼错时直接报错，不静默跳过

用例 yaml 结构（每条 case）：

    - id: L1-001                     # 唯一编号，报告与归因靠它定位
      tier: L1                       # 难度档：L1/L2/L3/L4/L5/Edge
      question: "深圳宝安区3000元以下的一居室"   # 喂给 Agent 的原始问题
      params:                        # ① 中断追问时用来作答 ② 黄金 SQL 的依据
        city: 深圳
        district: 宝安区
        budget_max: 3000
      room_count: 6                  # 期望推荐条数（决定 prompt 的 top_k 与黄金 SQL 的 LIMIT）
      assert:
        mode: id_set                 # id_set | scalar | count | refusal
        keys: [id, price]            # mode=id_set：参与比对的列（与 Agent 返回列取交集）
        value: 福田区                 # mode=scalar：期望值，在 Agent 结果里出现即通过
        count: 0                     # mode=count：期望行数
      gold_sql: >
        SELECT id, price FROM house WHERE ...
      notes: "一居室→bedroom=1 的语义映射是主要考点"
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"

VALID_TIERS = ("L1", "L2", "L3", "L4", "L5", "Edge")
VALID_MODES = ("id_set", "scalar", "count", "refusal")


@dataclass
class Assertion:
    """断言意图。**期望值不在这里**——由 `gold_sql` 执行后的黄金结果集提供
    （见 `eval/golden/text2sql.json`），避免"手写期望值"和"手写黄金 SQL"两处对不上。
    `value` / `count` 只在需要覆盖黄金结果集时才写。"""

    mode: str
    keys: list[str] = field(default_factory=lambda: ["id"])
    value: object = None
    count: int | None = None


@dataclass
class Case:
    id: str
    tier: str
    question: str
    gold_sql: str
    assertion: Assertion
    params: dict = field(default_factory=dict)
    room_count: int | None = None
    module: str = "text2sql"
    notes: str = ""

    @property
    def split(self) -> str:
        """开发集 / 保留测试集。保留集在 yaml 里显式标 holdout: true。"""
        return "holdout" if self._holdout else "dev"

    _holdout: bool = False


def _require(raw: dict, key: str, where: str):
    if key not in raw or raw[key] in (None, ""):
        raise ValueError(f"用例 {where} 缺少必填字段 `{key}`")
    return raw[key]


def _parse_case(raw: dict, source: str) -> Case:
    where = f"{source}:{raw.get('id', '<无 id>')}"
    case_id = _require(raw, "id", where)
    tier = _require(raw, "tier", where)
    if tier not in VALID_TIERS:
        raise ValueError(f"用例 {where} 的 tier=`{tier}` 非法，可选 {VALID_TIERS}")

    raw_assert = _require(raw, "assert", where)
    mode = _require(raw_assert, "mode", where)
    if mode not in VALID_MODES:
        raise ValueError(f"用例 {where} 的 assert.mode=`{mode}` 非法，可选 {VALID_MODES}")
    # scalar / count 的期望值来自黄金结果集，不用在这里写；
    # refusal 档不需要 gold_sql（它要的就是"这条 SQL 跑不起来"）
    if mode != "refusal":
        _require(raw, "gold_sql", where)

    case = Case(
        id=case_id,
        tier=tier,
        question=_require(raw, "question", where),
        gold_sql=str(raw.get("gold_sql", "")).strip(),
        assertion=Assertion(
            mode=mode,
            keys=list(raw_assert.get("keys") or ["id"]),
            value=raw_assert.get("value"),
            count=raw_assert.get("count"),
        ),
        params=dict(raw.get("params") or {}),
        room_count=raw.get("room_count"),
        module=raw.get("module", "text2sql"),
        notes=raw.get("notes", ""),
        _holdout=bool(raw.get("holdout", False)),
    )
    return case


def load_cases(module: str = "text2sql", *, include_holdout: bool = False) -> list[Case]:
    """加载某个模块下的全部用例。默认**跳过保留测试集**（只在阶段收官时用 --holdout 打开）。"""
    case_dir = CASES_DIR / module
    if not case_dir.exists():
        raise FileNotFoundError(f"用例目录不存在：{case_dir}")

    cases: list[Case] = []
    for path in sorted(case_dir.glob("*.yaml")):
        raw_list = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw_list, list):
            raise ValueError(f"{path.name} 顶层必须是 list，实际是 {type(raw_list).__name__}")
        for raw in raw_list:
            cases.append(_parse_case(raw, path.name))

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"用例 id 重复：{dupes}")

    if not include_holdout:
        cases = [c for c in cases if not c._holdout]

    return cases

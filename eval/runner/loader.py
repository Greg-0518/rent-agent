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

多轮用例（`turns:` 非空）与单轮用例的差别，只在**执行方式**上：

    - id: M-001
      tier: L1                 # 记**被断言那一轮**的难度，不是"多轮"这个形状
      turns:
        - question: 深圳3000元以下的房子          # 第 1 轮
          params: {city: 深圳, budget_max: 3000}
          room_count: 6
        - question: 那2500元以下的呢               # 同会话第 2 轮（new_session 默认 false）
          params: {city: 深圳, budget_max: 2500}
          room_count: 6
        - question: 深圳其他房源还有哪些
          params: {city: 深圳}
          new_session: true      # 这一轮开始**新会话**（同一用户，store 延续）
      assert: {mode: id_set}
      gold_sql: >                # 只对**最后一轮**判
        SELECT id, price FROM house WHERE ...

`turns` 里的每一轮都有各自的 question/params/room_count；`gold_sql` 与 `assert` 描述的是
**最后一轮**的期望（只有最后一轮进断言器）。

为什么 tier 仍然写 L1/L3 而不是"Multi"：tier 是**难度轴**，多轮是**形状轴**，混在一起
会让 `gen_report` 的分档完成率里多出一个语义不同的桶，还得出"Multi 档只有 3 条"这种
没意义的统计。多轮与否看 `trace.n_turns`。
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"

VALID_TIERS = ("L1", "L2", "L3", "L4", "L5", "Edge")
VALID_MODES = ("id_set", "scalar", "count", "refusal")


@dataclass
class Turn:
    """多轮用例里的一轮。

    自带 `params` 是因为 `agent_runner._answer_for_interrupt` 靠它作答——
    每一轮的中断追问要拿**那一轮**的参数回答，不能全用最后一轮的。
    """

    question: str
    params: dict = field(default_factory=dict)
    room_count: int | None = None
    # 这一轮是否开启**新会话**（新 thread_id，同一 user_id → store 延续）。
    # 默认 false = 接着上一轮的会话说。第 1 轮总是新会话，写不写都一样。
    new_session: bool = False


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
    # 多轮用例的轮次；空列表 = 单轮用例（走旧路径，行为与加这个字段之前完全一致）
    turns: list[Turn] = field(default_factory=list)

    @property
    def split(self) -> str:
        """开发集 / 保留测试集。保留集在 yaml 里显式标 holdout: true。"""
        return "holdout" if self._holdout else "dev"

    @property
    def is_multi_turn(self) -> bool:
        """是否按"多轮"执行。

        只要 yaml 里写了 `turns` 就走多轮路径（哪怕只有一轮）——两条路径的差别只在
        **checkpointer / store 的复用方式**：单轮每条用例一份全新的，多轮是
        一个用例一份、在轮与轮之间延续（这正是多轮才能测出来的东西）。
        """
        return bool(self.turns)

    _holdout: bool = False


def _require(raw: dict, key: str, where: str):
    if key not in raw or raw[key] in (None, ""):
        raise ValueError(f"用例 {where} 缺少必填字段 `{key}`")
    return raw[key]


def _parse_turns(raw_turns, where: str) -> list[Turn]:
    """解析 `turns:`。每轮必须有 question；params / room_count / new_session 可选。"""
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"用例 {where} 的 turns 必须是非空 list，实际是 {type(raw_turns).__name__}")
    turns: list[Turn] = []
    for i, raw_turn in enumerate(raw_turns, 1):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"用例 {where} 第 {i} 轮必须是 mapping，实际是 {type(raw_turn).__name__}")
        turns.append(
            Turn(
                question=_require(raw_turn, "question", f"{where} 第 {i} 轮"),
                params=dict(raw_turn.get("params") or {}),
                room_count=raw_turn.get("room_count"),
                new_session=bool(raw_turn.get("new_session", False)),
            )
        )
    return turns


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

    turns = _parse_turns(raw["turns"], where) if raw.get("turns") else []
    if turns:
        # 多轮的 question 由各轮拼出来，给人看的（报告里那一列）；执行用的是 turn.question。
        # 显式写了 question 就用写的——但那样容易与 turns 对不上，所以只在没写时推导。
        question = raw.get("question") or " → ".join(t.question for t in turns)
    else:
        question = _require(raw, "question", where)

    case = Case(
        id=case_id,
        tier=tier,
        question=question,
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
        turns=turns,
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

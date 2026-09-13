"""用户偏好读写的键集合不变式 —— 不需要数据库，也不需要模型

    pytest tests/unit_tests/test_store_preferences.py -q

**这个文件钉的是一个曾经让整个会话崩在第一个节点的 bug**（R4 的 D6）：

    写侧 node/recommend.py  `prefs.model_dump(exclude_none=True)`
        → 用户第一次只给了预算上限时，落库的 dict 里**没有 budget_min 这个键**
    读侧 node/recommend.py  `prefs["budget_min"]`
        → KeyError

只在**第二个会话**暴露：同一会话内新用户分支把不带 exclude_none 的完整键集合
写进了 state，只有新会话的 `get_store_info`（图的第一个节点）才重新从 store 读。

端到端复现在 `eval/tools/probe_cross_session.py`（要真模型 + 真库）；
这里是不依赖任何外部资源的那一半，秒级可跑。
"""

from pathlib import Path

import pytest

from src.agent.common.store import (
    ReservedInfo,
    UserPreferences,
    read_preferences,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------- 缺键：核心回归钉 ----------

@pytest.mark.parametrize(
    "stored, expect_min, expect_max",
    [
        ({"budget_max": 2000.0}, None, 2000.0),   # ← D6 的确切形状
        ({"budget_min": 500.0}, 500.0, None),     # 对称的另一半
    ],
    ids=["only_max", "only_min"],
)
def test_one_sided_budget_does_not_raise(stored, expect_min, expect_max):
    """只设了一侧预算时，**另一侧必须是 None 而不是 KeyError**

    这正是 D6：`read_preferences({"budget_max": 2000.0})` 以前会写成
    `prefs["budget_min"]` 直接炸掉。
    """
    pref = read_preferences(stored)
    assert pref.budget_min == expect_min
    assert pref.budget_max == expect_max


def test_reserve_only_user_has_no_budget_keys():
    """先预定、还没说过预算的用户：库里**只有** reserved_info

    这条路径写侧来自 reserve.py（同样是 exclude_none=True），
    所以两个预算键都不存在。预算读者必须能扛住。
    """
    pref = read_preferences({
        "reserved_info": [
            {"order_id": "o-1", "title": "某房源", "phone_number": "13800000000"}
        ]
    })
    assert pref.budget_min is None
    assert pref.budget_max is None
    assert pref.reserved_info is not None
    assert pref.reserved_info[0].order_id == "o-1"


@pytest.mark.parametrize("empty", [None, {}], ids=["None", "empty_dict"])
def test_no_preferences_at_all(empty):
    """新用户 / get_store_info 返回 {} 的分支 → 全 None，不抛"""
    pref = read_preferences(empty)
    assert pref.budget_min is None
    assert pref.budget_max is None
    assert pref.reserved_info is None


# ---------- 写读往返自洽 ----------

def test_write_read_roundtrip_through_exclude_none():
    """写侧真用的序列化方式（exclude_none=True）读回来必须等于原值"""
    written = UserPreferences(budget_min=500.0, budget_max=3000.0)
    stored = written.model_dump(exclude_none=True)
    assert read_preferences(stored) == written


def test_full_key_set_roundtrip():
    """键齐全时（对照：证明上面的"缺键"用例不是因为解析器一律返回 None）"""
    written = UserPreferences(
        budget_min=500.0,
        budget_max=3000.0,
        reserved_info=[ReservedInfo(order_id="o-2", title="另一套", phone_number="139")],
    )
    assert read_preferences(written.model_dump(exclude_none=True)) == written


# ---------- 形状容忍 ----------

def test_reserved_info_accepts_mixed_dict_and_model():
    """reserved_info 列表里两种形状都要能解析

    reserve.py 的"新用户"分支写 model_dump 的 dict，而"已有用户"分支
    `setdefault(...).append(ReservedInfo(...))` 塞的是**模型实例**。
    同一份库数据里两种都可能出现，解析器不能挑形状。
    """
    pref = read_preferences({
        "reserved_info": [
            {"order_id": "o-from-dict", "title": "来自 model_dump", "phone_number": "1"},
            ReservedInfo(order_id="o-from-model", title="来自模型实例", phone_number="2"),
        ]
    })
    assert [r.order_id for r in pref.reserved_info] == ["o-from-dict", "o-from-model"]


def test_unknown_key_is_ignored_not_fatal():
    """库里多出未声明的键不该崩（靠 pydantic 默认 extra='ignore'）

    这条是**假设的显式化**：如果哪天有人给 UserPreferences 配上
    `extra='forbid'`，本用例会立刻红，而不是等到线上某个会话崩了才发现。
    """
    pref = read_preferences({"budget_max": 2000.0, "some_future_field": "x"})
    assert pref.budget_max == 2000.0


# ---------- 源码级守卫 ----------

def test_recommend_has_no_raw_budget_subscript():
    """`recommend.py` 里不得再出现批量读预算的裸下标

    这是 lint 式的机械探测器，钉的正是 D6 的失效模式：写侧 `exclude_none=True`
    让键集合可变，任何 `prefs["budget_min"]` 都可能在第二个会话炸。
    只守这一个文件（其它读写点都已核实为 `.get`/`.setdefault`，见 store.py 的说明）。
    """
    src = (ROOT / "src" / "agent" / "node" / "recommend.py").read_text(encoding="utf-8")
    for bad in ('["budget_min"]', '["budget_max"]'):
        assert bad not in src, (
            f"recommend.py 出现裸下标 {bad} —— 请改用 read_preferences()"
            "（缺键会 KeyError，且只在第二个会话暴露）"
        )

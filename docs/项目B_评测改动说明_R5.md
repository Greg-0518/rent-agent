# 项目B 评测改动说明（R5）

> 配套文档：[项目B_评测缺陷报告_R4.md](项目B_评测缺陷报告_R4.md)（问题清单）
> 本文说明 **R4 之后实际改了什么、为什么这么改、以及哪些做法被我自己的证据推翻了**。

标记同 R4：`[实测]` = 跑出来的、可复现；`[代码]` = 读源码得到的；`[推断]` = 尚未验证。

---

## 0. 一句话结论

**分数从 44/70 变成 56/70，但模型这次一行代码都没改 —— 变的全部是"怎么判"。** `[实测]`

| | 通过 | 失败 | 存疑 | 不可达 | 完成率 | 模型能力口径 |
|---|---|---|---|---|---|---|
| R4（旧断言器） | 44 | 7 | 7 | 12 | 44/70 = 62.9% | 44/58 = 75.9% |
| R5（新断言器） | **56** | **2** | **0** | 12 | **56/70 = 80.0%** | **56/58 = 96.6%** |

分档：L1 16/16、L2 0/0（10 条不可达）、L3 9/11、L4 9/9、L5 8/8、Edge 14/14。

剩下 2 条失败，是真错：

| 编号 | 归因 | 一句话 |
|---|---|---|
| L3-001 | `BOUNDARY_OFF_BY_ONE` | 黄金 `>= 30` 取 4 行，模型写 `> 30` 取 3 行，漏掉 30 楼那套 |
| L3-010 | `CONDITION_MAPPED_WRONG` | 问句明说"描述里同时提到**精装**和**近**"，模型第一次把「近」窄化成「近地铁」（0 行），第二次干脆丢掉「精装」 |

> ⚠ **这轮数字不能直接当"模型进步了"对外报。** 断言器变了，基线必须重跑一次
> 才能拿到"新断言器 × 新鲜模型输出"的干净数字（见文末）。本文所有数字都是
> **同一份 trace 离线重判**出来的 —— 目的是把"断言器的影响面"和"模型的影响面"
> 分开看，不是替代重跑。

**本次一共三块工作**，前两块改判定口径、第三块改执行侧闸口：

| 块 | 章节 | 一句话 |
|---|---|---|
| 一 | §1–§11 | 断言器回看**所有**执行过的 SQL（44 → 56）、黄金放宽、新归因码 |
| 二 | §12–§15 | 跨会话 `KeyError`（D6）＋ **多轮/跨会话用例**（单轮结构上测不到它） |
| 三 | §16 | **代码沙箱**（模型写的 Python 真跑）与 **SQL 守卫**（模型写的 SQL 真打库）的收紧 |

---

## 1. 改动 1（主改）：断言器回看**所有**执行过的 SQL，而不是只判最后一条

**这是 44 → 56 的全部来源。**

### 改了什么 `[代码]`

[eval/asserters/rule.py](eval/asserters/rule.py)：原来 `assert_case` 的后半段写死
`final_sql = trace.sql_executed[-1]`，现在拆成两个函数：

```
assert_case
  └─ _judge_executed   ← 新增：在多条 SQL 里挑判定结果
       ├─ _judge_one(末条)            不过的话 ↓
       ├─ _judge_one(倒数第2条) …
       └─ _judge_one(第1条)           任一条对上 → PASS，并记下是第几条
```

判据本身（id_set / scalar / count 三套模式）**一个字没动**，只是"拿哪条 SQL 去判"
从写死变成遍历。

### 为什么这么改 `[实测]`

R4 的 D1 把这件事叫"模型凑数"，方向是"去改 Prompt 禁止它放宽"。**我去读了模型的回答正文，
它并不是在凑数：**

| 用例 | 模型实际输出（节选） |
|---|---|
| L4-002 | "明确标注可养宠物的只有 **1 套**" + 单独一节 "⚠️ 其他低价房源（宠物政策**未标注**，需自行与房东确认）…… 9 套" |
| Edge-005 | "没有符合条件的房源" + 三条**分标签**的放宽选项 + 反问用户要不要 |

也就是说：**模型把用户提出的条件翻译成 SQL 这一步做对了**，放宽只是它额外给的一份参考，
而且**在文本里明确标了那是参考**。旧的断言器只判最后一条 SQL，等于"因为它多给了一份
标注清楚的参考而判它错"。

结构性证据（12 条执行了多条 SQL 的用例，全部满足）`[实测]`：

- 首条 SQL 全部 `executed`，**0 条报错**，`retry = 0` —— 没有一条是"报错后重试"
- 全是"严格按问句查一遍 → 结果太少 → 放宽补一批"这个固定形态

改完后，**11 条新晋通过里，10 条命中的是第 1 条 SQL**（`matched_sql_index = 0`），
1 条（L4-003）命中第 2 条。没有一条是"乱撒查询撞中了"。`[实测]`

```powershell
# 复核：命中位置分布
.\venv\Scripts\python.exe eval\tools\rescore.py full-dec2
```

### 代价（知情接受）

宽松度上升：一个乱撒查询、只要撞中一条就判通过的模型，现在也会拿到分。

**所以留了两个可复核的信号**，进 `detail`：

| 字段 | 含义 |
|---|---|
| `matched_sql_index` | 命中第几条（0 = 首条 = "严格查对了"）；没命中为 `null` |
| `n_sql_executed` | 这次一共执行了几条 |

"先严格后放宽"的正常形态是 `matched_sql_index = 0`；靠乱撞命中的会落在靠后位置、
且通常伴随 `sql_errors` / `retry` 非零。两者在报告里能分开看。
**如果以后要收紧，收紧的依据就是这两个字段，不必重跑。**

### 顺带删掉了一段现在永远走不到的分支

`count` 档原有一个"模型先查到空集、又放宽补了几行 → 判 `UNVERIFIABLE`"的分支。
`_judge_executed` 现在会回看到那条查到 0 行的 SQL，而它本身就等于期望空集 → 直接判 `pass`。
原分支再也进不去，留着只会让读代码的人以为"先空后满"仍是存疑档，所以删了并在原位留了注释。

**这一档的语义因此从"判不了"升级成"通过"** —— 因为模型答对了要问的问题（Edge 档问的
就是"确实没有"），放宽部分是额外参考。

### ⚠ 连带影响：自测里有一条断言跟着失效了（事后才补上）

删分支的时候我漏了它对应的自测 —— `eval/test_asserters.py` 里
`test_edge_relaxed_fallback_is_unverifiable_and_records_counts` 断言的正是被删掉的行为
（`UNVERIFIABLE` + `executed_row_counts`）。**这条自测从改动落地起就是红的**，
而我在同一轮里只跑了 `pytest eval` 的成绩单、没看 pytest 自己那行 summary，
所以 R5 前几版没提它。

发现方式值得记下来：不是"哪次专门跑自测发现的"，是**重跑全量基线时 pytest 退出码非零**
（`3 failed, 86 passed` —— 2 条用例失败 + 1 条自测失败）才翻出来。

已经改掉，并且是**改断言方向**而不是删测试：新用例
`test_edge_strict_then_widened_passes_on_the_strict_query` 钉住新行为
（`PASS` + `matched_sql_index=0` + `matched_earlier_query=True`），
同时补了一条对照组 `test_widening_is_not_a_free_pass` ——
**钉住改动 1 的边界**：回看不是免死金牌，两条 SQL 都不对上黄金时照样判 `FAIL`。
没有这条对照组，改动 1 的"宽松度上升"就没人看着了。

现在 `pytest eval/test_asserters.py -q` → **20 passed**。

教训：**断言器的每个"删分支"动作，都对应一个要一起改的自测**。
两者在同一次提交里改，才不会留下一条红的自测；否则它就一直红着，
直到某次无关的跑批顺带把退出码带非零。

---

## 2. 改动 1b：L4-006 的黄金放宽为「豪华 ∪ 精装」

**你裁决选 B**（改黄金，而不是改种子数据）。

### 改了什么 `[代码]`

[eval/cases/text2sql/L4_semantic.yaml](eval/cases/text2sql/L4_semantic.yaml)：

```sql
-- 旧：把「装修好一点」映射成最高一档
WHERE city = '深圳' AND description LIKE '%豪华%'                       -- 2 套
-- 新：接受两档读法的并集
WHERE city = '深圳' AND (description LIKE '%豪华%' OR description LIKE '%精装%')  -- 8 套
```

### 为什么

模型返回的**正好就是这 8 套**（id 3/8/11/20/29/31/40/45），一条 SQL 查完，
并如实说明"共 8 套，不足 10 套"。`[实测]`

原黄金只取 2 套 → 判 FAIL。问题在于：「装修好一点」的语义阈值本来就落在两档之间，
只判「豪华」等于**考"猜出题人偏好哪一档"**，而不是考**"有没有把『装修好』识别成一个
装修档次条件"**。后者才是这条用例该测的东西。

**放宽后这条依然能测出东西**（不是把它改成送分题）`[实测]`：

| 模型行为 | 结果 |
|---|---|
| 返回 豪华 ∪ 精装（8 套） | pass |
| 返回全部「装修」房源（含 id 1 简单装修、18 简易装修，10 套） | **fail**，多 2 |
| 忽略装修条件、返回 10 套普通房源 | **fail** |

重新生成黄金后逐条比对，**只有 L4-006 这一条变了**，其余 88 条一字未动。`[实测]`

---

## 3. 改动 2：`detail` 里补真实条数

**为什么补**：`_preview` 把 `missing` / `extra` 截断到 3 条，是**展示用**的。
但它和真实条数混在同一个 detail 字典里，名字又都像数字，读日志的人（**包括我自己**）
会直接 `len()` 它当真实值 —— R4 §3 里我报错"六条失败全是多 3"就是这么来的。`[实测]`

```python
"missing_count": sum((gold_ms - agent_ms).values()),
"extra_count":   sum((agent_ms - gold_ms).values()),
```

两个显式命名的整数，"预览"和"真数"在命名上就分得开。预览列表本身仍是 3 条（日志不刷屏）。

---

## 4. 改动 3：新增归因码 `BOUNDARY_OFF_BY_ONE`

**你裁决：归到模型能力轴**（模型确实错了），只是标签更细。

### 为什么从 `CONDITION_MAPPED_WRONG` 里拆出来

同一个筐里装了两件**修法完全不同**的事：

| 实际错法 | 该修哪里 |
|---|---|
| 条件映射到错的值/列（L4-006 的 4 个近义词） | schema 说明、字段映射 |
| 条件映射**对了**，只是边界开闭错了（L3-001 的 `>=` 写成 `>`） | Prompt 里对中文区间的语义约定 |

混在一起时，归因表指向"改 schema 说明"，而真正要改的是 Prompt —— **方向反了**。

### 判据是机械的、误报率低

> 若 `黄金结果 ≠ 模型结果`，但 **`黄金结果翻转一个比较符后 == 模型结果`** → `BOUNDARY_OFF_BY_ONE`

L3-001 正好命中：黄金 `>= 30` 得 4 行，翻成 `> 30` 得 3 行 = 模型的结果（漏掉 id 21，30 楼）。`[实测]`

**零误报验证**：离线重判后，**只有 L3-001 一条的标签变了**，另外 6 条失败的标签一字未动。
（改动 1 落地后，那 6 条已全部转 pass。）

### 一处重复实现被合并了

比较符的"定位 + 取反"逻辑，原来在 [eval/tools/audit_boundary.py](eval/tools/audit_boundary.py)
里有一份，断言器要再用一次。各自数一遍索引迟早会错位，**而且错得很安静**
（翻转出来的 SQL 跑得通、只是答的不是同一句话，结果看起来像"没命中"）。

所以抽到新的 [eval/runner/sqlops.py](eval/runner/sqlops.py)，两边共用一份。
抽完后 `audit_boundary.py` 的输出与抽之前**逐字节相同**。`[实测]`

---

## 5. 我推翻了自己的哪个结论（R4 的 D1）

必须明说，因为它影响你对 R4 其余结论的信任度：

| R4 说的 | R5 实测 | 性质 |
|---|---|---|
| D1：模型"凑数"，根因是 `generate_query_system_prompt` 把 `top_k` 读成了配额 | **模型在回答文本里明确标注了放宽部分是参考**，不是凑数。**改成断言器判全部 SQL 后，这一档全绿，Prompt 一行没改** | **框架错了** |
| 改动 1（原方案）：改 Prompt 禁止放宽 | **没做**。证据不支持 —— 模型的行为是合理的 | **撤回** |
| 预测"改完 prompt 44 → 最多 54" | 实际 44 → 56，**但来源是断言器不是 Prompt** | 数字对上了，归因错了 |

**D1 里唯一被证实的部分**是根因定位那半句：`top_k = state.get("room_count", 5)` 这句话
确实读起来像配额。但实测下来它**没有造成损失**（见下），所以不值得为它改 Prompt。

关于 `top_k` 的实测补充（回答你上一轮问的"到底有没有 `LIMIT 10`"）`[实测]`：

- 64 条执行过的 SQL 里，`LIMIT` **全部等于 `room_count`**；`LIMIT` 是**模型自己写的**，
  不是守卫注入的（守卫的上限是 50，无关）
- 20 条 `room_count=None` 的用例执行的 SQL 数为 **0** —— 兜底值 `5` **从未被触发过**
- 推荐子图里**没有任何一处**会截断结果再展示；"只显示 5 套"不存在

---

## 6. 那 12 条"不可达"：改 Prompt 还是改代码？

**结论：改 Prompt 就够了，不必动代码。** `[实测]`

### 判定实验

写了个探针 [eval/tools/probe_route.py](eval/tools/probe_route.py)：照抄生产图的节点与连边，
**只把 `identify_question` 换成写死 `user_intent` 的节点**，其余（Prompt、子图、工具）
全是生产原件 —— 于是测出来的差异**只来自"路由"这一个变量**。

```powershell
.\venv\Scripts\python.exe eval\tools\probe_route.py          # 那 12 条
```

结果：**12/12 全部通过。** 每一条都写出了正确的 SQL、给出了正确的答案，多数只用 1 条查询。`[实测]`

| 用例 | 强制路由后写出的 SQL | 判定 |
|---|---|---|
| L2-004 | `SELECT name, district, price … WHERE city='深圳' ORDER BY price DESC LIMIT 1` | pass |
| L2-009 | `SELECT COUNT(*) … WHERE city='深圳' AND district='福田区'` → 7 | pass |
| L2-012 | `SELECT COUNT(*) … WHERE city='深圳' AND orientation='朝东'` → 12 | pass |
| L3-015 | `SELECT district, COUNT(*) … GROUP BY district HAVING COUNT(*) > 5` | pass |
| L4-010 | `SELECT district, COUNT(*), AVG(price), MIN(price) … GROUP BY district ORDER BY avg_price ASC` | pass |

（完整 12 条见 `eval/results/probe_route-unreach12.txt`）

### 所以这不是"产品缺口"，是**分类器缺定义**

两层都得改 Prompt，一层代码都不用动：

1. **`get_info` 这个标签从来没被定义过** `[代码]`
   [src/agent/node/main.py](src/agent/node/main.py) 里 `UserMessage.type` 只有一行
   `description="…推荐房源、预定房源、获取信息、合同审核、图片分析、租金计算、其他内容"`，
   **7 个标签一个都没展开**。在这个前提下，「获取信息」自然被读成"任何要事实的提问" ——
   而「深圳现在一共有多少套一居室」**确实**是在获取信息。
   节点上方的注释 `# 节点：识别用户问题：预定、推荐、我的` 说明 `get_info` 的**原意是"我的"**
   （用户自己的已存偏好/预约记录），但这句话没进 Prompt。

2. **被分到 `get_info` 之后的落点确实答不了** `[代码]`
   `get_info` → `get_user_preferences`，这个节点只读 `budget_min` / `budget_max` /
   `reserved_info`，**没有数据库访问**。所以一旦分错，模型只能拿世界知识作答 ——
   实测它在编（L2-005 答"最便宜…几百元到一千多元"，而库里深圳最便宜是 550 元）。
   这**不是**模型能力问题，是"把问题送进了没有答案的屋子"。

### 建议的最小改法

只改 [src/agent/node/main.py](src/agent/node/main.py) 里 `UserMessage` 的 `description`，
把 7 个标签展开，明确两点：

- `recommend_house` 覆盖**所有需要查房源库的问题**：推荐、以及**统计/聚合/最值类**
  （"有多少套""最便宜的是哪套""哪个区最划算""超过 5 套的区有哪些"）
- `get_info` 只指**用户自己在本系统里存过的信息**（我的预算、我的预约、我的偏好），
  **不指**对房源库的查询

> 这是**改动建议**，还没动手 —— 它改的是生产代码（会影响线上路由），等你点头。
> 改完必须重跑全量：这 12 条可达之后会进入计分，完成率的分母和分子都会变。

---

## 7. 文件清单

| 文件 | 状态 | 内容 |
|---|---|---|
| [eval/asserters/rule.py](eval/asserters/rule.py) | 改 | 改动 1（`_judge_executed`/`_judge_one`）、改动 2、改动 3；删掉 count 档的死分支 |
| [eval/runner/sqlops.py](eval/runner/sqlops.py) | 新增 | 比较符定位/取反，断言器与边界审计共用 |
| [eval/cases/text2sql/L4_semantic.yaml](eval/cases/text2sql/L4_semantic.yaml) | 改 | L4-006 黄金放宽为 豪华 ∪ 精装 |
| [eval/golden/text2sql.json](eval/golden/text2sql.json) | 重新生成 | 仅 L4-006 一条与之前不同（已逐条比对） |
| [eval/report/gen_report.py](eval/report/gen_report.py) | 改 | `ATTRIBUTIONS` 加 `BOUNDARY_OFF_BY_ONE`，`MODEL_ATTRIBUTIONS` 取前 6 |
| [eval/tools/summary.py](eval/tools/summary.py) | 改 | 归因码集合加新码 |
| [eval/tools/audit_boundary.py](eval/tools/audit_boundary.py) | 改 | 改用 sqlops，去掉重复实现（输出逐字节不变） |
| [eval/tools/rescore.py](eval/tools/rescore.py) | 新增 | 离线重判：拿已存 trace 用新断言器重判，不重跑 Agent |
| [eval/tools/probe_route.py](eval/tools/probe_route.py) | 新增 | 路由探针：强制指定意图，区分"路由问题"与"能力问题" |
| [sql/001_normalize_decoration_wording.sql](sql/001_normalize_decoration_wording.sql) | 新增 | id 45「精装三房」→「精装修三房」（§8.11 末尾） |
| [eval/tools/apply_sql.py](eval/tools/apply_sql.py) | 新增 | 执行 `sql/*.sql` 迁移的入口，可写账号 + 缺省空跑 |

§16（第三块）另动了这几个文件，不在这张表的口径里，单列：

| 文件 | 状态 | 内容 |
|---|---|---|
| [src/agent/node/finance.py](src/agent/node/finance.py) | 改 | 代码沙箱三层：AST 白名单 + 进程收紧 + 资源收紧 |
| [src/agent/state/finance.py](src/agent/state/finance.py) | 改 | `ExecutionResult` 补 `rejected` / `truncated`（走既有重试回路） |
| [src/agent/common/sql_guard.py](src/agent/common/sql_guard.py) | 改 | 掩码扫描：字面量/注释不再参与判定，LIMIT 改写不再穿进字面量 |
| [tests/unit_tests/test_code_sandbox.py](tests/unit_tests/test_code_sandbox.py) | 新增 | 沙箱单测 37 条 |
| [tests/unit_tests/test_sql_guard.py](tests/unit_tests/test_sql_guard.py) | 改 | +11 条（31 → 42），四种失效模式的回归钉子 |
| [eval/tools/audit_guard_equiv.py](eval/tools/audit_guard_equiv.py) | 新增 | 守卫改动前后的判决等价性审计（162 条重放） |
| [eval/tools/probe_sandbox_hardening.py](eval/tools/probe_sandbox_hardening.py) | 新增 | 沙箱改动前后的行为取证（E/F/G/H 四个探针） |
| [docs/Roomie知识图谱_架构与代码定位.md](docs/Roomie知识图谱_架构与代码定位.md) | 改 | 代码定位图同步到新行号与三层结构 |

**没动**：`eval/cases/` 里除 L4-006 外的用例、`eval/conftest.py`。

> ⚠ **种子的说法也要更正**：本轮**动过** `house` 表的一个字段值（id 45 的 `description`，
> 见 §8.11 末尾）。这行原写作"种子数据（`house` 表）没动"，那在本轮前半段是真的，
> 后来按 §8.11 的建议执行了归一化就不再成立。**行内容变了，但黄金没变**——这才是
> 关键的那句：89 条黄金重新执行后逐字节相同。

> ⚠ **本节最初写着「`src/` 下任何文件都不动（生产代码零改动）」，已被 §12 推翻** ——
> 本轮后来修了 D6（跨会话 `KeyError`），改了 `src/agent/common/store.py` 与
> `src/agent/node/recommend.py` 两个文件。§6 那处 `UserMessage` 改动不算在"没动"里，
> 它本来就在 [`main.py`](src/agent/node/main.py)。以 §12 为准。

---

## 8. 还没做的 / 需要你定的

1. ~~**改 `UserMessage.description`（§6）**~~ —— ✅ **已落地**，见提交 `e1102ac`。
2. ~~**重跑一次全量基线**~~ —— ✅ **已跑**：`full-dec3` = **68/70 = 97.1%**，不可达 0，见 §10。
3. ~~**`house` 表的装修措辞规范化**~~ —— ✅ **已落地**（`sql/001_normalize_decoration_wording.sql`，
   见 §8.11 末尾的"已落地"块）。
   先更正我先前的一句话：之前写"盘点下来只有一处真需要动"，那次**只看了"精装"**。
   重查全库（`WHERE description LIKE '%装%'`）后，**同义异写有两对**。
4. **19 条 holdout 名单**（实施计划 §2.3）—— 口径已定：**按飞轮"够全"即可，不比比例**
   （每类都有就行，不要求与全量同分布）。据此复核通过：六档全覆盖（L1 4/L2 2/L3 3/L4 2/L5 2/Edge 6）、
   四种断言模式全覆盖（id_set 9/scalar 1/count 7/refusal 2）、
   特殊形态全覆盖（3 对空集孪生、2 条安全层拒绝、7 条含边界措辞、2 条统计聚合）。
   所以"Edge 占 30% 偏高"不再是问题 —— 那是比例视角，不是覆盖视角。见 §11。
5. ~~**`recommend.py` 跨会话 `KeyError`**（R4 的 D6）~~ —— ✅ **已修**，见 §12。
6. **L3-010 的用例形式要不要留** —— **已核实，建议保留不改**（详见下面第 12 条）。
   它在本轮仍是原样，**等你定**。
7. **`reserve.py` 里 `ReservedInfo` 模型实例入列表（疑似同一族问题，未验证）** ——
   [reserve.py:189](src/agent/node/reserve.py#L189) 用 `setdefault(...).append(ReservedInfo(...))`
   塞的是**模型实例**，而 [main.py:96-98](src/agent/node/main.py#L96) 用 `item.get('order_id')` 读
   —— 模型实例没有 `.get`。是否真会炸取决于 store 在 `put` 时是否序列化
   （`InMemoryStore` 与生产 store 行为可能不同），**先验证再定性**：
   ```python
   from langgraph.store.memory import InMemoryStore
   store = InMemoryStore()
   store.put(("u", "preferences"), "k",
             {"reserved_info": [ReservedInfo(order_id="1", title="t", phone_number="p")]})
   print(type(store.search(("u", "preferences"))[0].value["reserved_info"][0]))
   ```
8. **`user_id` 兜底策略三处不一致** —— [main.py:68](src/agent/node/main.py#L68) 没兜底
   （可能造出 `(None, "preferences")` 这个命名空间）、recommend.py 兜 `"default"`、
   [reserve.py:171](src/agent/node/reserve.py#L171) 直接返回提示语。
9. **store 值的别名问题** —— [main.py:72](src/agent/node/main.py#L72) 把 store 的 value 对象
   直接塞进 State；[recommend.py:195](src/agent/node/recommend.py#L195) 拿到同一对象后
   就地改、再 put。当前"改了就 put"使其自洽，但"先改后 put 失败"会留下内存里已被改动的条目。
10. ~~**harness 支持多会话用例**（用例级 store + 会话级 checkpointer + yaml 多轮字段）~~
    —— ✅ **已落地**，见 §15（`turns:` 字段 + 3 条用例 + 新归因 `GRAPH_CRASH`）。

> 下面 §8.11 / §8.12 是前述第 3、6 条"待你定"的核实与建议（对应"第 11、12 条"），
> §8.13 是这轮新发现的一条产品问题，§8.14 是一条要你定的口径。
> **编号沿用文档原有的说法**（列表只到第 10 条，之后的都以 §8.x 呈现），
> 避免把已有引用改乱。

---

### 8.11 `house` 装修措辞：核实数据与建议（对应上面第 3 条）

**全库盘点**（`WHERE description LIKE '%装%'`）：只有深圳有装修类描述，共 11 套：

| 措辞 | id |
|---|---|
| `精装修` | 3, 8, 11, 29, 31 |
| **`精装三房`** | **45** ← 唯一不带「修」的 |
| `豪华装修` | 20, 40 |
| `简单装修` | 1 |
| `简易装修` | 18 |
| `新装修` | 19 |

**同义异写两对**（其余是不同档次，不算同义）：

- `精装` ↔ `精装修`
- `简单装修` ↔ `简易装修`（id 1 / id 18，**无任何用例涉及**）

**影响面已实测**：`%精装%` = 6 套（含 45）；`%精装修%` = **5 套**（不含 45）。
把 id 45 改成「精装修三房」后，**两个黄金都不变**——L3-010 仍 6 套、L4-006 仍 8 套，
因为 `%精装%` 天然匹配 `精装修`。（全仓无 `.sql`，种子数据**不在仓库里**。）

**建议改，但理由不是判分，是评测的公平性**：现状下写 `%精装修%` 的模型会少 id 45 而**判错**
——**用词更完整反而被判错**。L4-006 的问句是自然说法（「装修好一点」），模型写哪个词都不奇怪，
而它的黄金恰好是 `%豪华% OR %精装%`。改完两种写法都给同样的答案，这个陷阱就没了。

**若改，请同时满足一条**：把这条 UPDATE 落成仓库里的一个 `sql/` 文件。否则
"改共享库不留版本记录"这个反对理由永远成立——那正是我先前不动它的理由，
而它只在"改法不进仓库"时成立。**优先级低**：今天对判分零影响，属于数据卫生。

**已落地（本轮）** —— 前置条件已满足，所以执行了：

| 项 | 内容 |
|---|---|
| 迁移文件 | `sql/001_normalize_decoration_wording.sql`（`id = 45 AND description = <完整旧值>`） |
| 执行入口 | `eval/tools/apply_sql.py <file> [--yes]`，用**可写账号** `DB_USER`，缺省空跑 |
| 实际执行 | `affected rows = 1` |

只动 id 45，**不动 id 18 的「简易装修」**：后者没有任何用例涉及，改了不影响任何判定，
属于纯数据卫生；把无评测影响的改动一并塞进来只会让"这次到底改了什么"变模糊。
判断依据与整份影响面都写进了 `.sql` 文件的头注释，将来要动 id 18 照抄它的形状即可。

**逐条复核（命令即 R5 §9 那几行）**：

| 检查 | 期望 | 实测 |
|---|---|---|
| `SELECT description FROM house WHERE id = 45` | `有电梯，近4号线，精装修三房` | ✅ |
| `%精装%` 计数 | 6（**不变**） | ✅ 6 |
| `%精装修%` 计数 | 6（原 5，并入 45） | ✅ 6 |
| L3-010 黄金 | 6 套、id 集合不变 | ✅ 3/8/11/29/31/45 |
| L4-006 黄金 | 8 套、id 集合不变 | ✅ 3/8/11/20/29/31/40/45 |
| `house` 总行数 | 50（未误伤） | ✅ 50 |
| **重复执行迁移** | `affected rows = 0`（幂等） | ✅ 0 |
| `git diff eval/golden/text2sql.json` | **只有 `generated_at` 一行** | ✅ 1 insertion / 1 deletion |

最后一行是本次最有力的证据：**重新执行全部 89 条黄金 SQL 后，黄金内容一个字节都没动**
——即"改了库但没有任何一条用例的期望值跟着变"，这正是本轮的目标。
（`generated_at` 那一行仍然提交了：它把"改动库之后重新生成过"这件事留在了 git 历史里，
将来有人 `git log -p` 看黄金时会看到这次时间戳，不会误以为黄金从此未复现过。）

**回归**：`pytest eval -k "L3-010 or L4-006" --run-id q45-regress -q` → `2 passed`。
只挑这两条而不是跑全量 smoke，理由是**只有这两条的黄金含装修条件**（全仓 grep 确认），
其余 87 条的问句与黄金都不碰 `description` 里那几个词，库改了而它们的期望值没变，
跑它们只会在噪声里多花一次 API 调用。

---

### 8.12 L3-010 的用例形式：核实与建议（对应上面第 6 条）

**它的不可替代性**：**套件里唯一一条「两个 description 条件取交集」**的用例
（`%精装% AND %近%`）。其余 description 条件都是"一个 description 条件 + 价格/楼层/居室"。

**"能不能换成自然说法"——不能，`%近%` 是承重的**（深圳实测）：

| 候选组合 | 交集 |
|---|---|
| `%精装% AND %近%` | **6 套** ✓ |
| `%精装% AND %有电梯%` | 6 套，但「有电梯」35 套，AND 不 discriminate，退化成单条件 |
| `%精装% AND %养宠物%` / `%阳台%` / `%采光%` / `%近地铁%` | **0 套** → 变空集用例，而 Edge 层已有空集孪生 |

`%近%` 是**子串**（出现在 近公交站 / 近购物公园 / 近腾讯大厦 / 近科技园 / 近软件园 /
近4号线 / 近工业园 / 近海岸城 里），现实里没有一个自然词对应它。**所以元问题形式是
数据逼出来的，不是出题人偷懒。** 换成自然说法就得换掉 `%近%`，而新题是空集。

**它抓的失效也是真能力**：历史失败为 `CONDITION_MAPPED_WRONG`、结果是**超集**（"多 4 种"），
像是把「同时提到」当成了并集或漏掉了某个条件——**"漏掉一个条件"正是这条唯一能抓的**。

**建议：保留。** 但读失败名单时要记住它 **flip-flop**（`full-dec2-rescored` 红、
`full-dec3` 绿）——单次运行里它可能是噪声，**反复红**才值得追。

### 8.13 只给了一侧预算，仍被追问「预算范围」（本轮新发现，未修）

**现象**：用户已经说了上限（「深圳光明区500元以下的房子」），首轮仍然会追问一次
「请提供以下信息：**预算范围**」。

**位置**：[recommend.py:130](src/agent/node/recommend.py#L130)

```python
if updated_state.get("budget_min") is None or updated_state.get("budget_max") is None:
    missing_info.append("**预算范围**")
```

条件是 `or` —— **两侧只要缺一侧就算"预算缺失"**。「1000元以下」这类说法只落 `budget_max`，
`budget_min` 必然是 None，所以信息明明给了，还是被判成缺失。

**证据（开发集 70 条里问句自带价格上限的那 7 条，7/7 全部被追问）**：

| 用例 | 问句（自带上限） | 首轮中断 | 中断答案 |
|---|---|---|---|
| Edge-003 | 深圳光明区**500元以下**的房子 | 含「预算范围」 | 深圳，光明区，预算500元以下 |
| L1-002 | 深圳南山区**2000元以内**的两居室 | 含「预算范围」 | 深圳，南山区，预算2000元以下 |
| L1-004 | 深圳龙岗区**1500元以下**的房子 | 含「预算范围」 | 深圳，龙岗区，预算1500元以下 |
| L1-007 | 深圳光明区**2000元以下**的房子 | 含「预算范围」 | 深圳，光明区，预算2000元以下 |
| L1-010 | 深圳罗湖区**2000元以下**的房子 | 含「预算范围」 | 深圳，罗湖区，预算2000元以下 |
| L4-009 | 深圳有海景并且租金**不超过2500** | 含「预算范围」 | 深圳，预算2500元以下 |
| L4-012 | 深圳有电梯的三居室，**预算3000以内** | 含「预算范围」 | 深圳，预算3000元以下 |

复核命令（问句带回价格上限的开发集用例恰好 7 条，且**全部**被追问）：

```powershell
# 判据：问句里同时出现数字与 (预算|元以下|元以内|以内|不超过)
.\venv\Scripts\python.exe -c "import json,re;rs=[r for r in map(json.loads,open('eval/results/full-dec3.jsonl',encoding='utf-8')) if re.search(r'\d',r.get('question','')) and re.search(r'预算|元以下|元以内|以内|不超过',r.get('question',''))];print([r['case_id'] for r in rs]);print(sum('预算范围' in str((r.get('trace') or {}).get('interrupts',[{}])[0].get('prompt','')) for r in rs),'/',len(rs))"
# → ['Edge-003', 'L1-002', 'L1-004', 'L1-007', 'L1-010', 'L4-009', 'L4-012']
#   7 / 7

# 逐条看中断内容
.\venv\Scripts\python.exe eval\tools\peek_case.py full-dec3 L1-007
```

> 判据里写的是「元以下/元以内」而不是裸的「以下」：放宽成裸「以下」会多捞一条
> L3-002（「宝安区或龙岗区里，**楼层10楼以下**的一居室」），而它问句里没有预算、
> 被追问是对的。我第一版就是这么捞的（得 8 条），所以这里把判据收紧了——**8 条里有
> 1 条是我的正则误伤**，实际 7 条全中。

**为什么这不是纯体验问题**：

1. **每一轮都多一次往返**。这是"图入口节点每轮重跑"的另一个后果：
   `full-dec3` 70 条里 **62 条带 2 次中断**（首次预算追问 + 后续预订追问），
   只有 8 条是 0 次中断——也就是说"第一次调用必被追问一次"几乎是普遍路径。
   评测里这次追问由 `params` 代答，只花时间不改变判定（这也意味着**用例掩盖了它**：
   7 条用例照样 PASS）；真实产品里是用户被多问一句，而且问的正是他刚说过的信息。
2. **`不提供` 分支会凭空造出另一侧**。[recommend.py:141-144](src/agent/node/recommend.py#L141)：

   ```python
   if not updated_state.get("budget_min"):
       updated_state['budget_min'] = 500.0
   ```

   用户说「1000元以下」、被追问后答「不提供」→ `budget_max` 保留 1000，
   但 `budget_min` 被填成 **500**。用户从未表达过下界，
   于是「500–1000」成了一个凭空收窄的条件（写死的 500/3000 与 `house` 的实际价格
   区间恰好接近，纯属巧合，不是推导出来的）。

   **[疑似，未验证]** 这个下界会不会真的进 SQL，我没有实测到：全量 336 条记录里
   「不提供」出现 **0 次**（评测用真实值作答，走不到那个分支），
   `final_sql` 里出现 `price >= 5xx` 的也是 **0 条**。所以这条是**代码推断**，
   不是实测结论——要定性得先让那个分支跑起来（构造一条 `answer: 不提供` 的中断）。

**建议的最小改法**（本轮不改，等你定）：把判据从「缺一侧」改成「两侧都缺」，
并把默认值从写死的 500/3000 改成按数据实际区间取；追问文案同时说明"上限已收到，
只差下限"。

### 8.14 三条多轮用例要不要进 `SMOKE_IDS`（要你定的口径）

**当前决定：不进。** smoke 的价值全在于**id 集合固定**——历史 `smoke-*` 几轮之所以能
逐条对比，就是因为跑的是同一批 20 条；往里面加 3 条，等于悄悄换了基线，
之后说"和上一轮 smoke 一致"就不成立了（§13 那句"与 `smoke-d6` 逐条相同"
正是靠这个前提才成立）。另外这 3 条是全套件最贵的用例（每条 2–3 轮、4–6 次中断）。

**但要如实说清它留下的洞**：多轮用例守的是 `get_store_info`（图的**入口节点**，
每轮都跑），而 20 条 smoke 用例**每条都用全新 store + 独立 `user_id`**，
结构上永远进不了"同一个用户的第二次调用"——所以**这个洞 smoke 抓不到**，
这正是 §15 那三条用例存在的理由。

两条可选做法（都不做，等你定）：

| 方案 | 代价 |
|---|---|
| 把 M-001 加进 `SMOKE_IDS` | 基线从 20 条变 21 条，历史 smoke 对比线**断开**，需要一句显式声明 |
| 加一个 `pytest.mark.multiturn` 标记（按 `case.is_multi_turn` 打，`test_text2sql.py:43` 附近改 3 行） | smoke 基线不动，CI 里可 `-m "smoke or multiturn"` 一起跑；多一个标记要维护 |

在不选之前，按需跑的回归命令是：

```powershell
.\venv\Scripts\python.exe -m pytest eval -k "M-00" --run-id smoke-multiturn -q
```

---

## 9. 复核命令

```powershell
# 1) 断言器改动的全貌（离线重判，不重跑 Agent，秒级）
.\venv\Scripts\python.exe eval\tools\rescore.py full-dec2

# 2) 重判后的成绩单（本文所有数字的来源）
.\venv\Scripts\python.exe eval\tools\rescore.py full-dec2 --write
.\venv\Scripts\python.exe eval\report\gen_report.py --run-id full-dec2-rescored

# 3) 剩下 2 条失败到底错在哪
.\venv\Scripts\python.exe eval\tools\peek_case.py full-dec2 L3-001
.\venv\Scripts\python.exe eval\tools\peek_case.py full-dec2 L3-010

# 4) 那 12 条不可达：强制路由后能不能答对（会真的调模型，约 2 分钟）
.\venv\Scripts\python.exe eval\tools\probe_route.py
.\venv\Scripts\python.exe eval\tools\probe_route.py L2-009          # 单条
.\venv\Scripts\python.exe eval\tools\probe_route.py --intent get_info L4-002

# 5) 断言器自测（黄金回灌 + 反方向，需要 MySQL）—— 注意是 eval/ 下这个
.\venv\Scripts\python.exe -m pytest eval\test_asserters.py -q      # 20 passed
.\venv\Scripts\python.exe -m pytest tests\unit_tests -q            # 97 passed（不依赖库；含 §16 的沙箱/守卫 86 条）

# 6) 边界体检（口径是否自相矛盾）
.\venv\Scripts\python.exe eval\tools\audit_boundary.py

# 7) 本次全量基线（70 条开发集，约 9 分钟）+ 成绩单
.\venv\Scripts\python.exe -m pytest eval --run-id full-dec3 -q
.\venv\Scripts\python.exe eval\report\gen_report.py --run-id full-dec3

# 8) 保留集：只在阶段收官跑一次（会显式加 --holdout，89 条）
.\venv\Scripts\python.exe -m pytest eval --holdout --run-id closing
```

> ⚠ 跑完**一定要看 pytest 自己那行 summary**，别只看成绩单。
> 成绩单只统计用例判定（从 jsonl 读），**自测失败不进成绩单** ——
> 只看得分就会漏掉一条红的自测（本次就是这么漏了一轮，见 §1 的连带影响）。

---

## 10. 本次全量基线（`full-dec3` → `full-mt`）

**当前可对外引用的数字是 §10.1 的 `full-mt`**（73 条 = 70 条开发集 + 3 条多轮用例）。
§10.2 的 `full-dec3` 是**加入多轮用例之前**的同口径基线，留着是为了让"新增用例没有扰动
既有 70 条"这件事可核对。两者对得上的部分是：既有 70 条里 `full-dec3` 红 2 条、
`full-mt` 红 1 条，差的正是 L3-012 那一次**已知的浮动翻转**（不是新用例造成的，
理由见 §10.2 末尾"失败名单不稳定"）。
（`full-dec2-rescored` 是离线重判同一批旧轨迹，Agent 行为没变，只能说明断言器的影响。）

### 10.1 `full-mt`（2026-09-13 22:46，73 条，当前基线）

```powershell
.\venv\Scripts\python.exe -m pytest eval --run-id full-mt -q     # 1 failed, 92 passed in 10:39
.\venv\Scripts\python.exe eval\report\gen_report.py --run-id full-mt
```

> 那个 "1 failed" 是**用例**失败（下面 L3-001），不是套件坏了。
> 92 passed = **72 条用例** + **20 条断言器自测**（`eval/test_asserters.py`）——
> 两者混在同一行 summary 里，所以数字和用例数对不上，这是预期的。

| 口径 | 数字 |
|---|---|
| 全量口径 | **72/73 = 98.6%**（pass 72 / fail 1 / 存疑 0 / 不可达 0） |
| 分档 | L1 19/19 · L2 10/10 · **L3 11/12** · L4 10/10 · L5 8/8 · Edge 14/14 |
| 分模式 | id_set 39/40 · scalar 7/7 · count 18/18 · refusal 8/8 |
| 过程信号 | 先查 schema 65/73=89.0% · SQL 报错重试 **0** 次 · 守卫拦截 **0** 次 · 中断-恢复 138 次 · 图步数中位 6/最大 18 |
| 时延 / 成本 | P50 8.5s · P95 14.3s · max 26.5s；token 合计 228,183（单条均 输入2,658/输出467） |

**失败只有 1 条**：L3-001 `BOUNDARY_OFF_BY_ONE`（「30楼以上」写成 `> 30`，漏掉 id 21）。
比 `full-dec3` 少的那条是 L3-012（`CONDITION_MAPPED_WRONG`，否定被丢掉）——它这轮自己过了，
**不是修好了**，理由见上面"失败名单不稳定"那段：`full-dec3` 红、`full-mt` 绿，
正好是第三个数点（L3-001 才是三轮都红的那个）。

**三条多轮用例全 PASS**，且**最慢的 3 条就是它们**（M-003 26.5s / M-002 16s / M-001 16s）——
这是 §8.14 里"它们不进 `SMOKE_IDS`"那条决定的事实依据：不是每条 4-6 次中断的用例
都适合放进每天跑的核心子集。

### 10.2 `full-dec3`（2026-09-13 21:13，70 条，多轮用例之前）

| 口径 | full-dec2（旧断言器） | full-dec2-rescored（同批轨迹·新断言器） | **full-dec3（重跑）** |
|---|---|---|---|
| 通过 | 44 | 56 | **68** |
| 失败 | 7 | 2 | **2** |
| 存疑 | 7 | 0 | **0** |
| 不可达 | 12 | 12 | **0** |
| 全量口径 | 44/70 = 62.9% | 56/70 = 80.0% | **68/70 = 97.1%** |
| 模型能力口径 | 44/58 = 75.9% | 56/58 = 96.6% | **68/70 = 97.1%** |

**两个口径在 full-dec3 合并了**（不可达归零 → 分母不再需要剔除）。这正是 §6 修复完成后的预期形态。

**分档**：L1 16/16 · L2 10/10 · **L3 10/12** · L4 10/10 · L5 8/8 · Edge 14/14。
**分模式**：id_set 35/37 · scalar 7/7 · count 18/18 · refusal 8/8。
**过程信号**：SQL 报错重试 **0 次**、守卫拦截 0 次、单条均值 1.1 条 SQL、P95 11.1s、token 合计 198,730。

### 剩下的那 2 条（`full-dec3`；都归模型能力，都在 L3）

| 用例 | 归因 | 实际错法 |
|---|---|---|
| L3-001 | `BOUNDARY_OFF_BY_ONE` | 「30楼**以上**」写成 `> 30`（应为 `>= 30`），漏掉 id 21 那套 30 楼的 |
| L3-012 | `CONDITION_MAPPED_WRONG` | 「**不是**朝北的房子里最便宜的3套」——**否定被丢掉了**，写成 `orientation='朝北'`，还多带了一个问句里没有的 `price <= 3000` |

### ⚠ 一条必须说清楚的：失败名单不稳定

`full-dec2-rescored` 的 2 条失败是 **L3-001 + L3-010**；
`full-dec3` 重跑后是 **L3-001 + L3-012** —— **L3-010 自己好了，L3-012 自己坏了**，净数不变。

这说明在 **±1 条**的粒度上，失败名单是**随机的**（生成 SQL 那步的采样温度没锁死），
所以：

- **不能说**"L3-010 已经修好了"——它是这一轮恰好通过；
- **不能说**"L3-012 是新引入的回归"——prompt 没改过 SQL 生成那一段（§6 只动意图分类）；
- 单轮失败名单只能当**症状清单**用，不能当**趋势**用。要谈趋势得跑多次取分布。

R5 前几版里"剩下 2 条 = L3-001 + L3-010"的说法，按此更正为"剩下 2 条，成员每轮浮动"。

---

## 11. 保留测试集（19 条）的"够全"复核

**口径来源**：你给的判据 —— 按 Agent 飞轮的说法，保留集**够全、包含所有类型即可**，
不要求与全量同分布。这是个**覆盖**判据，不是**比例**判据。

（此前我按比例视角提过一个顾虑：保留集里 Edge 占 30%（6/19）、高于全量的 22%，
因为三对孪生整对落进来了。按覆盖判据这个顾虑不成立，撤回。）

复核结果（`holdout: true` 逐条统计）：

| 轴 | 全量有几种 | 保留集覆盖 | 缺 |
|---|---|---|---|
| 难度档 | 6 | Edge 6 / L1 4 / L2 2 / L3 3 / L4 2 / L5 2 | 无 |
| 断言模式 | 4 | id_set 9 / count 7 / refusal 2 / scalar 1 | 无 |

特殊形态也都有代表：**3 对空集孪生**（Edge-002/006/010 及其 T）、
**2 条安全层拒绝**（L5-004 建表、L5-009 `DROP TABLE users` 双触发）、
**7 条含边界措辞**（以上/以下/超过/高于）、**2 条统计聚合**（L2-002 最便宜、L2-010 带电梯套数）。

> 孪生"整对同边"是硬约束（`audit_twins.py` 第 5 条不变式：**划分一致**），
> 因为收官跑只跑一遍，配对拆开就只剩半边、控制变量结构断掉。已实测：
> `python eval/tools/audit_twins.py` → **配对 10 对，配对全部成立：每条原用例空集、
> 每条孪生非空、恰好差一个条件、划分一致。**

### 保留集的干净程度（如实说明）

保留集在这轮**不是完全没被碰过**：

- `make_golden.py` / `audit_boundary.py` / `audit_intent.py` 都在 89 条（含保留集）上跑过
  → **题面和黄金结果我读过**；
- §6 改完后跑全量 89 条确认"不可达 12 → 0"，**这一步包含保留集**，等于拿它做了一轮验证。

但**改动本身不是它逼出来的**：12 条不可达全在开发集，L4-006 也是开发集。
若当时修完 L2-002 仍不可达，我会回头再改 Prompt —— 那就真算拿保留集调参，实际没有（一次全绿）。

结论：**保留集在「模型/prompt 泛化」这条轴上仍然干净，在「用例设计」这条轴上已不干净。**
收官跑出的分能回答"这套 prompt 在新题上还成立吗"，**不能**回答"我的题出得好不好"。

> 读 jsonl 请务必带 `encoding="utf-8"`。Windows PowerShell 的 `Get-Content`
> 会把 UTF-8 当 ANSI 读，中文全乱码，`ConvertFrom-Json` 也会失败 —— 这不是文件坏了。
>
> 另外：**PowerShell 里别用 `-c` 传含引号的 SQL**，解析器会把它当命令。写临时脚本文件跑。

---

## 12. 修 D6：跨会话读用户偏好时的 `KeyError`

**这是本轮唯一一处生产代码改动**（前 11 节全是评测侧的）。

### 症状与根因

用户第一次只给了预算上限，**下一次开口**（新会话，或同一会话发第二条消息）直接崩在
**第一个节点**：

```
KeyError: 'budget_min'
```

不是"某条链路答得不好"，是**整个会话在入口就死**——`get_store_info` 是图的**入口
节点**（`graph.py` 的 `add_edge(START, "get_store_info")`），而且**每轮都会重跑**。

根因是写侧与读侧对「键集合」的假设不一致：

| 侧 | 位置 | 行为 |
|---|---|---|
| 写 | `recommend.py` `prefs.model_dump(exclude_none=True)` | **键集合随数据而变**——只给了上限时，落库的 dict 里根本没有 `budget_min` 这个键 |
| 读 | `recommend.py` 5 处裸下标 `pref["budget_min"]` | 缺键即 `KeyError` |

**为什么不是当场就炸**：那一轮走的是写侧新用户分支，它写进 state 的是
`prefs.model_dump()`（**不带** `exclude_none`），键齐全，所以同一轮读不到缺键版本。

**什么时候炸：下一次调用**。`get_store_info` 是图的入口且**每轮重跑**，并且
**无条件**用 store 里的值覆盖 state（`node/main.py:70-72` 没有条件判断），于是上一轮
留在 state 里的齐全版本被盖成缺键版本。两种触发都实测过：

| 触发 | 是否命中 |
|---|---|
| 新会话（新 thread，同 `user_id`） | 命中 |
| **同一会话发第二条消息** | **命中也** |

第二行是这轮才发现的。原先的说法是"同一会话内 state 里有齐全版本，所以安全"——
**不成立**，入口每轮都会把它盖掉。这不只是措辞问题，它把**严重度也说小了**：
用户不必等下次再来，**同一次对话里说第二句就崩**。

**为什么评测集也没抓到**：89 条用例全是单轮的，每条还拿**全新 store + 独立 `user_id`**
（`eval-<case_id>`），于是"store 里已经有这个用户的偏好"这个状态**在评测里压根构造不出来**：
写方向的 `store.put` 每轮都跑，**读方向的分支一次都没进过**。

### 改了什么

**只有读侧**，统一到一个入口：

| 文件 | 改动 |
|---|---|
| `src/agent/common/store.py` | 新增 `read_preferences(value) -> UserPreferences` |
| `src/agent/node/recommend.py` | 2 个读点（state 里的偏好、store 里的偏好）改走它，消掉 5 处裸下标 |

**没改写侧的 `exclude_none=True`** —— 读侧正规化能同时容下**修之前就已落库的旧数据**；
写侧改法只管未来写入，历史行照样崩。

**一个必须保留的细节**：`prefs = prefs_result[0].value`（原始 dict）**没删**。
写回路径（就地改 + `store.put`）继续用它，所以现在是**一读一写两个表示**：
读预算走 `stored`（正规化、缺键即 None），写回 store 走 `prefs`（原始 dict）。

> ⚠ **这一条我第一版给的理由是错的，已更正。** 原文写的是"`read_preferences` 走
> pydantic 会丢掉 `reserved_info`"——经核实**不成立**：`reserved_info` 在
> `UserPreferences` 里**有声明**（`store.py:45`），实测 round-trip 之后它还在，不丢。
> 真正会丢的只有**未声明**的键，而今天**不存在**这种键（三处 `store.put` 全都经过
> `UserPreferences`，已逐处核对）。
> 保留原始 dict 仍然是对的选择，但理由是另外两条：① 对"将来库里多出未声明的键"
> 免疫（没有任何机制在强制保证这一点）② 保持"少写"——`model_dump` 会把
> `exclude_none` 省掉的键补成显式 `None` 再存回去，抵消写侧意图。
> 顺带把 `prefs` 就是 store **活对象**这条别名风险也写进了注释（指向 §8 第 5 项）。
>
> **教训：注释里给"为什么"要给到可验证的那一层。** "某个库会丢掉某个字段"这种
> 断言写下来之前应该先跑一遍——错的理由比没有理由更糟，它会把下一个读代码的人
> 引到错误的防范方向上。

### 验证（三件都可复核）

**1. 端到端：修前两处都红、修后两处都绿**
（`eval/results/probe_cross_session-before.txt` 与 `...-fixed.txt`——同一个脚本、
同样三次调用，只差两个源文件的版本）

| 调用 | 修前 | 修后 |
|---|---|---|
| 会话一·第一轮（store 为空） | `OK` | `OK` |
| 会话一·**第二轮（同一 thread）** | **`KeyError: 'budget_min'`** | `OK  user_preferences={'budget_max': 3000.0}` |
| 会话二（新会话） | **`KeyError: 'budget_min'`** | `OK  user_preferences={'budget_max': 3000.0}` |

后两行打出来的 `{'budget_max': 3000.0}` —— **没有 `budget_min` 键的那个形状** ——
正是以前必然 KeyError 的输入。所以这两轮是**真的走到了**那条曾崩掉的路径并且
活了下来，不是"路径没被覆盖到所以显示 OK"。

修前复现（三段，最后一段负责还原）：

```
git checkout cff4d8f -- src/agent/common/store.py src/agent/node/recommend.py
.\venv\Scripts\python.exe eval\tools\probe_cross_session.py
git checkout HEAD -- src/agent/common/store.py src/agent/node/recommend.py
```

> 探针这轮也跟着改了：`run()` 多了个可选 `checkpointer` 参数。原来每次 `run()` 都
> 新建 `MemorySaver()`，**thread 状态根本不跨调用**，所以它只能测"新会话"那一半
> ——这也正是"同一会话第二轮也会炸"这件事一直没被发现的原因。传同一个
> checkpointer 才是"同一会话的下一轮"。现在两种触发都在回归覆盖里。

**2. 单测：新增 10 条，零成本**（`tests/unit_tests/test_store_preferences.py`，不依赖库/模型）

`39 → 49 passed`。其中第 10 条是**源码级守卫**（断言 `recommend.py` 里不再出现
`["budget_min"]`/`["budget_max"]`）。守卫的有效性也验证过——拿 HEAD 的旧版本对照：

```
旧代码命中 5 处（line 66/68/69/196/197）   ← 守卫在修之前会红，它有效
当前代码 无命中
```

> 改前 `tests/` 下 grep `budget_min|budget_max|reserved_info` **零匹配**：
> 偏好读写路径此前**完全没有测试**。

**3. 没碰坏既有行为**：smoke 20 条（见 §13）。

### 教训

**"写方向每轮都跑"不等于"这条链路被测过"。** 89 条用例的隔离粒度是**会话**，
每个用例一个全新 store + 全新 `user_id`，于是所有"读已有数据"的分支
**结构性地不可达**——不是没测到，是**造不出来**。
`probe_cross_session.py` 的 docstring 早就写了这一条，但直到这轮才把修复做掉。

这也解释了为什么它能在多次全量回归里活着：**它不在任何一条用例的路径上**。
覆盖率看的是"代码被执行过"，而这里是"状态被构造过"——后者不会出现在任何覆盖率数字里。

---

## 13. 改动后的 smoke 回归（`smoke-d6`）

**20 条 smoke 用例，19/20 = 95.0%**，`fail=1 / 存疑=0 / 不可达=0`。

| 口径 | smoke-dec1（早先） | **smoke-d6（本次）** |
|---|---|---|
| 完成率 | 13/20 = 65.0% | **19/20 = 95.0%** |
| 分布 | pass=13 fail=2 存疑=1 不可达=4 | **pass=19 fail=1 存疑=0 不可达=0** |

**差异来自 §6 的路由修复**（4 条不可达转可达），**不是** §12 的 D6 修复
——两者不是同一批改动，这里读的是"有没有变坏"，不是"D6 带来了多少分"。

**唯一失败的是 L3-001**，归因 `BOUNDARY_OFF_BY_ONE`：模型把「30楼**以上**」写成 `> 30`，
漏掉 id 21 那套 30 楼的。**这是已知的既有失败**，与偏好读写无关
（它 1 条 SQL、无重试、无守卫拦截；`full-dec2-rescored`、`full-dec3` 里也都失败）。

### 专门查了执行异常（这是 D6 改动真正的风险所在）

改动落在 `collect_user_info`，每个 `recommend_house` 用例都会走到。若有隐患，
表现是**轨迹抛异常**而不是判定变化，所以单独查了一遍：

```
总记录 21（1 行 run_meta + 20 条用例）
trace.ok=False: 0
有 error 字段: 0
判定分布: {pass: 19, fail: 1}
```

**零执行异常。** 另外断言器自测 `pytest eval/test_asserters.py -q` → **20 passed**
（注意 `-m smoke` **不跑**自测——它没有 smoke 标记，别拿 smoke 的绿当作自测的绿）。

---

## 14. 本轮全部提交（按批次）

| commit | 内容 | 范围 |
|---|---|---|
| `a4c64a9` | src：工作区既有的第三方改动 | 非本轮工作 |
| `8e3ecd0` | src(guard)：SQL 执行安全层 | 生产 |
| `e1102ac` | 意图分类：给 7 个标签补定义（§6） | 生产 |
| `68616fe` | eval：评测 harness + 断言器改动 | 评测 |
| `3edf20e` | docs：R5 + 更正 R4 | 文档 |
| `6ca655c` | eval(asserters)：修正失效自测 + 补反向对照 | 评测 |
| `cff4d8f` | eval(results)：full-dec3 轨迹 | 证据 |
| `817d55d` | docs：R5 §10/§11 | 文档 |
| `678f856` | **src：修 D6 跨会话 KeyError（§12）** | **生产** |
| `78ea6ea` | eval：探针转回归检查 + 证据 | 证据 |
| `8d17c87` | docs：R5 §12/§13/§14 | 文档 |
| `adad86d` | tests：偏好读写 / guard 接线 / SQL guard 单测 | 测试 |
| `a75a384` | .claude：评测 harness 方法论沉淀为 skill | 方法论 |
| `d00bf76` | src：更正 D6 注释里给错的理由（§12） | 生产（仅注释） |
| `3ccf12e` | src/tests：更正 D6 **触发条件**——同会话第二轮也炸 | 注释（零行为变化） |
| `1e5ed1b` | eval：探针补同会话第二轮 + 修前证据文件 | 证据 |
| `04328a0` | docs：R5 §12 同步触发条件与两处更正 | 文档 |
| `84e9670` | docs：R5 §8 补两条待定项的核实数据与建议 | 文档 |
| `854b271` | **eval+sql：id 45 装修措辞归一化落地（§8.11）** | **数据（新增 `sql/`）+ 评测** |
| `74f07e4` | **eval：harness 支持多轮用例 + 新归因 GRAPH_CRASH（§15）** | **评测** |
| `8b4166f` | eval：多轮用例的证据（修前红/修后绿/旧断言器会跳过） | 证据 |
| `245e9cb` | docs+eval：R5 §8.13/§8.14/§15 与基线换代 §10.1（+ `full-mt` 轨迹） | 文档 + 证据 |
| `1a604d3` | docs：R5 §14 补两个真实提交号 | 文档 |
| `a69fd5c` | **src：收紧代码沙箱与 SQL 守卫（§16）** | **生产** |
| `9aac5ea` | tests：沙箱 37 条 + SQL 守卫 +11 条 | 测试 |
| `855f8e4` | eval(tools)：守卫等价性审计 + 沙箱行为取证 | 评测工具 |
| `aeb514c` | eval(results)：收紧后的 L5 端到端轨迹 | 证据 |
| `09aa155` | docs：R5 §16 —— 沙箱与守卫的收紧（第三块） | 文档 |
| `289fb06` | eval(results)：§16.5 ⑤ 真实 SQL 过新守卫（smoke，22 条 SQL / 0 误拒） | 证据 |

> **§16 那两笔 docs 提交只留了 §16 正文的落点**（一笔是 §16 全文，一笔是 §16.5 ⑤ 的
> 补充）。它们的哈希不含在表内 —— 哈希要等提交之后才知道。要看它们：
> `git log --oneline -3 -- "docs/项目B_评测改动说明_R5.md"`。

> `.claude/settings.local.json`（本机权限白名单，含 `D:\` 绝对路径与 `PowerShell(*)`
> 通配）**未提交**，留在工作区。
> `scripts/langgraph_log.txt` 的删除是**别人** staged 的，一直在工作区未提交，
> 不属于本轮。`pyproject.toml` 的修改同理。

---

## 15. 多轮 / 跨会话用例（这次改动的第二块）

### 为什么非加不可 `[实测]`

原套件的 70 条**全是单轮的**，而且每条都拿全新的 store + 独立 user_id
（`eval-<case_id>`）。于是 `recommend.py` 里**读**方向那条分支——"store 里已经有这个
用户的偏好"——**一次都没被走到过**。§12 那个 D6 就长在这条分支里。

也就是说：**D6 不是"碰巧没被测到"，是单轮用例结构上测不到**——它需要"同一个用户的
第二次调用"。修完 §12 之后，用例层仍然没有任何东西能保证它不再回来。
这一节补的就是这个。

### 三条用例各自钉一个触发条件

| 用例 | 触发方式 | 除了"没崩"，还钉什么 |
|---|---|---|
| `M-001` | 同会话第二轮 | 记忆里的预算被真的用上（少了它结果变超集） |
| `M-002` | 新会话（同一用户） | 跨会话记忆：第二轮问句里没有任何预算数字 |
| `M-003` | 同会话第三轮 | 「冲突以最新用户消息为主」这条明文约定 |

「同会话第二轮」值得单独说：大多数人（包括我一开始）会以为跨会话 bug 只在"新会话"
出现。实际 `get_store_info` 是图的**入口**（`graph.py:34` 的 `add_edge(START, ...)`），
**每轮都重跑**并覆盖 state，所以同一个会话里发第二条消息就命中——用户不必等下次再来。
这条也是为什么 M-003 崩在第 2 轮而不是第 3 轮：第 2 轮就断了，第 3 轮压根没跑到。
**只看最后一轮的轨迹会漏掉它**，所以 `turn_errors` 按轮记、任一轮崩都判失败。

### 隔离粒度是"用例"，不是"轮"

这是实现里唯一需要想清楚的地方（也是设计文档 §2.3 早就写下的目标语义）：

```
一个用例  →  一份 store + 一份 checkpointer      ← 轮与轮之间延续
新会话    →  只换 thread_id，不换 store / user_id  ← 同一用户的第二次会话
```

另外三处细节，都是有意的：

1. **每轮只取本轮新增的消息**。同会话的后续轮次里 checkpointer 的 `messages` 是
   **累积**的；不切片就会把前面轮次的 SQL 一起算进 `sql_executed`，而
   `_judge_executed` 会"回头认前面任意一条"——于是**最后一轮答错**可能被前面某轮
   答对的 SQL 判成 PASS。`trace.turn_sql_counts` 把这件事变成可核对的数字
   （实测三条用例都是 `[1]`/`[1,1]`/`[1,1,1]`，即确实只算了自己那一轮）。
2. **步数/中断上限按轮算**。多轮会让总步数成倍增长，拿总步数当上限等于用轮数
   稀释保护阈值。
3. **崩了就不再往后跑**。后面的轮次会踩在一个残缺状态上，跑出来的东西既不是
   "模型答错了"也不是"产品对了"，没有解释价值。

### 给断言器加了 `GRAPH_CRASH`（本次唯一动到共享判定口径的地方）

**必须加的理由是实测出来的，不是推演**：用**旧**断言器跑修前的代码，那 3 条全被判成
`unverifiable`（存疑）然后 **skip**——因为旧的 `assert_case` 从头到尾没看
`trace.ok/error`，而崩在入口节点时 `user_intent` 与 `tool_calls` 都是空的，
正好落进"没执行任何 SQL"那段里的 `if not trace.user_intent and not trace.tool_calls`
分支。**没有这个新归因，这次新加的用例是没牙的**：报告里显示"存疑"而不是"失败"，
只把存疑比例从 0/70 抬到 3/73 ≈ 4%，连 conftest 的 20% 门槛都不会触发。
原始输出在 `eval/results/multi-turn-oldAsserter.jsonl`。

顺带发现 L5（refusal）更危险：崩了自然"没有执行危险 SQL"，按现有分支会判 **PASS**。
这次一并堵上（该分支在 `assert_case` 的最前面，先于所有判定）。

**影响面已核实为零**：`eval/results/` 下 10 份 jsonl 共 **301 条**用例记录里，
`trace.ok=False` 出现 **0 次**——所以没有任何一条历史用例的判定会因此改变。
核查命令（可复核）：

```powershell
# 逐份 jsonl 数 trace.ok=False，并打印它们原来的判定
# （脚本见 §9 的复核命令一节，输出：用例 301 条，ok=False 0 条）
```

`GRAPH_CRASH` 排在 `MODEL_ATTRIBUTIONS` 之外（要修的是代码，不是 Prompt），
与 `GUARD_BYPASS`/`ROUTED_AWAY` 同类处理。

### 证据：修前红、修后绿 `[实测]`

| 组 | 代码 | 断言器 | 结果 |
|---|---|---|---|
| A | 修前（`cff4d8f`） | 新 | **3 条全 FAIL**，归因 `GRAPH_CRASH`，`KeyError: 'budget_min'` |
| B | 修前（`cff4d8f`） | 旧 | **2 条全 skip**（存疑）—— 崩溃被静默吞掉 |
| C | 修后 | 新 | **3 条全 PASS** |

复现命令、原始输出、逐轮提取结果（证明预算确实来自 store）都在
[eval/results/multi_turn-before-fix.txt](eval/results/multi_turn-before-fix.txt)。
另外 `smoke-multiturn` 那轮确认 runner 重构对既有单轮路径**零影响**：
非 pass 名单与 `smoke-d6` 逐条相同（都只有 L3-001 / `BOUNDARY_OFF_BY_ONE`）。

**并入全量后（§10.1 的 `full-mt`，73 条）**：3 条全 PASS，整体 72/73 = 98.6%。
它们同时是**全套件最慢的 3 条**（M-003 26.5s / M-002 16s / M-001 16s，P95 是 14.3s）——
"值得守"和"该不该进每天跑的核心子集"是两件事，见 §8.14。

### 两条设计上的取舍（都会被读代码的人问）

1. **`tier` 仍写被断言那一轮的难度，没有新开一个 `Multi` 档**。tier 是**难度轴**，
   多轮是**形状轴**；混在一起会让分档完成率里多出一个语义不同的桶，还会得出
   "Multi 档只有 3 条"这种没有意义的统计。多轮与否看 `trace.n_turns`（0 = 单轮）。
2. **后续轮次的 `params` 刻意不写 `budget_max`**。params 是给中断追问作答用的，
   而预算正是这几条要考的东西——写进去等于把答案递过去：记忆失效时会触发一次追问，
   那一问一答就把预算补齐了，用例随即失去鉴别力。实测"城市"确实靠 params 补上
   （问句里没有城市），而预算**没有**被补上，所以"预算来自 store"这个结论是干净的。

   同理：**上限刻意避开"正好等于某套房源的 price"**。实跑里模型写的是
   `price < 1000` 而不是 `<= 1000`；只要没有房源价格正好等于上限，两种写法结果相同，
   用例就不会因为"模型选了哪个比较符"而随机红绿（否则它是一条伪装的
   `BOUNDARY_OFF_BY_ONE`）。M-002 的上限因此从 1050（正好是 id 18 的价格）改成 1000，
   黄金从 2 行变成 1 行——宁可黄金少一行，不要一条会随机红绿的用例。

### ⚠ 这轮顺带发现一条产品问题（未修，见 §8 第 13 条）

三条用例的第一轮都触发了「请提供**预算范围**」的追问，而用户其实已经说了预算上限
（"1000元以下"）。原因在 [recommend.py:130](src/agent/node/recommend.py#L130)：
`missing_info` 要求 `budget_min` **和** `budget_max` 都存在，只给了一侧就算缺失。
这不是我这轮要改的东西，但它解释了为什么中断几乎是普遍路径：`full-dec3` 70 条里
**62 条带 2 次中断**（首次预算追问 + 后续预订追问），问句里自带价格上限那 7 条
**7/7 全部被追问**——证据、复核命令、以及"`不提供` 会凭空补出 `budget_min=500`"
那条（疑似，未验证）都在 §8.13。

---

## 16. 沙箱与 SQL 守卫的收紧（这次改动的第三块）

### 16.0 为什么是这两处

它们是"**模型生成的代码**"和"**模型生成的 SQL**"的最后一道闸口——两条**输入完全不可信**
的路径：

| 闸口 | 位置 | 拦什么 |
|---|---|---|
| 代码沙箱 | [node/finance.py](src/agent/node/finance.py) | 租金计算节点把模型写的 Python **真的跑起来** |
| SQL 守卫 | [common/sql_guard.py](src/agent/common/sql_guard.py) | text2SQL 把模型写的 SQL **真的打到 MySQL 上** |

升级前**两个文件一条测试都没有**（`tests/` 下 grep `sql_guard` / `execute_code_sandbox`
零匹配）。所以这一节的写法和前面几节一样：**先取证，再改，改完把证据钉成测试和可复跑的命令。**

取证工具 [eval/tools/probe_sandbox_hardening.py](eval/tools/probe_sandbox_hardening.py)
把 `git show HEAD:src/agent/node/finance.py` 取出来当**独立模块**加载，
于是同一个探测脚本能同时喂给"旧"和"新"两版实现——差别只可能来自被测代码本身，
不来自探测代码：

```powershell
.\venv\Scripts\python.exe eval\tools\probe_sandbox_hardening.py          # 旧 = HEAD
.\venv\Scripts\python.exe eval\tools\probe_sandbox_hardening.py --old HEAD~1
```

（与 §16.5 那个审计工具同一个纪律：**提交前跑才有"前后"**，提交后它退化成
一份"当前实现会拒绝哪四类行为"的自检。）

### 16.1 代码沙箱：四类行为，改前改后 `[实测]`

| # | 探测（模型生成的代码里真这么写就能做到） | 升级前 | 升级后 |
|---|---|---|---|
| E | `import os` 读环境变量 | `env_visible=True`；`cwd=D:\kindsOfProject\rent-agent`（**仓库根**） | `rejected=True` · `banned_import:os`，**解释器根本没启动** |
| F | `subprocess.run(...)` 起子进程 + `open('.tmp_pwned.txt','w')` 往仓库写文件 | `subprocess_ok=True` `write_ok=True` | `rejected=True` · `banned_import:subprocess` |
| G | `while True: pass`（死循环） | 超时被杀 ✅（**这条原来就是好的**） | 同左（保留） |
| H | `while True: print('x'*100)`（输出洪水） | `stdout_len=0`——输出全丢，5 秒在烧内存 | `stdout_len=200000`，封顶保留 + `truncated=True`，并被报成超时 |

E 这条的严重度得说清楚：**`.env` 里的 API key 就在子进程的环境变量里**，
而 `cwd` 是仓库根——一次 `open('src/agent/common/llm.py','w')` 就能改生产代码。
升级前的实现是 `subprocess.run(code, capture_output=True, timeout=...)`，
只做了一件事：超时。G 之所以"原来就是好的"，是因为那是它唯一做了的事。

### 16.2 三层，各自的理由

**① 静态白名单** [`check_code_safety`](src/agent/node/finance.py#L97)：AST 走两遍，
模型不可信，所以在**起进程之前**就判。它是个纯函数（不吃环境、不起进程），
所以能逐条列规则做单测。四类规则：

- `banned_node:` Global / Nonlocal / ClassDef / AsyncFuncDef / Await / AsyncFor / AsyncWith / Yield / YieldFrom / Delete
- `banned_import:` 只允许 `math statistics decimal fractions itertools functools collections datetime json`
- `banned_name:` `eval exec compile open input __builtins__ ...`（含 dunder 名）
- `banned_call:` 调用既不在内置白名单、也不是本文件自己定义的函数
- 另加 `Attribute.attr` 以下划线开头 → `private_attr:`

**为什么模块白名单放得比较宽（`math/decimal/statistics`…）**：prompt 明文要求"计算金额保留
2 位小数"，`Decimal` 是最自然的写法。白名单收得过紧的代价不是"更安全"，而是
`fix_code` 反复重试、最后把一条本该答对的题答成答不出——**安全层和用例通过率是同一个
预算的两端**，所以放行清单要按"这个节点真的需要算什么"来定，不是按"最小权限"来定。

**② 进程收紧**：`sys.executable`（不依赖 PATH 上的 `python`）+ `-I -B -X utf8`
（隔离模式：忽略 `PYTHON*` 环境变量、不把 cwd 加进 `sys.path`、不写 `.pyc`），
`cwd` 给一次性临时目录（**代码里的相对路径不再落到仓库**），
环境变量**整个不继承**（`_sandbox_env` 只给 `PATH`/`TEMP`/`TMP`）。

> `_sandbox_env` 里**刻意没有**写 `PYTHONPATH=…` 之类的硬化项：`-I` 已经隐含 `-E`，
> 写了也是惰性的。这种"写了不生效"的假硬化比不写更坏——下一个人会以为它在起作用。

**③ 资源收紧**：超时杀进程（保留超时前输出）、输出**读干 + 截断**（两个线程各自
drain 到 200k 字符后继续读但丢弃——**只截断不读干，子进程会卡在写管道上**，
最后被误报成"超时"）、临时目录 `finally` 删。

### 16.3 我自己在实现里踩出来的两个坑（都留了测试）

1. **白名单把"调用自己定义的函数"也拦了**：`def area(w, h): ...; print(area(3,4))`
   → `banned_call:area`。修法是两遍走：第一遍先收集本文件 `FunctionDef` 的名字与
   赋值目标（Store 上下文），第二遍才判 `Call`。顺带确认"改名绕过黑名单"仍然堵着——
   `x = eval; x()` 里的 `eval` 是 Name 节点，照样命中 `banned_name:eval`（有测试钉住）。
2. **输出为 0**：第一版重写后探针 H 报 `stdout_len=0`，看起来像"截断生效"，
   实际是**子进程被 kill 时缓冲里的东西还没进管道**。这不是这次要修的问题
   （真实用例里模型很少写 `flush=True`），但测试里得写清楚，否则
   `test_timeout_keeps_partial_output` 会变成一条随机红绿的测试。

### 16.4 SQL 守卫：掩码扫描

原来的实现是**先剥注释、再对整串原文做正则**。于是**字符串字面量里的内容会参与判定**——
四种失效模式（两种误拒、一种篡改、一种静默改语义）：

| 失效模式 | 输入 | 旧版行为 |
|---|---|---|
| 误拒 | `... WHERE description LIKE '%;%'` | `allowed=False` · `multi_statement`——分号在**字面量里** |
| **篡改** | `... LIKE '%LIMIT 1000%'` | `allowed=True`，但 SQL 被**改写成** `... LIKE '%LIMIT 50%'` |
| 误拒 | `... LIKE '%DROP%'` | `allowed=False` · `deny_keyword:DROP` |
| 静默改语义 | `... WHERE id IN (SELECT id FROM house LIMIT 999)` | 子查询的 `LIMIT 999` 被当成顶层，**收紧成 50** |

第二条最难看：它不改判定、只改 SQL，**没有任何日志或原因会暴露它**——
用户问"描述里含 LIMIT 1000 的房源"，拿回的是另一批房子。

**修法**：`_mask()` 把**字符串字面量**与注释的内部字符替换成空格、**长度不变**，
所有关键字/表名/分号/顶层 LIMIT 的判定都在掩码串上做；只有 LIMIT 的**插入位置**
仍用原文的偏移。等长是关键——掩码串的下标和原文一一对应，`_enforce_limit` 才能
同时拿到"判定用串"和"改写用原文"。

**两处刻意不掩码**（都写在 `_mask` 的 docstring 里，因为都是踩出来的）：

- **反引号**：它是**标识符**不是字面量。掩掉内容会让 `SELECT * FROM \`users\`` 里的表名
  消失，**表黑名单被绕过**——这是我在实现过程中真的踩出来的漏判（`FROM \`users\``
  一度变成放行），不是推演。修完 `` FROM `users` `` 重新返回 `deny_table:users`。
- **`#` 行注释**：`#` 出现在反引号标识符里会把本行剩下的一切掩成空格 → **fail-open**。
  本项目 SQL 里从来没出现过 `#` 注释（336 条历史轨迹 0 次），所以干脆不认它。
  代价是"用 `#` 当注释的 SQL"可能被误拒——**方向是安全的那一侧**。

### 16.5 证据 `[实测]`

**① 等价性审计（新增工具）** [eval/tools/audit_guard_equiv.py](eval/tools/audit_guard_equiv.py)：
取 `HEAD` 版守卫当独立模块，把**历史轨迹里真正过过守卫的每一条 SQL**（`sql_executed`
+ `final_sql` + 全部黄金 SQL）重放给新旧两版，逐条比 `allowed / rule / 改写后的 SQL`。
零 LLM 成本、不连数据库：

```powershell
.\venv\Scripts\python.exe eval\tools\audit_guard_equiv.py   # 必须在**提交前**跑，见下
```

```
— 内置样例（本次要修的四种失效模式）—
  ≠ 分号在字面量里（误拒 · multi_statement）
  ≠ LIMIT 在字面量里（篡改 · 把 '%LIMIT 1000%' 改成 '%LIMIT 50%'）
  ≠ 写关键字在字面量里（误拒 · deny_keyword:DROP）
  ≠ 子查询 LIMIT 被当顶层收紧（静默改语义）

— 真实数据重放 —
重放 SQL 162 条（已去重）
  结论相同 162 条
  变宽松 0 条   变严格 0 条   其它 0 条
```

四条样例全部 `≠` 是"**确实改了**"，162 条真实 SQL 全部相同是"**没有连带伤**"——
这两句话要一起说才有意义。**所以这轮没有重跑全量评测**：守卫的输入集合在真实数据上
判决等价，重跑只会得到一份新的随机失败名单，不会得到新信息。

> 该工具的 docstring 里写死了一条纪律：它比的是"改前 vs 改后"，**提交之后再跑就是
> 拿同一版比自己，输出必然是 0**。工具会从 git ref 取旧版，所以提交后它只会变成
> 一个恒真检查——放进 CI 之前必须知道这件事。

**② 单测**：`test_sql_guard.py` **31 → 42 条**（新增 11 条，全是上面那四种失效模式的
回归钉子，含反引号绕过那两条）+ `test_guard_wiring.py` 7 条（接线加在 `sql_db_query`
之外，与守卫本身解耦）；`test_code_sandbox.py` **新增 37 条**（放行 8 / 拦截 19 /
进程接线 2 / 资源与端到端 8）。全量：

```powershell
.\venv\Scripts\python.exe -m pytest tests\unit_tests -q     # 97 passed
```

**③ 真实模型跑一遍**（静态白名单是新增闸口，必须确认**模型自己写的代码不会被误杀**——
单测只能用手写样例）：

```powershell
.\venv\Scripts\python.exe .tmp_finance_e2e.py               # 临时脚本，已删
# 复核时等价的做法：把两句问话喂给 finance_graph.invoke()，看 execution_result
```

```
深圳三套房子月租 1200/1850/2300，押一付三共需多少？
  → exit=0 rejected=False 重试 0，stdout='三套房子月租合计：5350 元\n押一付三共需准备：21400.00 元'
租金 2500 元，年涨幅 3%，两年后月租？（保留 2 位小数）
  → exit=0 rejected=False 重试 0，stdout='2652.25'
```

两条都 `rejected=False`、答案正确、**一次重试都没触发**——白名单没有误伤。

**④ L5（拒答档）端到端**：`eval/results/sandbox-guard-l5.jsonl`，8 条 **全 PASS**。

> ⚠ **但这 8 条没有一条证明"拒绝路径是通的"**，如实记账：`guard_fired_times: 0`、
> `model_declined_itself: true` 出现在 **8/8** 条里——模型在生成 SQL 之前就自己拒了，
> 守卫一次都没被执行到。所以"守卫接线在真实图里确实生效"的证据是
> [tests/unit_tests/test_guard_wiring.py](tests/unit_tests/test_guard_wiring.py)
> （把 `validate_sql` 换成桩，看它是否被调用、拒绝后是否不再执行），不是这一跑。
> 这一跑能证明的只有一件事：**收紧之后 L5 档的 8 条全部照旧通过**。
> 同类问题 §8.14 已经写过一次（"值得守"和"测到了"是两件事），这里再记一次。

**⑤ 真实 SQL 过新守卫（补 ④ 那个缺口）**：上面两跑都不足以说明"新的守卫在真实路径上
能用"——L5 那 8 条 **一条 SQL 都没执行**。所以另跑了一轮 smoke（20 条）：

```powershell
.\venv\Scripts\python.exe -m pytest eval -m smoke --run-id smoke-guard -q   # 149s
```

```
用例 20 条，判定 {'pass': 19, 'fail': 1}
真正执行过的 SQL 共 22 条 → 全部经过新的 validate_sql；守卫拒绝 0 次
非 pass： [('L3-001', 'fail', 'BOUNDARY_OFF_BY_ONE')]
```

两件事各自成立：**22 条真实 SQL 走完了新代码**（掩码、`_top_level_limit_spans`、
`_trim_trailing_comments` 都真的被调用过），且**没有一条合法 SQL 被误拒**；
非 pass 名单与 `smoke-d6` 基线逐条相同（只有 L3-001，归因也是既有那个）。
证据 `eval/results/smoke-guard.jsonl`。

> 仍未覆盖的：**真实 SQL 触发"拒绝"那条分支**（这轮守卫拒绝 0 次）。原因和 ④ 一样——
> 模型自己就把危险 SQL 挡在生成阶段了。所以"拒绝后不再执行"这条断言，
> 证据仍然只有接线单测。

### 16.6 明确没做 / 残留缺口

- **没上容器、不装 Docker**。作过比较：Windows 11 **家庭版**没有 Windows Sandbox
  组件，容器路线要 Docker Desktop + WSL2 + **重启**，把一个"给生成的代码加一道闸口"的
  需求变成项目级依赖（每个人的机器都要装）。当前三层挡住的是"读秘密 / 起进程 /
  写文件 / 打死沙箱"，**代价是零依赖**。
- **没有内存 / CPU 配额**。`resource` 是 POSIX-only，Windows 上得走 Job Object
  （`pywin32`，新依赖）。现在只有"墙钟超时"，所以**一条 `x = [0]*10**9` 能把内存吃满**——
  这是本节最实的残留缺口，写在这里而不是含糊过去。
- **没有容器级文件系统隔离**：临时目录只是"干净的 cwd"，不是"看不见的根"。
  配合"环境变量不继承"，能走的路已经很少，但**不是内核级隔离**。
- **没引入 `sqlglot`**。用真正的 SQL 解析器是更正确的路线，但它是新依赖；
  掩码方案零依赖且"162 条等价"已经量过。真要做，`audit_guard_equiv.py` 现成能用
  来量"换成解析器之后判决变了多少条"。
- **没动生成侧**（prompt / 模型选择）。这一节只收紧执行侧——按 §3.2 的说法，
  prompt 是请求，执行侧白名单是唯一能兜住的。

### 16.7 复核命令（本节自包含）

```powershell
# 守卫：改动前后的判决对照（**只能在提交前跑**，见 16.5）
.\venv\Scripts\python.exe eval\tools\audit_guard_equiv.py

# 沙箱：同一个探测脚本喂给 HEAD 版与工作区版（提交前跑）
.\venv\Scripts\python.exe eval\tools\probe_sandbox_hardening.py
#   预期：E/F 旧版 `env_visible=True`/`write_ok=True` → 新版 `rejected=True`；
#        H 旧版 stdout_len=0 → 新版 200000

# 单测
.\venv\Scripts\python.exe -m pytest tests\unit_tests -q            # 97 passed
.\venv\Scripts\python.exe -m pytest tests\unit_tests\test_code_sandbox.py tests\unit_tests\test_sql_guard.py tests\unit_tests\test_guard_wiring.py -q   # 86 passed

# L5 端到端（会真的调模型；8 条，`-k L5` 选出来正好是那 8 条）
.\venv\Scripts\python.exe -m pytest eval -k "L5" --run-id sandbox-guard-l5 -q
```


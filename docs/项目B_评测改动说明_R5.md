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

**没动**：`src/` 下**任何**文件（生产代码零改动）、`eval/cases/` 里除 L4-006 外的用例、
`eval/conftest.py`、种子数据（`house` 表）。

---

## 8. 还没做的 / 需要你定的

1. **改 `UserMessage.description`（§6）** —— 唯一还没落地的实质改动，改的是生产代码，等你点头。
2. **重跑一次全量基线** —— 现在这个 56 是离线重判的。断言器和黄金都变了，
   必须重跑才能得到可对外引用的数字。建议**先改完 §6 再跑**，否则要跑两次。
3. **`house` 表的词汇规范化（你提的「装修好类似意思的都转成豪华」）** —— **没做**。
   盘点下来只有一处真需要动：id 45 写的是「近4号线，**精装**三房」，而其余 5 套用的是
   「**精装修**」。改成「精装修三房」后 `%精装修%` 从 5 套变 6 套，**但所有黄金的结果都不变**
   （L3-010 仍 6 套、L4-006 仍 8 套）。
   没动的理由：种子数据**不在仓库里**（只有 MySQL 里有），改它是改共享库、且不留版本记录，
   而它对当前判分的影响是零。**要改的话说一声**，一条 UPDATE 就行。
4. **19 条 holdout 名单**（实施计划 §2.3）—— 仍等你签字。
5. **`recommend.py` 跨会话 `KeyError`**（R4 的 D6）—— 未修。

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

# 5) 断言器单元测试
.\venv\Scripts\python.exe -m pytest tests/unit_tests -q            # 39 passed

# 6) 边界体检（口径是否自相矛盾）
.\venv\Scripts\python.exe eval\tools\audit_boundary.py
```

> 读 jsonl 请务必带 `encoding="utf-8"`。Windows PowerShell 的 `Get-Content`
> 会把 UTF-8 当 ANSI 读，中文全乱码，`ConvertFrom-Json` 也会失败 —— 这不是文件坏了。
>
> 另外：**PowerShell 里别用 `-c` 传含引号的 SQL**，解析器会把它当命令。写临时脚本文件跑。

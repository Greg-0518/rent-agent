# Roomie Agent 代码知识图谱

- **版本**：v1.0 ｜ 2026-09-13（基于源码逐文件实读生成，非口述）
- **源码位置**：`upload/rent-agent-extracted/agent - 副本/`（本副本内 import 路径为 `src.agent.*`，对应原工程 `src/agent/` 目录）
- **用途**：三层导航——L1 看懂全局架构 → L2 吃透模块流程 → L3 秒查"面试官问到 X，代码在哪一行"
- **配套**：`interview_prep/Roomie面试题库_源码深挖.md`（深挖题）、`interview_prep/项目B_评测Harness/`（评测设计）
- **执行侧闸门的原理**：[代码沙箱与SQL守卫_原理说明.md](代码沙箱与SQL守卫_原理说明.md)（代码沙箱三层 + SQL 守卫掩码扫描，含威胁模型与残留缺口）

---

## 目录

- **第一部分 L1：项目一页纸**（是什么 / 技术栈 / 一图流架构）
- **第二部分 L1：主图流程**（意图路由 7 分支）
- **第三部分 L2：六大模块地图**（每个模块：职责 → 流程 → 关键技术 → 面试考点）
  - 3.1 意图路由与用户记忆（主图大脑）
  - 3.2 房源推荐子图（text-to-SQL 循环）
  - 3.3 合同审核子图（RAG 四段流水线）
  - 3.4 租金计算子图（代码生成-执行-修复循环）
  - 3.5 预定子图（人机协作信息收集）
  - 3.6 多模态看房（Qwen-VL）
- **第四部分 L2：状态设计**（State / ContextSchema / Store 三层持久化）
- **第五部分 L3：代码定位索引**（文件 → 函数 → 行号 → 一句话职责）
- **第六部分 L2：五个可深挖机制**（interrupt / 降级链 / 强制工具调用 / 延迟初始化 / 重试控制）
- **第七部分：已知问题与改进方向**（损伤标注 + 与项目 B 的衔接）

---

# 第一部分 L1：项目一页纸

**一句话**：基于 LangGraph 的租房 Multi-Agent 助手——主图做意图路由，七个业务分支各为独立子图，覆盖"找房（text-to-SQL）→ 看房（多模态）→ 算租金（代码沙箱）→ 审合同（法律 RAG）→ 下订单（人机协作）"完整闭环。

**技术栈**：

| 层 | 选型 |
|---|---|
| 编排 | LangGraph（StateGraph / 条件边 / interrupt / ToolNode / BaseStore） |
| LLM 接入 | LangChain `init_chat_model` → DeepSeek（chat / reasoner 双模型，temperature=0） |
| 结构化输出 | pydantic schema + `with_structured_output`（意图识别、需求提取、条款提取、风险分析四处） |
| 数据库 | MySQL（pymysql）+ LangChain SQLDatabaseToolkit（sql_db_schema / sql_db_query 工具） |
| RAG | BM25(jieba) + FAISS(bge-small-zh-v1.5) 混合检索 0.4/0.6 → bge-reranker-v2-m3 重排 → LLM 查询改写 |
| 多模态 | Qwen-VL-Max（DashScope 兼容模式），降级 DeepSeek |
| 代码沙箱 | tempfile 隔离 + subprocess 执行，timeout=30s，失败自动修复重试 ≤3 次 |
| 持久化 | LangGraph Store（namespace=(user_id, "preferences")）跨会话用户画像 |

---

# 第二部分 L1：主图流程

入口 `graph.py`（97 行，全项目最短但最重要的文件）：

```
START
  └→ get_store_info        查 Store 取用户偏好（node/main.py:39）
  └→ identify_question     LLM 结构化输出判意图，7 类（node/main.py:23）
  └→ router_message 条件路由（graph.py:38-56）
       ├─ recommend_house ──→ [recommended_graph 子图] ──→ need_resume ①
       ├─ reserve_house ────→ [reserve_graph 子图] ──→ END
       ├─ contract_audit ───→ set_contract_text ──→ [contract_graph 子图] ──→ END
       ├─ image_analysis ───→ image_analysis 节点 ──→ END
       ├─ rent_calc ────────→ [finance_graph 子图] ──→ END
       ├─ get_info ─────────→ get_user_preferences ──→ END
       └─ others ───────────→ [extend_graph 兜底闲聊] ──→ END

① recommended_graph 完成后进入 need_reserve（interrupt 问"是否预定"）：
   答"需要" → reserve_graph；否则 END（graph.py:68,71-83）
```

**面试一句话**：这不是单 Agent 加一堆工具，而是 Supervisor 模式——主图只做"识别+路由"，业务逻辑全部下沉子图，子图之间通过共享 State 主图字段（user_preferences / contract_text）传递数据。

---

# 第三部分 L2：六大模块地图

## 3.1 意图路由与用户记忆（主图大脑）

- **流程**：进图先查 Store（跨会话画像）→ LLM 对**最后一条用户消息**做 7 类意图分类（pydantic `UserMessage` 强约束）→ 条件边分发
- **关键技术**：`with_structured_output` 让分类结果必须是枚举值，不可能出 schema 外的意图；`filter_messages(include_types="human")` 只取用户侧消息
- **面试考点**：为什么意图识别只用最后一条消息而不带全历史（省 token、避免历史意图污染）；为什么用结构化输出而不用 function calling 分类（输出 schema 是硬约束）

## 3.2 房源推荐子图（text-to-SQL 循环）——全项目技术含量最高

- **流程**（`recommend.py` + `node/recommend.py`）：
  1. `collect_user_info`：pydantic `UserInfo`（8 字段）提取需求 → 缺城市/预算时 `interrupt` 追问 → 用户答"不提供"则填默认值（随机城市/500-3000 元/6 套）→ **预算写回 Store（只扩不缩合并）** → 拼规范化查询消息
  2. `list_tables` → `call_get_schema`（`tool_choice="any"` 强制调 sql_db_schema）→ `get_schema`（ToolNode）
  3. `generate_query`：text-to-SQL prompt（注入 dialect、top_k=room_count、**禁 DML**）→ `check_query`（SQL 专家 checklist 复查改写）→ `run_query`（ToolNode 执行，执行前**必过安全层** `common/sql_guard.validate_sql`：只读白名单 + 强制 LIMIT + 敏感表黑名单）
  4. **执行结果回灌 generate_query 循环**，直到 LLM 不再发起 tool call（`should_continue`）→ END
- **关键技术**：SQLDatabaseToolkit 工具集；DB 连不上时 `_DB_AVAILABLE=False` 全节点降级返回错误提示（node/recommend.py:249-274）；check_query 复用原 message id 保持消息链一致（:341）
- **面试考点**：为什么"生成→检查→执行→回灌"循环而不是一次生成（自修复）；check_query 检查哪些 SQL 常见错误（NULL NOT IN / UNION vs UNION ALL / BETWEEN 边界 / 类型不匹配）；为什么禁 DML 要写进 prompt **而且**要再加一层执行侧白名单（prompt 只是请求，模型可以不听；执行侧白名单是唯一能兜住的——见 `common/sql_guard.py`；**掩码扫描的原理与四种被修掉的失效模式见 [代码沙箱与SQL守卫_原理说明.md](代码沙箱与SQL守卫_原理说明.md) §6**）

## 3.3 合同审核子图（RAG 四段流水线）

- **流程**（`contract.py` + `node/contract.py`）：`clause_extraction`（9 类条款结构化提取，空白条款标 is_blank）→ `law_retrieval`（逐条款检索法条）→ `risk_analysis`（逐条款分析：checklist_map 按条款类型注入审查清单 + **强制引用法条编号**）→ `report_generation`（五段式报告 + 空白核心条款"高-合同不成立风险"警告）
- **关键技术**：`make_law_retrieval_node(get_retriever)` **工厂函数延迟初始化**——图编译不触发模型下载/索引构建，首次执行才加载；检索器三级降级（FAISS 挂→纯 BM25；reranker 挂→跳过重排；HF 不可达→纯 BM25）；索引持久化带 docs hash 变更检测（common/retriever.py:170-181）
- **面试考点**：混合检索为什么 BM25+向量加权融合（术语精确匹配+语义泛化，0.4/0.6）；为什么法律场景必须有 reranker（召回≠排序，CrossEncoder 精排）；查询改写的作用（口语→法律术语，rewriter prompt 在 retriever.py:118-131）

## 3.4 租金计算子图（代码生成-执行-修复循环）

- **流程**（`finance.py` + `node/finance.py`）：`generate_code`（```python 块提取，无块则按行前缀兜底提取）→ `execute_code`（**静态白名单** `check_code_safety` → 一次性临时目录写盘 + `sys.executable -I -B -X utf8` 子进程跑，产出 ExecutionResult{stdout/stderr/exit_code/execution_time/timed_out/rejected/truncated}）→ `should_retry`：成功→`generate_answer`；失败且 retry_count<3→`correct_error`（LLM 拿 stderr 修代码）→ 回执行；3 次仍败→give_up END
- **关键技术**：错误信息直接回灌 LLM（stderr 是最好的修复上下文）；重试计数放 State 里由条件边读取；沙箱三层——AST 白名单（禁 import 白名单外的模块/魔术属性/`open|eval|exec|input`）、进程收紧（`-I` 隔离、cwd=临时目录、**env 不继承父进程**）、资源收紧（超时杀进程、输出截断到 200k 字符）
- **面试考点**：为什么 LLM 算租金要生成代码而不是直接算（数值计算要确定性，LLM 心算会错）；沙箱还剩什么缺口（**没有内存/CPU 配额**——Windows 上没有 stdlib 的 rlimit，也没有容器级文件系统隔离；真隔离要上容器）；**三层各自的原理、AST 为什么不能用正则代替、白名单为什么故意放宽见 [代码沙箱与SQL守卫_原理说明.md](代码沙箱与SQL守卫_原理说明.md) §1–§5、§7**

## 3.5 预定子图（人机协作信息收集）

- **流程**（`reserve.py` + `node/reserve.py`）：`get_title` → `get_phone`（11 位/纯数字/1 开头/3-9 号段校验）→ `get_id`（18 位 + **ISO 7064 MOD 11-2 校验码验证**，node/reserve.py:73-107）→ `add_reserve_message`（拼单）→ `call_orders`（bind generate_orders 工具）→ ToolNode 执行
- **关键技术**：三个信息节点都是"state 已有值→跳过 / 从消息提取→失败则 interrupt 追问"三段式；`generate_orders` 是 `@tool` + `InjectedStore`——工具内部直接写 Store（生成 uuid 工单号，追加 reserved_info 列表）
- **面试考点**：为什么用 interrupt 而不是让 LLM 多轮对话收集（信息收集要确定性+可校验，不能靠模型自觉）；手机号/身份证为什么代码校验而不是 LLM 校验（规则确定的事不进模型）

## 3.6 多模态看房（Qwen-VL）

- **流程**（`node/vision.py`）：`_extract_image`（消息里挖图片：多模态 content 块 / URL 正则 / 本地路径三种来源）→ base64 编码 → `qwen-vl-max`（DashScope OpenAI 兼容模式；无 key 则降级 DeepSeek）七维分析（房间类型/装修/家具/采光/面积/评价/租金估算）→ `issue_detection_node` 隐患检测 → `report_generation_node` 看房报告（逐房间打分/隐患清单/性价比）
- **面试考点**：视觉模型怎么选（qwen-vl-max 国内可达+中文场景强）；图片输入为什么 base64 而非 URL（本地文件无公网地址）

---

# 第四部分 L2：状态设计（三层持久化）

| 层 | 载体 | 内容 | 生命周期 |
|---|---|---|---|
| 会话内 | 各子图 State（均继承 `MessagesState`，消息自动 reducer 合并） | 主图 State：user_intent / user_preferences / contract_text / audit_report；子图各自扩展字段 | 单次 run |
| 运行时注入 | `ContextSchema`（common/content.py） | user_id | 每次 invoke 传入 |
| 跨会话 | LangGraph `BaseStore` | namespace=(user_id, "preferences")：预算上下限 + 预订工单列表（common/store.py pydantic 模型） | 永久 |

**数据流转细节**（面试深挖点）：
- 主图 State 的 user_preferences 由 `get_store_info` 注入，子图通过共享字段读取（LangGraph 子图 state 映射）
- 预算合并是**只扩不缩**：新预算比存的下限更低才更新 min、比上限更高才更新 max（node/recommend.py:200-223）
- 订单写入用 `prefs.setdefault('reserved_info', []).append(...)`（node/reserve.py:189）

---

# 第五部分 L3：代码定位索引

> 用法：面试官问"你的 X 是怎么做的"→ 查表 → 直接翻到文件行号开讲。

### 主图与路由

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| 图的构建/全部节点注册 | graph.py:21-32 | 11 个节点 add_node |
| 意图路由函数（7 分支） | graph.py:38-56 | user_intent → 子图名 |
| 推荐→预定中断衔接 | graph.py:68,71-83 | should_resume 条件边 |
| 7 类意图 schema | node/main.py:14-19 | UserMessage(Literal 枚举) |
| 意图识别节点 | node/main.py:23-35 | 只用最后一条 human 消息 |
| Store 读画像 | node/main.py:39-50 | namespace=(user_id,"preferences") |
| interrupt 问是否预定 | node/main.py:54-60 | "需要/不需要" |
| 偏好查询回复 | node/main.py:64-92 | 禁编造 prompt |
| 合同文本提取 | node/main.py:96-99 | filter human 消息 |

### 房源推荐（text-to-SQL）

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| 子图编排（循环结构） | recommend.py:16-50 | generate→check→run→回 generate |
| 需求提取 8 字段 schema | node/recommend.py:21-55 | UserInfo(pydantic) |
| 缺失追问+默认值 | node/recommend.py:120-141 | interrupt + "不提供"分支 |
| 预算只扩不缩持久化 | node/recommend.py:170-223 | Store 合并逻辑 |
| DB 连接与降级 | node/recommend.py:237-274 | SQLDatabaseToolkit + _DB_AVAILABLE |
| text-to-SQL 主 prompt（禁 DML） | node/recommend.py:300-319 | dialect/top_k 注入 |
| SQL 质检 checklist | node/recommend.py:321-342 | 8 类常见错误复查 |
| **执行侧安全层**（只读白名单/强制 LIMIT/表黑名单，**掩码扫描**） | common/sql_guard.py:280-336 | validate_sql（109-172 是 `_mask`） |
| 安全层单测（42 条 + 接线 7 条） | tests/unit_tests/test_sql_guard.py + test_guard_wiring.py | 接线加在了 sql_db_query 之外 |
| 保留推荐参数模板 | state/recommend.py:19-40 | get_recommend_info |

### 合同审核（RAG）

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| 混合检索+重排+降级全逻辑 | common/retriever.py:186-314 | build_law_retriever |
| LLM 查询改写（口语→法言法语） | common/retriever.py:104-134 | RewrittenRetriever |
| 分块策略（按"第X条"切） | common/retriever.py:246-250 | RecursiveCharacterTextSplitter |
| PDF 清洗（去页眉/URL/页码） | common/retriever.py:39-71 | _clean_page_text |
| 索引 hash 变更检测 | common/retriever.py:170-181 | 增量重建依据 |
| 检索器延迟初始化工厂 | contract.py:19-26 + node/contract.py:254-276 | 编译期不加载模型 |
| 9 类条款提取 prompt | node/contract.py:74-94 | 空白标 is_blank |
| 逐条款法条检索 | node/contract.py:266-274 | clause_type+content 做查询 |
| 类型化审查清单 | node/contract.py:162-174 | checklist_map 6 类 |
| 风险分析主 prompt（强制法条号） | node/contract.py:176-197 | 民法典 712/722/497/585 |
| 报告生成（空白条款高危警告） | node/contract.py:301-372 | 五段式结构 |
| Clause/RiskAnalysis 数据结构 | state/contract.py:9-25 | TypedDict |

### 租金计算（沙箱）

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| 代码生成（块提取+兜底） | node/finance.py:15-36 | generate_code |
| **沙箱静态白名单**（AST：模块/属性/名字/调用四类规则） | node/finance.py:42-160 | check_code_safety |
| **沙箱执行**（临时目录 + `-I -B -X utf8` + 最小 env + 超时/输出上限） | node/finance.py:162-308 | execute_code_sandbox |
| LLM 修复代码 | node/finance.py:311-317 | fix_code |
| 重试控制（≤3 次） | node/finance.py:365-372 | should_retry |
| ExecutionResult/FinanceState | state/finance.py:9-47 | 七字段执行结果（含 rejected/truncated） |
| 沙箱单测（37 条） | tests/unit_tests/test_code_sandbox.py | 放行/拦截/进程接线/超时截断 |

### 预定

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| title/phone/id 三段收集 | node/reserve.py:13-127 | state 优先→消息提取→interrupt |
| 手机号校验 | node/reserve.py:34-51 | 11位/数字/1开头/3-9号段 |
| 身份证校验码算法 | node/reserve.py:73-107 | ISO 7064 MOD 11-2 |
| 工单工具（写 Store） | node/reserve.py:144-197 | generate_orders @tool+InjectedStore |
| 工具循环 | reserve.py:27-35 | tools_condition 条件边 |

### 多模态

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| VL 模型选择与降级 | node/vision.py:15-22 | qwen-vl-max→DeepSeek |
| 图片来源三种挖掘 | node/vision.py:41-56 | content块/URL正则/本地路径 |
| 七维分析 prompt | node/vision.py:31-39 | 含租金估算 |
| 隐患检测+看房报告 | node/vision.py:71-138 | issue/report 两节点 |

### 公共层

| 要讲的东西 | 位置 | 一句话 |
|---|---|---|
| 双模型接入（chat/reasoner） | common/llm.py:4-14 | temperature=0 全局 |
| 用户画像模型 | common/store.py:6-48 | ReservedInfo/UserPreferences |
| 会话上下文 | common/content.py:4-6 | ContextSchema(user_id) |
| 兜底闲聊子图 | extend.py:7-22 | 单节点 MessagesState |

---

# 第六部分 L2：五个可深挖机制（面试弹药）

1. **interrupt 人机协作（4 处）**：collect_user_info（缺参数）、need_reserve（是否预定）、get_title/get_phone/get_id（信息收集）。共同模式：**state 已有值先跳过 → 从最新消息提取 → 失败才 interrupt**。深挖点：interrupt 后 resume 如何恢复现场（LangGraph checkpoint；所以代码里到处 print ENTER state——调试中断恢复的痕迹，node/recommend.py:101-108、node/reserve.py:14-31）
2. **检索器三级降级链**：embedding 挂→纯 BM25；reranker 挂→跳过精排；整库空→兜底文档检索器。深挖点：为什么降级而不是报错（可用性优先，法律问答 BM25 也能答个七八成）
3. **强制工具调用**：call_get_schema 和 check_query 都用 `tool_choice="any"`。深挖点：什么场景必须强制（确定下一步就是工具，不让模型自由发挥跑偏）
4. **延迟初始化**：make_law_retrieval_node 工厂把重资源（HF 模型下载/索引构建）推迟到首次节点执行。深挖点：图编译期和执行期的职责分离
5. **循环终止条件**：text-to-SQL 靠"LLM 不再发 tool call"自然终止；finance 靠 retry_count 硬上限 3。深挖点：两种终止策略的适用场景（对话型 vs 任务型）

---

# 第七部分：已知问题与改进方向

### 7.1 本副本的三处代码损伤（疑似导出/复制丢失，非逻辑 bug）

以下三处 `.tool_calls` 相关代码文本不完整，是同一模式（属性名/左值丢失），原始工程大概率是完好的，**面试演示前务必用原工程跑一遍确认**：

- `recommend.py:36`：`if not last_messages.:` —— 应为 `if not last_messages.tool_calls:`（循环终止判断）
- `node/recommend.py:280-286`：`list_tables` 内手工构造 tool call 的字典和 AIMessage 参数名丢失 —— 应为 `tool_call = {"name": "sql_db_list_tables", ...}` 与 `AIMessage(content="", tool_calls=[...])`
- `node/recommend.py:337-338`：` = state["messages"][-1].[0]` / `HumanMessage(content=["args"]["query"])` —— 应为 `query = state["messages"][-1].tool_calls[0]["args"]["query"]` 与 `HumanMessage(content=query["args"]["query"])`

### 7.2 与项目 B（评测 Harness）的衔接点

| 本项目现状 | 项目 B 对应改造 |
|---|---|
| text-to-SQL 禁 DML 只在 prompt 里 | 执行侧白名单（SELECT 开头校验+强制 LIMIT）→ 评测 L5 档成立前提 |
| 沙箱仅 timeout=30 | 资源限制（项目 B M0） |
| check_query 质检 8 类错误 | 评测归因四分类可直接对齐 |
| 无任何评测 | 50-100 条用例回归（设计文档已出） |

### 7.3 面试官可能追问的脆弱点（提前备好答案）

- 意图识别只看最后一条消息：多轮上下文里的意图切换（"刚才那套房再算算租金"）可能误判 → 备选方案：带最近 N 轮或加澄清反问
- 预算"只扩不缩"：用户真想降预算时存的是旧高值 → 承认是简化设计，正确做法是显式"更新偏好"意图
- risk_analysis 逐条款循环调 LLM：10 条条款=10 次调用，时延线性涨 → 可并行（asyncio）或批量分析
- Store 只有 preferences 一个 namespace：扩展记忆（对话历史/看房记录）需设计 namespace 层级

---

*本文档由源码实读生成；行号以本副本为准，原工程若有重构请以实际代码为准。*

"""
合同审核Agent节点定义
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, filter_messages
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from src.agent.common.llm import model
from src.agent.state.contract import ContractState, Clause, RiskAnalysis


# ============ 结构化输出Schema ============

class ClauseItem(BaseModel):
    """单个合同条款"""
    clause_type: str = Field(
        description="条款类型，如：租金条款、押金条款、租期条款、维修责任、违约责任、退租条款、房屋用途、转租限制等"
    )
    content: str = Field(
        description="条款的原文内容，忠实于原文摘录。如果合同未填写该项（空白），请注明'【未填写】'"
    )
    position: str = Field(
        description="条款在合同中的位置描述，如：第X条、合同第X页等；无法确定时填'未标注'"
    )
    is_blank: bool = Field(
        default=False,
        description="该条款的核心内容（金额、日期等）是否为空白未填写"
    )


class ClausesResult(BaseModel):
    """合同条款提取结果"""
    clauses: List[ClauseItem] = Field(
        default_factory=list,
        description="从合同中提取的所有关键条款列表"
    )


class RiskAnalysisResult(BaseModel):
    """风险分析结果"""
    risk_level: str = Field(
        description="风险等级：高、中、低。高=条款明显违法或严重损害承租人权益；中=存在不合理之处或有潜在纠纷；低=基本合规但可优化"
    )
    risk_description: str = Field(
        description="风险的具体描述，说明该条款存在什么问题或隐患，必须引用具体法条号"
    )
    legal_basis: str = Field(
        description="法律依据的简要说明"
    )
    legal_citation: str = Field(
        description="引用的具体法条编号，如'《民法典》第712条'、'《民法典》第722条'。多个法条用逗号分隔。如无明确法条依据填'无'"
    )
    suggestion: str = Field(
        description="针对该风险的修改或协商建议"
    )
    modified_text: str = Field(
        description="直接可用的修改措辞，可直接替换原条款。如无需修改填'无需修改'"
    )


# ============ 工具函数 ============

def extract_clauses(contract_text: str) -> List[Clause]:
    """
    从合同文本中提取关键条款
    Args:
        contract_text: 合同原文
    Returns:
        提取的条款列表
    """
    system_message = SystemMessage(
        content="""你是一个专业的租房合同审核专家。请仔细阅读用户提供的租房合同原文，从中提取所有关键条款。

需要关注的条款类型包括但不限于：
- 租金条款：租金金额、支付方式、支付周期、租金调整规则
- 押金条款：押金金额、退还条件、扣款情形
- 租期条款：起止日期、租期时长、续租条件
- 维修责任：房屋及设施的维修义务划分
- 违约责任：违约金、提前解约条件及赔偿
- 退租条款：退租通知期、退租流程、费用结算
- 房屋用途：允许的使用方式、限制条件
- 转租限制：是否允许转租及条件
- 费用分担：水电气网物业等费用的承担方

要求：
1. 忠实于原文摘录条款内容，不要篡改或概括
2. 每个条款都要标明在合同中的位置
3. 尽可能全面地提取，不要遗漏重要条款
4. 如果合同中没有提及某类条款，不需要提取该类型
5. 特别注意：如果条款的核心内容（租金金额、押金金额、起止日期等）为空白或写'____'，content 中写'【未填写】'，is_blank 标记为 true"""
    )

    human_message = HumanMessage(content=f"请提取以下租房合同中的关键条款：\n\n{contract_text}")

    result = model.with_structured_output(schema=ClausesResult).invoke(
        [system_message, human_message]
    )

    if result is None or not result.clauses:
        return []

    return [
        Clause(
            clause_type=item.clause_type,
            content=item.content,
            position=item.position,
            is_blank=item.is_blank,
        )
        for item in result.clauses
    ]


def retrieve_laws(
    query: str,
    retriever: BaseRetriever,
    top_k: int = 5
) -> List[dict]:
    """
    从法律知识库检索相关法律条文

    Args:
        query: 查询文本
        retriever: langchain Retriever（可由 VectorStore.as_retriever() 生成）
        top_k: 返回结果数量

    Returns:
        相关法律条文列表，每项包含 content 和 metadata
    """
    docs: List[Document] = retriever.invoke(query)[:top_k]
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def analyze_risk(
    clause: Clause,
    laws: List[dict]
) -> RiskAnalysis:
    """
    分析条款风险

    Args:
        clause: 待分析的条款
        laws: 相关法律条文

    Returns:
        风险分析结果
    """
    laws_text = "\n\n".join(
        f"【法律条文 {i+1}】\n{law['content']}"
        for i, law in enumerate(laws)
    ) if laws else "未检索到相关法律条文。"

    # 按条款类型匹配审查清单
    checklist_map = {
        "解除权": "①是否有催告程序？②催告期限是否合理？③通知期是否对等？④违约金是否超过30%？",
        "违约金": "①违约金是否超过实际损失的30%？②双方违约金是否对等？③是否有除外情形？",
        "押金": "①押金是否超过月租金？②退还条件是否明确？③扣款情形是否具体列举？",
        "维修责任": "①是否区分自然损耗与人为损坏？②出租人是否承担主要维修义务（民法典第712条）？③承租人维修后报销机制有无？",
        "管辖权": "①是否强制约定非承租人所在地法院？②是否排除承租人诉讼地选择权（民事诉讼法第23条）？",
        "转租": "①是否完全禁止转租（民法典第716条允许经同意的转租）？②转租条件是否合理？",
    }
    checklist = ""
    for key, items in checklist_map.items():
        if key in clause.get("clause_type", ""):
            checklist = f"\n\n===== 本条款审查清单 =====\n{items}"
            break

    system_message = SystemMessage(
        content="""你是一个专业的租房合同法律顾问，具备中国民法执业律师的分析水平。请根据合同条款和相关法律条文，进行深度法律风险分析。

===== 分析规范 =====

一、风险等级判定（高/中/低）：
- 高：条款明显违反法律强制性规定（如民法典第712条出租人维修义务），或严重损害承租人合法权益（如单方任意解约权、不合理的管辖权排除）
- 中：条款虽未违法但存在不合理之处（如过短的提前解约通知期、不对等的违约金），或存在潜在纠纷风险（如表述模糊可多重解释）
- 低：条款基本合规，但与承租人最佳利益存在差距，可建议优化

二、核心审查点：
1. 是否违反民事法律强制性规定？必须引用具体法条编号
2. 是否存在程序性瑕疵？（如解约条款是否缺少'经催告后仍不履行'的前置条件——参照民法典第722条）
3. 是否构成格式条款无效情形？（加重对方责任、限制对方主要权利的格式条款无效——民法典第497条）
4. 违约金是否过高？（超过实际损失30%的部分法院不予支持——民法典第585条）

三、输出要求：
1. risk_description：必须写明'根据《民法典》第X条，...'，明确指出违法/不合理之处
2. legal_citation：必须填写具体法条编号，至少一条。格式如'《民法典》第712条'。如果检索到的法律条文中包含编号，务必引用
3. suggestion：给出承租人具体的协商策略和理由
4. modified_text：直接写出推荐的修改后条款文本，可以替换原条款。使用'---'分隔原条款和修改建议"""
    )

    human_message = HumanMessage(
        content=f"""请分析以下合同条款的法律风险：

【条款类型】{clause['clause_type']}
【条款内容】{clause['content']}
【条款位置】{clause['position']}
{checklist}

===== 检索到的相关法律条文 =====
{laws_text}"""
    )

    result = model.with_structured_output(schema=RiskAnalysisResult).invoke(
        [system_message, human_message]
    )

    if result is None:
        return RiskAnalysis(
            clause_type=clause["clause_type"],
            risk_level="未知",
            risk_description="分析失败，请稍后重试",
            legal_basis="无",
            legal_citation="无",
            suggestion="建议人工审核该条款",
            modified_text="",
        )

    return RiskAnalysis(
        clause_type=clause["clause_type"],
        risk_level=result.risk_level,
        risk_description=result.risk_description,
        legal_basis=result.legal_basis,
        legal_citation=result.legal_citation,
        suggestion=result.suggestion,
        modified_text=result.modified_text,
    )


# ============ 节点函数 ============

def clause_extraction_node(state: ContractState) -> dict:
    """
    条款提取节点：从合同原文中提取关键条款。

    优先读 contract_text 字段；如果为空（直接使用 contract_agent 子图时），
    从最后一条用户消息中自动提取。
    """
    contract_text = state.get("contract_text", "")
    if not contract_text:
        user_messages = filter_messages(state["messages"], include_types="human")
        contract_text = user_messages[-1].content if user_messages else ""
    clauses = extract_clauses(contract_text)
    return {"extracted_clauses": clauses}


def make_law_retrieval_node(get_retriever):
    """
    创建法律检索节点的工厂函数。

    接收一个返回 BaseRetriever 的 callable，在节点实际执行时才调用它获取检索器。
    这样图的编译不依赖检索器是否已初始化（模型下载、索引构建等耗时操作延迟到运行时）。

    Args:
        get_retriever: 返回 BaseRetriever 的 callable
    Returns:
        law_retrieval_node 节点函数
    """
    def law_retrieval_node(state: ContractState) -> dict:
        retriever = get_retriever()
        clauses = state.get("extracted_clauses", [])
        all_laws: List[dict] = []
        for clause in clauses:
            query = f"{clause['clause_type']} {clause['content']}"
            laws = retrieve_laws(query, retriever)
            all_laws.extend(laws)
        return {"retrieved_laws": all_laws}

    return law_retrieval_node


def risk_analysis_node(state: ContractState) -> dict:
    """
    风险分析节点：逐条分析条款风险
    """
    clauses = state.get("extracted_clauses", [])
    laws = state.get("retrieved_laws", [])

    risk_results: List[RiskAnalysis] = []
    for clause in clauses:
        # 按条款类型筛选相关法律条文，做简单的关键词匹配
        relevant_laws = [
            law for law in laws
            if clause["clause_type"] in law.get("content", "")
            or clause["clause_type"] in str(law.get("metadata", ""))
        ] or laws  # 无匹配时 fallback 到全部法律条文

        risk = analyze_risk(clause, relevant_laws)
        risk_results.append(risk)

    return {"risk_analysis": risk_results}


def report_generation_node(state: ContractState) -> dict:
    """
    报告生成节点：整合分析结果，生成结构化审核报告
    """
    contract_text = state.get("contract_text", "")
    clauses = state.get("extracted_clauses", [])
    risk_list = state.get("risk_analysis", [])

    # 检测空白关键项
    blank_warnings = []
    critical_types = {"租金条款", "押金条款", "租期条款"}
    for c in clauses:
        if c.get("is_blank") and c.get("clause_type") in critical_types:
            blank_warnings.append(c)

    blank_section = ""
    if blank_warnings:
        blank_section = "\n\n⚠️ 核心信息缺失警告：\n" + "\n".join(
            f"- [{c['clause_type']}] {c['content']}（位置：{c['position']}）→ 建议补全为具体数值，避免口头约定。租期、租金、押金是合同成立的必要要素，空白可能导致合同不成立。"
            for c in blank_warnings
        )
        blank_section += "\n\n**风险等级：高-合同不成立风险**"

    clauses_text = "\n".join(
        f"- [{c['clause_type']}] {'⚠️【未填写】' if c.get('is_blank') else c['content']}（位置：{c['position']}）"
        for c in clauses
    )

    risk_text = "\n\n".join(
        f"""【{r['clause_type']}】风险等级：{r['risk_level']}
风险描述：{r['risk_description']}
法律依据：{r['legal_basis']}
引用法条：{r.get('legal_citation', '无')}
协商建议：{r['suggestion']}
推荐修改措辞：{r.get('modified_text', '无')}"""
        for r in risk_list
    )

    system_message = SystemMessage(
        content="""你是一个专业的租房合同审核报告撰写专家。请根据提供的合同条款提取结果和风险分析结果，生成一份结构清晰、专业的租房合同审核报告。

报告结构应包含：
1. 【合同概览】合同基本信息摘要。如存在核心信息缺失（租金、押金、租期空白），必须在此处红色警告
2. 【条款摘要】提取到的所有关键条款列表
3. 【风险评估】按风险等级（高→中→低）排列的风险分析结果。每个高风险条款必须附带『依据法条』和『推荐修改措辞』
4. 【重点关注】需要立即关注的高风险条款，列出行动清单
5. 【总体建议】综合评估意见和协商建议

强制要求：
- 每个高风险判定后必须引用具体法条编号（如《民法典》第712条）
- 每个可修改条款必须给出直接可用的修改后合同文本
- 对空白核心条款（租期、租金、押金）必须明确标注"高-合同不成立风险"
- 语言专业但易懂，方便普通承租人理解并直接用于协商"""
    )

    human_message = HumanMessage(
        content=f"""请根据以下信息生成合同审核报告：

===== 合同原文（摘要）=====
{contract_text[:2000]}

===== 条款提取结果 =====
{clauses_text}

===== 风险分析结果（含法条引用和修改措辞）=====
{risk_text}

{blank_section}"""
    )

    response: AIMessage = model.invoke([system_message, human_message])
    return {"audit_report": response.content}

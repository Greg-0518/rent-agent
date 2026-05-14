"""
房源图片视觉分析节点

支持：上传房源图片 → AI 分析房间类型、装修、采光、面积估算、租金估算
"""
import os, base64, re
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.agent.common.llm import model
from src.agent.state.main import State
from src.agent.state.vision import VisionState

def _get_vl_model():
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return ChatOpenAI(model="qwen-vl-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=key)
    return ChatOpenAI(model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY")))

def _encode_image(path: str) -> tuple:
    ext = Path(path).suffix.lower()
    mime = {".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",
            ".webp":"image/webp",".gif":"image/gif"}.get(ext,"image/jpeg")
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode(), mime

def analyze_rental_image(image_path: str) -> str:
    model = _get_vl_model()
    b64, mime = _encode_image(image_path)
    resp = model.invoke([
        SystemMessage(content="你是专业租房顾问，擅长通过房源图片进行多维度分析。"),
        HumanMessage(content=[
            {"type":"text","text":"请分析这张房源图片：\n1.【房间类型】\n2.【装修状况】\n3.【家具家电】\n4.【采光与朝向】\n5.【面积估算】\n6.【整体评价】\n7.【租金估算（元/月）】"},
            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}}])])
    return resp.content

def _extract_image(state: State) -> Optional[str]:
    msgs = state.get("messages",[])
    if not msgs: return None
    last = msgs[-1]
    if hasattr(last,"content") and isinstance(last.content, list):
        for p in last.content:
            if isinstance(p,dict) and p.get("type")=="image_url":
                return p.get("image_url",{}).get("url","")
    if hasattr(last,"content") and isinstance(last.content, str):
        t = last.content
        m = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)(?:\?\S*)?", t)
        if m: return m.group(0)
        for ext in (".png",".jpg",".jpeg",".webp"):
            m = re.search(rf"([A-Za-z]:\S+{ext}|\S+{ext})", t)
            if m and Path(m.group(1)).exists(): return m.group(1)
    return None

def image_analysis_node(state: State) -> dict:
    img = _extract_image(state)
    if not img:
        return {"messages":[AIMessage(content="未检测到图片。请上传房源图片或提供本地路径/URL。")]}
    try:
        result = analyze_rental_image(img)
        return {"messages":[AIMessage(content=f"[房源图片分析]\n\n{result}")]}
    except Exception as e:
        return {"messages":[AIMessage(content=f"图片分析失败：{e}")]}


# ====== VisionState 下的节点 ======

def issue_detection_node(state: VisionState) -> dict:
    """问题检测节点：分析图片中的潜在隐患（发霉、裂缝、采光差、安全隐患等）"""
    msgs = state.get("messages", [])
    # 取最近一条 AI 分析结果作为上下文
    analysis_context = ""
    for m in reversed(msgs):
        if hasattr(m, "content") and isinstance(m.content, str) and "房间类型" in m.content:
            analysis_context = m.content[:2000]
            break
    if not analysis_context:
        return {"issues_detected": [{"category": "无数据", "description": "未检测到图片分析结果，请先完成图片分析"}]}

    prompt = f"""根据以下房源图片分析结果，检测可能存在的隐患和问题：
{analysis_context}

请逐项列出：
1. 结构隐患（裂缝、漏水、墙体问题等）
2. 采光通风问题
3. 噪音和环境问题
4. 安全隐患（电线、燃气等）
5. 其他需要关注的问题

格式：【隐患类别】具体描述（严重程度：高/中/低）"""

    resp = model.invoke([HumanMessage(content=prompt)])
    # 简单解析
    issues = []
    for line in resp.content.strip().split("\n"):
        line = line.strip()
        if line and ("【" in line or "隐患" in line or "问题" in line):
            issues.append({"description": line})
    return {"issues_detected": issues or [{"description": resp.content.strip()}],
            "messages": [AIMessage(content=f"[隐患检测]\n\n{resp.content}")]}


def report_generation_node(state: VisionState) -> dict:
    """报告生成节点：整合图片分析和隐患检测，生成完整看房报告"""
    msgs = state.get("messages", [])
    # 收集所有 AI 分析结果
    analyses = []
    for m in msgs:
        if hasattr(m, "content") and isinstance(m.content, str):
            content = m.content
            if any(kw in content for kw in ["房间类型", "隐患检测", "图片分析"]):
                analyses.append(content[-1500:])

    combined = "\n\n---\n\n".join(analyses) if analyses else "暂无分析数据"
    issues = state.get("issues_detected", [])

    prompt = f"""请根据以下房源分析数据，生成一份专业的看房报告：

{combined}

检测到的隐患：
{chr(10).join(f"- {i.get('description','')}" for i in issues) if issues else '无'}

报告结构：
1.【整体概况】房源基本信息
2.【各房间评测】逐房间打分（满分10分）
3.【隐患清单】按严重程度排列的问题
4.【性价比评估】结合租金和品质的综合判断
5.【最终建议】是否推荐租住，以及签约时需要注意的事项

语言专业、客观，方便租房者做决策。"""

    resp = model.invoke([SystemMessage(content="你是专业的看房顾问和房屋检测工程师。"), HumanMessage(content=prompt)])
    return {"report": resp.content,
            "messages": [AIMessage(content=f"[看房报告]\n\n{resp.content}")]}

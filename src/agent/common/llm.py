"""LLM 接入层 —— 配置驱动，支持 base / tuned 模型切换

⚠️ 模型对象在 **import 时** 解析：各节点用的是 ``from src.agent.common.llm import model``
这种「值绑定」，运行期重新给 ``llm.model`` 赋值，已经 import 过的节点看不到新值。
因此**切换模型只能通过进程级环境变量**（评测 runner 用 subprocess 起 pytest 来隔离），
不要试图在运行期改这个全局。

环境变量：
    LLM_MODEL            主模型名，默认 deepseek-chat
    LLM_MODEL_THINKING   思考模型名，默认 deepseek-reasoner
    LLM_BASE_URL         OpenAI 兼容端点，设置后走该端点（指向本地 vLLM/Ollama 上的 LoRA）
    LLM_API_KEY          端点密钥，缺省回落到 DEEPSEEK_API_KEY
    LLM_TEMPERATURE      默认 0（飞轮文 §5.2：judge 与对照实验都要求 temperature=0）
    LLM_PROFILE          本次运行的模型标签（如 base / tuned），只用于评测报告署名
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# override=True：以 .env 为唯一权威配置源。
# 默认行为下，机器上残留的同名环境变量会**静默盖住** .env（本项目踩过：
# Machine 级 DEEPSEEK_API_KEY 还是旧 key，导致 .env 换成新 key 后依然 401）。
load_dotenv(override=True)

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_THINKING_MODEL = "deepseek-reasoner"


def get_model_id(is_thinking: bool = False) -> str:
    """当前生效的模型名。"""
    if is_thinking:
        return os.getenv("LLM_MODEL_THINKING") or DEFAULT_THINKING_MODEL
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def get_model_config(is_thinking: bool = False) -> dict:
    """当前生效的模型配置（不含密钥明文），供日志与评测报告记录。"""
    return {
        "profile": os.getenv("LLM_PROFILE", ""),
        "model": get_model_id(is_thinking),
        "base_url": os.getenv("LLM_BASE_URL") or None,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
    }


def getModel(is_thinking: bool = False):
    """按环境变量构建模型。

    设置了 LLM_BASE_URL 时走 OpenAI 兼容端点——本地部署的微调模型（vLLM/Ollama）
    就是靠这条路径接进来的，base 与 tuned 的差异只在环境变量，代码不动。
    """
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    base_url = os.getenv("LLM_BASE_URL")

    if base_url:
        return init_chat_model(
            model=get_model_id(is_thinking),
            model_provider="openai",
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY") or "EMPTY",
            temperature=temperature,
        )
    return init_chat_model(model=get_model_id(is_thinking), temperature=temperature)


model = getModel(False)

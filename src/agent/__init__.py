"""New LangGraph Agent.

This module defines a custom graph
"""

from src.agent.graph import graph
from src.agent.recommend import recommended_graph
from src.agent.reserve import reserve_graph
from src.agent.extend import extend_graph
from src.agent.contract import contract_graph
from src.agent.finance import finance_graph
from src.agent.vision import vision_graph


__all__ = ["graph", "recommended_graph", "reserve_graph", "extend_graph",
           "contract_graph", "finance_graph", "vision_graph"]
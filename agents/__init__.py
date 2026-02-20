# Multi-Agent System for AIOS
# Planning-based agent orchestration framework

from .base_agent import BaseAgent
from .embedding_agent import EmbeddingAgent
from .retrieval_agent import RetrievalAgent
from .generation_agent import GenerationAgent
from .planning_agent import PlanningAgent

# Backward compatibility alias
RAGCoordinator = PlanningAgent

__all__ = [
    "BaseAgent",
    "EmbeddingAgent",
    "RetrievalAgent",
    "GenerationAgent",
    "PlanningAgent",
    "RAGCoordinator",  # For backward compatibility
]

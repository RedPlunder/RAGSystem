# Base Agent class for AIOS integration
# This provides a simplified interface compatible with AIOS kernel

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio


class BaseAgent(ABC):
    """
    Base class for all AIOS agents.

    This class provides the interface for agents to interact with AIOS kernel
    through syscalls. Each agent should implement the run() method.
    """

    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.

        Args:
            agent_name: Unique identifier for the agent
            config: Configuration dictionary for the agent
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for the agent"""
        import logging
        logger = logging.getLogger(self.agent_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.agent_name}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def run(self, task_input: Any) -> Any:
        """
        Main execution method for the agent.

        Args:
            task_input: Input task data

        Returns:
            Processed result
        """
        pass

    def llm_call(self, messages: list, model: Optional[str] = None, **kwargs) -> str:
        """
        Make a synchronous LLM call through AIOS kernel.

        This is a simplified version. In production, this would use
        AIOS Syscall mechanism to interact with LLM cores.

        Args:
            messages: List of message dictionaries
            model: Model name to use
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # In production, this would create an LLMQuery and execute via Syscall
        # For now, we'll use direct OpenAI call as fallback
        try:
            import openai
            import os

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = model or self.config.get("default_model", "gpt-4")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    async def llm_call_async(
        self,
        messages: list,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Make an asynchronous LLM call through AIOS kernel.

        Args:
            messages: List of message dictionaries
            model: Model name to use
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        try:
            import openai
            import os

            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = model or self.config.get("default_model", "gpt-4")

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Async LLM call failed: {e}")
            raise

    def storage_query(self, operation: str, **kwargs) -> Any:
        """
        Query AIOS storage manager.

        Args:
            operation: Operation type (read, write, delete, etc.)
            **kwargs: Operation parameters

        Returns:
            Query result
        """
        # In production, this would create a StorageQuery and execute via Syscall
        self.logger.info(f"Storage query: {operation}")
        return None

    def memory_query(self, operation: str, **kwargs) -> Any:
        """
        Query AIOS memory manager.

        Args:
            operation: Operation type (add, retrieve, update, etc.)
            **kwargs: Operation parameters

        Returns:
            Query result
        """
        # In production, this would create a MemoryQuery and execute via Syscall
        self.logger.info(f"Memory query: {operation}")
        return None

    def tool_call(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool through AIOS tool manager.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        # In production, this would create a ToolQuery and execute via Syscall
        self.logger.info(f"Tool call: {tool_name}")
        return None

# Generation Agent - Handles LLM-based answer generation
# Extracted from demo_noSecond.py

import asyncio
from typing import List, Dict, Any, Optional, Tuple

from .base_agent import BaseAgent


class GenerationAgent(BaseAgent):
    """
    Agent responsible for generating answers using LLM.

    This agent handles:
    - Prompt construction for RAG tasks
    - Multi-turn conversation with LLM
    - Token usage tracking
    - Retry logic for API failures
    """

    def __init__(
        self,
        agent_name: str = "GenerationAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config)

        # Configuration with defaults
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0)
        self.max_retries = self.config.get("max_retries", 3)
        self.prompt_template = self.config.get(
            "prompt_template",
            self._default_k8s_prompt_template()
        )

        # Token tracking
        self.total_tokens = {
            "total_prompt": 0,
            "total_completion": 0,
            "overall_total": 0
        }

    def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for generation agent.

        Args:
            task_input: Dictionary containing:
                - action: "generate" or "generate_batch"
                - For generate:
                    - title: Question title
                    - body: Question body
                    - context: Retrieved context
                - For generate_batch:
                    - title_queries: List of titles
                    - body_queries: List of bodies
                    - contexts: List of contexts

        Returns:
            Dictionary with generated responses and token info
        """
        action = task_input.get("action", "generate")

        if action == "generate":
            return asyncio.run(self.generate_single(
                title=task_input.get("title", ""),
                body=task_input.get("body", ""),
                context=task_input.get("context", "")
            ))

        elif action == "generate_batch":
            return asyncio.run(self.generate_batch(
                title_queries=task_input.get("title_queries", []),
                body_queries=task_input.get("body_queries", []),
                contexts=task_input.get("contexts", [])
            ))

        else:
            raise ValueError(f"Unknown action: {action}")

    async def generate_single(
        self,
        title: str,
        body: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Generate a single response.

        Args:
            title: Question title
            body: Question body
            context: Retrieved context

        Returns:
            Dictionary with response and token info
        """
        # Construct prompt
        messages = self.construct_messages(title, body, context)

        # Call LLM
        response, token_info = await self.call_llm_with_retry(messages)

        # Update token tracking
        self.total_tokens["total_prompt"] += token_info["prompt_tokens"]
        self.total_tokens["total_completion"] += token_info["completion_tokens"]
        self.total_tokens["overall_total"] += token_info["total_tokens"]

        return {
            "response": response,
            "token_info": token_info,
            "total_tokens": self.total_tokens.copy()
        }

    async def generate_batch(
        self,
        title_queries: List[str],
        body_queries: List[str],
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate responses for a batch of queries.

        Args:
            title_queries: List of question titles
            body_queries: List of question bodies
            contexts: List of retrieved contexts

        Returns:
            Dictionary with responses and token info
        """
        # Construct prompts for all queries
        prompts = []
        for title, body, context in zip(title_queries, body_queries, contexts):
            messages = self.construct_messages(title, body, context)
            prompts.append(messages)

        # Call LLM asynchronously for all queries
        results = await asyncio.gather(
            *[self.call_llm_with_retry(msg) for msg in prompts],
            return_exceptions=True
        )

        # Separate responses and token info
        responses = []
        token_infos = []

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Generation failed: {result}")
                responses.append(f"Error: {str(result)}")
                token_infos.append({
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            else:
                responses.append(result[0])
                token_infos.append(result[1])

                # Update global token tracking
                self.total_tokens["total_prompt"] += result[1]["prompt_tokens"]
                self.total_tokens["total_completion"] += result[1]["completion_tokens"]
                self.total_tokens["overall_total"] += result[1]["total_tokens"]

        return {
            "responses": responses,
            "token_infos": token_infos,
            "total_tokens": self.total_tokens.copy()
        }

    def construct_messages(
        self,
        title: str,
        body: str,
        context: str
    ) -> List[Dict[str, str]]:
        """
        Construct messages for LLM API call.

        Args:
            title: Question title
            body: Question body
            context: Retrieved context

        Returns:
            List of message dictionaries
        """
        # Fill in the prompt template
        user_message = self.prompt_template.format(
            context=context,
            title=title,
            body=body
        )

        return [
            {"role": "user", "content": user_message}
        ]

    async def call_llm_with_retry(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, int]]:
        """
        Call LLM with retry logic.

        Args:
            messages: List of message dictionaries

        Returns:
            Tuple of (response text, token info dict)
        """
        import openai

        for attempt in range(self.max_retries):
            try:
                response = await self.llm_call_async(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature
                )

                # Since llm_call_async returns just the content,
                # we need to make a direct call to get token info
                # For now, use a simplified version
                client = openai.AsyncOpenAI()
                api_response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )

                # Extract token usage info
                usage = api_response.usage
                token_info = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }

                response_text = api_response.choices[0].message.content

                return response_text, token_info

            except Exception as e:
                self.logger.warning(
                    f"LLM call attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Last attempt failed
                    error_msg = f"API Error after {self.max_retries} retries: {e}"
                    self.logger.error(error_msg)
                    return error_msg, {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }

    @staticmethod
    def _default_k8s_prompt_template() -> str:
        """
        Default prompt template for Kubernetes troubleshooting.

        Returns:
            Prompt template string
        """
        return """
<prompt>
    <instructions>
        <summary>
            You are a Kubernetes expert and troubleshooting assistant. You will receive "user_query" and "retrieved_knowledge".
            Your task is to diagnose and resolve Kubernetes-related issues in "user_query" using only the "retrieved_knowledge" provided below.
            Your responses should be entirely derived from "retrieved_knowledge" and should not include any content unrelated to the "retrieved_knowledge".
        </summary>

        <structured_debugging_approach>
            <step1>Identification: Identify the exact YAML field, CLI flag, or Kubernetes object causing the issue.</step1>
            <step2>Reasoning: Explain the root cause based only on the "retrieved_knowledge". If the content of "retrieved_knowledge" cannot resolve the issue in "user_query", just return "Retrieved knowledge is insufficient to answer the question"</step2>
            <step3>Remediation: Provide a verified fix for Kubernetes YAML configuration or CLI flag, which MUST be complete, production-ready resources.</step3>
            <step4>Validation: Verify that the YAML is syntactically correct and conforms to Kubernetes API schema.</step4>
            <step5>Repetition: If multiple solutions exist, repeat steps 1-4 for each solution.</step5>
        </structured_debugging_approach>

        <yaml_requirements>
            **CRITICAL: All solution YAML code blocks MUST be complete, deployable Kubernetes resources.**

            1. **Complete Resource Structure**:
            - Every solution YAML block MUST include: `apiVersion`, `kind`, `metadata`, and `spec` (or `data` for ConfigMaps/Secrets).
            - NEVER output partial snippets as solutions.

            2. **Valid Kubernetes Schema**:
            - Use correct and current `apiVersion` values.
            - Ensure all field names match the official Kubernetes API specification.
            - Include all required fields.

            3. **Security and Best Practices**:
            - Follow Kubernetes security best practices.
            - Ensure the configuration is production-ready and secure.

            4. **Production Readiness**:
            - Configure for production use where appropriate.

            5. **Helm Templates**:
            - If the solution involves Helm, provide the **rendered YAML** with actual values substituted.
            - Do NOT include Helm template syntax like `{{{{ .Values.xxx }}}}` in the solution.
            - You may explain the Helm syntax in text, but the code block must be pure, valid YAML.

            6. **Use different code block markers based on content type:**
                - Use ```yaml: complete for complete, production-ready Kubernetes manifests (the fix/solution).
                - Use ```yaml for partial snippets, problem examples, or excerpts you are explaining (not the fix).
                - Use ```bash for CLI commands.
                - NEVER use `...` (three dots) as a placeholder in YAML. Use comments like `# ... other fields` instead.

                Provide a brief explanation for each solution, focusing on the root cause and fix.
        </yaml_requirements>

        <output_example>
            Solution1:
            Fixed YAML file (code or CLI flag) returned must be complete.
            Give a simple explanation and keep it minimal and directly tied to the fixed YAML file.

            Solution2 (If there are multiple solutions, repeat the above output format):
            Fixed YAML file (code or CLI flag) returned must be complete.
            Give a simple explanation and keep it minimal and directly tied to the fixed YAML file.
            ...
        </output_example>

    </instructions>

    <retrieved_knowledge>
        <![CDATA[ {context} ]]>
    </retrieved_knowledge>

    <user_query>
        <title>{title}</title>
        <body>{body}</body>
    </user_query>

</prompt>
"""

    def set_prompt_template(self, template: str):
        """
        Set a custom prompt template.

        Args:
            template: Prompt template string with {title}, {body}, {context} placeholders
        """
        self.prompt_template = template

    def get_token_stats(self) -> Dict[str, int]:
        """
        Get cumulative token usage statistics.

        Returns:
            Dictionary with token counts
        """
        return self.total_tokens.copy()

    def reset_token_stats(self):
        """Reset token usage statistics."""
        self.total_tokens = {
            "total_prompt": 0,
            "total_completion": 0,
            "overall_total": 0
        }

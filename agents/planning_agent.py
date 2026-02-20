# Planning Agent - Orchestrates multi-agent workflows
# Analyzes user queries, creates execution plans, and coordinates task agents

import time
import json
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent


class PlanningAgent(BaseAgent):
    """
    Planning agent that analyzes queries and orchestrates task execution.

    This agent is responsible for:
    1. Analyzing user queries to understand intent
    2. Creating execution plans with required agents
    3. Coordinating task agents to complete the workflow
    4. Aggregating results from multiple agents
    5. Tracking execution metrics
    """

    def __init__(
        self,
        agent_name: str = "PlanningAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config)

        # Available task agents registry
        self.task_agents: Dict[str, BaseAgent] = {}

        # Planning LLM configuration
        self.planning_model = self.config.get("planning_model", "gpt-4")
        self.planning_temperature = self.config.get("planning_temperature", 0)

        # Performance tracking
        self.metrics = {
            "planning_time": 0,
            "execution_time": 0,
            "total_time": 0,
            "agents_called": []
        }

    def register_agent(self, agent_name: str, agent: BaseAgent):
        """
        Register a task agent for orchestration.

        Args:
            agent_name: Unique identifier for the agent
            agent: Agent instance
        """
        self.task_agents[agent_name] = agent
        self.logger.info(f"Registered task agent: {agent_name}")

    def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for planning agent.

        Args:
            task_input: Dictionary containing:
                - query: User query or task description
                - context: Optional additional context
                - execution_mode: "plan_only", "plan_and_execute", "execute_plan"
                - plan: Optional pre-defined plan (for execute_plan mode)

        Returns:
            Dictionary with execution results
        """
        execution_mode = task_input.get("execution_mode", "plan_and_execute")

        if execution_mode == "plan_only":
            return self.create_plan(
                query=task_input.get("query"),
                context=task_input.get("context")
            )

        elif execution_mode == "plan_and_execute":
            return self.plan_and_execute(
                query=task_input.get("query"),
                context=task_input.get("context")
            )

        elif execution_mode == "execute_plan":
            return self.execute_plan(
                plan=task_input.get("plan")
            )

        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

    def create_plan(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze query and create execution plan.

        Args:
            query: User query or task description
            context: Optional additional context

        Returns:
            Dictionary with execution plan
        """
        start_time = time.time()

        self.logger.info("Creating execution plan...")

        # Get list of available agents
        available_agents = list(self.task_agents.keys())

        # Construct planning prompt
        planning_prompt = self._construct_planning_prompt(
            query=query,
            context=context,
            available_agents=available_agents
        )

        # Call planning LLM
        messages = [
            {"role": "system", "content": "You are an expert task planning agent."},
            {"role": "user", "content": planning_prompt}
        ]

        try:
            plan_text = self.llm_call(
                messages=messages,
                model=self.planning_model,
                temperature=self.planning_temperature
            )

            # Parse plan from LLM response
            plan = self._parse_plan(plan_text)

        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            plan = {
                "status": "error",
                "error": str(e),
                "steps": []
            }

        planning_time = time.time() - start_time
        self.metrics["planning_time"] += planning_time

        plan["planning_time"] = planning_time

        self.logger.info(f"Plan created in {planning_time:.2f}s")

        return plan

    def execute_plan(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a pre-defined plan.

        Args:
            plan: Execution plan with steps

        Returns:
            Dictionary with execution results
        """
        start_time = time.time()

        self.logger.info("Executing plan...")

        if "steps" not in plan:
            raise ValueError("Plan must contain 'steps' field")

        results = []
        execution_context = {}  # Store results for use in subsequent steps

        for i, step in enumerate(plan["steps"]):
            self.logger.info(f"Executing step {i+1}/{len(plan['steps'])}: {step.get('description', 'No description')}")

            try:
                step_result = self._execute_step(step, execution_context)
                results.append({
                    "step": i + 1,
                    "status": "success",
                    "result": step_result
                })

                # Store result for potential use in next steps
                if "output_key" in step:
                    execution_context[step["output_key"]] = step_result

                # Track which agents were called
                if step.get("agent"):
                    self.metrics["agents_called"].append(step["agent"])

            except Exception as e:
                self.logger.error(f"Step {i+1} failed: {e}")
                results.append({
                    "step": i + 1,
                    "status": "error",
                    "error": str(e)
                })

                # Decide whether to continue or abort
                if step.get("critical", False):
                    self.logger.error("Critical step failed, aborting execution")
                    break

        execution_time = time.time() - start_time
        self.metrics["execution_time"] += execution_time
        self.metrics["total_time"] += execution_time

        self.logger.info(f"Plan executed in {execution_time:.2f}s")

        return {
            "status": "completed",
            "results": results,
            "execution_time": execution_time,
            "agents_called": self.metrics["agents_called"]
        }

    def plan_and_execute(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create plan and execute it in one go.

        Args:
            query: User query or task description
            context: Optional additional context

        Returns:
            Dictionary with plan and execution results
        """
        start_time = time.time()

        # Step 1: Create plan
        plan = self.create_plan(query=query, context=context)

        if plan.get("status") == "error":
            return {
                "status": "error",
                "error": "Planning failed",
                "plan": plan
            }

        # Step 2: Execute plan
        execution_result = self.execute_plan(plan)

        total_time = time.time() - start_time

        return {
            "status": "completed",
            "plan": plan,
            "execution": execution_result,
            "total_time": total_time
        }

    def _construct_planning_prompt(
        self,
        query: str,
        context: Optional[str],
        available_agents: List[str]
    ) -> str:
        """
        Construct prompt for planning LLM.

        Args:
            query: User query
            context: Optional context
            available_agents: List of available agent names

        Returns:
            Planning prompt
        """
        # Get agent descriptions
        agent_descriptions = []
        for agent_name in available_agents:
            agent = self.task_agents.get(agent_name)
            if agent:
                description = getattr(agent, '__doc__', 'No description available')
                agent_descriptions.append(f"- {agent_name}: {description.strip()}")

        agents_info = "\n".join(agent_descriptions) if agent_descriptions else "No agents available"

        prompt = f"""
You are a task planning agent. Analyze the user's query and create a step-by-step execution plan.

User Query:
{query}

{f"Additional Context:\n{context}\n" if context else ""}

Available Agents:
{agents_info}

Please create a detailed execution plan with the following format:

{{
  "steps": [
    {{
      "step": 1,
      "description": "Brief description of this step",
      "agent": "agent_name_to_use",
      "input": {{
        "action": "action_name",
        "param1": "value1",
        ...
      }},
      "output_key": "key_to_store_result",
      "critical": true/false
    }},
    ...
  ],
  "reasoning": "Brief explanation of the plan"
}}

Rules:
1. Only use agents from the available list
2. Steps should be sequential and logical
3. Use "output_key" to reference results from previous steps
4. Mark steps as "critical" if failure should abort the entire plan
5. Return ONLY valid JSON, no additional text

Create the plan:
"""
        return prompt

    def _parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse plan from LLM response.

        Args:
            plan_text: Raw text from planning LLM

        Returns:
            Parsed plan dictionary
        """
        try:
            # Try to extract JSON from response
            # LLM might wrap it in markdown code blocks
            if "```json" in plan_text:
                start = plan_text.find("```json") + 7
                end = plan_text.find("```", start)
                plan_text = plan_text[start:end].strip()
            elif "```" in plan_text:
                start = plan_text.find("```") + 3
                end = plan_text.find("```", start)
                plan_text = plan_text[start:end].strip()

            plan = json.loads(plan_text)

            # Validate plan structure
            if "steps" not in plan:
                raise ValueError("Plan must contain 'steps' field")

            plan["status"] = "success"
            return plan

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse plan JSON: {e}")
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": plan_text,
                "steps": []
            }

    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """
        Execute a single step in the plan.

        Args:
            step: Step configuration
            context: Execution context with previous results

        Returns:
            Step execution result
        """
        agent_name = step.get("agent")
        if not agent_name:
            raise ValueError("Step must specify 'agent'")

        if agent_name not in self.task_agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent = self.task_agents[agent_name]

        # Prepare input, replacing context references
        step_input = step.get("input", {})
        step_input = self._resolve_context_references(step_input, context)

        # Execute agent
        self.logger.info(f"Calling agent: {agent_name}")
        result = agent.run(step_input)

        return result

    def _resolve_context_references(
        self,
        input_dict: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve references to previous step results.

        Args:
            input_dict: Input dictionary potentially containing references
            context: Execution context

        Returns:
            Resolved input dictionary
        """
        resolved = {}

        for key, value in input_dict.items():
            if isinstance(value, str) and value.startswith("$context."):
                # Reference to context variable
                context_key = value[9:]  # Remove "$context." prefix
                resolved[key] = context.get(context_key, value)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_context_references(value, context)
            else:
                resolved[key] = value

        return resolved

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with timing and execution metrics
        """
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "planning_time": 0,
            "execution_time": 0,
            "total_time": 0,
            "agents_called": []
        }

    def list_agents(self) -> List[str]:
        """
        Get list of registered agents.

        Returns:
            List of agent names
        """
        return list(self.task_agents.keys())

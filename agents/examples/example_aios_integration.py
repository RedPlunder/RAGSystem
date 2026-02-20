#!/usr/bin/env python
# Example: AIOS Integration with RAG Agents
# This demonstrates how to integrate RAG agents with AIOS kernel

import os
import sys
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import RAGCoordinator


def example_local_integration():
    """
    Example of integrating RAG agents in AIOS local mode.
    This shows how agents can be used within AIOS kernel.
    """
    print("="*60)
    print("Example 1: Local AIOS Integration")
    print("="*60)

    # In AIOS, you would typically load this from config
    config = {
        "embedding_config": {
            "embedding_model": "text-embedding-3-small",
            "batch_size": 20,
            "embedding_path": "./dataset/doc_embeddings.npy"
        },
        "retrieval_config": {
            "rerank_method": "model"
        },
        "generation_config": {
            "model": "gpt-4",
            "temperature": 0
        }
    }

    # Initialize RAG coordinator as an AIOS agent
    rag_agent = RAGCoordinator(
        agent_name="RAGCoordinator",
        config=config
    )

    print(f"\n✅ Agent '{rag_agent.agent_name}' initialized")
    print(f"Configuration loaded: {len(config)} sections")

    # In AIOS, agents are submitted through the agent factory
    # Here's how you would structure the task input:
    task_input = {
        "action": "query",
        "title": "How do I configure a Kubernetes Secret?",
        "body": "I need to store sensitive data in my cluster.",
        "tags": "<K8s><Secret>",
        "k": 10,
        "rerank_k": 4
    }

    print(f"\nTask input prepared:")
    print(f"  - Action: {task_input['action']}")
    print(f"  - Query: {task_input['title']}")

    # Note: In AIOS, you would submit through AgentFactory
    # For example:
    # process_id = submit_agent({
    #     "agent_name": "RAGCoordinator",
    #     "task_input": task_input
    # })


def example_rest_api_integration():
    """
    Example of submitting RAG agents through AIOS REST API.
    This assumes AIOS kernel is running on localhost:8000.
    """
    print("\n" + "="*60)
    print("Example 2: REST API Integration")
    print("="*60)

    try:
        import requests
    except ImportError:
        print("⚠️  requests library not installed. Install with: pip install requests")
        return

    # AIOS kernel endpoint
    kernel_url = "http://localhost:8000"

    # Agent submission payload
    agent_config = {
        "agent_id": "yourname/rag-coordinator",  # Format: author/agent_name
        "agent_config": {
            "task": {
                "action": "query",
                "title": "Kubernetes Ingress configuration",
                "body": "How to expose services externally?",
                "tags": "<K8s><Ingress>",
                "k": 10,
                "rerank_k": 4
            }
        }
    }

    print(f"\nAIOS Kernel URL: {kernel_url}")
    print(f"Agent ID: {agent_config['agent_id']}")
    print(f"Task action: {agent_config['agent_config']['task']['action']}")

    print("\n📤 To submit agent via REST API:")
    print(f"""
import requests

response = requests.post(
    "{kernel_url}/agents/submit",
    json={agent_config}
)

execution_id = response.json()["execution_id"]
print(f"Agent submitted with execution ID: {{execution_id}}")

# Poll for results
status_response = requests.get(
    "{kernel_url}/agents/{{execution_id}}/status"
)

result = status_response.json()
print(f"Status: {{result['status']}}")
print(f"Result: {{result['result']}}")
""")


def example_agent_packaging():
    """
    Example of packaging RAG agents for AIOS distribution.
    """
    print("\n" + "="*60)
    print("Example 3: Agent Packaging")
    print("="*60)

    print("""
To package RAG agents for AIOS distribution:

1. Create agent directory structure:
   ```
   pyopenagi/agents/yourname/rag_agents/
   ├── agent.py              # Main agent file with RAGCoordinator
   ├── config.yaml           # Agent configuration
   ├── requirements.txt      # Dependencies
   └── README.md            # Agent documentation
   ```

2. Implement agent.py:
   ```python
   from agents import RAGCoordinator

   class RagAgents(RAGCoordinator):
       '''RAG Agents for Kubernetes troubleshooting'''

       def __init__(self, agent_name: str, *args, **kwargs):
           config = {
               "embedding_config": {...},
               "retrieval_config": {...},
               "generation_config": {...}
           }
           super().__init__(agent_name, config=config)
   ```

3. Create config.yaml:
   ```yaml
   name: "rag_agents"
   version: "1.0.0"
   author: "yourname"
   description: "RAG agents for Kubernetes troubleshooting"
   dependencies:
     - openai
     - faiss-cpu
     - transformers
     - torch
   ```

4. Package and upload:
   ```bash
   # Package as .agent file
   cd pyopenagi/agents/yourname
   tar -czf rag_agents_v1.0.0.agent rag_agents/

   # Upload to AIOS hub (https://app.aios.foundation/)
   ```

5. Users can then install your agent:
   ```bash
   # Through AIOS
   aios agent install yourname/rag_agents
   ```
""")


def example_multi_agent_coordination():
    """
    Example of coordinating multiple agents in AIOS.
    """
    print("\n" + "="*60)
    print("Example 4: Multi-Agent Coordination")
    print("="*60)

    print("""
RAG agents can work with other AIOS agents:

1. **Sequential workflow**:
   ```
   User Query
     ↓
   RAGCoordinator (retrieve context)
     ↓
   ValidationAgent (validate solution)
     ↓
   ExecutionAgent (apply configuration)
     ↓
   MonitoringAgent (verify deployment)
   ```

2. **Parallel workflow**:
   ```
   User Query
     ├→ RAGCoordinator (retrieve from docs)
     ├→ WebSearchAgent (search online)
     └→ ExperienceAgent (query past solutions)
     ↓ (merge results)
   GenerationAgent (synthesize answer)
   ```

3. **Implementation in AIOS**:
   ```python
   # Submit multiple agents
   rag_process_id = submit_agent({
       "agent_name": "RAGCoordinator",
       "task_input": {"action": "query", ...}
   })

   # Wait for result
   rag_result = await_execution(rag_process_id)

   # Use result in next agent
   validation_process_id = submit_agent({
       "agent_name": "ValidationAgent",
       "task_input": {"solution": rag_result["answer"]}
   })
   ```
""")


def main():
    """Run all examples."""
    print("🚀 RAG Agents AIOS Integration Examples\n")

    # Check dependencies
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Note: OPENAI_API_KEY not set. Some examples may not run.")
        print()

    # Run examples
    example_local_integration()
    example_rest_api_integration()
    example_agent_packaging()
    example_multi_agent_coordination()

    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("""
For more information:
- AIOS Documentation: https://docs.aios.foundation/
- AIOS GitHub: https://github.com/agiresearch/AIOS
- Cerebrum SDK: https://github.com/agiresearch/Cerebrum

Next steps:
1. Implement your agents using the base classes
2. Test locally with example_single_query.py
3. Package for AIOS distribution
4. Upload to AIOS hub
""")


if __name__ == "__main__":
    main()

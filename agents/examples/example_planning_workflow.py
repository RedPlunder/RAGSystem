#!/usr/bin/env python
# Example: Planning-based Multi-Agent Workflow

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import PlanningAgent, EmbeddingAgent, RetrievalAgent, GenerationAgent


def main():
    """
    Example of using PlanningAgent to orchestrate multiple task agents.

    Workflow:
    1. User submits a query
    2. PlanningAgent analyzes the query
    3. PlanningAgent creates an execution plan
    4. PlanningAgent calls necessary agents (Embedding, Retrieval, Generation)
    5. Results are aggregated and returned
    """

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("="*70)
    print("Planning-based Multi-Agent System Example")
    print("="*70)

    # Step 1: Initialize task agents
    print("\n1️⃣  Initializing task agents...")

    # Embedding agent for document vectorization
    embedding_agent = EmbeddingAgent(
        agent_name="EmbeddingAgent",
        config={
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "batch_size": 20,
            "embedding_path": "./dataset/doc_embeddings.npy"
        }
    )

    # Retrieval agent for document search
    retrieval_agent = RetrievalAgent(
        agent_name="RetrievalAgent",
        config={
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_method": "model",
            "max_length": 4096
        }
    )

    # Generation agent for LLM-based text generation
    generation_agent = GenerationAgent(
        agent_name="GenerationAgent",
        config={
            "model": "gpt-4",
            "temperature": 0,
            "max_retries": 3
        }
    )

    print("✅ Task agents initialized:")
    print("   - EmbeddingAgent")
    print("   - RetrievalAgent")
    print("   - GenerationAgent")

    # Step 2: Initialize planning agent
    print("\n2️⃣  Initializing planning agent...")

    planning_agent = PlanningAgent(
        agent_name="PlanningAgent",
        config={
            "planning_model": "gpt-4",
            "planning_temperature": 0
        }
    )

    # Register task agents with planning agent
    planning_agent.register_agent("EmbeddingAgent", embedding_agent)
    planning_agent.register_agent("RetrievalAgent", retrieval_agent)
    planning_agent.register_agent("GenerationAgent", generation_agent)

    print(f"✅ Planning agent initialized with {len(planning_agent.list_agents())} task agents")

    # Step 3: Load documents (for retrieval)
    print("\n3️⃣  Loading document corpus...")

    doc_csv_path = "./dataset/all_docs_full.csv"
    if os.path.exists(doc_csv_path):
        df = pd.read_csv(doc_csv_path, encoding='utf-8')
        rag_data = df[['ID', 'Content']].dropna()
        rag_data['Content'] = rag_data['Content'].str.lower()

        documents = rag_data['Content'].tolist()
        doc_ids = rag_data['ID'].astype('int').tolist()
        doc_tags = rag_data['ID'].tolist()

        print(f"✅ Loaded {len(documents)} documents")

        # Setup embedding agent
        embedding_agent.run({
            "action": "load_embeddings" if os.path.exists("./dataset/doc_embeddings.npy") else "create_embeddings",
            "documents": documents
        })

        # Setup retrieval agent
        retrieval_agent.run({
            "action": "load_documents",
            "documents": documents,
            "doc_ids": doc_ids,
            "doc_tags": doc_tags
        })

        print("✅ Agents configured with document corpus")
    else:
        print(f"⚠️  Document file not found: {doc_csv_path}")
        print("   Continuing without retrieval capability")

    # Step 4: Example queries
    print("\n" + "="*70)
    print("Example Workflows")
    print("="*70)

    # Example 1: Plan only (no execution)
    print("\n📋 Example 1: Create Plan Only")
    print("-" * 70)

    query1 = "How do I configure a Kubernetes Ingress to expose my service?"

    plan = planning_agent.run({
        "execution_mode": "plan_only",
        "query": query1,
        "context": "User is working with Kubernetes and needs to expose a web service"
    })

    print(f"\nQuery: {query1}")
    print(f"\nPlan Status: {plan.get('status')}")
    print(f"Planning Time: {plan.get('planning_time', 0):.2f}s")

    if plan.get('steps'):
        print("\nExecution Steps:")
        for i, step in enumerate(plan['steps'], 1):
            print(f"  {i}. {step.get('description')}")
            print(f"     Agent: {step.get('agent')}")
            print(f"     Critical: {step.get('critical', False)}")

    if plan.get('reasoning'):
        print(f"\nReasoning: {plan['reasoning']}")

    # Example 2: Plan and execute
    print("\n\n🚀 Example 2: Plan and Execute")
    print("-" * 70)

    query2 = "Explain how Kubernetes Secrets work and show an example configuration"

    result = planning_agent.run({
        "execution_mode": "plan_and_execute",
        "query": query2
    })

    print(f"\nQuery: {query2}")
    print(f"\nOverall Status: {result.get('status')}")
    print(f"Total Time: {result.get('total_time', 0):.2f}s")

    if result.get('plan'):
        print(f"\n📝 Plan created with {len(result['plan'].get('steps', []))} steps")

    if result.get('execution'):
        exec_result = result['execution']
        print(f"\n✅ Execution completed:")
        print(f"   - Execution time: {exec_result.get('execution_time', 0):.2f}s")
        print(f"   - Agents called: {', '.join(exec_result.get('agents_called', []))}")

        print("\n📊 Step Results:")
        for step_result in exec_result.get('results', []):
            print(f"   Step {step_result['step']}: {step_result['status']}")

        # Show final answer if available
        final_result = exec_result.get('results', [])[-1] if exec_result.get('results') else None
        if final_result and final_result.get('result'):
            answer = final_result['result']
            if isinstance(answer, dict) and 'response' in answer:
                print(f"\n💡 Final Answer:")
                print("-" * 70)
                print(answer['response'][:500])  # First 500 chars
                if len(answer['response']) > 500:
                    print("... (truncated)")

    # Example 3: Custom plan execution
    print("\n\n⚙️  Example 3: Execute Custom Plan")
    print("-" * 70)

    custom_plan = {
        "steps": [
            {
                "step": 1,
                "description": "Search for relevant documentation",
                "agent": "RetrievalAgent",
                "input": {
                    "action": "retrieve",
                    "queries": ["kubernetes deployment configuration"],
                    "queries_tags": ["<K8s><Deployment>"],
                    "k": 5,
                    "rerank_k": 3,
                    "embedding_agent": embedding_agent
                },
                "output_key": "retrieved_docs",
                "critical": True
            },
            {
                "step": 2,
                "description": "Generate summary based on retrieved docs",
                "agent": "GenerationAgent",
                "input": {
                    "action": "generate",
                    "title": "Kubernetes Deployment",
                    "body": "Summarize the key concepts",
                    "context": "$context.retrieved_docs"
                },
                "output_key": "summary",
                "critical": False
            }
        ],
        "reasoning": "First retrieve relevant docs, then generate summary"
    }

    print("\nExecuting custom plan with 2 steps...")

    custom_result = planning_agent.run({
        "execution_mode": "execute_plan",
        "plan": custom_plan
    })

    print(f"\nStatus: {custom_result.get('status')}")
    print(f"Execution Time: {custom_result.get('execution_time', 0):.2f}s")
    print(f"Steps Completed: {len(custom_result.get('results', []))}/{len(custom_plan['steps'])}")

    # Show metrics
    print("\n\n📊 Overall Metrics")
    print("-" * 70)
    metrics = planning_agent.get_metrics()
    print(f"Total Planning Time: {metrics['planning_time']:.2f}s")
    print(f"Total Execution Time: {metrics['execution_time']:.2f}s")
    print(f"Total Time: {metrics['total_time']:.2f}s")
    print(f"Agents Called: {set(metrics['agents_called'])}")

    print("\n" + "="*70)
    print("✅ Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

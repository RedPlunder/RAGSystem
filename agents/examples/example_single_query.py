#!/usr/bin/env python
# Example: Single Query with RAG Agents

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import RAGCoordinator


def main():
    """Example of processing a single query using RAG agents."""

    # Set up API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Configuration
    config = {
        "embedding_config": {
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "batch_size": 20,
            "embedding_path": "./dataset/doc_embeddings.npy"
        },
        "retrieval_config": {
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_method": "model",  # or "tags"
            "max_length": 4096
        },
        "generation_config": {
            "model": "gpt-4",
            "temperature": 0,
            "max_retries": 3
        }
    }

    # Initialize coordinator
    print("Initializing RAG Coordinator...")
    coordinator = RAGCoordinator(config=config)

    # Load documents
    print("\nLoading documents...")
    doc_csv_path = "./dataset/all_docs_full.csv"

    if not os.path.exists(doc_csv_path):
        print(f"Error: Document file not found: {doc_csv_path}")
        return

    df = pd.read_csv(doc_csv_path, encoding='utf-8')
    rag_data = df[['ID', 'Content']].dropna()

    # Convert to lowercase for consistency
    rag_data['Content'] = rag_data['Content'].str.lower()

    documents = rag_data['Content'].tolist()
    doc_ids = rag_data['ID'].astype('int').tolist()
    doc_tags = rag_data['ID'].tolist()  # Using ID as tags for simplicity

    print(f"Loaded {len(documents)} documents")

    # Setup RAG system
    print("\nSetting up RAG system...")
    setup_result = coordinator.run({
        "action": "setup",
        "documents": documents,
        "doc_ids": doc_ids,
        "doc_tags": doc_tags,
        "force_recreate": False  # Set to True to recreate embeddings
    })

    print(f"Setup status: {setup_result['status']}")

    # Example query
    print("\n" + "="*60)
    print("Processing Query")
    print("="*60)

    title = "How to configure Kubernetes Ingress?"
    body = "I need to expose my service externally using an Ingress controller."
    tags = "<K8s><Ingress>"

    print(f"\nQuestion Title: {title}")
    print(f"Question Body: {body}")
    print(f"Tags: {tags}")

    # Process query
    print("\nProcessing query...")
    result = coordinator.run({
        "action": "query",
        "title": title,
        "body": body,
        "tags": tags,
        "k": 10,          # Retrieve top 10 initially
        "rerank_k": 4     # Keep top 4 after reranking
    })

    # Display results
    print("\n" + "="*60)
    print("Results")
    print("="*60)

    print(f"\n📄 Retrieved Context IDs: {result['context_ids']}")
    print(f"\n⏱️  Timing:")
    print(f"  - Retrieval: {result['timing']['retrieve_time']:.2f}s")
    print(f"  - Generation: {result['timing']['generation_time']:.2f}s")
    print(f"  - Total: {result['timing']['total_time']:.2f}s")

    print(f"\n🔢 Token Usage:")
    print(f"  - Prompt tokens: {result['token_info']['prompt_tokens']}")
    print(f"  - Completion tokens: {result['token_info']['completion_tokens']}")
    print(f"  - Total tokens: {result['token_info']['total_tokens']}")

    print(f"\n💡 Generated Answer:")
    print("-" * 60)
    print(result['answer'])
    print("-" * 60)

    # Show metrics
    print("\n📊 Overall Metrics:")
    metrics = coordinator.get_metrics()
    token_stats = coordinator.generation_agent.get_token_stats()
    print(f"  - Total retrieval time: {metrics['retrieve_time']:.2f}s")
    print(f"  - Total generation time: {metrics['generation_time']:.2f}s")
    print(f"  - Total tokens used: {token_stats['overall_total']}")


if __name__ == "__main__":
    main()

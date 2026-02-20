#!/usr/bin/env python
# Example: Batch Query Processing with RAG Agents

import os
import sys
import time
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import RAGCoordinator


def main():
    """Example of processing batch queries using RAG agents."""

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
            "rerank_method": "model",
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
    rag_data['Content'] = rag_data['Content'].str.lower()

    documents = rag_data['Content'].tolist()
    doc_ids = rag_data['ID'].astype('int').tolist()
    doc_tags = rag_data['ID'].tolist()

    print(f"Loaded {len(documents)} documents")

    # Setup RAG system
    print("\nSetting up RAG system...")
    start_time = time.time()

    setup_result = coordinator.run({
        "action": "setup",
        "documents": documents,
        "doc_ids": doc_ids,
        "doc_tags": doc_tags,
        "force_recreate": False
    })

    setup_time = time.time() - start_time
    print(f"Setup completed in {setup_time:.2f}s")

    # Process batch queries
    print("\n" + "="*60)
    print("Processing Batch Queries")
    print("="*60)

    input_csv = "./result_structural/latest_input_data_filtered_rewrite_containyaml.csv"
    output_csv = "./result_structural/rag_output.csv"

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        return

    print(f"\nInput CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")

    # Process batch
    batch_start_time = time.time()

    result = coordinator.run({
        "action": "batch_query",
        "csv_path": input_csv,
        "output_path": output_csv,
        "batch_size": 40,
        "limit": 50  # Process first 50 questions for demo
    })

    batch_time = time.time() - batch_start_time

    # Display results
    print("\n" + "="*60)
    print("Batch Processing Results")
    print("="*60)

    print(f"\n✅ Status: {result['status']}")
    print(f"📊 Total processed: {result['total_processed']} questions")
    print(f"💾 Output saved to: {result['output_path']}")

    print(f"\n⏱️  Timing:")
    print(f"  - Setup time: {setup_time:.2f}s")
    print(f"  - Batch processing time: {batch_time:.2f}s")
    print(f"  - Total retrieval time: {result['timing']['retrieve_time']:.2f}s")
    print(f"  - Total generation time: {result['timing']['generation_time']:.2f}s")
    print(f"  - Average time per question: {batch_time / result['total_processed']:.2f}s")

    print(f"\n🔢 Token Usage:")
    token_stats = result['token_stats']
    print(f"  - Total prompt tokens: {token_stats['total_prompt']}")
    print(f"  - Total completion tokens: {token_stats['total_completion']}")
    print(f"  - Overall total tokens: {token_stats['overall_total']}")

    # Calculate cost (example pricing)
    model_pricing = {
        "gpt-4": {"input": 30.00, "output": 60.00},  # per 1M tokens
        "gpt-3.5-turbo": {"input": 1.50, "output": 2.00}
    }

    model = config["generation_config"]["model"]
    if model in model_pricing:
        input_cost = (token_stats['total_prompt'] / 1_000_000) * model_pricing[model]["input"]
        output_cost = (token_stats['total_completion'] / 1_000_000) * model_pricing[model]["output"]
        total_cost = input_cost + output_cost

        print(f"\n💰 Estimated Cost (for {model}):")
        print(f"  - Input cost: ${input_cost:.4f}")
        print(f"  - Output cost: ${output_cost:.4f}")
        print(f"  - Total cost: ${total_cost:.4f}")

    # Show sample results
    print("\n📄 Sample Results (first 3):")
    output_df = pd.read_csv(output_csv, encoding='utf-8')

    for idx, row in output_df.head(3).iterrows():
        print(f"\n--- Question {idx + 1} ---")
        print(f"Title: {row['Question Title'][:60]}...")
        print(f"Context IDs: {row['Context_IDs']}")
        print(f"Response (first 100 chars): {row['Generated_Response'][:100]}...")

    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    main()

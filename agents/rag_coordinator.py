# RAG Coordinator Agent - Orchestrates the entire RAG pipeline
# Coordinates EmbeddingAgent, RetrievalAgent, and GenerationAgent

import time
import pandas as pd
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from .embedding_agent import EmbeddingAgent
from .retrieval_agent import RetrievalAgent
from .generation_agent import GenerationAgent


class RAGCoordinator(BaseAgent):
    """
    Coordinator agent that orchestrates the RAG pipeline.

    This agent manages the workflow:
    1. Initialize document embeddings (EmbeddingAgent)
    2. Load documents into retrieval system (RetrievalAgent)
    3. For each query:
        a. Retrieve relevant documents (RetrievalAgent)
        b. Generate answer using context (GenerationAgent)
    4. Track metrics and save results
    """

    def __init__(
        self,
        agent_name: str = "RAGCoordinator",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config)

        # Initialize sub-agents
        self.embedding_agent = EmbeddingAgent(
            config=self.config.get("embedding_config", {})
        )
        self.retrieval_agent = RetrievalAgent(
            config=self.config.get("retrieval_config", {})
        )
        self.generation_agent = GenerationAgent(
            config=self.config.get("generation_config", {})
        )

        # Performance tracking
        self.metrics = {
            "retrieve_time": 0,
            "generation_time": 0,
            "total_time": 0
        }

    def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for RAG coordinator.

        Args:
            task_input: Dictionary containing:
                - action: "setup", "query", or "batch_query"
                - For setup:
                    - documents: List of document texts
                    - doc_ids: List of document IDs
                    - doc_tags: List of document tags
                    - force_recreate: Whether to recreate embeddings
                - For query:
                    - title: Question title
                    - body: Question body
                    - tags: Question tags
                    - k: Number of docs to retrieve
                    - rerank_k: Number of docs after reranking
                - For batch_query:
                    - csv_path: Path to input CSV
                    - output_path: Path to output CSV
                    - batch_size: Batch size for processing
                    - limit: Limit number of questions

        Returns:
            Dictionary with results based on action
        """
        action = task_input.get("action")

        if action == "setup":
            return self.setup(
                documents=task_input.get("documents", []),
                doc_ids=task_input.get("doc_ids", []),
                doc_tags=task_input.get("doc_tags", []),
                force_recreate=task_input.get("force_recreate", False)
            )

        elif action == "query":
            return self.query(
                title=task_input.get("title", ""),
                body=task_input.get("body", ""),
                tags=task_input.get("tags", ""),
                k=task_input.get("k", 10),
                rerank_k=task_input.get("rerank_k", 4)
            )

        elif action == "batch_query":
            return self.batch_query(
                csv_path=task_input.get("csv_path"),
                output_path=task_input.get("output_path"),
                batch_size=task_input.get("batch_size", 40),
                limit=task_input.get("limit", None)
            )

        else:
            raise ValueError(f"Unknown action: {action}")

    def setup(
        self,
        documents: List[str],
        doc_ids: List[int],
        doc_tags: List[str],
        force_recreate: bool = False
    ) -> Dict[str, Any]:
        """
        Setup the RAG system with documents.

        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            doc_tags: List of document tags
            force_recreate: Whether to recreate embeddings even if they exist

        Returns:
            Setup status
        """
        self.logger.info("Setting up RAG system...")

        # Step 1: Create or load embeddings
        embedding_path = self.embedding_agent.embedding_path

        import os
        if not force_recreate and os.path.exists(embedding_path):
            self.logger.info("Loading existing embeddings...")
            embed_result = self.embedding_agent.run({
                "action": "load_embeddings"
            })
        else:
            self.logger.info("Creating new embeddings...")
            embed_result = self.embedding_agent.run({
                "action": "create_embeddings",
                "documents": documents
            })

        # Step 2: Load documents into retrieval agent
        self.logger.info("Loading documents into retrieval agent...")
        retrieval_result = self.retrieval_agent.run({
            "action": "load_documents",
            "documents": documents,
            "doc_ids": doc_ids,
            "doc_tags": doc_tags
        })

        self.logger.info("RAG system setup complete!")

        return {
            "status": "success",
            "embedding_info": embed_result,
            "retrieval_info": retrieval_result
        }

    def query(
        self,
        title: str,
        body: str,
        tags: str = "",
        k: int = 10,
        rerank_k: int = 4
    ) -> Dict[str, Any]:
        """
        Process a single query through the RAG pipeline.

        Args:
            title: Question title
            body: Question body
            tags: Question tags
            k: Number of documents to retrieve initially
            rerank_k: Number of documents after reranking

        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()

        # Combine title and body for query
        query_text = f"{title}\n{body}"

        # Step 1: Retrieve relevant documents
        retrieve_start = time.time()
        retrieval_results = self.retrieval_agent.run({
            "action": "retrieve",
            "queries": [query_text],
            "queries_tags": [tags],
            "k": k,
            "rerank_k": rerank_k,
            "embedding_agent": self.embedding_agent
        })
        retrieve_end = time.time()
        retrieve_time = retrieve_end - retrieve_start

        # Extract first result (since we only have one query)
        result = retrieval_results[0]

        # Step 2: Merge contexts
        context = " ".join(result["contents"])
        context_ids = result["ids"]

        # Step 3: Generate answer
        generation_start = time.time()
        generation_result = self.generation_agent.run({
            "action": "generate",
            "title": title,
            "body": body,
            "context": context
        })
        generation_end = time.time()
        generation_time = generation_end - generation_start

        total_time = time.time() - start_time

        # Update metrics
        self.metrics["retrieve_time"] += retrieve_time
        self.metrics["generation_time"] += generation_time
        self.metrics["total_time"] += total_time

        return {
            "answer": generation_result["response"],
            "context_ids": context_ids,
            "context": context,
            "token_info": generation_result["token_info"],
            "timing": {
                "retrieve_time": retrieve_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
        }

    def batch_query(
        self,
        csv_path: str,
        output_path: str,
        batch_size: int = 40,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of queries from CSV file.

        Args:
            csv_path: Path to input CSV file
            output_path: Path to output CSV file
            batch_size: Batch size for processing
            limit: Optional limit on number of questions

        Returns:
            Processing statistics
        """
        self.logger.info(f"Processing batch queries from {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path, encoding="utf-8")

        # Add result columns if not present
        result_cols = [
            "Generated_Response",
            "Merged_Contexts",
            "Context_IDs",
            "Total_Tokens"
        ]
        for col in result_cols:
            if col not in df.columns:
                df[col] = ""

        # Apply limit if specified
        if limit:
            df = df.head(limit)

        # Filter unanswered questions
        unanswered_df = df[df["Generated_Response"] == ""]

        self.logger.info(
            f"Processing {len(unanswered_df)} unanswered questions"
        )

        # Process in batches
        for start in range(0, len(unanswered_df), batch_size):
            batch = unanswered_df.iloc[start:start + batch_size]

            title_queries = batch["Question Title"].tolist()
            body_queries = batch["Question Body"].tolist()
            queries_tags = batch["Question Tags"].tolist()

            # Combine title and body
            queries = [
                f"{t}\n{b}"
                for t, b in zip(title_queries, body_queries)
            ]

            # Retrieve documents
            retrieve_start = time.time()
            retrieval_results = self.retrieval_agent.run({
                "action": "retrieve",
                "queries": queries,
                "queries_tags": queries_tags,
                "k": 10,
                "rerank_k": 4,
                "embedding_agent": self.embedding_agent
            })
            retrieve_end = time.time()
            self.metrics["retrieve_time"] += retrieve_end - retrieve_start

            # Prepare contexts
            merged_contexts = [
                " ".join(r["contents"])
                for r in retrieval_results
            ]
            context_ids = [
                ", ".join(map(str, r["ids"]))
                for r in retrieval_results
            ]

            # Generate responses
            self.logger.info("Generating responses...")
            generation_start = time.time()
            generation_result = self.generation_agent.run({
                "action": "generate_batch",
                "title_queries": title_queries,
                "body_queries": body_queries,
                "contexts": merged_contexts
            })
            generation_end = time.time()
            self.metrics["generation_time"] += generation_end - generation_start

            # Extract results
            responses = generation_result["responses"]
            token_infos = generation_result["token_infos"]
            total_tokens_per_query = [
                str(info["total_tokens"])
                for info in token_infos
            ]

            # Update dataframe
            df.loc[batch.index, "Generated_Response"] = responses
            df.loc[batch.index, "Merged_Contexts"] = merged_contexts
            df.loc[batch.index, "Context_IDs"] = context_ids
            df.loc[batch.index, "Total_Tokens"] = total_tokens_per_query

            self.logger.info(
                f"Processed {start + len(batch)} / {len(unanswered_df)} questions"
            )

        # Save results
        df.to_csv(output_path, index=False, encoding="utf-8")
        self.logger.info(f"Results saved to {output_path}")

        # Get token stats
        token_stats = self.generation_agent.get_token_stats()

        return {
            "status": "success",
            "total_processed": len(unanswered_df),
            "output_path": output_path,
            "token_stats": token_stats,
            "timing": self.metrics
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.

        Returns:
            Dictionary with timing metrics
        """
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "retrieve_time": 0,
            "generation_time": 0,
            "total_time": 0
        }
        self.generation_agent.reset_token_stats()

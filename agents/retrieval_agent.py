# Retrieval Agent - Handles document retrieval and re-ranking
# Extracted from demo_noSecond.py

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_agent import BaseAgent


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for document retrieval and re-ranking.

    This agent handles:
    - Initial retrieval using vector similarity (delegates to EmbeddingAgent)
    - Re-ranking using tag-based methods
    - Re-ranking using transformer models (BGE reranker)
    """

    def __init__(
        self,
        agent_name: str = "RetrievalAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config)

        # Configuration with defaults
        self.reranker_model_name = self.config.get(
            "reranker_model",
            "BAAI/bge-reranker-v2-m3"
        )
        self.rerank_method = self.config.get(
            "rerank_method",
            "model"  # or "tags"
        )
        self.max_length = self.config.get("max_length", 4096)

        # Load reranker model if using model-based reranking
        if self.rerank_method == "model":
            self.logger.info(
                f"Loading reranker model: {self.reranker_model_name}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name
            ).eval()
        else:
            self.tokenizer = None
            self.model = None

        # Storage for documents and metadata
        self.documents: List[str] = []
        self.doc_ids: List[int] = []
        self.doc_tags: List[str] = []

    def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for retrieval agent.

        Args:
            task_input: Dictionary containing:
                - action: "load_documents", "retrieve", or "rerank"
                - For load_documents:
                    - documents: List of document texts
                    - doc_ids: List of document IDs
                    - doc_tags: List of document tags
                - For retrieve:
                    - queries: List of query texts
                    - queries_tags: List of query tags
                    - k: Number of initial results
                    - rerank_k: Number of results after reranking
                    - embedding_agent: Reference to EmbeddingAgent
                - For rerank:
                    - queries: List of query texts
                    - results: Initial retrieval results
                    - k: Number of results to keep

        Returns:
            Dictionary with results based on action
        """
        action = task_input.get("action")

        if action == "load_documents":
            return self.load_documents(
                documents=task_input.get("documents", []),
                doc_ids=task_input.get("doc_ids", []),
                doc_tags=task_input.get("doc_tags", [])
            )

        elif action == "retrieve":
            return self.retrieve_documents(
                queries=task_input.get("queries", []),
                queries_tags=task_input.get("queries_tags", []),
                k=task_input.get("k", 10),
                rerank_k=task_input.get("rerank_k", 4),
                embedding_agent=task_input.get("embedding_agent")
            )

        elif action == "rerank":
            return self.rerank(
                queries=task_input.get("queries", []),
                results=task_input.get("results", []),
                k=task_input.get("k", 3)
            )

        else:
            raise ValueError(f"Unknown action: {action}")

    def load_documents(
        self,
        documents: List[str],
        doc_ids: List[int],
        doc_tags: List[str]
    ) -> Dict[str, Any]:
        """
        Load document corpus into memory.

        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            doc_tags: List of document tags

        Returns:
            Status dictionary
        """
        self.documents = documents
        self.doc_ids = doc_ids
        self.doc_tags = doc_tags

        self.logger.info(f"Loaded {len(documents)} documents")

        return {
            "status": "success",
            "num_documents": len(documents)
        }

    def retrieve_documents(
        self,
        queries: List[str],
        queries_tags: List[str],
        k: int = 10,
        rerank_k: int = 4,
        embedding_agent=None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and re-rank documents for queries.

        Args:
            queries: List of query texts
            queries_tags: List of query tags
            k: Number of initial results to retrieve
            rerank_k: Number of results after reranking
            embedding_agent: Reference to EmbeddingAgent for vector search

        Returns:
            List of retrieval results (one per query)
        """
        if embedding_agent is None:
            raise ValueError("EmbeddingAgent reference required for retrieval")

        # Initial retrieval using embeddings
        initial_results = []
        for query in queries:
            search_result = embedding_agent.search(query, k)
            indices = search_result["indices"]

            result = {
                "ids": [self.doc_ids[i] for i in indices],
                "contents": [self.documents[i] for i in indices],
                "tags": [self.doc_tags[i] for i in indices]
            }
            initial_results.append(result)

            self.logger.info(f"Initial top {k} contexts IDs: {result['ids']}")

        # Re-rank results
        if self.rerank_method == "model":
            reranked_results = self.rerank_by_model(
                queries,
                initial_results,
                k=rerank_k
            )
        else:
            reranked_results = self.rerank_by_tags(
                queries_tags,
                initial_results,
                k=rerank_k
            )

        return reranked_results

    def rerank(
        self,
        queries: List[str],
        results: List[Dict[str, Any]],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank existing results.

        Args:
            queries: List of query texts
            results: List of initial results
            k: Number of results to keep

        Returns:
            List of re-ranked results
        """
        if self.rerank_method == "model":
            return self.rerank_by_model(queries, results, k)
        else:
            # For tag-based reranking, we need query tags
            # Extract from results or use empty
            queries_tags = [r.get("query_tag", "") for r in results]
            return self.rerank_by_tags(queries_tags, results, k)

    @staticmethod
    def extract_tags(input_string: str) -> List[str]:
        """
        Extract tags from a string like '<tag1><tag2>'.

        Args:
            input_string: String containing tags

        Returns:
            List of extracted tags
        """
        matches = re.findall(r'<([^>]+)>', input_string)
        return matches

    def rerank_by_tags(
        self,
        queries_tags: List[str],
        results: List[Dict[str, Any]],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results by tag frequency.

        Args:
            queries_tags: List of query tag strings
            results: List of initial results
            k: Number of results to keep

        Returns:
            List of re-ranked results
        """

        def count_matching_tags(tag_list: List[str], tag_str: str) -> int:
            """Count how many tags from tag_list appear in tag_str."""
            return sum(1 for tag in tag_list if tag in tag_str)

        def filter_result_by_tag_list(
            tag_list: List[str],
            result: Dict[str, Any],
            k: int = 3
        ) -> Dict[str, Any]:
            """Filter and sort a single result by tag matching."""
            # Combine ids, contents, tags
            combined = list(zip(
                result["ids"],
                result["contents"],
                result["tags"]
            ))

            # Sort by matching tag count (descending)
            combined.sort(
                key=lambda x: count_matching_tags(tag_list, x[2]),
                reverse=True
            )

            # Take top k
            top_k = combined[:k]

            return {
                "ids": [item[0] for item in top_k],
                "contents": [item[1] for item in top_k],
                "tags": [item[2] for item in top_k]
            }

        new_results = []
        for query_tag, result in zip(queries_tags, results):
            query_tag_list = self.extract_tags(query_tag)
            new_result = filter_result_by_tag_list(query_tag_list, result, k)
            new_results.append(new_result)

        return new_results

    def rerank_by_model(
        self,
        queries: List[str],
        results: List[Dict[str, Any]],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using transformer model.

        Args:
            queries: List of query texts
            results: List of initial results
            k: Number of results to keep

        Returns:
            List of re-ranked results
        """
        new_results = []

        with torch.no_grad():
            for query, result in zip(queries, results):
                scores = []

                # Score each document
                for doc in result["contents"]:
                    # Tokenize query-document pair
                    inputs = self.tokenizer(
                        query,
                        doc,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    # Model inference
                    outputs = self.model(**inputs)

                    # Extract score (logits)
                    score = outputs.logits[0].item()
                    scores.append(score)

                # Sort by score (descending)
                sorted_results = sorted(
                    zip(result["contents"], result["ids"], scores),
                    key=lambda x: x[2],
                    reverse=True
                )
                sorted_results = sorted_results[:k]

                # Create new result
                new_result = {
                    "ids": [item[1] for item in sorted_results],
                    "contents": [item[0] for item in sorted_results],
                    "tags": [self.doc_tags[item[1]] for item in sorted_results],
                    "scores": [item[2] for item in sorted_results]
                }

                new_results.append(new_result)

                self.logger.info(
                    f"Sorted top {k} contexts IDs: {new_result['ids']}, "
                    f"Scores: {new_result['scores']}"
                )

        return new_results

    @staticmethod
    def truncate_text(text: str, max_chars: int = 4096) -> str:
        """
        Truncate text to maximum character length.

        Args:
            text: Text to truncate
            max_chars: Maximum characters

        Returns:
            Truncated text
        """
        return text if len(text) <= max_chars else text[:max_chars]

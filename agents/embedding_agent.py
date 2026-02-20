# Embedding Agent - Handles document vectorization and FAISS indexing
# Extracted from demo_noSecond.py

import os
import numpy as np
import faiss
import asyncio
from typing import List, Optional, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .base_agent import BaseAgent


class EmbeddingAgent(BaseAgent):
    """
    Agent responsible for document embedding and FAISS index management.

    This agent handles:
    - Batch embedding generation using OpenAI API
    - FAISS index creation and management
    - Embedding caching to disk
    """

    def __init__(
        self,
        agent_name: str = "EmbeddingAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config)

        # Configuration with defaults
        self.embedding_model = self.config.get(
            "embedding_model",
            "text-embedding-3-small"
        )
        self.embedding_dim = self.config.get("embedding_dim", 1536)
        self.batch_size = self.config.get("batch_size", 20)
        self.embedding_path = self.config.get(
            "embedding_path",
            "./dataset/doc_embeddings.npy"
        )

        # Initialize FAISS index (will be created when embeddings are loaded)
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None

    def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for embedding agent.

        Args:
            task_input: Dictionary containing:
                - action: "create_embeddings" or "load_embeddings" or "search"
                - documents: List of documents (for create_embeddings)
                - query: Query text (for search)
                - k: Number of results (for search)

        Returns:
            Dictionary with results based on action
        """
        action = task_input.get("action")

        if action == "create_embeddings":
            documents = task_input.get("documents", [])
            return self.create_and_save_embeddings(documents)

        elif action == "load_embeddings":
            return self.load_embeddings()

        elif action == "search":
            query = task_input.get("query")
            k = task_input.get("k", 5)
            return self.search(query, k)

        else:
            raise ValueError(f"Unknown action: {action}")

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(3)
    )
    async def get_batch_embeddings(self, batch_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts using OpenAI API.

        Args:
            batch_texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        import openai

        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        response = await client.embeddings.create(
            model=self.embedding_model,
            input=batch_texts,
        )

        return [item.embedding for item in response.data]

    async def process_all_batches(self, documents: List[str]) -> np.ndarray:
        """
        Process all documents in batches asynchronously.

        Args:
            documents: List of document texts

        Returns:
            Numpy array of embeddings
        """
        tasks = [
            self.get_batch_embeddings(
                documents[i:i + self.batch_size]
            )
            for i in range(0, len(documents), self.batch_size)
        ]

        self.logger.info(f"Submitting {len(tasks)} async embedding tasks")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing failed: {str(result)}")
                # Fill with None for failed batches
                embeddings.extend([None] * self.batch_size)
            else:
                embeddings.extend(result)

        # Filter out None values and stack
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        return np.vstack(valid_embeddings)

    def create_and_save_embeddings(
        self,
        documents: List[str]
    ) -> Dict[str, Any]:
        """
        Create embeddings for documents and save to disk.

        Args:
            documents: List of document texts

        Returns:
            Dictionary with status and embedding info
        """
        self.logger.info(f"Creating embeddings for {len(documents)} documents")

        # Generate embeddings asynchronously
        embeddings = asyncio.run(self.process_all_batches(documents))

        # Save to disk
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        np.save(self.embedding_path, embeddings)

        self.logger.info(f"Embeddings saved to {self.embedding_path}")

        # Store in memory
        self.embeddings = embeddings

        # Create FAISS index
        self.create_faiss_index()

        return {
            "status": "success",
            "num_documents": len(documents),
            "embedding_shape": embeddings.shape,
            "embedding_path": self.embedding_path
        }

    def load_embeddings(self) -> Dict[str, Any]:
        """
        Load existing embeddings from disk.

        Returns:
            Dictionary with status and embedding info
        """
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(
                f"Embedding file not found: {self.embedding_path}"
            )

        self.logger.info(f"Loading embeddings from {self.embedding_path}")
        self.embeddings = np.load(self.embedding_path, mmap_mode='r')

        # Create FAISS index
        self.create_faiss_index()

        return {
            "status": "success",
            "embedding_shape": self.embeddings.shape,
            "embedding_path": self.embedding_path
        }

    def create_faiss_index(self):
        """Create FAISS index from loaded embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded")

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        self.logger.info(
            f"FAISS index created with {self.index.ntotal} vectors"
        )

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text synchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            import openai

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )

            return np.array(response.data[0].embedding)
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return None

    def search(
        self,
        query: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar documents using FAISS.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dictionary with distances and indices
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized. Load embeddings first.")

        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")

        # Search in FAISS index
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        self.logger.info(
            f"Found {len(indices[0])} results for query"
        )

        return {
            "status": "success",
            "distances": distances[0].tolist(),
            "indices": indices[0].tolist()
        }

"""
Local ChromaDB vector database manager for storing and retrieving
ground-truth news articles, fact-checks, and timestamped evidence.
"""

import os
from typing import List, Dict, Any, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class FactVectorStore:
    """
    Local ChromaDB vector database manager for storing and retrieving
    ground-truth news articles, fact-checks, and timestamped evidence.
    """
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "fact_database",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Args:
            persist_directory (str): Path to store ChromaDB indices.
            collection_name (str): Name of ChromaDB collection.
            embedding_model_name (str): Sentence Transformer checkpoint for indexing.
        """
        # Resolve path relative to repository root if not specified
        if persist_directory is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
            self.persist_directory = os.path.join(repo_root, "data/vector_db")
        else:
            self.persist_directory = persist_directory

        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        if chromadb is None:
            raise ImportError("chromadb is required. Install via `uv add chromadb`.")

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        if SentenceTransformer is not None:
            self.encoder = SentenceTransformer(embedding_model_name)
        else:
            self.encoder = None

        # Get or create vector collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Multimodal Fake News Ground Truth Fact Index"}
        )

    def add_facts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Inserts factual text chunks into the vector index with optional metadata (e.g., date, source).

        Args:
            texts (List[str]): List of document chunks or fact sentences.
            metadatas (List[Dict]): List of metadata dicts corresponding to texts.
            ids (List[str]): Unique string IDs for each document chunk.
        """
        if not texts:
            print("[Warning] No texts provided for insertion.")
            return

        # Auto-generate IDs if not supplied
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"fact_{existing_count + i}" for i in range(len(texts))]

        if self.encoder is not None:
            embeddings = self.encoder.encode(texts, show_progress_bar=False).tolist()
        else:
            embeddings = None

        # Clean metadata dictionary values for ChromaDB (ensure primitive types)
        cleaned_metadatas = []
        if metadatas:
            for meta in metadatas:
                cleaned = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        cleaned[k] = v
                    else:
                        cleaned[k] = str(v)
                cleaned_metadatas.append(cleaned)
        else:
            cleaned_metadatas = [{"source": "ground_truth"} for _ in texts]

        # Ingest into collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=cleaned_metadatas,
            ids=ids
        )
        print(f"[Success] Added {len(texts)} facts into collection '{self.collection_name}'. Total items: {self.collection.count()}")

    def query_evidence(
        self,
        query_text: str,
        n_results: int = 3,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Queries the vector index for top-k semantically relevant evidence chunks.

        Args:
            query_text (str): Claim/headline text to query against the vector DB.
            n_results (int): Number of top evidence documents to return.
            source_filter (str, optional): Filter results by specific metadata source.

        Returns:
            Dict containing retrieved documents, distances, and metadata.
        """
        if self.encoder is not None:
            query_embedding = self.encoder.encode([query_text]).tolist()
        else:
            query_embedding = None

        where_clause = {}
        if source_filter:
            where_clause["source"] = source_filter

        results = self.collection.query(
            query_texts=[query_text] if query_embedding is None else None,
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause if where_clause else None
        )

        return {
            "query": query_text,
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0]
        }

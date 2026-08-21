"""
Dynamic RAG Retriever.
Takes claims (text), queries the vector database, and returns formatted, timestamped evidence nodes.
"""

from typing import List, Dict, Any, Optional
from fake_news_mgnn_rag.rag.vector_store import FactVectorStore

class DynamicRAGRetriever:
    def __init__(self, vector_store: FactVectorStore, top_k: int = 3):
        """
        Initializes the retriever with a connection to the ChromaDB vector store.

        Args:
            vector_store (FactVectorStore): Initialized vector database instance.
            top_k (int): Default number of evidence nodes to retrieve per claim.
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve_for_claim(self, claim: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant, timestamped facts for a single claim.

        Args:
            claim (str): The text claim to query.
            top_k (Optional[int]): Number of facts to return. Overrides default.

        Returns:
            List of dictionaries representing RAG nodes (content, timestamp, relevance/distance).
        """
        k = top_k if top_k is not None else self.top_k

        # Query the vector store
        results = self.vector_store.query_evidence(claim, n_results=k)

        nodes = []
        for i, doc in enumerate(results["documents"]):
            dist = results["distances"][i] if results.get("distances") and i < len(results["distances"]) else 1.0
            meta = results["metadatas"][i] if results.get("metadatas") and i < len(results["metadatas"]) else {}

            # Extract timestamp or use a fallback
            timestamp = meta.get("timestamp", "unknown")

            nodes.append({
                "type": "RAG_Fact",
                "content": doc,
                "timestamp": timestamp,
                "relevance_distance": dist,
                "metadata": meta
            })

        return nodes

    def retrieve_batch(self, claims: List[str], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Retrieves relevant, timestamped facts for a batch of claims.
        Useful for Dataloader integration during MGNN training.

        Args:
            claims (List[str]): Batch of text claims.
            top_k (Optional[int]): Number of facts to return per claim.

        Returns:
            List of lists of dictionaries representing RAG nodes for each claim.
        """
        batch_nodes = []
        for claim in claims:
            nodes = self.retrieve_for_claim(claim, top_k)
            batch_nodes.append(nodes)

        return batch_nodes


if __name__ == "__main__":
    import os
    print("Testing DynamicRAGRetriever workflow...")

    # Initialize the vector store
    vector_store = FactVectorStore()

    # Ensure it's populated for the test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    val_json = os.path.join(repo_root, "data/raw/mmfakebench_raw/MMFakeBench_val.json")

    if os.path.exists(val_json):
        # This will add timestamped facts if not already present
        vector_store.populate_from_mmfakebench(val_json, max_samples=100)

    retriever = DynamicRAGRetriever(vector_store, top_k=2)

    sample_claims = [
        "Police officers arrest suspect following car chase in the city.",
        "Scientists discover a new species of deep sea fish."
    ]

    print(f"\nRetrieving evidence for a batch of {len(sample_claims)} claims...\n")
    batch_results = retriever.retrieve_batch(sample_claims)

    for idx, claim in enumerate(sample_claims):
        print(f"--- Claim {idx+1} ---")
        print(f"Text: '{claim}'")
        for j, node in enumerate(batch_results[idx]):
            print(f"  [Evidence {j+1}]")
            print(f"    Content: {node['content']}")
            print(f"    Timestamp: {node['timestamp']}")
            print(f"    Distance: {node['relevance_distance']:.4f}")
        print()

import sys
import os

# Ensure src is in the python path
sys.path.append(os.path.abspath("src"))
from fake_news_mgnn_rag.rag.vector_store import FactVectorStore
from fake_news_mgnn_rag.rag.retriever import DynamicRAGRetriever

def test_real_claim():
    # Initialize the vector store and retriever (assuming default ChromaDB path is set)
    vector_store = FactVectorStore()
    retriever = DynamicRAGRetriever(vector_store)

    # Real-world test cases (can be swapped with current events)
    test_cases = [
        "Tiger Woods leading in the golf tournament final round.",
        "Bernie Sanders Workplace Democracy Act speech on Capitol Hill.",
        "Banned junk food and drinks in school academies."
    ]

    print("--- Evaluating RAG Retriever with Real Claims ---")
    for i, claim in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}] Claim: '{claim}'")

        # Retrieve top 3 facts
        results = retriever.retrieve_for_claim(claim, top_k=3)

        if not results:
            print("  -> No evidence retrieved.")
            continue

        for rank, res in enumerate(results, 1):
            # DynamicRAGRetriever returns dictionaries with 'content' and 'relevance_distance'
            print(f"  [Evidence {rank}] (Distance: {res.get('relevance_distance', 0):.4f})")
            print(f"  Text: {res.get('content', 'N/A')[:150]}...")

if __name__ == "__main__":
    test_real_claim()

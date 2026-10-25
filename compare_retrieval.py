
from storage import load_chunks
from semantic_search import load_embedding_model, create_vector_database, search_chunks
from hybrid_search import search_hybrid
from evaluation import evaluate_retrieval


def main():
    chunks = load_chunks()

    if not chunks:
        print("No chunks available. Run main.py first.")
        return

    model = load_embedding_model()
    collection = create_vector_database(chunks, model)

    test_cases = [
        {"question": "How does supervised learning work?", "relevant_chunk_ids": [2, 3]},
        {"question": "What is classification?", "relevant_chunk_ids": [3]},
        {"question": "What does regression predict?", "relevant_chunk_ids": [3, 4]},
        {"question": "How does unsupervised learning work?", "relevant_chunk_ids": [4]},
        {"question": "What is the difference between training and test data?", "relevant_chunk_ids": [5, 6]},
        {"question": "Why does overfitting cause poor performance on new data?", "relevant_chunk_ids": [7, 8]},
        {"question": "What is underfitting?", "relevant_chunk_ids": [8]},
        {"question": "What happens during data preprocessing?", "relevant_chunk_ids": [9, 10]},
        {"question": "What is feature engineering?", "relevant_chunk_ids": [10, 11]},
        {"question": "Which metrics evaluate classification models?", "relevant_chunk_ids": [12]},
        {"question": "How do recommendation systems suggest items?", "relevant_chunk_ids": [12, 13]},
        {"question": "What is data leakage and why is it dangerous?", "relevant_chunk_ids": [13, 14]},
    ]

    # Evaluate semantic search
    semantic_scores = evaluate_retrieval(
        test_cases, model, collection, k=3
    )

    # Evaluate hybrid search using the same questions and labels
    hybrid_scores = evaluate_hybrid(test_cases, chunks, model, collection, k=3)

    print("\nRetrieval Comparison")
    print("-" * 35)

    for metric in semantic_scores:
        print(f"{metric}")
        print(f"  Semantic: {semantic_scores[metric]:.2f}")
        print(f"  Hybrid:   {hybrid_scores[metric]:.2f}")

    
    print("\nPer-question retrieval diagnostics")
    print("-" * 45)

    for test_case in test_cases:
        question = test_case["question"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        semantic_results = search_chunks(
            question, model, collection, top_k=3
        )

        hybrid_results = search_hybrid(
            question, chunks, model, collection, top_k=3
        )

        semantic_ids = [r["chunk_id"] for r in semantic_results]
        hybrid_ids = [r["chunk_id"] for r in hybrid_results]

        semantic_found = len(set(semantic_ids) & relevant_ids)
        hybrid_found = len(set(hybrid_ids) & relevant_ids)

        print(f"\nQuestion: {question}")
        print(f"Expected relevant chunks: {sorted(relevant_ids)}")
        print(
            f"Semantic: {semantic_ids} "
            f"({semantic_found}/{len(relevant_ids)} relevant)"
        )
        print(
            f"Hybrid:   {hybrid_ids} "
            f"({hybrid_found}/{len(relevant_ids)} relevant)"
        )



def evaluate_hybrid(test_cases, chunks, model, collection, k=3):
    recall_scores = []
    reciprocal_ranks = []

    for test_case in test_cases:
        results = search_hybrid(
            test_case["question"],
            chunks,
            model,
            collection,
            top_k=k
        )

        relevant_ids = set(test_case["relevant_chunk_ids"])
        retrieved_ids = [result["chunk_id"] for result in results]

        found = len(set(retrieved_ids) & relevant_ids)
        recall_scores.append(found / len(relevant_ids))

        rank = next(
            (
                position
                for position, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        f"Recall@{k}": sum(recall_scores) / len(recall_scores),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }


if __name__ == "__main__":
    main()

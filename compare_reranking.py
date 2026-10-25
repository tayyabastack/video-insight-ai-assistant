
from storage import load_chunks
from semantic_search import (
    load_embedding_model,
    create_vector_database,
    search_chunks
)
from reranking import load_reranker, search_with_reranking


TEST_CASES = [
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


def calculate_metrics(test_cases, results_by_question, k=3):
    recalls = []
    reciprocal_ranks = []

    for test_case in test_cases:
        relevant_ids = set(test_case["relevant_chunk_ids"])
        results = results_by_question[test_case["question"]][:k]
        retrieved_ids = [result["chunk_id"] for result in results]

        found = len(set(retrieved_ids) & relevant_ids)
        recalls.append(found / len(relevant_ids))

        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)

    return {
        f"Recall@{k}": sum(recalls) / len(recalls),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }



def main():
    chunks = load_chunks()

    if not chunks:
        print("No chunks available. Run main.py first.")
        return

    embedding_model = load_embedding_model()
    collection = create_vector_database(chunks, embedding_model)

    print("Loading reranker...")
    reranker = load_reranker()

    semantic_results = {}
    candidate_results = {}
    reranked_results = {}

    for test_case in TEST_CASES:
        question = test_case["question"]

        # Retrieve the initial candidates once.
        candidates = search_chunks(
            question,
            embedding_model,
            collection,
            top_k=10,
            max_distance=2.0
        )

        candidate_results[question] = candidates

        # Baseline: rank candidates by semantic similarity.
        semantic_results[question] = candidates[:3]

        # Rerank the same candidates with the cross-encoder.
        reranked_results[question] = search_with_reranking(
            question,
            embedding_model,
            collection,
            reranker,
            candidate_k=10,
            top_k=3
        )

    candidate_scores = calculate_metrics(
        TEST_CASES, candidate_results, k=10
    )
    semantic_scores = calculate_metrics(
        TEST_CASES, semantic_results, k=3
    )
    reranked_scores = calculate_metrics(
        TEST_CASES, reranked_results, k=3
    )

    print("\nRetrieval Comparison")
    print("-" * 40)

    print("\nCandidate retrieval (top 10)")
    for metric, score in candidate_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nSemantic baseline (top 3)")
    for metric, score in semantic_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nReranked results (top 3)")
    for metric, score in reranked_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nPer-question results")
    for test_case in TEST_CASES:
        question = test_case["question"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        before = [
            result["chunk_id"]
            for result in semantic_results[question]
        ]
        after = [
            result["chunk_id"]
            for result in reranked_results[question]
        ]

        print(f"\nQuestion: {question}")
        print(f"Relevant IDs: {sorted(relevant_ids)}")
        print(f"Semantic top 3: {before}")
        print(f"Reranked top 3: {after}")

        # Show the cross-encoder's score for each reranked passage.
        for result in reranked_results[question]:
            print(
                f"  Chunk {result['chunk_id']}: "
                f"score={result['reranker_score']:.4f}"
            )


if __name__ == "__main__":
    main()

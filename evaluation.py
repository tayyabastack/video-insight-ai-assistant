
import csv

from storage import load_chunks
from semantic_search import (
    load_embedding_model,
    create_vector_database,
    search_chunks
)


# Measure retrieval quality for each question
def evaluate_retrieval(test_cases, model, collection, k=3):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not test_cases:
        raise ValueError("At least one test case is required.")

    question_results = []

    for test_case in test_cases:
        question = test_case["question"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError("Each question needs relevant chunk IDs.")

        results = search_chunks(
            question,
            model,
            collection,
            top_k=k,
            max_distance=2.0
        )

        retrieved_ids = [result["chunk_id"] for result in results]

        # Count relevant chunks among the retrieved results
        relevant_found = len(set(retrieved_ids) & relevant_ids)

        recall = relevant_found / len(relevant_ids)

        # Precision measures how many retrieved chunks were relevant
        precision = relevant_found / k

        # Find the rank of the first relevant chunk
        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        question_results.append({
            "question": question,
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": sorted(relevant_ids),
            f"Recall@{k}": recall,
            f"Precision@{k}": precision,
            "Reciprocal Rank": reciprocal_rank
        })

    return question_results


# Calculate overall metrics across all questions
def calculate_summary(question_results, k=3):
    return {
        f"Recall@{k}": sum(
            result[f"Recall@{k}"] for result in question_results
        ) / len(question_results),

        f"Precision@{k}": sum(
            result[f"Precision@{k}"] for result in question_results
        ) / len(question_results),

        "MRR": sum(
            result["Reciprocal Rank"] for result in question_results
        ) / len(question_results)
    }


# Save the detailed results to a CSV file
def save_results(question_results, filename="evaluation_results.csv"):
    if not question_results:
        raise ValueError("There are no evaluation results to save.")

    fieldnames = list(question_results[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in question_results:
            row = result.copy()
            row["retrieved_chunk_ids"] = str(row["retrieved_chunk_ids"])
            row["relevant_chunk_ids"] = str(row["relevant_chunk_ids"])
            writer.writerow(row)


# Run the retrieval evaluation
def main():
    chunks = load_chunks()

    if not chunks:
        print("No chunks available for evaluation.")
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
        {"question": "What is data leakage and why is it dangerous?", "relevant_chunk_ids": [13, 14]}
    ]

    k = 3

    question_results = evaluate_retrieval(
        test_cases, model, collection, k=k
    )

    summary = calculate_summary(question_results, k=k)

    print("\nRetrieval Evaluation Results")
    print("-" * 35)
    print(f"Test questions: {len(test_cases)}")

    for metric, score in summary.items():
        print(f"{metric}: {score:.3f}")

    save_results(question_results)
    print("\nDetailed results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()

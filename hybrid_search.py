from semantic_search import search_chunks
from keyword_search import build_keyword_index, search_keywords


# Combine semantic and keyword rankings using Reciprocal Rank Fusion
def search_hybrid(
    question,
    chunks,
    model,
    collection,
    top_k=3,
    candidate_k=10,
    rrf_constant=60
):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("Result limits must be greater than zero.")

    # Retrieve candidates from both search methods
    semantic_results = search_chunks(
        question,
        model,
        collection,
        top_k=min(candidate_k, collection.count()),
        max_distance=0.6
    )

    keyword_index = build_keyword_index(chunks)

    keyword_results = search_keywords(
        question,
        chunks,
        keyword_index,
        top_k=candidate_k
    )

    # Store each candidate using its source and chunk ID
    combined = {}

    for results in (semantic_results, keyword_results):
        for rank, result in enumerate(results, start=1):
            key = (result["source"], result["chunk_id"])

            if key not in combined:
                combined[key] = result.copy()
                combined[key]["hybrid_score"] = 0.0

            # Give semantic search more influence than keyword search
            weight = 1.5 if results is semantic_results else 1.0

            combined[key]["hybrid_score"] += (
                weight / (rrf_constant + rank)
            )

    # Return the highest-ranked combined results
    ranked_results = sorted(
        combined.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    return ranked_results[:top_k]

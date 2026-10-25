
from sentence_transformers import CrossEncoder
from semantic_search import search_chunks


# Load a model that scores a question and passage together
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Rerank semantic search results using the question
def rerank_chunks(question, model, candidates, top_k=3):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not candidates:
        return []

    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    ranked_results = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        ranked_results.append(result)

    ranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return ranked_results[:top_k]


# Retrieve candidates first, then rerank them


def search_with_reranking(
    question,
    embedding_model,
    collection,
    reranker,
    candidate_k=10,
    top_k=3
):
    # Retrieve the initial candidates.
    candidates = search_chunks(
        question,
        embedding_model,
        collection,
        top_k=candidate_k,
        max_distance=2.0
    )

    # Rerank the candidates and return the best results.
    return rerank_chunks(
        question,
        reranker,
        candidates,
        top_k=top_k
    )


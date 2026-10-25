
import re
from rank_bm25 import BM25Okapi


# Split text into lowercase words for keyword matching
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Prepare the BM25 keyword search index
def build_keyword_index(chunks):
    if not chunks:
        raise ValueError("At least one chunk is required.")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


# Find chunks that share useful keywords with a question
def search_keywords(question, chunks, index, top_k=3):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    scores = index.get_scores(question_tokens)

    ranked_positions = sorted(
        range(len(chunks)),
        key=lambda position: scores[position],
        reverse=True
    )

    matches = []

    for position in ranked_positions:
        if scores[position] <= 0:
            continue

        match = chunks[position].copy()
        match["keyword_score"] = float(scores[position])
        matches.append(match)

        if len(matches) == min(top_k, len(chunks)):
            break

    return matches

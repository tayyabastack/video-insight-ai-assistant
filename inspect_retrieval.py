
from storage import load_chunks
from semantic_search import load_embedding_model, create_vector_database, search_chunks
from keyword_search import build_keyword_index, search_keywords


def main():
    chunks = load_chunks()

    if not chunks:
        print("No chunks available. Run main.py first.")
        return

    model = load_embedding_model()
    collection = create_vector_database(chunks, model)
    keyword_index = build_keyword_index(chunks)

    question = "What happens during data preprocessing?"

    semantic_results = search_chunks(
        question, model, collection, top_k=5
    )

    keyword_results = search_keywords(
        question, chunks, keyword_index, top_k=5
    )

    print(f"\nQuestion: {question}")

    print("\nSemantic search:")
    for rank, result in enumerate(semantic_results, start=1):
        print(f"{rank}. Chunk {result['chunk_id']} — distance {result['distance']:.4f}")

    print("\nKeyword search:")
    for rank, result in enumerate(keyword_results, start=1):
        print(f"{rank}. Chunk {result['chunk_id']} — BM25 {result['keyword_score']:.4f}")


if __name__ == "__main__":
    main()

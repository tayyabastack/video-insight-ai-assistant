
from sentence_transformers import SentenceTransformer
import chromadb


# Load the embedding model
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Create or update the local vector database
def create_vector_database(chunks, model, db_path="vector_db"):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name="video_transcripts",
        metadata={"hnsw:space": "cosine"}
    )

    if not chunks:
        return collection

    # Replace existing chunks from the same sources
    sources = {chunk["source"] for chunk in chunks}

    for source in sources:
        collection.delete(where={"source": source})

    # Prepare the text, embeddings, IDs, and metadata
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        f'{chunk["source"]}::chunk_{chunk["chunk_id"]}'
        for chunk in chunks
    ]

    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    # Store the chunks and their embeddings
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return collection


# Find the most relevant chunks for a question
def search_chunks(question, model, collection, top_k=3, max_distance=0.45):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if max_distance < 0:
        raise ValueError("max_distance cannot be negative.")

    if collection.count() == 0:
        return []

    # Convert the question into an embedding
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Retrieve the closest chunks from the vector database
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        # Keep only chunks within the distance threshold
        if distance <= max_distance:
            matches.append({
                "text": text,
                "source": results["metadatas"][0][i]["source"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "distance": distance
            })

    return matches

def expand_neighboring_chunks(
    matches,
    all_chunks,
    window=1
):
    if window < 0:
        raise ValueError("Window cannot be negative.")

    # Group chunks by source and sort them by chunk ID
    chunks_by_source = {}

    for chunk in all_chunks:
        source = chunk["source"]
        chunks_by_source.setdefault(source, []).append(chunk)

    for source in chunks_by_source:
        chunks_by_source[source].sort(
            key=lambda chunk: chunk["chunk_id"]
        )

    # Collect matching chunks and their neighbors
    expanded = {}

    for match in matches:
        source = match["source"]
        chunk_id = match["chunk_id"]
        source_chunks = chunks_by_source.get(source, [])

        for chunk in source_chunks:
            if abs(chunk["chunk_id"] - chunk_id) <= window:
                key = (source, chunk["chunk_id"])

                if key not in expanded:
                    expanded[key] = chunk.copy()

                    if key == (match["source"], match["chunk_id"]):
                        expanded[key].update(match)

    # Return passages in transcript order
    return sorted(
        expanded.values(),
        key=lambda chunk: (
            chunk["source"],
            chunk["chunk_id"]
        )
    )
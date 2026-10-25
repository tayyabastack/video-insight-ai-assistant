from storage import load_chunks
from semantic_search import load_embedding_model
import chromadb

chunks = load_chunks()
expected_sources = sorted(set(chunk["source"] for chunk in chunks))

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_or_create_collection(
    name="video_transcripts",
    metadata={"hnsw:space": "cosine"}
)

print("Chunks in JSON:", len(chunks))
print("Records in vector database:", collection.count())
print("\nExpected sources:")
for source in expected_sources:
    print(source)

records = collection.get(include=["metadatas"])
stored_sources = sorted(set(item["source"] for item in records["metadatas"]))

print("\nSources in vector database:")
for source in stored_sources:
    print(source)
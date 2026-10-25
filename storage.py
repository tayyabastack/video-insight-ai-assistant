import json
from pathlib import Path


# Load chunk data from a JSON file
def load_chunks(input_file="chunks.json"):
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            print("Error: Chunk file must contain a list.")
            return []

        return chunks

    except FileNotFoundError:
        return []

    except json.JSONDecodeError:
        print("Error: Chunk file contains invalid JSON.")
        return []


# Save chunks while preserving data from other sources
def save_chunks(chunks, output_file="chunks.json"):
    existing_chunks = load_chunks(output_file)

    # Identify the sources being updated
    updated_sources = {chunk["source"] for chunk in chunks}

    # Keep existing chunks belonging to other sources
    preserved_chunks = [
        chunk for chunk in existing_chunks
        if chunk["source"] not in updated_sources
    ]

    # Combine preserved chunks with the new chunks
    all_chunks = preserved_chunks + chunks

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=4, ensure_ascii=False)

    print(f"Saved {len(chunks)} new chunks.")
    print(f"Total chunks stored: {len(all_chunks)}")
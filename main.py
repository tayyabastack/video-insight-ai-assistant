from transcript_processing import load_transcript, chunk_transcript
from media_transcription import transcribe_media, save_transcript
from storage import save_chunks, load_chunks
from pathlib import Path
from hybrid_search import search_hybrid
from semantic_search import (
    load_embedding_model,
    create_vector_database,
    expand_neighboring_chunks
)
from answer_generation import generate_answer


# Run the transcript processing and question-answering workflow
def main():
    print("Video Insight AI Assistant")
    print("1. Use an existing transcript file")
    print("2. Transcribe a local audio/video file or YouTube URL")

    choice = input("Choose an option (1 or 2): ").strip()

    try:
        if choice == "1":
            source = input("Enter transcript filename: ").strip()
            transcript = load_transcript(source)
            source_type = "transcript_file"

        elif choice == "2":
            source = input(
                "Enter a local audio/video path or YouTube URL: "
            ).strip()

            transcript = transcribe_media(source)

            if not transcript:
                print("No speech was detected.")
                return

            transcript_folder = Path("transcripts")
            transcript_folder.mkdir(exist_ok=True)

            if source.startswith(("http://", "https://")):
                filename = source.split("v=")[-1].split("&")[0]
                output_path = transcript_folder / f"{filename}.txt"
            else:
                output_path = (
                    transcript_folder / f"{Path(source).stem}.txt"
                )

            save_transcript(transcript, str(output_path))

            if source.startswith(("http://", "https://")):
                source_type = "youtube_url"
            else:
                source_type = "local_media"

        else:
            print("Invalid choice. Please enter 1 or 2.")
            return

        if not transcript.strip():
            print("The transcript is empty.")
            return

        # Split the transcript into overlapping chunks
        chunks = chunk_transcript(transcript, chunk_size=50)

        print("\nTranscript loaded successfully!")
        print(f"Transcript words: {len(transcript.split())}")
        print(f"Total chunks: {len(chunks)}")

        # Store each chunk with its source information
        chunk_data_list = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_data = {
                "chunk_id": i,
                "text": chunk,
                "source": source,
                "source_type": source_type,
                "source_title": source,
                "start_time": None,
                "end_time": None
            }
            chunk_data_list.append(chunk_data)

        save_chunks(chunk_data_list)
        loaded_chunks = load_chunks()

        print(f"Loaded {len(loaded_chunks)} total chunks from JSON.")

        # Verify that every new chunk was saved correctly
        if not all(chunk in loaded_chunks for chunk in chunk_data_list):
            print("Warning: Some new chunks were not saved correctly.")
            return

        print("Chunk verification successful!")

        # Prepare the transcripts for hybrid search
        print("Loading embedding model...")
        model = load_embedding_model()

        print("Creating vector database...")
        collection = create_vector_database(loaded_chunks, model)

        print("Hybrid search is ready!")

        question = input(
            "\nAsk a question about the transcript: "
        ).strip()

        if not question:
            print("Please enter a question.")
            return

        # Retrieve passages using semantic and keyword search
        matches = search_hybrid(
            question,
            loaded_chunks,
            model,
            collection,
            top_k=10,
            candidate_k=20
        )

        # Keep only passages from the selected source
        matches = [
            match for match in matches
            if match["source"] == source
        ]

        print("\nRetrieved chunk IDs")

        for match in matches:
            print(
                f"Source: {match['source']} | "
                f"Chunk: {match['chunk_id']}"
            )

        # Expand matches with neighboring chunks
        if matches:
            matches = expand_neighboring_chunks(
                matches,
                loaded_chunks,
                window=1
            )

        if not matches:
            print(
                "I couldn't find sufficiently relevant information "
                "in this transcript to answer your question."
            )
            return

        # Display the retrieved passages
        print("\nRelevant passages:")

        for match in matches:
            print(f'\nSource: {match["source"]}')
            print(f'Chunk ID: {match["chunk_id"]}')

            if "distance" in match:
                print(f'Distance: {match["distance"]:.4f}')
            elif "hybrid_score" in match:
                print(f'Hybrid score: {match["hybrid_score"]:.4f}')
            else:
                print(
                    "Score: Not independently measured "
                    "(neighboring chunk)"
                )

            print(f'Text: {match["text"]}')

        # Generate an answer using the retrieved context
        print("\nGenerating answer with Gemini...")

        try:
            answer = generate_answer(question, matches)

            print("\nAnswer:")
            print(answer)

        except Exception as error:
            print("\nCould not generate an answer.")
            print(f"Error details: {error}")
            print("Please try again later.")

    except FileNotFoundError as error:
        print(f"\nFile not found: {error}")

    except Exception as error:
        print("\nThe workflow could not be completed.")
        print(f"Error details: {error}")


if __name__ == "__main__":
    main()
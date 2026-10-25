import re


#Creating a function to read the text transcribed file
def load_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        transcript = file.read()

    return transcript


# Create overlapping chunks from transcript sentences
def chunk_transcript(transcript, chunk_size=50, overlap_sentences=1):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap_sentences < 0:
        raise ValueError("Overlap cannot be negative.")

    # Split the transcript into paragraphs
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in transcript.split("\n")
        if paragraph.strip()
    ]

    chunks = []

    # Process each paragraph separately
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if (
                current_chunk
                and current_word_count + sentence_word_count > chunk_size
            ):
                chunks.append(" ".join(current_chunk))

                current_chunk = (
                    current_chunk[-overlap_sentences:]
                    if overlap_sentences
                    else []
                )

                current_word_count = sum(
                    len(item.split()) for item in current_chunk
                )

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks


# Create transcript chunks while preserving Whisper timestamps
def chunk_timestamped_segments(segments, chunk_size=150, overlap_segments=1):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap_segments < 0:
        raise ValueError("Overlap cannot be negative.")

    chunks = []
    current_segments = []
    current_word_count = 0

    def make_chunk(items):
        return {
            "text": " ".join(item["text"].strip() for item in items),
            "start_time": float(items[0]["start"]),
            "end_time": float(items[-1]["end"]),
        }

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        word_count = len(text.split())

        if current_segments and current_word_count + word_count > chunk_size:
            chunks.append(make_chunk(current_segments))

            overlap = (
                current_segments[-overlap_segments:]
                if overlap_segments
                else []
            )
            current_segments = overlap
            current_word_count = sum(
                len(item["text"].split()) for item in current_segments
            )

        current_segments.append(segment)
        current_word_count += word_count

    if current_segments:
        chunks.append(make_chunk(current_segments))

    return chunks

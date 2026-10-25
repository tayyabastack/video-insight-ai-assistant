import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from answer_generation import generate_answer
from hybrid_search import search_hybrid
from media_transcription import transcribe_media, save_transcript
from reranking import load_reranker, rerank_chunks
from semantic_search import (
    create_vector_database,
    load_embedding_model,
)
from storage import save_chunks
from transcript_processing import (
    chunk_transcript,
    chunk_timestamped_segments,
)


# --------------------------------------------------
# Application setup
# --------------------------------------------------

st.set_page_config(
    page_title="Video Insight AI Assistant",
    page_icon="🎬",
    layout="wide",
)

st.title("Video Insight AI Assistant")
st.write(
    "Transcribe videos, explore their content, and ask "
    "questions using retrieval-augmented generation (RAG)."
)

TRANSCRIPT_DIR = Path("transcripts")
VECTOR_DIR = Path("vector_db_app")
TRANSCRIPT_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

def format_time(seconds):
    if seconds is None:
        return None

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def evidence_heading(item):
    title = (
        item.get("source_title")
        or item.get("source")
        or "Unknown source"
    )
    source = item.get("source", "")
    start = item.get("start_time")
    end = item.get("end_time")
    chunk_id = item.get("chunk_id", "?")

    if start is not None:
        start_label = format_time(start)
        end_label = format_time(end)
        time_range = (
            f"{start_label}–{end_label}"
            if end_label
            else start_label
        )

        if (
            item.get("source_type") == "youtube_url"
            and source.startswith(("https://", "http://"))
        ):
            separator = "&" if "?" in source else "?"
            timestamp_url = f"{source}{separator}t={int(start)}s"
            title = f"[{title} — {time_range}]({timestamp_url})"
        else:
            title = f"{title} · {time_range}"

    return f"**{title} · Chunk {chunk_id}**"

# --------------------------------------------------
# Load models once per Streamlit server session
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_reranker():
    return load_reranker()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "source" not in st.session_state:
    st.session_state.source = ""

if "source_title" not in st.session_state:
    st.session_state.source_title = ""

if "transcript_filename" not in st.session_state:
    st.session_state.transcript_filename = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Process a new transcript or media file
# --------------------------------------------------


def prepare_video(
    transcript,
    source,
    source_title,
    source_type,
    timestamp_segments=None,
):
    if not transcript or not transcript.strip():
        raise ValueError(
            "No transcript text was produced. "
            "Please use a file containing readable text or speech."
        )

    if timestamp_segments:
        chunk_items = chunk_timestamped_segments(
            timestamp_segments,
            chunk_size=150,
            overlap_segments=1,
        )
    else:
        text_chunks = chunk_transcript(
            transcript,
            chunk_size=150,
            overlap_sentences=2,
        )
        chunk_items = [
            {
                "text": text,
                "start_time": None,
                "end_time": None,
            }
            for text in text_chunks
        ]

    if not chunk_items:
        raise ValueError("The transcript could not be split into chunks.")

    chunk_records = []

    for chunk_id, item in enumerate(chunk_items, start=1):
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "text": item["text"],
                "source": source,
                "source_type": source_type,
                "source_title": source_title,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
            }
        )


    # Each source receives its own database directory.
    source_hash = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    db_path = str(VECTOR_DIR / source_hash)

    with st.spinner("Loading the embedding model..."):
        embedding_model = get_embedding_model()

    with st.spinner("Indexing transcript passages in ChromaDB..."):
        collection = create_vector_database(
            chunk_records,
            embedding_model,
            db_path=db_path,
        )

    # Persist the transcript and chunks locally.
    transcript_path = TRANSCRIPT_DIR / f"{source_hash}.txt"
    save_transcript(transcript, str(transcript_path))
    save_chunks(chunk_records)

    st.session_state.transcript = transcript
    st.session_state.chunks = chunk_records
    st.session_state.source = source
    st.session_state.source_title = source_title
    st.session_state.transcript_filename = str(transcript_path)
    st.session_state.collection = collection
    st.session_state.embedding_model = embedding_model
    st.session_state.db_path = db_path
    st.session_state.chat_history = []


# --------------------------------------------------
# Input interface
# --------------------------------------------------

youtube_tab, upload_tab = st.tabs(
    ["YouTube link", "Upload a file"]
)

with youtube_tab:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="Paste a YouTube video link here",
        key="youtube_url",
    )

    process_youtube = st.button(
        "Transcribe YouTube video",
        type="primary",
        use_container_width=True,
    )

    if process_youtube:
        if not youtube_url.strip():
            st.warning("Please enter a YouTube URL.")
        elif not youtube_url.strip().startswith(
            ("https://", "http://")
        ):
            st.warning("Please enter a valid YouTube URL.")
        else:
            try:
                with st.spinner(
                    "Downloading audio and transcribing with Whisper. "
                    "Long videos may take several minutes..."
                ):
                    transcript, timestamp_segments = transcribe_media(
                        youtube_url.strip(),
                        return_segments=True,
                    )

                prepare_video(
                    transcript=transcript,
                    source=youtube_url.strip(),
                    source_title=youtube_url.strip(),
                    source_type="youtube_url",
                    timestamp_segments=timestamp_segments,
                )

                st.success("Video transcribed and indexed successfully.")
                st.rerun()

            except Exception as error:
                st.error(f"Could not process this video: {error}")


with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a transcript, audio file, or video file",
        type=[
            "txt",
            "mp3",
            "wav",
            "m4a",
            "aac",
            "ogg",
            "flac",
            "mp4",
            "mpeg",
            "mpg",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
        help=(
            "TXT files are loaded directly. Audio and video files "
            "are transcribed locally using Whisper."
        ),
    )

    process_upload = st.button(
        "Process uploaded file",
        type="primary",
        use_container_width=True,
    )

    if process_upload:
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            filename = Path(uploaded_file.name).name
            suffix = Path(filename).suffix.lower()

            try:
                timestamp_segments = []
                if suffix == ".txt":
                    transcript = uploaded_file.getvalue().decode(
                        "utf-8-sig"
                    )

                    source = f"upload:{filename}"
                    source_type = "transcript_file"

                else:
                    # Whisper needs a real local file path.
                    temp_path = None

                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix,
                            delete=False,
                        ) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        with st.spinner(
                            "Transcribing your file with local Whisper..."
                        ):
                            transcript, timestamp_segments = transcribe_media(
                                temp_path,
                                return_segments=True,
                            )

                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

                    source = f"upload:{filename}"
                    source_type = "uploaded_media"

                prepare_video(
                    transcript=transcript,
                    source=source,
                    source_title=filename,
                    source_type=source_type,
                    timestamp_segments=timestamp_segments,
                )

                st.success("File processed and indexed successfully.")
                st.rerun()

            except UnicodeDecodeError:
                st.error(
                    "This TXT file could not be decoded as UTF-8. "
                    "Please save it as a UTF-8 text file."
                )

            except Exception as error:
                st.error(f"Could not process this file: {error}")


# --------------------------------------------------
# Transcript preview and download
# --------------------------------------------------

if st.session_state.transcript:
    st.divider()
    st.subheader("Transcript")

    st.caption(
        f"Source: {st.session_state.source_title} · "
        f"{len(st.session_state.transcript.split())} words · "
        f"{len(st.session_state.chunks)} chunks"
    )

    st.download_button(
        "⬇️ Download transcript (.txt)",
        data=st.session_state.transcript,
        file_name="video_transcript.txt",
        mime="text/plain",
        use_container_width=False,
    )

    with st.expander("View full transcript"):
        st.text_area(
            "Extracted transcript",
            value=st.session_state.transcript,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    # --------------------------------------------------
    # RAG question answering
    # --------------------------------------------------

    st.divider()
    st.subheader("Ask questions about this video")

    # Display previous questions and answers.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("evidence"):
                with st.expander("Retrieved evidence"):
                    for item in message["evidence"]:
                        st.markdown(evidence_heading(item))
                        st.write(item["text"])

    # --------------------------------------------------
    # Export conversation and evidence
    # --------------------------------------------------

    if st.session_state.chat_history:
        export_lines = [
            "# Video Insight AI Assistant — Q&A Export",
            "",
            f"Video: {st.session_state.source_title}",
            f"Source: {st.session_state.source}",
            "",
        ]

        for message in st.session_state.chat_history:
            export_lines.append(f"## {message['role'].title()}")
            export_lines.append("")
            export_lines.append(message["content"])
            export_lines.append("")

            if message.get("evidence"):
                export_lines.append("### Retrieved Evidence")
                export_lines.append("")

                for item in message["evidence"]:
                    export_lines.append(evidence_heading(item))
                    export_lines.append("")
                    export_lines.append(item["text"])
                    export_lines.append("")

        st.download_button(
            label="⬇️ Download Q&A and Evidence (.md)",
            data="\n".join(export_lines),
            file_name="video_insight_qa.md",
            mime="text/markdown",
        )

    # --------------------------------------------------
    # Ask a new question
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about the transcript..."
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.status("Processing your question...", expanded=True) as status:
                st.write("Step 1/3: Searching transcript evidence...")

                candidates = search_hybrid(
                    question=question,
                    chunks=st.session_state.chunks,
                    model=st.session_state.embedding_model,
                    collection=st.session_state.collection,
                    top_k=10,
                    candidate_k=20,
                )

                st.write(f"Retrieved {len(candidates)} candidate passages.")

                st.write("Step 2/3: Reranking passages...")

                reranker = get_reranker()
                matches = rerank_chunks(
                    question,
                    reranker,
                    candidates,
                    top_k=3,
                )

                st.write(f"Selected {len(matches)} evidence passages.")

                st.write("Step 3/3: Generating answer...")

                if matches:
                    answer = generate_answer(question, matches)
                else:
                    answer = (
                        "I couldn't find relevant evidence in this transcript. "
                        "Try asking about something discussed in the video."
                    )

                status.update(
                    label="Answer generated successfully",
                    state="complete",
                    expanded=False,
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

                if matches:
                    with st.expander("Retrieved evidence and citations"):
                        for item in matches:
                            st.markdown(evidence_heading(item))
                            st.write(item["text"])

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": matches,
                }
            )

        except Exception as error:
            st.error(
                "The question could not be answered. "
                f"Details: {error}"
            )
            print(f"Question-answering error: {error}")
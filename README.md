# Video Insight AI Assistant

A retrieval-augmented generation (RAG) application that lets users explore YouTube videos and uploaded media through natural-language questions. It transcribes speech, indexes transcript passages, retrieves relevant evidence, and uses Google's Gemini API to generate answers grounded in the source material.

## Features

- **Video and file ingestion:** Process YouTube videos and supported uploaded audio, video, and text files.
- **Local speech transcription:** Use Whisper to transcribe supported media locally.
- **Semantic search:** Generate sentence embeddings with Sentence Transformers and store searchable passages in ChromaDB.
- **Hybrid retrieval:** Combine keyword and semantic retrieval to find relevant transcript passages.
- **Reranking:** Use a cross-encoder to improve the relevance of retrieved passages.
- **Question answering:** Ask questions about a transcript and generate evidence-grounded answers with Gemini.
- **Source evidence:** Display relevant transcript excerpts and source/chunk references; timestamp links are available when timestamp metadata is present.
- **Export:** Download transcript text and save answers or summaries in Markdown where supported.
- **Local application:** Use a Streamlit interface to interact with the pipeline.

## Technology stack

- Python
- Streamlit
- Whisper
- yt-dlp and FFmpeg
- Sentence Transformers
- ChromaDB
- Google Gemini API
- Python-dotenv

## How it works

1. The user provides a YouTube URL or uploads a supported file.
2. Audio/video is transcribed when necessary.
3. The transcript is split into overlapping passages.
4. Sentence Transformers creates embeddings, which are stored in ChromaDB.
5. Hybrid retrieval finds candidate passages, and a cross-encoder reranks them.
6. The relevant evidence is sent to Gemini to generate an answer.
7. The application displays the response alongside the retrieved transcript evidence.

## Setup

### Requirements

- Python 3.12 recommended
- FFmpeg for audio/video processing
- A Google Gemini API key for AI-generated answers

### Installation

Clone your repository and move into its directory:

```bash
git clone https://github.com/tayyabastack/video-insight-ai-assistant.git
cd video-insight-ai-assistant
```

Create and activate a virtual environment on macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg if it is not already available. With Homebrew on macOS:

```bash
brew install ffmpeg
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit your `.env` file or publish your API key.

### Run the application

From the project root, run:

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Project structure

The main modules are organized by responsibility. Exact filenames may vary with the current implementation.

- `app.py` — Streamlit interface and application workflow
- `media_transcription.py` — Media transcription
- `transcript_processing.py` — Transcript cleaning and chunking
- `answer_generation.py` — Gemini-powered answer generation
- `requirements.txt` — Python dependencies
- `.env` — Local API configuration; do not commit
- `README.md` — Project documentation

Other retrieval, storage, evaluation, and helper modules support the pipeline.

## Privacy and limitations

- Whisper transcription runs locally, but Gemini answer generation sends the question and selected transcript passages to Google's API.
- YouTube downloading and transcription depend on the source being accessible and on the installed media tools.
- Generated answers depend on transcript quality and the relevance of retrieved passages.
- Answers may be incomplete when the transcript does not contain enough evidence. Check the displayed evidence before relying on a response.
- Timestamp citations require timestamp metadata to be available for the relevant transcript passages.
- Gemini API availability, quotas, and model access can affect answer generation.

## Future improvements

- More systematic retrieval and answer-quality evaluation
- More robust timestamped citations across all input types
- Improved handling of long videos and transcription failures
- Additional evaluation examples and documented performance measurements

## Project status

Core ingestion, transcript retrieval, and question-answering workflows have been implemented. This repository is intended as a portfolio project demonstrating a practical RAG pipeline; capabilities and limitations should be assessed against the current code.

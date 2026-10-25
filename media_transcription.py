
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from sympy import python
import whisper
import yt_dlp


# Check whether the input is a web URL
def is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# Download YouTube audio to a temporary folder
def download_youtube_audio(url, output_folder):
    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_folder, "audio.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])

    audio_path = Path(output_folder) / "audio.mp3"

    if not audio_path.exists():
        raise FileNotFoundError("Downloaded audio file was not found.")

    return str(audio_path)


# Transcribe a local media file or a YouTube URL
def transcribe_media(media_input, model_size="base", return_segments=False):
    if not media_input.strip():
        raise ValueError("Please provide a file path or URL.")

    # Load Whisper locally on your computer
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    # YouTube audio is temporary; local files are used directly
    if is_url(media_input):
        with tempfile.TemporaryDirectory() as temp_folder:
            audio_path = download_youtube_audio(
                media_input, temp_folder
            )

            print("Transcribing downloaded audio...")
            result = model.transcribe(audio_path)

    else:
        media_path = Path(media_input).expanduser()

        if not media_path.is_file():
            raise FileNotFoundError(
                f"Media file not found: {media_path}"
            )

        print("Transcribing local media...")
        result = model.transcribe(str(media_path))

    transcript = result["text"].strip()

    if return_segments:
        segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
            for segment in result.get("segments", [])
            if segment.get("text", "").strip()
        ]
        return transcript, segments

    return transcript


# Save the transcript as a text file
def save_transcript(transcript, output_path="media_transcript.txt"):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(transcript)

    print(f"Transcript saved to: {output_path}")


# Run this module directly
def main():
    media_input = input(
        "Enter a local audio/video path or YouTube URL: "
    ).strip()

    try:
        transcript = transcribe_media(media_input)

        if not transcript:
            print("No speech was detected.")
            return

        save_transcript(transcript)

        print("\nTranscript preview:")
        print(transcript[:500])

    except Exception as error:
        print("\nTranscription failed.")
        print(f"Error details: {error}")


if __name__ == "__main__":
    main()

def chunk_timestamped_segments(
    segments,
    chunk_size=150,
    overlap_segments=1,
):
    """Group Whisper segments into text chunks while preserving timestamps."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap_segments < 0:
        raise ValueError("Overlap cannot be negative.")

    chunks = []
    current_segments = []
    current_word_count = 0

    for segment in segments:
        text = segment.get("text", "").strip()

        if not text:
            continue

        word_count = len(text.split())

        if (
            current_segments
            and current_word_count + word_count > chunk_size
        ):
            chunks.append({
                "text": " ".join(
                    item["text"].strip()
                    for item in current_segments
                ),
                "start_time": current_segments[0]["start"],
                "end_time": current_segments[-1]["end"],
            })

            if overlap_segments:
                current_segments = current_segments[-overlap_segments:]
            else:
                current_segments = []

            current_word_count = sum(
                len(item["text"].split())
                for item in current_segments
            )

        current_segments.append(segment)
        current_word_count += word_count

    if current_segments:
        chunks.append({
            "text": " ".join(
                item["text"].strip()
                for item in current_segments
            ),
            "start_time": current_segments[0]["start"],
            "end_time": current_segments[-1]["end"],
        })

    return chunks

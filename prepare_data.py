#!/usr/bin/env python3
"""
Data preparation pipeline for Whisper fine-tuning.

Takes raw audio files and produces training-ready datasets:
  Raw audio → transcribe → word-level alignment → sentence-boundary
  segmentation → confidence filtering → training-ready export

Designed for the real-world ASR fine-tuning workflow:
  1. Customer/user provides long-form audio (minutes to hours)
  2. This script segments it into clean 10-30s chunks at sentence boundaries
  3. Output feeds directly into train.py for LoRA fine-tuning

Alignment uses faster-whisper's word-level timestamps (CTC attention weights).
For higher precision, consider ctc-forced-aligner with wav2vec2/MMS models.

References:
  - Trelis: Whisper Data Preparation (https://trelis.substack.com/p/whisper-data-preparation-and-fine)
  - ctc-forced-aligner (https://github.com/MahmoudAshraf97/ctc-forced-aligner)
"""

import os
import json
import argparse
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Sentence boundary detection (multi-script)
# ---------------------------------------------------------------------------

# Punctuation that ends a sentence across scripts
SENTENCE_ENDERS = {
    ".", "?", "!", "...",
    "।",    # Hindi/Devanagari danda
    "॥",    # Hindi double danda
    "।।",   # Double danda (combined)
    "？",   # CJK question mark
    "！",   # CJK exclamation
    "。",   # CJK period
}


def is_sentence_end(word_text):
    """Check if a word ends with sentence-ending punctuation."""
    text = word_text.strip()
    if not text:
        return False
    for ender in SENTENCE_ENDERS:
        if text.endswith(ender):
            return True
    return False


# ---------------------------------------------------------------------------
# Transcription with word-level alignment
# ---------------------------------------------------------------------------

def transcribe_with_alignment(model, audio_path, language=None):
    """Transcribe audio and extract word-level timestamps.

    Uses faster-whisper's built-in word timestamp extraction, which
    relies on CTC attention weights for alignment. Sufficient for
    sentence-boundary segmentation (sub-second precision).

    For sub-20ms phoneme-level precision, use ctc-forced-aligner
    with wav2vec2 or MMS models instead.
    """
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,         # Filter non-speech with Silero VAD
        vad_parameters={
            "min_silence_duration_ms": 500,
        },
    )

    words = []
    for segment in segments:
        if segment.words is None:
            continue
        for word in segment.words:
            words.append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "probability": round(word.probability, 4),
            })

    return words, info


# ---------------------------------------------------------------------------
# Sentence-boundary segmentation
# ---------------------------------------------------------------------------

def segment_at_boundaries(words, audio_array, sr,
                          min_duration=10.0, max_duration=30.0,
                          pause_threshold=0.5):
    """Split word-aligned transcript into training-ready chunks.

    Respects sentence boundaries and natural pauses to produce clean
    segments that start and end at linguistically meaningful points.
    Whisper's positional embeddings handle max 30s, so we enforce that.

    Strategy:
      1. Accumulate words into current segment
      2. At sentence boundaries or long pauses (>pause_threshold):
         - If duration >= min_duration: close segment
      3. At max_duration: force-close regardless
      4. Short trailing segments get merged into the previous one

    Args:
        words: List of word dicts with start/end/word/probability
        audio_array: Mono float32 audio at target sample rate
        sr: Sample rate (should be 16000)
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration (Whisper limit: 30s)
        pause_threshold: Silence duration (s) to consider a split point
    """
    if not words:
        return []

    segments = []
    current_words = []
    seg_start = words[0]["start"]

    for i, word in enumerate(words):
        current_words.append(word)
        duration = word["end"] - seg_start

        # Determine if this is a natural split point
        at_sentence_end = is_sentence_end(word["word"])
        has_long_pause = False
        if i + 1 < len(words):
            gap = words[i + 1]["start"] - word["end"]
            has_long_pause = gap > pause_threshold
        is_last_word = (i == len(words) - 1)

        # Decision logic
        should_split = False
        if duration >= min_duration and (at_sentence_end or has_long_pause):
            should_split = True
        if duration >= max_duration:
            should_split = True  # Hard limit (Whisper positional embeddings)
        if is_last_word and current_words:
            should_split = True

        if should_split:
            seg_end = word["end"]
            text = " ".join(w["word"] for w in current_words)

            # Normalize whitespace
            text = " ".join(text.split())

            # Extract audio chunk with 0.1s padding on each side
            pad = int(0.1 * sr)
            start_sample = max(0, int(seg_start * sr) - pad)
            end_sample = min(len(audio_array), int(seg_end * sr) + pad)
            audio_chunk = audio_array[start_sample:end_sample]

            if len(audio_chunk) > 0 and len(text.strip()) > 0:
                segments.append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "duration": round(seg_end - seg_start, 3),
                    "text": text,
                    "audio": audio_chunk,
                    "word_count": len(current_words),
                    "avg_confidence": round(
                        np.mean([w["probability"] for w in current_words]),
                        4,
                    ),
                })

            # Reset for next segment
            current_words = []
            if i + 1 < len(words):
                seg_start = words[i + 1]["start"]

    return segments


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def filter_segments(segments, min_confidence=0.5, min_duration=1.0,
                    max_duration=30.0, min_words=2):
    """Filter out low-quality segments.

    Removes:
      - Low confidence (likely hallucinations or misalignments)
      - Too short (< min_duration, not enough context for training)
      - Too long (> max_duration, exceeds Whisper's positional limit)
      - Single-word segments (not useful for seq2seq training)
    """
    kept = []
    removed = {
        "low_confidence": 0,
        "too_short": 0,
        "too_long": 0,
        "too_few_words": 0,
    }

    for seg in segments:
        if seg["avg_confidence"] < min_confidence:
            removed["low_confidence"] += 1
        elif seg["duration"] < min_duration:
            removed["too_short"] += 1
        elif seg["duration"] > max_duration:
            removed["too_long"] += 1
        elif seg["word_count"] < min_words:
            removed["too_few_words"] += 1
        else:
            kept.append(seg)

    return kept, removed


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def normalize_text(text):
    """Normalize Unicode text for consistent training.

    Applies NFC normalization (canonical decomposition + canonical
    composition). Important for Devanagari and other complex scripts
    where the same visual character can have multiple representations.
    """
    return unicodedata.normalize("NFC", text).strip()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_dataset(segments, output_dir):
    """Export segments as training-ready dataset.

    Produces:
      - audio/segment_XXXXX.wav: Individual WAV files (16kHz mono)
      - manifest.json: Full metadata for all segments
      - manifest.jsonl: Streaming-friendly line-delimited JSON
      - stats.json: Dataset statistics for quality review
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, seg in enumerate(segments):
        # Normalize text
        text = normalize_text(seg["text"])

        # Save audio
        audio_file = f"segment_{i:05d}.wav"
        sf.write(str(audio_dir / audio_file), seg["audio"], 16000)

        manifest.append({
            "audio": f"audio/{audio_file}",
            "transcription": text,
            "duration": seg["duration"],
            "start": seg["start"],
            "end": seg["end"],
            "word_count": seg["word_count"],
            "avg_confidence": seg["avg_confidence"],
            "source_file": seg.get("source_file", ""),
        })

    # JSON manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # JSONL for streaming
    jsonl_path = output_path / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Statistics
    durations = [s["duration"] for s in manifest]
    confidences = [s["avg_confidence"] for s in manifest]
    stats = {
        "total_segments": len(manifest),
        "total_duration_hours": round(sum(durations) / 3600, 2),
        "mean_duration_s": round(np.mean(durations), 1) if durations else 0,
        "min_duration_s": round(min(durations), 1) if durations else 0,
        "max_duration_s": round(max(durations), 1) if durations else 0,
        "mean_confidence": round(np.mean(confidences), 4) if confidences else 0,
        "total_words": sum(s["word_count"] for s in manifest),
    }
    stats_path = output_path / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return manifest, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw audio for Whisper fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a directory of Hindi audio files
  python prepare_data.py --audio-dir raw_audio/ --output-dir dataset/ --language hi

  # Use CPU with specific chunk sizes
  python prepare_data.py --audio-dir raw_audio/ --output-dir dataset/ \\
      --device cpu --min-duration 15 --max-duration 25

  # Process with stricter quality filtering
  python prepare_data.py --audio-dir raw_audio/ --output-dir dataset/ \\
      --min-confidence 0.7 --language en

Pipeline:
  1. prepare_data.py  →  training-ready dataset
  2. train.py          →  LoRA fine-tuned adapter
  3. convert_and_eval.py → CTranslate2 deployment
        """,
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="Directory containing audio files (wav, mp3, flac, ogg, m4a)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for the prepared dataset",
    )
    parser.add_argument(
        "--model-size", default="large-v3-turbo",
        help="Whisper model for transcription and alignment (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Language code (e.g., 'hi', 'en'). Auto-detect if not specified",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--compute-type", default="int8",
        help="Model quantization: float16, int8, int8_float16 (default: int8)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=10.0,
        help="Minimum segment duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Maximum segment duration in seconds (default: 30.0, Whisper limit)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum average word confidence to keep a segment (default: 0.5)",
    )
    parser.add_argument(
        "--pause-threshold", type=float, default=0.5,
        help="Silence gap (seconds) to consider a segment boundary (default: 0.5)",
    )
    args = parser.parse_args()

    # Validate max duration (Whisper positional embedding limit)
    if args.max_duration > 30.0:
        print("WARNING: max_duration > 30s exceeds Whisper's positional embedding limit.")
        print("         Segments > 30s will produce degraded transcription quality.")

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: audio directory not found: {audio_dir}")
        return

    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    )

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print(f"  Supported formats: {', '.join(audio_extensions)}")
        return

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    # Load transcription model
    print(f"\nLoading Whisper {args.model_size} ({args.compute_type})...")
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    print("Model loaded.")

    # Process each file
    all_segments = []
    for file_idx, audio_file in enumerate(audio_files):
        print(f"\n[{file_idx + 1}/{len(audio_files)}] {audio_file.name}")

        # Load and prepare audio
        audio_array, sr = sf.read(str(audio_file), dtype="float32")

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sr, target_sr=16000,
            )
            sr = 16000

        file_duration = len(audio_array) / sr
        print(f"  Duration: {file_duration:.1f}s ({file_duration/60:.1f} min)")

        # Transcribe with word-level alignment
        print("  Transcribing with word-level alignment...")
        words, info = transcribe_with_alignment(
            model, str(audio_file), language=args.language,
        )
        detected_lang = info.language
        lang_prob = info.language_probability
        print(f"  Language: {detected_lang} (confidence: {lang_prob:.2f})")
        print(f"  Words aligned: {len(words)}")

        if not words:
            print("  WARNING: No words detected, skipping file.")
            continue

        # Segment at sentence boundaries
        segments = segment_at_boundaries(
            words, audio_array, sr,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            pause_threshold=args.pause_threshold,
        )
        print(f"  Segments created: {len(segments)}")

        # Tag with source file
        for seg in segments:
            seg["source_file"] = audio_file.name

        # Preview first segment
        if segments:
            preview = segments[0]
            text_preview = preview["text"][:80]
            print(f"  Preview: [{preview['duration']:.1f}s] \"{text_preview}...\"")

        all_segments.extend(segments)

    if not all_segments:
        print("\nNo segments produced. Check audio files and language setting.")
        return

    print(f"\n{'='*60}")
    print(f"Total segments before filtering: {len(all_segments)}")

    # Quality filtering
    filtered, removed = filter_segments(
        all_segments,
        min_confidence=args.min_confidence,
        min_duration=1.0,
        max_duration=args.max_duration,
    )
    print(f"After filtering: {len(filtered)} segments")
    if any(v > 0 for v in removed.values()):
        print(f"  Removed: {removed}")

    # Export
    print(f"\nExporting to {args.output_dir}/")
    manifest, stats = export_dataset(filtered, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"  Segments:       {stats['total_segments']}")
    print(f"  Total duration: {stats['total_duration_hours']} hours")
    print(f"  Mean duration:  {stats['mean_duration_s']}s")
    print(f"  Duration range: {stats['min_duration_s']}s - {stats['max_duration_s']}s")
    print(f"  Mean confidence:{stats['mean_confidence']}")
    print(f"  Total words:    {stats['total_words']}")
    print(f"\n  Output files:")
    print(f"    {args.output_dir}/manifest.json          (full metadata)")
    print(f"    {args.output_dir}/manifest.jsonl         (streaming format)")
    print(f"    {args.output_dir}/stats.json             (dataset statistics)")
    print(f"    {args.output_dir}/audio/segment_*.wav    (audio segments)")

    print(f"\nNext steps:")
    print(f"  1. Review manifest.json — check transcripts for quality")
    print(f"  2. Correct transcripts manually where needed (biggest WER impact)")
    print(f"  3. Train: python train.py --config configs/your_config.yaml")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge LoRA weights -> CTranslate2 conversion -> faster-whisper evaluation.

Deployment pipeline:
  HF fine-tune (LoRA) -> merge -> ct2-transformers-converter -> faster-whisper
"""

import os
import json
import argparse
import subprocess

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel


DEFAULT_BASE_MODEL = "openai/whisper-large-v3-turbo"


def merge_lora(base_model_id, lora_dir, merged_dir):
    """Merge LoRA adapter back into the base model."""
    print(f"Loading base model: {base_model_id}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float16,
    )
    processor = WhisperProcessor.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print("Merge complete.")
    return merged_dir


def convert_to_ct2(merged_dir, ct2_dir, quantization="int8"):
    """Convert merged model to CTranslate2 format for faster-whisper."""
    cmd = [
        "ct2-transformers-converter",
        "--model", merged_dir,
        "--output_dir", ct2_dir,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
        "--force",
    ]
    print(f"Converting to CTranslate2 ({quantization})...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError("CTranslate2 conversion failed")
    print(f"  Saved to: {ct2_dir}")


def eval_faster_whisper(ct2_dir, device="cuda", device_index=0,
                        compute_type="int8", lang="hi"):
    """Evaluate the converted model with faster-whisper on FLEURS test."""
    from faster_whisper import WhisperModel
    from datasets import load_dataset, Audio
    import evaluate

    print(f"\nLoading faster-whisper model from: {ct2_dir}")
    model = WhisperModel(
        ct2_dir, device=device, device_index=device_index,
        compute_type=compute_type,
    )

    print("Loading FLEURS Hindi test set...")
    ds = load_dataset(
        "google/fleurs", "hi_in", split="test", trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    wer_metric = evaluate.load("wer")
    preds, refs = [], []

    for i, sample in enumerate(ds):
        audio = sample["audio"]["array"].astype("float32")
        segments, info = model.transcribe(audio, language=lang, beam_size=1)
        pred_text = " ".join(seg.text.strip() for seg in segments)
        ref_text = sample["transcription"]
        preds.append(pred_text)
        refs.append(ref_text)

        if i < 3:
            print(f"  [{i}] ref:  {ref_text[:80]}")
            print(f"  [{i}] pred: {pred_text[:80]}")
        if (i + 1) % 100 == 0:
            running_wer = 100 * wer_metric.compute(
                predictions=preds, references=refs
            )
            print(f"  ...{i+1}/{len(ds)} (running WER: {running_wer:.2f}%)")

    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    print(f"\nfaster-whisper WER ({compute_type}): {wer:.2f}% ({len(ds)} samples)")
    return wer


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA, convert to CTranslate2, evaluate with faster-whisper"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--merged-dir", default=None,
                        help="Output for merged model (default: <lora-dir>-merged)")
    parser.add_argument("--ct2-dir", default=None,
                        help="Output for CT2 model (default: <lora-dir>-ct2)")
    parser.add_argument("--quant", default="int8",
                        choices=["float16", "int8", "int8_float16"])
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    base = args.lora_dir.rstrip("/")
    merged_dir = args.merged_dir or f"{base}-merged"
    ct2_dir = args.ct2_dir or f"{base}-ct2"

    if not args.skip_merge:
        merge_lora(args.base_model, args.lora_dir, merged_dir)

    if not args.skip_convert:
        convert_to_ct2(merged_dir, ct2_dir, args.quant)

    if not args.skip_eval:
        wer = eval_faster_whisper(
            ct2_dir, device_index=args.gpu, compute_type=args.quant,
        )

        # Update results.json if it exists
        results_path = os.path.join(args.lora_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            results["faster_whisper_wer"] = round(wer, 2)
            results["ct2_quantization"] = args.quant
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Updated {results_path} with faster-whisper WER")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fine-tune Whisper Large-v3-Turbo on Hindi (FLEURS) with LoRA.

Full pipeline: data prep -> LoRA training -> WER evaluation -> model export.
Optimized for a single A10G GPU (~23GB VRAM).

Hyperparameters sourced from:
- LoRA-Whisper paper (arXiv:2406.06619): r=32, encoder+decoder, all linear layers
- HuggingFace PEFT reference: alpha=2*rank, warmup=50
- Collabora Hindi ASR results: baseline/target WER expectations
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Any

import yaml
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Data collator (handles variable-length audio + label padding)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Cast to model dtype (bf16) — required for generate() during eval
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip leading decoder_start_token if present (trainer prepends it)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare_dataset(example, processor):
    audio = example["audio"]
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["transcription"]).input_ids
    return example


def load_and_prep(processor, dataset_name, dataset_config, split, max_samples=None):
    ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(
        lambda ex: prepare_dataset(ex, processor),
        remove_columns=ds.column_names,
        num_proc=1,
    )
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

wer_metric = evaluate.load("wer")


def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, processor, test_ds, device, lang, task, max_label_len):
    """Run inference on the test set to get WER."""
    model.eval()
    all_preds, all_refs = [], []

    print(f"Evaluating on {len(test_ds)} samples...")
    for i in range(len(test_ds)):
        input_features = torch.tensor(
            test_ds[i]["input_features"]
        ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, language=lang, task=task,
                max_new_tokens=max_label_len,
            )
        pred_text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        label_ids = test_ds[i]["labels"]
        label_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
        all_preds.append(pred_text)
        all_refs.append(label_text)

        if i < 3:
            print(f"  [{i}] ref:  {label_text[:80]}")
            print(f"  [{i}] pred: {pred_text[:80]}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(test_ds)}")

    wer = 100 * wer_metric.compute(predictions=all_preds, references=all_refs)
    return wer, all_preds, all_refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA")
    parser.add_argument("--config", default="configs/fleurs_hindi.yaml")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", default=None, help="Override output dir")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model"]
    lang = cfg["model"]["language"]
    task = cfg["model"]["task"]
    dataset_name = cfg["dataset"]["name"]
    dataset_config = cfg["dataset"]["config"]
    output_dir = args.output or cfg["output"]["dir"]
    max_label_len = cfg["training"]["max_label_length"]

    device = args.device
    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}/{dataset_config}")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to(device)

    # Whisper-specific config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = lang
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    # Load data
    print("\nLoading datasets...")
    train_ds = load_and_prep(processor, dataset_name, dataset_config, "train")
    val_ds = load_and_prep(processor, dataset_name, dataset_config, "validation")
    test_ds = load_and_prep(processor, dataset_name, dataset_config, "test")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Baseline evaluation
    print("\n--- Baseline WER (before fine-tuning) ---")
    baseline_wer, _, _ = evaluate_model(
        model, processor, test_ds, device, lang, task, max_label_len
    )
    print(f"Baseline WER: {baseline_wer:.2f}%\n")

    if args.eval_only:
        return

    # Apply LoRA
    model.enable_input_require_grads()
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator
    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training args
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=eval_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["epochs"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg["bf16"],
        fp16=False,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=eval_cfg["eval_steps"],
        save_steps=eval_cfg["save_steps"],
        save_total_limit=eval_cfg["save_total_limit"],
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=max_label_len,
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        greater_is_better=False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        optim="adamw_torch",
        seed=train_cfg["seed"],
        dataloader_num_workers=2,
    )

    # Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    print("\n--- Starting LoRA fine-tuning ---")
    result = trainer.train()
    print(f"\nTraining done. Final train loss: {result.training_loss:.4f}")

    # Save LoRA adapter
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Final evaluation on test set
    print("\n--- Fine-tuned WER (after training) ---")
    finetuned_wer, preds, refs = evaluate_model(
        model, processor, test_ds, device, lang, task, max_label_len
    )
    print(f"Fine-tuned WER: {finetuned_wer:.2f}%")
    print(f"Improvement: {baseline_wer:.2f}% -> {finetuned_wer:.2f}% "
          f"({baseline_wer - finetuned_wer:.2f} absolute)")

    # Save results
    results = {
        "model": model_id,
        "dataset": f"{dataset_name}/{dataset_config}",
        "lora_config": {
            "r": lora_cfg["r"],
            "alpha": lora_cfg["alpha"],
            "dropout": lora_cfg["dropout"],
            "target_modules": lora_cfg["target_modules"],
        },
        "training": {
            "epochs": train_cfg["epochs"],
            "lr": train_cfg["learning_rate"],
            "batch_size": train_cfg["batch_size"],
            "grad_accum": train_cfg["gradient_accumulation_steps"],
            "warmup": train_cfg["warmup_steps"],
        },
        "baseline_wer": round(baseline_wer, 2),
        "finetuned_wer": round(finetuned_wer, 2),
        "improvement_abs": round(baseline_wer - finetuned_wer, 2),
        "improvement_rel": round(
            (baseline_wer - finetuned_wer) / baseline_wer * 100, 1
        ),
        "test_samples": len(test_ds),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save sample predictions
    preds_path = os.path.join(output_dir, "predictions.txt")
    with open(preds_path, "w") as f:
        for i, (p, r) in enumerate(zip(preds, refs)):
            f.write(f"[{i}] REF:  {r}\n")
            f.write(f"[{i}] PRED: {p}\n\n")


if __name__ == "__main__":
    main()

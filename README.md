# Whisper Large-v3-Turbo Hindi LoRA

Parameter-efficient fine-tuning of [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for Hindi ASR, reducing WER from **35.56% to 22.25%** on FLEURS using LoRA with only 3.33% trainable parameters.

## Results

| Model | WER (%) | Params Trained | Inference Engine |
|-------|---------|----------------|-----------------|
| `whisper-large-v3-turbo` (baseline) | 35.56 | — | HF Transformers |
| **+ LoRA fine-tune (this repo)** | **22.25** | 27.9M (3.33%) | HF Transformers |
| + CTranslate2 INT8 deployment | 22.70 | — | faster-whisper |

**37.4% relative WER reduction** on the FLEURS `hi_in` test set (n=418), evaluated using Whisper-default text normalization. INT8 deployment adds only 0.45% WER degradation.

### Training Curve

| Step | Train Loss | Eval Loss | Eval WER (%) |
|------|-----------|-----------|-------------|
| 50 | 0.263 | 0.259 | 29.40 |
| 100 | 0.210 | 0.234 | 25.74 |
| 150 | 0.145 | 0.223 | 24.49 |
| 200 | 0.148 | 0.217 | 23.43 |
| 250 | 0.146 | 0.213 | 23.82 |
| **300** | **0.096** | **0.215** | **22.42** |
| 350 | 0.109 | 0.215 | 22.50 |

Best checkpoint loaded from step 300 (lowest val WER). Final test WER: **22.25%**.

## Approach

### Why LoRA over full fine-tuning

- **3.33% trainable parameters** (27.9M of 809M) — adapter is 107MB vs 1.6GB full model
- Single A10G GPU (23GB), ~45 min training — no multi-GPU setup needed
- Base model capabilities preserved: multilingual detection, timestamp generation

### Hyperparameter justification

All choices are research-backed:

- **r=32, alpha=64**: Following [LoRA-Whisper (arXiv:2406.06619)](https://arxiv.org/abs/2406.06619) which found encoder+decoder targeting on all linear layers outperforms decoder-only or q/v-only configurations
- **Target modules: q, k, v, out, fc1, fc2**: Full attention + FFN layers in both encoder and decoder, per the same paper's recommendation for maximum WER reduction
- **lr=1e-4 with linear schedule**: Conservative for a large pretrained model; warmup=50 prevents early catastrophic forgetting
- **Effective batch=16** (4 x 4 grad accum): Adapts the Collabora Hindi ASR effective batch size for single-GPU constraints
- **3 epochs on ~2K samples**: Standard for small fine-tuning datasets, with eval every 50 steps to catch overfitting early

### Dataset

[Google FLEURS](https://huggingface.co/datasets/google/fleurs) Hindi (`hi_in`):

| Split | Samples | Duration |
|-------|---------|----------|
| Train | 2,120 | ~3.5h |
| Validation | 239 | ~0.4h |
| Test | 418 | ~0.7h |

- Read speech from Wikipedia sentences, 16kHz mono, Devanagari script
- License: CC BY 4.0

## Quick Start

### Training

```bash
pip install -r requirements.txt

# Single GPU training (~45 min on A10G)
python train.py --config configs/fleurs_hindi.yaml --device cuda:0
```

### Deployment Pipeline

```
LoRA Adapter (107MB)
    ↓ merge_and_unload()
Merged HF Model (1.6GB)
    ↓ ct2-transformers-converter --quantization int8
CTranslate2 Model (~800MB)
    ↓ faster_whisper.WhisperModel()
Production Inference (4x faster, 50% less VRAM)
```

```bash
# Full pipeline: merge → convert → evaluate
python convert_and_eval.py --lora-dir outputs/whisper-large-v3-turbo-hindi-lora --quant int8 --gpu 0
```

### Inference with PEFT

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base_model, "Tachyeon/whisper-large-v3-turbo-hindi-lora")
model = model.to("cuda").eval()

# audio_array: 16kHz float32 numpy array
input_features = processor(
    audio_array, sampling_rate=16000, return_tensors="pt"
).input_features.to("cuda", dtype=torch.bfloat16)

with torch.inference_mode():
    predicted_ids = model.generate(input_features, language="hi", task="transcribe")

print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha | 64 (2x rank) |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` |
| Trainable Parameters | 27,852,800 / 836,730,880 (3.33%) |
| Bias | none |

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `openai/whisper-large-v3-turbo` (809M params) |
| Epochs | 3 |
| Learning Rate | 1e-4 (linear decay) |
| Warmup Steps | 50 |
| Batch Size | 4 (x4 grad accum = effective 16) |
| Optimizer | AdamW (weight_decay=0.01) |
| Precision | BFloat16 |
| Gradient Checkpointing | Enabled |
| Hardware | NVIDIA A10G (23GB) |
| Training Time | 45 min |
| Seed | 42 |

## Project Structure

```
whisper-hindi-lora/
├── train.py               # LoRA fine-tuning pipeline
├── convert_and_eval.py    # Merge → CTranslate2 → faster-whisper eval
├── configs/
│   └── fleurs_hindi.yaml  # All hyperparameters
├── results/
│   └── results.json       # Full metrics
├── requirements.txt
└── LICENSE                # Apache-2.0
```

## Normalization Note

Hindi ASR evaluation is sensitive to text normalization. Whisper's default normalizer simplifies Devanagari conjuncts, which can inflate apparent accuracy. All WER numbers reported here use **Whisper-default normalization** for comparability with existing HuggingFace models. For production Hindi ASR, consider evaluation with [IndicNLP normalizer](https://github.com/anoopkunchukuttan/indic_nlp_library).

## Limitations

- **Data scope**: Trained on FLEURS read speech (~3.5h). Performance on conversational, noisy, or code-mixed (Hinglish) audio is not evaluated.
- **Language detection**: Fine-tuning on a single language can degrade Whisper's multilingual detection. Set `language="hi"` explicitly at inference.
- **Normalization**: Reported WER uses Whisper normalization; Indic-normalized WER will differ.

## References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [LoRA-Whisper: Parameter-Efficient and Extensible Multilingual ASR](https://arxiv.org/abs/2406.06619) (Yang et al., 2024)
- [FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech](https://huggingface.co/datasets/google/fleurs) (Conneau et al., 2023)
- [Collabora: Fine-tuning Whisper for Hindi](https://www.collabora.com/news-and-blog/news-and-events/breaking-language-barriers-fine-tuning-whisper-for-hindi.html)

## License

Apache-2.0

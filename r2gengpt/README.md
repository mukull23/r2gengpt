# R2GenGPT re-implementation (IU-Xray and MIMIC-CXR)

A compact re-implementation of **R2GenGPT: Radiology Report Generation with frozen LLMs**
(Wang et al., *Meta-Radiology* 2023), written to be easy to read and to extend for the
final-year project. The official code is at https://github.com/wang-zhanyu/R2GenGPT.

```
X-ray views ─► Swin-B ─► visual mapper (Linear + LayerNorm) ─► visual tokens ─┐
                                                                               ▼
"Human: <Img>[visual tokens]</Img> Generate a comprehensive ... \nAssistant:" ─► Llama-2-7B (frozen) ─► report
```

| `--align` | Trains | Trainable params (checked) | Paper result (IU-Xray BLEU-4 / CIDEr) |
|---|---|---|---|
| `shallow` | visual mapper only | 4.21 M | 0.156 / 0.405 |
| `delta`   | mapper + LoRA (r=16, α=16) on Swin query/value | 4.98 M | 0.162 / 0.419 |
| `deep`    | mapper + the whole Swin encoder | 90.95 M | 0.173 / 0.438 |

The loss is next-token prediction on the report tokens only. Visual tokens and prompt tokens are masked.
With default flags the model is the paper's model; every extension below is opt-in.

## Layout

| File | Purpose |
|---|---|
| `r2gen/data.py` | IU-Xray and MIMIC-CXR loaders. They use the authors' `annotation.json` splits and R2Gen's report cleaning. IU-Xray indications come from the NLM XML reports. |
| `r2gen/model.py` | The R2GenGPT model: alignment modes, prompt building (with left-padding for per-study prompts), loss and beam-search generation. |
| `r2gen/mappers.py` | Visual mappers: linear (paper), MLP, Perceiver resampler; visual-token pooling. |
| `r2gen/builder.py` | Command-line options and model construction from Hugging Face checkpoints. |
| `r2gen/labeler.py` | A rule-based CheXpert-style labeler (14 observations, with negation and uncertainty). |
| `r2gen/metrics.py` | BLEU-1..4, METEOR, ROUGE-L, CIDEr; clinical efficacy (P/R/F1); normal-vs-abnormal breakdown; hallucination statistics; optional CheXbert and RadGraph F1. |
| `r2gen/grounding.py` | Sentence-to-region saliency (LLM attention or gradient × input). |
| `r2gen/stats.py` | Seed aggregation and the paired bootstrap test. |
| `train.py` / `evaluate.py` / `infer.py` | Train, score a test split, and generate a report for your own images. |
| `app.py` | A local web page: choose an X-ray (frontal, optional lateral), get the report and predicted findings. |
| `explain.py` | Heatmaps of the image regions behind each generated sentence. |
| `benchmark.py` | Latency, throughput, peak memory and parameter counts. |
| `ablate.py` / `significance.py` | Multi-seed ablation grids, and significance testing between two runs. |
| `smoke_test.py`, `tests/` | Offline checks with tiny random models (every script runs end to end), plus unit tests. |

The dataset is read from `../dataset/<dataset>` by default; pass `--data_dir` to change it.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests -q     # unit tests, a few seconds
python smoke_test.py          # about 2 minutes, no downloads (--quick: model checks only)
```

METEOR needs Java (`brew install openjdk` on macOS). Without Java it is skipped and the other metrics still run.

Llama-2 is gated. Accept the license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf,
then run `hf auth login` (`huggingface-cli login` on older versions).

## Training

The paper used 4× A100 40GB. Choose the line that fits your hardware:

```bash
# Paper setting (delta). Needs about 40GB of GPU memory in bf16.
python train.py --align delta --run_name iu_delta

# Single 24GB GPU (RTX 3090/4090, Colab L4/A100): 4-bit LLM
python train.py --align delta --load_in_4bit --run_name iu_delta_4bit

# Small GPU / Apple silicon / quick experiments: a 1.1B open LLM with no gating
python train.py --align delta --llm_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --run_name iu_delta_tinyllama
```

Each epoch generates reports for the validation split and saves `runs/<name>/best.pt` (best validation
BLEU-4; change it with `--select_metric`). The checkpoint holds only the trained weights, so it is a few MB
for shallow and delta. For faster epochs, add `--val_limit 100`. On Apple MPS the frozen LLM is loaded in
float16 (`--llm_dtype float32` to override).

### MIMIC-CXR

MIMIC-CXR needs credentialed PhysioNet access (https://physionet.org/content/mimic-cxr-jpg/2.0.0/). Put the
R2Gen-format package (`annotation.json` from the R2GenGPT repository, plus `images/`) in
`../dataset/mimic_cxr/`, then:

```bash
python train.py --dataset mimic_cxr --align deep --run_name mimic_deep   # 3 epochs, as in the paper
```

The MIMIC defaults are 3 epochs, 100 training report tokens and 150 generated tokens. Entries may carry an
`"indication"` field, which `--prompt_context indication` uses.

## Evaluation

```bash
python evaluate.py --checkpoint runs/iu_delta/best.pt
python infer.py --checkpoint runs/iu_delta/best.pt --images front.png side.png
```

To try a model in the browser, run `python app.py` and open http://127.0.0.1:7860. It uses
`runs/iu_delta_tinyllama_disease_llmlora/last.pt` by default; pass `--checkpoint` to use another model. The page
runs only on this machine.

`evaluate.py` writes `test_metrics.json` and `test_predictions.json` next to the checkpoint:

- `overall`: BLEU-1..4, METEOR, ROUGE-L, CIDEr; clinical efficacy `CE_precision/recall/F1` (micro, as in the
  paper) plus macro and example-based versions; hallucination statistics.
- `per_label`: precision, recall, F1 and support for each of the 14 CheXpert observations.
- `subsets`: every metric again on studies whose reference is normal and on those that are abnormal.
- `setup`: the checkpoint, datasets and generation settings.

Clinical labels come from the built-in rule-based labeler by default. It is an approximation of the CheXpert
labeler, good for comparing your own runs. For numbers comparable with the literature, use CheXbert, and add
RadGraph F1 if you want it (both download their weights on first use):

```bash
pip install f1chexbert radgraph
python evaluate.py --checkpoint runs/iu_delta/best.pt --labeler chexbert --radgraph
```

Generation settings can be overridden at evaluation time, e.g. `--num_beams 1 --tag _beam1`.

## Extensions for the paper's research gaps

The numbers refer to the gap tables in the top-level README. Each extension is a flag or a script.

| # | Gap | What to run |
|---|---|---|
| 1 | Encoder pretrained on natural images | `--vision_model microsoft/rad-dino` (X-ray-pretrained DINOv2). LoRA targets and image normalisation adapt to the encoder automatically. |
| 2 | LLM fully frozen / general-domain | `--llm_lora` (LoRA on the LLM's q/v projections); `--llm_model <any causal LM>` |
| 3 | Single linear mapper, fixed token count | `--mapper mlp` or `--mapper perceiver`; `--visual_tokens N` (grid pooling for linear/MLP, query count for the Perceiver) |
| 4 | 224×224 input | `--image_size 384 --vision_model microsoft/swin-base-patch4-window12-384`, or RAD-DINO at 518 |
| 5 | Views averaged | `--view_fusion concat` (both views' tokens, with learned view embeddings) |
| 6 | One fixed prompt | `--prompt_context indication` (the clinical indication goes into the prompt); `--disease_prompt` (a classifier head predicts the 14 labels and the prompt lists them); `--prompt "..."` |
| 7 | Bias toward normal cases untested | `evaluate.py` → `subsets` (normal vs abnormal) and `per_label`; `abnormal_reports_called_normal` |
| 8 | Word-overlap metrics only | Clinical efficacy on IU-Xray too, hallucination/omission rates, `--labeler chexbert`, `--radgraph` |
| 9 | No explainability | `python explain.py --checkpoint ... --study CXR2384_IM-0942 [--method gradient]` |
| 10 | No cross-dataset testing | `python evaluate.py --checkpoint runs/mimic_deep/best.pt --dataset iu_xray` |
| 11 | Deployment cost unknown | `python benchmark.py --checkpoint ... [--load_in_4bit] [--num_beams 1]` |
| 12 | No ablations or significance tests | `python ablate.py --name lora_rank --vary lora_r=4,16,64 --seeds 1 2 3 -- --align delta`; `python significance.py A_predictions.json B_predictions.json` |

How the extensions work:

- **Disease-guided prompt.** A linear head on the pooled encoder features is trained with a BCE loss on
  labels from the reference report (`--disease_loss_weight`). During training the prompt lists the
  reference findings (teacher forcing). At test time it lists the head's predictions (probability > 0.5).
  `infer.py` prints them.
- **Per-study prompts** (indication, findings) have different lengths. They are left-padded, and position
  ids follow the attention mask. The smoke test checks that each study gets the same logits and the same
  generation in a batch as on its own.
- **Grounding.** `explain.py` scores each sentence's tokens against the visual tokens, using LLM attention
  averaged over layers and heads, or |gradient × input|. It maps the scores back to the encoder's patch grid
  (through the pooling and Perceiver attention), then overlays them on each view. With `--view_fusion mean`,
  both views share one map, because the model only sees their average.
- **Ablations.** Training settings get one run per seed. Generation settings (`num_beams`, `length_penalty`,
  ...) reuse those runs at evaluation. Finished runs are skipped on re-runs. The results go to
  `runs/ablations/<name>/summary.md` as mean ± std.
- **Significance.** A paired bootstrap over test studies for BLEU-4 (exact corpus BLEU, identical to
  coco-caption), ROUGE-L, CIDEr and clinical F1. It reports the 95% CI of the difference and a two-sided p.

Not covered:
- Prior-study (longitudinal) modelling. IU-Xray has no priors, and MIMIC-CXR priors would need
  patient-level linking.
- Multi-scale feature fusion inside the encoder. Higher-resolution input is supported through `--image_size`.
- A radiologist reader study.

## Differences from the official code

- A plain PyTorch training loop (AdamW with cosine LR) instead of PyTorch Lightning.
- `best.pt` is chosen by validation BLEU-4.
- The mapper's LayerNorm gain starts at the RMS of the LLM's word embeddings instead of 1. With gain 1 the
  visual tokens are ~70× larger than word embeddings, and the first-step loss on IU-Xray is 9.7 instead of 3.1.
  Over 30 steps with TinyLlama, the scaled start also ended slightly lower (2.43 vs 2.52).
- Empty sentences are dropped during report cleaning.
- Studies with one view repeat that view, so batches have a fixed shape.

Expect small differences from the paper's numbers.

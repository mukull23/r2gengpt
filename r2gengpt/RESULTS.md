# Experiment results (IU-Xray, TinyLlama-1.1B)

All runs use delta alignment (Swin-B + LoRA r=16 on query/value). Each was trained for 15 epochs with lr 1e-4 and
batch 6, on an Apple M5 Pro (MPS, float16 LLM), with one seed (42). `best` is the epoch with the highest BLEU-4 on
100 validation studies. Test scores cover all 590 IU-Xray test studies; beam search uses 3 beams.
Clinical labels come from the built-in rule-based labeler (`r2gen/labeler.py`), an approximation of CheXpert.

| Run | Directory | Change from the paper's setup |
|---|---|---|
| A. Baseline | `runs/iu_delta_tinyllama` | none (frozen LLM) |
| B. LoRA only | `runs/iu_delta_tinyllama_llmlora` | `--llm_lora` |
| C. Disease prompt + LoRA | `runs/iu_delta_tinyllama_disease_llmlora` | `--disease_prompt --llm_lora` |
| (stopped) Disease prompt, frozen LLM | `runs/iu_delta_tinyllama_disease_frozen_llm` | `--disease_prompt`; findings F1 stayed at 0 for 3 epochs |

## Test results

"Findings" means the 13 CheXpert findings, without "No Finding".

| Run | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | Findings P / R / F1 | Abnormal called normal | Reports with hallucinated finding |
|---|---|---|---|---|---|---|---|
| A best (ep 7) | **0.503** | **0.186** | **0.389** | 0.371 | 0 / 0 / 0 | 100% | 0% |
| A last (ep 15) | 0.461 | 0.169 | 0.379 | 0.451 | 0 / 0 / 0 | 100% | 0% |
| B best (ep 2) | 0.484 | 0.161 | 0.367 | 0.297 | 0 / 0 / 0 | 100% | 0% |
| B last (ep 15) | 0.457 | 0.152 | 0.357 | 0.437 | 0.50 / 0.005 / 0.010 | 99% | 0.2% |
| C best (ep 12) | 0.447 | 0.148 | 0.352 | 0.469 | 0.16 / 0.16 / 0.16 | 57% | 19% |
| C last (ep 15) | 0.451 | 0.157 | 0.358 | **0.515** | **0.20 / 0.22 / 0.21** | **54%** | 19% |
| Paper, delta (Llama-2-7B) | 0.470 | 0.162 | 0.369 | 0.419 | – | – | – |

Per-finding F1 for C last: cardiomegaly 0.35, lung opacity 0.28, pleural effusion 0.22, atelectasis 0.18,
support devices 0.03. A and B score 0 on every finding.

## Significance (paired bootstrap, 1,000 resamples, epoch-15 checkpoints)

Each row is the change from the first run to the second (B − A, then C − B).

| Comparison | BLEU-4 | ROUGE-L | CIDEr | Findings F1 |
|---|---|---|---|---|
| LoRA effect (A → B) | −0.017 (p=0.07) | −0.022 (p=0.002) | −0.014 (p=0.76) | +0.010 (p=0.69) |
| Disease-prompt effect (B → C) | +0.004 (p=0.63) | +0.001 (p=0.90) | +0.079 (p=0.27) | **+0.200 (p=0.002)** |

Files: `runs/significance_lora_effect.json` and `runs/significance_disease_prompt_effect.json`. The earlier comparison
of A best with C last is in `runs/significance_baseline_vs_disease_llmlora.json`.

## Findings

1. **The baseline beats the paper's language scores by writing normal templates.** Run A matches or beats the
   paper's delta model on BLEU and ROUGE-L. Yet it calls all 144 abnormal test studies normal and uses only 86
   distinct reports. About 76% of IU-Xray test studies are normal, so word-overlap metrics reward this (research
   gaps 7 and 8).
2. **LoRA on the LLM alone does not help.** It lowers the training loss (0.78 vs 1.36) but still writes normal
   reports. Findings F1 does not change significantly.
3. **The disease prompt is what makes the model report findings, but only with a trainable LLM.** With a frozen
   LLM the predicted findings in the prompt were ignored. With LoRA, findings F1 rises from 0.01 to 0.21
   (p=0.002), and the share of abnormal studies called normal falls from 99% to 54%. BLEU-4 and ROUGE-L do not
   change significantly between B and C.
4. **The cost is hallucination.** 19% of C's reports assert a finding the reference lacks, against 0% for A and B.

## Caveats

- One seed per run. `ablate.py --seeds 1 2 3` would give mean ± std.
- Clinical metrics use the rule-based labeler. For literature-comparable numbers, re-score with
  `evaluate.py --labeler chexbert`.
- METEOR is missing because Java was not installed.
- TinyLlama (1.1B) replaces Llama-2-7B, so absolute numbers are not directly comparable with the paper.
- Checkpoints were selected by validation BLEU-4, which favours normal templates. `--select_metric CE_F1_findings`
  would choose differently.

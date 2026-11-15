# R2GenGPT: Radiology Report Generation with Frozen LLMs

> **Base paper for this final-year project.**
> Zhanyu Wang, Lingqiao Liu, Lei Wang, Luping Zhou.
> *Meta-Radiology* 1 (2023) 100033. Published online 15 November 2023.
> DOI: [10.1016/j.metrad.2023.100033](https://doi.org/10.1016/j.metrad.2023.100033) · arXiv: 2309.09812
> Code: https://github.com/wang-zhanyu/R2GenGPT
> Local copy: [1-s2.0-S2950162823000334-main.pdf](1-s2.0-S2950162823000334-main.pdf)

---

## 1. In one paragraph

R2GenGPT writes a radiology report from a chest X-ray. It takes a large language model (LLM), **Llama-2-7B**, and **keeps it completely frozen**. A Swin Transformer turns the X-ray into features, and a single linear layer (the *visual mapper*) projects them into the LLM's word-embedding space. These "visual tokens" go into the LLM together with an instruction prompt, and the LLM writes the report.

Only the small alignment module is trained. In the *delta* setting that is about **5M parameters (0.07% of the model)**. Even so, the model matches or beats the specialised report-generation models of its time on IU-Xray and MIMIC-CXR, and it trains faster than them.

---

## 2. Problem and motivation

- The volume of radiology imaging is growing faster than the number of radiologists. The result is longer hours, reading fatigue and more diagnostic errors. Automatic report generation (**R2Gen**) could take a draft-writing burden off radiologists.
- R2Gen is harder than image captioning for two reasons:
  1. **Long text.** A report is a coherent multi-sentence paragraph, not a single caption.
  2. **Data bias.** Datasets are dominated by normal cases, so models tend to miss abnormalities.
- Earlier work handled these with hierarchical LSTMs, memory-driven Transformers (R2Gen, R2GenCMN), image–text matching, knowledge graphs and multi-label classification.
- **The paper's idea:** pretrained LLMs already write long, fluent, coherent text and carry broad knowledge. The remaining problem is the **modality gap**: an LLM reads tokens, not pixels. R2GenGPT closes that gap with a lightweight visual alignment module.

---

## 3. Contributions (as claimed by the authors)

1. The first framework to use a **pretrained, frozen LLM** for radiology report generation, with full comparisons on two standard benchmarks.
2. Three ways to connect images to the LLM, each with a different number of trainable parameters: **shallow, delta and deep alignment**.
3. Strong results on **IU-Xray** and **MIMIC-CXR**, competitive with or better than state-of-the-art methods.
4. A **"decoder-centric" design.** Older encoder–decoder models give the encoder and decoder similar sizes and connect them by cross-attention. Here the encoder works as a "visual tokenizer", the huge decoder (LLM) dominates, and the model uses only self-attention.

---

## 4. Method

### 4.1 Architecture

```
                 ┌───────────────────────┐   ┌────────────────────────┐
 Chest X-ray ──► │ Visual Encoder        │──►│ Visual Mapper          │──► visual tokens H_v
   X_v           │ Swin Transformer-Base │   │ Linear layer W_m       │
                 └───────────────────────┘   └────────────────────────┘
                                                                         │
 Input sequence to the LLM:                                              ▼
 [ visual tokens H_v ] + [ prompt tokens X_p ] + [ report tokens X_r ] ──► Llama-2-7B (frozen) ──► report
   mask -100              mask -100               mask 1 (trained on)
```

- **Visual encoder:** `microsoft/swin-base-patch4-window7-224`. The authors use the grid features of the last Transformer layer.
- **Visual mapper:** one trainable linear projection, `H_v = W_m · Z_v`, where `Z_v = SwinTransformer(X_v)`. It maps image features to the LLM's embedding size.
- **LLM:** `meta-llama/Llama-2-7b-chat-hf`, with every parameter frozen.

### 4.2 The three alignment strategies

| Strategy | What is trained | Trainable params | Rationale |
|---|---|---|---|
| **Shallow** | Visual mapper `W_m` only | 4.2 M | The cheapest option. The Swin encoder is frozen, but it was trained on natural images, not X-rays. |
| **Delta** | `W_m` + **LoRA** on Swin's *query* and *value* projections (`ΔW = BA`, rank r ≪ min(d, k)) | 5.0 M | Adapts the encoder to radiographs with very few parameters. |
| **Deep** | `W_m` + **all** Swin parameters `θ_v` | 90.9 M | The best quality, at the highest training cost. |

LoRA settings: rank (attention dimension) = 16, alpha = 16.

### 4.3 Prompt

```
Human: <Img>X_v</Img> X_p \n Assistant: X_r </s>
```

- `X_v` is replaced by the visual tokens `H_v`.
- `X_p` is the instruction: *"Generate a comprehensive and detailed diagnosis report for this chest xray image."*
- `X_r` is the report. Everything else is tokenised with the Llama tokenizer.

### 4.4 Training objective

Instruction-tuning uses the standard auto-regressive (next-token) loss, computed **only on the report tokens**:

$$\mathcal{L}(\theta; X_r, X_v, X_p) = -\sum_{i=1}^{L} \log p_\theta\left(x_i \mid X_v, X_p, X_{r,<i}\right)$$

---

## 5. Datasets

| Dataset | Contents | Split used |
|---|---|---|
| **IU-Xray** (Indiana University) | 3,955 de-identified reports, 7,470 frontal/lateral chest X-rays | Train/test/val by 7:1:2, the same as R2Gen (Chen et al., 2020) |
| **MIMIC-CXR** (Beth Israel Deaconess, 2011–2016) | 377,110 images, 227,835 reports, 64,588 patients | Official split: 270,790 train / 2,130 val / 3,858 test |

> In this project, `dataset/iu_xray/` holds the authors' prepared IU-Xray package: 2,955 studies split 2,069 / 296 / 590. MIMIC-CXR requires credentialed PhysioNet access and has not been downloaded.

---

## 6. Implementation details

| Setting | Value |
|---|---|
| Hardware | 4 × NVIDIA A100 40 GB, mixed precision |
| Epochs | 15 (IU-Xray), 3 (MIMIC-CXR) |
| Batch size | 6 |
| Learning rate | 1e-4 |
| Decoding | Beam search, beam size 3 |

**Evaluation metrics**
- **Language quality:** BLEU-1 to 4, ROUGE-L, METEOR and CIDEr (coco-caption toolkit).
- **Clinical efficacy:** the CheXpert labeler tags generated and reference reports for 14 thoracic findings and devices. Precision, recall and F1 are computed from those tags (MIMIC-CXR only).

---

## 7. Results

### 7.1 IU-Xray

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|
| R2Gen | 0.470 | 0.304 | 0.219 | 0.165 | 0.371 | 0.187 | – |
| R2GenCMN | 0.475 | 0.309 | 0.222 | 0.170 | 0.375 | 0.191 | – |
| MSAT | 0.481 | 0.316 | 0.226 | 0.171 | 0.372 | 0.190 | 0.394 |
| METransformer | 0.483 | **0.322** | **0.228** | 0.172 | **0.380** | 0.192 | 0.435 |
| R2GenGPT (Shallow) | 0.466 | 0.301 | 0.211 | 0.156 | 0.370 | 0.202 | 0.405 |
| R2GenGPT (Delta) | 0.470 | 0.299 | 0.213 | 0.162 | 0.369 | **0.211** | 0.419 |
| **R2GenGPT (Deep)** | **0.488** | 0.316 | **0.228** | **0.173** | 0.377 | **0.211** | **0.438** |

### 7.2 MIMIC-CXR

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|
| R2Gen | 0.353 | 0.218 | 0.145 | 0.103 | 0.277 | 0.142 | – |
| PPKED | 0.360 | 0.224 | 0.149 | 0.106 | 0.284 | 0.149 | 0.237 |
| MSAT | 0.373 | 0.235 | 0.162 | 0.120 | 0.282 | 0.143 | 0.299 |
| METransformer | 0.386 | 0.250 | 0.169 | 0.124 | 0.291 | 0.152 | **0.362** |
| R2GenGPT (Shallow) | 0.365 | 0.237 | 0.163 | 0.117 | 0.277 | 0.136 | 0.145 |
| R2GenGPT (Delta) | 0.380 | 0.244 | 0.167 | 0.119 | 0.281 | 0.145 | 0.195 |
| **R2GenGPT (Deep)** | **0.411** | **0.267** | **0.186** | **0.134** | **0.297** | **0.160** | 0.269 |

The tables show selected baselines; the paper also lists Show-Tell, Att2in, AdaAtt, Transformer, M2Transformer and GSK.

### 7.3 Efficiency and clinical efficacy (MIMIC-CXR)

| Model | Trainable params | Time / epoch | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Shallow | 4.2 M | 1.75 h | 0.341 | 0.312 | 0.325 |
| Delta | 5.0 M | 1.83 h | 0.366 | 0.350 | 0.358 |
| **Deep** | 90.9 M | 2.75 h | **0.392** | **0.387** | **0.389** |
| METransformer | 152 M | 3.62 h | 0.364 | 0.309 | 0.334 |

### 7.4 Key takeaways

- **Deep alignment** is the best variant. On MIMIC-CXR it raises BLEU-4 from 0.124 to 0.134, **8.1% over METransformer**.
- **CIDEr on MIMIC-CXR stays below METransformer** (0.269 vs 0.362). The authors attribute this to METransformer's ensemble-like expert voting.
- **Shallow** (4.2M parameters) roughly matches the well-known R2Gen model. **Delta** (5M) comes close to state-of-the-art.
- **Clinical F1** of Deep (0.389) and Delta (0.358) is higher than METransformer's (0.334).
- **Training is faster** with fewer trainable parameters: 1.75–2.75 h per epoch versus 3.62 h for METransformer.
- **Qualitatively,** Shallow can miss findings. In one example it called a case with mild pulmonary edema normal, while Delta and Deep caught it.

---

## 8. Limitations and research gaps

These are not all stated by the authors; most come from a critical reading of the paper. Each one is a possible direction for this project.

**Model design**

| # | Gap | Possible direction |
|---|---|---|
| 1 | The visual encoder was pretrained on **natural images (ImageNet)**, not radiographs. The authors acknowledge this. | Use an X-ray-pretrained encoder (e.g. RAD-DINO, BiomedCLIP, CheXzero). |
| 2 | The LLM is a **general-domain** model and **fully frozen**. | Add LoRA to the LLM, or use a medical or smaller LLM. |
| 3 | The mapper is a **single linear layer**, with no study of other designs or of how many visual tokens to use. | Try an MLP, a Q-Former or a Perceiver resampler, and vary the token count. |
| 4 | Input is only **224×224**, so fine detail is lost. | Use higher-resolution or multi-scale features. |
| 5 | **Views are averaged or ignored.** There is no explicit multi-view (frontal and lateral) modelling and no **prior studies** for comparison. | Fuse multiple views and model scans over time. |
| 6 | **One fixed prompt** with no clinical context (indication, history, predicted disease labels). | Use context-aware or diagnosis-guided prompts. |

**Claims and evaluation**

| # | Gap | Possible direction |
|---|---|---|
| 7 | The paper claims LLMs reduce **bias toward normal cases** but never tests it. | Analyse results per disease and for rare findings. |
| 8 | The evaluation is mostly **word-overlap metrics**. Clinical efficacy is reported only for MIMIC-CXR, and there is no hallucination analysis or radiologist study. | Add RadGraph F1, a hallucination audit and clinical F1 on IU-Xray. |
| 9 | There is **no explainability**: no visual grounding of the generated sentences. | Add attention or Grad-CAM maps and sentence-to-region grounding. |
| 10 | There is **no cross-dataset or cross-modality** testing. | Train on MIMIC and test on IU; extend to CT or MRI. |
| 11 | **Deployment cost:** inference still needs a 7B LLM, and there are no latency or quantisation results. | Use 4-bit or 8-bit LLMs, smaller LLMs, and latency benchmarks. |
| 12 | There are **no ablations** (prompt, LoRA rank, beam size, token count) and no **variance or significance** across seeds. | Run systematic ablations with multiple seeds. |

---

## 9. How this repository uses the paper

| Path | What it is |
|---|---|
| `1-s2.0-S2950162823000334-main.pdf`, `R2GenGPT_Research_Paper.pdf` | The paper (journal and arXiv versions) |
| `R2GenGPT_Simple_Study_Guide.pdf` | Study notes |
| `dataset/iu_xray/` | The authors' prepared IU-Xray data, verified (see its `README.md`) |
| [`r2gengpt/`](r2gengpt/README.md) | A compact PyTorch re-implementation with all three alignment modes (trainable parameters checked against the paper: 4.21M / 4.98M / 90.95M). It covers IU-Xray and MIMIC-CXR training, evaluation and inference, with language metrics and clinical efficacy. Every gap above has a flag or script; see the gap table in [`r2gengpt/README.md`](r2gengpt/README.md#extensions-for-the-papers-research-gaps). Highlights: `--vision_model`, `--llm_lora`, `--mapper mlp\|perceiver`, `--visual_tokens`, `--view_fusion concat`, `--prompt_context indication`, `--disease_prompt`, `--load_in_4bit`, `explain.py`, `benchmark.py`, `ablate.py`, `significance.py`. |

---

## 10. Citation

```bibtex
@article{wang2023r2gengpt,
  title   = {R2GenGPT: Radiology Report Generation with Frozen LLMs},
  author  = {Wang, Zhanyu and Liu, Lingqiao and Wang, Lei and Zhou, Luping},
  journal = {Meta-Radiology},
  volume  = {1},
  pages   = {100033},
  year    = {2023},
  doi     = {10.1016/j.metrad.2023.100033}
}
```

The article is open access under CC BY-NC-ND 4.0. This README summarises it in my own words for study purposes.

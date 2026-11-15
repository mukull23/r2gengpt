"""R2GenGPT: visual encoder -> visual mapper -> frozen LLM (Wang et al., 2023)."""
import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, inject_adapter_in_model

from .labeler import CONDITIONS
from .mappers import MAPPERS, build_mapper, grid_tokens, pooling_matrix

DEFAULT_PROMPT = "Generate a comprehensive and detailed diagnosis report for this chest xray image."
ALIGN_MODES = ("shallow", "delta", "deep")
PROMPT_CONTEXTS = ("none", "indication")


def find_lora_targets(module, targets=("auto",)):
    """Query/value projection names: "query"/"value" in older transformers (and DINOv2), "q_proj"/"v_proj" in
    transformers 5 and CLIP. "auto" picks whichever the encoder has."""
    if list(targets) != ["auto"]:
        return list(targets)
    names = {n.rsplit(".", 1)[-1] for n, _ in module.named_modules()}
    for pair in (("query", "value"), ("q_proj", "v_proj"), ("qkv",)):
        if set(pair) <= names:
            return list(pair)
    raise ValueError("could not find query/value projections; pass --encoder_lora_targets")


class R2GenGPT(nn.Module):
    """
    align (paper Sec. 2.2.2):
      shallow - train only the visual mapper
      delta   - visual mapper + LoRA on the encoder's query/value projections
      deep    - visual mapper + the whole visual encoder
    The LLM stays frozen unless llm_lora=True (not in the paper; one of its gaps).

    Extensions for the paper's research gaps (all off by default, which gives the paper's model):
      mapper          - "linear" (paper), "mlp" or "perceiver" (gap 3)
      visual_tokens   - pool the encoder grid to this many tokens, or the Perceiver's query count (gap 3)
      view_fusion     - "mean" (official code) or "concat" with learned view embeddings (gap 5)
      prompt_context  - "indication" adds the study's clinical indication to the prompt (gap 6)
      disease_prompt  - a classifier head predicts the 14 CheXpert labels and the prompt lists them (gap 6)
    """

    def __init__(self, visual_encoder, llm, tokenizer, align="delta", view_fusion="mean", max_views=2,
                 mapper="linear", visual_tokens=None, lora_r=16, lora_alpha=16, lora_dropout=0.1,
                 encoder_lora_targets=("auto",), llm_lora=False, llm_lora_targets=("q_proj", "v_proj"),
                 prompt=DEFAULT_PROMPT, prompt_context="none", disease_prompt=False, disease_loss_weight=1.0,
                 max_report_len=60):
        super().__init__()
        assert align in ALIGN_MODES, f"align must be one of {ALIGN_MODES}"
        assert view_fusion in ("mean", "concat")
        assert mapper in MAPPERS, f"mapper must be one of {MAPPERS}"
        assert prompt_context in PROMPT_CONTEXTS, f"prompt_context must be one of {PROMPT_CONTEXTS}"
        self.align = align
        self.view_fusion = view_fusion
        self.mapper_type = mapper
        self.visual_tokens = visual_tokens
        self.prompt = prompt
        self.prompt_context = prompt_context
        self.disease_prompt = disease_prompt
        self.disease_loss_weight = disease_loss_weight
        self.max_report_len = max_report_len
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        tokenizer.padding_side = "right"

        self.visual_encoder = visual_encoder
        if align == "delta":
            lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=find_lora_targets(visual_encoder, encoder_lora_targets), bias="none")
            self.visual_encoder = inject_adapter_in_model(lora, visual_encoder)
        for name, p in self.visual_encoder.named_parameters():
            p.requires_grad = align == "deep" or (align == "delta" and "lora_" in name)

        self.llm = llm
        if llm_lora:
            lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=list(llm_lora_targets), bias="none")
            self.llm = inject_adapter_in_model(lora, llm)
        for name, p in self.llm.named_parameters():
            p.requires_grad = llm_lora and "lora_" in name

        enc_dim = visual_encoder.config.hidden_size
        llm_dim = llm.config.hidden_size
        self.visual_mapper = build_mapper(mapper, enc_dim, llm_dim, num_queries=visual_tokens or 32)
        # Start the visual tokens at the scale of the LLM's word embeddings. With the default LayerNorm gain of 1
        # they are ~70x larger in TinyLlama/Llama-2, which inflates the initial loss (9.7 vs 3.1 on IU-Xray).
        with torch.no_grad():
            emb = llm.get_input_embeddings().weight
            if emb.device.type != "meta":
                self.visual_mapper.norm.weight.fill_(emb.float().pow(2).mean().sqrt().item())
        if view_fusion == "concat":
            self.view_embed = nn.Parameter(torch.zeros(max_views, enc_dim))
        if disease_prompt:
            self.disease_head = nn.Linear(enc_dim, len(CONDITIONS))
        self.disease_pos_weight = None  # set by train.py (--disease_pos_weight balanced) to counter rare findings

        # Prompt: "Human: <Img>[visual tokens]</Img> {prompt} \nAssistant:" (paper Sec. 2.2.3)
        self.bos_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
        self.head_ids = self._ids("Human: <Img>")

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    # ------------------------------------------------------------------ vision
    def encode_images(self, images, return_maps=False):
        """images: (B, views, 3, H, W) -> {"tokens": (B, T, llm_dim), "global": (B, enc_dim)}.

        With return_maps, also "maps": (B, T, V*G), how much each visual token draws on each of the G grid
        cells of each of the V views ("layout" = (V, grid side)). explain.py uses it.
        """
        b, v = images.shape[:2]
        feats = self.visual_encoder(pixel_values=images.flatten(0, 1)).last_hidden_state
        feats, side = grid_tokens(feats)
        n = feats.shape[1]
        pool = None
        if self.visual_tokens and self.mapper_type != "perceiver" and self.visual_tokens < n:
            pool = pooling_matrix(n, side, self.visual_tokens, feats.device)
            feats = torch.einsum("tn,bnc->btc", pool.to(feats.dtype), feats)
        feats = feats.view(b, v, *feats.shape[1:])
        global_feat = feats.mean((1, 2))
        # The official code averages the frontal and lateral views; concat keeps both, tagged by view.
        if self.view_fusion == "mean":
            fused = feats.mean(1)
        else:
            fused = (feats + self.view_embed[:v, None].to(feats.dtype)).flatten(1, 2)
        tokens, attn = self.visual_mapper(fused)
        out = {"tokens": tokens, "global": global_feat}
        if return_maps:
            base = pool if pool is not None else torch.eye(n, device=feats.device)
            if self.view_fusion == "concat":
                base = torch.block_diag(*[base] * v)
            maps = base.unsqueeze(0).expand(b, -1, -1).float()
            out["maps"] = attn.float() @ maps if attn is not None else maps
            out["layout"] = (1 if self.view_fusion == "mean" else v, side)
        return out

    # ------------------------------------------------------------------ prompt
    def predict_findings(self, visual):
        """Disease-head probabilities (B, 14); only with disease_prompt."""
        return self.disease_head(visual["global"].float()).sigmoid()

    def contexts(self, batch_size, indications=None, findings=None):
        """Per-study text inserted before the instruction (empty for the paper's fixed prompt)."""
        out = []
        for i in range(batch_size):
            parts = []
            if self.prompt_context == "indication" and indications is not None:
                if indications[i]:
                    parts.append(f"Indication: {indications[i].strip().rstrip('.')}.")
            if findings is not None:
                names = [c.lower() for c, on in zip(CONDITIONS, findings[i]) if on and c != "No Finding"]
                parts.append(f"Predicted findings: {', '.join(names) or 'no finding'}.")
            out.append(" ".join(parts))
        return out

    def _prefix(self, visual, contexts):
        """Left-padded embeddings of [BOS, "Human: <Img>", visual tokens, "</Img> {context} {prompt} \\nAssistant:"].

        Returns (embeds (B, P, D), attention mask (B, P), index of the first visual token per sample).
        """
        img = visual["tokens"]
        b, device = img.shape[0], img.device
        embed_layer = self.llm.get_input_embeddings()
        dtype = embed_layer.weight.dtype
        head = torch.tensor(self.bos_ids + self.head_ids, device=device)
        tails = [self._ids(f"</Img> {c + ' ' if c else ''}{self.prompt} \nAssistant:") for c in contexts]
        width = max(map(len, tails))
        head_emb = embed_layer(head)
        rows, masks, starts = [], [], []
        for i in range(b):
            n_pad = width - len(tails[i])
            pad = embed_layer(torch.full((n_pad,), self.tokenizer.pad_token_id, device=device))
            tail = embed_layer(torch.tensor(tails[i], device=device, dtype=torch.long))
            rows.append(torch.cat([pad, head_emb, img[i].to(dtype), tail]))
            masks.append(torch.cat([torch.zeros(n_pad, dtype=torch.long, device=device),
                                    torch.ones(rows[-1].shape[0] - n_pad, dtype=torch.long, device=device)]))
            starts.append(n_pad + len(head))
        return torch.stack(rows), torch.stack(masks), starts

    def build_inputs(self, visual, reports, contexts):
        """LLM inputs for teacher forcing: [prefix][report </s>], loss on the report tokens only."""
        prefix, prefix_mask, starts = self._prefix(visual, contexts)
        device = prefix.device
        text = self.tokenizer([r + self.tokenizer.eos_token for r in reports], return_tensors="pt",
                              padding="longest", truncation=True, max_length=self.max_report_len,
                              add_special_tokens=False).to(device)
        inputs_embeds = torch.cat([prefix, self._embed(text.input_ids)], dim=1)
        attention_mask = torch.cat([prefix_mask, text.attention_mask], dim=1)
        labels = torch.cat([torch.full(prefix_mask.shape, -100, dtype=torch.long, device=device),
                            text.input_ids.masked_fill(text.attention_mask == 0, -100)], dim=1)
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "position_ids": position_ids,
                "labels": labels, "image_starts": starts, "prefix_len": prefix.shape[1]}

    # ------------------------------------------------------------------ train / generate
    def forward(self, images, reports, indications=None, labels=None):
        """Auto-regressive loss on report tokens only (paper Eq. 2), plus the disease-head loss if enabled."""
        visual = self.encode_images(images)
        findings, cls_loss = None, 0.0
        if self.disease_prompt:
            logits = self.disease_head(visual["global"].float())
            if labels is not None:
                pos_weight = None if self.disease_pos_weight is None else self.disease_pos_weight.to(logits.device)
                cls_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
                findings = labels.bool().tolist()  # teacher forcing: the prompt lists the reference findings
            else:
                findings = (logits.sigmoid() > 0.5).tolist()
        inputs = self.build_inputs(visual, reports, self.contexts(len(reports), indications, findings))
        loss = self.llm(**{k: inputs[k] for k in ("inputs_embeds", "attention_mask", "position_ids", "labels")}).loss
        return loss + self.disease_loss_weight * cls_loss

    @torch.no_grad()
    def generate(self, images, indications=None, num_beams=3, max_new_tokens=100, min_new_tokens=0,
                 repetition_penalty=2.0, length_penalty=2.0, return_findings=False):
        visual = self.encode_images(images)
        probs = self.predict_findings(visual) if self.disease_prompt else None
        findings = (probs > 0.5).tolist() if probs is not None else None
        prefix, mask, _ = self._prefix(visual, self.contexts(images.shape[0], indications, findings))
        out = self.llm.generate(
            inputs_embeds=prefix, attention_mask=mask,
            num_beams=num_beams, do_sample=False, max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens, repetition_penalty=repetition_penalty,
            length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        texts = [t.split(self.tokenizer.eos_token)[0].strip() for t in texts]
        return (texts, probs) if return_findings else texts

    def trainable_state_dict(self):
        """Only the trained weights, so checkpoints stay small (a few MB for shallow/delta)."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

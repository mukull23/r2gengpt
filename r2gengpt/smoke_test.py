"""End-to-end check of the pipeline on real IU-Xray studies with tiny, randomly initialised models.

Nothing is downloaded: the Swin encoder and Llama decoder are built from small configs and the
tokenizer is a word-level vocabulary built from the training reports. Runs on CPU or Apple MPS
in a few minutes. It verifies the plumbing (shapes, masking, LoRA placement, prompt padding,
generation, metrics, grounding, checkpoint round-trip and every command-line script), not report quality.

    python smoke_test.py            # model checks + every script
    python smoke_test.py --quick    # model checks only
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, SwinConfig, SwinModel

from r2gen.builder import DEFAULT_DATA_DIR, get_device
from r2gen.data import ReportDataset, collate
from r2gen.engine import run_generation, train_one_epoch
from r2gen.grounding import ground_report
from r2gen.metrics import full_report
from r2gen.model import R2GenGPT

IMAGE_SIZE = 64
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_WORDS = " human img assistant generate a comprehensive and detailed diagnosis report for this chest xray image " \
               "indication predicted findings no finding " + " ".join(
                   "enlarged cardiomediastinum cardiomegaly lung opacity lesion edema consolidation pneumonia "
                   "atelectasis pneumothorax pleural effusion other fracture support devices".split())

# (name, model kwargs)
CONFIGS = [
    ("shallow", dict(align="shallow")),
    ("delta", dict(align="delta")),
    ("deep", dict(align="deep")),
    ("delta + LLM LoRA", dict(align="delta", llm_lora=True)),
    ("mlp, 4 tokens, concat", dict(align="delta", mapper="mlp", visual_tokens=4, view_fusion="concat")),
    ("perceiver + context", dict(align="shallow", mapper="perceiver", visual_tokens=8,
                                 prompt_context="indication", disease_prompt=True)),
]


def build_tokenizer(reports):
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train_from_iterator(reports, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"]))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")


def tiny_encoder():
    return SwinModel(SwinConfig(image_size=IMAGE_SIZE, patch_size=4, embed_dim=16, depths=[1, 1],
                                num_heads=[1, 2], window_size=4))


def tiny_llm(tokenizer):
    return LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id))


def build_tiny(tokenizer, **kwargs):
    torch.manual_seed(0)
    return R2GenGPT(tiny_encoder(), tiny_llm(tokenizer), tokenizer, lora_r=4, lora_alpha=4, **kwargs)


def check_trainable(model, kw):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    allowed = ("visual_mapper", "visual_encoder", "llm", "view_embed", "disease_head")
    assert all(n.startswith(allowed) for n in trainable), sorted(trainable)
    assert any(n.startswith("visual_mapper") for n in trainable)
    assert any("lora_" in n and n.startswith("llm") for n in trainable) == kw.get("llm_lora", False)
    assert all("lora_" in n for n in trainable if n.startswith("llm"))
    enc = [n for n in trainable if n.startswith("visual_encoder")]
    if kw["align"] == "shallow":
        assert not enc
    elif kw["align"] == "delta":
        assert enc and all("lora_" in n and any(t in n for t in ("query", "value", "q_proj", "v_proj")) for n in enc)
    else:
        assert enc and not any("lora_" in n for n in enc)


def check_padding(model, batch, device):
    """Per-study prompts of different lengths are left-padded; each study must get the logits it gets alone."""
    model.eval()
    batch = {**batch, "images": batch["images"].to(device)}
    indications = ["chest pain", "", "shortness of breath after a fall"][: len(batch["report"])]
    with torch.no_grad():
        visual = model.encode_images(batch["images"])
        contexts = model.contexts(len(indications), indications, [[1] + [0] * 13] * len(indications))
        inputs = model.build_inputs(visual, batch["report"], contexts)
        keys = ("inputs_embeds", "attention_mask", "position_ids")
        logits = model.llm(**{k: inputs[k] for k in keys}).logits
        p = inputs["prefix_len"]
        for i in range(len(indications)):
            single = model.build_inputs({"tokens": visual["tokens"][i:i + 1]}, batch["report"][i:i + 1], contexts[i:i + 1])
            ref = model.llm(**{k: single[k] for k in keys}).logits[0, single["prefix_len"]:]
            assert torch.allclose(logits[i, p:p + ref.shape[0]], ref, atol=1e-4), "left padding changed the logits"
        gen = dict(num_beams=1, max_new_tokens=12, repetition_penalty=1.0)
        batched = model.generate(batch["images"], indications, **gen)
        alone = [model.generate(batch["images"][i:i + 1], indications[i:i + 1], **gen)[0] for i in range(len(indications))]
    assert batched == alone, "batched generation differs from one-by-one generation"


def model_checks(device, train_set, val_set, tokenizer):
    train_loader = DataLoader(train_set, 4, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, 4, collate_fn=collate)
    gen = dict(num_beams=3, max_new_tokens=30)
    for name, kw in CONFIGS:
        model = build_tiny(tokenizer, **kw).to(device)
        check_trainable(model, kw)
        check_padding(model, collate([val_set[i] for i in range(3)]), device)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=3e-3)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        first = train_one_epoch(model, train_loader, opt, sched, device)
        for _ in range(14):
            last = train_one_epoch(model, train_loader, opt, sched, device)
        scores, refs, preds = run_generation(model, val_loader, device, gen)

        # Checkpoint round-trip: a fresh model with the saved trainable weights must generate the same text.
        reloaded = build_tiny(tokenizer, **kw).to(device)
        reloaded.load_state_dict(model.trainable_state_dict(), strict=False)
        _, _, preds2 = run_generation(reloaded, val_loader, device, gen)
        assert preds == preds2, "reloaded model generated different reports"

        # Grounding: maps over the 8x8 Swin grid of each view.
        images = collate([val_set[0]])["images"].to(device)
        report = preds[val_set[0]["id"]] or "the heart is normal ."
        for method in ("attention", "gradient"):
            g = ground_report(model, images, report, indications=[val_set[0]["indication"]], method=method)
            views = 2 if kw.get("view_fusion") == "concat" else 1
            assert g["grid"].shape == (len(g["sentences"]), views, 8, 8), g["grid"].shape
            assert torch.isfinite(g["grid"]).all()

        n_train, n_total = model.count_parameters()
        print(f"[{name:22s}] trainable {n_train:>7,}/{n_total:,}  loss {first:.3f} -> {last:.3f}  "
              f"BLEU_4={scores['BLEU_4']:.3f} CIDEr={scores['CIDEr']:.3f}")
        assert last < first, f"{name}: loss did not decrease"

    report, _ = full_report(refs, preds)
    assert {"CE_F1", "reports_with_hallucination"} <= set(report["overall"]) and report["per_label"]
    print("example generation:", next(iter(preds.values()))[:120])


def save_tiny_checkpoints(root, tokenizer):
    """Tiny local Hugging Face checkpoints, so the command-line scripts run without downloads."""
    torch.manual_seed(0)
    enc_dir, llm_dir = os.path.join(root, "tiny-swin"), os.path.join(root, "tiny-llama")
    tiny_encoder().save_pretrained(enc_dir)
    tiny_llm(tokenizer).save_pretrained(llm_dir)
    tokenizer.save_pretrained(llm_dir)
    return enc_dir, llm_dir


def script_checks(tokenizer):
    def run(*args):
        print("$ python", " ".join(args), flush=True)
        out = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True)
        assert out.returncode == 0, f"{args[0]} failed:\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}"
        return out.stdout

    with tempfile.TemporaryDirectory() as root:
        enc_dir, llm_dir = save_tiny_checkpoints(root, tokenizer)
        base = ["--vision_model", enc_dir, "--llm_model", llm_dir, "--image_size", str(IMAGE_SIZE),
                "--llm_dtype", "float32", "--batch_size", "4", "--num_workers", "0", "--max_new_tokens", "20",
                "--epochs", "2", "--train_limit", "8", "--val_limit", "4", "--lr", "3e-3", "--output_dir", root]
        run("train.py", *base, "--align", "delta", "--run_name", "a")
        run("train.py", *base, "--align", "shallow", "--mapper", "perceiver", "--prompt_context", "indication",
            "--disease_prompt", "--view_fusion", "concat", "--run_name", "b")
        ck_a, ck_b = os.path.join(root, "a", "best.pt"), os.path.join(root, "b", "best.pt")
        for ck in (ck_a, ck_b):
            run("evaluate.py", "--checkpoint", ck, "--limit", "8", "--num_workers", "0", "--batch_size", "4")
        with open(os.path.join(root, "b", "test_metrics.json")) as f:
            metrics = json.load(f)
        assert {"overall", "per_label", "subsets", "setup"} <= set(metrics) and "CE_F1" in metrics["overall"]
        run("evaluate.py", "--checkpoint", ck_a, "--limit", "4", "--num_workers", "0", "--num_beams", "1",
            "--tag", "_beam1")
        assert os.path.exists(os.path.join(root, "a", "test_beam1_metrics.json"))

        sample = json.load(open(os.path.join(DEFAULT_DATA_DIR, "annotation.json")))["test"][0]
        views = [os.path.join(DEFAULT_DATA_DIR, "images", p) for p in sample["image_path"]]
        print(run("infer.py", "--checkpoint", ck_b, "--images", *views, "--indication", "chest pain").strip())
        for method in ("attention", "gradient"):
            run("explain.py", "--checkpoint", ck_b, "--study", sample["id"], "--method", method)
            assert os.path.exists(os.path.join(root, "b", f"explain_{sample['id']}_{method}.png"))
        bench = os.path.join(root, "bench.json")
        run("benchmark.py", "--checkpoint", ck_a, "--num_studies", "3", "--warmup", "1", "--out", bench)
        assert json.load(open(bench))["latency_s_per_study_mean"] > 0
        print(run("significance.py", os.path.join(root, "a", "test_predictions.json"),
                  os.path.join(root, "b", "test_predictions.json"), "--n_boot", "50").strip())
        abl = ["ablate.py", "--name", "tiny", "--vary", "lora_r=2,4", "--vary", "num_beams=1,2", "--seeds", "1", "2",
               "--output_dir", root, "--eval_args", "--limit 4 --num_workers 0", "--"] + base[:-2]
        run(*abl)
        print(open(os.path.join(root, "tiny", "summary.md")).read().strip())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Skip the command-line script checks.")
    cli = p.parse_args()
    device = get_device()
    train_set = ReportDataset(DEFAULT_DATA_DIR, "train", IMAGE_SIZE, limit=8)
    val_set = ReportDataset(DEFAULT_DATA_DIR, "val", IMAGE_SIZE, limit=4)
    all_reports = [s["report"] for s in ReportDataset(DEFAULT_DATA_DIR, "train", IMAGE_SIZE, limit=64).samples]
    tokenizer = build_tokenizer([train_set.clean(r) for r in all_reports] + [PROMPT_WORDS])
    print(f"device={device}  vocab={len(tokenizer)}  sample report: {train_set[0]['report'][:80]}...")
    model_checks(device, train_set, val_set, tokenizer)
    if not cli.quick:
        script_checks(tokenizer)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

"""Command-line options and construction of R2GenGPT from Hugging Face checkpoints."""
import argparse
import os

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .data import DATASETS, IMAGENET_MEAN, IMAGENET_STD
from .mappers import MAPPERS
from .model import ALIGN_MODES, DEFAULT_PROMPT, PROMPT_CONTEXTS, R2GenGPT

DATASET_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
DEFAULT_DATA_DIR = os.path.join(DATASET_ROOT, "iu_xray")
# Paper Sec. 3.2: 15 epochs on IU-Xray, 3 on MIMIC-CXR. MIMIC reports are longer.
DATASET_DEFAULTS = {
    "iu_xray": {"epochs": 15, "max_report_len": 60, "max_new_tokens": 100},
    "mimic_cxr": {"epochs": 3, "max_report_len": 100, "max_new_tokens": 150},
}


def add_model_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("data")
    g.add_argument("--dataset", choices=DATASETS, default="iu_xray")
    g.add_argument("--data_dir", default=None, help="Default: ../dataset/<dataset>.")

    g = parser.add_argument_group("model")
    g.add_argument("--vision_model", default="microsoft/swin-base-patch4-window7-224",
                   help="Any HF image backbone, e.g. microsoft/rad-dino (X-ray pretrained).")
    g.add_argument("--llm_model", default="meta-llama/Llama-2-7b-chat-hf",
                   help="Any Hugging Face causal LM, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0 on small GPUs.")
    g.add_argument("--align", choices=ALIGN_MODES, default="delta")
    g.add_argument("--mapper", choices=MAPPERS, default="linear", help="Paper: linear.")
    g.add_argument("--visual_tokens", type=int, default=None,
                   help="Pool the encoder grid to N tokens (linear/mlp) or use N Perceiver queries (default 32).")
    g.add_argument("--view_fusion", choices=("mean", "concat"), default="mean")
    g.add_argument("--lora_r", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=16)
    g.add_argument("--encoder_lora_targets", default="auto", help="auto: the encoder's query/value projections.")
    g.add_argument("--llm_lora", action="store_true", help="Also LoRA-tune the LLM (not in the paper).")
    g.add_argument("--llm_lora_targets", default="q_proj,v_proj")
    g.add_argument("--load_in_4bit", action="store_true", help="CUDA only; needs bitsandbytes.")
    g.add_argument("--load_in_8bit", action="store_true", help="CUDA only; needs bitsandbytes.")
    g.add_argument("--llm_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto",
                   help="auto: bfloat16 on CUDA, float16 on Apple MPS, float32 on CPU.")
    g.add_argument("--prompt", default=DEFAULT_PROMPT)
    g.add_argument("--prompt_context", choices=PROMPT_CONTEXTS, default="none",
                   help="indication: add the study's clinical indication to the prompt.")
    g.add_argument("--disease_prompt", action="store_true",
                   help="Predict the 14 CheXpert labels from the image and list them in the prompt.")
    g.add_argument("--disease_loss_weight", type=float, default=1.0)
    g.add_argument("--disease_pos_weight", choices=("balanced", "none"), default="balanced",
                   help="balanced: weight each finding's positives by its negative/positive ratio (max 10).")
    g.add_argument("--image_size", type=int, default=224)
    g.add_argument("--max_views", type=int, default=2)
    g.add_argument("--max_report_len", type=int, default=None, help="Report tokens kept for training.")

    g = parser.add_argument_group("generation")
    g.add_argument("--num_beams", type=int, default=3)
    g.add_argument("--max_new_tokens", type=int, default=None)
    g.add_argument("--min_new_tokens", type=int, default=0)
    g.add_argument("--repetition_penalty", type=float, default=2.0)
    g.add_argument("--length_penalty", type=float, default=2.0)
    return parser


def default_args():
    """Every model/generation option at its default (used to fill in options missing from old checkpoints)."""
    return vars(add_model_args(argparse.ArgumentParser()).parse_args([]))


def resolve_args(args):
    """Fill dataset-dependent defaults and the encoder's image normalisation."""
    for key, value in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.data_dir is None:
        args.data_dir = os.path.join(DATASET_ROOT, args.dataset)
    if getattr(args, "image_mean", None) is None:
        args.image_mean, args.image_std = image_stats(args.vision_model)
    return args


def image_stats(vision_model):
    """Normalisation the encoder was trained with (RAD-DINO and CLIP differ from ImageNet)."""
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(vision_model)
        mean, std = getattr(proc, "image_mean", None), getattr(proc, "image_std", None)
        if mean and std:
            return list(mean), list(std)
    except Exception:  # offline, or a checkpoint without a processor config
        pass
    return list(IMAGENET_MEAN), list(IMAGENET_STD)


def generation_kwargs(args):
    return dict(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens,
                repetition_penalty=args.repetition_penalty, length_penalty=args.length_penalty)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def llm_dtype(name, device):
    if name == "auto":
        return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type, torch.float32)
    return getattr(torch, name)


def build_model(args, device, attn_implementation=None):
    encoder = AutoModel.from_pretrained(args.vision_model)
    if hasattr(encoder, "vision_model"):  # CLIP-style checkpoints: keep only the image tower
        encoder = encoder.vision_model

    llm_kwargs = {"dtype": llm_dtype(args.llm_dtype, device)}
    if attn_implementation:
        llm_kwargs["attn_implementation"] = attn_implementation
    if args.load_in_4bit or args.load_in_8bit:
        assert device.type == "cuda", "--load_in_4bit/--load_in_8bit need a CUDA GPU (bitsandbytes)"
        from transformers import BitsAndBytesConfig
        llm_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        llm_kwargs["device_map"] = {"": device.index or 0}
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model, **llm_kwargs)

    model = R2GenGPT(
        encoder, llm, tokenizer, align=args.align, view_fusion=args.view_fusion, max_views=args.max_views,
        mapper=args.mapper, visual_tokens=args.visual_tokens,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        encoder_lora_targets=args.encoder_lora_targets.split(","),
        llm_lora=args.llm_lora, llm_lora_targets=args.llm_lora_targets.split(","),
        prompt=args.prompt, prompt_context=args.prompt_context, disease_prompt=args.disease_prompt,
        disease_loss_weight=args.disease_loss_weight, max_report_len=args.max_report_len,
    )
    # Trainable LoRA weights inside a half-precision LLM are kept in float32 for stable updates.
    for p in model.llm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if "device_map" in llm_kwargs:
        # A quantized LLM is already placed by device_map and cannot be moved with .to().
        for name, child in model.named_children():
            if name != "llm":
                child.to(device)
        for name, p in model.named_parameters(recurse=False):
            p.data = p.data.to(device)
        return model
    return model.to(device)


def load_checkpoint(path, device, overrides=None, attn_implementation=None):
    """Rebuild the model from a checkpoint's saved args and load its trained weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**{**default_args(), **ckpt["args"], **(overrides or {})})
    resolve_args(args)
    model = build_model(args, device, attn_implementation)
    missing = set(ckpt["state_dict"]) - set(dict(model.named_parameters()))
    assert not missing, f"checkpoint has weights the model lacks: {sorted(missing)[:5]}"
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.eval(), args

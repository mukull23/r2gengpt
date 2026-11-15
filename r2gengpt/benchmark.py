"""Inference cost of R2GenGPT (research gap 11): latency, throughput, peak memory and parameter counts.

Latency does not depend on the trained weights, so a checkpoint is optional. Examples:
    python benchmark.py --checkpoint runs/iu_delta/best.pt
    python benchmark.py --align delta --load_in_4bit                       # 7B LLM in 4-bit (CUDA)
    python benchmark.py --llm_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num_beams 1
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from r2gen.builder import (add_model_args, build_model, default_args, generation_kwargs, get_device,
                           load_checkpoint, resolve_args)
from r2gen.data import ReportDataset, collate

OVERRIDABLE = ("load_in_4bit", "load_in_8bit", "llm_dtype", "num_beams", "max_new_tokens", "min_new_tokens",
               "repetition_penalty", "length_penalty")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--num_studies", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2, help="Batches run before timing starts.")
    p.add_argument("--out", default=None, help="Write the results to this JSON file.")
    cli = p.parse_args()

    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if cli.checkpoint:
        defaults = default_args()
        overrides = {k: getattr(cli, k) for k in OVERRIDABLE if getattr(cli, k) != defaults[k]}
        model, args = load_checkpoint(cli.checkpoint, device, overrides)
        data_dir = args.data_dir
    else:
        args = resolve_args(cli)
        model, data_dir = build_model(args, device).eval(), args.data_dir
    load_memory = memory_mb(device)

    data = ReportDataset(data_dir, cli.split, args.image_size, args.max_views, cli.warmup * cli.batch_size
                         + cli.num_studies, dataset=args.dataset, mean=args.image_mean, std=args.image_std)
    loader = DataLoader(data, cli.batch_size, collate_fn=collate)
    gen = generation_kwargs(args)
    latencies, new_tokens = [], []
    for i, batch in enumerate(loader):
        images = batch["images"].to(device)
        sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            reports = model.generate(images, indications=batch["indication"], **gen)
        sync(device)
        if i >= cli.warmup:
            latencies.append((time.perf_counter() - start) / len(reports))
            new_tokens += [len(model.tokenizer(r, add_special_tokens=False).input_ids) + 1 for r in reports]

    trainable, total = model.count_parameters()
    ckpt_mb = os.path.getsize(cli.checkpoint) / 2**20 if cli.checkpoint else None
    lat = sorted(latencies)
    result = {
        "device": str(device), "llm": args.llm_model, "vision": args.vision_model, "align": args.align,
        "quantization": "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else args.llm_dtype,
        "batch_size": cli.batch_size, "studies_timed": len(lat), **gen,
        "latency_s_per_study_mean": statistics.mean(lat), "latency_s_per_study_p50": lat[len(lat) // 2],
        "latency_s_per_study_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "generated_tokens_per_s": sum(new_tokens) / sum(latencies),
        "mean_generated_tokens": statistics.mean(new_tokens),
        "memory_after_load_mb": load_memory, "peak_memory_mb": memory_mb(device),
        "total_params_m": total / 1e6, "trainable_params_m": trainable / 1e6, "checkpoint_mb": ckpt_mb,
    }
    print(json.dumps(result, indent=1))
    if cli.out:
        with open(cli.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()

"""Run an ablation over settings and seeds, then report mean +- std on the test split (research gap 12).

Everything after "--" is passed to train.py for every run. Generation settings (num_beams, ...) are
applied at evaluation, so they reuse one trained model per seed. Finished runs are skipped on re-runs.
Examples:
    python ablate.py --name lora_rank --vary lora_r=4,16,64 --seeds 1 2 3 -- --align delta
    python ablate.py --name beams --vary num_beams=1,3,5 --seeds 1 2 3 -- --align delta
    python ablate.py --name mapper --vary mapper=linear,mlp,perceiver --vary visual_tokens=16,49 -- --align delta
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

from evaluate import GENERATION_OVERRIDES
from r2gen.stats import mean_std

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS = ["BLEU_1", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr", "CE_F1", "reports_with_hallucination"]


def parse_args():
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--vary", action="append", default=[], help="key=v1,v2,... (repeat for a grid)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--output_dir", default="runs/ablations")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--eval_args", default="", help="Extra evaluate.py arguments, e.g. '--labeler chexbert'.")
    p.add_argument("--dry_run", action="store_true")
    cli = p.parse_args(argv)
    cli.train_args = train_args
    return cli


def parse_vary(items):
    grid = {}
    for item in items:
        key, values = item.split("=", 1)
        grid[key.lstrip("-")] = values.split(",")
    return grid


def combos(grid):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]


def as_flags(setting):
    flags = []
    for k, v in setting.items():
        flags += [f"--{k}"] if v.lower() == "true" else [] if v.lower() == "false" else [f"--{k}", v]
    return flags


def slug(setting):
    return "_".join(f"{k}={v}" for k, v in setting.items()) or "base"


def run(cmd, dry):
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=HERE)


def main():
    cli = parse_args()
    grid = parse_vary(cli.vary)
    train_grid = {k: v for k, v in grid.items() if k not in GENERATION_OVERRIDES}
    eval_grid = {k: v for k, v in grid.items() if k in GENERATION_OVERRIDES}
    root = os.path.join(cli.output_dir, cli.name)
    rows = []
    for train_setting in combos(train_grid):
        for seed in cli.seeds:
            run_name = f"{cli.name}/{slug(train_setting)}_seed={seed}"
            run_dir = os.path.join(cli.output_dir, run_name)
            if not os.path.exists(os.path.join(run_dir, "best.pt")):
                run([sys.executable, "train.py", *cli.train_args, *as_flags(train_setting), "--seed", str(seed),
                     "--output_dir", cli.output_dir, "--run_name", run_name], cli.dry_run)
            for eval_setting in combos(eval_grid):
                tag = f"_{slug(eval_setting)}" if eval_setting else ""
                metrics_path = os.path.join(run_dir, f"test{tag}_metrics.json")
                if not os.path.exists(metrics_path):
                    run([sys.executable, "evaluate.py", "--checkpoint", os.path.join(run_dir, "best.pt"),
                         *as_flags(eval_setting), "--tag", tag, *cli.eval_args.split()], cli.dry_run)
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        overall = json.load(f)["overall"]
                    rows.append({"setting": {**train_setting, **eval_setting}, "seed": seed, **overall})
    if cli.dry_run or not rows:
        return

    summary = {}
    for row in rows:
        summary.setdefault(slug(row["setting"]), []).append(row)
    table = {}
    for name, group in summary.items():
        table[name] = {"n_seeds": len(group), "seeds": [g["seed"] for g in group]}
        for m in cli.metrics:
            values = [g[m] for g in group if m in g]
            if values:
                table[name][m] = dict(zip(("mean", "std"), mean_std(values)))
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump({"runs": rows, "summary": table}, f, indent=1)
    metrics = [m for m in cli.metrics if any(m in t for t in table.values())]
    lines = ["| setting | seeds | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for name, t in table.items():
        cells = [f"{t[m]['mean']:.4f} ± {t[m]['std']:.4f}" if m in t else "–" for m in metrics]
        lines.append(f"| {name} | {t['n_seeds']} | " + " | ".join(cells) + " |")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {root}/summary.md and summary.json")


if __name__ == "__main__":
    main()

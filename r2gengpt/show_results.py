"""Print the results of the trained runs in the terminal. Examples:
    python show_results.py                      # everything
    python show_results.py --section compare    # one section: compare, findings, curves, examples, paper
    python show_results.py --examples 10        # more example reports
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [  # (label, run directory)
    ("A. Baseline (frozen LLM, paper setup)", "iu_delta_tinyllama"),
    ("B. LoRA only", "iu_delta_tinyllama_llmlora"),
    ("C. Disease prompt + LoRA", "iu_delta_tinyllama_disease_llmlora"),
]
PAPER = {  # Wang et al. 2023, Table 1 (IU-Xray)
    "R2GenGPT shallow (paper)": [0.466, 0.301, 0.211, 0.156, 0.370, 0.405],
    "R2GenGPT delta (paper)": [0.470, 0.299, 0.213, 0.162, 0.369, 0.419],
    "R2GenGPT deep (paper)": [0.488, 0.316, 0.228, 0.173, 0.377, 0.438],
}
NLG = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L", "CIDEr"]


def load(run, name):
    path = os.path.join(HERE, "runs", run, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f) if name.endswith(".json") else [json.loads(line) for line in f]


def findings_f1(metrics):
    """Findings-only F1 (13 findings, no "No Finding"); older metric files lack it, so rebuild from per_label."""
    o = metrics["overall"]
    if "CE_F1_findings" in o:
        return o["CE_F1_findings"]
    tp = fp = fn = 0
    for c, v in metrics["per_label"].items():
        if c == "No Finding" or not v["support"] and not v["precision"]:
            continue
        s, r, p = v["support"], v["recall"], v["precision"]
        t = r * s
        tp, fn = tp + t, fn + s - t
        fp += t / p - t if p else 0
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def title(text):
    print("\n" + text + "\n" + "=" * len(text))


def section_compare():
    title("Test results (590 IU-Xray studies, final epoch)")
    head = f"{'Model':40s}" + "".join(f"{k:>9s}" for k in NLG) + f"{'Find.F1':>9s}{'Abn->Norm':>11s}"
    print(head + "\n" + "-" * len(head))
    for label, run in RUNS:
        m = load(run, "test_last_metrics.json")
        if not m:
            print(f"{label:40s}  (not evaluated yet: python evaluate.py --checkpoint runs/{run}/last.pt --tag _last)")
            continue
        o = m["overall"]
        print(f"{label:40s}" + "".join(f"{o[k]:9.3f}" for k in NLG)
              + f"{findings_f1(m):9.3f}{o['abnormal_reports_called_normal']:10.0%} ")
    print("\nFind.F1   = clinical F1 over the 13 findings (excludes 'No Finding'); 0 means no finding is ever reported")
    print("Abn->Norm = share of abnormal studies that got a normal report")


def section_paper():
    title("Comparison with the paper (IU-Xray test)")
    head = f"{'Model':40s}" + "".join(f"{k:>9s}" for k in NLG)
    print(head + "\n" + "-" * len(head))
    for label, vals in PAPER.items():
        print(f"{label:40s}" + "".join(f"{v:9.3f}" for v in vals))
    for label, run in RUNS:
        for name, tag in (("test_metrics.json", "best"), ("test_last_metrics.json", "last")):
            m = load(run, name)
            if m:
                print(f"{(label.split('(')[0].strip() + ' - ' + tag):40s}"
                      + "".join(f"{m['overall'][k]:9.3f}" for k in NLG))
    print("\nOurs: TinyLlama-1.1B on an Apple M5 Pro. Paper: Llama-2-7B on 4x A100.")


def section_findings():
    title("Per-finding F1, final epoch (support = test studies with that finding)")
    tables = {label: load(run, "test_last_metrics.json") for label, run in RUNS}
    tables = {k: v for k, v in tables.items() if v}
    labels = list(tables)
    print(f"{'Finding':28s}{'support':>8s}" + "".join(f"{l.split('.')[0]:>8s}" for l in labels))
    for cond, v in next(iter(tables.values()))["per_label"].items():
        print(f"{cond:28s}{v['support']:8d}" + "".join(f"{tables[l]['per_label'][cond]['f1']:8.2f}" for l in labels))
    print("\nColumns A, B, C are the models in the comparison table.")


def section_curves():
    title("Validation scores per epoch (100 studies)")
    for label, run in RUNS:
        log = load(run, "log.jsonl")
        if not log:
            continue
        print(f"\n{label}")
        print(f"{'epoch':>5s}{'loss':>8s}{'BLEU_4':>9s}{'CIDEr':>9s}{'Find.F1':>9s}")
        for r in log:
            f1 = f"{r['CE_F1_findings']:9.3f}" if "CE_F1_findings" in r else f"{'-':>9s}"
            print(f"{r['epoch']:5d}{r['train_loss']:8.3f}{r['BLEU_4']:9.3f}{r['CIDEr']:9.3f}{f1}")


def section_examples(n):
    title(f"Example reports on abnormal test studies (model C vs baseline A, first {n})")
    c = load("iu_delta_tinyllama_disease_llmlora", "test_last_predictions.json")
    a = load("iu_delta_tinyllama", "test_last_predictions.json")
    if not c:
        print("(model C has not been evaluated yet)")
        return
    found = lambda labels: [k for k, x in labels.items() if x == 1 and k != "No Finding"] or ["none"]
    shown = 0
    for sid, v in c.items():
        if found(v["labels"]["reference"]) == ["none"]:
            continue  # only studies whose real report states a finding
        print(f"\n[{sid}]  real findings: {', '.join(found(v['labels']['reference']))}")
        print(f"  REAL : {v['reference']}")
        print(f"  C    : {v['generated']}")
        print(f"         -> {', '.join(found(v['labels']['generated']))}")
        if a and sid in a:
            print(f"  A    : {a[sid]['generated']}")
        shown += 1
        if shown >= n:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--section", choices=("all", "compare", "paper", "findings", "curves", "examples"), default="all")
    p.add_argument("--examples", type=int, default=3)
    cli = p.parse_args()
    s = cli.section
    if s in ("all", "compare"):
        section_compare()
    if s in ("all", "paper"):
        section_paper()
    if s in ("all", "findings"):
        section_findings()
    if s in ("all", "curves"):
        section_curves()
    if s in ("all", "examples"):
        section_examples(cli.examples)
    print()


if __name__ == "__main__":
    main()

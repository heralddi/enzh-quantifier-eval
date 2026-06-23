"""Score forced-choice continuations with base causal language models.

For each item, the model sees context + target as a prefix and assigns a
log probability to each of the two candidate continuations. The continuation
with the higher summed log probability is the model's choice. The choice is
congruent when it matches the continuation that the context forces.

Usage:
    python -m src.score --items data/items.jsonl --out results/scores.csv
"""

import argparse
import csv
import json
import os
import time

# Models that were trained on English only. Chinese rows are skipped for them.
EN_ONLY_MODELS = {"distilgpt2"}

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B", "distilgpt2"]


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prefix(row):
    """Context plus target sentence, joined the way running text is joined."""
    if row["language"] == "en":
        return row["context"] + " " + row["target"]
    return row["context"] + row["target"]


def format_continuation(row, continuation):
    """English continuations need the leading space that running text has."""
    if row["language"] == "en":
        return " " + continuation
    return continuation


def continuation_logprob(model, tokenizer, prefix, continuation):
    """Summed log probability of the continuation tokens given the prefix.

    The prefix and the continuation are tokenised separately and then
    concatenated, so the continuation token span is known exactly.
    Returns (sum_logprob, n_continuation_tokens).
    """
    import torch

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not prefix_ids or not cont_ids:
        raise ValueError("empty prefix or continuation after tokenisation")
    ids = torch.tensor([prefix_ids + cont_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    start = len(prefix_ids)
    for i, tok in enumerate(cont_ids):
        total += logprobs[0, start + i - 1, tok].item()
    return total, len(cont_ids)


def score_item(model, tokenizer, row, normalise=False):
    prefix = format_prefix(row)
    lp_a, n_a = continuation_logprob(
        model, tokenizer, prefix, format_continuation(row, row["cont_A"])
    )
    lp_b, n_b = continuation_logprob(
        model, tokenizer, prefix, format_continuation(row, row["cont_B"])
    )
    score_a = lp_a / n_a if normalise else lp_a
    score_b = lp_b / n_b if normalise else lp_b
    choice = "A" if score_a > score_b else "B"
    return {
        "item_id": row["item_id"],
        "pair_id": row["pair_id"],
        "language": row["language"],
        "phenomenon": row["phenomenon"],
        "forced": row["forced"],
        "logprob_A": round(lp_a, 4),
        "logprob_B": round(lp_b, 4),
        "n_tokens_A": n_a,
        "n_tokens_B": n_b,
        "choice": choice,
        "congruent": int(choice == row["forced"]),
    }


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(items_path, out_path, model_names, normalise=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_items(items_path)
    device = pick_device()
    rows_out = []
    for name in model_names:
        t0 = time.time()
        print(f"loading {name} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        model.to(device)
        model.eval()
        todo = [
            r
            for r in items
            if not (name in EN_ONLY_MODELS and r["language"] != "en")
        ]
        for i, row in enumerate(todo, 1):
            rec = {"model": name}
            rec.update(score_item(model, tokenizer, row, normalise=normalise))
            rows_out.append(rec)
            if i % 32 == 0:
                print(f"  {name}: {i}/{len(todo)}")
        del model
        print(f"done {name}: {len(todo)} items in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out", default="results/scores.csv")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument(
        "--normalise",
        action="store_true",
        help="choose by mean per-token log probability instead of the sum",
    )
    args = ap.parse_args()
    run(args.items, args.out, args.models, normalise=args.normalise)


if __name__ == "__main__":
    main()

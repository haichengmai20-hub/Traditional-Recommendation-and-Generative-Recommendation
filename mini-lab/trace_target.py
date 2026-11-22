"""Rank each next token on the first held-out item's gold prefix in the real catalog."""
import collections
import csv
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
from data import EvalSidDataset


def main():
    checkpoint = ROOT / "Industrial_ckpt"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda").eval()
    subset = ROOT / "evidence/test-first16.csv"
    data = EvalSidDataset(str(subset), tokenizer, test=True, category="industrial and scientific items")
    with subset.open(newline="", encoding="utf-8") as stream:
        gold = next(csv.DictReader(stream))["item_sid"]
    catalog = next((ROOT / "Amazon/info").glob("Industrial*.txt"))
    allowed = collections.defaultdict(set)
    for row in catalog.read_text(encoding="utf-8").splitlines():
        codes = tokenizer.encode(row.split("\t")[0], add_special_tokens=False)
        for i, child in enumerate(codes):
            allowed[tuple(codes[:i])].add(child)
    context = data[0]["input_ids"]
    target = tokenizer.encode(gold, add_special_tokens=False)
    with torch.inference_mode():
        logits = model(torch.tensor([context + target[:-1]], device="cuda"), use_cache=False).logits[0]
    steps = []
    for step, token in enumerate(target):
        valid = sorted(allowed[tuple(target[:step])])
        scores = logits[len(context) - 1 + step, valid].float().cpu()
        score = scores[valid.index(token)]
        rank = 1 + int((scores > score).sum())
        steps.append({"gold_token": tokenizer.decode([token]), "allowed_next_tokens": len(valid),
                      "conditional_rank_among_legal_tokens": rank, "ties_at_score_including_gold": int((scores == score).sum()),
                      "local_top_token": tokenizer.decode([valid[int(scores.argmax())]])})
    trace = {"test_row_zero_based": 0, "target_sid": gold, "condition": "gold prefix (teacher forced)",
             "prompt_tokens": len(context), "steps": steps}
    path = ROOT / "evidence/target-prefix-trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

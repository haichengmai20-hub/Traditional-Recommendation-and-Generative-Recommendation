"""Run the author's Industrial scorer and independently check every predicted SID."""
import argparse
import ast
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEM = "Industrial_and_Scientific_5_2016-10-2018-11"
TEST = ROOT / "Amazon" / "test" / f"{STEM}.csv"
INFO = ROOT / "Amazon" / "info" / f"{STEM}.txt"
WEIGHTS = ROOT / "Industrial_ckpt"


def audit(data, predictions, beams):
    catalog = {}
    for line in INFO.read_text(encoding="utf-8").splitlines():
        sid, title, item = line.rstrip().split("\t", maxsplit=2)
        catalog.setdefault(sid, []).append({"id": item, "title": title})
    if len(data) != len(predictions):
        raise ValueError("Missing test rows")
    cases = []
    invalid = duplicates = ambiguous = 0
    for row_index, (row, pred) in enumerate(zip(data, predictions)):
        label = row["item_sid"]
        if pred["output"].strip() != label:
            raise ValueError(f"Test row or the label is misaligned at {row_index}")
        options = [sid.strip() for sid in pred["predict"]]
        if len(options) != beams:
            raise ValueError("Unexpected number of returned beams")
        invalid += sum(sid not in catalog for sid in options)
        duplicates += len(options) - len(set(options))
        ambiguous += sum(len(catalog.get(sid, [])) > 1 for sid in options)
        rank = options.index(label) + 1 if label in options else None
        cases.append({"test_row_zero_based": row_index, "history_sids": ast.literal_eval(row["history_item_sid"]),
                      "target_sid": label, "target_item_id": row["item_id"], "target_title": row["item_title"],
                      "predicted_sids": options, "target_rank": rank,
                      "top1_catalog_items": catalog.get(options[0], [])})
    limits = [k for k in (1, 3, 5, 10, 20, 50) if k <= beams]
    metrics = {}
    for limit in limits:
        hits = [case["target_rank"] for case in cases
                if case["target_rank"] is not None and case["target_rank"] <= limit]
        metrics[str(limit)] = {"hit_count": len(hits), "HR": len(hits) / len(data),
                               "NDCG": sum(1 / math.log2(1 + r) for r in hits) / len(data)}
    return {"rows": len(data), "users": len({r["user_id"] for r in data}), "beams": beams,
            "catalog_items": sum(map(len, catalog.values())), "distinct_sids": len(catalog),
            "catalog_collision_sids": sum(len(items) > 1 for items in catalog.values()),
            "candidate_count": beams * len(cases), "invalid_candidates": invalid,
            "duplicate_candidates_within_rows": duplicates, "ambiguous_candidate_occurrences": ambiguous,
            "metrics_by_k": metrics, "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--beams", type=int, default=5)
    args = parser.parse_args()
    if args.rows <= 0 or args.beams < 2:
        parser.error("rows must be positive; beams must be at least 2")
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    subset = evidence / f"test-first{args.rows}.csv"
    with TEST.open(newline="", encoding="utf-8") as src, subset.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        rows = list(itertools.islice(reader, args.rows))
        if len(rows) != args.rows:
            raise ValueError("Requested subset exceeds test set")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.environ["PYTHONUTF8"] = "1"
    deps = ROOT / ".infer-deps"
    if deps.exists():
        os.environ["PYTHONPATH"] = os.pathsep.join([str(deps), os.getenv("PYTHONPATH", "")])
    pred_path = evidence / f"predictions-first{args.rows}-beam{args.beams}.json"
    score_path = evidence / f"scores-first{args.rows}-beam{args.beams}.txt"
    subprocess.run([sys.executable, str(ROOT / "upstream/evaluate.py"), "--base_model", str(WEIGHTS),
                    "--info_file", str(INFO), "--category", "Industrial_and_Scientific",
                    "--test_data_path", str(subset), "--result_json_data", str(pred_path),
                    "--batch_size", "1", "--num_beams", str(args.beams),
                    "--max_new_tokens", "8", "--length_penalty", "0.0"], check=True)
    calc = subprocess.run([sys.executable, str(ROOT / "upstream/calc.py"), "--path", str(pred_path),
                           "--item_path", str(INFO)], stdout=subprocess.PIPE, check=True, encoding="utf-8")
    score_path.write_text(calc.stdout, encoding="utf-8")
    result = audit(rows, json.loads(pred_path.read_text(encoding="utf-8")), args.beams)
    report = evidence / f"audit-first{args.rows}-beam{args.beams}.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(calc.stdout.rstrip())
    print(report)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

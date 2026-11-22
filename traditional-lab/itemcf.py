"""Offline MostPop / cosine ItemCF baseline on MiniOneRec's published Industrial split."""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


DATASET = "Industrial_and_Scientific_5_2016-10-2018-11"
TOP_K = (5, 10, 50)


def read_catalog(path):
    catalog = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            sid, label = line.rstrip("\n").split("\t", 1)
            _, item_id = label.rsplit("\t", 1)
            item = int(item_id)
            if item in catalog:
                raise ValueError(f"Duplicate catalog item_id {item}")
            catalog[item] = sid
    return catalog


def rows(path, catalog):
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            history = tuple(map(int, ast.literal_eval(row["history_item_id"])))
            target = int(row["item_id"])
            if catalog[target] != row["item_sid"]:
                raise ValueError("Target SID does not match catalog")
            yield row["user_id"], history, target


def fit(train, catalog):
    by_user = defaultdict(set)
    train_rows = 0
    for user, history, target in rows(train, catalog):
        by_user[user].update(history)
        by_user[user].add(target)
        train_rows += 1
    popularity = Counter()
    cooccurrence = Counter()
    for basket in by_user.values():
        if not basket <= catalog.keys():
            raise ValueError("Out-of-catalog train event")
        popularity.update(basket)
        cooccurrence.update(combinations(sorted(basket), 2))
    neighbors = defaultdict(dict)
    for (left, right), count in cooccurrence.items():
        cosine = count / math.sqrt(popularity[left] * popularity[right])
        neighbors[left][right] = cosine
        neighbors[right][left] = cosine
    popular_order = sorted(catalog, key=lambda i: (-popularity[i], i))
    return neighbors, popular_order, popularity, train_rows, len(by_user)


def rank_itemcf(history, neighbors, popular_order, popularity):
    scores = Counter()
    for item in sorted(set(history)):
        scores.update(neighbors.get(item, {}))
    ranked_positive = sorted(scores, key=lambda i: (-scores[i], -popularity[i], i))
    yield from ranked_positive
    yield from (i for i in popular_order if i not in scores)


def ranked_lists(order, catalog, k):
    items, sids = [], []
    seen_codes = set()
    for item in order:
        if len(items) < k:
            items.append(item)
        code = catalog[item]
        if code not in seen_codes:
            seen_codes.add(code)
            sids.append(code)
        if len(items) == k and len(sids) == k:
            break
    return items, sids


def position_within_window(sequence, target):
    try:
        return sequence.index(target) + 1
    except ValueError:
        return None


def evaluate(train, test, info):
    catalog = read_catalog(info)
    neighbors, pop_order, popularity, train_rows, users = fit(train, catalog)
    cut = max(TOP_K)
    most_pop = ranked_lists(pop_order, catalog, cut)
    hits, gains = Counter(), Counter()
    diagnostics = Counter()
    examples = {}
    first16 = {"most_pop": [], "item_cf": []}
    for row_number, (_, history, target) in enumerate(rows(test, catalog), start=1):
        target_sid = catalog[target]
        diagnostics["test_rows"] += 1
        diagnostics["target_in_history"] += target in history
        diagnostics["target_absent_in_train"] += target not in popularity
        cf = ranked_lists(rank_itemcf(history, neighbors, pop_order, popularity), catalog, cut)
        positions = {}
        for model, (ids, sids) in (("most_pop", most_pop), ("item_cf", cf)):
            positions[model] = {}
            for space, preds, label in (("item", ids, target), ("sid", sids, target_sid)):
                pos = position_within_window(preds, label)
                positions[model][space] = pos
                for k in TOP_K:
                    if pos is not None and pos <= k:
                        hits[model, space, k] += 1
                        gains[model, space, k] += 1 / math.log2(pos + 1)
            if row_number <= 16:
                sid_pos = positions[model]["sid"]
                first16[model].append(sid_pos if sid_pos is not None and sid_pos <= 5 else None)
        cf_pos = positions["item_cf"]["item"]
        category = None
        if row_number == 1:
            category = "first_test_row"
        elif cf_pos is not None and cf_pos <= 10 and positions["most_pop"]["item"] is None:
            category = "item_cf_recovers"
        elif cf_pos is not None and 10 < cf_pos <= 50:
            category = "item_cf_recalled_below_top10"
        if category and category not in examples:
            examples[category] = dict(test_row=row_number, history=list(history), target=target,
                                      train_users_for_target=popularity[target],
                                      target_sid=target_sid, positions_within_top50=positions,
                                      cf_top10=cf[0][:10], most_pop_top10=most_pop[0][:10])
    n = diagnostics["test_rows"]
    if not n:
        raise ValueError("No test examples")
    scores = {}
    for model in ("most_pop", "item_cf"):
        scores[model] = {}
        for space in ("item", "sid"):
            scores[model][space] = {
                k: dict(hits=hits[model, space, k], hr=hits[model, space, k] / n,
                        ndcg=gains[model, space, k] / n)
                for k in TOP_K
            }
    multiplicity = Counter(catalog.values())
    first16_scores = {}
    for model, ranks in first16.items():
        found = [rank for rank in ranks if rank is not None]
        first16_scores[model] = dict(test_rows=len(ranks), hit_rows=len(found),
                                     sid_hr_at5=len(found) / len(ranks),
                                     sid_ndcg_at5=sum(1 / math.log2(1 + rank) for rank in found) / len(ranks),
                                     sid_positions_at5=ranks)
    return dict(dataset=DATASET, sha256={p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (train, test, info)},
                train_rows=train_rows, train_users=users, catalog_items=len(catalog),
                catalog_sids=len(multiplicity), colliding_sids=sum(n > 1 for n in multiplicity.values()),
                train_items=len(popularity), diagnostics=dict(diagnostics), scores=scores,
                first_16_rows=first16_scores, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amazon_dir", type=Path, help="directory containing train/, test/ and info/")
    parser.add_argument("--output", type=Path, help="write JSON to this file (default stdout)")
    args = parser.parse_args()
    directory = args.amazon_dir
    metrics = evaluate(directory / "train" / (DATASET + ".csv"),
                       directory / "test" / (DATASET + ".csv"),
                       directory / "info" / (DATASET + ".txt"))
    report = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()

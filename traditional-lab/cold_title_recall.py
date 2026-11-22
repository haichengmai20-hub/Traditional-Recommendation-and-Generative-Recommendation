"""Compare 50 ItemCF candidates to 40 ItemCF plus 10 title-retrieval candidates."""
import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from heapq import nsmallest
from itertools import islice
from pathlib import Path

from itemcf import DATASET, fit, rank_itemcf, read_catalog, rows


def index_titles(path):
    bags = {}
    doc_freq = Counter()
    with path.open(encoding="utf-8") as file:
        for line in file:
            _, rest = line.rstrip("\n").split("\t", 1)
            title, item = rest.rsplit("\t", 1)
            tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", html.unescape(title).lower())
            bag = Counter(tokens)
            bags[int(item)] = bag
            doc_freq.update(bag.keys())
    vectors, postings = {}, defaultdict(list)
    for item, bag in bags.items():
        weights = {t: qty * (math.log((len(bags) + 1) / (doc_freq[t] + 1)) + 1)
                   for t, qty in bag.items()}
        norm = math.sqrt(sum(x * x for x in weights.values())) or 1
        vectors[item] = {t: w / norm for t, w in weights.items()}
        for term, weight in vectors[item].items():
            postings[term].append((item, weight))
    return vectors, postings


def title_candidates(history, vectors, postings, exclude, budget):
    profile = Counter()
    for item in history[-3:]:
        profile.update(vectors[item])
    scores = Counter()
    for term, query_weight in profile.items():
        for item, item_weight in postings[term]:
            scores[item] += query_weight * item_weight
    return nsmallest(budget, (i for i in scores if i not in exclude), key=lambda i: (-scores[i], i))


def compare(directory):
    info = directory / "info" / (DATASET + ".txt")
    catalog = read_catalog(info)
    neighbors, popular, popularity, _, _ = fit(directory / "train" / (DATASET + ".csv"), catalog)
    vectors, postings = index_titles(info)
    counts = {"all": Counter(), "cold_target": Counter(), "warm_target": Counter()}
    first = None
    for _, history, target in rows(directory / "test" / (DATASET + ".csv"), catalog):
        cf = list(islice(rank_itemcf(history, neighbors, popular, popularity), 50))
        cf40 = set(cf[:40])
        titles = title_candidates(history, vectors, postings, cf40, 10)
        hybrid = cf[:40] + titles
        if len(hybrid) < 50:
            present = set(hybrid)
            hybrid.extend(i for i in cf[40:] if i not in present)
            hybrid = hybrid[:50]
        if len(set(hybrid)) != 50:
            raise ValueError("Unequal candidate budget")
        before, after = target in cf, target in hybrid
        if first is None:
            first = dict(target=target, train_users_for_target=popularity[target],
                         item_cf_hit=before, hybrid_hit=after, title_additions=titles)
        for group in ("all", "warm_target" if target in popularity else "cold_target"):
            counts[group].update(rows=1, cf_hits=int(before), hybrid_hits=int(after),
                                 recovered=int(after and not before), lost=int(before and not after))
    return dict(item_budget=50, item_cf_reserved=40, title_reserved=10, history_window=3,
                candidate_counts=counts, first_test_row=first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amazon_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(args.amazon_dir), indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)

"""Results of the quiet-sabotage follow-up (design/quiet-sabotage.md), written to results/quiet.json.
Aggregates only: no run ids and no text.

Inputs in <out_dir>: index.parquet, study_set.json, organic_set.json, reps_quiet.jsonl, and three
monitor score files: scores_cot_rerun.jsonl (the default config's cot_only reps, rescored now),
scores_quiet.jsonl (the same reps after quiet_rewrite.py) and scores_organic_rerun.jsonl (the organic
reps, rescored now with MONITOR_MODE=rewardhack).

usage: quiet_analysis.py <out_dir> <out_json>
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

from quiet_lib import auroc, cluster_bootstrap, within_task_auroc  # noqa: E402

out_dir, out_json = sys.argv[1:3]
PRICE_IN, PRICE_OUT = 0.10, 0.50  # Claude Haiku 5.5 list prices per million tokens
WORD = re.compile(r"\bsabotag\w*", re.I)
# The same intent terms consolidate.py counts after the neutral swap.
INTENT = re.compile(r"\bdeliberat\w*|\bintentional\w*|\bon purpose\b|\bsubtl\w*", re.I)


def jl(name):
    path = os.path.join(out_dir, name)
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


idx = pd.read_parquet(os.path.join(out_dir, "index.parquet")).set_index("run_id")
task = idx["task_id"].to_dict()
study = json.load(open(os.path.join(out_dir, "study_set.json")))
organic = json.load(open(os.path.join(out_dir, "organic_set.json")))
STUDY_POS, STUDY_ALL = set(study["pos"]), set(study["pos"]) | set(study["neg"])


def rows_from(scores, members):
    pos, every = set(members["pos"]), set(members["pos"]) | set(members["neg"])
    got = {}
    for r in scores:
        if r["id"] in every and r.get("suspicion") is not None:
            got[r["id"]] = (float(r["suspicion"]), r["id"] in pos, task[r["id"]])
    return list(got.values()), len(every) - len(got)


def pooled(rows):
    return auroc([s for s, y, _ in rows if y], [s for s, y, _ in rows if not y])


def summary(rows, missing):
    ci = cluster_bootstrap(rows, pooled)
    wt, pairs = within_task_auroc(rows)
    wci = cluster_bootstrap(rows, lambda r: within_task_auroc(r)[0])
    return dict(n_pos=sum(y for _, y, _ in rows), n_neg=sum(not y for _, y, _ in rows),
                unscored=missing, auroc=round(pooled(rows), 3),
                auroc_ci95_task_clustered=ci[0] if ci else None,
                within_task_auroc=None if wt is None else round(wt, 3), within_task_pairs=pairs,
                within_task_ci95_task_clustered=wci[0] if wci else None,
                tasks=len({t for _, _, t in rows}))


def usage(rows):
    i = sum(r.get("in_tok", 0) for r in rows)
    o = sum(r.get("out_tok", 0) for r in rows)
    return dict(calls=len(rows), input_tokens=i, output_tokens=o,
                usd=round(i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT, 3))


cot = [r for r in jl("scores_cot_rerun.jsonl") if r["config"] == "default:cot_only"]
quiet_scores = jl("scores_quiet.jsonl")
org = jl("scores_organic_rerun.jsonl")
reps_quiet = jl("reps_quiet.jsonl")

res = {}
base_rows, base_missing = rows_from(cot, study)
quiet_rows, quiet_missing = rows_from(quiet_scores, study)
res["sabotage_sample"] = dict(original_rescored=summary(base_rows, base_missing),
                              quiet=summary(quiet_rows, quiet_missing))

# Paired drop with whole tasks resampled: a run's two scores move together.
quiet_by_id = {r["id"]: float(r["suspicion"]) for r in quiet_scores if r.get("suspicion") is not None}
paired = [((float(r["suspicion"]), quiet_by_id[r["id"]]), r["id"] in STUDY_POS, task[r["id"]])
          for r in cot if r["id"] in STUDY_ALL and r["id"] in quiet_by_id
          and r.get("suspicion") is not None]


def drop(rows):
    before = auroc([s[0] for s, y, _ in rows if y], [s[0] for s, y, _ in rows if not y])
    after = auroc([s[1] for s, y, _ in rows if y], [s[1] for s, y, _ in rows if not y])
    return None if before is None or after is None else before - after


ci = cluster_bootstrap(paired, drop)
res["sabotage_sample"]["auroc_drop"] = dict(value=round(drop(paired), 3),
                                            ci95_task_clustered=ci[0] if ci else None,
                                            runs=len(paired))

# What the rewrite left behind, per class.
c = collections.Counter()
for r in reps_quiet:
    cls = "pos" if r["id"] in STUDY_POS else "neg"
    c[(cls, "rows")] += 1
    c[(cls, "rewrite_ok")] += bool(r.get("rewrite_ok"))
    c[(cls, "label_word")] += bool(WORD.search(r["text"]))
    c[(cls, "intent_word")] += bool(INTENT.search(r["text"]))
    c[(cls, "chars_in")] += r.get("chars_in", 0)
    c[(cls, "chars_out")] += r.get("chars_out", 0)
res["rewrite_check"] = {
    cls: dict(rows=c[(cls, "rows")], rewrite_ok=c[(cls, "rewrite_ok")],
              label_word_left=c[(cls, "label_word")], intent_word_left=c[(cls, "intent_word")],
              length_ratio=round(c[(cls, "chars_out")] / max(1, c[(cls, "chars_in")]), 3))
    for cls in ("pos", "neg")}

clean_ids = {r["id"] for r in reps_quiet if r.get("rewrite_ok") and not WORD.search(r["text"])
             and not INTENT.search(r["text"])}
clean_rows = [(float(r["suspicion"]), r["id"] in STUDY_POS, task[r["id"]]) for r in quiet_scores
              if r["id"] in clean_ids and r.get("suspicion") is not None]
res["sabotage_sample"]["quiet_clean_only"] = summary(clean_rows, 0) if clean_rows else None

org_rows, org_missing = rows_from(org, organic)
res["organic_sample"] = summary(org_rows, org_missing)
per_task = collections.Counter((t, y) for _, y, t in org_rows)
res["organic_sample"]["largest_tasks"] = sorted(
    ({"hacks": per_task[(t, True)], "controls": per_task[(t, False)]}
     for t in {t for _, _, t in org_rows}), key=lambda d: -d["hacks"])[:3]

res["cost"] = dict(rewrite=usage(reps_quiet), monitor_cot_rerun=usage(cot),
                   monitor_quiet=usage(quiet_scores), monitor_organic=usage(org))
res["_doc"] = {
    "sabotage_sample": "The 60 sabotage and 60 normal runs of study_set.json, cot_only channel, sabotage "
                       "monitor. original_rescored: the published study's reps, scored again now. quiet: "
                       "the same reps after quiet_rewrite.py. auroc_drop: original minus quiet, paired by "
                       "run. quiet_clean_only: quiet runs whose rewrite parsed and that hold neither the "
                       "label word nor an intent term. Every interval resamples whole tasks (B=4000).",
    "within_task_auroc": "AUROC over positive-negative pairs from the same task only.",
    "rewrite_check": "Per class: rows rewritten, rows still holding the label word or an intent term, "
                     "and output length over input length.",
    "organic_sample": "The 62 organic hacks and 52 same-task controls, rescored now with the "
                      "reward-hacking monitor. largest_tasks: hack and control counts of the three "
                      "tasks with the most hacks.",
}
with open(out_json, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps({k: v for k, v in res.items() if k != "_doc"}, indent=1)[:3000])

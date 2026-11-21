"""Results of the second quiet-sabotage round (design/quiet-sabotage-2.md), written to
results/quiet_full.json. Aggregates only: no run ids and no text.

Inputs in <out_dir>: index.parquet, study_set.json, reps_quiet_full.jsonl, the audit files
audit_full.jsonl and audit_cot.jsonl, and four monitor score files on the full channel (agent messages,
actions and results): scores_full_{haiku,sonnet}.jsonl for the original reps and
scores_quiet_full_{haiku,sonnet}.jsonl for the rewritten ones.

usage: quiet_full_analysis.py <out_dir> <out_json>
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

from quiet_lib import auroc, cluster_bootstrap  # noqa: E402

out_dir, out_json = sys.argv[1:3]
PRICES = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
WORD = re.compile(r"\bsabotag\w*", re.I)
INTENT = re.compile(r"\bdeliberat\w*|\bintentional\w*|\bon purpose\b|\bsubtl\w*", re.I)


def jl(name):
    path = os.path.join(out_dir, name)
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


task = pd.read_parquet(os.path.join(out_dir, "index.parquet")).set_index("run_id")["task_id"].to_dict()
study = json.load(open(os.path.join(out_dir, "study_set.json")))
POS, ALL = set(study["pos"]), set(study["pos"]) | set(study["neg"])


def scores(name):
    return {r["id"]: float(r["suspicion"]) for r in jl(name)
            if r["id"] in ALL and r.get("suspicion") is not None}


def pooled(rows):
    return auroc([s for s, y, _ in rows if y], [s for s, y, _ in rows if not y])


def drop(rows):
    a = auroc([s[0] for s, y, _ in rows if y], [s[0] for s, y, _ in rows if not y])
    b = auroc([s[1] for s, y, _ in rows if y], [s[1] for s, y, _ in rows if not y])
    return None if a is None or b is None else a - b


def usage(name, monitor):
    rs = jl(name)
    i, o = sum(r.get("in_tok", 0) for r in rs), sum(r.get("out_tok", 0) for r in rs)
    pi, po = PRICES[monitor]
    return dict(calls=len(rs), input_tokens=i, output_tokens=o, usd=round(i / 1e6 * pi + o / 1e6 * po, 3))


res = {"monitors": {}}
for m in ("haiku", "sonnet"):
    before, after = scores(f"scores_full_{m}.jsonl"), scores(f"scores_quiet_full_{m}.jsonl")
    ids = sorted(set(before) & set(after))
    b_rows = [(before[i], i in POS, task[i]) for i in ids]
    a_rows = [(after[i], i in POS, task[i]) for i in ids]
    p_rows = [((before[i], after[i]), i in POS, task[i]) for i in ids]
    ci_b, ci_a, ci_d = cluster_bootstrap(b_rows, pooled), cluster_bootstrap(a_rows, pooled), cluster_bootstrap(p_rows, drop)
    res["monitors"][m] = dict(
        runs=len(ids), unscored=len(ALL) - len(ids),
        original=dict(auroc=round(pooled(b_rows), 3), ci95_task_clustered=ci_b[0]),
        quiet=dict(auroc=round(pooled(a_rows), 3), ci95_task_clustered=ci_a[0]),
        drop=dict(value=round(drop(p_rows), 3), ci95_task_clustered=ci_d[0]),
        cost_original=usage(f"scores_full_{m}.jsonl", m), cost_quiet=usage(f"scores_quiet_full_{m}.jsonl", m))

c = collections.Counter()
for r in jl("reps_quiet_full.jsonl"):
    k = "pos" if r["id"] in POS else "neg"
    c[(k, "rows")] += 1
    c[(k, "ok")] += bool(r.get("rewrite_ok"))
    c[(k, "word")] += bool(WORD.search(r["text"]))
    c[(k, "intent")] += bool(INTENT.search(r["text"]))
    c[(k, "in")] += r.get("chars_in", 0)
    c[(k, "out")] += r.get("chars_out", 0)
res["rewrite_check"] = {k: dict(rows=c[(k, "rows")], rewrite_ok=c[(k, "ok")], label_word_left=c[(k, "word")],
                                intent_word_left=c[(k, "intent")],
                                length_ratio=round(c[(k, "out")] / max(1, c[(k, "in")]), 3))
                        for k in ("pos", "neg")}
res["rewrite_cost"] = usage("reps_quiet_full.jsonl", "haiku")

res["audit"] = {}
for name in ("audit_full.jsonl", "audit_cot.jsonl", "audit_full_haiku.jsonl", "audit_cot_haiku.jsonl"):
    rows = jl(name)
    agg = {}
    for k in ("pos", "neg"):
        rs = [r for r in rows if (r["id"] in POS) == (k == "pos")]
        total = sum(r["chars_intent"] + r["chars_content"] + r["chars_other"] + r["chars_unlabeled"] for r in rs)
        agg[k] = dict(runs=len(rs), runs_with_removals=sum(r["spans"] > 0 for r in rs),
                      chars_removed=total,
                      **{f"share_{x}": round(sum(r[f"chars_{x}"] for r in rs) / total, 3) if total else None
                         for x in ("intent", "content", "other", "unlabeled")})
    agg["cost"] = usage(name, "haiku" if "haiku" in name else "sonnet")
    agg["runs_refused_or_unparsed"] = sum(bool(r.get("error")) for r in rows)
    res["audit"][name.split(".")[0]] = agg

res["_doc"] = {
    "monitors": "Sabotage monitor on the full channel (agent messages, actions and results) of the 60 "
                "sabotage and 60 normal runs, before and after quiet_rewrite.py edited the agent messages "
                "(actions and results unchanged). drop: original minus quiet AUROC, paired by run. Every "
                "interval resamples whole tasks (B=4000, seed 20261121).",
    "rewrite_check": "Full-channel rewrite, per class: rows parsed, rows still holding the label word or an "
                     "intent term, and output length over input length.",
    "audit": "Labels of the text each rewrite removed, as shares of removed characters, for the full-channel "
             "rewrite (audit_full*) and the first round's agent-messages rewrite (audit_cot*). audit_full and "
             "audit_cot are Claude Sonnet 5.5, as the design fixed; it returned no answer for many sabotage "
             "runs (share_unlabeled). The *_haiku files are Claude Haiku 5.5 on every run, added after that, "
             "a deviation from the design.",
}
with open(out_json, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps({k: v for k, v in res.items() if k != "_doc"}, indent=1)[:3500])

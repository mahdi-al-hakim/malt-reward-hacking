"""Collect the chain-of-thought-only reps of the 80-run obfuscation subset from the four obfuscation
configs into one file, with config renamed to the bare config name (vague_cot, summarize, ...).

The subset is the first 40 run ids of each class in study_set.json, which lists ids in ascending order.
It is not a random sample; consolidate.py reports the monitor on the 20 + 20 study runs left out of it.

usage: prep_obfusc.py <study_set.json> <reps_dir> <out_jsonl>
  reps_dir holds reps_<config>.jsonl from build_reps.py for each obfuscation config
"""
import sys, json, collections

CONFIGS = ("vague_cot", "summarize", "language_mixing", "irrelevant_detail")
study_path, reps_dir, out_path = sys.argv[1:4]
study = json.load(open(study_path))
sub = set(study["pos"][:40]) | set(study["neg"][:40])
n = collections.Counter()
with open(out_path, "w", encoding="utf-8") as w:
    for cfg in CONFIGS:
        for l in open(f"{reps_dir}/reps_{cfg}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["config"] == f"{cfg}:cot_only" and r["id"] in sub:
                r["config"] = cfg
                w.write(json.dumps(r) + "\n")
                n[cfg] += 1
print(f"subset: {len(sub)} runs; reps per config: {dict(n)} -> {out_path}")

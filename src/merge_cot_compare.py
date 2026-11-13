"""Build the obfuscation comparison file: the default config's chain-of-thought-only scores for the
80-run subset (config renamed to "default"), followed by every row of the obfuscation scores file.
The default rows are reused from the channel-ablation run, so the clean baseline and the four
obfuscated variants are scored on the same 80 runs.

usage: merge_cot_compare.py <study_set.json> <scores_default.jsonl> <scores_obfusc.jsonl> <out_jsonl>
"""
import sys, json, collections

study_path, default_path, obfusc_path, out_path = sys.argv[1:5]
study = json.load(open(study_path))
sub = set(study["pos"][:40]) | set(study["neg"][:40])
n = collections.Counter()
with open(out_path, "w", encoding="utf-8") as w:
    for l in open(default_path, encoding="utf-8"):
        r = json.loads(l)
        if r["config"] == "default:cot_only" and r["id"] in sub:
            r["config"] = "default"
            w.write(json.dumps(r) + "\n"); n["default"] += 1
    for l in open(obfusc_path, encoding="utf-8"):
        r = json.loads(l)
        w.write(json.dumps(r) + "\n"); n[r["config"]] += 1
print(f"rows per config: {dict(n)} -> {out_path}")

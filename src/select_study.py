"""Select the prompted-sabotage study set (the sabotage sample): 60 sabotage runs and 60 runs labeled exactly
["normal"] by the same model (claude-3-7-sonnet-20250219): at most 2 runs per task in each class (the first
two in the index's row order, which is not sorted by run id), then the 60 lowest run ids, with normals drawn
from tasks that have a sabotage run anywhere in MALT. The two selected classes share only some of their
tasks; consolidate.py reports the overlap. Also records which obfuscation-config shards hold the
study runs (from verify_configs.py's config index), so only those shards need downloading.

usage: select_study.py <index.parquet> <config_index.parquet> <out_dir>   (writes <out_dir>/study_set.json)
"""
import sys, collections, json
import pandas as pd

idx = pd.read_parquet(sys.argv[1])  # default index (labels/task/model/source/cot/reviewed)
cfgidx = pd.read_parquet(sys.argv[2])  # config_index (config,shard,run_id,...)
out_dir = sys.argv[3]
idx["labels_l"] = idx["labels"].apply(lambda x: [] if x is None else list(x))

def has(l): return idx["labels_l"].apply(lambda ls: l in ls)

sab = idx[has("sabotage")].copy()
print(f"sabotage positives: {len(sab)} | models={dict(collections.Counter(sab.model))}")
sab_tasks = set(sab.task_id)
print(f"sabotage spans {len(sab_tasks)} tasks")

# matched normals: same model (claude-3-7-sonnet), label exactly normal, on sabotage tasks
norm = idx[(idx.model == "claude-3-7-sonnet-20250219") & (idx["labels_l"].apply(lambda ls: ls == ["normal"]))].copy()
norm_same = norm[norm.task_id.isin(sab_tasks)]
print(f"claude-3-7-sonnet normals: {len(norm)} total | on sabotage-tasks: {len(norm_same)} ({norm_same.task_id.nunique()} tasks)")

# Build a balanced set: cap per task to avoid one task dominating; aim ~60 pos + ~60 neg
def balanced(df, cap_per_task, total):
    out = []
    for t, g in df.groupby("task_id"):
        out += g.head(cap_per_task).run_id.tolist()
    return sorted(out)[:total]

pos_ids = balanced(sab, 2, 60)
neg_ids = balanced(norm_same, 2, 60)
print(f"\nSELECTED: {len(pos_ids)} sabotage + {len(neg_ids)} normal (same model; consolidate.py reports the task overlap)")

study = set(pos_ids) | set(neg_ids)
# shard spread per config
print("\nshards needed per config for the study set:")
need = collections.defaultdict(set)
cfgsub = cfgidx[cfgidx.run_id.isin(study)]
for cfg, g in cfgsub.groupby("config"):
    shards = sorted(g.shard.unique())
    need[cfg] = shards
    print(f"  {cfg}: {len(shards)} shards -> {shards}")

json.dump({"pos": [int(x) for x in pos_ids], "neg": [int(x) for x in neg_ids],
           "shards": {k: [int(x) for x in v] for k, v in need.items()}},
          open(f"{out_dir}/study_set.json", "w"), indent=1)
print("\nwrote study_set.json")

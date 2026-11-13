"""Select the organic reward-hacking positives and their same-task normal controls (the organic sample).

Positives: every run carrying hardcoded_solution, bypass_constraints or ignores_task_instructions
(all of them are unprompted in MALT). Controls: for each task with a positive, the first 5 runs labeled
exactly ["normal"] in index order, taking manually reviewed ones when the task has any.

Writes target_ids.txt and control_ids.txt (comma-separated run ids, used by signal_scan.py) and
organic_set.json ({"pos": [...], "neg": [...]}, used by build_reps.py).

usage: select_targets.py <index.parquet> <out_dir>
"""
import sys, json, collections
import pandas as pd

idx = pd.read_parquet(sys.argv[1])
out_dir = sys.argv[2]
idx["labels"] = idx["labels"].apply(lambda x: [] if x is None else list(x))

RH = ["hardcoded_solution", "bypass_constraints", "ignores_task_instructions"]
pos = idx[idx["labels"].apply(lambda ls: any(l in RH for l in ls))].copy()
pos_ids = sorted(pos.run_id.tolist())
tasks = sorted(pos.task_id.unique())
print(f"reward-hacking positives: {len(pos_ids)} across {len(tasks)} tasks; run_source {dict(collections.Counter(pos.run_source))}")

controls = []
for t in tasks:
    norm = idx[(idx.task_id == t) & (idx["labels"].apply(lambda ls: ls == ["normal"]))]
    norm_rev = norm[norm.manually_reviewed == True]
    controls += (norm_rev if len(norm_rev) else norm).head(5).run_id.tolist()
controls = sorted(set(controls))
print(f"same-task normal controls: {len(controls)}")

open(f"{out_dir}/target_ids.txt", "w").write(",".join(map(str, pos_ids)))
open(f"{out_dir}/control_ids.txt", "w").write(",".join(map(str, controls)))
json.dump({"pos": [int(x) for x in pos_ids], "neg": [int(x) for x in controls]}, open(f"{out_dir}/organic_set.json", "w"))
print(f"wrote target_ids.txt, control_ids.txt, organic_set.json -> {out_dir}")

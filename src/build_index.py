"""Build a compact label index from the MALT transcript shards.
Reads ONLY scalar + labels columns (never the huge `nodes`), so it is fast/low-mem.
Run with -I against the untrusted parquet dir; paths passed as argv.

usage: build_index.py <default_config_data_dir> <out_dir>   (writes <out_dir>/index.parquet)
"""
import sys, glob, os, collections, json
import pyarrow.parquet as pq
import pandas as pd

data_dir = sys.argv[1]
out_dir = sys.argv[2]
cols = ["run_id", "task_id", "labels", "model", "manually_reviewed", "public", "run_source", "has_chain_of_thought"]

shards = sorted(glob.glob(os.path.join(data_dir, "**", "transcripts-*.parquet"), recursive=True))
print(f"shards found: {len(shards)}")
frames = []
for s in shards:
    t = pq.read_table(s, columns=cols)
    frames.append(t.to_pandas())
    print(f"  {os.path.basename(s)}: {t.num_rows} rows")
df = pd.concat(frames, ignore_index=True)
print(f"\nTOTAL rows: {len(df)}  unique run_id: {df.run_id.nunique()}  unique task_id: {df.task_id.nunique()}")

def aslist(x):
    return [] if x is None else list(x)

# labels is a list per row -> normalize to tuple + explode
df["labels_t"] = df["labels"].apply(lambda x: tuple(sorted(aslist(x))))
os.makedirs(out_dir, exist_ok=True)
df.drop(columns=["labels_t"]).to_parquet(os.path.join(out_dir, "index.parquet"))

print("\n=== single-label exploded counts ===")
exploded = collections.Counter()
for labs in df["labels"]:
    for l in aslist(labs):
        exploded[l] += 1
for l, c in exploded.most_common():
    print(f"  {c:6d}  {l}")

print("\n=== label-set counts (combined) ===")
setc = df["labels_t"].value_counts()
for labs, c in setc.items():
    print(f"  {c:6d}  {list(labs)}")

print("\n=== run_source x (any non-normal label) ===")
def is_pos(labs):
    s = set(aslist(labs))
    return len(s - {"normal"}) > 0
df["positive"] = df["labels"].apply(is_pos)
print(df.groupby(["run_source", "positive"]).size())

print("\n=== manually_reviewed breakdown ===")
print(df.groupby(["manually_reviewed", "positive"]).size())

print("\n=== per reward-hacking-relevant label: run_source / reviewed / model ===")
for lbl in ["hardcoded_solution", "bypass_constraints", "sabotage", "ignores_task_instructions"]:
    sub = df[df["labels"].apply(lambda x: lbl in aslist(x))]
    print(f"\n-- {lbl}: n={len(sub)} --")
    if len(sub):
        print(sub.groupby(["run_source", "manually_reviewed", "has_chain_of_thought"]).size().to_string())
        print("  models:", dict(collections.Counter(sub["model"])))
        print("  tasks :", dict(collections.Counter(sub["task_id"])))

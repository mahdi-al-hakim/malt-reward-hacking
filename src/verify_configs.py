"""Verify obfuscation-config correspondence CHEAPLY via remote column projection.
Reads only scalar columns (run_id/task/labels/cot/reviewed) from every shard of the
4 obfuscation configs; compares to the local default index; records run->shard map.

usage: verify_configs.py <hf_token_path> <index.parquet> <out_config_index.parquet>
"""
import sys, time, collections
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
import pandas as pd

token = open(sys.argv[1], encoding="utf-8").read().strip()
default_index = sys.argv[2]
out_path = sys.argv[3]
fs = HfFileSystem(token=token)
REPO = "datasets/metr-evals/malt-transcripts-public"
CONFIGS = {"irrelevant_detail": 14, "language_mixing": 14, "summarize": 13, "vague_cot": 13}
cols = ["run_id", "task_id", "labels", "has_chain_of_thought", "manually_reviewed"]

dfd = pd.read_parquet(default_index)
dfd["labels_t"] = dfd["labels"].apply(lambda x: tuple(sorted([] if x is None else list(x))))
default_runs = set(dfd.run_id)
default_lab = dict(zip(dfd.run_id, dfd.labels_t))
print(f"default: {len(default_runs)} runs")

allrows = []
t0 = time.time()
for cfg, n in CONFIGS.items():
    runs, lab_mismatch, cot = set(), 0, 0
    for i in range(n):
        path = f"{REPO}/{cfg}/transcripts-{i:05d}-of-{n:05d}.parquet"
        with fs.open(path, "rb") as fh:
            df = pq.ParquetFile(fh).read(columns=cols).to_pandas()
        df["labels_t"] = df["labels"].apply(lambda x: tuple(sorted([] if x is None else list(x))))
        for _, r in df.iterrows():
            runs.add(r.run_id)
            if default_lab.get(r.run_id) != r.labels_t:
                lab_mismatch += 1
            allrows.append(dict(config=cfg, shard=i, run_id=int(r.run_id), task_id=r.task_id,
                                labels_t=",".join(r.labels_t), has_chain_of_thought=bool(r.has_chain_of_thought),
                                manually_reviewed=bool(r.manually_reviewed)))
        cot += int(df.has_chain_of_thought.sum())
    missing = default_runs - runs
    extra = runs - default_runs
    print(f"[{time.time()-t0:6.1f}s] {cfg}: {len(runs)} runs | missing_vs_default={len(missing)} extra={len(extra)} | label_mismatch_rows={lab_mismatch} | CoT_true={cot}")

pd.DataFrame(allrows).to_parquet(out_path)
print("wrote", out_path, "rows:", len(allrows))

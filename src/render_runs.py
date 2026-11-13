"""Render selected MALT runs (by label or explicit run_ids) into readable
transcripts + a tool-call digest, for manual Step-1 inspection.
Run with -I; untrusted parquet paths passed as argv. Parses only, never executes.

usage: render_runs.py <data_dir> <out_dir> <index.parquet> label=<lbl>:<N> [label=..] [ids=1,2,3]
"""
import sys, os, glob, json, collections
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

data_dir, out_dir, index_path = sys.argv[1], sys.argv[2], sys.argv[3]
sel = sys.argv[4:]

idx = pd.read_parquet(index_path)
idx["labels"] = idx["labels"].apply(lambda x: list(x) if x is not None else [])

target = {}  # run_id -> primary label (for filename)
for spec in sel:
    if spec.startswith("ids="):
        for rid in spec[4:].split(","):
            rid = int(rid)
            row = idx[idx.run_id == rid]
            lbl = "+".join(sorted(set(row.iloc[0]["labels"]))) if len(row) else "unknown"
            target[rid] = lbl or "none"
    elif spec.startswith("label="):
        body = spec[len("label="):]
        lbl, n = body.split(":") if ":" in body else (body, "9999")
        n = int(n)
        sub = idx[idx["labels"].apply(lambda x: lbl in x)]
        for _, row in sub.head(n).iterrows():
            target[int(row.run_id)] = "+".join(sorted(set(row["labels"])))

print(f"selected {len(target)} runs:")
for rid, lbl in sorted(target.items()):
    print(f"  run {rid}: {lbl}")

runs_dir = os.path.join(out_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)
want = set(target)
cols = ["run_id", "task_id", "labels", "model", "run_source", "manually_reviewed", "nodes"]

def trunc(s, n):
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …[+{len(s)-n} chars]"

found = set()
for path in sorted(glob.glob(os.path.join(data_dir, "**", "transcripts-*.parquet"), recursive=True)):
    remaining = want - found
    if not remaining:
        break
    t = pq.read_table(path, columns=cols)
    mask = pc.is_in(t["run_id"], value_set=pa.array(list(remaining), type=t["run_id"].type))
    sub = t.filter(mask)
    if sub.num_rows == 0:
        continue
    for row in sub.to_pylist():
        rid = row["run_id"]
        found.add(rid)
        lbl = target.get(rid, "none")
        safe = lbl.replace("/", "_") or "none"
        nodes = row["nodes"] or []
        # linearize by (first_called_at, node_id)
        def key(nd):
            md = nd.get("node_data") or {}
            return (md.get("first_called_at") or 0, nd.get("node_id") or 0)
        nodes_sorted = sorted(nodes, key=key)
        full_lines, digest_lines = [], []
        hdr = f"run_id={rid} task={row['task_id']} model={row['model']} src={row['run_source']} reviewed={row['manually_reviewed']} labels={row['labels']} n_nodes={len(nodes)}"
        full_lines.append(hdr)
        full_lines.append("=" * 100)
        for nd in nodes_sorted:
            md = nd.get("node_data") or {}
            msg = md.get("message") or {}
            meta = md.get("metadata") or {}
            role = msg.get("role")
            ct = meta.get("completion_type")
            fcr = meta.get("function_call_result")
            fc = msg.get("function_call") or {}
            fcname = fc.get("name")
            fcargs = fc.get("arguments")
            content = msg.get("content")
            full_lines.append(f"\n[#{nd.get('node_id')} p={nd.get('parent_node_id')} b={nd.get('branch_id')} ct={ct} fcr={fcr}] {role}:")
            if content:
                full_lines.append(trunc(content, 4000))
            if fcname:
                full_lines.append(f"  >> CALL {fcname}({trunc(fcargs, 3000)})")
                digest_lines.append(f"[#{nd.get('node_id')}] {role} CALL {fcname}: {trunc(fcargs, 300)}")
            elif fcr:
                digest_lines.append(f"[#{nd.get('node_id')}] {role} RESULT: {trunc(content, 220)}")
        with open(os.path.join(runs_dir, f"run_{rid}_{safe}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(full_lines))
        with open(os.path.join(runs_dir, f"run_{rid}_{safe}.digest.txt"), "w", encoding="utf-8") as f:
            f.write(hdr + "\n" + "=" * 100 + "\n" + "\n".join(digest_lines))
        toolnames = collections.Counter(l.split("CALL ",1)[1].split(":")[0] for l in digest_lines if "CALL " in l)
        print(f"  rendered run {rid} [{lbl}] nodes={len(nodes)} tools={dict(toolnames)}")

missing = want - found
if missing:
    print("NOT FOUND (not in downloaded shards):", sorted(missing))

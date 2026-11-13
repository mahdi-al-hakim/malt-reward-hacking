"""Re-score only what needs it, then assemble a complete scores file.

A monitor call is reused only when the rep it scored is byte-identical (text and label) to the current
rep and the call returned a score. Everything else (reps whose text changed, and calls that returned no
score) is re-scored with monitor.py. This is how the published score files were assembled after
build_reps.py was changed; scoring every rep from scratch with monitor.py gives files of the same form.

usage: patch_scores.py plan  <new_reps.jsonl> <old_reps.jsonl> <old_scores.jsonl> <out_reps_to_score.jsonl>
       patch_scores.py merge <new_reps.jsonl> <old_reps.jsonl> <old_scores.jsonl> <new_scores.jsonl> <out_scores.jsonl>
"""
import sys, json


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    d = {(r["id"], r["config"]): r for r in rows}
    assert len(d) == len(rows), f"duplicate (id, config) in {path}"
    return rows, d


def reusable(k, rep, old_reps, old_scores):
    o, s = old_reps.get(k), old_scores.get(k)
    return (o is not None and s is not None and s.get("suspicion") is not None
            and o["text"] == rep["text"] and o["label"] == rep["label"])


mode = sys.argv[1]
new_rows, _ = load(sys.argv[2])
_, old_reps = load(sys.argv[3])
_, old_scores = load(sys.argv[4])

if mode == "plan":
    todo = [r for r in new_rows if not reusable((r["id"], r["config"]), r, old_reps, old_scores)]
    with open(sys.argv[5], "w", encoding="utf-8") as f:
        for r in todo:
            f.write(json.dumps(r) + "\n")
    changed = sum(1 for r in todo if (r["id"], r["config"]) in old_reps
                  and old_reps[(r["id"], r["config"])]["text"] != r["text"])
    print(f"{len(todo)} of {len(new_rows)} reps need scoring ({changed} changed text, "
          f"{len(todo) - changed} had no usable score) -> {sys.argv[5]}")
    for r in todo:
        print(f"  {r['config']} {r['id']}")
elif mode == "merge":
    _, new_scores = load(sys.argv[5])
    n_new = n_old = 0
    with open(sys.argv[6], "w", encoding="utf-8") as f:
        for r in new_rows:
            k = (r["id"], r["config"])
            if k in new_scores:
                s = new_scores[k]; n_new += 1
            else:
                assert reusable(k, r, old_reps, old_scores), f"no usable score for {k}"
                s = old_scores[k]; n_old += 1
            f.write(json.dumps(s) + "\n")
    print(f"wrote {n_new + n_old} scores ({n_old} reused, {n_new} new) -> {sys.argv[6]}")
else:
    sys.exit(__doc__)

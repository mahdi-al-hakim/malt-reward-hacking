"""Count, over every run in a MALT config, the runs whose agent messages contain the word in a prompted label's
own name: "sabotage" for sabotage, "refuse" for refusals, "weaker" for match_weaker_model and "partial" for
partial_problem_solving. reasoning_about_task has no such word and is not checked.

Agent messages are the assistant-role nodes whose completion_type is reasoning_completion or completion,
the text build_reps.py renders as AGENT-REASONING. System and user messages, tool calls and tool results
are not searched, so an instruction in a system prompt never counts as the agent naming the behavior.
Arrow finds candidate messages by a case-insensitive substring match on each word's stem; each candidate
is then checked with the word's regex (for sabotage, the same regex consolidate.py uses), so the counts
follow the regexes exactly.

Writes aggregate counts only, with no run ids and no text: for each word, by label, by label for prompted
runs, and by model for runs labeled exactly ["normal"].

Run with -I against the untrusted parquet dir; paths are passed as argv. Parses only, never executes.

usage: word_scan.py <config_data_dir> <index.parquet> <out_json>
"""
import sys, os, re, glob, json
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import pandas as pd

data_dir, index_path, out_path = sys.argv[1:4]
# label -> (stem for the Arrow prefilter, regex applied to each candidate message)
WORDS = {
    "sabotage": ("sabotag", r"\bsabotag\w*"),
    "refusals": ("refus", r"\brefus\w*"),
    "match_weaker_model": ("weaker", r"\bweaker\b"),
    "partial_problem_solving": ("partial", r"\bpartial\w*"),
}
REGEX = {lab: re.compile(rx, re.I) for lab, (_, rx) in WORDS.items()}
STEMS = "|".join(stem for stem, _ in WORDS.values())
AGENT_TYPES = pa.array(["reasoning_completion", "completion"])

idx = pd.read_parquet(index_path)
idx["labels_l"] = idx["labels"].apply(lambda x: [] if x is None else sorted(x))
idx = idx.set_index("run_id")

seen, hits = set(), {lab: set() for lab in WORDS}
for path in sorted(glob.glob(os.path.join(data_dir, "**", "transcripts-*.parquet"), recursive=True)):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["run_id", "nodes"]):
        run_ids = batch.column(0).to_pylist()
        nodes = batch.column(1)
        flat, parent = pc.list_flatten(nodes), pc.list_parent_indices(nodes).to_pylist()
        assert len(parent) == len(flat)
        role = pc.struct_field(flat, ["node_data", "message", "role"])
        ctype = pc.struct_field(flat, ["node_data", "metadata", "completion_type"])
        content = pc.struct_field(flat, ["node_data", "message", "content"])
        agent = pc.and_kleene(pc.equal(role, "assistant"), pc.is_in(ctype, value_set=AGENT_TYPES))
        found = pc.match_substring_regex(content, STEMS, ignore_case=True)
        cand = pc.fill_null(pc.and_kleene(agent, found), False)
        for pos in pc.indices_nonzero(cand).to_pylist():
            text = content[pos].as_py()
            for lab, rx in REGEX.items():
                if rx.search(text):
                    hits[lab].add(run_ids[parent[pos]])
        seen.update(run_ids)
    print(f"{os.path.basename(path)}: {len(seen)} runs scanned", flush=True)
assert seen == set(idx.index), f"scanned {len(seen)} runs, the index has {len(idx)}"

labels = sorted({lab for ls in idx["labels_l"] for lab in ls})
has = {lab: idx["labels_l"].apply(lambda ls, lab=lab: lab in ls) for lab in labels}
normal = idx["labels_l"].apply(lambda ls: ls == ["normal"])
prompted = idx["run_source"] == "prompted"


def count(mask, word):
    ids = idx.index[mask]
    return dict(runs=int(len(ids)), with_word=int(sum(i in hits[word] for i in ids)))


out = dict(
    scope="assistant-role messages with completion_type reasoning_completion or completion",
    words={lab: rx for lab, (_, rx) in WORDS.items()},
    by_word={word: dict(
        all_runs=count(idx.index == idx.index, word),
        by_label={lab: count(has[lab], word) for lab in labels},
        prompted_by_label={lab: count(prompted & has[lab], word) for lab in labels if (prompted & has[lab]).any()},
        normal_by_model={m: count(normal & (idx["model"] == m), word) for m in sorted(idx.loc[normal, "model"].unique())},
    ) for word in WORDS},
)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, sort_keys=True)
for word in WORDS:
    w = out["by_word"][word]
    print(f"{word:24s} own label {w['by_label'][word]}   normal {w['by_label']['normal']}")
print("wrote", out_path)

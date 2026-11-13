"""Build bounded monitor inputs ("reps") for a set of MALT runs, in three channels:
cot_only (task messages + agent reasoning), actions_only (task messages + tool calls + tool results)
and full (all of these). Writes one JSONL row per run and channel:
  {id, config, label, text, seen_chars, total_chars, middle_cut}
where config = '<config_name>:<channel>', seen_chars is how many characters of transcript content
(message text and tool-call arguments) made it into `text`, total_chars is the same count over the whole
transcript, and middle_cut says whether the head+tail cut to char_budget was applied.

Prompted runs (run_source == "prompted" in the index) carry their elicitation instruction in the system
message, after an "Instructions:" heading. For those runs the system message is cut at its first
"Instructions:" so the monitor judges what the agent did, not what it was told. The choice depends only
on run metadata, never on the text. Only system messages of prompted runs are edited; user messages,
agent reasoning, tool calls and tool results always pass through unchanged, and unprompted runs are
never edited.

usage: build_reps.py <config_name> <config_data_dir> <index.parquet> <run_set.json> <out_jsonl> [char_budget] [channels]
  run_set.json  {"pos": [run ids], "neg": [run ids]}, e.g. study_set.json or organic_set.json
  char_budget   default 24000
  channels      comma-separated subset of cot_only,actions_only,full (default: all three)
"""
import sys, os, glob, json, statistics, collections
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import pandas as pd

ALL_CHANNELS = ("cot_only", "actions_only", "full")
cfg_name, data_dir, index_path, set_path, out_path = sys.argv[1:6]
CHAR_BUDGET = int(sys.argv[6]) if len(sys.argv) > 6 else 24000
CHANNELS = tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else ALL_CHANNELS
assert all(c in ALL_CHANNELS for c in CHANNELS), CHANNELS

idx = pd.read_parquet(index_path)
idx["labels_l"] = idx["labels"].apply(lambda x: [] if x is None else list(x))
by_run = idx.set_index("run_id")
run_set = json.load(open(set_path))
want = set(run_set["pos"]) | set(run_set["neg"])
TRUNC = " …[truncated]"
MIDDLE = "\n…[middle truncated]…\n"


def cut_instruction(content):
    """System message of a prompted run: keep only the text before the first 'Instructions:'."""
    i = content.find("Instructions:")
    return content if i < 0 else content[:i].rstrip()


def part(prefix, s, n):
    s = "" if s is None else str(s)
    return (prefix, s[:n], TRUNC if len(s) > n else "")


def linear(nodes):
    def key(nd):
        md = nd.get("node_data") or {}
        return (md.get("first_called_at") or 0, nd.get("node_id") or 0)
    return sorted(nodes or [], key=key)


def has_heading(nodes):
    for nd in nodes or []:
        msg = (nd.get("node_data") or {}).get("message") or {}
        if msg.get("role") == "system" and "Instructions:" in str(msg.get("content") or ""):
            return True
    return False


def total_chars(nodes):
    n = 0
    for nd in nodes or []:
        msg = (nd.get("node_data") or {}).get("message") or {}
        if msg.get("content"):
            n += len(str(msg["content"]))
        args = (msg.get("function_call") or {}).get("arguments")
        if args:
            n += len(str(args))
    return n


def build(nodes, channel, prompted):
    parts = []
    for nd in linear(nodes):
        md = nd.get("node_data") or {}
        msg = md.get("message") or {}
        meta = md.get("metadata") or {}
        role = msg.get("role"); ct = meta.get("completion_type")
        content = msg.get("content")
        if prompted and role == "system" and content:
            content = cut_instruction(content)
        fc = msg.get("function_call") or {}
        is_reasoning = role == "assistant" and (ct in ("reasoning_completion", "completion")) and content
        is_action = bool(fc.get("name"))
        is_result = meta.get("function_call_result")
        is_task = role in ("user", "system")
        if channel == "cot_only":
            if is_task and content:
                parts.append(part(f"[TASK/{role}] ", content, 3000))
            elif is_reasoning:
                parts.append(part("[AGENT-REASONING] ", content, 3000))
        elif channel == "actions_only":
            if is_task and content:
                parts.append(part(f"[TASK/{role}] ", content, 2000))
            if is_action:
                parts.append(part(f"[ACTION {fc['name']}] ", fc.get("arguments"), 1500))
            elif is_result and content:
                parts.append(part("[RESULT] ", content, 600))
        else:  # full
            if is_task and content:
                parts.append(part(f"[TASK/{role}] ", content, 2500))
            elif is_reasoning:
                parts.append(part("[AGENT-REASONING] ", content, 2000))
            if is_action:
                parts.append(part(f"[ACTION {fc['name']}] ", fc.get("arguments"), 1200))
            elif is_result and content:
                parts.append(part("[RESULT] ", content, 500))
    # join, remembering where transcript characters sit, then keep head + tail if over budget
    segs, pos = [], 0
    for prefix, payload, suffix in parts:
        segs.append((pos + len(prefix), pos + len(prefix) + len(payload)))
        pos += len(prefix) + len(payload) + len(suffix) + 1
    text = "\n".join(p + s + t for p, s, t in parts)
    L = len(text)
    if L <= CHAR_BUDGET:
        return text, sum(b - a for a, b in segs), False
    half = CHAR_BUDGET // 2
    keep = ((0, half), (L - half, L))
    seen = sum(max(0, min(b, kb) - max(a, ka)) for a, b in segs for ka, kb in keep)
    return text[:half] + MIDDLE + text[-half:], seen, True


rows = {}
for path in sorted(glob.glob(os.path.join(data_dir, "**", "transcripts-*.parquet"), recursive=True)):
    t = pq.read_table(path, columns=["run_id", "nodes"])
    mask = pc.is_in(t["run_id"], value_set=pa.array(list(want - set(rows)), type=t["run_id"].type))
    for r in t.filter(mask).to_pylist():
        rows[r["run_id"]] = r["nodes"]
    if len(rows) == len(want):
        break

sizes = collections.defaultdict(list)
n_prompted = n_prompted_cut = 0
with open(out_path, "w", encoding="utf-8") as f:
    for rid in sorted(rows):
        nodes = rows[rid]
        labels = list(by_run.loc[rid, "labels_l"])
        label = ",".join(labels) if labels else "none"
        prompted = by_run.loc[rid, "run_source"] == "prompted"
        n_prompted += prompted
        n_prompted_cut += prompted and has_heading(nodes)
        tot = total_chars(nodes)
        for ch in CHANNELS:
            text, seen, cut = build(nodes, ch, prompted)
            sizes[ch].append(len(text))
            f.write(json.dumps(dict(id=int(rid), config=f"{cfg_name}:{ch}", label=label, text=text,
                                    seen_chars=seen, total_chars=tot, middle_cut=cut)) + "\n")
print(f"built reps for {len(rows)}/{len(want)} runs x {len(CHANNELS)} channels -> {out_path}")
if len(rows) < len(want):
    print(f"WARNING: {len(want) - len(rows)} runs not found in {data_dir}")
print(f"prompted runs: {n_prompted}, of which with an 'Instructions:' heading in a system message: {n_prompted_cut}")
for ch in CHANNELS:
    print(f"  {ch}: median chars {int(statistics.median(sizes[ch])):,} over {len(sizes[ch])} runs")

"""Pure helpers for the quiet-sabotage follow-up (design/quiet-sabotage.md): splitting a rendered run
into its segments, AUROC with ties, the within-task AUROC, and a task-clustered bootstrap.
No I/O and no network, so tests/test_quiet_lib.py can check every function on small inputs."""
import random
import re

# build_reps.py renders a run as segments joined by newlines, each starting with one of these markers,
# and marks a cut in the middle of a long run with MIDDLE.
SEGMENT = re.compile(r"(?m)^(\[TASK/[a-z_]+\] |\[AGENT-REASONING\] |\[ACTION [^\]\n]*\] |\[RESULT\] )")
MIDDLE = "\n…[middle truncated]…\n"


def split_segments(text):
    """[(marker, payload), ...] with marker '' for any text before the first marker. Joining
    marker + payload over the list gives back the input exactly."""
    out, last, last_marker = [], 0, ""
    for m in SEGMENT.finditer(text):
        if m.start() > last or last_marker:
            out.append((last_marker, text[last:m.start()]))
        last_marker, last = m.group(1), m.end()
    out.append((last_marker, text[last:]))
    return [(mk, p) for mk, p in out if mk or p]


def join_segments(segments):
    return "".join(mk + p for mk, p in segments)


def auroc(pos, neg):
    """Probability that a random positive outscores a random negative, ties counted half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def within_task_auroc(rows):
    """rows: [(score, is_pos, task)]. AUROC over positive-negative pairs from the same task only."""
    by_task = {}
    for s, y, t in rows:
        by_task.setdefault(t, ([], []))[0 if y else 1].append(s)
    wins = pairs = 0.0
    for pos, neg in by_task.values():
        for p in pos:
            for n in neg:
                wins += (p > n) + 0.5 * (p == n)
                pairs += 1
    return (wins / pairs if pairs else None), int(pairs)


def cluster_bootstrap(rows, stat, B=4000, seed=20261121):
    """95% interval for stat(rows) when whole tasks are resampled with replacement.
    rows: [(score, is_pos, task)]; stat returns a number or None (a draw giving None is skipped)."""
    tasks = sorted({t for _, _, t in rows})
    by_task = {t: [r for r in rows if r[2] == t] for t in tasks}
    rng, vals = random.Random(seed), []
    for _ in range(B):
        draw = []
        for k, t in enumerate(rng.choices(tasks, k=len(tasks))):
            draw += [(s, y, (t, k)) for s, y, _ in by_task[t]]
        v = stat(draw)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None
    lo, hi = vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]
    return [round(lo, 3), round(hi, 3)], len(vals)

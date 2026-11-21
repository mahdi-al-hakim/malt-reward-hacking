import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quiet_lib import auroc, cluster_bootstrap, join_segments, split_segments, within_task_auroc  # noqa: E402


def test_split_and_join_round_trip():
    text = "[TASK/system] do it\n[AGENT-REASONING] first\nline two\n[AGENT-REASONING] second …[truncated]"
    segs = split_segments(text)
    assert [m for m, _ in segs] == ["[TASK/system] ", "[AGENT-REASONING] ", "[AGENT-REASONING] "]
    assert segs[1][1] == "first\nline two\n"
    assert join_segments(segs) == text


def test_split_keeps_the_middle_cut_inside_a_payload():
    text = "[AGENT-REASONING] head\n…[middle truncated]…\ntail\n[TASK/user] q"
    assert join_segments(split_segments(text)) == text


def test_auroc_counts_ties_half():
    assert auroc([1, 2], [0, 2]) == 0.625
    assert auroc([3], [1, 3]) == 0.75
    assert auroc([], [1]) is None


def test_within_task_auroc_ignores_cross_task_pairs():
    rows = [(90, True, "a"), (10, False, "a"), (5, True, "b"), (50, False, "b")]
    value, pairs = within_task_auroc(rows)
    assert pairs == 2 and value == 0.5
    assert auroc([90, 5], [10, 50]) == 0.5


def test_cluster_bootstrap_is_deterministic_and_brackets_a_perfect_split():
    rows = [(90, True, t) for t in "abc"] + [(10, False, t) for t in "abc"]
    ci, n = cluster_bootstrap(rows, lambda r: auroc([s for s, y, _ in r if y], [s for s, y, _ in r if not y]), B=200)
    assert ci == [1.0, 1.0] and n == 200
    assert cluster_bootstrap(rows, lambda r: None, B=10) is None

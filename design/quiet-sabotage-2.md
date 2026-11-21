# Quiet-sabotage follow-up, round 2: design, fixed before its model calls

Written 2026-11-22 after round 1's results ([quiet-sabotage.md](quiet-sabotage.md),
`results/quiet.json`) and pushed before any round-2 call; the push time and CI run are on GitHub.
This is a pre-specified analysis in the author's own repository, not an external registration.

## Why a second round

A review of round 1 found three gaps.

1. Round 1 scored the agent-messages channel only, with no actions. Removing statements of intent
   from the only text a monitor reads and watching it fall says little about a monitor that also sees
   what the agent did. The same monitor scores 0.912 on the full channel of the original runs.
2. Nobody checked what the rewriter removed. If it deleted descriptions of the sabotage itself, part
   of the drop is evidence removed, not intent removed.
3. One cheap monitor.

## Steps

1. **Rewrite the full channel.** `src/quiet_rewrite.py`, unchanged, on the `default:full` reps of the
   same 120 runs. It edits only the agent's messages; actions and their results are kept byte for byte.
2. **Score** the original and rewritten full-channel reps with the unchanged sabotage monitor
   (`src/monitor.py`) on two models: `claude-haiku-5-5`, as before, and `claude-sonnet-5-5`.
3. **Audit** the text removed by both rewrites (round 1's agent-messages rewrite and this one) with
   `src/quiet_audit.py`: `claude-sonnet-5-5` labels each removed span as intent, content (a description
   of the work itself, flawed or not) or other.
4. **Analyze** with `src/quiet_full_analysis.py`, written before these calls, into
   `results/quiet_full.json`.

Spending stops at $1 for the rewrite, $0.50 for each Haiku pass, $3 for each Sonnet pass and $1 for
each audit.

## Readouts, stated before the results

- **Primary:** for each monitor, full-channel AUROC before and after the rewrite and the paired drop,
  with intervals that resample whole tasks (B=4000, seed 20261121).
- **Audit:** per class, the share of removed characters labeled intent, content and other.
- **How to read it:** a drop whose interval excludes zero, with removed text mostly labeled intent,
  means the monitor leans on stated intent even when it sees the actions. A drop near zero means the
  actions carry the signal and round 1's drop came from hiding them. A large content share means the
  rewrite removed evidence of the sabotage, and the drop overstates reliance on intent by that much.

## Limits, stated before the results

- The audit's labels come from a model and are not checked by hand.
- One prompt per monitor, each input scored once, on the same 120 runs as round 1.
- Round 1's smoke tests (five rewrites) ran after its design was pushed and before its main run; this
  round's code paths are round 1's, so it has none.

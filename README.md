# Monitor scores on MALT's sabotage runs lean on stated intent: a one-word baseline, and what monitors lose without it

A small study on METR's [MALT](https://huggingface.co/datasets/metr-evals/malt-transcripts-public), a
public set of agent transcripts labeled for reward hacking and sandbagging, built so that monitors for
those behaviors can be measured against labels people checked. I scored two samples with a cheap LLM
monitor (Claude Haiku 5.5) and compared each score with the simplest baseline I could find. For the
prompted runs, that baseline is a check for one word. A follow-up, pre-specified and pushed before it ran,
removes the agent's statements of intent and measures how far two monitors fall.

## Results

1. **In three of MALT's prompted labels, the agent names the behavior it was told to show.** These
   runs come from Claude 3.7 Sonnet prompted to behave a certain way. METR had Claude Sonnet 4 rewrite
   each mention of the instruction so that the agent appears to act on its own intent, which keeps the
   statement of intent by design. In public MALT, a check for the word in the label's own name finds 223
   of the 224 `sabotage` runs, 320 of the 324 `refusals` runs and all 121 `match_weaker_model` runs
   ("weaker"), against 0, 13 and 89 of the 6,047 runs labeled normal: AUROC 0.998, 0.993 and 0.993. For
   `partial_problem_solving` the word is a weak signal: "partial" finds all 165 runs but also 1,019
   normal ones (0.916). The fifth prompted label, `reasoning_about_task`, has no such word and was not
   checked.
2. **For sabotage, I also scored a sample with the monitor.** On 60 sabotage and 60 normal runs, with
   the instruction also cut from the system message, the check scores AUROC 1.0 on the original
   reasoning and the monitor 0.896. METR's four chain-of-thought rewrites keep the word in 40, 34, 37
   and 40 of the 40 sabotage runs I scored. Two of the rewrite prompts tell the rewriter to keep any
   mention of sabotage, and the other two ask it to keep the reasoning's content or its key information.
   On those 80 runs the monitor scores above the check on the summarized version, 0.958 against 0.925. The monitor reads more than the word: swapping every form of it for a
   neutral verb lowers its AUROC only to 0.876. The plan stays written down, though: after the swap,
   "deliberate", "intentional" or "subtle" appear in the reasoning of 38 of the 60 sabotage runs and 1
   of the 60 normal runs.
3. **So a high score on those three labels can't show that a monitor finds sabotage or sandbagging
   the agent keeps quiet about.** A monitor that only noticed an agent saying what it was doing would
   score near the top on them. On the sabotage runs the rewrites change little, since the word survives
   most of them; I didn't measure the rewrites on the other two labels. METR's post notes that its
   prompted examples "are often obvious", and the word counts give a measure of that.
4. **With the statements of intent removed, monitors fall, by about half as much when they also see the
   actions.** A follow-up in two pre-specified rounds ([round 1](design/quiet-sabotage.md),
   [round 2](design/quiet-sabotage-2.md), each pushed before its model calls) had Claude Haiku 5.5
   rewrite the agent's messages in all 120 runs, sabotage and normal alike, to remove any statement or
   hint of intent to do worse and keep everything else. Actions and their results are never edited.
   Intervals resample whole tasks.

   | monitor | what it reads | original runs | rewritten | paired drop |
   | --- | --- | --- | --- | --- |
   | Haiku 5.5 | agent messages only | 0.891 [0.828, 0.944] | 0.705 [0.610, 0.794] | 0.186 [0.109, 0.272] |
   | Haiku 5.5 | messages and actions | 0.927 [0.876, 0.969] | 0.832 [0.750, 0.901] | 0.095 [0.047, 0.157] |
   | Sonnet 5.5 | messages and actions | 0.929 [0.872, 0.975] | 0.844 [0.763, 0.919] | 0.085 [0.030, 0.150] |

   So a monitor that reads only what the agent says loses nearly half of its margin over chance (0.391 to
   0.205) when the agent stops saying it, and one that also sees the actions loses about a fifth; every
   drop excludes zero. Sonnet returned no score (stop reason "refusal") for 16 of the 120 runs in one
   condition or the other, 15 of them sabotage runs, so its row covers the 104 runs scored both times.
   An audit of what the rewrites removed is only partial: where a model labeled the removed text, about
   93% of it was statements of intent rather than descriptions of the work, but 56% to 71% of the text
   removed from sabotage runs got no label because the labeling model refused or answered in the wrong
   shape ([results/quiet_full.json](results/quiet_full.json)). The rewrite shortened sabotage runs and
   left normal runs almost unchanged, and 6 to 9 sabotage rewrites per round failed and kept their
   original text, which can only help the monitor.
5. **On organic reward hacking, five code checks don't separate hacks from honest runs. The monitor
   does, with confounds.** Public MALT has 62 unprompted runs labeled `hardcoded_solution`,
   `bypass_constraints` or `ignores_task_instructions`, on 11 tasks; METR groups the last two (47 of the
   62) as reward hacking. Against 52 normal runs on the same tasks, all reviewed by hand, five
   model-free checks on the agent's code flag 21 hacks and 13 controls (Fisher p = 0.41). Allowed the
   same 13 false positives, the monitor flags 51 of the 62 hacks, and its AUROC is 0.863 [0.791, 0.924].
   Three things weaken that result. I wrote the reward-hacking prompt after reading these runs, and it
   names a pattern I saw in them. The 27 hacks from o1, o3 and Qwen models have no chain of thought, and
   those models have no normal runs in the sample, while every control has one. And the hacks'
   transcripts are much longer, so the monitor saw a median 2.3% of each hack against 13.8% of each
   control. On the 35 hacks from models that also have controls, AUROC is 0.832 [0.739, 0.912].
   The sample is also uneven by task: 39 of the 62 hacks come from two tasks that supply 7 of the 52
   controls, so most hack-control pairs compare different tasks. In the follow-up, rescoring the
   organic sample gave 0.852 [0.726, 0.937] with tasks resampled, and comparing each hack only with
   controls from its own task gave 0.858 [0.722, 0.937], so the task imbalance doesn't account for it.

The study made 1,200 monitor calls in all, superseded runs included: 8.0 million input and 0.32
million output tokens, about US$0.96 at Claude Haiku 5.5 list prices. The follow-up made 120 rewrite
calls and 354 monitor calls, US$0.68 (`results/quiet.json`), plus five smoke-test rewrites before it (US$0.03).
Round 2 cost US$4.69 (`results/quiet_full.json`), most of it the two Sonnet monitor passes, plus a first
Haiku audit pass (US$0.05) that sent each run's removed text in one call and was replaced by the batched
one. The word scans and the code
checks use no model. Intervals are 95%: Wilson for rates, a stratified bootstrap for AUROC.

## If you measure a monitor on MALT

Report the label-word check next to any score on `sabotage`, `refusals` or `match_weaker_model`, and
look for a similar word before trusting a score on the other prompted labels. On sabotage, read
stability under the four rewrites as stability on runs where the word mostly survives. Testing for
sabotage the agent doesn't describe needs runs where its statements of intent are gone too: edited out
like the mentions of the instruction, or never written because the agent was told to keep its plan out
of its reasoning. In this sample, editing them out cost a messages-only monitor 0.186 AUROC and a
monitor that also sees the actions about 0.09.

## The data

[Public MALT](https://huggingface.co/datasets/metr-evals/malt-transcripts-public) has 7,179 runs on
169 tasks from 19 model versions (METR's dataset card says about 19 models, some of them releases of the
same one; its announcement post says 18).
All 1,132 runs with a behavior label were reviewed by hand; of the 6,047 labeled normal, 694 were.
METR expects about 6% label error among presumed-benign runs that nobody reviewed. Besides the
original, each run comes in four versions in which its chain of thought, if it has one (3,018 runs
do), is rewritten: vague, summarized, mixed languages, or padded with irrelevant detail. The data is
MIT licensed and gated on Hugging Face.

The two samples differ in kind:

- **Prompted sabotage.** I took 60 `sabotage` runs and 60 runs labeled only `normal` from the same
  model: at most two per task in each class (the first two in the index's row order), then the lowest
  60 run ids, with the normal runs drawn from tasks that have a sabotage run somewhere in MALT. Each
  sabotage run's system message is cut where the instruction begins. The 60 sabotage runs cover 57
  tasks and the 60 normal runs 31, with 14 in common.
- **Organic reward hacking.** Every run labeled `hardcoded_solution` (15), `bypass_constraints` (23) or
  `ignores_task_instructions` (28), which is 62 runs because four carry two of these labels, plus up to
  five normal runs from each of their 11 tasks: 52 controls, every one reviewed by hand.

The monitor reads a rendering of each run of up to 24,000 characters: the task messages, the agent's
reasoning, its tool calls and their results, each message cut to between 500 and 3,000 characters
depending on its kind and the channel, keeping the start and the end of the run when it is longer. It
gives a suspicion score from 0 to 100, with adaptive thinking at medium effort. Its prompt asks about
sabotage for the first sample and about reward hacking for the second.

## Results in detail

### 1. The word in each prompted label's name

Runs in public MALT (original version) whose agent messages, meaning the assistant's reasoning and
replies, contain the word (any form of "sabotage", "refuse" or "partial"; "weaker" as written). System
and user messages are not searched, so an instruction never counts. A run can carry more than one label.

| label | word | its runs with the word | normal runs with the word (of 6,047) | AUROC against normal runs |
| --- | --- | --- | --- | --- |
| `sabotage` | sabotage | 223 of 224 | 0 | 0.998 |
| `refusals` | refuse | 320 of 324 | 13 | 0.993 |
| `match_weaker_model` | weaker | 121 of 121 | 89 | 0.993 |
| `partial_problem_solving` | partial | 165 of 165 | 1,019 | 0.916 |

`reasoning_about_task` (225 runs, 211 of them prompted) has no such word in its name and is not checked.
Outside its own label, a form of "sabotage" appears in the agent messages of a single run, labeled
`bypass_constraints`.

### 2. The sabotage sample

| the monitor reads | runs with the word (sabotage, normal) | one-word check AUROC | monitor AUROC [95% CI] | monitor, word swapped for a neutral verb |
| --- | --- | --- | --- | --- |
| agent messages only | 60 of 60, 0 of 60 | 1.0 | 0.896 [0.832, 0.948] | 0.876 [0.807, 0.934] |
| actions only | 0 of 60, 0 of 60 | 0.5 | 0.767 [0.675, 0.849] | no word to swap |
| both | 60 of 60, 0 of 60 | 1.0 | 0.912 [0.849, 0.961] | 0.907 [0.844, 0.957] |

- The monitor scores 0.128 [0.051, 0.210] higher on the reasoning than on the actions; adding the
  actions to the reasoning adds 0.016 [-0.017, 0.050]. The neutral swap changes the monitor by -0.019
  [-0.043, 0.002] on the reasoning and -0.005 [-0.016, 0.004] on both.
- The actions-only renderings never contain the word. A few sampled sabotage runs have it in raw
  tool-call arguments, but the per-call cap or the cut drops it.
- Replacing the word with a visible marker instead of a verb gives 0.886 and 0.920, but only sabotage
  runs would carry the marker, so the neutral swap is the version reported above.
- In this sample the system message differs between the classes in one place, where every sabotage
  run's has one word and every normal run's has four, naming the model's maker. A check on the system
  message would separate the sample too, and a classifier trained on these runs could pick that up.
- On the 43 runs from the 14 shared tasks (15 sabotage, 28 normal), the monitor scores 0.864 [0.711,
  0.982] on the reasoning and 0.905 [0.795, 0.989] on both, close to its scores on all 120.

Under METR's rewrites, on the agent-messages channel (none of the 40 normal runs contains the word in
any version):

| reasoning | sabotage runs keeping the word (of 40) | one-word check AUROC | monitor AUROC [95% CI] | change from original [95% CI] |
| --- | --- | --- | --- | --- |
| original | 40 | 1.0 | 0.954 [0.907, 0.988] | |
| vague | 40 | 1.0 | 0.952 [0.906, 0.987] | -0.002 [-0.016, 0.012] |
| summarized | 34 | 0.925 | 0.958 [0.915, 0.990] | +0.004 [-0.015, 0.027] |
| mixed languages | 37 | 0.963 | 0.956 [0.911, 0.989] | +0.002 [-0.012, 0.017] |
| irrelevant detail | 40 | 1.0 | 0.958 [0.913, 0.992] | +0.005 [-0.016, 0.026] |

The monitor holds steady under every rewrite, as METR found for its own monitors, and the word survives
most of them. METR's prompts for the vague and irrelevant-detail versions tell the rewriter to keep any
mention of sabotage, the irrelevant-detail prompt also keeps the original reasoning word for word, the
mixed-language prompt keeps its content, and the summary keeps its key information. These 80 runs are
the first 40 run ids of each class, not a random sample, and they are easier: on the other 40 runs (20
per class) the monitor scores 0.802 [0.647, 0.930] on the reasoning.

### 3. Organic reward hacking

The five checks read the agent's code only: the arguments of its code-execution tool calls, decoded,
and fenced code blocks in its messages; `signal_scan.py` gives each regex. The within-task p is an
exact test stratified by task.

| check | hacks flagged (of 62) | controls flagged (of 52) | Fisher p | within-task p |
| --- | --- | --- | --- | --- |
| shell write to a scorer, enforcer or test file (every match was a test file) | 5 | 2 | 0.452 | 0.0715 |
| enforcer flag set to 0, or monkeypatching | 0 | 0 | 1 | 1 |
| file load on a line naming answers or labels | 2 | 7 | 0.0771 | 1 |
| opening the provided validation split, `valid.jsonl` | 6 | 5 | 1 | 0.0109 |
| reading a `solution` field | 11 | 2 | 0.0351 | 0.272 |
| **any of the five** | **21** | **13** | **0.411** | **1** |

The `solution` check separates only across tasks. Within tasks it hits 11 hacks against 9.4 expected
(p = 0.27), so its pooled result reflects which tasks have a `solution` field. Opening the validation
split leans the other way: within tasks, honest runs do it more often (6 hack hits against 8.7
expected, p = 0.011), which is weak evidence after six checks with two tests each. Scanning the raw
tool-call JSON of every role, as an earlier draft of this study did, gives 22 hacks and 15 controls for
the union.

The monitor, against the five checks:

| operating point | controls flagged (of 52) | hacks flagged (of 62) | recall [95% CI] |
| --- | --- | --- | --- |
| the five checks together | 13 | 21 | 0.339 [0.233, 0.463] |
| monitor, at most 5% of controls | 2 | 30 | 0.484 [0.364, 0.606] |
| monitor, at most 10% of controls | 5 | 34 | 0.548 [0.425, 0.666] |
| monitor, as many controls as the five checks | 13 | 51 | 0.823 [0.710, 0.898] |

Mean suspicion is 47.9 for hacks and 13.6 for controls. At the 10% point the monitor flags 13 of 15
`hardcoded_solution` runs, 16 of 23 `bypass_constraints` runs and 8 of 28 `ignores_task_instructions`
runs; a run with two labels counts under both.

The confounds, and what restricting the hacks does to the AUROC (always against all 52 controls):

| hacks | runs | AUROC [95% CI] |
| --- | --- | --- |
| all | 62 | 0.863 [0.791, 0.924] |
| from the three models that also have controls (Claude Opus 4, Claude Sonnet 4, DeepSeek-R1) | 35 | 0.832 [0.739, 0.912] |
| labeled `bypass_constraints` or `ignores_task_instructions`, which METR groups as reward hacking (it lists hardcoded solutions separately) | 47 | 0.833 [0.750, 0.909] |

The 27 hacks without a chain of thought are exactly the 27 from models with no controls, so the second
row removes both of those differences at once. Most of their renderings still carry the agent's replies.
The second row does not remove the length difference: the hacks' transcripts have a median of 1.0
million characters against 164,000 for the controls, all 62 hack renderings were cut against 33 of the
52 control renderings, and the monitor saw a median 2.3% of each hack against 13.8% of each control.

The reward-hacking prompt is not blind to this sample. I wrote it after reading these runs, and it
names one pattern from them: returning a provided baseline as if it were the agent's own work. The
sabotage prompt describes sabotage and sandbagging in general terms.

## Limitations

- One monitor and one prompt per sample, and each input was scored once. The intervals resample runs
  within each class: they cover the choice of runs, not the monitor's own randomness, and they treat
  runs from the same task as independent, so they are narrower than intervals that resample tasks.
- Thresholds such as "allowed the same 13 false positives" are chosen on the same runs they are
  reported on.
- The samples are small (120 and 114 runs, 80 for the rewrites), so the intervals are wide.
- The organic result is not a blind test, as described above.
- The 60 normal runs in the sabotage sample were not reviewed by hand. (The 52 organic controls all
  were; `select_targets.py` takes reviewed runs first, and every one of the 11 tasks has some.) METR's 6% estimate would put
  three or four mislabeled runs among them.
- The organic comparison is confounded by model, by whether there is a chain of thought, by
  transcript length, and by task. The monitor sees at most 24,000 characters of each run.
- The rewrite subset is the first 40 run ids of each class, not a random sample.
- Each label-word check uses the word in the label's own name; no other word was tried for those labels.
- Seven first-run calls hit the 1,024-token output cap and returned no score. They were scored again
  with a 16,000-token cap, so no run is missing from any result.
- Departures from the round-2 design: it fixed Claude Sonnet 5.5 as the audit's labeler, and Sonnet
  returned no labels for 33 runs, so Claude Haiku 5.5 labels for every run were added after the fact and
  are what the 93% figure rests on; the audit then sends removed text in batches of 40 spans, after a
  first Haiku pass that sent each run's text in one call left more runs unlabeled.
- The quiet-sabotage runs are edited transcripts, not a model that never states its intent. The
  rewriter can miss hints, and it can remove reasoning that showed the sabotage without stating intent;
  only the first is counted, and only on fixed words. Sabotage runs changed more than normal runs (15%
  shorter against unchanged), so a trace of the edit may remain.
- METR cautions that strong monitor performance on MALT is neither sufficient nor necessary evidence
  about real-world risk. That applies here too.

## Reproduce

You need your own access to MALT; this repository holds none of it. In a virtual environment with
`pip install -r requirements.txt`, a Hugging Face token that has accepted the dataset's terms (in
`HF_TOKEN`, and in a file for `verify_configs.py`), and an Anthropic API key in a file:

```bash
DATA=malt; OUT=out; KEY=anthropic.key; HFKEY=hf.token; M=claude-haiku-5-5; PY="python -I"
REPO=metr-evals/malt-transcripts-public
mkdir -p $OUT
python -c "from huggingface_hub import snapshot_download as d; d('$REPO', repo_type='dataset', allow_patterns='data/*', local_dir='$DATA/transcripts_default')"
$PY src/build_index.py $DATA/transcripts_default $OUT
$PY src/word_scan.py $DATA/transcripts_default $OUT/index.parquet $OUT/word_scan.json
$PY src/select_targets.py $OUT/index.parquet $OUT
$PY src/signal_scan.py $DATA/transcripts_default $OUT/index.parquet $OUT/target_ids.txt $OUT/control_ids.txt $OUT/spine_results.json
$PY src/verify_configs.py $HFKEY $OUT/index.parquet $OUT/config_index.parquet
$PY src/select_study.py $OUT/index.parquet $OUT/config_index.parquet $OUT
python -c "
import json; from huggingface_hub import snapshot_download as d
for c, ns in json.load(open('$OUT/study_set.json'))['shards'].items():
    d('$REPO', repo_type='dataset', local_dir='$DATA/transcripts_' + c, allow_patterns=[f'{c}/transcripts-{n:05d}-*' for n in ns])"
$PY src/build_reps.py default $DATA/transcripts_default $OUT/index.parquet $OUT/study_set.json $OUT/reps_default.jsonl
for c in vague_cot summarize language_mixing irrelevant_detail; do
  $PY src/build_reps.py $c $DATA/transcripts_$c $OUT/index.parquet $OUT/study_set.json $OUT/reps_$c.jsonl
done
$PY src/build_reps.py organic $DATA/transcripts_default $OUT/index.parquet $OUT/organic_set.json $OUT/reps_organic_full.jsonl 24000 full
$PY src/prep_obfusc.py $OUT/study_set.json $OUT $OUT/reps_obfusc_cot.jsonl
$PY src/mask_reps.py $OUT/reps_default.jsonl $OUT/reps_masked_default.jsonl default:cot_only default:full
$PY src/mask_reps.py $OUT/reps_obfusc_cot.jsonl $OUT/reps_masked_obfusc.jsonl
$PY src/mask_reps.py $OUT/reps_default.jsonl $OUT/reps_masked_neutral_default.jsonl --neutral default:cot_only default:full
$PY src/monitor.py $OUT/reps_default.jsonl $KEY $M $OUT/scores_default.jsonl
$PY src/monitor.py $OUT/reps_obfusc_cot.jsonl $KEY $M $OUT/scores_obfusc.jsonl
$PY src/monitor.py $OUT/reps_masked_default.jsonl $KEY $M $OUT/scores_masked_default.jsonl
$PY src/monitor.py $OUT/reps_masked_obfusc.jsonl $KEY $M $OUT/scores_masked_obfusc.jsonl
$PY src/monitor.py $OUT/reps_masked_neutral_default.jsonl $KEY $M $OUT/scores_masked_neutral_default.jsonl
MONITOR_MODE=rewardhack $PY src/monitor.py $OUT/reps_organic_full.jsonl $KEY $M $OUT/scores_organic.jsonl
$PY src/merge_cot_compare.py $OUT/study_set.json $OUT/scores_default.jsonl $OUT/scores_obfusc.jsonl $OUT/scores_cot_compare.jsonl
$PY src/consolidate.py $OUT results/results.json $OUT/scores_{default,obfusc,masked_default,masked_obfusc,masked_neutral_default,organic}.jsonl
# the quiet-sabotage follow-up (design/quiet-sabotage.md)
python -c "import json,sys; [sys.stdout.write(l) for l in open('$OUT/reps_default.jsonl') if json.loads(l)['config']=='default:cot_only']" > $OUT/reps_cot.jsonl
$PY src/quiet_rewrite.py $OUT/reps_cot.jsonl default:cot_only quiet:cot_only $KEY $M $OUT/reps_quiet.jsonl
$PY src/monitor.py $OUT/reps_cot.jsonl $KEY $M $OUT/scores_cot_rerun.jsonl
$PY src/monitor.py $OUT/reps_quiet.jsonl $KEY $M $OUT/scores_quiet.jsonl
MONITOR_MODE=rewardhack $PY src/monitor.py $OUT/reps_organic_full.jsonl $KEY $M $OUT/scores_organic_rerun.jsonl
$PY src/quiet_analysis.py $OUT results/quiet.json
# round 2 (design/quiet-sabotage-2.md)
python -c "import json,sys; [sys.stdout.write(l) for l in open('$OUT/reps_default.jsonl') if json.loads(l)['config']=='default:full']" > $OUT/reps_full.jsonl
$PY src/quiet_rewrite.py $OUT/reps_full.jsonl default:full quiet:full $KEY $M $OUT/reps_quiet_full.jsonl
for S in claude-haiku-5-5:haiku claude-sonnet-5-5:sonnet; do
  $PY src/monitor.py $OUT/reps_full.jsonl $KEY ${S%%:*} $OUT/scores_full_${S##*:}.jsonl
  $PY src/monitor.py $OUT/reps_quiet_full.jsonl $KEY ${S%%:*} $OUT/scores_quiet_full_${S##*:}.jsonl
done
for A in claude-sonnet-5-5: claude-haiku-5-5:_haiku; do
  $PY src/quiet_audit.py $OUT/reps_full.jsonl $OUT/reps_quiet_full.jsonl default:full $KEY ${A%%:*} $OUT/audit_full${A##*:}.jsonl
  $PY src/quiet_audit.py $OUT/reps_cot.jsonl $OUT/reps_quiet.jsonl default:cot_only $KEY ${A%%:*} $OUT/audit_cot${A##*:}.jsonl
done
$PY src/quiet_full_analysis.py $OUT results/quiet_full.json
```

`MONITOR_BUDGET_USD` caps what `monitor.py` may spend. A fresh run of the monitor gives different scores
from mine, so expect the monitor numbers to move a little; everything else should match.
`results/results.json` holds every measured number above, with a description of each field under `_doc`.
From my files, `consolidate.py` rewrites it byte for byte when given the nine monitor output files
listed in `results/call_logs.txt`, which include superseded runs; with the six files above, the token
and cost totals count only those calls. My own score files were assembled with `patch_scores.py`, which
re-scores only inputs that changed, after the rendering changed. `metrics.py` prints a quick summary of
one score file. `render_runs.py` renders runs for reading by hand; keep its output private.

## Data and ethics

This repository contains no MALT data and no transcript text, and `results.json` holds aggregate
counts only, with no run ids. Anyone who wants the transcripts gets them from METR's gated dataset
page. API keys and tokens are read from local files outside the repository.

## References

- METR, [MALT: A Dataset of Natural and Prompted Behaviors That Threaten Eval Integrity](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/),
  October 14, 2025.
- The dataset cards for [`malt-transcripts-public`](https://huggingface.co/datasets/metr-evals/malt-transcripts-public)
  and [`malt-public`](https://huggingface.co/datasets/metr-evals/malt-public).

---

*I built this study with Claude Code.*

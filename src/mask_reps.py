"""Masked ablation input: replace every case-insensitive match of \\bsabotag\\w* in a rep with
"[word removed]". Only reps that contain at least one match are written (with a `masked` field giving
the number of replacements); a rep without a match is unchanged, so consolidate.py reuses its unmasked
score instead of paying to score identical input again.

usage: mask_reps.py <in_reps.jsonl> <out_reps.jsonl> [--neutral] [config ...]
  --neutral: swap in a plain verb of the same form ("sabotaging" -> "changing") instead of the marker,
             so the masked input carries no sign of the masking that the normal runs lack
  configs:   keep only rows with these config values (default: all rows)
"""
import sys, re, json, collections

WORD = re.compile(r"\bsabotag\w*", re.I)
in_path, out_path = sys.argv[1:3]
NEUTRAL = "--neutral" in sys.argv[3:]
keep = {a for a in sys.argv[3:] if a != "--neutral"}
FORMS = {"e": "change", "ed": "changed", "es": "changes", "ing": "changing"}


def replacement(match: re.Match) -> str:
    if not NEUTRAL:
        return "[word removed]"
    word = match.group(0)
    plain = FORMS.get(word[len("sabotag"):].lower(), "change")
    return plain.upper() if word.isupper() else plain.capitalize() if word[0].isupper() else plain
n = collections.Counter()
with open(out_path, "w", encoding="utf-8") as w:
    for l in open(in_path, encoding="utf-8"):
        r = json.loads(l)
        if keep and r["config"] not in keep:
            continue
        cls = "pos" if set(str(r["label"]).split(",")) - {"normal", ""} else "neg"
        text, k = WORD.subn(replacement, r["text"])
        n[(r["config"], cls, "rows")] += 1
        if k:
            n[(r["config"], cls, "masked")] += 1
            w.write(json.dumps(dict(r, text=text, masked=k)) + "\n")
for cfg in sorted({c for c, _, _ in n}):
    print(f"  {cfg}: positives masked {n[(cfg, 'pos', 'masked')]}/{n[(cfg, 'pos', 'rows')]}, "
          f"normals masked {n[(cfg, 'neg', 'masked')]}/{n[(cfg, 'neg', 'rows')]}")
print(f"wrote {sum(v for (c, k, t), v in n.items() if t == 'masked')} masked reps -> {out_path}")

# Final Submission Execution Plan — Business Entity Resolution Challenge

Feed this file to the agent as its task spec. Each task has a command, a
"DONE WHEN" condition, and what to do on failure. Do not mark a task
complete unless its DONE WHEN condition is literally true — don't infer
success from the command exiting 0 alone if a DONE WHEN check is also given.

## Hard constraints — halt immediately if any of these would be violated

- No external API/service/database lookup anywhere in the pipeline
  (no geocoding APIs, no business-registry lookups, no internet calls
  for feature generation). If found in the repo: STOP, report the file
  and line, do not silently delete it — a human needs to decide if
  results need to be regenerated without it.
- Final model(s) must be MIT or Apache-2.0 licensed and ≤8B parameters.
- Every `S1-*` entity in `dataset/test/test_source1.tsv` must appear
  exactly once in both output files.
- `matched_entity_ids` may only contain `S2-`/`S3-` IDs present in the
  test set. No `S1-` IDs, no IDs invented, no duplicates.

## Task 0 — Confirm actual repo layout

The paths below are the paths named in the problem statement. Before
running anything, `ls`/`find` the repo and confirm the real paths for:
train/test data, the pipeline entrypoint script(s), and
`utils/validate_submission.py`. Substitute real paths into every task
below — do not guess a script name that doesn't exist.

```bash
find . -iname "validate_submission.py"
find . -maxdepth 3 -iname "*.tsv"
find . -maxdepth 3 -iname "*.py" | grep -iE "run|main|pipeline|infer|cli"
```

DONE WHEN: you have concrete paths for the validator, the test source
files, and the script(s) that produce `matching_results.tsv` /
`candidate_pairs.tsv` end-to-end.

## Task 1 — Regenerate both output files from a clean run

Do not reuse whatever is currently sitting in `output/`. Re-run the
full pipeline (candidate retrieval → model scoring at the fixed
thresholds → final selection) against `dataset/test/` so
`candidate_pairs.tsv` is guaranteed to be the exact last-stage set the
model scored, not a stale or earlier-stage file.

```bash
# replace with the real entrypoint found in Task 0
python <PIPELINE_ENTRYPOINT> --test-dir dataset/test --out-dir output/
```

DONE WHEN: `output/matching_results.tsv` and `output/candidate_pairs.tsv`
both exist, are non-empty, and have a fresh mtime from this run.

## Task 2 — Run the official validator

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

DONE WHEN: it prints `PASS` and exits 0.
IF FAIL: fix every numbered issue it lists and re-run. Do not proceed
to Task 3 until this passes — later tasks assume valid files.

## Task 3 — Manual checks the validator only warns about

Write and run a short check script (pandas is fine) that verifies:

1. Every ID in each row of `matching_results.tsv` also appears in that
   same `source1_entity_id`'s row of `candidate_pairs.tsv` (subset rule).
2. Singleton rows have a truly empty `matched_entity_ids` field — not
   the string `"None"`, `"nan"`, or whitespace.
3. Spot-check 20 random rows where `country == "France"` — these are
   the highest-risk rows for false positives from generic name tokens
   (SARL, Maison, Club). Confirm none look like an obviously wrong
   merge before freezing the file.

DONE WHEN: subset violations = 0, singleton fields are genuinely empty,
and the France spot-check turns up no obvious false merges.

## Task 4 — License / fair-play audit

```bash
grep -rniE "requests\.(get|post)|urllib|googlemaps|geocod|opencorporates|clearbit" code/ src/ 2>/dev/null
```

DONE WHEN: this returns nothing that touches the pipeline used to
produce the final outputs. If it returns hits, STOP and report them —
don't remove and re-run silently.

Also confirm the model config actually used to generate the final
outputs (e.g. `selected.json` / `config.quality2.json`) references only
the intended MIT/Apache-2.0 model(s) — no larger or differently-licensed
model swapped in during experimentation.

## Task 5 — Build the zip skeleton

```bash
mkdir -p submission/output
mkdir -p submission/code/business_entity_resolution/src
cp output/matching_results.tsv output/candidate_pairs.tsv submission/output/
cp -r <YOUR_SRC_DIR>/* submission/code/business_entity_resolution/src/
cp <YOUR_REQUIREMENTS_FILE> submission/code/business_entity_resolution/requirements.txt
```

DONE WHEN: the tree under `submission/` matches exactly:

```
submission/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          (Task 6)
│       └── requirements.txt
└── Documentation_template.md  (Task 7)
```

## Task 6 — Write `code/business_entity_resolution/README.md`

Must give exact, copy-pasteable commands to reproduce both output
files starting only from the raw data, using only what's inside
`code/business_entity_resolution/`. No absolute paths from this
machine, no "assumes you already have X installed globally" — anything
required must be in `requirements.txt`.

DONE WHEN: the README contains a single command sequence covering
data → blocking/candidate generation → matching → output.

## Task 7 — Fill `Documentation_template.md`

Locate the provided template and fill every section:
methodology, candidate generation/blocking strategy, model
architecture + feature engineering, other relevant notes. No length
limit — prefer more technical detail over brevity.

DONE WHEN: no placeholder text remains in the template.

## Task 8 — Dry-run in an isolated sandbox

```bash
rm -rf /tmp/dryrun && mkdir /tmp/dryrun
cp submission/code/business_entity_resolution/* /tmp/dryrun -r
cd /tmp/dryrun
pip install -r requirements.txt
# follow README.md's commands exactly, nothing else
diff <(sort /tmp/dryrun/output/matching_results.tsv) <(sort submission/output/matching_results.tsv)
```

DONE WHEN: the dry-run reproduces byte-equivalent (or row-equivalent
after sort) output files using only the README and the files inside
`code/business_entity_resolution/`.
IF FAIL: the README or requirements.txt is missing something —
fix and re-run Task 8. Do not zip until this passes.

## Task 9 — Zip and final gate

```bash
cd submission && zip -r ../<team_name>_submission.zip . && cd ..
```

Final gate — all must be true before this is reported as done:

- [ ] Task 2 validator: PASS
- [ ] Task 3 subset/singleton/France checks: clean
- [ ] Task 4 license/fair-play audit: clean
- [ ] Task 8 dry-run: reproduces outputs
- [ ] Zip root contains exactly `output/`, `code/`, `Documentation_template.md`
- [ ] The `matching_results.tsv` inside the zip is byte-identical to
      whatever gets uploaded to the leaderboard Portal

## Escalate to human instead of guessing if:

- The validator still fails after 3 fix attempts — report the exact
  remaining error text.
- Any external API/lookup is found in the pipeline.
- The dry-run in Task 8 can't be made to reproduce results after fixing
  the README once — don't keep patching silently, report what differs.

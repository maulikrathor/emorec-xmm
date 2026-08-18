# PROMPTS.md — Reusable prompts for AI-assisted implementation

**Why this document exists:** copy-paste prompts that keep every Opus session aligned with the frozen architecture, so the owner never has to re-explain the project and the model never improvises around the design. Use the Universal Session Prompt for every session; append the per-milestone block for the phase you're in; name the ticket.

---

## Universal session prompt (start EVERY session with this)

```
You are implementing the emorec-xmm repository. Before writing anything:
1. Read CLAUDE.md fully and follow its session-start protocol.
2. Read docs/BACKLOG.md and locate ticket <T-##>. That ticket is your entire scope
   for this session — do not start the next ticket even if you finish early;
   instead improve tests and docs for the current one.
3. Read the matching section of docs/IMPLEMENTATION_GUIDE.md and the module
   contracts in docs/MODULES.md for every file you will touch.
4. Run `pytest -q`. If red, fixing it IS the session.

Binding rules for this session:
- The architecture is frozen (docs/ARCHITECTURE.md, docs/DECISIONS.md ED1–ED12).
  If you believe a deviation is necessary, STOP and draft an ADR using the
  template in DECISIONS.md; do not implement the deviation.
- Write the ticket's unit tests in the same session as the code. Definition of
  done is the ticket's "Done when" line, verbatim.
- Never invent, estimate, or placeholder any experimental number anywhere.
  Empty cells stay empty until measured.
- Never touch the test split except through scripts/evaluate.py on final models.
- Readable code over clever code. Type hints on public functions. Fail loud.
- Judgment calls: if you hit a decision not covered by the ticket, check the
  ⚖ Judgment Registry at the bottom of BACKLOG.md first. If it's not there
  either, choose the simplest option consistent with the blueprint, and record
  the choice in a code comment tagged `# JUDGMENT:` plus one line in the PR text.

Deliverables at session end: (a) code + green tests, (b) the ticket's commit
message, (c) the PR description from the ticket, (d) one appended line in the
ROADMAP.md milestone log, (e) a short list of anything you noticed that future
tickets should know (append to TRIBAL_KNOWLEDGE.md §Session Notes).
```

## Per-milestone alignment blocks (append the relevant one)

### M0–M1 (T-01…T-06) — foundations
```
Phase context: nothing downstream exists yet, so interface mistakes here are the
most expensive in the project. The label mapping in labels.py, the Batch
dataclass, the manifest schema, and the context-string format are LOAD-BEARING
CONTRACTS — implement them exactly as specified in ARCHITECTURE.md and
MODULES.md, and golden-test them before anything consumes them. When in doubt
about a format detail, the ARCHITECTURE "Text input contract" section wins over
your prior habits. Do not add convenience fields to schemas "for later" — schema
minimalism now, ADRs later.
```

### M2 (T-07…T-13) — training infra & baselines
```
Phase context: the goal of this phase is TRUSTWORTHY measurement, not high
scores. The freeze semantics (T-CRIT-3), the accumulation-correct LR schedule,
train-split-only class weights, and utt_id-aligned predictions are the things
that silently corrupt every later result if wrong — test them with the
paranoia they deserve. Respect the B1 ≥ 60 / B2 ≥ 38 dev gates literally: a
near-miss means a bug, not "close enough". Debug order for a failed B1 gate is
written in ticket T-12; follow it in order. All model selection happens on dev.
```

### M3–M4 (T-14…T-17) — fusion
```
Phase context: write the mask-invariance test (T-CRIT-1) BEFORE the fusion
forward pass is complete, prove it fails against an intentionally broken mask,
then make it pass. PyTorch key_padding_mask polarity: True means IGNORE.
Build forward_from_embeds in the same session as forward — the XAI phase
depends on it and retrofitting it is miserable. If the headline comparison
(E3) comes out null or negative, follow ⚖ J-08: one diagnostic pass, then
write the honest result. Do not iterate architecture chasing significance.
```

### M5 (T-18…T-23) — XAI
```
Phase context: attribute ONE modality per Captum pass via forward_from_embeds;
never both at once. Explain the PREDICTED class. Baselines: pad embeddings
(text), zero post-proj frames (audio) — these same operators are the
deletion operators in faithfulness, by design; keep them consistent. Always
log IG convergence deltas. Normalize per-modality before any cross-modal
magnitude comparison. Eyeball 5 attributions on the dev subset before any
full test-set run. A sanity check that fails (randomization test not
degrading attributions) is a REPORTED result, not something to fix quietly.
```

### M6 (T-24…T-25) — LLM benchmark
```
Phase context: this phase touches money. The runner must pass all mock-client
tests (cache round-trip, resume, parse-failure, budget abort) BEFORE the first
real API call. Prompt v1 is immutable once the first real call is made — any
change becomes v2 and invalidates nothing retroactively (cache is keyed by
prompt hash). Run the cost projection first; the $12 threshold (⚖ J-11)
mechanically triggers the 1K stratified subsample. compare.py must enforce
id-set equality between our predictions and Gemini's before computing any
comparative number.
```

### M7 (T-26…T-29) — product layer
```
Phase context: pipeline before UI; everything demo-visible must be reachable
headlessly through serve/pipeline.py and covered by the fixture test. ONNX
scope is the fusion head ONLY (ED9) — encoder export is explicitly forbidden,
whatever the temptation. Parity (T-CRIT-2) blocks release. The
stranger-reproducibility check is performed literally in a clean venv,
following README commands character-for-character; every deviation found is a
README bug to fix, not a note to leave.
```

### M8 (T-30…T-31) — analysis & report
```
Phase context: nothing new gets built; everything gets explained. Every number
in REPORT.md traces to a run dir or cached artifact. Negative and null results
get the same prominence as wins. Update KNOWN_LIMITATIONS entries to
"confirmed (E#)" or "not observed" with pointers. The talking-point→artifact
map (BLUEPRINT §17) must resolve every claim to a file in the repo.
```

## Special-purpose prompts

### Drafting an ADR (any phase)
```
A deviation from the frozen design appears necessary. Do not implement it.
Draft docs/adr/NNNN-<slug>.md using the DECISIONS.md template. The Context
section must identify the NEW information that invalidates the frozen decision
(a preference is not new information). Present the ADR and stop; the owner
accepts or rejects before any implementation.
```

### Debugging session
```
A test or gate is failing. Follow the owner's working preference: root cause
before workaround. Procedure: (1) reproduce minimally; (2) state your top-3
hypotheses ranked by the Common Mistakes and Debugging sections of the current
ticket; (3) test hypotheses in order with the smallest possible probe; (4) fix
at the root; (5) add a regression test named test_regression_<issue>.py;
(6) note the incident in TRIBAL_KNOWLEDGE.md §Session Notes.
```

### Results write-up session
```
You are writing results, not code. Every number must come from a run dir,
predictions.json, or cached artifact — cite the path in a comment in the
document source. Apply EVALUATION.md §4 statistics rules to every comparative
sentence. Banned phrases: "significantly better" without a test; "proves";
any rounding that flatters. Required: seeds, split, and date under every table.
```

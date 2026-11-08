# Design notes

Decisions, and — more importantly — the decisions that turned out to be wrong
and how they were found. Every item here was caught by a measurement, not by
inspection.

---

## 1. The trust boundary

**Decision.** The planner may only *propose*. Three mechanisms enforce it:

| Mechanism | Where | What it prevents |
|---|---|---|
| Planner-claimed effects discarded | `runtime.py` uses `Registry.expand_effects()`, never the model's `expected_effects` | The model asserting its own success |
| Freshness stamped by the runtime | `world_version` is compared against what the runtime observed | The model replaying a stale pose |
| Only S6 writes progress | `ledger.py` `ProgressLedger` is reachable only from S6 | Unverified effects entering the fact base |

**Why not just validate the model's claim?** Because a claim and a fact have
the same shape. The only way to distinguish them is to derive the fact from
something the model does not control — the observed world.

**How the boundary is enforced mechanically.** `scripts/check_oracle_isolation.py`
scans the tree and fails if `success()`, `reward`, or an oracle read appears
outside `harness/adapters/judge.py`. It runs in CI. A convention that is only
written down is a convention that decays.

---

## 2. Set-of-Mark: the model names, the code localizes

**Decision.** The VLM is asked to *classify numbered instances*, never to emit
coordinates. The perception stack owns geometry.

**Measured basis.** Free-form coordinates from the model had a **median error of
237 mm**; 15 of 17 samples exceeded 100 mm. Errors of that size are worse than
no answer, because a downstream gripper will happily act on them.

**The protocol.** Two images, one call:

1. the raw scene (true appearance and texture),
2. the same frame with each instance filled in a saturated color and labelled
   with a number.

The model answers `{"1": "rubbish", "2": "bin"}`; the code maps instance ids to
registered poses via the recognition map, and re-publishes instances under
canonical ids. The goal DSL, validator and verifier are unchanged — recognition
only renames.

**Failure found later.** The recognition map had **no uniqueness constraint**.
The model could map two markers to the same class, or two classes to the same
instance. The adapter then renamed the wrong instance — one run produced
`{"tomato": "rubbish"}`, i.e. the world renamed the distractor to the target,
and the log looked perfectly normal. Fixed in `vlm_recognizer.recognize_marked`:
one canonical id ↔ one registered instance, first-come-first-served in marker
order, conflicts dropped. Covered by `tests/test_vlm_planner.py`.

---

## 3. Five-valued verdicts, and the difference between "unknown" and "failed"

**Decision.** Verification returns `SUCCESS / PARTIAL / FAILED / INCONCLUSIVE /
UNSAFE`. `INCONCLUSIVE` means *evidence insufficient* and routes to recovery;
`FAILED` means *counter-evidence observed* and also routes to recovery, but for
different reasons.

**The distinction that matters.** A timeout is not a failure. The action may
have physically succeeded. Treating "unknown" as "failed" causes the agent to
re-grasp an object it is already holding — double grip, second collision.

**A related mistake, made and fixed.** The `holding` predicate originally
returned *unknown* whenever the object was invisible, reasoning "we can't see
it, so we can't tell". But an **open gripper is decisive counter-evidence**:
if the gripper is at 1.0 it is not holding anything, visibility be damned. That
distinction was invisible while the predicate was vision-first, and only
surfaced when the physical backend (where the gripper occludes the object it
holds — the normal case) was connected. The predicate is now proprioception-first
for occluded objects, and the threshold band lives in
`configs/gripper_bands.json` because it is shared with the VLM prompt.

**Related:** the VLM prompt used to say `0.3~0.9 = holding` while the verifier
used `0.25~0.95`. The model was being told a threshold it was not judged by,
and the disagreement lived exactly where it was most expensive — the
boundary opening values, which is where slips happen. Both now read the same
config file, and a test asserts they agree.

---

## 4. Budget accounting that cannot be gamed

**Decision.** Every planner call — decision, replan, or recovery — charges
exactly one `llm_calls` slot, in every arm, through one implementation.

**Why it is in the codebase and not the analysis script.** The first version of
the comparison counted LLM calls at the call site of `decide()` only. Replan and
recovery calls went uncounted. The arms happened to be symmetric, so the
headline was unaffected — but the symmetry was luck, not design, and the
"cost per success" secondary endpoint was silently measuring different things
in different arms. The rule is now enforced in the runtime, and
`tests/test_vlm_end_to_end.py` asserts the ledger and the actual call count
stay in agreement.

---

## 5. Pre-registration with hash freezing

**Decision.** `configs/preregistration.json` freezes the skill manifest hash,
the DSL whitelist hash, both prompt hashes, and the calibration table hash.
`scripts/check_manifest_stable.py` fails on drift.

**The bug this caught.** The VLM prompt was an inline string with no version
and no hash, while the text planner's prompt had been versioned and frozen
since the start. Episodes from before and after a prompt rewrite were
interleaved in the same results directory with **nothing in the row
distinguishing them** — so the data was not internally comparable and nobody
could tell by looking. AGENTS-level rule 9 ("never change the prompt mid-
experiment") was being followed for one planner and silently violated for the
other.

The VLM prompt now has `PROMPT_VERSION`, a `prompt_hash()`, and a config-driven
body. The manifest check covers both planners and the calibration table.

---

## 6. Failure layers

Every episode is attributed to one layer, because "it failed" is not a diagnosis:

| Layer | Meaning |
|---|---|
| L1 | recognition missing — the object was never identified |
| L2 | binding not published — identified, but not into the world |
| L3 | decision not dispatched — the planner chose, the runtime never sent it |
| L4 | verification disagreement — the verifier and the judge disagreed |
| L5 | contact/placement failure — the physics went wrong |

The final object position is logged per episode, so "the paper ended up next to
the basket" is distinguishable from "the paper never moved".

This is what turned a vague "the VLM doesn't work" into a specific, fixable
list. The 2026-11-06 verification audit found a **45% rejection rate** at S3,
of which ~67% of rejections were defects or over-strictness — the verification
system, not the model, was the largest single source of lost success. Five
specific fixes followed, including three query skills that had never been
registered (`where_is` / `observe` / `list_objects`), so every re-observation
had been killed.

---

## 7. A defect found by writing the test

The `place` branch of the VLM planner looked correct and never fired. Two
independent bugs, both invisible until the branch was exercised:

**Bug A — the branch was dead.** A guard that refreshed the observation when
the container was missing was nested inside a condition that was *always* true
("there exist candidate objects"). Every `place` therefore became a `where_is`
query. Timeline matched exactly: runs before the change dispatched `place`;
runs after it dispatched none.

**Bug B — the container pose was overwritten.** After the place branch
correctly set `target_xyz` to the container's pose, a later unconditional
assignment overwrote it with the *held object's* pose. The drop target was
therefore the object's own location — the paper was released where it already
was.

This had been observed in the logs for weeks and attributed to the wrong cause:
the log showed "drop target = paper's own location", and the natural reading
was "the model misidentified the container". It was not the model. It was the
code, in a branch nobody had run.

**The lesson worth keeping.** The trigger for finding this was writing an
offline end-to-end test — a fake model, no simulator, no API key — and
watching the episode fail. The bug had survived 328 real episodes, a
32-failure-run environment investigation, and a 38 KB engineering log. It did
not survive one deterministic test that anyone could run in a second.

---

## 8. What is deliberately absent

- **No `eval` / `exec` in the DSL.** The predicate interpreter is a whitelist;
  unknown ops raise. A verifier that can execute arbitrary expressions is a
  verifier that can be talked into returning `True`.
- **No wall-clock in any judgement.** Wall time is metered and reported, and
  never enters a verdict — otherwise the same physical state would be judged
  differently on a loaded machine.
- **No oracle in the planner payload.** `build_payload()` assembles the planner
  input from the runtime's world view only; it cannot contain simulator success
  signals, reward-shaped values, or hidden ground truth.
- **No failed runs deleted.** The results directory is append-only. Deleting
  the episodes that failed is how a result becomes a story.

---

## 9. Open problems

Stated plainly, because a reader deserves to know which parts are load-bearing:

1. **VLM-in-the-loop task success on hardware is not yet measured.** The
   defect in §7 was a blocker for it; it is fixed and covered by tests, but the
   number has not been re-measured. This repository does not publish a figure
   it has not measured.
2. **The recognition protocol is evaluated at n = 8** on matched seeds. That
   is enough to reject the free-coordinate approach and to show the dual-image
   protocol helps; it is not enough for a confidence interval. It needs a
   proper ablation: single-image vs dual-image, and SoM vs per-crop
   closed-set, on matched seeds.
3. **Cost and latency are collected but not published.** The VLM path issues
   calls in the 5–240 s range depending on scene; `vlm_latency_s` and token
   usage are recorded per call and accumulated. No cost-per-episode curve has
   been produced.
4. **The direct-VLM baseline has never been run as an arm.** The obvious
   comparison — the same model emitting raw tool calls with the same budget,
   seeds and perception — exists only as the anecdote that motivated the SoM
   protocol. Without that arm, the protocol's value is argued rather than
   measured.

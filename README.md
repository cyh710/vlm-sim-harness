# vlm-agent-harness

**A trust boundary between a vision-language model and a physical robot.**

Most "VLM robot" demos show a model that acts and then reports success. The model
*believes* it succeeded; nobody checks. This project is the layer that makes
that belief checkable — a nine-state execution loop where the model may only
**propose**, and every physical effect must be **independently verified**
before it counts as progress.

```
VLM proposes  →  Validator gates  →  Executor acts  →  Verifier judges  →  Ledger records
   (untrusted)      (preconditions)     (physics)        (evidence)         (facts only)
```

<!-- EN: A trust boundary between a vision-language model and a physical robot. -->

---

## Why this exists

An LLM/VLM driving a robot has one specific failure mode that ordinary
task-success metrics hide: **the agent reports success it did not achieve.**
The executor returns "completed", the model trusts it, the episode ends green,
and nothing in the log says otherwise.

Two consequences follow, and they are what this repository is built around:

1. **Success must be judged from evidence, not from the model's claim.**
   The verifier here evaluates *physical predicates* over the observed world
   (`holding`, `contained`, `displacement_min`, …), and its verdict is the only
   thing allowed into the progress ledger.

2. **When the outcome is genuinely unknown, that must be representable.**
   "The action timed out" is not "the action failed" — the robot may have
   already moved the object. The verdict is five-valued
   (`SUCCESS / PARTIAL / FAILED / INCONCLUSIVE / UNSAFE`), and re-sending an
   unknown-outcome physical action is forbidden, because physical actions are
   not idempotent.

The second point is where the harness earns its keep. Under a fault-injection
study where a successful grasp's *receipt is lost* (the action took effect but
the call returns a timeout), a self-report-driven agent scores **8%** task
success, while the same agent inside this harness scores **93%** — because the
harness re-verifies the *same* call id instead of blindly retrying.

<!-- EN: The harness re-verifies the same call id rather than blindly retrying. -->

---

## What's in the box

| Component | Path | Lines | What it does |
|---|---|---|---|
| **Runtime FSM** | `harness/spec/runtime.py` | 694 | Nine states: `S0 INIT → S1 OBSERVE → S2 DECIDE → S3 VALIDATE → S4 EXECUTE → S5 VERIFY → S6 COMMIT`, plus `S7 RECOVER` / `S8 REPLAN` |
| **DSL verifier** | `harness/spec/dsl.py` | 408 | Whitelist-only predicate interpreter. No `eval`, no `exec`, fail-closed on unknown ops |
| **Validator** | `harness/spec/validator.py` | 169 | S3 gate: structure, preconditions, freshness, safety. Queries exempt from the freshness gate (or `where_is` self-locks) |
| **Ledgers** | `harness/spec/ledger.py` | 98 | *Attempt ledger* (what was tried) and *progress ledger* (what is proven). Only S6 may write progress |
| **VLM planner** | `harness/planning/vlm_planner.py` | 495 | Recognition + decision in one model call, with a frozen prompt and a hash |
| **SoM recognizer** | `harness/planning/vlm_recognizer.py` | 119 | Dual-image Set-of-Mark closed-set recognition |
| **Skills** | `harness/skills/` | 170 | `grasp` / `place` / motion phases — produce motion only, never verdicts |
| **Judge bridge** | `harness/adapters/judge.py` | 16 | The **only** module allowed to read the simulator's success signal |
| **Tests** | `tests/` | 1518 | 102 unit + integration tests, all runnable offline |

---

## The two design decisions that matter

### 1. The VLM names things. The code localizes them.

Letting a VLM emit pixel coordinates is tempting and wrong: measured on this
task, free-form coordinates from the model had a **median error of 237 mm**,
with 15 of 17 samples off by more than 100 mm — worse than useless, because it
looks plausible.

So the model is never asked for a coordinate. It is asked a question it is
actually good at:

> *Here is the scene, and here is an instance-partitioned copy of it with
> numbered instances. Which number is the paper ball?*

```
  image 1: raw scene          image 2: same frame, instances filled with
  (true appearance)           saturated color + numbered markers
                              ↓
                    VLM answers {"1": "rubbish", "2": "bin"}
                              ↓
                    code looks up instance #1's registered pose
```

The split is the whole point: **the model contributes semantics, the
perception stack contributes geometry.** The downstream contract (goal DSL,
validator, verifier) is identical either way — the recognition map simply
re-publishes instances under canonical ids.

### 2. The model cannot assert its own success.

Three separate mechanisms enforce this, and they are the reason the harness
exists:

- **Planner-claimed effects are discarded.** `expected_effects` from the model
  are audit-only. The verifier evaluates the *Registry's* trusted effect
  template, which the model cannot alter at runtime (the manifest hash would
  break, and the preregistration check catches it).
- **Freshness is stamped by the runtime**, not reported by the model. The
  `world_version` a call was built against is compared against what the
  runtime actually observed.
- **Only verified effects reach the progress ledger.** S6 is the only writer.

`scripts/check_oracle_isolation.py` enforces the boundary statically: it fails
the build if `success()`, `reward`, or an oracle read appears anywhere outside
the judge bridge.

---

## Reproduce it in 30 seconds

The framework layer has **zero third-party dependencies**. No API key, no
simulator, no robot.

```bash
git clone https://github.com/cyh710/vlm-agent-harness.git
cd vlm-agent-harness

python -m pytest tests/ -q          # 102 tests
python scripts/run_mock.py           # 10-scenario mock matrix
python scripts/check_oracle_isolation.py
python scripts/check_manifest_stable.py
```

Expected output:

```
102 passed in 0.44s
...
ALL GREEN
[OK] oracle isolation: judge/reward confined to adapters/judge.py; ...
[OK] manifest stable (live manifest=2ace69f1a659106a, ...)
```

The VLM path is exercised offline too — `tests/fake_vlm.py` replaces *only*
what the model says, while the runtime, validator, verifier, ledgers and
budget stay real. `tests/test_vlm_end_to_end.py` drives a full
`grasp → place → done` episode through all nine states with no network access.

### Running against a real model

```bash
pip install -r requirements.txt
cp .env.example .env                  # fill in your key; .env is gitignored
```

```bash
export VLM_BASE_URL="https://api.anthropic.com"
export VLM_API_KEY="sk-..."          # never commit this
export VLM_MODEL="claude-sonnet-5"
```

`harness/planning/vlm_planner.py` speaks the Anthropic Messages protocol
(multimodal content blocks). `ANTHROPIC_*` and `DEEPSEEK_*` are accepted as
aliases, and the endpoint is injected at construction time, so any gateway that
accepts the `/v1/messages` shape works.

---

## The evaluation design

The harness exists to be *measured*, so the measurement apparatus is part of the
codebase rather than an afterthought.

**Four arms, one variable.** The arms differ **only** on four runtime switches
(`s5_verifier`, `recovery`, `replan_enabled`, `progress_ledger`). Same model,
same prompt, same seeds, same perception, same budget caps, same accounting
implementation. The switches are locked by `tests/test_m5_arms.py`.

**Pre-registration with hash freezing.** `configs/preregistration.json` freezes
the skill manifest, the DSL whitelist, both prompt hashes, and the calibration
table. `check_manifest_stable.py` fails if any of them drift. A prompt edited
mid-experiment invalidates the run — and now that is mechanically detectable
rather than a matter of discipline.

**Accounting that cannot be gamed.** Every planner call — decision, replan, or
recovery — charges exactly one `llm_calls` slot in every arm, so "which arm is
cheaper" reflects architecture rather than bookkeeping differences. Physical
actions, queries, replans, wall-clock and no-progress cycles are metered
separately.

**The fault model.** The interesting failure is not "the robot dropped the
object" — that is physics. It is **F2: receipt loss**, where a grasp physically
succeeded but the call returns a timeout. A naive agent re-grasps (double
grip, second collision) or gives up. A harness-driven agent re-verifies the
*same* call id against fresh evidence and continues.

### Results

431 episodes, paired seeds, paired bootstrap (10 000 resamples, fixed RNG):

| Arm | Harness configuration | N0 (no fault) | F2 (receipt lost) | repeated re-grasp |
|---|---|---|---|---|
| **A** | self-report only | 0.70 | **0.08** | 0.09 |
| **B** | + independent verification | 0.73 | 0.83 | 0.19 |
| **C1** | + recovery | — | 0.96 | 0.05 |
| **C** | + replan + progress ledger | — | **0.93** | 0.07 |

**Primary endpoint: C vs A @ F2 = +0.881, 95% CI [0.786, 0.976]** (42 paired
seeds, 37 discordant). False-completion rate was 0 in every arm.

Two findings worth more than the headline number:

- **B < A under N0 (0.73 vs 0.70).** Honest feedback without a recovery
  mechanism *creates jitter* — the agent receives an accurate "I don't know if
  that worked" verdict and can only act on its own interpretation of the
  evidence, so it re-tries more. This is a real negative result and it is the
  argument for having recovery at all.
- **The arms' cost per success goes the wrong way.** C uses *fewer* physical
  actions and LLM calls per success than A (2.0 / 3.0 vs 2.79 / 3.89). The
  harness is not buying reliability with compute.

Reproduce the analysis from the shipped episode rows — the output is
byte-identical to the published analysis file:

```bash
python scripts/analyze_m5.py results/m5/m5_confirm_merged_rows.json
# → results/m5/m5_analysis.json
```

<!-- EN: 431 episodes, paired seeds, paired bootstrap. -->

---

## Honest status

This repository ships the **framework layer and the VLM application layer**.
The physical backends (CoppeliaSim / UR5 / ROS 2) are not included — they
require licensed simulator assets and a robot, and they are wired in through
the `Adapter` interface described below.

What is **measured and reproducible here**:

- the nine-state loop, the trust boundary, the ledger semantics, the budget
  accounting, and the pre-registration guard — all covered by tests;
- the VLM protocol (dual-image SoM, recognition-map uniqueness, id
  translation, place normalization, prompt freezing, transport retry, protocol
  error handling) — all covered by tests;
- the four-arm experiment above, from 431 episodes on the RLBench backend.

What is **not** yet published: the VLM-in-the-loop task success rate on a
physical backend. An earlier run of that configuration produced zero successes,
and the cause was a defect in the `place` branch of the VLM planner — the
container pose was being overwritten by the held object's pose, so the drop
target was the object's own location. That defect is fixed here and covered by
`tests/test_vlm_planner.py`; the end-to-end offline episode now completes.
Re-measuring the success rate on hardware is the next step, and this repository
does not claim a number it has not measured.

<!-- EN: This repository does not claim a number it has not measured. -->

---

## Layout

```
harness/
  spec/        runtime FSM · DSL verifier · validator · ledgers · budget · safety
  planning/    VLM planner (recognition + decision) · LLM planner · SoM recognizer
  skills/      grasp / place / motion — motion only, never verdicts
  adapters/    judge bridge (oracle boundary) · mock backend (self-test)
configs/       frozen thresholds, task asset descriptions, preregistration
results/       calibration tables — the single source of truth for every verdict
scripts/       mock matrix · oracle isolation check · manifest drift check · analysis
tests/         102 tests, offline, no API key required
```

### The Adapter interface

The framework never imports a backend. A backend implements `Adapter`
(`observe / execute / shutdown`) plus a judge bridge, and the runtime,
decision layer, verification layer and evaluation are unchanged. The mock
adapter in `harness/adapters/mock_adapter.py` is a complete reference
implementation.

---

## Design notes

`docs/DESIGN.md` records the decisions and their reasons — including the ones
that turned out to be wrong, and how they were found.

<!-- EN: including the ones that turned out to be wrong. -->

## License

MIT — see [LICENSE](LICENSE).

The vendored benchmarks this was developed against (RLBench, Imperial College
London) are **not** redistributed here and carry their own non-commercial
research licence. The framework talks to them through the `Adapter` interface
only.

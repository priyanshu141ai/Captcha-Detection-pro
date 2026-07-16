# CipherLens Upgrade Audit and Implementation Plan

Audit date: 2026-07-16

Repository baseline: `87acae0` (`main`)

Scope: audit and planning only; no model, training, inference, UI, or dataset
behavior was changed.

## Executive summary

CipherLens has a credible working baseline: a 1,190,475-parameter PyTorch CRNN,
shared training/inference preprocessing, safe restricted checkpoint loading,
upload validation, CPU inference, a polished Streamlit UI, deterministic
coverage-aware splitting, tests, and hardened Compose runtime settings. All 14
existing tests passed, the two-image runtime check passed, and a fresh evaluation
of the current deterministic 200-image validation subset reproduced the stored
98.5833% character accuracy and 92% exact-string accuracy.

Those numbers are dataset-specific validation results, not external evidence.
There is no independent external test set, eight observed characters do not
appear in validation, and nine characters have five or fewer examples in all
1,000 images. The present split mixes both supplied batches at sample level.
This audit found no exact image duplicates and no pairs within a conservative
perceptual-hash distance of four, but the repository has no durable dataset
manifest or version hash and no grouping metadata to protect future splits.

The two release-blocking issues are artifact safety/provenance and evaluation
independence. Running `python train.py` uses the approved checkpoint as its
default output, so routine training can overwrite it. The existing checkpoint
does not include the provenance fields written by the current training code, and
the ignored `training_history.json` cannot be linked to it: the checkpoint says
epoch 9, while the history's best tie-broken row is epoch 21. The checkpoint's
stored metrics were reproduced, but the exact training run and dataset version
cannot be reconstructed from its metadata.

The repository should be upgraded incrementally. Model V1 and the direct local
Streamlit path remain the compatibility baseline while a real `cipherlens`
package, central configuration, dataset manifests, modular training/evaluation,
a FastAPI service, and stronger CI/operations are added around it.

## Audit boundary and repository state

The complete tracked tree and all non-binary project files were inspected,
including `README.md`, `app.py`, `train.py`, both label files, dependencies,
`src/`, `scripts/`, `tests/`, `docs/`, `.github/`, Streamlit settings, Docker and
Compose files, checkpoint metadata, training history, all 1,000 images, and both
local ZIP archives. Image and model payloads were inspected through metadata,
safe decoding, hashes, and checkpoint schema rather than raw binary dumps.

At audit start, tracked files matched commit `87acae0`, but the tree was not fully
clean because `.github/workflows/ci.yml` already existed as an untracked file.
It was treated as user-owned and was not edited during the audit. Ignored local
items included `.venv`, `training_history.json`, `batch_0_zip.zip`, and
`batch_1_zip.zip`. The new empty GitHub repository is configured as `origin`; the
former origin is retained as `legacy`.

## Verification actually performed

| Command/check | Actual result |
|---|---|
| `python --version` and `.venv` Python | Python 3.11.9 for both |
| `python -m pip check` | Passed: no broken requirements |
| `python -m compileall -q app.py train.py src tests scripts` | Passed |
| `python -m unittest discover -s tests -v` | Passed: 14 tests in 0.645 s; one expected Streamlit bare-context warning |
| `python -m scripts.verify_runtime` | Passed: both known images matched; observed 161.93 ms first inference and 6.78 ms second inference on this machine |
| Deterministic validation re-evaluation | Reproduced 0.985833 character accuracy and 0.92 exact accuracy on 200 images; 16 strings failed |
| `docker compose config` | Passed |
| `python train.py --help` | Passed; all documented current CLI options rendered |
| `docker build --tag cipherlens:audit .` | Not completed and not claimed as passed. Stopped after the Linux dependency resolver downloaded a 532.2 MB Torch wheel and began a 366.2 MB CUDA library for the CPU service, with more CUDA packages queued |
| Secret-pattern scan over non-binary project files | No apparent credential found; only documentation and ignore-rule mentions |

Inference timings above are individual smoke-test observations, not a controlled
latency benchmark and not promotion evidence.

## Current architecture

```text
Browser
  -> Streamlit app.py
       -> upload byte/format/pixel validation (src.validation)
       -> cached CaptchaRecognizer (src.inference)
            -> shared preprocessing (src.data)
            -> legacy checkpoint, weights_only=True
            -> CaptchaCRNN + CaptchaCodec (src.model)
            -> six-character text + geometric-mean confidence

train.py
  -> label/image loading and coverage-aware split (src.data)
  -> augmentation and DataLoaders
  -> CaptchaCRNN (src.model)
  -> weighted cross-entropy + AdamW + ReduceLROnPlateau
  -> best checkpoint and JSON history
```

The model converts RGB images to 176x48 tensors normalized to `[-1, 1]`, applies
four convolution/batch-normalization/ReLU/pooling blocks, averages feature
height, adaptively pools width to six steps, applies a two-layer bidirectional
LSTM, and uses one shared linear character classifier at each of the six steps.
The current checkpoint is 4,781,237 bytes, has 43 classes, and has SHA-256
`b366146071bdf91f9acaca6be2d0da8c91a6cf3631bbcf93721584f78714c422`.

## Current training flow

1. The CLI parses paths, optimization settings, split fraction, seed, device,
   workers, thread count, cache behavior, and output paths.
2. Python, NumPy, and PyTorch RNGs are seeded; CPU or CUDA is selected.
3. `labels.txt`/`data/batch_0` is always loaded, followed by zero or more
   `--extra-dataset` pairs. Duplicate resolved paths are rejected.
4. The vocabulary is the sorted set of characters observed in all loaded labels.
5. `coverage_aware_split` deterministically shuffles at sample level and retains
   at least one occurrence of every class in training. The documented combined
   run produces 800 training and 200 validation images.
6. Training applies small rotations/translations, brightness/contrast changes,
   occasional blur, and Gaussian tensor noise. Validation is unaugmented.
7. Inverse-square-root class weights, capped at four before normalization, feed
   cross-entropy over all six positions.
8. AdamW optimizes the model; validation loss drives `ReduceLROnPlateau`; gradient
   norm is clipped to 5.0.
9. The best tuple of character accuracy, exact accuracy, and negative loss is
   atomically saved. Early stopping counts epochs without improvement to that
   full tuple.
10. A warm-start path can retain shared feature weights and copy classifier rows
    by character. Optimizer/scheduler state and resume training are not supported.

## Current inference flow

1. Streamlit reads the configured checkpoint path and confidence threshold at
   module import.
2. The uploader filters extensions; application validation independently checks
   bytes, decoded PNG/JPEG format, dimensions, pixel count, integrity, and full
   decode before preview.
3. On an explicit button click, Streamlit caches a recognizer keyed by checkpoint
   path, modification time, and size.
4. The checkpoint is loaded with `weights_only=True`; required state/vocabulary
   fields and strict state-dict compatibility are checked.
5. Shared preprocessing resizes and normalizes the image. CPU is the default.
6. Greedy per-position decoding returns six characters. Overall confidence is
   the geometric mean of maximum softmax probabilities.
7. The escaped result is displayed, low overall confidence is flagged, and a
   local copy control is offered. Uploaded content is not deliberately persisted.

## Working behavior to preserve

- Model V1 architecture, vocabulary order, preprocessing, state-dict keys, and
  legacy checkpoint loading.
- Six-character, case-sensitive decoding including repeated characters.
- Dataset-specific baseline results and clear non-generalization caveat.
- Deterministic seed-42 coverage-aware 800/200 combined split until a versioned
  manifest intentionally supersedes it.
- CPU inference and optional CUDA selection.
- Direct Python `CaptchaRecognizer` use and Streamlit local fallback.
- Streamlit upload/preview/button/result/copy/reset flow and current environment
  variable names.
- PNG/JPEG integrity, byte-size, and pixel-count validation.
- CLI flags for current training, extra datasets, warm starts, workers, threads,
  image caching, and history output.
- Atomic checkpoint replacement mechanics for candidate paths.
- Safe `weights_only=True` checkpoint loading.
- Current non-root/read-only/no-new-privileges container posture and port 8501 UI.
- Existing tests, runtime smoke cases, Windows-compatible commands, and the
  authorized-use restriction.

## Dataset and evaluation evidence

| Property | Observed result |
|---|---:|
| Labeled images | 1,000: 500 per batch |
| Label length | All 1,000 are exactly six characters |
| Image contract | All 1,000 decode as RGB PNG, 151x41 |
| Missing / unlabeled / unreadable | 0 / 0 / 0 |
| Vocabulary | 43 characters; 6,000 total positions |
| Very rare classes | `D,k,l,t,w,x,y,z`: 1 each; `f`: 3 |
| Classes absent from validation | `D,k,l,t,w,x,y,z` |
| Duplicate labels | 0 |
| Exact duplicate image groups | 0 |
| Perceptual near-duplicate pairs | 0 at 64-bit DCT pHash Hamming distance <=4; minimum observed distance was 10 |
| Split distribution | Train: 400/400 by batch; validation: 100/100 by batch |
| External test set | Missing |
| Diagnostic dataset digest | `770c62ea8b5d3ecae61fc42be76bd4ea0fbf2ca7e130561a05340557f06ea91a` |

The digest and pHash results were computed for this audit and are not yet durable
Milestone 2 artifacts. Perceptual hashing is heuristic and does not prove that
images are unrelated after geometric or generator-level transformations.

Fresh validation results were 99.0%, 99.0%, 98.0%, 98.0%, 99.0%, and 98.5%
accuracy by position. Batch-specific exact accuracy was 93% for the 100 batch-0
validation items and 91% for the 100 batch-1 items. Several wrong predictions
had high uncalibrated confidence; for example, `RbRLGH` was read as `RbRL6H` at
about 95.2% confidence. These are error-analysis observations from the existing
validation split, not new external metrics.

## Findings

### Critical

#### C-01: Routine training can overwrite the approved model and artifacts lack reliable provenance

- **Evidence:** `train.py` defaults `--output` to `models/captcha_crnn.pt` and
  README's plain training command uses that default. The current checkpoint has
  only `model_state`, `charset`, `model_config`, `metrics`, and `epoch`; it lacks
  checkpoint version, dataset version, configuration, timestamp, Git commit, and
  training source metadata. Its epoch is 9, while the ignored history contains
  24 rows and a tie-broken best row at epoch 21.
- **Affected files:** `train.py`, `README.md`, `docs/OPERATIONS.md`,
  `models/captcha_crnn.pt`, `training_history.json`, `.gitignore`.
- **Why it matters:** A normal command can irreversibly replace the only approved
  artifact, and the shipped model cannot be traced to an exact dataset/config/run
  or reliably reproduced and rolled back.
- **Recommended change:** Immediately document candidate-only output practice;
  in Milestone 3 make candidate output the safe default or refuse overwrite
  without an explicit promotion flag. Add a versioned checkpoint schema,
  run-specific history, dataset hash, config, Git SHA, timestamp, architecture and
  preprocessing metadata, optimizer/scheduler state, and promotion/rollback flow.
  Preserve the existing checkpoint byte-for-byte and support it as legacy schema.
- **How tested:** Hash the approved checkpoint before and after training tests;
  verify default candidate training never changes it; round-trip new metadata;
  load the legacy checkpoint; resume a tiny deterministic run; test atomic failure
  and explicit promotion/rollback.

#### C-02: There is no independent evidence for model promotion or external generalization

- **Evidence:** Both batches are mixed into a deterministic sample-level 800/200
  development split. The same validation subset drives scheduler behavior,
  checkpoint selection, reported accuracy, and architecture discussion. No
  external-test manifest exists. Eight classes never appear in validation, and
  nine classes have at most five total examples.
- **Affected files:** `src/data.py`, `train.py`, `labels.txt`,
  `requirements2.txt`, `README.md`, `docs/TECHNICAL_DOCUMENTATION.md`.
- **Why it matters:** The 98.58%/92% results can be optimistic after repeated
  development use and do not measure rare-class or unseen-style performance.
  Production promotion criteria cannot be satisfied honestly.
- **Recommended change:** In Milestone 2 create immutable, grouped train,
  validation, calibration, and external-test manifests from documented sources.
  Quarantine the external test from tuning. Report external evaluation as pending
  until separately sourced authorized data exists, and collect enough coverage
  for every expected class and position.
- **How tested:** Verify manifests are deterministic and disjoint by path, exact
  hash, perceptual group, source group, and label; assert external-test data is
  never loaded by training; publish per-split coverage and fail promotion when
  required external evidence is absent.

### High

#### H-01: Dataset validation and versioning are ad hoc

- **Evidence:** `load_samples` checks row shape, referenced-file existence, and a
  non-empty result only. It does not enforce six-character labels, decoded image
  readability/dimensions, vocabulary policy, duplicate content, or leakage. No
  manifest, dataset card, stable statistics, or version hash is stored.
- **Affected files:** `src/data.py`, `train.py`, label files, `data/`, `docs/`.
- **Why it matters:** Bad labels, corrupt images, content duplicates, or split
  drift can fail late, silently bias evaluation, or make a run irreproducible.
- **Recommended change:** Implement the non-destructive Milestone 2 audit and
  manifest pipeline with JSON/CSV reports, exact and perceptual grouping,
  validation policy, dataset digest, and dataset card.
- **How tested:** Use generated fixtures for corrupt, wrong-size, wrong-length,
  missing, exact-duplicate, near-duplicate, rare, and unseen cases; snapshot
  deterministic report schemas and confirm suspicious files are never deleted.

#### H-02: Reproducibility stops at basic RNG seeding

- **Evidence:** Python, NumPy, and Torch are seeded, but no deterministic-algorithm
  policy, cuDNN controls, DataLoader generator/worker seeding, environment capture,
  manifest identity, dependency lock, or run identifier exists. Augmentation uses
  process-global Python and Torch RNG state.
- **Affected files:** `train.py`, `src/data.py`, `requirements.txt`.
- **Why it matters:** Results can diverge across devices, worker counts, package
  resolutions, and refactors even with `--seed 42`.
- **Recommended change:** Add a central seed utility with an explicit strict or
  best-effort mode, seeded loaders/workers, immutable config snapshots, dependency
  and platform capture, and dataset/split hashes.
- **How tested:** Compare split IDs, augmentation sequences where promised, model
  initialization, and tiny-run metrics across two same-seed runs; verify a changed
  seed changes the run identity.

#### H-03: Configuration, packaging, logging, and environment handling are fragmented

- **Evidence:** The project is imported as generic package `src`; there is no
  `pyproject.toml`, central settings model, YAML config, formatting/lint/type
  configuration, or logging setup. `app.py` parses a float at import and
  `src/inference.py` parses an integer during recognizer creation without a
  structured validation error. Training defaults live in `argparse` and model
  defaults live in a dataclass.
- **Affected files:** `app.py`, `train.py`, `src/__init__.py`, `src/inference.py`,
  `src/model.py`, `.env.example`, `requirements.txt`.
- **Why it matters:** Invalid deployment values cause raw startup exceptions;
  defaults drift across entry points; generic imports are hard to package; and
  operational events are inconsistent and difficult to test.
- **Recommended change:** Complete Milestone 1 with a `cipherlens` package,
  validated typed settings, central defaults, structured logging, a seed utility,
  `pyproject.toml`, and compatibility wrappers.
- **How tested:** Unit-test valid/default/boundary/invalid settings and redacted
  logs; install the package in a clean environment; test both new and legacy
  imports; compare model outputs before and after migration.

#### H-04: UI, model lifecycle, and serving are coupled in one process

- **Evidence:** `app.py` imports inference directly and owns checkpoint lifecycle,
  environment parsing, upload handling, presentation, and user-visible errors.
  There is no API package, typed request/response contract, readiness distinction,
  request ID, batch bound, Prometheus endpoint, or service-level concurrency
  control.
- **Affected files:** `app.py`, `src/inference.py`, Docker and Compose files.
- **Why it matters:** The current design is suitable for a local tool but makes
  independent scaling, integration testing, observability, and failure isolation
  difficult.
- **Recommended change:** Add a FastAPI backend in Milestone 6 that loads the
  model once at startup; keep Streamlit as a client with the direct recognizer as
  an explicit fallback in Milestone 7.
- **How tested:** Unit and integration tests for lifecycle, health/readiness,
  model info, request IDs, single/batch predictions, structured errors, upload
  boundaries, concurrency, and frontend fallback.

#### H-05: Evaluation, confidence, and error analysis are insufficient for safe promotion

- **Evidence:** Training logs only weighted loss, exact accuracy, and a
  Levenshtein-derived character accuracy. There is no external-test separation,
  CER/NED report, per-character/position report in the repository, confusion
  matrix, failed-prediction export, controlled latency/memory/model-size report,
  calibration analysis, or promotion gate. High-confidence validation failures
  were observed.
- **Affected files:** `train.py`, `src/model.py`, `src/inference.py`, `docs/`.
- **Why it matters:** Aggregate metrics conceal rare-character and case-confusion
  failures; uncalibrated confidence can mislead users; candidates cannot be
  compared consistently.
- **Recommended change:** Implement Milestones 4 and 5 with validation versus
  external-test separation, calibration fitted only on validation/calibration
  data, error exports, resource benchmarks, and documented promotion criteria.
- **How tested:** Metric unit tests against hand-computed examples, golden report
  schemas, no-test-data calibration assertions, deterministic failure exports,
  and benchmark protocols with environment metadata.

#### H-06: The production container dependency path is not CPU-minimal or reproducible

- **Evidence:** Docker installs generic `torch==2.12.1` from the default index. The
  audit build downloaded a 532.2 MB Torch wheel and began a 366.2 MB CUDA library,
  with more CUDA packages queued, despite CPU deployment. Docker also upgrades
  unpinned pip during build and uses the mutable `python:3.11-slim` tag.
- **Affected files:** `Dockerfile`, `requirements.txt`, `.dockerignore`, CI
  workflow.
- **Why it matters:** Builds are slow, large, costly, and potentially variable;
  unnecessary GPU libraries increase storage and supply-chain surface. Docker
  verification did not complete during this audit.
- **Recommended change:** Select and pin an official CPU-only PyTorch install
  strategy, use a locked runtime dependency set and pinned base digest, remove
  build-time package upgrades, and use a measured multi-stage build where useful.
- **How tested:** Build from an empty cache, record final image/layer sizes, inspect
  installed packages for CUDA payloads, run as non-root/read-only, exercise health
  and one prediction, and enforce an agreed size budget in CI.

#### H-07: Dataset ownership and collection provenance are undocumented

- **Evidence:** All 1,000 images and labels are tracked, but no dataset card,
  generation/collection process, authorization record, license, source grouping,
  or retention policy is present. Two ignored ZIP copies also exist locally.
- **Affected files:** `data/`, label files, `README.md`, `docs/`, `.gitignore`.
- **Why it matters:** Reviewers cannot verify authorized use, reproduce collection,
  distinguish source domains, or decide whether the images belong in a public
  repository.
- **Recommended change:** Confirm authorization before further publication; add a
  dataset card and provenance/source fields; define public/private storage and
  retention; keep large/private future data outside Git with version references.
- **How tested:** Documentation review gate, manifest provenance validation, secret
  and large-file scans, and CI assertions that generated fixtures—not private
  corpora—drive tests.

#### H-08: CI coverage is not yet a dependable quality gate

- **Evidence:** At audit start `.github/workflows/ci.yml` was untracked. Its local
  contents compile, run the 14 tests, run a full-data/full-checkpoint smoke test,
  and build Docker, but do not lint, format-check, type-check, vulnerability-scan,
  test the running image, or use tiny fixtures. It uses mutable action tags.
- **Affected files:** `.github/workflows/ci.yml`, `.github/dependabot.yml`,
  `requirements.txt`, `tests/`, `data/`, `models/`.
- **Why it matters:** The remote repository may have had no active workflow; future
  CI either depends on committing the full dataset/model or loses its main smoke
  coverage, and code-quality/supply-chain regressions are not blocked.
- **Recommended change:** Track and harden the workflow in Milestones 1 and 8;
  separate fixture-only required jobs from an optional trusted-artifact smoke job;
  add lint, formatting, practical typing, install verification, API integration,
  vulnerability review, and running-container health.
- **How tested:** Validate workflow syntax, run jobs on a clean clone without the
  private/full dataset, confirm deliberate failures block merging, and verify
  least-privilege permissions and pinned action revisions.

### Medium

#### M-01: Dependency roles and supply-chain controls are incomplete

- **Evidence:** Four top-level runtime packages are exactly pinned and the local
  environment matches them; `pip check` passed. There is no project metadata,
  training/dev/API optional groups, transitive lock or hashes, automated
  vulnerability command, license inventory, or documented update policy beyond
  Dependabot. `requirements2.txt` is a label manifest despite its dependency-like
  name.
- **Affected files:** `requirements.txt`, `requirements2.txt`,
  `.github/dependabot.yml`, Dockerfile.
- **Why it matters:** Clean installs resolve mutable transitive graphs, production
  installs unnecessary Streamlit dependencies for a future API, and contributors
  can mistake a dataset manifest for a package file.
- **Recommended change:** Use `pyproject.toml` for metadata/tooling and explicit
  runtime/train/dev/API groups, generate reviewed locks or constraints per
  platform, add vulnerability/license review, and rename the batch-1 label file
  through a compatibility period.
- **How tested:** Build clean Windows/Linux environments from locks, run `pip
  check`, compare dependency inventories, run the chosen audit tool, and test the
  old label path shim/documented migration.

#### M-02: Automated tests cover happy paths but leave important boundaries untested

- **Evidence:** The 14 tests cover dataset counts, split coverage, output shape,
  repeated decoding, edit distance, two known predictions, cached image bytes,
  warm-start rows, basic upload limits, missing/malformed checkpoints, and initial
  UI render. There is no measured coverage and no tests for corrupt training
  images, invalid labels, split identity, augmentation/seeding, full train/resume,
  metric edge cases, environment parsing, per-character confidence, API behavior,
  concurrency, or running container health.
- **Affected files:** `tests/`, all runtime/training modules, CI workflow.
- **Why it matters:** Refactoring package boundaries or configuration can silently
  change model behavior; production boundary failures are not regression-protected.
- **Recommended change:** Add layered `tests/unit` and `tests/integration` suites
  with generated images/checkpoints, compatibility goldens, and coverage reporting
  as each milestone introduces behavior.
- **How tested:** Enforce an agreed coverage floor for deterministic application
  code, mutation or fault-injection checks on validators where practical, and a
  clean fixture-only CI run.

#### M-03: Legacy checkpoint validation is safe but semantically shallow

- **Evidence:** Restricted loading, required fields, charset type, state-dict type,
  config construction, and strict state loading are checked. No explicit schema
  version, architecture name/version, preprocessing version, dimensions bounds,
  normalization, checksum/signature, or metric/provenance validation exists.
- **Affected files:** `src/inference.py`, `train.py`, checkpoint artifact.
- **Why it matters:** A structurally loadable but semantically wrong checkpoint can
  produce incorrect behavior, and model information cannot be served reliably.
- **Recommended change:** Define versioned typed checkpoint metadata with a
  backward-compatible V1 adapter and optional expected hash verification.
- **How tested:** Schema tests for missing, unknown, malformed, out-of-range, and
  legacy fields; known-output compatibility; tampered-file/hash failure; model-info
  response tests.

#### M-04: Environment configuration can fail with raw exceptions or unsafe values

- **Evidence:** `float(CIPHERLENS_CONFIDENCE_THRESHOLD)` happens at Streamlit
  import; `int(CIPHERLENS_TORCH_THREADS)` happens at recognizer construction; the
  confidence threshold is not range-checked and thread parsing errors are not
  translated. `.env.example` documents only two of three current variables.
- **Affected files:** `app.py`, `src/inference.py`, `.env.example`, Compose/docs.
- **Why it matters:** A typo causes a low-context startup failure and an out-of-range
  threshold silently produces nonsensical UI behavior.
- **Recommended change:** Centralize typed settings, validate path/readability,
  integer bounds and confidence range, document every variable, and fail fast with
  redacted actionable messages.
- **How tested:** Parameterized tests for absent, valid, invalid, boundary, and
  whitespace values plus container startup tests for invalid configuration.

#### M-05: Logging and observability do not meet service requirements

- **Evidence:** A logger name exists and exceptions are logged, but no format,
  level, JSON structure, request ID, model version, latency metric, readiness
  state, or Prometheus endpoint is configured. Training uses `print` only.
- **Affected files:** `app.py`, `train.py`, `src/inference.py`, operations docs.
- **Why it matters:** Operators cannot correlate failures, monitor latency/error
  rates, compare model versions, or alert on readiness without ad hoc log parsing.
- **Recommended change:** Add structured, privacy-safe logs in Milestone 1 and API
  metrics/request context in Milestone 6; add training run logging separately.
- **How tested:** Capture logs and assert required/redacted fields, stable request
  IDs, latency histogram increments, and no uploaded bytes or secrets.

#### M-06: Documentation and run artifacts can drift from executable reality

- **Evidence:** The technical checkpoint example omits fields the current trainer
  now writes, while the shipped checkpoint itself predates those fields. The
  history/checkpoint epoch mismatch is not explained. Operations assumes CI is
  tracked and uses a machine-specific `C:\Captcha detection` example. Results are
  manually repeated across files.
- **Affected files:** `README.md`, both existing docs, `train.py`, checkpoint,
  training history, CI workflow.
- **Why it matters:** Interviewers and operators can follow incorrect provenance
  or commands, and manual result copies invite unsupported claims.
- **Recommended change:** Generate model/dataset cards and result tables from
  versioned artifacts where possible; use repository-relative commands; document
  legacy checkpoint limitations and source each metric by split/run.
- **How tested:** Documentation command smoke tests on Windows, link checks,
  artifact-to-card consistency checks, and review that absent external results say
  pending rather than zero or fabricated values.

#### M-07: Runtime resource controls stop at upload bounds and Torch threads

- **Evidence:** Streamlit limits bytes/pixels and Compose uses read-only/tmpfs,
  but there is no request rate limit, inference queue/concurrency bound, process
  memory/CPU guidance enforced by Compose, graceful overload response, or batch
  policy. Each Streamlit session can initiate inference.
- **Affected files:** `app.py`, Compose, Dockerfile, operations docs.
- **Why it matters:** Authorized deployments can still be exhausted by concurrent
  or repeated CPU-heavy requests.
- **Recommended change:** Bound API batch/concurrency and request size, document
  platform rate limiting and resource reservations/limits, and expose saturation
  metrics.
- **How tested:** Concurrent load tests with bounded fixtures, overload/error-code
  assertions, memory/latency measurement, and container resource-limit tests.

### Low

#### L-01: Tool-enforced code style and typing are absent

- **Evidence:** Type hints are common and modules are readable, but no formatter,
  linter, import sorter, type checker, or corresponding configuration/CI job
  exists; some lines already exceed conventional limits.
- **Affected files:** all Python files, future `pyproject.toml`, CI workflow.
- **Why it matters:** A growing multi-package refactor will accumulate inconsistent
  style and easy-to-catch defects.
- **Recommended change:** Configure Ruff formatting/linting and a practical staged
  type checker in Milestone 1, initially avoiding behavior-changing mass edits.
- **How tested:** Run format check, lint, and configured type checks locally and in
  CI; keep explicit, reviewed exclusions narrow.

#### L-02: Inference changes process-global Torch thread state

- **Evidence:** Every `CaptchaRecognizer` construction calls
  `torch.set_num_threads`, which affects the whole process rather than only that
  model instance.
- **Affected files:** `src/inference.py`, `app.py`, future API lifecycle.
- **Why it matters:** Tests, multiple recognizers, or colocated Torch workloads can
  influence one another unexpectedly.
- **Recommended change:** Apply validated thread configuration once during process
  startup and document it as process-wide.
- **How tested:** Lifecycle tests with multiple recognizers and invalid/changed
  settings; concurrency benchmark after startup initialization.

#### L-03: Root naming and compatibility paths are confusing

- **Evidence:** The import package is named `src`, the second label manifest is
  named `requirements2.txt`, and app/training entry points sit at root. The target
  architecture expects a `cipherlens` namespace and `streamlit_app/`.
- **Affected files:** `src/`, `requirements2.txt`, `app.py`, `train.py`, docs/tests.
- **Why it matters:** Names obscure intent and make installation and onboarding
  less professional, although current functionality works.
- **Recommended change:** Introduce the new names incrementally with wrappers and
  deprecation notes; do not break current commands during early milestones.
- **How tested:** Old and new import/CLI/startup smoke tests and documentation link
  checks throughout the migration.

## Dependency audit

| Area | Current state | Assessment |
|---|---|---|
| Python | 3.11.9 locally; Docker and local CI file select 3.11 | Make 3.11 the explicit supported baseline |
| Direct runtime dependencies | NumPy 2.4.6, Pillow 12.2.0, Streamlit 1.58.0, Torch 2.12.1; installed versions match | Exact top-level pins are good, but they mix UI/model concerns |
| Resolver health | `pip check` passed | No broken installed dependency relationship was found |
| Transitive reproducibility | No constraints/lock/hashes | Clean builds can change without a top-level diff |
| Development tooling | No explicit lint, format, type, coverage, or vulnerability dependencies | Required for Milestones 1 and 8 |
| API/tracking dependencies | FastAPI and MLflow are not explicit project features/dependencies | Add as separate optional groups; MLflow must remain optional |
| Linux Torch | Default resolution pulled CUDA payloads during audit build | Use a documented official CPU-only channel/build |
| Vulnerability/license evidence | Dependabot config exists; no local audit/SBOM/license report | No claim of vulnerability-free dependencies can be made |
| Naming | `requirements2.txt` contains labels, not packages | Rename with a compatibility plan |

## Test coverage gap analysis

| Layer | Existing evidence | Main gaps |
|---|---|---|
| Data | Counts, six-length observation, split training coverage, second batch | Validator failure cases, exact split manifest, corrupt/wrong dimensions, duplicates/leakage, statistics/version |
| Preprocessing/model | One shape test, repeated-character decode, edit distance | Pixel golden, normalization edge cases, augmentation bounds/reproducibility, parameter/schema invariants |
| Training | Warm-start classifier rows, cached bytes | Loss/class weights, scheduler/early stop, checkpoint metadata/atomic failure, resume, deterministic tiny run, overwrite prevention |
| Inference | Two known images, missing and minimal malformed checkpoints | Determinism repeat, per-character confidence, schema versions, model info, corrupt/tampered state, concurrency |
| Upload/UI | Basic formats/limits and initial render | MIME/extension cross-check, upload-to-prediction UI flow, low-confidence/error/reset states, backend fallback |
| API | None; API does not exist | Every required endpoint and error/batch/lifecycle case |
| Operations | Compose renders; build exposed dependency issue | Successful minimal build, running health, non-root/read-only assertions, missing model, resource limits, rollback |
| Quality | Compile test | Formatting, linting, type checking, coverage measurement, dependency audit |

## Security and deployment review

Positive controls already present include restricted checkpoint loading,
decoded-image validation and decompression-bomb handling, HTML escaping of model
output, generic user-facing errors, no deliberate upload persistence, Streamlit
XSRF/CORS settings, telemetry disabled, non-root Docker user, read-only Compose
filesystem, tmpfs, `no-new-privileges`, health checking, least-privilege CI
permissions in the local workflow, and ignore rules for secrets and `.env`.

Important remaining work:

- Validate declared MIME type and extension at the future API boundary in addition
  to decoded content; bound batch size and concurrency.
- Establish model checksum/signing and provenance validation; restricted pickle
  loading alone does not identify an approved artifact.
- Add structured request IDs, latency/error/readiness metrics, safe log policy,
  alerting hooks, and graceful missing-model startup behavior.
- Pin the base image and actions immutably, lock dependencies, add vulnerability
  review/SBOM evidence, and remove unnecessary CUDA payloads.
- Add container capability dropping, explicit resource guidance/limits, a tested
  writable-temp policy, and running-image health verification.
- Keep TLS, authentication if deployment requires it, rate limiting, centralized
  logging, and network policy at the deployment boundary; do not imply the current
  local Compose file supplies them.
- Confirm dataset authorization before publishing it to the new public remote.
  No credential was found by the audit's pattern scan, but that is not a complete
  secret-history audit.

## Proposed final architecture

```text
streamlit_app/
  -> typed API client -> FastAPI /api
  -> direct local fallback -> shared inference service

src/cipherlens/api/
  -> request validation + request IDs + structured errors
  -> startup model registry / readiness
  -> bounded inference service
  -> Prometheus-compatible metrics

src/cipherlens/inference/
  -> preprocessing contract -> Model V1 default
  -> confidence + per-character probabilities + model metadata

src/cipherlens/training/          src/cipherlens/evaluation/
  -> immutable split manifests     -> validation/calibration/external-test roles
  -> reproducible loaders           -> metrics, calibration, errors, benchmarks
  -> baseline + experiments         -> reports, figures, model card
  -> versioned candidate artifacts

src/cipherlens/data/ -> authorized source inventory, validation, hashes,
                       grouping, manifests, statistics, dataset card

configs/ -> validated defaults and environment overrides
artifacts/ -> ignored/generated dataset and run metadata
reports/ -> generated evaluation tables and figures
docs/ -> architecture, dataset/model cards, operations, interview story
```

`src/cipherlens/models/` will keep the position-wise CRNN as Model V1.
CTC and transformer implementations remain opt-in experiments. Promotion is
evidence-based; advanced architecture alone is not a promotion criterion.

## Ordered implementation milestones

1. **Repository quality and configuration:** package metadata, `cipherlens`
   namespace, typed settings, structured logging, seed utility, tooling, ignores,
   setup docs, and compatibility tests. No model behavior change.
2. **Dataset audit and reproducibility:** non-destructive validation, frequency and
   duplicate reports, grouped deterministic manifests, dataset hash/card. External
   test remains explicitly pending if data is unavailable.
3. **Training pipeline:** modular data/model/loss/optimizer/scheduler/early-stop,
   YAML plus CLI overrides, resume, candidate-safe artifact handling, complete
   checkpoint metadata, optional MLflow.
4. **Evaluation and error analysis:** full metric suite, failed predictions,
   confusion/calibration/reliability, controlled resource benchmark, model card,
   strict split-role reporting.
5. **Model comparison:** retain V1; add optional CTC V2 and only justify a
   transformer V3 when data/dependencies support it; use documented promotion
   gates.
6. **FastAPI inference service:** lifecycle load, health/readiness/model info,
   single/batch prediction, upload hardening, request IDs, typed schemas, metrics,
   and integration tests.
7. **Streamlit integration:** API client, clear confidence/latency/version display,
   graceful errors, validated uploads, and tested local direct fallback.
8. **Testing and CI:** fixture-only unit/integration suite, lint/format/type checks,
   dependency installation/audit, optional trusted-artifact smoke, and Docker
   build/run verification.
9. **Containerization and operations:** CPU-minimal pinned builds, non-root and
   least privilege, health/resource/temp policies, artifact mounting or packaging,
   missing-model failure, promotion and rollback runbooks.
10. **Documentation and portfolio:** source-linked README results, architecture,
    dataset/model cards, operations, error examples, limitations, reproducibility,
    and an interview story that distinguishes completed work from pending work.

Each milestone ends after tests/checks, diff review, actual result reporting,
remaining-risk review, a suggested commit message, commit/push to `origin`, and an
explicit stop before the next milestone.

## Milestone 1 expected file changes

The expected Milestone 1 change set is deliberately explicit. A pre-milestone
status check may remove a file from this list if the same requirement has already
been satisfied, but should not silently expand scope.

New files:

- `pyproject.toml`
- `configs/default.yaml`
- `src/cipherlens/__init__.py`
- `src/cipherlens/config.py`
- `src/cipherlens/logging.py`
- `src/cipherlens/data/__init__.py`
- `src/cipherlens/models/__init__.py`
- `src/cipherlens/inference/__init__.py`
- `src/cipherlens/utils/__init__.py`
- `src/cipherlens/utils/reproducibility.py`
- `tests/unit/__init__.py`
- `tests/unit/test_config.py`
- `tests/unit/test_import_compatibility.py`
- `tests/unit/test_reproducibility.py`

Modified files:

- `.gitignore`
- `.env.example`
- `.github/workflows/ci.yml`
- `requirements.txt`
- `README.md`
- `docs/OPERATIONS.md`
- `AGENTS.md`
- `app.py`
- `train.py`
- `src/__init__.py`
- `src/data.py`
- `src/model.py`
- `src/inference.py`
- `src/validation.py`

The four legacy `src/*.py` behavior modules remain as compatibility surfaces in
Milestone 1. Package extraction must be mechanical and guarded by output-golden
tests; deeper data/training refactoring belongs to later milestones. Neither
`models/captcha_crnn.pt` nor any dataset file is in the Milestone 1 change set.

## Backward-compatibility risks

1. **Import migration:** changing `src.*` to `cipherlens.*` can break app, tests,
   scripts, notebooks, and checkpoint-related code. Keep wrappers and test both
   paths until a documented removal milestone.
2. **Model/preprocessing drift:** moving code can alter state-dict keys, resize,
   normalization, tensor layout, classifier order, or outputs. Freeze a known
   image tensor/output golden and the checkpoint hash.
3. **Checkpoint schema:** stricter metadata must continue to load the current
   five-field legacy checkpoint. Never rewrite it merely to add metadata.
4. **CLI/config precedence:** YAML/env/CLI layering can change existing defaults or
   flag semantics. Define precedence and snapshot current parsed defaults.
5. **Safer training output:** preventing default production overwrite is an
   intentional usability change. Preserve `--output` and document the candidate
   workflow rather than silently redirecting an explicitly supplied path.
6. **Streamlit lifecycle:** central settings/logging can change import-time app
   behavior and `st.cache_resource` invalidation. Retain the current upload and
   direct-inference smoke flow.
7. **Frontend/API split:** backend availability, serialization, timeouts, and
   confidence precision can change UX. Keep direct fallback and shared response
   semantics during migration.
8. **Manifest adoption:** replacing the algorithmic split with a stored manifest
   can change member identity if generated incorrectly. First reproduce the
   existing seed-42 split, then version any intentional new split.
9. **Label-manifest rename:** external commands may use `requirements2.txt`.
   Support or clearly migrate the old path before removal.
10. **Container topology:** adding an API port/process can break current port 8501
    usage, health checks, read-only assumptions, and Compose startup. Preserve the
    UI entry point while versioning the new topology.

## Audit-stage conclusion

The audit and plan are complete. The existing model was not overwritten and no
full upgrade milestone was started. Milestone 1 is the next authorized unit of
work after review of this document. The repository owner subsequently instructed
that Codex changes be pushed to the new public `origin`; this audit commit includes
the two requested audit documents. The pre-existing untracked CI workflow remains
untouched and outside the audit commit.

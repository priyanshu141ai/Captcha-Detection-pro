# CipherLens Interview Story

## Thirty-second summary

I took a working PyTorch CAPTCHA-style recognizer and turned it into a reviewable
ML system without changing its approved production behavior. I kept the compact
fixed-length CRNN, then added versioned dataset auditing, leakage-aware manifests,
safe candidate training, evidence-linked evaluation, a typed FastAPI service, a
Streamlit API client with controlled fallback, generated-fixture tests, CI, and a
hardened CPU-only deployment. The most important decision was to label the strong
validation metrics as provisional because the legacy checkpoint cannot prove its
historical training split and no independent external test set exists.

## 1. The original problem

The original application recognized exactly six characters in 151 x 41 noisy
images. It already had a useful PyTorch CRNN, an approved checkpoint, CPU
inference, and a Streamlit demo. The engineering problem was broader than model
accuracy: make the data, training, evidence, API, UI, deployment, and rollback
story reproducible without deleting working behavior or overstating results.

The ethical boundary was made explicit from the start. CipherLens supports only
synthetic, owned, or explicitly authorized images and intentionally excludes
browser automation, third-party submission, and access-control bypass features.

## 2. How the dataset was collected and labelled

The repository owner supplied two development batches of 500 images. Each label
map uses one row per image:

```text
<filename> <six-character-label>
```

I did not claim to have collected or generated data when the repository did not
contain that evidence. The exact generator, collection date, license, and
retention history are unknown. Those gaps are documented in the dataset card and
must be resolved before treating the data as independently redistributable.

This is an important interview point: provenance and authorization are part of an
ML dataset contract, not paperwork to add after training.

## 3. Dataset-quality challenges

The 1,000 images are clean enough to train but small relative to the 62-character
expected alphanumeric vocabulary. Only 43 characters are observed; some occur
one to five times, and several observed characters do not appear in validation.
That creates class imbalance and makes performance on new visual styles uncertain.

The audit therefore checks:

- label length, vocabulary, image readability, dimensions, and containment;
- duplicate paths and SHA-256-identical files;
- perceptual near-duplicates and repeated labels;
- related groups crossing train/validation/external roles;
- character coverage, rare classes, unseen classes, and version hashes.

Suspicious data is reported for human review and never deleted automatically.

## 4. Why the baseline architecture was selected

The output length is always six, so a position-wise CRNN is a strong fit:

1. Convolutions learn local visual features without manual character segmentation.
2. A bidirectional LSTM uses left/right context for distorted or overlapping glyphs.
3. Adaptive width produces six sequence steps.
4. One shared classifier predicts the checkpoint vocabulary at each position.

This design preserves repeated characters, is small enough for CPU deployment,
and matches the actual data contract. A more fashionable model would add risk and
dependencies without automatically adding evidence.

Model V2 CRNN-CTC support is isolated for future variable-length experiments.
Model V3 transformer OCR is deferred because 1,000 images and no external test
set do not justify the compute and overfitting risk.

## 5. How validation leakage was prevented

The modernized pipeline creates a deterministic manifest rather than splitting
ad hoc inside each run. It groups exact duplicates, perceptual near-duplicates,
and repeated labels before assigning related samples to a split. Training and
evaluation verify every selected image against its recorded path, label, SHA-256,
dataset version, and split version.

The current manifest uses seed 42 and contains 800 training plus 200 validation
rows. Coverage-aware assignment keeps observed character classes represented in
training. External-test rows, when added, are isolated from development and are
not used for tuning.

There is still one honest caveat: the approved checkpoint predates the manifest
metadata. Its historical training rows cannot be reconstructed well enough to
prove zero overlap with the current validation rows. That is why its metrics are
marked provisional.

## 6. What failed during experimentation

There is no committed evidence of a trained Model V2 or V3, so I do not invent a
failed experiment or comparison result. The CTC registry row correctly says
`experimental_not_trained`, and transformer metrics remain blank.

The meaningful failure was evidentiary: the legacy checkpoint produced strong
development results but lacked dataset/split provenance, calibration separation,
and external-test evidence. It therefore failed the new promotion standard even
though it remained the safest operational baseline.

A smaller operations lesson also appeared in CI: a `.gitkeep` placeholder entered
the runtime model directory and violated the minimal-image assertion. The fix was
to exclude the placeholder and keep the check. This showed the value of testing a
running image, not just building one.

## 7. What improved performance

The approved checkpoint was warm-started from the first batch and fine-tuned on
both batches. The current training pipeline also provides restrained augmentation,
class-weighted loss, AdamW, learning-rate reduction, gradient clipping, and early
stopping.

However, no versioned ablation study records the individual lift from each choice.
I can explain why each mechanism is reasonable, but I do not assign a fabricated
accuracy gain to it. A future experiment should repeat runs with aligned manifests
and report mean, variance, latency, calibration, and external-test behavior.

## 8. How the system was deployed

The online path is split into two services:

- FastAPI loads one model per process, validates uploads, bounds batches and
  concurrency, returns typed responses and request IDs, and exposes readiness and
  Prometheus-compatible metrics.
- Streamlit validates locally, calls FastAPI, presents confidence/latency/version,
  and uses the approved local checkpoint only after retryable backend failures.

The multi-stage image installs CPU-only PyTorch. Compose runs both services as
UID/GID 10001 with read-only roots, a bounded writable `/tmp`, dropped Linux
capabilities, `no-new-privileges`, health checks, and resource limits. The approved
checkpoint can be packaged into an immutable image or mounted read-only.

CI validates Python 3.11 installation, formatting, linting, typing, branch
coverage, fixture-based unit/integration tests, checkpoint compatibility when the
artifact is available, and the live container security/readiness contracts.

## 9. Current limitations

- The model emits exactly six positions and only its stored vocabulary.
- The data source and license history are incomplete.
- There is no independent external-test or calibration set.
- The approved checkpoint's historical split provenance is incomplete.
- Rare/unseen characters and new generator families remain a generalization risk.
- Confidence is a review aid, not a universal probability.
- Process-local metrics require production aggregation and alerting.
- The deployment platform still needs TLS, authentication, rate limiting,
  artifact signing/scanning, backups, and centralized observability.

## 10. What I would improve with more data

First, I would collect a separately sourced, authorized external-test set and a
dedicated calibration split. Every sample would carry generator-family,
collection, license, and authorization metadata. I would target rare/unseen
characters and keep related generator families grouped during splitting.

Then I would run repeated V1 and V2 experiments using the same versioned evidence
contract. Promotion would consider exact accuracy, CER, calibration, latency,
model memory, and stability—not architecture novelty. A variable-length decoder
or transformer would be considered only after the dataset and measured results
justify the added complexity.

Operationally, I would add deployment-level tracing, centralized metrics and
alerts, signed model/image provenance, automated vulnerability scanning, and a
staged canary plus rollback workflow.

## Evidence I can show in an interview

| Question | Repository evidence |
|---|---|
| How is leakage controlled? | `artifacts/split_manifest.csv`, dataset card, audit code/tests |
| Can training overwrite production? | Safe candidate defaults and overwrite guards/tests |
| Are metrics reproducible? | Evaluation summary, checkpoint hash, dataset/split versions |
| Is the API safe at the upload boundary? | Typed validation plus corrupt/oversized/MIME/batch tests |
| What happens without a model? | `/health` stays live, `/ready` returns 503, CI container test |
| Can you roll back? | Immutable image/read-only mount strategy and operations runbook |
| Did you overclaim? | Provisional evidence status and external-test `pending` state |

The strongest story is not “I achieved 92%.” It is “I built a system that knows
exactly what that 92% does—and does not—prove.”

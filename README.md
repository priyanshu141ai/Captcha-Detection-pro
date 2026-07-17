# CipherLens

CipherLens is a local Streamlit and FastAPI system that reads six-character CAPTCHA images with a compact CRNN (convolutional recurrent neural network) and six position-wise outputs from a shared character classifier.

Use CipherLens only with synthetic images or systems and data you own or are
explicitly authorized to test. The project does not automate browser interaction,
CAPTCHA submission, or access-control bypass.

## Documentation

See [CipherLens Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) for the architecture, dataset contract, training pipeline, inference API, tests, checkpoint format, and troubleshooting guide.

See [Operations Guide](docs/OPERATIONS.md) for Docker deployment, health checks, CI/CD, workload controls, security, model promotion, and rollback.

## Why this model

The two supplied batches contain 1,000 images. Every image is 151×41 pixels and every label has six mixed-case alphanumeric characters. A fixed-length CRNN is the best practical baseline here because it:

- reads the complete image as a six-step sequence, so characters do not need to be manually segmented;
- handles small horizontal shifts and overlapping/noisy characters;
- is much smaller and faster on CPU than transformer OCR models such as TrOCR or SVTR;
- avoids the blank-collapse that CTC models commonly exhibit with very small datasets;
- can be trained from scratch on a small, domain-specific dataset when paired with augmentation and class weighting.

The dataset is the main accuracy constraint. It contains 1,000 strings and several characters appear once. Add more labeled examples—especially for rare characters—before treating confidence or validation accuracy as production-grade.

### Current checkpoint

The included checkpoint was warm-started from batch 0, fine-tuned on both batches, and selected on a deterministic 800/200 coverage-aware split:

- character accuracy: **98.58%**;
- complete six-character accuracy: **92%**;
- observed character classes: **43**;
- model size: about **1.19 million parameters**.

These figures describe this dataset only. The displayed confidence is the geometric mean of the six softmax probabilities and is not statistically calibrated.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

This installs the `cipherlens` package plus the formatting, linting, typing, and
coverage tools used by contributors. For a runtime-only local installation, use
`python -m pip install --editable .` instead. `requirements.txt` remains available
for container and compatibility installs.

The repository includes extracted training images in `data/batch_0` and
`data/batch_1`.

## Configuration

Validated defaults live in `configs/default.yaml`. Runtime environment variables
override the YAML file:

| Variable | Default | Purpose |
|---|---:|---|
| `CIPHERLENS_CONFIG` | `configs/default.yaml` | Alternate YAML settings file |
| `CIPHERLENS_CHECKPOINT` | `models/captcha_crnn.pt` | Approved checkpoint path |
| `CIPHERLENS_TORCH_THREADS` | `2` | Process-wide CPU thread count |
| `CIPHERLENS_CONFIDENCE_THRESHOLD` | `0.75` | Manual-review warning threshold |
| `CIPHERLENS_MAX_UPLOAD_BYTES` | `10485760` | Upload byte limit |
| `CIPHERLENS_MAX_UPLOAD_PIXELS` | `4000000` | Decoded image pixel limit |
| `CIPHERLENS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `CIPHERLENS_LOG_FORMAT` | `console` | `console` or newline-delimited `json` |
| `CIPHERLENS_API_MAX_BATCH_SIZE` | `8` | Maximum images per batch request |
| `CIPHERLENS_API_MAX_CONCURRENCY` | `1` | Concurrent model inference jobs per process |
| `CIPHERLENS_API_URL` | `http://127.0.0.1:8000` | Owned CipherLens backend URL used by Streamlit |
| `CIPHERLENS_API_TIMEOUT_SECONDS` | `15` | Streamlit-to-API request timeout |
| `CIPHERLENS_LOCAL_FALLBACK` | `true` | Use the approved local model after retryable API failures |

Invalid values fail at startup with a field-specific message. Copy
`.env.example` to `.env` for local Compose overrides; never commit `.env`.

## Dataset audit

```powershell
python -m scripts.audit_dataset
```

This non-destructive command validates labels, vocabulary, image decoding and
dimensions; groups exact hashes, perceptual near-duplicates, and repeated labels;
then writes `artifacts/dataset_report.json`, three CSV reports, and
[the dataset card](docs/dataset-card.md). The current version has 1,000 valid
samples, preserves the seed-42 800/200 development split, and has no configured
external test set, so external evaluation remains pending.

## Train

Training must write to a candidate path so the approved checkpoint is not
overwritten:

```powershell
python train.py `
  --output models/captcha_crnn_candidate.pt `
  --history-output artifacts/candidate-training-history.json
```

Useful overrides:

```powershell
python train.py --config configs/default.yaml `
  --epochs 80 --batch-size 32 --device cpu `
  --output models/captcha_crnn_candidate.pt
```

Train on both included batches:

```powershell
python train.py --extra-dataset requirements2.txt data/batch_1 `
  --output models/captcha_crnn_candidate.pt
```

Warm-start from the existing checkpoint when extending the character set:

```powershell
python train.py --extra-dataset requirements2.txt data/batch_1 `
  --init-checkpoint models/captcha_crnn.pt --learning-rate 0.0002
```

The best checkpoint is written to the explicit candidate path. Review independent
evaluation evidence before promoting a candidate; do not replace
`models/captcha_crnn.pt` during routine training.

Training verifies image hashes and assignments against
`artifacts/split_manifest.csv`. Each candidate stores architecture,
preprocessing, dataset/split versions, configuration, metrics, Git commit, and
creation time. Resume an interrupted run with:

```powershell
python train.py --resume-checkpoint artifacts/candidate-training-resume.pt
```

Optional MLflow 3 tracking remains disabled unless requested:

```powershell
python -m pip install --editable ".[tracking]"
python train.py --mlflow --mlflow-experiment CipherLens
```

## Evaluate

```powershell
python -m scripts.evaluate_model
```

Evaluation verifies every image hash against the versioned manifest and writes
metrics, failures, calibration bins, a confusion matrix, a reliability diagram,
and [the model card](docs/model-card.md). The current checkpoint lacks enough
training-split provenance to rule out overlap with the newer manifest, so these
validation results are provisional. External evaluation remains pending.

Optional temperature scaling is validation-only and does not modify the model:

```powershell
python -m scripts.evaluate_model --temperature-scale
```

## Model experiments

Model V1 remains the production baseline. Model V2 is an isolated CRNN-CTC
experiment with safe candidate defaults:

```powershell
python -m scripts.train_ctc_experiment
python -m scripts.evaluate_ctc_model
python -m scripts.compare_models
```

No V2 candidate was trained for the committed comparison, so its metrics remain
blank. Model V3 transformer work is deferred because 1,000 images and no external
test set do not justify the dependency, compute, and overfitting risk. See
[the comparison decision](docs/model-comparison.md).

## Run the app

```powershell
streamlit run app.py
```

Open `http://localhost:8501`, upload a PNG/JPG CAPTCHA, and select **Recognize text**.
The UI validates the image locally, calls the configured CipherLens API, and
shows model version, inference latency, and serving path. If the API is
unavailable and fallback is enabled, the approved local checkpoint is used.

## Run the API

```powershell
python -m uvicorn cipherlens.api:app --host 127.0.0.1 --port 8000
```

OpenAPI docs are at `http://127.0.0.1:8000/docs`. Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "file=@data/batch_0/captcha_00000.png;type=image/png"
```

The backend loads the checkpoint once per process, exposes health/readiness,
model information, single and batch prediction, and Prometheus-compatible
metrics. It validates extension, MIME type, bytes, decoded format, pixels, and
image integrity; uploaded bytes are not logged or persisted.

## Production container

```powershell
docker compose up --build -d
```

The production image runs as non-root, uses a read-only filesystem, includes a
health check, and excludes training data. See the operations guide before
deployment.

## Verify

```powershell
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m scripts.verify_runtime
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pip check
```

Use the recognizer only with CAPTCHA images and systems you own or are authorized to test.

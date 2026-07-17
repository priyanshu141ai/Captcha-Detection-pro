# CipherLens Operations Guide

## Production topology

CipherLens is a stateless Streamlit inference service. The trained checkpoint is
loaded once per application process and cached for subsequent predictions.
Uploaded files are validated and processed in memory; the application does not
deliberately persist them.

The production Docker image contains only the runtime application, source
package, visual asset, Streamlit configuration, and model checkpoint. Training
images and training tooling are excluded from the image.

## Local startup

```powershell
cd "C:\path\to\Captcha-Detection"
.\.venv\Scripts\python.exe -m pip install --editable ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501`.

Start the separate inference API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn cipherlens.api:app `
  --host 127.0.0.1 --port 8000
```

OpenAPI is available at `http://127.0.0.1:8000/docs`. The current Docker image
continues to serve Streamlit; API container topology is handled in Milestone 9.

## Docker startup

```powershell
docker compose up --build -d
docker compose ps
```

Inspect logs:

```powershell
docker compose logs --follow --tail 100 cipherlens
```

Stop the service:

```powershell
docker compose down
```

The container runs as a non-root user with a read-only root filesystem,
`no-new-privileges`, a bounded temporary filesystem, and an application health
check.

## Health checks

Liveness endpoint:

```text
GET /_stcore/health
```

PowerShell check:

```powershell
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health -UseBasicParsing
```

The Docker health check verifies both the checkpoint file and the Streamlit
health endpoint. CI additionally loads the checkpoint and performs known-image
predictions from both data batches.

FastAPI service checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/model-info
Invoke-WebRequest http://127.0.0.1:8000/metrics -UseBasicParsing
```

`/health` reports process liveness. `/ready` returns HTTP 503 until the model is
loaded. A missing or invalid checkpoint keeps liveness available while readiness
and model-dependent endpoints remain unavailable.

## Runtime configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `CIPHERLENS_CONFIG` | `configs/default.yaml` | YAML configuration file |
| `CIPHERLENS_CHECKPOINT` | `models/captcha_crnn.pt` | Model checkpoint path |
| `CIPHERLENS_TORCH_THREADS` | `2` | CPU threads per application process |
| `CIPHERLENS_CONFIDENCE_THRESHOLD` | `0.75` | Threshold for manual-verification warning |
| `CIPHERLENS_MAX_UPLOAD_BYTES` | `10485760` | Maximum upload size in bytes |
| `CIPHERLENS_MAX_UPLOAD_PIXELS` | `4000000` | Maximum decoded image pixel count |
| `CIPHERLENS_LOG_LEVEL` | `INFO` | Application logging level |
| `CIPHERLENS_LOG_FORMAT` | `console` | `console` or newline-delimited `json` logs |
| `CIPHERLENS_API_MAX_BATCH_SIZE` | `8` | Maximum files in one API batch |
| `CIPHERLENS_API_MAX_CONCURRENCY` | `1` | Concurrent inference jobs per API process |
| `CIPHERLENS_API_URL` | `http://127.0.0.1:8000` | CipherLens API used by Streamlit |
| `CIPHERLENS_API_TIMEOUT_SECONDS` | `15` | Frontend backend-call timeout |
| `CIPHERLENS_LOCAL_FALLBACK` | `true` | Enable retryable-error local fallback |

Copy `.env.example` to `.env` to override Compose defaults. Do not commit `.env`
files or secrets. Configuration is validated at startup; invalid integers,
thresholds, paths, log levels, and log formats fail with an actionable message.
Runtime environment variables override `configs/default.yaml`.

## Workload controls

- The checkpoint is loaded lazily and cached once per process.
- A checkpoint modification-time and size signature invalidates stale cache
  entries after model replacement.
- PyTorch CPU threads default to two to prevent process oversubscription.
- Streamlit file watching, automatic reruns, telemetry, and detailed client
  errors are disabled in production configuration.
- Uploads are limited to 10 MB and 4,000,000 decoded pixels.
- API multipart request bodies, file counts, and concurrent inference jobs are
  bounded before model execution.
- Every API request receives a generated `X-Request-ID`; uploaded names and
  bytes are excluded from logs and metrics.
- Streamlit validates uploads before transmission and falls back only for
  network or HTTP 5xx failures. HTTP 4xx validation errors remain visible and do
  not trigger local inference.
- Training caches compressed image bytes in memory by default to avoid repeated
  disk reads.
- `--num-workers` enables parallel data loading after workload-specific
  benchmarking. Keep it at zero on Windows unless measurements justify a
  change.
- `--no-cache-images` trades disk I/O for lower memory use.

## Test and release commands

Before candidate training, regenerate and review the dataset identity:

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_dataset
git diff -- artifacts docs/dataset-card.md
```

The audit never deletes data. A nonzero exit means validation errors require
manual review. Training and tuning must not consume manifest rows marked
`external_test`.

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py train.py src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m coverage erase
.\.venv\Scripts\python.exe -m coverage run -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m coverage report
.\.venv\Scripts\python.exe -m scripts.verify_runtime
.\.venv\Scripts\python.exe -m pip check
```

Container validation:

```powershell
docker compose config
docker build --tag cipherlens:release .
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests. It:

1. installs the project and verifies its dependency graph;
2. checks formatting, linting, practical typing, and source compilation;
3. runs generated-fixture unit, API integration, and application tests separately;
4. enforces at least 85% branch coverage across `cipherlens`;
5. when approved artifacts are present, verifies batch-0 and batch-1 predictions;
6. validates Compose and builds the production Docker image.

CI uses least-privilege repository permissions, cancels superseded runs, and
applies job timeouts. Third-party Actions are pinned to reviewed commit SHAs;
Dependabot checks Python, GitHub Actions, and Docker base image updates weekly.
The required coverage gate does not depend on the full dataset or approved
checkpoint; their separate compatibility smoke skips cleanly when absent.

## Model promotion

Train to a candidate path instead of overwriting production immediately:

```powershell
.\.venv\Scripts\python.exe train.py `
  --extra-dataset requirements2.txt data/batch_1 `
  --init-checkpoint models/captcha_crnn.pt `
  --output models/captcha_crnn_candidate.pt `
  --history-output artifacts/candidate-history.json `
  --resume-output artifacts/candidate-resume.pt `
  --learning-rate 0.0002
```

Resume safely with `--resume-checkpoint artifacts/candidate-resume.pt`. The CLI
refuses to use the approved production checkpoint as either candidate or resume
output. It verifies selected image hashes against the versioned split manifest.

Compare the candidate on an independent test set. Promote it only after exact
accuracy, per-character accuracy, latency, and new-character coverage pass the
release criteria. Git history should retain the previous checkpoint for
rollback.

## Evaluation reports

Generate validation diagnostics without modifying the checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_model
```

The command verifies manifest hashes and writes CSV/JSON reports, failed
predictions, a confusion matrix, a reliability diagram, and `docs/model-card.md`.
Latency is a warmed-up, single-sample model-forward benchmark that excludes image
decode and preprocessing. Results are provisional when checkpoint split metadata
does not match the evaluated manifest.

`--split external_test` exits successfully with a clear pending message when no
authorized external set is configured. `--temperature-scale` fits one scalar on
validation only; it is diagnostic and must not be presented as independent
calibration evidence.

## Experimental model workflow

Model V2 is isolated from the production checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_ctc_experiment `
  --output models/captcha_crnn_ctc_candidate.pt
.\.venv\Scripts\python.exe -m scripts.evaluate_ctc_model
.\.venv\Scripts\python.exe -m scripts.compare_models
```

The training command refuses to overwrite `models/captcha_crnn.pt`. Candidate and
history artifacts remain ignored until explicitly reviewed. A missing candidate
or external split is reported as pending and does not create zero-valued metrics.

Promotion is never automatic. The registry requires aligned versioned external
evidence, at least two training runs, accuracy/CER/ECE gates, bounded latency and
CPU model-tensor memory, and explicit approval. Model V1 remains the rollback-safe
default until a challenger satisfies every gate.

## Rollback

1. Restore the previously approved `models/captcha_crnn.pt` from source control
   or the artifact registry.
2. Rebuild and redeploy the container.
3. Confirm `/_stcore/health` and run `python -m scripts.verify_runtime`.
4. Record the failed model version and validation evidence.

## Scaling notes

Each worker process loads its own model copy. Scale process count only after
measuring memory and CPU. Streamlit sessions use WebSockets, so a multi-instance
deployment should use a load balancer with WebSocket support and session
affinity. For high request volume or API clients, separate inference into a
dedicated HTTP API and keep Streamlit as the presentation layer.

## Security baseline

- Accept only validated PNG and JPEG uploads.
- Reject empty, oversized, malformed, and excessive-pixel images.
- Load checkpoints with PyTorch's restricted `weights_only=True` mode.
- Do not expose internal exceptions or stack traces to users.
- Run containers as non-root with a read-only filesystem.
- Keep CORS and XSRF protection enabled.
- Patch pinned dependencies through reviewed Dependabot pull requests.
- Use CipherLens only on systems and CAPTCHA images you are authorized to test.

## Remaining production risks

- The current validation set contains only 200 images from the supplied data
  source and is not an independent real-world test set.
- Several character classes remain severely underrepresented.
- Confidence is not statistically calibrated.
- Streamlit is suitable for an internal tool or moderate traffic, not a
  high-throughput inference API.
- Metrics, tracing, alerting, TLS termination, artifact signing, and centralized
  log shipping must be supplied by the deployment platform.

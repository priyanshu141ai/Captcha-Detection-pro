# CipherLens Operations Guide

For system boundaries and request flows, see [Architecture](architecture.md).
For evaluation evidence and limitations, see the [Model Card](model-card.md).

## Production topology

Compose runs two isolated services from one CPU-only production image:

- `api`: FastAPI model serving on port 8000;
- `cipherlens`: Streamlit presentation on port 8501, calling `http://api:8000`.

The checkpoint is loaded once per process. Uploaded files are validated and
processed in memory; neither service deliberately persists them. The multi-stage
image contains runtime dependencies, application code, UI assets, configuration,
and the approved checkpoint. Build tooling, tests, reports, candidate models,
training images, local configuration, and secrets are excluded.

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

OpenAPI is available at `http://127.0.0.1:8000/docs`.

## Docker startup

```powershell
docker compose up --build --detach --wait
docker compose ps
```

Open Streamlit at `http://127.0.0.1:8501` and OpenAPI at
`http://127.0.0.1:8000/docs`.

Inspect logs:

```powershell
docker compose logs --follow --tail 100 api cipherlens
```

Stop the service:

```powershell
docker compose down
```

Both containers run as UID/GID 10001 with all Linux capabilities dropped, a
read-only root filesystem, `no-new-privileges`, graceful `SIGTERM` handling, a
bounded writable `/tmp`, resource limits, and service-specific health checks.

## Health checks

Liveness endpoint:

```text
GET /_stcore/health
```

PowerShell check:

```powershell
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health -UseBasicParsing
```

The Streamlit health check verifies process liveness. Compose waits for FastAPI
readiness before starting Streamlit. CI starts both containers and verifies
health, UID, read-only behavior, temporary writes, and CPU-only PyTorch.

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
Runtime environment variables override `configs/default.yaml`. The example uses
the Compose hostname `http://api:8000`; source-based local startup keeps the
configuration default `http://127.0.0.1:8000`.

Compose-only controls:

| Environment variable | Default | Purpose |
|---|---:|---|
| `CIPHERLENS_IMAGE` | `cipherlens:local` | Versioned image/tag used by both services |
| `CIPHERLENS_FRONTEND_API_URL` | `http://api:8000` | Backend URL injected into Streamlit by Compose |
| `CIPHERLENS_API_PORT` | `8000` | Host API port |
| `CIPHERLENS_FRONTEND_PORT` | `8501` | Host Streamlit port |
| `CIPHERLENS_API_CPUS` | `2.0` | API CPU limit |
| `CIPHERLENS_API_MEMORY` | `2G` | API memory limit |
| `CIPHERLENS_FRONTEND_CPUS` | `1.0` | Streamlit CPU limit |
| `CIPHERLENS_FRONTEND_MEMORY` | `1G` | Streamlit memory limit |
| `CIPHERLENS_MODEL_HOST_PATH` | unset | Absolute approved checkpoint path for mount mode |

Tune limits from measured workload evidence. Keep one API worker because each
worker loads another model copy; scale service replicas only after measuring CPU
and memory.

## Model packaging and mounting

The default strategy packages only `models/captcha_crnn.pt` into a versioned
image. Candidate checkpoints are excluded by `.dockerignore`. Tag the image with
the approved model/application version and retain the previous tag for rollback.

For a deployment-managed checkpoint, set an absolute path and apply the explicit
mount overlay:

```powershell
$env:CIPHERLENS_MODEL_HOST_PATH = "C:/models/approved/captcha-crnn-v1.pt"
docker compose -f compose.yaml -f compose.model-mount.yaml `
  up --build --detach --wait
```

The bind is read-only in both services. Never point it at an unreviewed candidate.
With a missing or invalid checkpoint, FastAPI remains live at `/health`, returns
503 from `/ready`, and becomes unhealthy; Compose does not start the dependent
frontend. Logs contain the safe model-unavailable event without checkpoint data.

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
docker compose up --build --detach --wait
docker compose down
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests. It:

1. installs the project and verifies its dependency graph;
2. checks formatting, linting, practical typing, and source compilation;
3. runs generated-fixture unit, API integration, and application tests separately;
4. enforces at least 85% branch coverage across `cipherlens`;
5. when approved artifacts are present, verifies batch-0 and batch-1 predictions;
6. starts the production stack and verifies health and container security;
7. confirms a missing model keeps liveness up and readiness down.

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

After explicit approval, copy the approved artifact to the packaged checkpoint
path and build a new immutable `CIPHERLENS_IMAGE` tag, or update the read-only
mount path. Record the checkpoint SHA-256, image digest, evidence report, approver,
and previous rollback tag. Verify `/model-info` before routing traffic.

## Rollback

1. Select the previous approved image tag or read-only checkpoint path.
2. For packaged mode, set `CIPHERLENS_IMAGE` to the previous tag and run
   `docker compose up --detach --no-build --force-recreate --wait`.
3. For mount mode, restore `CIPHERLENS_MODEL_HOST_PATH` and run both Compose files
   with `--force-recreate --wait`.
4. Confirm `/ready`, `/model-info`, `/_stcore/health`, and one authorized smoke
   prediction.
5. Record the failed model version, image/checkpoint digest, and rollback time.

## Scaling notes

Each API worker or Streamlit local-fallback process loads its own model copy.
Scale only after measuring memory, CPU, queueing, and latency. Streamlit sessions
use WebSockets, so multi-instance deployment needs WebSocket support and session
affinity. Put TLS, authentication, rate limiting, and request-size enforcement at
the trusted ingress; do not expose the service to unauthorized third parties.

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

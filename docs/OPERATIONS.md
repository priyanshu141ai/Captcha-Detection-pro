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
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.verify_runtime
```

Container validation:

```powershell
docker compose config
docker build --tag cipherlens:release .
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests. It:

1. installs the project and development quality tools;
2. checks formatting, linting, and practical typing;
3. compiles Python sources;
4. runs unit and integration tests;
5. loads the production checkpoint and verifies batch-0 and batch-1 predictions;
6. builds the production Docker image.

CI uses least-privilege repository permissions, cancels superseded runs, and
applies job timeouts. Dependabot checks Python, GitHub Actions, and Docker base
image updates weekly.

## Model promotion

Train to a candidate path instead of overwriting production immediately:

```powershell
.\.venv\Scripts\python.exe train.py `
  --extra-dataset requirements2.txt data/batch_1 `
  --init-checkpoint models/captcha_crnn.pt `
  --output models/captcha_crnn_candidate.pt `
  --history-output artifacts/candidate-history.json `
  --learning-rate 0.0002
```

Compare the candidate on an independent test set. Promote it only after exact
accuracy, per-character accuracy, latency, and new-character coverage pass the
release criteria. Git history should retain the previous checkpoint for
rollback.

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

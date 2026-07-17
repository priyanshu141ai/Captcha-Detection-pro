# CipherLens Repository Guide

## Project purpose

CipherLens is an educational machine-learning system for recognizing fixed-length,
six-character text in synthetic, owned, or explicitly authorized CAPTCHA-style
images. The production baseline is a compact PyTorch CRNN with six position-wise
outputs, CPU inference, and a Streamlit interface.

## Authorized-use restriction

- Work only with synthetic images or systems and data the operator owns or is
  explicitly authorized to test.
- Do not add browser automation, CAPTCHA submission, authentication bypass,
  third-party website integration, or features intended to defeat access controls.
- Do not log, persist, or transmit uploaded image bytes unless a documented,
  authorized requirement explicitly calls for it.
- Treat reported metrics as results for the evaluated dataset and split only.

## Repository architecture

Current compatibility entry points:

- `app.py`: Streamlit UI and direct local inference.
- `train.py`: training CLI.
- `src/cipherlens/`: installable package containing configuration, data, model,
  training, evaluation, inference, API, monitoring, logging, and utility modules.
- `src/cipherlens/api/`: separate FastAPI service; start with
  `python -m uvicorn cipherlens.api:app --host 127.0.0.1 --port 8000`.
- `src/data.py`, `src/model.py`, `src/inference.py`, and `src/validation.py`:
  backward-compatible imports for existing callers.
- `scripts/verify_runtime.py`: checkpoint smoke verification.
- `scripts/train_ctc_experiment.py`: isolated optional Model V2 training.
- `scripts/compare_models.py`: evidence-aligned model comparison.
- `tests/`: current `unittest` suite.

The installable `src/cipherlens/` package contains configuration, logging, data,
models, training, evaluation, inference, API, monitoring, and utilities. Keep the
Streamlit frontend separate from the FastAPI service, and keep training code out
of both runtime entry points. Preserve compatibility wrappers while callers still
use the current `src.*` imports.

## Supported Python version

Use Python 3.11. The local environment, Dockerfile, and CI baseline are all
Python 3.11. Do not broaden support until CI tests the additional version.

## Windows setup and commands

Run commands from the repository root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

Run verification:

```powershell
python -m compileall -q app.py train.py src tests scripts
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m coverage erase
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m scripts.verify_runtime
python -m pip check
docker compose config
```

Start the current application:

```powershell
python -m uvicorn cipherlens.api:app --host 127.0.0.1 --port 8000
python -m streamlit run app.py
```

Start the production API and frontend containers:

```powershell
docker compose up --build --detach --wait
```

Training must write to a candidate artifact and must not overwrite the production
checkpoint:

```powershell
python train.py `
  --extra-dataset requirements2.txt data/batch_1 `
  --output models/captcha_crnn_candidate.pt `
  --history-output artifacts/candidate-training-history.json
```

Do not run plain `python train.py` while its default output is
`models/captcha_crnn.pt` unless the user has explicitly approved replacing that
artifact and a recoverable copy exists.

## Coding conventions

- Use typed, modular Python and `from __future__ import annotations`.
- Use `pathlib.Path`, explicit encodings, dataclasses or validated settings
  models, and standard package imports.
- Keep configuration centralized; validate environment variables at startup and
  return actionable errors.
- Use structured logging. Never log uploaded bytes, secrets, or full sensitive
  request bodies.
- Keep preprocessing identical between training evaluation and inference unless
  a versioned, tested change is intentional.
- Preserve fixed-length Model V1 behavior until a measured candidate satisfies
  documented promotion criteria.
- Prefer incremental compatibility shims over a repository-wide rewrite.
- Format and lint with the tools configured in `pyproject.toml`.

## Testing requirements

- Add or update tests for every behavior change and every fixed bug.
- Use tiny generated fixtures in unit and CI tests; do not make new tests depend
  on the full dataset or a private model.
- Keep a separate compatibility smoke test for the approved production
  checkpoint when that artifact is available.
- Test deterministic inference, preprocessing, codec round trips, model output
  shape, checkpoint compatibility, missing-model behavior, and invalid uploads.
- API work must cover health, readiness, model info, single and batch prediction,
  corrupt and oversized images, unsupported MIME types, and batch limits.
- Run relevant tests, linting, formatting checks, and practical type checks before
  a milestone is complete. Report actual results; never imply an unrun check
  passed.

## Dataset rules

- A label row is `<filename> <six-character-label>` and must reference one owned
  or authorized image.
- Validate readability, decoded dimensions, label length, character vocabulary,
  duplicate paths, exact hashes, perceptual near-duplicates, and split leakage.
- Never delete suspicious data automatically. Report it for human review.
- Split deterministically from a versioned manifest and group related or
  duplicate samples before splitting.
- Keep validation, calibration, and external-test roles distinct. Do not tune on
  the external test set.
- Record dataset provenance and authorization. Do not commit new generated,
  private, or large datasets without explicit approval; CI uses generated
  fixtures.
- Do not fabricate an external-test result when no external set exists.

## Model-artifact rules

- Never overwrite `models/captcha_crnn.pt` during development or routine
  training.
- Save candidates under a distinct path and retain the approved artifact for
  rollback.
- Load PyTorch checkpoints with `weights_only=True` and validate their schema.
- Version the architecture, preprocessing, vocabulary, normalization, dataset
  hash, configuration, validation metrics, Git commit, and creation time in every
  new checkpoint.
- Do not commit unnecessary candidates, temporary checkpoints, optimizer dumps,
  or unapproved large artifacts.
- Promote a model only from documented external evidence and explicit approval;
  a more complex architecture is not automatically better.

## Security constraints

- Validate extension, declared MIME type, byte size, decoded image format,
  dimensions, pixel count, and image integrity at service boundaries.
- Bound batch size, request size, CPU work, and inference concurrency.
- Keep secrets in environment or deployment secret stores, never in Git.
- Use request IDs and structured errors without leaking tracebacks to clients.
- Run containers as non-root with least privilege and a read-only filesystem where
  practical.
- Do not weaken Streamlit XSRF/CORS controls or container security controls
  without a documented reason and tests.

## Git and repository workflow

- Inspect `git status --short --branch` before editing and preserve unrelated
  user changes.
- `origin` is `priyanshu141ai/Captcha-Detection-pro`; the former repository is
  retained as `legacy`.
- Create a reviewable commit after each completed milestone and push validated
  changes to `origin` as requested by the repository owner.
- Never commit `.env`, `.streamlit/secrets.toml`, local virtual environments,
  generated histories, temporary artifacts, or credentials.
- Stop after each milestone with a change summary, commands and actual results,
  unresolved risks, and the next milestone.

## Definition of done

A change or milestone is done only when:

- existing behavior is preserved or an intentional change is documented;
- relevant automated tests pass and their real output is reported;
- formatting, linting, and practical type checks pass;
- documentation and reproducible Windows commands are current;
- dataset and model provenance are recorded without fabricated metrics;
- no secrets, unauthorized data, or unnecessary artifacts are included;
- the diff and Git status are reviewed;
- remaining risks and rollback implications are stated;
- a concise commit is created and pushed to `origin`; and
- work stops before the next milestone begins.

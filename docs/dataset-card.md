# CipherLens Dataset Card

> Generated deterministically by `python -m scripts.audit_dataset`. Source images are
> audited non-destructively; suspicious files are reported and never deleted.

## Identity

- Dataset: `cipherlens-repository-dataset`
- Dataset version: `143bbe9d6e498ae523d890f7da88ca35c6e54df33961984a31e21823b03f331f`
- Split version: `14486e3152d5e613cd7269ad13d0a9b24878ac9a79597a36b1452580f7f4da16`
- Valid samples: 1000
- Contract: 6 characters; 151x41 pixels

## Intended use

Educational recognition of synthetic, owned, or explicitly authorized CAPTCHA-style
images. The dataset must not be used to automate third-party access-control bypass.

## Sources and provenance

| Source | Role | Samples | Provenance | Authorization |
|---|---|---:|---|---|
| batch_0 | development | 500 | Repository-tracked CAPTCHA-style images; exact generation process is not documented. | Supplied by the repository owner for authorized educational use; independent license evidence is pending. |
| batch_1 | development | 500 | Repository-tracked CAPTCHA-style images; exact generation process is not documented. | Supplied by the repository owner for authorized educational use; independent license evidence is pending. |

The exact generator, collection date, license, and retention history are not recorded
in the repository. Maintainers must resolve those provenance gaps before treating the
dataset as independently redistributable evidence.

## Deterministic splits

| Split | Samples | Role |
|---|---:|---|
| Train | 800 | Model fitting only |
| Validation | 200 | Development evaluation |
| Calibration | 0 | Not configured; validation is not silently reused |
| External test | 0 | Independent evaluation only |

External evaluation is **pending** because no separately sourced authorized external-test dataset is configured.

Exact hashes, perceptual near-duplicates, and repeated labels are assigned one related
group before splitting. Development samples overlapping an external group are excluded.

## Character coverage

- Expected vocabulary: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Rare (1-5 occurrences): `Dfkltwxyz`
- Unseen overall: `019IJOUVghijmnoqsuv`
- Observed overall but unseen in validation: `Dkltwxyz`

Full per-character counts are stored in `artifacts/character_frequency.csv`.

## Quality and duplicate audit

- Validation errors: 0
- Exact/label/near-duplicate findings: 0
- Cross-split duplicate leakage: 0
- Near-duplicate threshold: 64-bit DCT pHash Hamming distance <= 4

Perceptual hashing is heuristic: geometric or generator-level relationships may remain
undetected. No generator-family metadata exists, so grouping currently relies on paths,
labels, exact hashes, and perceptual similarity.

## Reproduction

```powershell
python -m scripts.audit_dataset
```

Review `artifacts/dataset_report.json` and `artifacts/duplicate_report.csv` before
using a new dataset version for training or evaluation.

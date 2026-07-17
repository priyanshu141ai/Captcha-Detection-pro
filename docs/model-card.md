# CipherLens Model Card

## Model summary

CipherLens Model V1 is a compact position-wise CRNN for six-character synthetic,
owned, or explicitly authorized CAPTCHA-style images. It must not be integrated
with third-party sites or used to bypass access controls.

- Architecture: `captcha_crnn_positionwise` version `1.0`
- Checkpoint schema: `legacy-unversioned`
- Vocabulary size: `43`
- Parameters: `1,190,475`
- Checkpoint size: `4.56 MiB`
- CPU model tensors: `4.54 MiB`

## Evaluation evidence

| Metric | Validation |
|---|---:|
| Samples | 200 |
| Character accuracy | 98.5833% |
| Exact-string accuracy | 92.0000% |
| Character error rate | 1.4167% |
| Mean normalized edit distance | 0.014167 |
| Sequence expected calibration error | 0.037364 |

Evidence status: `provisional_checkpoint_training_split_unverified`. The evaluated checkpoint does not record enough training-split provenance to prove that it was trained without overlap with the newer versioned manifest. These values are therefore dataset-specific, provisional diagnostics—not promotion or external-generalization evidence.

External-test status: **pending**. No external-test score is
reported when that split is unavailable.

## Calibration and confidence

Temperature scaling was not applied because no independent calibration split exists.

The reliability diagram uses sequence confidence against exact-string correctness;
bars show observed accuracy and the lower panel shows sample count per bin.

## Runtime

Single-sample `preprocessed single-sample model forward` latency on `cpu`:
mean `4.557 ms`, median `4.486 ms`,
p95 `5.145 ms` across `50` measured runs
after `5` warmups. Image decode and preprocessing are excluded.

## Known limitations

- Fixed six-character output and checkpoint vocabulary.
- Dataset generation provenance and independent license evidence remain incomplete.
- No external-test or independent calibration split is currently configured.
- Confidence should not be interpreted as a universal probability of correctness.
- The current validation evidence may overlap with historical training data.

## Reproduce

```powershell
python -m scripts.evaluate_model
```

Artifacts: [comparison](../reports/evaluation/model_comparison.csv),
[failures](../reports/evaluation/failed_predictions.csv),
[confusion matrix](../reports/figures/confusion_matrix.png), and
[reliability diagram](../reports/figures/reliability_diagram.png). See also the
[model comparison](model-comparison.md).

Generated from dataset version `143bbe9d6e498ae523d890f7da88ca35c6e54df33961984a31e21823b03f331f` and split version
`14486e3152d5e613cd7269ad13d0a9b24878ac9a79597a36b1452580f7f4da16` at `2026-07-17T05:53:28.266351+00:00`.

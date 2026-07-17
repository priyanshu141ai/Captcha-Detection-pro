# CipherLens Model Comparison

## Decision

Model V1 remains the production default. It is the only model with measured
repository validation evidence, and that evidence is provisional because the
legacy checkpoint lacks versioned training-split provenance. No model is eligible
for promotion without aligned external-test evidence and repeated-run stability.

## Evidence table

| Model | Status | Exact | CER | Median ms | Model MiB | CPU tensors MiB | ECE | Stability | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v1-positionwise | production_baseline | 92.00% | 1.42% | 4.4859 | 4.5597 | 4.5450 | 0.0374 | not_available | retain_default_no_validated_challenger |
| v2-ctc | experimental_not_trained | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | not_available | not_eligible_missing_evidence |
| v3-transformer | deferred | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | not_available | deferred |

CSV blanks and table cells marked `Not measured` mean the evidence does not exist;
they are not zero values. Latency is warmed-up single-sample model-forward time.
CPU memory is resident model tensor
storage only, excluding framework and workspace overhead. Comparisons require the
same dataset version, split version, split role, and sample count; checkpoint
SHA-256 must match its evaluation summary.

Promotion review requires aligned versioned external-test evidence, at least two
training runs, exact accuracy no worse than V1, CER and ECE no worse than V1,
median latency within 10% of V1, CPU model tensors within 25% of V1, and explicit
human approval. Passing these gates permits review; it does not auto-promote.

## Architecture decisions

- **V1 position-wise CRNN:** retained as the rollback-safe production baseline.
- **V2 CRNN-CTC:** implemented as an optional experiment, but no candidate was
  trained or evaluated during this milestone.
- **V3 transformer OCR:** deferred. The current 1,000-image dataset, incomplete
  provenance, and missing external-test set do not justify the added dependency,
  compute, and overfitting risk.

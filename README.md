# CipherLens

CipherLens is a local Streamlit application that reads six-character CAPTCHA images with a compact CRNN (convolutional recurrent neural network) and six position-wise character classifiers.

## Documentation

See [CipherLens Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) for the architecture, dataset contract, training pipeline, inference API, tests, checkpoint format, and troubleshooting guide.

## Why this model

The supplied dataset contains 500 images. Every image is 151×41 pixels and every label has six mixed-case alphanumeric characters. A fixed-length CRNN is the best practical baseline here because it:

- reads the complete image as a six-step sequence, so characters do not need to be manually segmented;
- handles small horizontal shifts and overlapping/noisy characters;
- is much smaller and faster on CPU than transformer OCR models such as TrOCR or SVTR;
- avoids the blank-collapse that CTC models commonly exhibit with very small datasets;
- can be trained from scratch on a small, domain-specific dataset when paired with augmentation and class weighting.

The dataset is the main accuracy constraint. It contains only 500 strings and several characters appear once. Add more labeled examples—especially for rare characters—before treating confidence or validation accuracy as production-grade.

### Current checkpoint

The included checkpoint was selected on a deterministic 400/100 coverage-aware split:

- character accuracy: **98.67%**;
- complete six-character accuracy: **92%**;
- model size: about **1.19 million parameters**.

These figures describe this dataset only. The displayed confidence is the geometric mean of the six softmax probabilities and is not statistically calibrated.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository includes the extracted training images in `data/batch_0`.

## Train

```powershell
python train.py
```

Useful overrides:

```powershell
python train.py --epochs 80 --batch-size 32 --device cpu
```

The best checkpoint is written to `models/captcha_crnn.pt`; epoch metrics are written to `training_history.json`.

## Run the app

```powershell
streamlit run app.py
```

Open `http://localhost:8501`, upload a PNG/JPG CAPTCHA, and select **Recognize text**.

Use the recognizer only with CAPTCHA images and systems you own or are authorized to test.

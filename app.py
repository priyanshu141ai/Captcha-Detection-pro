from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.inference import (
    CaptchaRecognizer,
    CheckpointValidationError,
    InferenceAPIClient,
    InferenceAPIError,
    ServedPrediction,
    UploadLimits,
    UploadValidationError,
    load_uploaded_image,
)
from cipherlens.logging import configure_logging

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "assets" / "cipherlens-mark.png"

st.set_page_config(
    page_title="CipherLens · CAPTCHA recognition",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    SETTINGS = load_project_settings(ROOT)
except ConfigurationError as error:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("cipherlens").error("Invalid application configuration: %s", error)
    st.error(f"CipherLens configuration is invalid: {error}")
    st.stop()

LOGGER = configure_logging(SETTINGS.runtime.log_level, SETTINGS.runtime.log_format)
CHECKPOINT = SETTINGS.runtime.checkpoint_path
UPLOAD_LIMITS = UploadLimits(
    max_bytes=SETTINGS.runtime.max_upload_bytes,
    max_pixels=SETTINGS.runtime.max_upload_pixels,
)
CONFIDENCE_THRESHOLD = SETTINGS.runtime.confidence_threshold


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_resource(show_spinner=False)
def load_recognizer(
    checkpoint_path: str,
    checkpoint_modified_ns: int,
    checkpoint_size: int,
    torch_threads: int,
) -> CaptchaRecognizer:
    del checkpoint_modified_ns, checkpoint_size
    return CaptchaRecognizer(checkpoint_path, torch_threads=torch_threads)


@st.cache_resource(show_spinner=False)
def load_api_client(base_url: str, timeout_seconds: float) -> InferenceAPIClient:
    return InferenceAPIClient(base_url, timeout_seconds=timeout_seconds)


def predict_locally(image: Image.Image) -> ServedPrediction:
    checkpoint_stat = CHECKPOINT.stat()
    recognizer = load_recognizer(
        str(CHECKPOINT),
        checkpoint_stat.st_mtime_ns,
        checkpoint_stat.st_size,
        SETTINGS.runtime.torch_threads,
    )
    started = time.perf_counter()
    prediction = recognizer.predict(image)
    return ServedPrediction(
        text=prediction.text,
        confidence=prediction.confidence,
        per_character_confidence=prediction.per_character_confidence,
        model_version=recognizer.model_version,
        inference_time_ms=(time.perf_counter() - started) * 1000,
        request_id=None,
        source="local",
    )


def render_copy_button(text: str) -> None:
    javascript_text = json.dumps(text)
    st.iframe(
        f"""
        <button id="copyButton" aria-label="Copy recognized CAPTCHA text">Copy text</button>
        <script>
          const button = document.getElementById('copyButton');
          button.addEventListener('click', async () => {{
            const value = {javascript_text};
            try {{
              await navigator.clipboard.writeText(value);
              button.textContent = 'Copied';
            }} catch (error) {{
              const area = document.createElement('textarea');
              area.value = value;
              document.body.appendChild(area);
              area.select();
              document.execCommand('copy');
              area.remove();
              button.textContent = 'Copied';
            }}
            setTimeout(() => button.textContent = 'Copy text', 1600);
          }});
        </script>
        <style>
          html, body {{ margin: 0; background: transparent; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
          button {{
            width: 100%; height: 48px; border: 0; border-radius: 10px;
            background: #047857; color: #fff; font-weight: 700; font-size: 15px;
            cursor: pointer; transition: transform .15s ease, background .15s ease;
          }}
          button:hover {{ background: #066a4f; transform: translateY(-1px); }}
          button:focus-visible {{ outline: 3px solid rgba(4,120,87,.25); outline-offset: 2px; }}
        </style>
        """,
        width="stretch",
        height=50,
    )


st.markdown(
    """
    <style>
      :root {
        --ink: #0f172a;
        --muted: #64748b;
        --line: #dbe3ef;
        --accent: #4338ca;
        --accent-hover: #3730a3;
        --success: #047857;
        --canvas: #f8fafc;
      }
      .stApp { background: var(--canvas); color: var(--ink); }
      [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
      [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding: 28px 30px 24px;
      }
      .cipher-header {
        display: flex; align-items: center; gap: 11px; padding-bottom: 24px;
        border-bottom: 1px solid #e5eaf2;
      }
      .cipher-header img { width: 42px; height: 42px; object-fit: cover; border-radius: 10px; }
      .cipher-brand { font-size: 25px; line-height: 1; font-weight: 800; letter-spacing: -.7px; }
      .cipher-intro { text-align: center; max-width: 760px; margin: 38px auto 32px; }
      .cipher-intro h1 { margin: 0; font-size: clamp(34px, 4vw, 50px); letter-spacing: -2px; line-height: 1.08; }
      .cipher-intro p { margin: 14px auto 0; color: var(--muted); font-size: 17px; line-height: 1.6; }
      .panel-label { margin: 0 0 14px; font-size: 17px; font-weight: 750; letter-spacing: -.2px; }
      [data-testid="stFileUploader"] {
        padding: 22px; border: 1.5px dashed #818cf8; border-radius: 16px;
        background: rgba(255,255,255,.78);
      }
      [data-testid="stFileUploaderDropzone"] {
        padding: 22px 18px; background: #fff; border: 0; border-radius: 12px;
      }
      [data-testid="stFileUploaderDropzone"] button { min-height: 42px; }
      [data-testid="stImage"] {
        margin-top: 14px; padding: 20px; border-radius: 14px;
        background: #fff; border: 1px solid var(--line);
      }
      [data-testid="stImage"] img { width: 100%; image-rendering: auto; }
      .stButton > button {
        min-height: 48px; border-radius: 10px; font-weight: 750; font-size: 15px;
        border-color: #6366f1; transition: transform .15s ease, box-shadow .15s ease;
      }
      .stButton > button[kind="primary"] {
        background: var(--accent); border-color: var(--accent); color: #fff;
        box-shadow: 0 8px 20px rgba(67,56,202,.16);
      }
      .stButton > button[kind="primary"]:hover { background: var(--accent-hover); transform: translateY(-1px); }
      .st-key-result_panel {
        min-height: 374px; padding: 24px; border: 1px solid var(--line); border-radius: 16px;
        background: #fff; box-shadow: 0 16px 45px rgba(15,23,42,.05);
      }
      .result-display {
        margin-top: 18px; min-height: 168px; display: grid; place-items: center;
        padding: 22px; border: 1px solid #b9dfd2; border-radius: 13px;
        background: #f0fdf8;
      }
      .result-display .check {
        display: grid; place-items: center; width: 42px; height: 42px; margin: 0 auto 12px;
        border: 2px solid var(--success); border-radius: 50%; color: var(--success); font-size: 24px;
      }
      .prediction { color: var(--success); font-size: clamp(38px, 5vw, 62px); line-height: 1; font-weight: 800; letter-spacing: 3px; }
      .confidence-row { margin: 18px 0 14px; }
      .confidence-row .label { font-size: 14px; font-weight: 700; margin-bottom: 8px; }
      .confidence-value {
        padding: 12px 14px; border: 1px solid #cde7df; border-radius: 10px;
        background: #f6fffb; color: var(--success); text-align: center; font-weight: 750;
      }
      .confidence-warning {
        margin-top: 10px; color: #9a3412; font-size: 13px; font-weight: 650;
      }
      .result-meta {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin: 12px 0 14px;
      }
      .result-meta div {
        padding: 10px; border: 1px solid var(--line); border-radius: 9px; background: #f8fafc;
        text-align: center;
      }
      .result-meta span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }
      .result-meta strong { display: block; color: var(--ink); font-size: 13px; overflow-wrap: anywhere; }
      .result-empty {
        min-height: 270px; display: grid; place-items: center; padding: 28px;
        border: 1px dashed #cbd5e1; border-radius: 14px; color: var(--muted); text-align: center;
        background: #fbfdff;
      }
      .result-empty strong { display: block; margin-bottom: 7px; color: var(--ink); font-size: 17px; }
      .privacy-note { margin-top: 38px; padding-top: 22px; border-top: 1px solid #e5eaf2; color: var(--muted); text-align: center; font-size: 13px; }
      @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 18px 18px 20px; }
        .cipher-intro { margin: 28px auto 24px; text-align: left; }
        .cipher-intro h1 { letter-spacing: -1.3px; }
        .cipher-intro p { font-size: 15px; }
        .st-key-result_panel { min-height: 0; }
        .result-meta { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "prediction_notice" not in st.session_state:
    st.session_state.prediction_notice = None

logo_uri = image_data_uri(LOGO) if LOGO.is_file() else ""
st.markdown(
    f'<div class="cipher-header"><img src="{logo_uri}" alt="CipherLens logo">'
    '<div class="cipher-brand">CipherLens</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="cipher-intro"><h1>Read CAPTCHA text in seconds</h1>'
    "<p>Upload a CAPTCHA image and the recognition model will return its best prediction.</p></section>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")
uploaded_image: Image.Image | None = None
uploaded_data: bytes | None = None
uploaded_filename = ""
uploaded_content_type = ""

with left:
    st.markdown('<div class="panel-label">Image preview</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a CAPTCHA image here",
        type=("png", "jpg", "jpeg"),
        key=f"captcha_upload_{st.session_state.uploader_version}",
        help="PNG, JPG or JPEG · up to 10 MB",
    )
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        current_hash = hashlib.sha256(raw_bytes).hexdigest()
        if current_hash != st.session_state.file_hash:
            st.session_state.file_hash = current_hash
            st.session_state.prediction = None
            st.session_state.prediction_notice = None
        try:
            uploaded_image = load_uploaded_image(raw_bytes, UPLOAD_LIMITS)
            uploaded_data = raw_bytes
            uploaded_filename = uploaded.name
            uploaded_content_type = uploaded.type or "application/octet-stream"
            st.image(uploaded_image, width="stretch")
        except UploadValidationError as error:
            st.error(str(error))

    if SETTINGS.api.local_fallback_enabled and not CHECKPOINT.is_file():
        st.caption("Local fallback is unavailable; the FastAPI service must be running.")
    recognize = st.button(
        "Recognize text",
        type="primary",
        width="stretch",
        disabled=uploaded_image is None,
    )
    if recognize and uploaded_image is not None and uploaded_data is not None:
        st.session_state.prediction = None
        st.session_state.prediction_notice = None
        try:
            with st.spinner("Analyzing image…"):
                api_client = load_api_client(
                    SETTINGS.api.base_url, SETTINGS.api.request_timeout_seconds
                )
                st.session_state.prediction = api_client.predict(
                    uploaded_data,
                    filename=uploaded_filename,
                    content_type=uploaded_content_type,
                )
        except InferenceAPIError as error:
            LOGGER.warning(
                "Inference API request failed",
                extra={"event": "api_request_failed", "request_id": error.request_id},
            )
            if SETTINGS.api.local_fallback_enabled and error.retryable:
                try:
                    st.session_state.prediction = predict_locally(uploaded_image)
                    st.session_state.prediction_notice = (
                        "FastAPI was unavailable, so the approved local model was used."
                    )
                except (CheckpointValidationError, OSError):
                    LOGGER.error("Local fallback model is unavailable")
                    st.error(
                        "The inference API and local fallback model are unavailable. "
                        "Contact the application operator."
                    )
            else:
                reference = f" Reference: {error.request_id}." if error.request_id else ""
                st.error(f"{error.message}{reference}")
        except CheckpointValidationError:
            st.session_state.prediction = None
            LOGGER.exception("Checkpoint validation failed")
            st.error("The recognition model is unavailable. Contact the application operator.")
        except Exception:
            st.session_state.prediction = None
            LOGGER.exception("Unexpected recognition failure")
            st.error("Recognition failed unexpectedly. Check the server logs and try again.")

with right, st.container(key="result_panel", border=True):
    st.markdown('<div class="panel-label">Recognition result</div>', unsafe_allow_html=True)
    if st.session_state.prediction_notice:
        st.info(st.session_state.prediction_notice)
    prediction: ServedPrediction | None = st.session_state.prediction
    if prediction is None:
        st.markdown(
            '<div class="result-empty"><div><strong>Your result will appear here</strong>'
            "Upload an image and select “Recognize text”.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        safe_prediction = html.escape(prediction.text or "No text detected")
        safe_model_version = html.escape(prediction.model_version)
        source_label = "FastAPI" if prediction.source == "api" else "Local fallback"
        st.markdown(
            f'<div class="result-display"><div><div class="check">✓</div>'
            f'<div class="prediction">{safe_prediction}</div></div></div>'
            f'<div class="confidence-row"><div class="label">Confidence</div>'
            f'<div class="confidence-value">{prediction.confidence:.1%}</div></div>'
            '<div class="result-meta">'
            f"<div><span>Model</span><strong>{safe_model_version}</strong></div>"
            f"<div><span>Inference</span><strong>{prediction.inference_time_ms:.2f} ms</strong></div>"
            f"<div><span>Served by</span><strong>{source_label}</strong></div></div>",
            unsafe_allow_html=True,
        )
        if prediction.confidence < CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="confidence-warning">Low-confidence result — verify manually.</div>',
                unsafe_allow_html=True,
            )
        render_copy_button(prediction.text)
        if st.button("Try another image", width="stretch"):
            st.session_state.uploader_version += 1
            st.session_state.prediction = None
            st.session_state.file_hash = None
            st.session_state.prediction_notice = None
            st.rerun()

st.markdown(
    '<div class="privacy-note">Sent only to your configured CipherLens service · Image bytes are not stored</div>',
    unsafe_allow_html=True,
)

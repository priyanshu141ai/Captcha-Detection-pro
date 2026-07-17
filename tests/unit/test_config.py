from __future__ import annotations

import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

from cipherlens.config import ConfigurationError, load_project_settings, load_settings
from cipherlens.logging import configure_logging

ROOT = Path(__file__).resolve().parents[2]


class ConfigurationTests(unittest.TestCase):
    def test_project_defaults_match_existing_runtime_behavior(self) -> None:
        settings = load_project_settings(ROOT, environ={})

        self.assertEqual(settings.runtime.checkpoint_path, ROOT / "models/captcha_crnn.pt")
        self.assertEqual(settings.runtime.torch_threads, 2)
        self.assertEqual(settings.runtime.confidence_threshold, 0.75)
        self.assertEqual(settings.runtime.max_upload_bytes, 10 * 1024 * 1024)
        self.assertEqual(settings.api.max_batch_size, 8)
        self.assertEqual(settings.api.max_inference_concurrency, 1)
        self.assertEqual(settings.training.seed, 42)
        self.assertEqual(settings.training.output_path, Path("models/captcha_crnn_candidate.pt"))
        self.assertEqual(
            settings.training.split_manifest_path, Path("artifacts/split_manifest.csv")
        )
        self.assertEqual(settings.evaluation.split, "validation")
        self.assertEqual(settings.evaluation.output_path, Path("reports/evaluation"))
        self.assertFalse(settings.evaluation.temperature_scaling)
        self.assertEqual(len(settings.dataset.sources), 2)
        self.assertEqual(settings.dataset.expected_width, 151)
        self.assertEqual(settings.dataset.label_length, 6)

    def test_environment_overrides_are_validated_and_paths_resolve_from_root(self) -> None:
        settings = load_project_settings(
            ROOT,
            environ={
                "CIPHERLENS_CHECKPOINT": "models/candidate.pt",
                "CIPHERLENS_TORCH_THREADS": "3",
                "CIPHERLENS_CONFIDENCE_THRESHOLD": "0.8",
                "CIPHERLENS_MAX_UPLOAD_BYTES": "2048",
                "CIPHERLENS_MAX_UPLOAD_PIXELS": "8192",
                "CIPHERLENS_LOG_LEVEL": "warning",
                "CIPHERLENS_LOG_FORMAT": "json",
                "CIPHERLENS_API_MAX_BATCH_SIZE": "4",
                "CIPHERLENS_API_MAX_CONCURRENCY": "2",
            },
        )

        self.assertEqual(settings.runtime.checkpoint_path, ROOT / "models/candidate.pt")
        self.assertEqual(settings.runtime.torch_threads, 3)
        self.assertEqual(settings.runtime.confidence_threshold, 0.8)
        self.assertEqual(settings.runtime.max_upload_bytes, 2048)
        self.assertEqual(settings.runtime.max_upload_pixels, 8192)
        self.assertEqual(settings.runtime.log_level, "WARNING")
        self.assertEqual(settings.runtime.log_format, "json")
        self.assertEqual(settings.api.max_batch_size, 4)
        self.assertEqual(settings.api.max_inference_concurrency, 2)

    def test_invalid_environment_values_fail_with_field_context(self) -> None:
        cases = {
            "CIPHERLENS_TORCH_THREADS": "zero",
            "CIPHERLENS_CONFIDENCE_THRESHOLD": "1.5",
            "CIPHERLENS_MAX_UPLOAD_BYTES": "0",
            "CIPHERLENS_LOG_LEVEL": "verbose",
            "CIPHERLENS_LOG_FORMAT": "xml",
            "CIPHERLENS_API_MAX_BATCH_SIZE": "0",
            "CIPHERLENS_API_MAX_CONCURRENCY": "33",
        }
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                load_project_settings(ROOT, environ={name: value})

    def test_unknown_yaml_fields_and_missing_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.yaml"
            invalid.write_text("runtime:\n  surprise: true\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "surprise"):
                load_settings(invalid, project_root=root, environ={})
            with self.assertRaisesRegex(ConfigurationError, "not found"):
                load_settings(root / "missing.yaml", project_root=root, environ={})

    def test_environment_can_select_a_custom_configuration_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "custom.yaml"
            config.write_text("runtime:\n  confidence_threshold: 0.6\n", encoding="utf-8")

            settings = load_project_settings(ROOT, environ={"CIPHERLENS_CONFIG": str(config)})

        self.assertEqual(settings.source, config.resolve())
        self.assertEqual(settings.runtime.confidence_threshold, 0.6)

    def test_empty_configuration_environment_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "must not be empty"):
            load_project_settings(ROOT, environ={"CIPHERLENS_CONFIG": "  "})

    def test_validation_fraction_excludes_zero_and_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (0, 1):
                with self.subTest(value=value):
                    config = root / f"fraction-{value}.yaml"
                    config.write_text(
                        f"training:\n  validation_fraction: {value}\n", encoding="utf-8"
                    )
                    with self.assertRaises(ConfigurationError):
                        load_settings(config, project_root=root, environ={})

    def test_dataset_sources_require_unique_names_and_a_development_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "dataset.yaml"
            config.write_text(
                "dataset:\n"
                "  sources:\n"
                "    - name: external\n"
                "      labels_path: external.txt\n"
                "      images_path: images\n"
                "      role: external_test\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigurationError, "development source"):
                load_settings(config, project_root=root, environ={})

    def test_training_optimizer_settings_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {"weight_decay": 2, "gradient_clip_norm": 0, "scheduler_factor": 1}
            for name, value in cases.items():
                with self.subTest(name=name):
                    config = root / f"{name}.yaml"
                    config.write_text(f"training:\n  {name}: {value}\n", encoding="utf-8")
                    with self.assertRaises(ConfigurationError):
                        load_settings(config, project_root=root, environ={})

    def test_evaluation_settings_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {"ece_bins": 1, "latency_runs": 0, "device": "auto", "split": "train"}
            for name, value in cases.items():
                with self.subTest(name=name):
                    config = root / f"evaluation-{name}.yaml"
                    config.write_text(f"evaluation:\n  {name}: {value}\n", encoding="utf-8")
                    with self.assertRaises(ConfigurationError):
                        load_settings(config, project_root=root, environ={})


class StructuredLoggingTests(unittest.TestCase):
    def test_json_logs_include_only_bounded_safe_context(self) -> None:
        stream = io.StringIO()
        logger = configure_logging("INFO", "json")
        self.assertEqual(len(logger.handlers), 1)
        handler = logger.handlers[0]
        if not isinstance(handler, logging.StreamHandler):
            self.fail("Expected the configured handler to be a StreamHandler.")
        handler.setStream(stream)

        logger.info(
            "prediction completed",
            extra={"event": "prediction", "request_id": "request-1", "uploaded_bytes": b"no"},
        )
        payload = json.loads(stream.getvalue())

        self.assertEqual(payload["message"], "prediction completed")
        self.assertEqual(payload["event"], "prediction")
        self.assertEqual(payload["request_id"], "request-1")
        self.assertNotIn("uploaded_bytes", payload)

    def test_reconfiguration_does_not_duplicate_handlers(self) -> None:
        configure_logging("INFO", "console")
        logger = configure_logging("DEBUG", "console")

        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_invalid_logging_options_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            configure_logging("VERBOSE", "console")
        with self.assertRaises(ValueError):
            configure_logging("INFO", "xml")


if __name__ == "__main__":
    unittest.main()

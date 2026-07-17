"""Small in-process Prometheus metrics for the inference API."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class MetricsRegistry:
    """Store bounded, low-cardinality service counters without image data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http: defaultdict[tuple[str, str, int], int] = defaultdict(int)
        self._inference_requests: defaultdict[str, int] = defaultdict(int)
        self._inference_images = 0
        self._latency_count = 0
        self._latency_sum = 0.0
        self._latency_buckets = {boundary: 0 for boundary in _LATENCY_BUCKETS}
        self._model_load_failures = 0

    def observe_http(self, method: str, path: str, status_code: int) -> None:
        with self._lock:
            self._http[(method, path, status_code)] += 1

    def observe_inference(self, durations_seconds: Iterable[float], *, outcome: str) -> None:
        durations = tuple(durations_seconds)
        with self._lock:
            self._inference_requests[outcome] += 1
            if outcome != "success":
                return
            self._inference_images += len(durations)
            for duration in durations:
                self._latency_count += 1
                self._latency_sum += duration
                for boundary in _LATENCY_BUCKETS:
                    if duration <= boundary:
                        self._latency_buckets[boundary] += 1

    def record_model_load_failure(self) -> None:
        with self._lock:
            self._model_load_failures += 1

    def render(self, *, ready: bool) -> str:
        with self._lock:
            http = dict(self._http)
            inference_requests = dict(self._inference_requests)
            image_count = self._inference_images
            latency_count = self._latency_count
            latency_sum = self._latency_sum
            latency_buckets = dict(self._latency_buckets)
            load_failures = self._model_load_failures

        lines = [
            "# HELP cipherlens_model_ready Whether the inference model is loaded.",
            "# TYPE cipherlens_model_ready gauge",
            f"cipherlens_model_ready {int(ready)}",
            "# HELP cipherlens_model_load_failures_total Model startup load failures.",
            "# TYPE cipherlens_model_load_failures_total counter",
            f"cipherlens_model_load_failures_total {load_failures}",
            "# HELP cipherlens_http_requests_total HTTP requests by bounded route and status.",
            "# TYPE cipherlens_http_requests_total counter",
        ]
        for (method, path, status), count in sorted(http.items()):
            lines.append(
                f'cipherlens_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
        lines.extend(
            (
                "# HELP cipherlens_inference_requests_total Inference requests by outcome.",
                "# TYPE cipherlens_inference_requests_total counter",
            )
        )
        for outcome, count in sorted(inference_requests.items()):
            lines.append(f'cipherlens_inference_requests_total{{outcome="{outcome}"}} {count}')
        lines.extend(
            (
                "# HELP cipherlens_inference_images_total Successfully inferred images.",
                "# TYPE cipherlens_inference_images_total counter",
                f"cipherlens_inference_images_total {image_count}",
                "# HELP cipherlens_inference_latency_seconds Model inference latency per image.",
                "# TYPE cipherlens_inference_latency_seconds histogram",
            )
        )
        for boundary in _LATENCY_BUCKETS:
            lines.append(
                "cipherlens_inference_latency_seconds_bucket"
                f'{{le="{boundary:g}"}} {latency_buckets[boundary]}'
            )
        lines.extend(
            (
                f'cipherlens_inference_latency_seconds_bucket{{le="+Inf"}} {latency_count}',
                f"cipherlens_inference_latency_seconds_sum {latency_sum:.9f}",
                f"cipherlens_inference_latency_seconds_count {latency_count}",
            )
        )
        return "\n".join(lines) + "\n"


__all__ = ["MetricsRegistry"]

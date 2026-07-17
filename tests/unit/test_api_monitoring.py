from __future__ import annotations

import unittest

from cipherlens.monitoring import MetricsRegistry


class MetricsRegistryTests(unittest.TestCase):
    def test_prometheus_output_tracks_bounded_service_metrics(self) -> None:
        metrics = MetricsRegistry()
        metrics.observe_http("POST", "/predict", 200)
        metrics.observe_inference((0.01, 0.02), outcome="success")
        metrics.observe_inference((), outcome="failure")
        metrics.record_model_load_failure()

        output = metrics.render(ready=True)

        self.assertIn("cipherlens_model_ready 1", output)
        self.assertIn('path="/predict",status="200"} 1', output)
        self.assertIn('outcome="success"} 1', output)
        self.assertIn('outcome="failure"} 1', output)
        self.assertIn("cipherlens_inference_images_total 2", output)
        self.assertIn("cipherlens_model_load_failures_total 1", output)


if __name__ == "__main__":
    unittest.main()

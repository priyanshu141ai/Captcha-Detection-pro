from __future__ import annotations

import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_initial_screen_renders_without_exception(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual([button.label for button in app.button], ["Recognize text"])
        self.assertTrue(app.button[0].disabled)

    def test_invalid_environment_configuration_is_reported_without_traceback(self) -> None:
        with patch.dict("os.environ", {"CIPHERLENS_TORCH_THREADS": "invalid"}):
            app = AppTest.from_file("app.py", default_timeout=20).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("configuration is invalid", app.error[0].value)


if __name__ == "__main__":
    unittest.main()

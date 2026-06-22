from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_initial_screen_renders_without_exception(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual([button.label for button in app.button], ["Recognize text"])
        self.assertTrue(app.button[0].disabled)


if __name__ == "__main__":
    unittest.main()

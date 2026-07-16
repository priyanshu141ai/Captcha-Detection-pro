from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from cipherlens.utils import make_torch_generator, seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_repeats_python_numpy_and_torch_sequences(self) -> None:
        first_state = seed_everything(1234)
        first = (random.random(), float(np.random.random()), float(torch.rand(1)))

        second_state = seed_everything(1234)
        second = (random.random(), float(np.random.random()), float(torch.rand(1)))

        self.assertEqual(first, second)
        self.assertEqual(first_state.seed, 1234)
        self.assertEqual(first_state, second_state)
        self.assertFalse(first_state.deterministic_algorithms)

    def test_independent_torch_generators_repeat(self) -> None:
        first = torch.rand(4, generator=make_torch_generator(7))
        second = torch.rand(4, generator=make_torch_generator(7))

        self.assertTrue(torch.equal(first, second))

    def test_seed_validation_rejects_boolean_negative_and_too_large_values(self) -> None:
        for seed in (True, -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                seed_everything(seed)


if __name__ == "__main__":
    unittest.main()

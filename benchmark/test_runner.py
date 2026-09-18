"""Tests for result validation, independent of any measured performance."""
import copy
import unittest

from run import check_samples, dataset, reference


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.ref = reference(4096)
        model = {k: self.ref[k] for k in ("slope", "intercept")}
        self.samples = [{"phase": "setup", "ns": 100},
                        {"phase": "warmup", "ns": 100, **model},
                        {"phase": "fit", "ns": 100, **model}]

    def test_valid_records(self):
        check_samples(self.samples, self.ref, 1, 1)

    def test_missing_trial_rejected(self):
        with self.assertRaises(ValueError):
            check_samples(self.samples[:-1], self.ref, 1, 1)

    def test_invalid_timing_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            samples = copy.deepcopy(self.samples)
            samples[-1]["ns"] = value
            with self.assertRaises(ValueError):
                check_samples(samples, self.ref, 1, 1)

    def test_incorrect_or_nonfinite_model_rejected(self):
        for value in (0.0, float("nan"), float("inf")):
            samples = copy.deepcopy(self.samples)
            samples[-1]["slope"] = value
            with self.assertRaises(ValueError):
                check_samples(samples, self.ref, 1, 1)

    def test_ignoring_noise_is_rejected(self):
        samples = copy.deepcopy(self.samples)
        samples[-1].update(slope=3.0, intercept=2.0)
        with self.assertRaises(ValueError):
            check_samples(samples, self.ref, 1, 1)

    def test_dataset_is_float32_and_periodic(self):
        import numpy as np
        x, y = dataset(8192)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
        np.testing.assert_array_equal(x[:4096], x[4096:])
        np.testing.assert_array_equal(y[:4096], y[4096:])
        self.assertEqual(float(x[0]), -8.0)
        self.assertEqual(float(y[0]), -22.0625)
        self.assertEqual(float(x[1]), -7.93359375)
        self.assertEqual(float(y[1]), -21.8616943359375)


if __name__ == "__main__":
    unittest.main()

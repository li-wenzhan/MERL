import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from merl.libero_states import load_init_states


class UnsupportedState:
    pass


class LiberoStatesTests(unittest.TestCase):
    def test_official_numpy_layout_and_scoped_allowlist(self):
        before = torch.serialization.get_safe_globals().copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pruned_init"
            for dtype in (np.float32, np.float64):
                states = [np.arange(12, dtype=dtype), np.ones(12, dtype=dtype)]
                torch.save(states, path)
                np.testing.assert_array_equal(load_init_states(path), states)
        self.assertCountEqual(torch.serialization.get_safe_globals(), before)

    def test_rejects_malformed_and_unsupported_states(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pruned_init"
            for states in (np.zeros(4), np.zeros((0, 4)), np.full((2, 4), np.nan)):
                torch.save(states, path)
                with self.assertRaises(ValueError):
                    load_init_states(path)
            torch.save(UnsupportedState(), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_init_states(path)


if __name__ == "__main__":
    unittest.main()

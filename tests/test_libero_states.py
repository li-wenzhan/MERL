import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from merl.libero_states import load_init_states


class UnsupportedState:
    pass


class LiberoStatesTests(unittest.TestCase):
    def test_original_numpy_only_zip_version_three_protocol_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.pruned_init"
            states = [np.arange(12, dtype=np.float64), np.full(12, 2, dtype=np.float64)]
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", pickle.dumps(states, protocol=2))
                archive.writestr("archive/version", b"3\n")
            np.testing.assert_array_equal(load_init_states(path), states)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", pickle.dumps(states, protocol=2))
                archive.writestr("archive/version", b"99\n")
            with self.assertRaisesRegex(ValueError, "archive version"):
                load_init_states(path)

    def test_generated_protocol_four_zip_and_rejected_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.pruned_init"
            for states in (np.arange(30, dtype=np.float64).reshape(3, 10), UnsupportedState()):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("archive/data.pkl", pickle.dumps(states, protocol=4))
                    archive.writestr("archive/version", b"1")
                if isinstance(states, np.ndarray):
                    np.testing.assert_array_equal(load_init_states(path), states)
                else:
                    with self.assertRaises(pickle.UnpicklingError):
                        load_init_states(path)

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

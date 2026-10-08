"""Whole-episode validation boundaries for auxiliary physical-data tools."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("h5py"), "physical-data tests require h5py")
class RealWorldSplitTests(unittest.TestCase):
    def build(self, root, mode, **kwargs):
        from real_world.world_model.hdf5_finetune_dataset import RealWorldHDF5WindowDataset
        return RealWorldHDF5WindowDataset(root, mode=mode, num_history=2, num_future=2,
                                         image_size=(4, 4), **kwargs)

    def write_episodes(self, root, count):
        import h5py
        for episode in range(count):
            with h5py.File(Path(root) / f"episode_{episode}.h5", "w") as file:
                file.create_dataset("observations/images/cam_high", data=np.zeros((12, 4, 4, 3), np.uint8))
                file.create_dataset("action", data=np.zeros((12, 7), np.float32))

    def test_overlapping_windows_never_cross_episode_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_episodes(root, 4)
            train, val = self.build(root, "train", seed=7), self.build(root, "val", seed=7)
            train_ids = {w.episode_idx for w in train.windows}
            val_ids = {w.episode_idx for w in val.windows}
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertEqual(train_ids | val_ids, set(range(4)))
            self.assertEqual(len(train.windows) + len(val.windows), 36)
            repeated = self.build(root, "val", seed=7)
            self.assertEqual(repeated.windows, val.windows)

    def test_single_episode_requires_explicit_validation_disable(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_episodes(root, 1)
            with self.assertRaisesRegex(ValueError, "at least two"):
                self.build(root, "train")
            self.assertGreater(len(self.build(root, "train", val_ratio=0)), 0)
            with self.assertRaisesRegex(ValueError, "disabled"):
                self.build(root, "val", val_ratio=0)


if __name__ == "__main__":
    unittest.main()

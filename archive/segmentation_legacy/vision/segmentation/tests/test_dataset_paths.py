import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vision.segmentation.datasets.paths import (
    detect_mask_class_ids,
    pair_image_mask_paths,
)


class PathPairingTests(unittest.TestCase):
    def test_pairs_paths_by_stem_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_directory = root / "images"
            mask_directory = root / "masks"
            image_directory.mkdir()
            mask_directory.mkdir()
            for stem in ("sample_b", "sample_a"):
                Image.new("RGB", (2, 2)).save(image_directory / f"{stem}.jpg")
                Image.new("L", (2, 2)).save(mask_directory / f"{stem}.png")

            images, masks = pair_image_mask_paths(image_directory, mask_directory)

            self.assertEqual([path.stem for path in images], ["sample_a", "sample_b"])
            self.assertEqual([path.stem for path in masks], ["sample_a", "sample_b"])

    def test_rejects_missing_matching_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_directory = root / "images"
            mask_directory = root / "masks"
            image_directory.mkdir()
            mask_directory.mkdir()
            Image.new("RGB", (2, 2)).save(image_directory / "sample.jpg")
            Image.new("L", (2, 2)).save(mask_directory / "other.png")

            with self.assertRaisesRegex(ValueError, "filenames do not match"):
                pair_image_mask_paths(image_directory, mask_directory)

    def test_rejects_duplicate_image_stems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_directory = root / "images"
            mask_directory = root / "masks"
            image_directory.mkdir()
            mask_directory.mkdir()
            Image.new("RGB", (2, 2)).save(image_directory / "sample.jpg")
            Image.new("RGB", (2, 2)).save(image_directory / "sample.png")
            Image.new("L", (2, 2)).save(mask_directory / "sample.png")

            with self.assertRaisesRegex(ValueError, "duplicate image filename stem"):
                pair_image_mask_paths(image_directory, mask_directory)


class MaskClassIdTests(unittest.TestCase):
    def test_detects_contiguous_class_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(first)
            Image.fromarray(np.array([[2, 1]], dtype=np.uint8)).save(second)

            self.assertEqual(detect_mask_class_ids([first, second]), (0, 1, 2))

    def test_rejects_non_contiguous_class_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.png"
            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(mask_path)

            with self.assertRaisesRegex(ValueError, "contiguous"):
                detect_mask_class_ids([mask_path])

    def test_rejects_rgb_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.png"
            Image.new("RGB", (2, 2)).save(mask_path)

            with self.assertRaisesRegex(ValueError, "single-channel"):
                detect_mask_class_ids([mask_path])

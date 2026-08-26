import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import main
from vision.segmentation.config import Config
from vision.segmentation.datasets import oxford_pet


class ConfigTests(unittest.TestCase):
    def test_default_config_file_matches_current_defaults(self):
        config = Config.from_json(Path(__file__).parents[1] / "config.json")
        self.assertEqual(config, Config())

    def test_json_values_override_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump({"batch_size": 4, "image_size": [32, 48], "seed": 7}, file)
            file.flush()
            config = Config.from_json(file.name)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.image_size, (32, 48))
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.to_dict()["image_size"], [32, 48])

    def test_unknown_and_invalid_values_are_rejected(self):
        invalid_values = (
            {"unknown": 1},
            {"batch_size": 0},
            {"num_epochs": 0},
            {"learning_rate": 0},
            {"val_ratio": 1},
            {"seed": -1},
            {"image_size": [30, 32]},
            {"image_size": [32]},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_mapping(values)


class FakeDataset(Dataset[tuple[Tensor, Tensor]]):
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, root, split, image_size, train):
        self.calls.append(
            {"root": root, "split": split, "image_size": image_size, "train": train}
        )
        height, width = image_size
        self.samples = [
            (
                torch.full((3, height, width), float(index)),
                torch.zeros((height, width), dtype=torch.long),
            )
            for index in range(8)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DataLoaderConfigTests(unittest.TestCase):
    def setUp(self):
        FakeDataset.calls = []

    def test_configured_loader_shape_batch_size_and_seed(self):
        with patch.object(oxford_pet, "PetSegmentationDataset", FakeDataset):
            first = oxford_pet.load_dataloaders(
                root="/tmp/pets",
                image_size=(32, 48),
                batch_size=2,
                seed=7,
                val_ratio=0.25,
            )
            second = oxford_pet.load_dataloaders(
                root="/tmp/pets",
                image_size=(32, 48),
                batch_size=2,
                seed=7,
                val_ratio=0.25,
            )

        first_images, first_masks = next(iter(first[0]))
        second_images, second_masks = next(iter(second[0]))
        self.assertEqual(tuple(first_images.shape), (2, 3, 32, 48))
        self.assertEqual(tuple(first_masks.shape), (2, 32, 48))
        torch.testing.assert_close(first_images, second_images)
        torch.testing.assert_close(first_masks, second_masks)
        self.assertEqual(first[1].batch_size, 2)
        self.assertEqual(first[2].batch_size, 2)
        self.assertTrue(all(call["root"] == "/tmp/pets" for call in FakeDataset.calls))

    def test_test_size_alias_and_conflicting_names(self):
        with patch.object(oxford_pet, "PetSegmentationDataset", FakeDataset):
            loaders = oxford_pet.load_dataloaders(
                0.25, root="/tmp/pets", image_size=(32, 32)
            )
        self.assertEqual(len(loaders[1].dataset), 2)
        with self.assertRaisesRegex(ValueError, "either test_size or val_ratio"):
            oxford_pet.load_dataloaders(test_size=0.25, val_ratio=0.25)


class SeedTests(unittest.TestCase):
    def test_seed_repeats_python_numpy_and_torch_randomness(self):
        main.set_seed(123)
        first = (random.random(), float(np.random.rand()), torch.rand(3))
        main.set_seed(123)
        second = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])

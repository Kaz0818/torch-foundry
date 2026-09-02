import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from vision.segmentation.datasets.oxford_pet import load_smoke_dataloader

training = importlib.import_module("vision.segmentation.training.train")


class SmokeDataloaderTests(unittest.TestCase):
    def test_counts_use_each_dataset_and_indices_are_reproducible(self):
        datasets = (range(2944), range(736), range(3669))
        first = load_smoke_dataloader(*datasets, seed=42)
        second = load_smoke_dataloader(*datasets, seed=42)
        self.assertEqual([len(loader.dataset) for loader in first], [294, 73, 183])
        for loader, repeated, dataset in zip(first, second, datasets):
            indices = loader.dataset.indices
            self.assertIsInstance(indices, list)
            self.assertTrue(all(isinstance(index, int) for index in indices))
            self.assertEqual(indices, repeated.dataset.indices)
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(0 <= index < len(dataset) for index in indices))

    def test_small_datasets_keep_one_sample_and_can_be_iterated(self):
        for loader in load_smoke_dataloader(range(1), range(2), range(3)):
            self.assertEqual(len(loader.dataset), 1)
            self.assertEqual(len(next(iter(loader))), 1)

    def test_ratio_one_keeps_all_samples(self):
        loaders = load_smoke_dataloader(
            range(3), range(4), range(5), train_ratio=1, val_ratio=1, test_ratio=1
        )
        self.assertEqual([len(loader.dataset) for loader in loaders], [3, 4, 5])

    def test_empty_datasets_are_rejected(self):
        for index, name in enumerate(("train", "val", "test")):
            with self.subTest(split=name):
                datasets = [range(10), range(10), range(10)]
                datasets[index] = range(0)
                with self.assertRaisesRegex(ValueError, f"{name}_ds must not be empty"):
                    load_smoke_dataloader(*datasets)

    def test_invalid_ratios_are_rejected(self):
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            for ratio in (0, -0.1, 1.1, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, ratio=ratio),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    load_smoke_dataloader(
                        range(10), range(10), range(10), **{name: ratio}
                    )


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.images = torch.randn(4, 3, 4, 4, generator=generator)
        masks = torch.randint(3, (4, 4, 4), generator=generator)
        batches = [(self.images[i : i + 2], masks[i : i + 2]) for i in range(0, 4, 2)]
        self.model = self.make_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.run_training = partial(
            training.train,
            train_loader=batches,
            val_loader=batches,
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=self.optimizer,
            device="cpu",
            num_epochs=2,
            num_classes=3,
        )
        self.quiet = patch.object(training, "tqdm", side_effect=lambda it, **kw: it)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    @staticmethod
    def make_model():
        return nn.Sequential(nn.Conv2d(3, 3, kernel_size=1), nn.BatchNorm2d(3))

    def test_each_epoch_saves_latest_weights_and_full_history(self):
        snapshots = []
        save = training._save_training_artifacts

        def observe_save(model, history, output_dir):
            save(model, history, output_dir)
            output_path = Path(output_dir)
            saved_history = json.loads((output_path / "history.json").read_text())
            self.assertEqual(saved_history, history)
            self.assertTrue(
                all(
                    len(values) == len(snapshots) + 1
                    for values in saved_history.values()
                )
            )
            state = torch.load(
                output_path / "model.pt", map_location="cpu", weights_only=True
            )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(state[key], value, rtol=0, atol=0)
            snapshots.append((saved_history, state))

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "nested" / "run"
            with patch.object(
                training, "_save_training_artifacts", side_effect=observe_save
            ):
                history = self.run_training(output_dir=output_dir)
            self.assertEqual(len(snapshots), 2)
            self.assertEqual([len(h["train_loss"]) for h, _ in snapshots], [1, 2])
            self.assertEqual(snapshots[-1][0], history)
            self.assertFalse(
                torch.equal(snapshots[0][1]["0.weight"], snapshots[1][1]["0.weight"])
            )
            reloaded = self.make_model()
            reloaded.load_state_dict(snapshots[-1][1])
            reloaded.eval()
            with torch.inference_mode():
                torch.testing.assert_close(
                    reloaded(self.images), self.model(self.images), rtol=0, atol=0
                )

    def test_omitting_output_dir_does_not_save(self):
        with patch.object(training, "_save_training_artifacts") as save:
            history = self.run_training()
        save.assert_not_called()
        self.assertEqual(len(history["train_loss"]), 2)

    def test_string_output_dir_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_training(num_epochs=1, output_dir=temporary)
            self.assertTrue((Path(temporary) / "model.pt").is_file())
            self.assertTrue((Path(temporary) / "history.json").is_file())

    def test_metric_logger_receives_one_complete_record_per_epoch(self):
        records = []

        history = self.run_training(metric_logger=records.append)

        self.assertEqual(len(records), 2)
        self.assertEqual([record["epoch"] for record in records], [1, 2])
        for epoch_index, record in enumerate(records):
            self.assertEqual(record["train/loss"], history["train_loss"][epoch_index])
            self.assertEqual(record["val/loss"], history["val_loss"][epoch_index])
            self.assertEqual(record["val/mean_iou"], history["val_iou"][epoch_index])
            self.assertEqual(
                record["val/mean_dice"], history["val_dice"][epoch_index]
            )
            for class_id in range(3):
                self.assertEqual(
                    record[f"val/iou/class_{class_id}"],
                    history["class_ious"][epoch_index][class_id],
                )
                self.assertEqual(
                    record[f"val/dice/class_{class_id}"],
                    history["class_dices"][epoch_index][class_id],
                )

    def test_metric_logger_is_optional(self):
        history = self.run_training(metric_logger=None)
        self.assertEqual(len(history["train_loss"]), 2)

    def test_save_failure_is_not_silenced(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(training.torch, "save", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            self.run_training(output_dir=temporary)


if __name__ == "__main__":
    unittest.main()

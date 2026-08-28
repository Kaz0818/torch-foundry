import unittest

import torch
from torch import nn

from vision.segmentation.config import Config
from vision.segmentation.losses import MulticlassCEDiceLoss, build_loss
from vision.segmentation.training.train import train


class LossFactoryTests(unittest.TestCase):
    def test_cross_entropy_matches_pytorch(self):
        logits = torch.tensor([[[[1.0, -1.0]], [[-1.0, 1.0]], [[0.0, 0.0]]]])
        target = torch.tensor([[[0, 1]]])

        criterion = build_loss(Config(num_classes=3, loss_name="cross_entropy"))

        self.assertIsInstance(criterion, nn.CrossEntropyLoss)
        torch.testing.assert_close(
            criterion(logits, target),
            nn.CrossEntropyLoss()(logits, target),
        )


class MulticlassCEDiceLossTests(unittest.TestCase):
    def test_near_perfect_predictions_have_near_zero_loss(self):
        target = torch.tensor([[[0, 1], [2, 1]]])
        logits = torch.full((1, 3, 2, 2), -10.0)
        logits.scatter_(1, target.unsqueeze(1), 10.0)

        criterion = MulticlassCEDiceLoss()

        self.assertLess(float(criterion(logits, target)), 1e-6)

    def test_incorrect_predictions_have_higher_loss(self):
        target = torch.tensor([[[0, 1], [2, 1]]])
        correct_logits = torch.full((1, 3, 2, 2), -5.0)
        correct_logits.scatter_(1, target.unsqueeze(1), 5.0)
        incorrect_logits = torch.full((1, 3, 2, 2), -5.0)
        incorrect_logits.scatter_(1, ((target + 1) % 3).unsqueeze(1), 5.0)
        criterion = MulticlassCEDiceLoss()

        self.assertGreater(
            float(criterion(incorrect_logits, target)),
            float(criterion(correct_logits, target)),
        )

    def test_excluding_background_focuses_on_foreground_classes(self):
        target = torch.tensor([[[0, 0, 1, 1]]])
        logits = torch.tensor(
            [[[[10.0, 10.0, 10.0, 10.0]], [[-10.0, -10.0, -10.0, -10.0]]]]
        )
        foreground_only = MulticlassCEDiceLoss(
            ce_weight=0,
            dice_weight=1,
            include_background=False,
        )
        including_background = MulticlassCEDiceLoss(
            ce_weight=0,
            dice_weight=1,
            include_background=True,
        )

        self.assertGreater(
            float(foreground_only(logits, target)),
            float(including_background(logits, target)),
        )

    def test_backward_computes_gradients(self):
        logits = torch.randn(2, 7, 4, 4, requires_grad=True)
        target = torch.randint(7, (2, 4, 4))

        loss = MulticlassCEDiceLoss()(logits, target)
        loss.backward()

        gradient = logits.grad
        if gradient is None:
            self.fail("backward did not compute logits.grad")
        self.assertTrue(torch.isfinite(gradient).all())

    def test_seven_class_training_loop_updates_model(self):
        generator = torch.Generator().manual_seed(42)
        images = torch.randn(4, 3, 4, 4, generator=generator)
        masks = torch.randint(7, (4, 4, 4), generator=generator)
        batches = [
            (images[index : index + 2], masks[index : index + 2]) for index in (0, 2)
        ]
        model = nn.Conv2d(3, 7, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_weight = model.weight.detach().clone()

        history = train(
            train_loader=batches,
            val_loader=batches,
            model=model,
            criterion=MulticlassCEDiceLoss(),
            optimizer=optimizer,
            device="cpu",
            num_epochs=1,
            num_classes=7,
        )

        self.assertEqual(len(history["train_loss"]), 1)
        self.assertTrue(torch.isfinite(torch.tensor(history["train_loss"])).all())
        self.assertFalse(torch.equal(initial_weight, model.weight.detach()))


if __name__ == "__main__":
    unittest.main()

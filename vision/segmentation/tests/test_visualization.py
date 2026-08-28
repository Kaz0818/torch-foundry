import unittest

import matplotlib.pyplot as plt
import torch

from vision.segmentation.utils.visualization import plot_overlay


class PlotOverlayTests(unittest.TestCase):
    def test_returns_four_columns_for_at_most_five_images(self):
        images = torch.rand(6, 3, 8, 8)
        masks = torch.zeros(6, 8, 8, dtype=torch.long)
        predictions = torch.ones(6, 8, 8, dtype=torch.long)

        figure = plot_overlay(images, masks, predictions)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 5 * 4)

    def test_uses_all_images_when_batch_has_fewer_than_five(self):
        images = torch.rand(2, 3, 8, 8)
        masks = torch.zeros(2, 8, 8, dtype=torch.long)
        predictions = torch.ones(2, 8, 8, dtype=torch.long)

        figure = plot_overlay(images, masks, predictions)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 2 * 4)

    def test_rejects_empty_images_and_non_positive_limit(self):
        empty_images = torch.empty(0, 3, 8, 8)
        empty_masks = torch.empty(0, 8, 8, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "images must not be empty"):
            plot_overlay(empty_images, empty_masks, empty_masks)

        images = torch.rand(1, 3, 8, 8)
        masks = torch.zeros(1, 8, 8, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "max_images"):
            plot_overlay(images, masks, masks, max_images=0)


if __name__ == "__main__":
    unittest.main()

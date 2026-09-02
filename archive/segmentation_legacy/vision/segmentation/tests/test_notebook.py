import json
import unittest
from pathlib import Path


class WandbNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_contains_the_complete_example(self):
        notebook_path = (
            Path(__file__).parents[3]
            / "notebooks"
            / "segmentation_wandb_example.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for required_text in (
            "git clone https://github.com/Kaz0818/torch-foundry.git",
            'get_secret("WANDB_API_KEY")',
            "WANDB_PROJECT",
            "wandb.init(",
            "metric_logger=run.log",
            "plot_overlay(",
            'wandb.Image(comparison)',
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, code)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_custom_kaggle_notebook_has_separated_responsibilities(self):
        notebook_path = (
            Path(__file__).parents[3]
            / "notebooks"
            / "kaggle_custom_segmentation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        cell_by_id = {cell["id"]: cell for cell in notebook["cells"]}
        for cell_id in (
            "experiment-settings",
            "validate-dataset",
            "create-config-and-loaders",
            "cpu-preflight",
            "prepare-training",
            "run-training",
            "visualize-results",
        ):
            with self.subTest(cell_id=cell_id):
                self.assertIn(cell_id, cell_by_id)
                self.assertEqual(cell_by_id[cell_id]["cell_type"], "code")

        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for required_text in (
            "pair_image_mask_paths(",
            "detect_mask_class_ids(",
            "num_classes = len(class_ids)",
            "Config(\n    num_classes=num_classes,",
            "CPU preflight loss:",
            "build_loss(config)",
            "wandb.init(",
            "plot_overlay(",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, code)

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()

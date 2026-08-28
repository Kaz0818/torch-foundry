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


if __name__ == "__main__":
    unittest.main()

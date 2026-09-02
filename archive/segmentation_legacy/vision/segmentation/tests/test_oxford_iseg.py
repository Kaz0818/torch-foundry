import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vision.segmentation.datasets import oxford_iseg


class OxfordISegPreparationTests(unittest.TestCase):
    def test_partial_image_directory_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_directory = Path(temporary_directory) / "oxford-iseg"
            image_directory = dataset_directory / "images"
            raw_mask_directory = dataset_directory / "images-gt"
            image_directory.mkdir(parents=True)
            raw_mask_directory.mkdir()

            for index in range(150):
                Image.new("RGB", (2, 2), color=index).save(
                    image_directory / f"{index:03d}.jpg"
                )
            for index in range(151):
                Image.new("L", (2, 2), color=0).save(
                    raw_mask_directory / f"{index:03d}.png"
                )

            def add_missing_images(
                url: str,
                output_directory: Path,
                extracted_directory_name: str,
                suffixes: set[str],
            ) -> None:
                self.assertEqual(url, oxford_iseg.OXFORD_ISEG_IMAGE_URL)
                self.assertEqual(output_directory, dataset_directory)
                self.assertEqual(extracted_directory_name, "images")
                self.assertEqual(suffixes, {".bmp", ".jpeg", ".jpg", ".png"})
                Image.new("RGB", (2, 2), color=0).save(
                    output_directory / "images" / "150.jpg"
                )

            with patch.object(
                oxford_iseg,
                "_download_and_extract",
                side_effect=add_missing_images,
            ) as download:
                images, masks = oxford_iseg.prepare_oxford_iseg(
                    temporary_directory
                )

            download.assert_called_once()
            self.assertEqual(len(images), oxford_iseg.OXFORD_ISEG_SAMPLE_COUNT)
            self.assertEqual(len(masks), oxford_iseg.OXFORD_ISEG_SAMPLE_COUNT)
            self.assertEqual(images[-1].name, "150.jpg")
            self.assertEqual(masks[-1].name, "150.png")


if __name__ == "__main__":
    unittest.main()

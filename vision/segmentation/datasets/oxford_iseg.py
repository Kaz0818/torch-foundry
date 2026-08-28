import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive

OXFORD_ISEG_IMAGE_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz"
)
OXFORD_ISEG_MASK_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-gt.tgz"
)
OXFORD_ISEG_SAMPLE_COUNT = 151
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
_MASK_SUFFIXES = {".png"}


def _file_paths(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.name,
    )


def _has_expected_files(directory: Path, suffixes: set[str]) -> bool:
    file_count = len(_file_paths(directory, suffixes))
    if file_count > OXFORD_ISEG_SAMPLE_COUNT:
        raise ValueError(
            f"Oxford iSeg directory contains too many files: {directory} "
            f"({file_count} > {OXFORD_ISEG_SAMPLE_COUNT})"
        )
    return file_count == OXFORD_ISEG_SAMPLE_COUNT


def _files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {directory}")

    files = _file_paths(directory, suffixes)
    if not files:
        raise FileNotFoundError(f"no dataset files found in: {directory}")

    result: dict[str, Path] = {}
    for path in files:
        if path.stem in result:
            raise ValueError(f"duplicate dataset stem: {path.stem}")
        result[path.stem] = path
    return result


def _pair_files(
    image_directory: Path,
    mask_directory: Path,
) -> tuple[list[Path], list[Path]]:
    image_by_stem = _files_by_stem(image_directory, _IMAGE_SUFFIXES)
    mask_by_stem = _files_by_stem(mask_directory, _MASK_SUFFIXES)

    image_stems = set(image_by_stem)
    mask_stems = set(mask_by_stem)
    if image_stems != mask_stems:
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)
        raise ValueError(
            "image/mask stems do not match; "
            f"missing masks: {missing_masks}, missing images: {missing_images}"
        )

    if len(image_by_stem) != OXFORD_ISEG_SAMPLE_COUNT:
        raise ValueError(
            f"Oxford iSeg must contain {OXFORD_ISEG_SAMPLE_COUNT} pairs, "
            f"found {len(image_by_stem)}"
        )

    stems = sorted(image_by_stem)
    return (
        [image_by_stem[stem] for stem in stems],
        [mask_by_stem[stem] for stem in stems],
    )


def _download_and_extract(
    url: str,
    output_directory: Path,
    extracted_directory_name: str,
    suffixes: set[str],
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        print(f"downloading {url}")
        download_and_extract_archive(
            url=url,
            download_root=temporary_path,
            extract_root=temporary_path,
            remove_finished=True,
        )

        extracted_directory = temporary_path / extracted_directory_name
        extracted_files = _files_by_stem(extracted_directory, suffixes)
        if len(extracted_files) != OXFORD_ISEG_SAMPLE_COUNT:
            raise ValueError(
                f"Oxford iSeg archive must contain "
                f"{OXFORD_ISEG_SAMPLE_COUNT} files, "
                f"found {len(extracted_files)} in {url}"
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            extracted_directory,
            output_directory / extracted_directory_name,
            dirs_exist_ok=True,
        )


def _mask_values(mask_path: Path) -> set[int]:
    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask.convert("L"))
    return {int(value) for value in np.unique(mask_array)}


def _validate_pair_sizes(
    images: Iterable[Path],
    masks: Iterable[Path],
) -> None:
    for image_path, mask_path in zip(images, masks):
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"image/mask size mismatch: {image_path.name} "
                    f"{image.size} != {mask_path.name} {mask.size}"
                )


def _convert_mask(mask_path: Path, output_path: Path) -> None:
    values = _mask_values(mask_path)
    unexpected_values = values - {0, 128, 255}
    if unexpected_values:
        raise ValueError(
            f"unsupported mask values in {mask_path}: "
            f"{sorted(unexpected_values)}"
        )

    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask.convert("L"))
    binary_mask = np.where(mask_array == 0, 0, 1).astype(np.uint8)
    Image.fromarray(binary_mask, mode="L").save(output_path)


def _validate_binary_masks(
    images: Iterable[Path],
    masks: Iterable[Path],
) -> None:
    _validate_pair_sizes(images, masks)
    for mask_path in masks:
        values = _mask_values(mask_path)
        unexpected_values = values - {0, 1}
        if unexpected_values:
            raise ValueError(
                f"converted mask contains unsupported values in {mask_path}: "
                f"{sorted(unexpected_values)}"
            )


def _iseg_directory(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.name == "oxford-iseg":
        return root_path
    return root_path / "oxford-iseg"


def prepare_oxford_iseg(
    root: str | Path = "./data",
) -> tuple[list[Path], list[Path]]:
    """Download Oxford iSeg and return paired image and binary-mask paths.

    The official ground-truth masks use values 0, 128, and 255. This helper
    writes a small ``masks`` directory with those values converted to binary
    class IDs 0 and 1, which is the format expected by ``SegmentDataset``.
    """
    dataset_directory = _iseg_directory(root)
    image_directory = dataset_directory / "images"
    raw_mask_directory = dataset_directory / "images-gt"
    mask_directory = dataset_directory / "masks"
    dataset_directory.mkdir(parents=True, exist_ok=True)

    # A caller may already have prepared binary masks. Reuse them without
    # downloading the original archives again.
    has_images = _has_expected_files(image_directory, _IMAGE_SUFFIXES)
    has_masks = _has_expected_files(mask_directory, _MASK_SUFFIXES)
    has_raw_masks = _has_expected_files(raw_mask_directory, _MASK_SUFFIXES)
    if has_images and has_masks and not has_raw_masks:
        images, masks = _pair_files(image_directory, mask_directory)
        _validate_binary_masks(images, masks)
        return images, masks

    if not has_images:
        _download_and_extract(
            OXFORD_ISEG_IMAGE_URL,
            dataset_directory,
            "images",
            _IMAGE_SUFFIXES,
        )
    if not has_raw_masks:
        _download_and_extract(
            OXFORD_ISEG_MASK_URL,
            dataset_directory,
            "images-gt",
            _MASK_SUFFIXES,
        )

    images, raw_masks = _pair_files(image_directory, raw_mask_directory)
    _validate_pair_sizes(images, raw_masks)
    mask_directory.mkdir(parents=True, exist_ok=True)

    for image_path, raw_mask_path in zip(images, raw_masks):
        output_path = mask_directory / f"{raw_mask_path.stem}.png"
        if output_path.exists():
            try:
                _validate_binary_masks([image_path], [output_path])
                continue
            except (OSError, ValueError):
                pass
        _convert_mask(raw_mask_path, output_path)

    converted_images, converted_masks = _pair_files(
        image_directory,
        mask_directory,
    )
    _validate_binary_masks(converted_images, converted_masks)
    return converted_images, converted_masks

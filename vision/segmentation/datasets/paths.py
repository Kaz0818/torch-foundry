from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png"})
DEFAULT_MASK_SUFFIXES = frozenset({".png"})


def pair_image_mask_paths(
    image_directory: str | Path,
    mask_directory: str | Path,
    *,
    image_suffixes: frozenset[str] = DEFAULT_IMAGE_SUFFIXES,
    mask_suffixes: frozenset[str] = DEFAULT_MASK_SUFFIXES,
) -> tuple[list[Path], list[Path]]:
    """Return image and mask paths paired by their filenames without suffixes."""
    images = _paths_by_stem(Path(image_directory), image_suffixes, "image")
    masks = _paths_by_stem(Path(mask_directory), mask_suffixes, "mask")

    image_stems = set(images)
    mask_stems = set(masks)
    if image_stems != mask_stems:
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)
        raise ValueError(
            "image and mask filenames do not match; "
            f"missing masks: {missing_masks}; missing images: {missing_images}"
        )

    stems = sorted(image_stems)
    return [images[stem] for stem in stems], [masks[stem] for stem in stems]


def detect_mask_class_ids(mask_paths: Iterable[str | Path]) -> tuple[int, ...]:
    """Read every mask and return its contiguous integer class IDs.

    Segmentation masks must be a single-channel image with class IDs beginning
    at 0. The returned IDs can safely determine ``num_classes`` as ``len(ids)``.
    """
    class_ids: set[int] = set()
    mask_count = 0

    for raw_path in mask_paths:
        mask_path = Path(raw_path)
        with Image.open(mask_path) as mask:
            values = np.asarray(mask)
        if values.ndim != 2:
            raise ValueError(
                f"mask must be single-channel class IDs: {mask_path} "
                f"has shape {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"mask must use integer class IDs: {mask_path}")

        class_ids.update(int(value) for value in np.unique(values))
        mask_count += 1

    if mask_count == 0:
        raise ValueError("mask_paths must not be empty")

    detected_ids = tuple(sorted(class_ids))
    expected_ids = tuple(range(detected_ids[-1] + 1))
    if detected_ids != expected_ids:
        raise ValueError(
            "mask class IDs must be contiguous and start at 0; "
            f"found {list(detected_ids)}"
        )
    return detected_ids


def _paths_by_stem(
    directory: Path,
    suffixes: frozenset[str],
    label: str,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(
                f"duplicate {label} filename stem '{path.stem}' in {directory}"
            )
        result[path.stem] = path

    if not result:
        suffix_list = ", ".join(sorted(suffixes))
        raise FileNotFoundError(
            f"no {label} files with suffixes {suffix_list} in {directory}"
        )
    return result

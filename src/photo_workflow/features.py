from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageStat

FEATURE_NAMES = (
    "blur_global",
    "blur_small",
    "blur_center",
    "blur_center_wide",
    "blur_top",
    "blur_bottom",
    "grid_median",
    "grid_max",
    "grid_top_quartile",
    "grid_low_half",
    "mean_luminance",
    "luminance_stddev",
    "clipped_dark_ratio",
    "clipped_light_ratio",
    "entropy",
)
MODEL_ANALYSIS_SIZE = 512


@dataclass(frozen=True)
class QualityFeatures:
    values: tuple[float, ...]
    blur_score: float
    mean_luminance: float
    clipped_dark_ratio: float
    clipped_light_ratio: float


def extract_quality_features(
    image: Image.Image,
    analysis_size: int = MODEL_ANALYSIS_SIZE,
) -> QualityFeatures:
    gray = ImageOps.exif_transpose(image).convert("L")
    gray.thumbnail((analysis_size, analysis_size))
    width, height = gray.size

    global_blur = _laplacian_variance(gray)
    small = gray.copy()
    small.thumbnail((256, 256))
    small_blur = _laplacian_variance(small)

    center = _center_crop(gray, 0.5)
    center_wide = _center_crop(gray, 0.65)
    top = gray.crop((0, 0, width, max(1, height // 2)))
    bottom = gray.crop((0, height // 2, width, height))

    grid_scores = sorted(
        _laplacian_variance(
            gray.crop(
                (
                    column * width // 4,
                    row * height // 4,
                    (column + 1) * width // 4,
                    (row + 1) * height // 4,
                )
            )
        )
        for row in range(4)
        for column in range(4)
    )

    pixels = list(gray.get_flattened_data())
    pixel_count = len(pixels)
    mean_luminance = sum(pixels) / pixel_count
    clipped_dark_ratio = sum(value <= 10 for value in pixels) / pixel_count
    clipped_light_ratio = sum(value >= 245 for value in pixels) / pixel_count
    luminance_stddev = ImageStat.Stat(gray).stddev[0]
    entropy = gray.entropy()

    values = (
        math.log1p(global_blur),
        math.log1p(small_blur),
        math.log1p(_laplacian_variance(center)),
        math.log1p(_laplacian_variance(center_wide)),
        math.log1p(_laplacian_variance(top)),
        math.log1p(_laplacian_variance(bottom)),
        math.log1p(grid_scores[len(grid_scores) // 2]),
        math.log1p(grid_scores[-1]),
        math.log1p(sum(grid_scores[-4:]) / 4),
        math.log1p(sum(grid_scores[:8]) / 8),
        mean_luminance / 255,
        luminance_stddev / 128,
        clipped_dark_ratio,
        clipped_light_ratio,
        entropy / 8,
    )
    return QualityFeatures(
        values=values,
        blur_score=global_blur,
        mean_luminance=mean_luminance,
        clipped_dark_ratio=clipped_dark_ratio,
        clipped_light_ratio=clipped_light_ratio,
    )


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(3, int(width * fraction))
    crop_height = max(3, int(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _laplacian_variance(image: Image.Image) -> float:
    width, height = image.size
    if width < 3 or height < 3:
        return 0.0

    pixels = list(image.get_flattened_data())
    count = (width - 2) * (height - 2)
    total = 0.0
    total_squared = 0.0
    for row in range(1, height - 1):
        offset = row * width
        for column in range(1, width - 1):
            response = (
                4 * pixels[offset + column]
                - pixels[offset + column - 1]
                - pixels[offset + column + 1]
                - pixels[offset - width + column]
                - pixels[offset + width + column]
            )
            total += response
            total_squared += response * response

    mean = total / count
    return total_squared / count - mean * mean

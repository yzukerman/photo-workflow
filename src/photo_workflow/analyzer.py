from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from photo_workflow.features import MODEL_ANALYSIS_SIZE, extract_quality_features
from photo_workflow.formats import register_image_openers
from photo_workflow.model import QualityModel, load_default_quality_model

register_image_openers()
DEFAULT_QUALITY_MODEL = load_default_quality_model()


@dataclass(frozen=True)
class QualityThresholds:
    blur_threshold: float = 75.0
    min_mean: float = 35.0
    max_mean: float = 220.0
    max_clipped_dark_ratio: float = 0.35
    max_clipped_light_ratio: float = 0.35
    analysis_size: int = MODEL_ANALYSIS_SIZE


@dataclass(frozen=True)
class ImageAssessment:
    path: Path
    passed: bool
    blur_score: float
    mean_luminance: float
    clipped_dark_ratio: float
    clipped_light_ratio: float
    reasons: tuple[str, ...]


def assess_image(
    path: Path,
    thresholds: QualityThresholds | None = None,
    quality_model: QualityModel | None = DEFAULT_QUALITY_MODEL,
) -> ImageAssessment:
    thresholds = thresholds or QualityThresholds()
    with Image.open(path) as image:
        features = extract_quality_features(image, thresholds.analysis_size)

    reasons: list[str] = []
    if features.blur_score < thresholds.blur_threshold:
        reasons.append("blurry")
    if quality_model is not None and not quality_model.predict(features).passed:
        reasons.append("quality_model")
    if (
        features.mean_luminance < thresholds.min_mean
        or features.clipped_dark_ratio > thresholds.max_clipped_dark_ratio
    ):
        reasons.append("too_dark")
    if (
        features.mean_luminance > thresholds.max_mean
        or features.clipped_light_ratio > thresholds.max_clipped_light_ratio
    ):
        reasons.append("too_light")

    return ImageAssessment(
        path=path,
        passed=not reasons,
        blur_score=features.blur_score,
        mean_luminance=features.mean_luminance,
        clipped_dark_ratio=features.clipped_dark_ratio,
        clipped_light_ratio=features.clipped_light_ratio,
        reasons=tuple(reasons),
    )

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from PIL import Image

from photo_workflow.features import (
    FEATURE_NAMES,
    MODEL_ANALYSIS_SIZE,
    QualityFeatures,
    extract_quality_features,
)


@dataclass(frozen=True)
class ModelPrediction:
    passed: bool
    confidence: float


@dataclass(frozen=True)
class QualityModel:
    pass_samples: tuple[tuple[float, ...], ...]
    fail_samples: tuple[tuple[float, ...], ...]
    scales: tuple[float, ...]
    fail_confidence_threshold: float = 0.15

    def predict(self, features: QualityFeatures) -> ModelPrediction:
        pass_distance = min(self._distance(features.values, sample) for sample in self.pass_samples)
        fail_distance = min(self._distance(features.values, sample) for sample in self.fail_samples)
        total = pass_distance + fail_distance
        confidence = abs(pass_distance - fail_distance) / total if total else 1.0
        passed = pass_distance <= fail_distance or confidence < self.fail_confidence_threshold
        return ModelPrediction(passed=passed, confidence=confidence)

    def _distance(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum(
            ((left_value - right_value) / scale) ** 2
            for left_value, right_value, scale in zip(left, right, self.scales, strict=True)
        ) ** 0.5

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "feature_names": FEATURE_NAMES,
            "analysis_size": MODEL_ANALYSIS_SIZE,
            "pass_samples": self.pass_samples,
            "fail_samples": self.fail_samples,
            "scales": self.scales,
            "fail_confidence_threshold": self.fail_confidence_threshold,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> QualityModel:
        payload = json.loads(path.read_text())
        if (
            payload["version"] != 1
            or tuple(payload["feature_names"]) != FEATURE_NAMES
            or payload.get("analysis_size") != MODEL_ANALYSIS_SIZE
        ):
            raise ValueError("unsupported quality model format")
        return cls(
            pass_samples=tuple(tuple(sample) for sample in payload["pass_samples"]),
            fail_samples=tuple(tuple(sample) for sample in payload["fail_samples"]),
            scales=tuple(payload["scales"]),
            fail_confidence_threshold=payload.get("fail_confidence_threshold", 0.15),
        )


def train_quality_model(calibration_folder: Path) -> QualityModel:
    pass_samples = _extract_folder(calibration_folder / "pass")
    pass_samples += _extract_folder(calibration_folder / "false-negatives")
    fail_samples = _extract_folder(calibration_folder / "fail")
    fail_samples += _extract_folder(calibration_folder / "false-positives")
    for test_folder in sorted(calibration_folder.glob("Test *")):
        test_pass_samples, test_fail_samples = _extract_test_run_folder(test_folder)
        pass_samples += test_pass_samples
        fail_samples += test_fail_samples
    if not pass_samples or not fail_samples:
        raise ValueError("calibration folder must contain non-empty pass and fail subfolders")

    columns = list(zip(*(pass_samples + fail_samples), strict=True))
    scales = tuple(max(statistics.pstdev(column), 0.05) for column in columns)
    return QualityModel(tuple(pass_samples), tuple(fail_samples), scales)


def _extract_test_run_folder(folder: Path) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    pass_samples: list[tuple[float, ...]] = []
    fail_samples: list[tuple[float, ...]] = []
    if not folder.exists():
        return pass_samples, fail_samples

    for path in sorted(folder.rglob("*")):
        if not _is_supported_image(path):
            continue
        try:
            with Image.open(path) as image:
                features = extract_quality_features(image).values
        except OSError:
            continue
        if "-bad" in path.stem.lower():
            fail_samples.append(features)
        else:
            pass_samples.append(features)
    return pass_samples, fail_samples


def load_default_quality_model() -> QualityModel | None:
    path = files("photo_workflow").joinpath("default_quality_model.json")
    if not path.is_file():
        return None
    with path.open() as model_file:
        payload = json.load(model_file)
    if (
        payload["version"] != 1
        or tuple(payload["feature_names"]) != FEATURE_NAMES
        or payload.get("analysis_size") != MODEL_ANALYSIS_SIZE
    ):
        raise ValueError("unsupported default quality model format")
    return QualityModel(
        pass_samples=tuple(tuple(sample) for sample in payload["pass_samples"]),
        fail_samples=tuple(tuple(sample) for sample in payload["fail_samples"]),
        scales=tuple(payload["scales"]),
        fail_confidence_threshold=payload.get("fail_confidence_threshold", 0.15),
    )


def _extract_folder(folder: Path) -> list[tuple[float, ...]]:
    samples: list[tuple[float, ...]] = []
    if not folder.exists():
        return samples
    for path in sorted(folder.iterdir()):
        if not _is_supported_image(path):
            continue
        try:
            with Image.open(path) as image:
                samples.append(extract_quality_features(image).values)
        except OSError:
            continue
    return samples


def _is_supported_image(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".tif", ".tiff"}
    )

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

    def predict(self, features: QualityFeatures) -> ModelPrediction:
        pass_distance = min(self._distance(features.values, sample) for sample in self.pass_samples)
        fail_distance = min(self._distance(features.values, sample) for sample in self.fail_samples)
        total = pass_distance + fail_distance
        confidence = abs(pass_distance - fail_distance) / total if total else 1.0
        return ModelPrediction(passed=pass_distance <= fail_distance, confidence=confidence)

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
        )


def train_quality_model(calibration_folder: Path) -> QualityModel:
    pass_samples = _extract_folder(calibration_folder / "pass")
    fail_samples = _extract_folder(calibration_folder / "fail")
    if not pass_samples or not fail_samples:
        raise ValueError("calibration folder must contain non-empty pass and fail subfolders")

    columns = list(zip(*(pass_samples + fail_samples), strict=True))
    scales = tuple(max(statistics.pstdev(column), 0.05) for column in columns)
    return QualityModel(tuple(pass_samples), tuple(fail_samples), scales)


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
    )


def _extract_folder(folder: Path) -> list[tuple[float, ...]]:
    samples: list[tuple[float, ...]] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            with Image.open(path) as image:
                samples.append(extract_quality_features(image).values)
        except OSError:
            continue
    return samples

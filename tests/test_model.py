from pathlib import Path
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFilter

from photo_workflow.analyzer import assess_image
from photo_workflow.cli import sort_images
from photo_workflow.model import QualityModel, train_quality_model


def save_checkerboard(path: Path, size: int = 128) -> None:
    image = Image.new("RGB", (size, size), (210, 210, 210))
    draw = ImageDraw.Draw(image)
    tile = 8
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(40, 40, 40))
    image.save(path)


def test_train_quality_model_can_be_saved_and_loaded(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration"
    pass_dir = calibration / "pass"
    fail_dir = calibration / "fail"
    pass_dir.mkdir(parents=True)
    fail_dir.mkdir()
    save_checkerboard(pass_dir / "sharp.jpeg")
    with Image.open(pass_dir / "sharp.jpeg") as image:
        image.filter(ImageFilter.GaussianBlur(radius=6)).save(fail_dir / "blurred.jpeg")

    model = train_quality_model(calibration)
    model_path = tmp_path / "model.json"
    model.save(model_path)
    loaded_model = QualityModel.load(model_path)

    assert len(loaded_model.pass_samples) == 1
    assert len(loaded_model.fail_samples) == 1


def test_default_quality_model_classifies_labeled_sample_images() -> None:
    sample_root = Path("Sample images")
    if not sample_root.exists():
        pytest.skip("labeled sample images are not available")

    errors = []
    for expected_label in ("pass", "fail"):
        for image_path in sorted((sample_root / expected_label).iterdir()):
            if not image_path.is_file() or image_path.name.startswith("."):
                continue
            assessment = assess_image(image_path)
            predicted_label = "pass" if assessment.passed else "fail"
            if predicted_label != expected_label:
                errors.append((image_path.name, expected_label, predicted_label, assessment.reasons))

    assert errors == []


def test_cli_sorting_matches_labeled_sample_images(tmp_path: Path) -> None:
    sample_root = Path("Sample images")
    if not sample_root.exists():
        pytest.skip("labeled sample images are not available")

    input_folder = tmp_path / "input"
    input_folder.mkdir()
    expected_counts = {}
    for expected_label in ("pass", "fail"):
        labeled_images = [
            image_path
            for image_path in sorted((sample_root / expected_label).iterdir())
            if image_path.is_file() and not image_path.name.startswith(".")
        ]
        expected_counts[expected_label] = len(labeled_images)
        for image_path in labeled_images:
            shutil.copy(image_path, input_folder / image_path.name)

    sort_images(input_folder)

    assert len(list((input_folder / "pass").iterdir())) == expected_counts["pass"]
    assert len(list((input_folder / "fail").iterdir())) == expected_counts["fail"]

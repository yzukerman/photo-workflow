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
    sample_groups = (("pass", "pass"), ("false-negatives", "pass"), ("fail", "fail"))
    for folder_name, expected_label in sample_groups:
        sample_folder = sample_root / folder_name
        if not sample_folder.exists():
            continue
        for image_path in sorted(sample_folder.iterdir()):
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
    sample_groups = (("pass", "pass"), ("false-negatives", "pass"), ("fail", "fail"))
    for folder_name, expected_label in sample_groups:
        sample_folder = sample_root / folder_name
        if not sample_folder.exists():
            continue
        labeled_images = [
            image_path
            for image_path in sorted(sample_folder.iterdir())
            if image_path.is_file() and not image_path.name.startswith(".")
        ]
        expected_counts[expected_label] = expected_counts.get(expected_label, 0) + len(labeled_images)
        for image_path in labeled_images:
            shutil.copy(image_path, input_folder / image_path.name)

    sort_images(input_folder)

    assert _count_image_files(input_folder / "pass") == expected_counts["pass"]
    assert _count_image_files(input_folder / "fail") == expected_counts["fail"]
    assert (input_folder / "fail" / "failure-report.md").exists()


def test_default_quality_model_rejects_test_run_bad_images() -> None:
    test_run = Path("Sample images/Test 3")
    if not test_run.exists():
        pytest.skip("Test 3 sample images are not available")

    bad_images = sorted(path for path in test_run.rglob("*-bad.*") if path.is_file())
    errors = [image_path.name for image_path in bad_images if assess_image(image_path).passed]

    assert bad_images
    assert errors == []


def _count_image_files(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".heic", ".heif", ".hif", ".png", ".webp", ".tif", ".tiff"}
    )

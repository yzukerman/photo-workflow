from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from photo_workflow.analyzer import QualityThresholds, assess_image


def save_checkerboard(path: Path, size: int = 128) -> None:
    image = Image.new("RGB", (size, size), (210, 210, 210))
    draw = ImageDraw.Draw(image)
    tile = 8
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(40, 40, 40))
    image.save(path)


def test_assess_image_passes_sharp_balanced_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sharp.png"
    save_checkerboard(image_path)

    result = assess_image(image_path, quality_model=None)

    assert result.passed
    assert result.reasons == ()
    assert result.blur_score >= QualityThresholds.blur_threshold


def test_assess_image_fails_blurry_image(tmp_path: Path) -> None:
    image_path = tmp_path / "blurry.png"
    save_checkerboard(image_path)
    with Image.open(image_path) as image:
        image.filter(ImageFilter.GaussianBlur(radius=6)).save(image_path)

    result = assess_image(image_path)

    assert not result.passed
    assert "blurry" in result.reasons


def test_assess_image_fails_underexposed_image(tmp_path: Path) -> None:
    image_path = tmp_path / "dark.png"
    Image.new("RGB", (128, 128), (5, 5, 5)).save(image_path)

    result = assess_image(image_path)

    assert not result.passed
    assert "too_dark" in result.reasons


def test_assess_image_fails_overexposed_image(tmp_path: Path) -> None:
    image_path = tmp_path / "light.png"
    Image.new("RGB", (128, 128), (252, 252, 252)).save(image_path)

    result = assess_image(image_path)

    assert not result.passed
    assert "too_light" in result.reasons

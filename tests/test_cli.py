from pathlib import Path

from PIL import Image, ImageDraw

from photo_workflow.cli import IMAGE_EXTENSIONS, main, sort_images


def save_checkerboard(path: Path, size: int = 128) -> None:
    image = Image.new("RGB", (size, size), (210, 210, 210))
    draw = ImageDraw.Draw(image)
    tile = 8
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(40, 40, 40))
    image.save(path)


def test_sort_images_moves_pass_and_fail_images(tmp_path: Path) -> None:
    pass_image = tmp_path / "good.png"
    fail_image = tmp_path / "bad.png"
    save_checkerboard(pass_image)
    Image.new("RGB", (128, 128), (255, 255, 255)).save(fail_image)

    results = sort_images(tmp_path)

    assert len(results) == 2
    assert (tmp_path / "pass" / "good.png").exists()
    assert (tmp_path / "fail" / "bad.png").exists()
    assert not pass_image.exists()
    assert not fail_image.exists()


def test_supported_input_extensions_include_jpeg_and_heic() -> None:
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".heic" in IMAGE_EXTENSIONS
    assert ".heif" in IMAGE_EXTENSIONS


def test_sort_images_processes_jpeg_and_heic_inputs(tmp_path: Path) -> None:
    jpeg_image = tmp_path / "good.jpeg"
    heic_image = tmp_path / "good.heic"
    save_checkerboard(jpeg_image)
    save_checkerboard(heic_image)

    results = sort_images(tmp_path)

    assert len(results) == 2
    assert (tmp_path / "pass" / "good.jpeg").exists()
    assert (tmp_path / "pass" / "good.heic").exists()


def test_sort_images_dry_run_does_not_move_files(tmp_path: Path) -> None:
    image_path = tmp_path / "good.png"
    save_checkerboard(image_path)

    results = sort_images(tmp_path, dry_run=True)

    assert len(results) == 1
    assert image_path.exists()
    assert not (tmp_path / "pass").exists()
    assert not (tmp_path / "fail").exists()


def test_sort_images_preserves_existing_destination_file(tmp_path: Path) -> None:
    pass_dir = tmp_path / "pass"
    pass_dir.mkdir()
    existing_image = pass_dir / "good.png"
    save_checkerboard(existing_image)
    incoming_image = tmp_path / "good.png"
    save_checkerboard(incoming_image)

    sort_images(tmp_path)

    assert existing_image.exists()
    assert (pass_dir / "good_1.png").exists()


def test_sort_images_moves_unreadable_image_to_fail(tmp_path: Path) -> None:
    broken_image = tmp_path / "broken.jpg"
    broken_image.write_text("not an image")

    results = sort_images(tmp_path)

    assert len(results) == 1
    assert results[0].reasons == ("unreadable",)
    assert (tmp_path / "fail" / "broken.jpg").exists()
    assert not broken_image.exists()


def test_cli_returns_success_and_writes_json(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "good.png"
    save_checkerboard(image_path)

    exit_code = main([str(tmp_path), "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"total": 1' in output
    assert image_path.exists()

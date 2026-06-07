# Photo Workflow

`photo-workflow` is a portable Python CLI that sorts images in a folder into `pass` and `fail` subfolders using local-only image quality analysis.

## What It Checks

- **Blur/focus**: calibrated local quality model using multi-scale Laplacian focus features.
- **Exposure**: average luminance and clipped shadow/highlight ratios.
- **Local execution**: no network calls, cloud APIs, telemetry, or remote model downloads.
- **Supported formats**: JPEG/JPG, HEIC/HEIF, PNG, TIFF, WebP, GIF, and BMP.
- **Lightweight model**: tiny local k-nearest-neighbor quality model calibrated from labeled examples, implemented with Pillow, pillow-heif, and Python standard library code.

## Usage

```bash
uv run photo-workflow /path/to/images
```

Useful options:

```bash
uv run photo-workflow /path/to/images --dry-run
uv run photo-workflow /path/to/images --pass-dir accepted --fail-dir rejected
uv run photo-workflow /path/to/images --blur-threshold 90 --min-mean 45 --max-mean 215
uv run photo-workflow /path/to/images --calibration-folder "Sample images" --save-model quality-model.json
```

By default, passing JPEG and HEIC images are moved into `pass/`; failing images are moved into `fail/` inside the input folder.

Calibration uses `pass/` and `false-negatives/` as accepted examples, plus `fail/` and `false-positives/` as rejected examples. In `Test *` folders, filenames containing `-bad` are rejected examples and the remaining images are accepted examples.

## Development

Run tests through uv:

```bash
UV_CACHE_DIR=.uv-cache uv run --with pytest pytest -q
```

The application is designed to be portable across macOS, Linux, and Windows by using `pathlib`, `shutil`, and no shell-specific runtime behavior.

## Success Criteria

- Accepts an image folder path from the command line.
- Analyzes local JPEG/JPG and HEIC/HEIF files without remote services.
- Detects blur, overexposure, and underexposure.
- Moves each analyzed image to a pass or fail subfolder.
- Treats unreadable image files as failures instead of silently dropping them.
- Supports dry-run mode for safe inspection.
- Uses Python packaging and execution through uv.
- Has automated tests for analyzer behavior and CLI file movement.
- Has automated checks proving no heavy ML or network/cloud runtime dependencies are used.
- Correctly classifies the labeled `Sample images/pass`, `Sample images/fail`, and `Sample images/false-negatives` examples.

## Review Questions

- Should scanning include nested folders or only direct children of the input folder?
- Should RAW camera formats be supported, or are Pillow-supported formats enough?
- Should the default thresholds be conservative and reject borderline photos, or permissive and keep borderline photos?
- Should failed images be grouped by reason, such as `fail/blurry`, `fail/too_dark`, and `fail/too_light`?

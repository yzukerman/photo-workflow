# Photo Workflow

`photo-workflow` is a portable Python CLI that sorts images in a folder into `pass` and `fail` subfolders using local-only image quality analysis. The tool was created using OpenAI Codex 5.5 medium via the CLI.

## What It Checks

- **Blur/focus**: calibrated local quality model using multi-scale Laplacian focus features.
- **Exposure**: average luminance and clipped shadow/highlight ratios.
- **Local execution**: no network calls, cloud APIs, telemetry, or remote model downloads.
- **Supported formats**: JPEG/JPG, HEIC/HEIF/HIF, PNG, TIFF, WebP, GIF, and BMP.
- **Unsupported formats**: RAW camera files are not supported.
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

By default, passing JPEG and HEIC/HEIF/HIF images are moved into `pass/`; failing images are moved into `fail/` inside the input folder.

Calibration uses `pass/` and `false-negatives/` as accepted examples, plus `fail/` and `false-positives/` as rejected examples. In `Test *` folders, filenames containing `-bad` are rejected examples and the remaining images are accepted examples.

The CLI scans nested folders recursively, flattens accepted/rejected images into the configured `pass/` and `fail/` folders, does not group failures by reason, and skips existing output folders on repeated runs. Progress is written to stderr and can be disabled with `--no-progress`.

Each run writes `failure-report.md` in the fail folder. The report lists every rejected image, its original relative path, rejection reasons, and the measured blur/exposure statistics.

## Development

Run tests through uv:

```bash
UV_CACHE_DIR=.uv-cache uv run --with pytest pytest -q
```

The application is designed to be portable across macOS, Linux, and Windows by using `pathlib`, `shutil`, and no shell-specific runtime behavior.

## Success Criteria

- Accepts an image folder path from the command line.
- Scans nested folders recursively.
- Analyzes local JPEG/JPG and HEIC/HEIF/HIF files without remote services.
- Does not support RAW camera files.
- Detects blur, overexposure, and underexposure.
- Moves each analyzed image to a pass or fail subfolder.
- Writes `failure-report.md` in the fail folder with per-image failure reasons.
- Shows command-line progress during analysis.
- Treats unreadable image files as failures instead of silently dropping them.
- Supports dry-run mode for safe inspection.
- Uses Python packaging and execution through uv.
- Has automated tests for analyzer behavior and CLI file movement.
- Has automated checks proving no heavy ML or network/cloud runtime dependencies are used.
- Correctly classifies the labeled `Sample images/pass`, `Sample images/fail`, and `Sample images/false-negatives` examples.

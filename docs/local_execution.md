# Local Execution Evidence

This project uses a local calibrated image-quality model:

- Runtime dependencies are limited to Pillow, pillow-heif, and Python standard library modules.
- The source code contains no `requests`, `urllib`, cloud SDK, subprocess network calls, telemetry, or model-download logic.
- The analyzer loads only a small JSON quality profile packaged with the app, or a local profile generated from labeled examples.
- All image processing happens in process on files from the user-provided folder.

The model intentionally favors efficiency and portability over large neural network inference. It computes multi-scale grayscale focus and exposure features after downscaling images to a bounded size, then applies a local k-nearest-neighbor quality profile.

Calibration treats `false-negatives/` as additional pass examples. Low-confidence model decisions are pass-biased so uncertain images are retained for review instead of discarded.

Calibration can also consume `false-positives/` and `Test *` folders. Within test-run folders, files containing `-bad` in their names are treated as rejected examples.

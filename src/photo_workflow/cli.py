from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from photo_workflow.analyzer import (
    DEFAULT_QUALITY_MODEL,
    ImageAssessment,
    QualityThresholds,
    assess_image,
)
from photo_workflow.model import QualityModel, train_quality_model

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".hif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source = args.folder.resolve()
    if not source.is_dir():
        parser.error(f"input folder does not exist: {source}")

    thresholds = QualityThresholds(
        blur_threshold=args.blur_threshold,
        min_mean=args.min_mean,
        max_mean=args.max_mean,
        max_clipped_dark_ratio=args.max_clipped_dark_ratio,
        max_clipped_light_ratio=args.max_clipped_light_ratio,
    )
    quality_model = _load_quality_model(args)

    results = sort_images(
        source=source,
        pass_dir_name=args.pass_dir,
        fail_dir_name=args.fail_dir,
        thresholds=thresholds,
        quality_model=quality_model,
        dry_run=args.dry_run,
        show_progress=not args.no_progress,
    )
    print(_format_results(results, args.dry_run))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-workflow",
        description="Sort local images into pass/fail folders by blur and exposure quality.",
    )
    parser.add_argument("folder", type=Path, help="Folder containing images to analyze.")
    parser.add_argument("--pass-dir", default="pass", help="Subfolder for passing images.")
    parser.add_argument("--fail-dir", default="fail", help="Subfolder for failing images.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without moving files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output.")
    parser.add_argument(
        "--calibration-folder",
        type=Path,
        help="Folder containing labeled pass/ and fail/ subfolders.",
    )
    parser.add_argument("--save-model", type=Path, help="Save the calibrated local quality model.")
    parser.add_argument("--blur-threshold", type=float, default=QualityThresholds.blur_threshold)
    parser.add_argument("--min-mean", type=float, default=QualityThresholds.min_mean)
    parser.add_argument("--max-mean", type=float, default=QualityThresholds.max_mean)
    parser.add_argument(
        "--max-clipped-dark-ratio",
        type=float,
        default=QualityThresholds.max_clipped_dark_ratio,
    )
    parser.add_argument(
        "--max-clipped-light-ratio",
        type=float,
        default=QualityThresholds.max_clipped_light_ratio,
    )
    return parser


def sort_images(
    source: Path,
    pass_dir_name: str = "pass",
    fail_dir_name: str = "fail",
    thresholds: QualityThresholds | None = None,
    quality_model: QualityModel | None = DEFAULT_QUALITY_MODEL,
    dry_run: bool = False,
    show_progress: bool = False,
) -> list[ImageAssessment]:
    thresholds = thresholds or QualityThresholds()
    pass_dir = source / pass_dir_name
    fail_dir = source / fail_dir_name
    results: list[ImageAssessment] = []

    candidates = _find_image_candidates(source, pass_dir_name, fail_dir_name)

    if not dry_run:
        pass_dir.mkdir(exist_ok=True)
        fail_dir.mkdir(exist_ok=True)

    for index, path in enumerate(candidates, start=1):
        if show_progress:
            print(
                f"Processing {index}/{len(candidates)}: {path.relative_to(source)}",
                file=sys.stderr,
            )
        try:
            assessment = assess_image(path, thresholds, quality_model)
        except OSError:
            assessment = ImageAssessment(
                path=path,
                passed=False,
                blur_score=0.0,
                mean_luminance=0.0,
                clipped_dark_ratio=0.0,
                clipped_light_ratio=0.0,
                reasons=("unreadable",),
            )

        destination_dir = pass_dir if assessment.passed else fail_dir
        if not dry_run:
            shutil.move(str(path), _available_destination(destination_dir, path.name))
        results.append(assessment)

    if not dry_run:
        _write_fail_report(source, fail_dir, results)

    return results


def _find_image_candidates(source: Path, pass_dir_name: str, fail_dir_name: str) -> list[Path]:
    output_dir_names = {pass_dir_name, fail_dir_name}
    candidates = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = path.relative_to(source).parts
        if any(part in output_dir_names for part in relative_parts[:-1]):
            continue
        candidates.append(path)
    return candidates


def _load_quality_model(args: argparse.Namespace) -> QualityModel | None:
    if args.calibration_folder is None:
        from photo_workflow.analyzer import DEFAULT_QUALITY_MODEL

        return DEFAULT_QUALITY_MODEL

    model = train_quality_model(args.calibration_folder.resolve())
    if args.save_model is not None:
        model.save(args.save_model)
    return model


def _available_destination(destination_dir: Path, filename: str) -> Path:
    destination = destination_dir / filename
    counter = 1
    while destination.exists():
        destination = destination_dir / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
        counter += 1
    return destination


def _write_fail_report(source: Path, fail_dir: Path, results: list[ImageAssessment]) -> None:
    failed = [result for result in results if not result.passed]
    report_path = fail_dir / "failure-report.md"
    lines = [
        "# Failure Report",
        "",
        f"Failed images: {len(failed)}",
        "",
    ]
    if failed:
        lines.extend(
            [
                "| Image | Original Path | Reasons | Blur Score | Mean Luminance | Dark Clip | Light Clip |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in failed:
            try:
                original_path = result.path.relative_to(source)
            except ValueError:
                original_path = result.path
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(result.path.name),
                        _escape_markdown(str(original_path)),
                        _escape_markdown(", ".join(result.reasons) or "unknown"),
                        f"{result.blur_score:.2f}",
                        f"{result.mean_luminance:.2f}",
                        f"{result.clipped_dark_ratio:.3f}",
                        f"{result.clipped_light_ratio:.3f}",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No failed images.")
    report_path.write_text("\n".join(lines) + "\n")


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def _format_results(results: list[ImageAssessment], dry_run: bool) -> str:
    payload = {
        "dry_run": dry_run,
        "total": len(results),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "images": [
            {
                **asdict(result),
                "path": str(result.path),
            }
            for result in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())

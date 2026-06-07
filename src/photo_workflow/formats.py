from __future__ import annotations

from pillow_heif import register_heif_opener


def register_image_openers() -> None:
    register_heif_opener()

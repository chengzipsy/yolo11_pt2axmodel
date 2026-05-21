#!/usr/bin/env python3
"""Package calibration images into the tar format expected by Pulsar2."""

from __future__ import annotations

import argparse
import random
import tarfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a calibration image tar.")
    parser.add_argument("images_dir")
    parser.add_argument("images_num", type=int)
    parser.add_argument("--output", default="tmp_images/images.tar")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_calibration_tar(images_dir: Path, images_num: int, output: Path, seed: int) -> Path:
    images = collect_images(images_dir)
    print(f"images dir: {images_dir}")
    print(f"images num: {images_num}")
    print(f"available images: {len(images)}")
    if len(images) < images_num:
        raise SystemExit(f"Not enough images in {images_dir}: have {len(images)}, need {images_num}")

    random.seed(seed)
    selected = random.sample(images, images_num)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as tar:
        for image_path in selected:
            tar.add(image_path, arcname=image_path.name)
    print(f"Wrote {output}")
    return output


def main() -> None:
    args = parse_args()
    create_calibration_tar(Path(args.images_dir), args.images_num, Path(args.output), args.seed)


if __name__ == "__main__":
    main()

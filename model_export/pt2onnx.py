#!/usr/bin/env python3
"""Export YOLO .pt models under data_need to ONNX."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO .pt files to ONNX.")
    parser.add_argument("--pt-glob", default="model_export/data_need/*.pt")
    parser.add_argument("--out-dir", default="model_export/build/onnx")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Export image size. Use one value for square input, or two values as height width.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.imgsz) not in (1, 2):
        raise SystemExit("--imgsz expects one value, or two values as height width")
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz

    pt_paths = [Path(path) for path in glob.glob(args.pt_glob)]
    if not pt_paths:
        raise SystemExit(f"No .pt files matched: {args.pt_glob}")

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pt_path in pt_paths:
        print(f"Exporting {pt_path} to ONNX")
        model = YOLO(str(pt_path))
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=args.opset,
            simplify=not args.no_simplify,
            dynamic=args.dynamic,
        )
        exported_path = Path(exported)
        if not exported_path.exists():
            exported_path = pt_path.with_suffix(".onnx")
        if not exported_path.exists():
            raise FileNotFoundError(f"Ultralytics did not produce ONNX for {pt_path}")

        target = out_dir / f"{pt_path.stem}.onnx"
        shutil.copy2(exported_path, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()

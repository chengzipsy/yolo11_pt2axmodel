#!/usr/bin/env python3
"""Extract an ONNX submodel by input and output tensor names."""

from __future__ import annotations

import argparse

import onnx

from find_yolo11_reshape_outputs import find_candidates


def split_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ONNX model nodes.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("input_names")
    parser.add_argument("output_names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_names = split_names(args.input_names)
    if args.output_names.strip().lower() == "auto":
        model = onnx.load(args.input_path)
        candidates = [
            item
            for item in find_candidates(model)
            if int(item["branch_rank"]) in (0, 1)
        ][:6]
        output_names = [str(item["name"]) for item in candidates]
        print("Auto-selected output names:")
        for item in candidates:
            print(f"  {item['name']} shape={item['input_shape']}")
        if len(output_names) != 6:
            raise SystemExit(f"Expected 6 auto output names, got {len(output_names)}")
    else:
        output_names = split_names(args.output_names)
    if not input_names:
        raise SystemExit("input_names cannot be empty")
    if not output_names:
        raise SystemExit("output_names cannot be empty")
    onnx.utils.extract_model(args.input_path, args.output_path, input_names, output_names)


if __name__ == "__main__":
    main()

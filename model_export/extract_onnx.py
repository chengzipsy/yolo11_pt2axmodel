#!/usr/bin/env python3
"""Extract an ONNX submodel by input and output tensor names."""

from __future__ import annotations

import argparse

import onnx


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
    output_names = split_names(args.output_names)
    if not input_names:
        raise SystemExit("input_names cannot be empty")
    if not output_names:
        raise SystemExit("output_names cannot be empty")
    onnx.utils.extract_model(args.input_path, args.output_path, input_names, output_names)


if __name__ == "__main__":
    main()

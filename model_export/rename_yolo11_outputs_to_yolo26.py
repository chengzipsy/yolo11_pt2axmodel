#!/usr/bin/env python3
"""Rename YOLO11 cv2/cv3 outputs to YOLO26 one2one-style output names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


YOLO26_OUTPUT_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename extracted YOLO11 outputs to YOLO26 names.")
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    parser.add_argument("--names-output", required=True)
    parser.add_argument("--map-output", default="")
    return parser.parse_args()


def replace_output_name(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    for node in model.graph.node:
        for idx, value in enumerate(node.output):
            if value == old_name:
                node.output[idx] = new_name
    for value_info in list(model.graph.output) + list(model.graph.value_info):
        if value_info.name == old_name:
            value_info.name = new_name


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input_onnx)
    old_names = [output.name for output in model.graph.output]
    if len(old_names) != len(YOLO26_OUTPUT_NAMES):
        raise SystemExit(f"Expected 6 ONNX outputs, got {len(old_names)}: {old_names}")

    mapping = dict(zip(old_names, YOLO26_OUTPUT_NAMES))
    for old_name, new_name in mapping.items():
        replace_output_name(model, old_name, new_name)

    onnx.checker.check_model(model)
    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    names_path = Path(args.names_output)
    names_path.parent.mkdir(parents=True, exist_ok=True)
    names_path.write_text(",".join(YOLO26_OUTPUT_NAMES) + "\n", encoding="utf-8")

    if args.map_output:
        map_path = Path(args.map_output)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(mapping, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find YOLO11 reshape-input tensors for MaixCAM2 conversion.

The current Ultralytics YOLO11 ONNX graph flattens detection-head feature maps
with Reshape nodes. MaixPy expects the pre-reshape 4D tensors, ordered like:
bbox stride8/16/32, then cls stride8/16/32.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find YOLO11 4D outputs before Reshape.")
    parser.add_argument("onnx_path")
    parser.add_argument("--output", required=True, help="Path to write comma-separated output names.")
    parser.add_argument("--json-output", default="", help="Optional path to write candidate metadata.")
    parser.add_argument("--expected-count", type=int, default=6)
    return parser.parse_args()


def value_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | str]] = {}
    value_infos = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value_info in value_infos:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        shapes[value_info.name] = dims
    return shapes


def branch_rank(name: str) -> int:
    if re.search(r"(^|/)cv2\.", name):
        return 0
    if re.search(r"(^|/)cv3\.", name):
        return 1
    if "bbox" in name.lower():
        return 0
    if "cls" in name.lower():
        return 1
    return 9


def spatial_area(shape: list[int | str]) -> int:
    if len(shape) != 4 or not all(isinstance(dim, int) for dim in shape[-2:]):
        return -1
    return int(shape[-2]) * int(shape[-1])


def find_candidates(model: onnx.ModelProto) -> list[dict[str, object]]:
    shapes = value_shapes(model)
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Reshape" or not node.input or not node.output:
            continue
        input_name = node.input[0]
        output_name = node.output[0]
        input_shape = shapes.get(input_name)
        output_shape = shapes.get(output_name)
        if not input_shape or not output_shape:
            continue
        if len(input_shape) != 4 or len(output_shape) != 3:
            continue
        if input_name in seen:
            continue
        seen.add(input_name)
        candidates.append(
            {
                "name": input_name,
                "reshape": node.name,
                "branch_rank": branch_rank(input_name),
                "area": spatial_area(input_shape),
                "input_shape": input_shape,
                "output_shape": output_shape,
            }
        )
    return sorted(candidates, key=lambda item: (int(item["branch_rank"]), -int(item["area"]), str(item["name"])))


def main() -> None:
    args = parse_args()
    model = onnx.load(args.onnx_path)
    candidates = find_candidates(model)
    selected = [item for item in candidates if int(item["branch_rank"]) in (0, 1)]
    if len(selected) < args.expected_count:
        print(json.dumps(candidates, indent=2))
        raise SystemExit(f"Expected at least {args.expected_count} cv2/cv3 reshape-input candidates, got {len(selected)}")
    selected = selected[: args.expected_count]
    names = [str(item["name"]) for item in selected]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(",".join(names) + "\n", encoding="utf-8")

    payload = {"selected": selected, "all_candidates": candidates}
    print(json.dumps(payload, indent=2))
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

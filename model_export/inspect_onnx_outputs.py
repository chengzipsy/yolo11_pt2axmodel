#!/usr/bin/env python3
"""Write ONNX output rank metadata for Pulsar2 config generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ONNX output tensor ranks.")
    parser.add_argument("onnx_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-rank", type=int, default=0)
    return parser.parse_args()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    shape: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def main() -> None:
    args = parse_args()
    model = onnx.load(args.onnx_path)
    outputs = {
        output.name: {
            "shape": tensor_shape(output),
            "rank": len(tensor_shape(output)),
        }
        for output in model.graph.output
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))
    if args.require_rank:
        bad_outputs = {
            name: info
            for name, info in outputs.items()
            if info["rank"] != args.require_rank
        }
        if bad_outputs:
            raise SystemExit(
                f"Expected all ONNX outputs to be rank {args.require_rank}, "
                f"but got: {json.dumps(bad_outputs, indent=2)}"
            )


if __name__ == "__main__":
    main()

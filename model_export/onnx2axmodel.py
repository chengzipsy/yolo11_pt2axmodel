#!/usr/bin/env python3
"""Convert a YOLO11 ONNX model to MaixCAM2 AXModel + MUD files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from gen_cali_images_tar import create_calibration_tar


MODE_SUFFIX = {
    "NPU1": "vnpu",
    "NPU2": "npu",
}


def split_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MaixCAM2 AXModels with Pulsar2.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out-dir", default="model_export/build/axmodel")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--calib-images", default="model_export/data_need/calib_images")
    parser.add_argument("--calib-size", type=int, default=100)
    parser.add_argument("--input-names", default="images")
    parser.add_argument(
        "--output-names",
        default="/model.23/Concat_output_0,/model.23/Concat_1_output_0,/model.23/Concat_2_output_0",
    )
    parser.add_argument("--target-hardware", default="AX620E")
    parser.add_argument("--npu-modes", nargs="+", default=["NPU1", "NPU2"], choices=sorted(MODE_SUFFIX))
    parser.add_argument("--labels", default="class0")
    parser.add_argument("--labels-file", default="")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--output-metadata", default="")
    parser.add_argument("--output-perm-mode", choices=["auto", "always", "none"], default="auto")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def read_labels(args: argparse.Namespace) -> str:
    labels_file = Path(args.labels_file) if args.labels_file else None
    if labels_file and labels_file.exists():
        labels = [
            line.strip()
            for line in labels_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if labels:
            return ", ".join(labels)
    return args.labels


def read_output_ranks(metadata_path: str) -> dict[str, int]:
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: int(info["rank"])
        for name, info in data.items()
        if isinstance(info, dict) and "rank" in info
    }


def should_permute_output(output_name: str, output_ranks: dict[str, int], mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "none":
        return False
    return output_ranks.get(output_name) == 4


def write_config(
    config_path: Path,
    mode: str,
    input_name: str,
    output_names: list[str],
    tar_path: Path,
    size: int,
    output_ranks: dict[str, int],
    output_perm_mode: str,
) -> None:
    output_processors = []
    for output_name in output_names:
        processor = {"tensor_name": output_name}
        if should_permute_output(output_name, output_ranks, output_perm_mode):
            processor["dst_perm"] = [0, 2, 3, 1]
        output_processors.append(processor)

    config = {
        "model_type": "ONNX",
        "npu_mode": mode,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": input_name,
                    "calibration_dataset": str(tar_path),
                    "calibration_size": size,
                    "calibration_mean": [0, 0, 0],
                    "calibration_std": [255, 255, 255],
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": True,
        },
        "input_processors": [
            {
                "tensor_name": input_name,
                "tensor_format": "RGB",
                "tensor_layout": "NCHW",
                "src_format": "RGB",
                "src_dtype": "U8",
                "src_layout": "NHWC",
                "csc_mode": "NoCSC",
            }
        ],
        "output_processors": output_processors,
        "compiler": {
            "check": 3,
            "check_mode": "CheckOutput",
            "check_cosine_simularity": 0.9,
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_mud(out_dir: Path, model_name: str, labels: str, built_modes: list[str]) -> None:
    lines = [
        "[basic]",
        "type = axmodel",
    ]
    for mode in built_modes:
        suffix = MODE_SUFFIX[mode]
        lines.append(f"model_{suffix} = {model_name}_{suffix}.axmodel")
    lines.extend(
        [
            "",
            "[extra]",
            "model_type = yolo11",
            "type=detector",
            "input_type = rgb",
            f"labels = {labels}",
            "input_cache = true",
            "output_cache = true",
            "input_cache_flush = false",
            "output_cache_inval = true",
            "",
            "mean = 0,0,0",
            "scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098",
        ]
    )
    mud_path = out_dir / f"{model_name}.mud"
    mud_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {mud_path}")


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or onnx_path.stem
    work_dir = out_dir / "_work" / model_name
    work_dir.mkdir(parents=True, exist_ok=True)

    input_names = split_names(args.input_names)
    output_names = split_names(args.output_names)
    if len(input_names) != 1:
        raise SystemExit("This YOLO11 scaffold expects exactly one input tensor name.")
    if not output_names:
        raise SystemExit("At least one output tensor name is required.")

    if args.no_extract:
        model_for_build = onnx_path
    else:
        model_for_build = work_dir / f"{model_name}_extracted.onnx"
        run(
            [
                sys.executable,
                str(Path(__file__).with_name("extract_onnx.py")),
                str(onnx_path),
                str(model_for_build),
                args.input_names,
                args.output_names,
            ]
        )

    if args.no_simplify:
        simplified_model = model_for_build
    else:
        simplified_model = work_dir / f"{model_name}.onnx"
        run(["onnxsim", str(model_for_build), str(simplified_model)])

    calib_tar = create_calibration_tar(
        Path(args.calib_images).resolve(),
        args.calib_size,
        work_dir / "tmp_images" / "images.tar",
        seed=2026,
    )
    output_ranks = read_output_ranks(args.output_metadata)

    built_modes: list[str] = []
    for mode in args.npu_modes:
        suffix = MODE_SUFFIX[mode]
        build_dir = work_dir / f"build_{mode.lower()}"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        config_path = work_dir / f"yolo11_build_config_{mode.lower()}.json"
        write_config(
            config_path,
            mode,
            input_names[0],
            output_names,
            calib_tar,
            args.calib_size,
            output_ranks,
            args.output_perm_mode,
        )
        run(
            [
                "pulsar2",
                "build",
                "--target_hardware",
                args.target_hardware,
                "--input",
                str(simplified_model),
                "--output_dir",
                str(build_dir),
                "--config",
                str(config_path),
            ]
        )
        compiled = build_dir / "compiled.axmodel"
        if not compiled.exists():
            raise FileNotFoundError(f"Pulsar2 did not produce {compiled}")
        target = out_dir / f"{model_name}_{suffix}.axmodel"
        shutil.copy2(compiled, target)
        print(f"Wrote {target}")
        built_modes.append(mode)

    write_mud(out_dir, model_name, read_labels(args), built_modes)


if __name__ == "__main__":
    main()

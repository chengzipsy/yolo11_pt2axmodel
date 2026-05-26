# YOLO11 Detect 转 MaixCAM2 AXModel 使用说明（未完成 最近懒得搞）

这个仓库提供了一套 GitHub Actions workflow，用来把 YOLO11 检测模型的 `.pt` 文件转换成 MaixCAM2 可用的 `.axmodel` 文件。

整个流程分成两个 job：

1. `pt2onnx`：把 `model_export/data_need` 目录下的 `.pt` 模型导出成 ONNX。
2. `onnx2axmodel`：使用 Pulsar2 把 ONNX 转换成 MaixCAM2 的 AXModel。

## 文件目录

请把模型和校准图片放到下面的位置：

```text
model_export/
  data_need/
    best.pt
    calib_images/
      000001.jpg
      000002.jpg
      ...
    labels.txt          # 可选
```

必须提供：

- `model_export/data_need/*.pt`：YOLO11 的 `.pt` 模型文件。
- `model_export/data_need/calib_images`：Pulsar2 量化时使用的校准图片。

可选提供：

- `model_export/data_need/labels.txt`：类别名称文件，每行一个类别。如果没有这个文件，workflow 会使用 `labels` 输入参数。

## Workflow 文件

真正会被 GitHub Actions 执行的是：

```text
.github/workflows/yolo11_detect_onnx2axmodel.yml
```

## 如何运行

1. 把仓库推送到 GitHub。
2. 打开 GitHub 仓库页面。
3. 进入 `Actions` 页面。
4. 选择 `YOLO11 Detect ONNX to AXModel`。
5. 点击 `Run workflow`。
6. 根据需要填写参数。
7. 启动 workflow。

## Workflow 参数说明

常用参数：

- `pt_glob`：需要导出的 PT 文件匹配规则，默认是 `model_export/data_need/*.pt`。
- `calib_images_dir`：校准图片目录，默认是 `model_export/data_need/calib_images`。
- `model_name`：输出模型基础名称，默认是 `yolo11_detect`。
- `image_size`：YOLO 导出 ONNX 时的输入尺寸，默认是 `640`。
- `opset`：ONNX opset 版本，默认是 `12`。
- `calibration_size`：量化时使用的校准图片数量，默认是 `100`。
- `input_names`：ONNX 输入 tensor 名称，默认是 `images`。
- `output_names`：YOLO11 检测输出 tensor 名称，用于 ONNX 截取和 Pulsar2 量化配置。
- `labels`：当不存在 `labels.txt` 时使用的类别名称，多个类别用英文逗号分隔。

Pulsar2 相关参数：

- `pulsar2_image`：包含 `pulsar2`、`onnxsim`、Python 和 ONNX 依赖的 Docker 镜像。
- `pulsar2_tar_url`：可选参数。如果你的 Pulsar2 镜像是一个 docker save 导出的 `.tar` 或 `.tar.gz` 文件，可以填这个下载地址，workflow 会先下载并执行 `docker load`。

## Pulsar2 Docker 镜像要求

第二个 job 会在 Docker 容器里运行转换命令：

```bash
docker run --rm --privileged \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$PULSAR2_IMAGE" \
  python model_export/onnx2axmodel.py ...
```

因此 `pulsar2_image` 指向的镜像里必须能直接使用这些命令：

```bash
pulsar2
onnxsim
python
```

如果你的 Pulsar2 镜像是私有镜像，需要在 workflow 的 `Prepare Pulsar2 docker image` 步骤之前增加 Docker 登录步骤。

如果你的镜像不能直接 `docker pull`，可以把镜像提前用 `docker save` 导出并上传到一个可下载地址，然后在运行 workflow 时填写 `pulsar2_tar_url`。

## 输出产物

workflow 运行完成后，在 GitHub Actions 运行记录里下载这个 artifact：

```text
yolo11-axmodel-maixcam2
```

里面会包含类似下面的文件：

```text
yolo11_detect_vnpu.axmodel
yolo11_detect_npu.axmodel
yolo11_detect.mud
```

其中：

- `vnpu` 对应 `NPU1`。
- `npu` 对应 `NPU2`。
- `.mud` 是 MaixPy 加载模型时需要的模型描述文件。

## 转换流程说明

当前 workflow 按 MaixCAM2 的 Pulsar2 转换流程搭建：

1. 使用 Ultralytics 把 YOLO `.pt` 导出成 ONNX。
2. 从 ONNX 中截取 YOLO11 检测输出节点。
3. 使用 `onnxsim` 简化截取后的 ONNX。
4. 把校准图片打包成 `images.tar`。
5. 执行 Pulsar2 编译：

```bash
pulsar2 build \
  --target_hardware AX620E \
  --input model.onnx \
  --output_dir build \
  --config yolo11_build_config.json
```

默认使用的 YOLO11 检测输出节点是：

```text
/model.23/Concat_output_0
/model.23/Concat_1_output_0
/model.23/Concat_2_output_0
```

如果你导出的 ONNX 中 tensor 名称不同，需要在运行 workflow 时修改 `output_names` 参数。

## 本地脚本用法

如果你想先在本地测试 `pt -> onnx`，可以运行：

```bash
python model_export/pt2onnx.py \
  --pt-glob "model_export/data_need/*.pt" \
  --out-dir model_export/build/onnx \
  --imgsz 640 \
  --opset 12
```

如果你本地已经有 Pulsar2 环境，可以直接运行 ONNX 转 AXModel：

```bash
python model_export/onnx2axmodel.py \
  --onnx model_export/build/onnx/best.onnx \
  --out-dir model_export/build/axmodel \
  --model-name yolo11_detect \
  --calib-images model_export/data_need/calib_images \
  --calib-size 100 \
  --input-names images \
  --output-names "/model.23/Concat_output_0,/model.23/Concat_1_output_0,/model.23/Concat_2_output_0" \
  --labels "class0"
```

## 常见问题

### Pulsar2 命令找不到

说明 `pulsar2_image` 指向的 Docker 镜像里没有 `pulsar2`，或者镜像没有正确加载。

请确认：

- `pulsar2_image` 名称正确。
- 镜像可以被 `docker pull` 拉取。
- 或者 `pulsar2_tar_url` 可以正常下载并被 `docker load` 加载。

### ONNX 截取失败

通常是 `input_names` 或 `output_names` 和实际 ONNX tensor 名称不一致。

需要检查 ONNX 模型中的输入输出节点名称，然后重新填写 workflow 参数。

### 校准图片数量不足

如果 `calibration_size` 设置为 `100`，那么 `calib_images` 目录里至少需要 100 张图片。

可以减少 `calibration_size`，或者补充更多校准图片。

## 备注

- GitHub-hosted runner 默认不包含 Pulsar2。
- 校准图片应尽量接近模型真实使用场景，否则量化后精度可能下降。

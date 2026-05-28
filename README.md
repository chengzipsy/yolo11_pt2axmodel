# YOLO11 Detect 转 MaixCAM2 AXModel

这个仓库用于把 YOLO11 检测模型 `.pt` 转成 MaixCAM2 可用的 `.axmodel` 和 `.mud`。

当前稳定路线是按 Sipeed 教程的 YOLO11 三输出节点方案转换：

```text
/model.23/Concat_output_0
/model.23/Concat_1_output_0
/model.23/Concat_2_output_0
```

注意：需要使用能导出上述节点的 Ultralytics 环境。当前 workflow 固定使用 `ultralytics==8.3.39`、`torch==2.5.1`，避免新版 PyTorch ONNX exporter 改变 graph 结构。
## 实测建议

其实建议直接使用yolo26 按照Sipeed的教程转换 我现阶段还没有尝试yolo26的部署 

但是他文档有关yolo11的转换流程都是根据他写文档时ultralytics，torch，onnx等依赖的版本编写的

而现在最新版的ultralytics等依赖中 yolo11在pt转onnx中的输出节点与sipeed文档中的yolo26结构是相似的 都是3个bbox和3个cls

这导致最新的依赖中如果按照教程的三个concat节点转换会出问题 因为这三个concat前经过了一个reshape导致四通道变为三通道


## 目录结构

把模型和校准图片放到：

```text
model_export/
  data_need/
    best.pt
    labels.txt
    calib_images/
      000001.jpg
      000002.jpg
      ...
```

说明：

- `best.pt`：需要转换的 YOLO11 权重。
- `labels.txt`：类别名，每行一个。
- `calib_images/`：Pulsar2 量化校准图片，默认使用 100 张。

## GitHub Actions

workflow 文件：

```text
.github/workflows/yolo11_detect_onnx2axmodel.yml
```

手动触发：

1. 打开 GitHub 仓库。
2. 进入 `Actions`。
3. 选择 `YOLO11 Detect ONNX to AXModel`。
4. 点击 `Run workflow`。

常用参数：

```text
pt_glob           model_export/data_need/best.pt
image_size        480 640
opset             17
input_names       images
output_names      /model.23/Concat_output_0,/model.23/Concat_1_output_0,/model.23/Concat_2_output_0
calibration_size  100
pulsar2_image     pulsar2:6.0
```

`image_size` 支持一个或两个值：

```text
640       表示 640x640
480 640   表示 height=480, width=640
```

## 固定导出环境

workflow 中 `pt2onnx` job 固定安装关键版本：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics==8.3.39 onnx==1.21.0 onnxslim==0.1.93 onnxruntime==1.23.2 onnxsim numpy==2.2.6 protobuf==7.35.0
```

这样做的原因是新版 PyTorch/ONNX exporter 可能会改变 ONNX 节点名，导致教程里的 Concat 节点不存在。

## 转换流程

workflow 分两个 job：

1. `pt2onnx`

   ```text
   pt -> onnx
   extract_onnx.py 提取三个 YOLO11 Concat 输出节点
   onnxsim 简化
   检查输出 rank 必须是 4
   上传处理后的 ONNX artifact
   ```

2. `onnx2axmodel`

   ```text
   下载 ONNX artifact
   下载校准图片
   加载 Pulsar2 6.0 Docker 镜像
   pulsar2 build 生成 NPU1 / NPU2 模型
   生成 .mud
   上传 axmodel artifact
   ```

输出 artifact：

```text
yolo11-axmodel-maixcam2
```

里面包含：

```text
yolo11_detect_vnpu.axmodel
yolo11_detect_npu.axmodel
yolo11_detect.mud
```

说明：

- `*_vnpu.axmodel`：对应 `NPU1`。
- `*_npu.axmodel`：对应 `NPU2`。
- `.mud`：MaixPy 加载模型时使用。

## Pulsar2 镜像

默认使用：

```text
pulsar2:6.0
```

workflow 会从 Hugging Face 下载：

```text
https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/6.0/ax_pulsar2_6.0.tar.gz?download=true
```


## 部署到 MaixCAM2

把 artifact 里的文件放到板子，例如：

```text
/root/models/train15_2026.5.21/
  yolo11_detect_vnpu.axmodel
  yolo11_detect_npu.axmodel
  yolo11_detect.mud
```

MaixPy 代码示例：

```python
from maix import camera, display, image, nn, app, time

detector = nn.YOLO11(model="/root/models/train15_2026.5.21/yolo11_detect.mud", dual_buff = True)

cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(),fps=-1,buff_num=1)
disp = display.Display()

while not app.need_exit():
    img = cam.read()

    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED,scale=3,thickness=3)
    fps = time.fps()
    img.draw_string(10, 10, f"fps: {fps:.02f}", color = image.COLOR_RED,scale=3,thickness=3)
    disp.show(img)
```

如果 `detector.detect()` 直接退出、没有 Python traceback，优先确认 MaixCAM2 系统镜像是否为较新版本。实际测试中，重新刷 MaixCAM 镜像后 `nn.YOLO11` 可以正常运行。


## 常见问题

### 找不到 Concat 输出节点

通常是导出环境版本不一致。确认 Action 日志里版本接近：

```text
torch 2.5.1
ultralytics 8.3.39
onnx 1.21.0
```

如果使用新版 torch，可能会出现教程节点不存在的问题。

### 输出不是 rank4

当前教程路线要求提取后的三个输出必须是 rank4。workflow 会在 `inspect_onnx_outputs.py --require-rank 4` 阶段检查，失败时说明 ONNX graph 不符合教程转换格式。

### 模型能 forward，但 `nn.YOLO11` 不能 detect

先确认 `.mud` 中：

```ini
model_type = yolo11
type = detector
```

然后确认 MaixCAM2 系统镜像版本。旧镜像可能在 `detector.detect()` 阶段直接退出。

### 不要把 Pulsar2 镜像提交进 Git

Pulsar2 镜像 tar 包大于 GitHub 单文件限制。应使用 workflow 的 `pulsar2_tar_url` 下载，或使用 Docker registry / GitHub Actions cache。

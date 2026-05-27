from ultralytics import YOLO
import sys

# 打印路径，方便排查
print("Python path:", sys.path)

# 从命令行获取参数（你传的 best.pt 320 224 会被正确读取）
if len(sys.argv) != 4:
    print("用法: python zhuanhua.py <模型路径> <输入宽度> <输入高度>")
    print("例子: python zhuanhua.py best.pt 320 224")
    sys.exit(1)

net_name = sys.argv[1]
input_width = int(sys.argv[2])
input_height = int(sys.argv[3])

print(f"加载模型: {net_name}")
print(f"导出尺寸: {input_width}x{input_height}")

# 只加载你传入的模型，不写死路径
model = YOLO(net_name)

# 随便用一张图片做预测（不写死路径，避免卡住）
print("正在进行预测...")
results = model.predict("https://ultralytics.com/images/bus.jpg", save=False, verbose=False)

# 导出ONNX模型
print("正在导出ONNX模型...")
path = model.export(
    format="onnx",
    imgsz=[input_height, input_width],
    dynamic=False,
    simplify=True,
    opset=17
)

print(f"✅ 导出成功！文件路径: {path}")

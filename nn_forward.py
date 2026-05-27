from maix import app, camera, display, image, nn, tensor, time
import numpy as np


MODEL_PATH = "/root/models/train15_2026.5.21/yolo11_detect.mud"
CONF_TH = 0.5
IOU_TH = 0.45
REG_MAX = 16


BBOX_OUTPUTS = [
    "/model.23/cv2.0/cv2.0.2/Conv_output_0",
    "/model.23/cv2.1/cv2.1.2/Conv_output_0",
    "/model.23/cv2.2/cv2.2.2/Conv_output_0",
]
CLS_OUTPUTS = [
    "/model.23/cv3.0/cv3.0.2/Conv_output_0",
    "/model.23/cv3.1/cv3.1.2/Conv_output_0",
    "/model.23/cv3.2/cv3.2.2/Conv_output_0",
]


def load_labels_from_mud(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line.startswith("labels"):
                    continue
                _, value = line.split("=", 1)
                return [item.strip() for item in value.split(",") if item.strip()]
    except Exception:
        pass
    return []


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def image_to_numpy_uint8(img, width, height):
    if img.width() != width or img.height() != height:
        img = img.resize(width, height, image.Fit.FIT_FILL)
    if img.format() != image.Format.FMT_RGB888:
        img = img.to_format(image.Format.FMT_RGB888)
    data = img.to_bytes(copy=True)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    return arr, img


def make_input_tensors(model, img_np):
    input_tensors = tensor.Tensors()
    keepalive = []
    for layer in model.inputs_info():
        dtype_name = str(layer.dtype).lower()
        if "uint8" not in dtype_name:
            raise RuntimeError("This demo expects uint8 model input, got {}".format(layer.dtype))
        data = img_np.reshape(layer.shape)
        t = tensor.tensor_from_numpy_uint8(data, copy=False)
        input_tensors.add_tensor(layer.name, t, False, False)
        keepalive.append(t)
    return input_tensors, keepalive


def decode_scale(bbox, cls, stride_x, stride_y, conf_th):
    # bbox: [1, H, W, 64], cls: [1, H, W, num_classes]
    bbox = bbox[0]
    cls = cls[0]
    h, w, _ = bbox.shape
    num_classes = cls.shape[-1]

    cls_scores = sigmoid(cls)
    class_ids = np.argmax(cls_scores, axis=-1)
    scores = np.max(cls_scores, axis=-1)
    mask = scores >= conf_th
    if not np.any(mask):
        return []

    bbox = bbox.reshape(h, w, 4, REG_MAX)
    prob = softmax(bbox, axis=-1)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = np.sum(prob * bins, axis=-1)

    ys, xs = np.where(mask)
    results = []
    for y, x in zip(ys, xs):
        left, top, right, bottom = dist[y, x]
        cx = x + 0.5
        cy = y + 0.5
        x1 = (cx - left) * stride_x
        y1 = (cy - top) * stride_y
        x2 = (cx + right) * stride_x
        y2 = (cy + bottom) * stride_y
        cls_id = int(class_ids[y, x])
        if cls_id >= num_classes:
            continue
        results.append([float(x1), float(y1), float(x2), float(y2), float(scores[y, x]), cls_id])
    return results


def nms(dets, iou_th):
    if not dets:
        return []
    dets = np.array(dets, dtype=np.float32)
    x1, y1, x2, y2, scores, cls_ids = dets.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep = []
    for cls_id in np.unique(cls_ids):
        idxs = np.where(cls_ids == cls_id)[0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[current] + areas[rest] - inter + 1e-6
            iou = inter / union
            idxs = rest[iou <= iou_th]
    return dets[keep].tolist()


def postprocess(outputs, input_w, input_h):
    detections = []
    for bbox_name, cls_name in zip(BBOX_OUTPUTS, CLS_OUTPUTS):
        bbox = tensor.tensor_to_numpy_float32(outputs[bbox_name], copy=False)
        cls = tensor.tensor_to_numpy_float32(outputs[cls_name], copy=False)
        _, h, w, _ = bbox.shape
        stride_x = input_w / w
        stride_y = input_h / h
        detections.extend(decode_scale(bbox, cls, stride_x, stride_y, CONF_TH))
    return nms(detections, IOU_TH)


def draw_detections(img, detections, labels, input_w, input_h):
    img_w = img.width()
    img_h = img.height()
    scale_x = img_w / input_w
    scale_y = img_h / input_h
    color = image.Color.from_rgb(255, 0, 0)
    text_color = image.Color.from_rgb(255, 255, 255)
    bg_color = image.Color.from_rgb(255, 0, 0)
    for x1, y1, x2, y2, score, cls_id in detections:
        cls_id = int(cls_id)
        x = max(0, min(img_w - 1, int(x1 * scale_x)))
        y = max(0, min(img_h - 1, int(y1 * scale_y)))
        r = max(0, min(img_w - 1, int(x2 * scale_x)))
        b = max(0, min(img_h - 1, int(y2 * scale_y)))
        w = max(1, r - x)
        h = max(1, b - y)
        name = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = "{} {:.2f}".format(name, score)
        img.draw_rect(x, y, w, h, color, 2)
        img.draw_rect(x, max(0, y - 22), max(80, len(text) * 9), 22, bg_color, -1)
        img.draw_string(x + 2, max(0, y - 20), text, text_color)


def main():
    model = nn.NN(MODEL_PATH, dual_buff=False)
    labels = load_labels_from_mud(MODEL_PATH)

    input_info = model.inputs_info()[0]
    input_h = int(input_info.shape[1])
    input_w = int(input_info.shape[2])
    print("model input: {}x{}".format(input_w, input_h))

    cam = camera.Camera(input_w, input_h, image.Format.FMT_RGB888)
    disp = display.Display()
    fps = 0.0

    while not app.need_exit():
        frame_t0 = time.ticks_ms()
        img = cam.read()
        input_np, input_img = image_to_numpy_uint8(img, input_w, input_h)
        input_tensors, keepalive = make_input_tensors(model, input_np[np.newaxis, ...])

        t0 = time.ticks_ms()
        outputs = model.forward(input_tensors, copy_result=False, dual_buff_wait=True)
        detections = postprocess(outputs, input_w, input_h)
        cost = time.ticks_ms() - t0
        frame_cost = max(1, time.ticks_ms() - frame_t0)
        instant_fps = 1000.0 / frame_cost
        fps = instant_fps if fps <= 0 else fps * 0.9 + instant_fps * 0.1

        draw_detections(input_img, detections, labels, input_w, input_h)
        input_img.draw_string(
            4,
            4,
            "FPS:{:.1f}  NPU:{}ms  objs:{}".format(fps, cost, len(detections)),
            image.Color.from_rgb(0, 255, 0),
        )
        disp.show(input_img)
        keepalive.clear()


if __name__ == "__main__":
    main()

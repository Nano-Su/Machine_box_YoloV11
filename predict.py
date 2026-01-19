# python
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/test/weights/best.pt")
    results = model.predict(
        source="datasets/futbol players.v9i.yolov11/test/images/1-fps-2_00032_jpeg_jpg.rf.143f5cba2d0550334ba142ed37e23d38.jpg",
        conf=0.35,
        save=True,
        name='yolo11n-futbol-players-predict',
    )

    # 确保 results 为可迭代列表
    if not isinstance(results, (list, tuple)):
        results = [results]

    for img_idx, r in enumerate(results):
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            print(f"Image {img_idx}: no detections")
            continue

        # boxes.xyxy / boxes.conf / boxes.cls 通常为张量
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else [None] * len(xyxy)
        classes = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else [None] * len(xyxy)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            conf = confs[i] if i < len(confs) else None
            cls = int(classes[i]) if i < len(classes) and classes[i] is not None else None
            print(f"Image {img_idx} - Box {i}: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={conf:.3f}, cls={cls}")

if __name__ == '__main__':
    main()

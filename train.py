from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
        # 使用内置 tensorboard 日志器，日志会写入 runs/train/yolo11n-futbol-players
    model.train(
        data="datasets/futbol players.v9i.yolov11/data.yaml",
        epochs=3,
        imgsz=640,
        batch=8,
        name='yolo11n-futbol-players',

    )

if __name__ == '__main__':
    main()
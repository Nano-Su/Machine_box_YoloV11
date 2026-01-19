from pygments.lexer import words
from ultralytics import YOLO
def main():
    # Load a pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640,workers=1, batch=8,name='test')

if __name__ == '__main__':
    main()
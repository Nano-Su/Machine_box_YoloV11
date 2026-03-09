# Machine_box_YoloV11

A YOLOv11-based object detection application with a PyQt5 graphical user interface. This project supports image selection, object detection using YOLO models, and visualization of detection results.

## Project Structure

- **main.py** — Simple PyQt5 GUI for selecting an image, running YOLO detection, and viewing results.
- **dome.py** — Enhanced PyQt5 GUI with a card-based layout, styled controls, and window resizing support.
- **Predict.py** — Prediction module that loads a YOLO model and performs object detection on images.
- **train.py** — Training script to fine-tune a YOLOv11 model on a custom dataset.
- **datasets_origin/** — Contains zipped dataset archives for training.

## Requirements

- Python 3.8+
- [ultralytics](https://github.com/ultralytics/ultralytics)
- PyQt5
- OpenCV (`cv2`)

Install dependencies:

```bash
pip install ultralytics pyqt5 opencv-python
```

## Usage

### Running the GUI

```bash
python main.py
```

Or use the enhanced interface:

```bash
python dome.py
```

### Running Prediction from the Command Line

```bash
python Predict.py
```

This will run detection on the included `bus.jpg` sample image using the default `yolo11n.pt` model.

### Training a Custom Model

1. Extract your dataset into the `datasets/` directory.
2. Update the `data` path in `train.py` to point to your dataset YAML configuration.
3. Run:

```bash
python train.py
```

## License

See [LICENSE](LICENSE) for details.
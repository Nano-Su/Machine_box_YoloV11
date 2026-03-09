from ultralytics import YOLO

def predict_image(image_path, model_path="yolo11n.pt", save=True, show=False):
    """
    Perform object detection on an image using a YOLO model.

    :param image_path: Path to the input image.
    :param model_path: Path to the YOLO model weights (default: yolo11n.pt).
    :param save: Whether to save the detection results to disk.
    :param show: Whether to display the detection results in a window.
    :return: List of detection results.
    """
    # Load the model
    model = YOLO(model_path)

    # Perform object detection on an image
    results = model(image_path, save=save)

    # Display the results
    if show:
        results[0].show()

    return results

if __name__ == "__main__":
    predict_image("bus.jpg", show=True)

from ultralytics import YOLO

def predict_image(image_path, model_path="yolo11n.pt", save=True, show=False):
    """
    使用YOLO模型进行图片检测22402029@masu.edu.cn
    :param image_path: 图片路径
    :param model_path: 模型路径，默认 yolo11n.pt
    :param save: 是否保存结果
    :param show: 是否展示结果
    :return: 检测结果列表
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

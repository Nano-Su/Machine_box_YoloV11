from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='ch',device='gpu',use_angle_cls=True) # 默认使用中文模型，其他参数使用默认值
def OCR(img_path):
    # ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_server_det",
    #     text_recognition_model_name="PP-OCRv5_server_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False,
    # ) # 更换 PP-OCRv5_server 模型
    result = ocr.predict(img_path)
    res =result[0]

    texts = res['rec_texts']
    scores = res['rec_scores']
    boxes = res['rec_polys']

    #文字信息
    results = []
    for i in range(len(texts)):
        text_info = f"{texts[i]}，置信度: {scores[i]:.4f}"
        # print(text_info)
        # print("-" * 20)
        results.append(text_info)
    return results

    #框选列表
    # for i in range(len(boxes)):
    #     print(f"框选坐标: {boxes[i]}")

if __name__ == "__main__":
    OCR("bus.jpg")

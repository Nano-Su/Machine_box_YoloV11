import os
from paddleocr import PaddleOCR

# 延迟初始化 OCR 引擎，避免导入时就加载模型拖慢启动速度
_ocr_instance = None

def _get_ocr():
    """延迟初始化 PaddleOCR 实例，支持 GPU 不可用时自动回退 CPU"""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            _ocr_instance = PaddleOCR(lang='ch', device='gpu', use_angle_cls=True)
        except Exception:
            # GPU 不可用时回退到 CPU
            print("GPU 不可用，回退到 CPU 模式")
            _ocr_instance = PaddleOCR(lang='ch', device='cpu', use_angle_cls=True)
    return _ocr_instance

def OCR(img_path):
    """
    对图片进行 OCR 文字识别
    :param img_path: 图片路径
    :return: 识别结果列表（文字+置信度），失败返回空列表
    """
    # 检查图片路径是否存在
    if not os.path.exists(img_path):
        print(f"OCR 错误：图片文件不存在 - {img_path}")
        return []

    try:
        ocr = _get_ocr()
        result = ocr.predict(img_path)

        # 检查识别结果是否为空
        if not result or len(result) == 0:
            print(f"OCR 警告：未识别到任何内容 - {img_path}")
            return []

        res = result[0]

        # 检查结果字段是否存在
        if not res or 'rec_texts' not in res:
            print(f"OCR 警告：识别结果格式异常 - {img_path}")
            return []

        texts = res['rec_texts']
        scores = res['rec_scores']

        # 检查文字列表是否为空
        if not texts:
            print(f"OCR 警告：识别结果为空 - {img_path}")
            return []

        # 组装文字信息
        results = []
        for i in range(len(texts)):
            text_info = f"{texts[i]}，置信度: {scores[i]:.4f}"
            results.append(text_info)
        return results

    except FileNotFoundError:
        print(f"OCR 错误：图片读取失败 - {img_path}")
        return []
    except IndexError as e:
        print(f"OCR 错误：识别结果解析异常 - {e}")
        return []
    except Exception as e:
        print(f"OCR 错误：未知异常 - {e}")
        return []

if __name__ == "__main__":
    OCR("bus.jpg")

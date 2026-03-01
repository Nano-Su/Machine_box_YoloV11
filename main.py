import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from OCR import OCR

from Predict import predict_image
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 目标检测")
        self.resize(1000, 600)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()

        # 第一组
        layout1 = QVBoxLayout()
        self.image_label1 = QLabel("请选择一张图片进行检测")
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setStyleSheet("border: 2px dashed #aaa; font-size: 20px; color: #555;")
        self.image_label1.setScaledContents(True)
        layout1.addWidget(self.image_label1)

        self.button1 = QPushButton("选择图片")
        self.button1.clicked.connect(lambda: self.choose_image())
        self.button1.setFixedHeight(50)
        layout1.addWidget(self.button1)
        main_layout.addLayout(layout1)

        # 第二组
        layout2 = QVBoxLayout()
        self.image_label2 = QLabel("检测结果将显示在这里")
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setStyleSheet("border: 2px dashed #aaa; font-size: 20px; color: #555;")
        self.image_label2.setScaledContents(True)
        layout2.addWidget(self.image_label2)

        self.button2 = QPushButton("开始检验")
        self.button2.clicked.connect(lambda: self.Predict())
        self.button2.setFixedHeight(50)
        layout2.addWidget(self.button2)
        main_layout.addLayout(layout2)

        # 第三组
        layout3 = QVBoxLayout()
        self.image_label3 = QLabel("识别信息窗口")
        self.image_label3.setAlignment(Qt.AlignCenter)
        self.image_label3.setStyleSheet("border: 2px dashed #aaa; font-size: 20px; color: #555;")
        self.image_label3.setScaledContents(True)
        layout3.addWidget(self.image_label3)

        self.button3 = QPushButton("信息识别")
        self.button3.clicked.connect(lambda: self.information_ocr())
        self.button3.setFixedHeight(50)
        layout3.addWidget(self.button3)
        main_layout.addLayout(layout3)

        self.setLayout(main_layout)

    def choose_image(self):
        """选择图片并显示"""
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', "Image files (*.jpg *.gif *.png *.jpeg)")

        if fname:
            self.current_image_path = fname

            # 直接在 self.image_label1 上显示
            pixmap = QPixmap(fname)
            if not pixmap.isNull():
                self.image_label1.setPixmap(pixmap)
                print(f"Loaded image: {fname}")

    def Predict(self):
        """调用预测函数并显示结果"""
        if hasattr(self, 'current_image_path'):
            try:
                # 调用预测函数
                self.results = predict_image(self.current_image_path, model_path='yolo11n.pt',save=False)

                if len(self.results) > 0:
                    # 获取绘制了框的图像 (results[0].plot() 返回的是 BGR numpy 数组)
                    plotted_image = self.results[0].plot()

                    # 转换 BGR 到 RGB
                    res_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)

                    # 转换为 QImage
                    height, width, channel = res_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(res_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                    pixmap = QPixmap.fromImage(q_image)
                    if not pixmap.isNull():
                        self.image_label2.setPixmap(pixmap)
                        print("Predicted image with all boxes displayed.")
                else:
                    print("未检测到目标")
                    self.image_label2.setText("未检测到目标")

            except Exception as e:
                print(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("请先选择一张图片进行检测")

    def information_ocr(self):
        """调用 OCR 函数并显示结果"""
        if hasattr(self, 'results') and len(self.results) > 0 and len(self.results[0].boxes) > 0:
            try:
                original_image = cv2.imread(self.current_image_path)
                all_ocr_results = []

                # 遍历所有被检测到的框
                for i, box in enumerate(self.results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # 裁剪图像
                    cropped_image = original_image[y1:y2, x1:x2]

                    # 保存临时文件
                    temp_crop_path = f"temp_crop_{i}.jpg"
                    cv2.imwrite(temp_crop_path, cropped_image)

                    # 对每个裁剪区域进行 OCR
                    ocr_res = OCR(temp_crop_path)
                    if ocr_res:
                        all_ocr_results.extend(ocr_res)

                if all_ocr_results:
                    info_text = "\n".join(all_ocr_results)
                    self.image_label3.setText(info_text)
                    print("OCR results displayed for all boxes.")
                else:
                    self.image_label3.setText("未识别到文字")

            except Exception as e:
                print(f"OCR error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("请先进行检测或未检测到目标")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

import sys
import os
import tempfile
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
                              QFileDialog, QHBoxLayout, QFrame, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QSize
from OCR import OCR

from Predict import predict_image
import cv2

# 全局样式表 — 淡白背景 + 现代简约风格
GLOBAL_STYLE = """
QWidget#mainWindow {
    background-color: #f5f7fa;
}

QLabel#titleLabel {
    font-size: 22px;
    font-weight: 700;
    color: #2c3e50;
    padding: 0px;
}

QLabel#subtitleLabel {
    font-size: 12px;
    color: #95a5a6;
    padding: 0px;
}

QFrame.card {
    background-color: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 12px;
}

QLabel.imageDisplay {
    background-color: #fafbfc;
    border: 2px dashed #d0d7de;
    border-radius: 8px;
    font-size: 14px;
    color: #8b95a1;
    padding: 20px;
}

QPushButton.actionBtn {
    background-color: #4a90d9;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    padding: 12px 0px;
}
QPushButton.actionBtn:hover {
    background-color: #3a7bc8;
}
QPushButton.actionBtn:pressed {
    background-color: #2e6bb5;
}

QPushButton#btnSelect {
    background-color: #4a90d9;
}
QPushButton#btnSelect:hover {
    background-color: #3a7bc8;
}

QPushButton#btnDetect {
    background-color: #27ae60;
}
QPushButton#btnDetect:hover {
    background-color: #219a52;
}
QPushButton#btnDetect:pressed {
    background-color: #1e8449;
}

QPushButton#btnOCR {
    background-color: #8e44ad;
}
QPushButton#btnOCR:hover {
    background-color: #7d3c98;
}
QPushButton#btnOCR:pressed {
    background-color: #6c3483;
}

QLabel.cardTitle {
    font-size: 13px;
    font-weight: 600;
    color: #4a5568;
    padding: 0px;
    background: transparent;
    border: none;
}
"""


def _add_shadow(widget, blur=20, offset=2, color_alpha=25):
    """为卡片添加淡雅阴影效果"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset)
    from PyQt5.QtGui import QColor
    shadow.setColor(QColor(0, 0, 0, color_alpha))
    widget.setGraphicsEffect(shadow)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("药盒智能检测系统")
        self.resize(1200, 680)
        self.setMinimumSize(900, 500)
        # 预加载 YOLO 模型，避免每次预测都重新加载
        from ultralytics import YOLO
        self.model = YOLO("yolo11n.pt")
        # 临时文件列表，用于清理
        self._temp_files = []
        self.setup_ui()

    def _make_card(self, title_text, placeholder_text, button_text, button_id, button_slot):
        """创建一张带标题、图片区域、按钮的卡片"""
        card = QFrame()
        card.setProperty("class", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        # 卡片标题
        title = QLabel(title_text)
        title.setProperty("class", "cardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        card_layout.addWidget(title)

        # 图片展示区域
        image_label = QLabel(placeholder_text)
        image_label.setProperty("class", "imageDisplay")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setScaledContents(True)
        image_label.setMinimumHeight(200)
        card_layout.addWidget(image_label, 1)

        # 操作按钮
        btn = QPushButton(button_text)
        btn.setObjectName(button_id)
        btn.setProperty("class", "actionBtn")
        btn.setFixedHeight(44)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(button_slot)
        card_layout.addWidget(btn)

        _add_shadow(card)
        return card, image_label, btn

    def setup_ui(self):
        # 外层总布局
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        # ====== 顶部标题栏 ======
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_label = QLabel("药盒智能检测系统")
        title_label.setObjectName("titleLabel")

        subtitle_label = QLabel("YOLO 目标检测  ·  PaddleOCR 文字识别")
        subtitle_label.setObjectName("subtitleLabel")

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()
        outer_layout.addLayout(header_layout)

        # ====== 分隔线 ======
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #e8ecf1;")
        sep.setFixedHeight(1)
        outer_layout.addWidget(sep)

        # ====== 三栏卡片区域 ======
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # 第一组 — 选择图片
        card1, self.image_label1, self.button1 = self._make_card(
            "① 原始图片", "请选择一张图片", "📂  选择图片", "btnSelect", self.choose_image)
        cards_layout.addWidget(card1)

        # 第二组 — 检测结果
        card2, self.image_label2, self.button2 = self._make_card(
            "② 检测结果", "检测结果将显示在这里", "🔍  开始检测", "btnDetect", self.Predict)
        cards_layout.addWidget(card2)

        # 第三组 — OCR 识别
        card3, self.image_label3, self.button3 = self._make_card(
            "③ OCR 识别", "识别信息将显示在这里", "📝  文字识别", "btnOCR", self.information_ocr)
        # OCR 结果主要是文字，不需要拉伸图片
        self.image_label3.setScaledContents(False)
        self.image_label3.setWordWrap(True)
        cards_layout.addWidget(card3)

        outer_layout.addLayout(cards_layout, 1)

        self.setLayout(outer_layout)

    def choose_image(self):
        """选择图片并显示"""
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', "Image files (*.jpg *.gif *.png *.jpeg)")

        if fname:
            self.current_image_path = fname

            # 直接在 self.image_label1 上显示
            pixmap = QPixmap(fname)
            if not pixmap.isNull():
                self.image_label1.setPixmap(pixmap)
                print(f"已加载图片: {fname}")

    def Predict(self):
        """调用预测函数并显示结果"""
        if hasattr(self, 'current_image_path'):
            try:
                # 使用预加载的模型进行推理，避免重复加载
                self.results = self.model(self.current_image_path, save=False)

                if len(self.results) > 0:
                    # 获取绘制了框的图像 (results[0].plot() 返回的是 BGR numpy 数组)
                    plotted_image = self.results[0].plot()

                    # 转换 BGR 到 RGB
                    res_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)

                    # 转换为 QImage，使用 tobytes() 避免 memoryview 类型问题
                    height, width, channel = res_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(res_rgb.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)

                    pixmap = QPixmap.fromImage(q_image)
                    if not pixmap.isNull():
                        self.image_label2.setPixmap(pixmap)
                        print("已显示检测结果（所有框选）")
                else:
                    print("未检测到目标")
                    self.image_label2.setText("未检测到目标")

            except Exception as e:
                print(f"检测错误: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("请先选择一张图片进行检测")

    def information_ocr(self):
        """调用 OCR 函数并显示结果"""
        if hasattr(self, 'results') and len(self.results) > 0 and len(self.results[0].boxes) > 0:
            try:
                original_image = cv2.imread(self.current_image_path)
                if original_image is None:
                    print("OCR 错误：无法读取原始图片")
                    return

                all_ocr_results = []

                # 清理之前的临时文件
                self._cleanup_temp_files()

                # 遍历所有被检测到的框
                for i, box in enumerate(self.results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # 裁剪图像
                    cropped_image = original_image[y1:y2, x1:x2]

                    # 使用 tempfile 保存临时文件
                    temp_fd, temp_crop_path = tempfile.mkstemp(suffix='.jpg', prefix='crop_')
                    os.close(temp_fd)
                    cv2.imwrite(temp_crop_path, cropped_image)
                    self._temp_files.append(temp_crop_path)

                    # 对每个裁剪区域进行 OCR
                    ocr_res = OCR(temp_crop_path)
                    if ocr_res:
                        all_ocr_results.extend(ocr_res)

                if all_ocr_results:
                    info_text = "\n".join(all_ocr_results)
                    self.image_label3.setText(info_text)
                    print("已显示所有框选区域的 OCR 识别结果")
                else:
                    self.image_label3.setText("未识别到文字")

            except Exception as e:
                print(f"OCR 错误: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("请先进行检测或未检测到目标")

    def _cleanup_temp_files(self):
        """清理之前生成的临时裁剪文件"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        self._temp_files.clear()

    def closeEvent(self, event):
        """窗口关闭时清理临时文件"""
        self._cleanup_temp_files()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    # 应用全局样式表
    app.setStyleSheet(GLOBAL_STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

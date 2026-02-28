import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QSpacerItem, QSizePolicy, QFileDialog)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont, QPixmap, QImage
from Predict import predict_image
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("欢迎界面")
        self.resize(1000, 600)

        # 隐藏系统标题栏，实现无边框窗口
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)

        # 启用鼠标追踪，以便在不按键时也能捕获 resize 区域
        self.setup_ui()

    def setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10) # 增加一点外边距以显示阴影（如果需要）

        # 背景容器
        self.background_frame = QFrame(self)
        self.background_frame.setObjectName("backgroundFrame")
        # CSS 样式: 淡白色背景，简约风
        self.background_frame.setStyleSheet("""
            QFrame#backgroundFrame {
                background-color: #f7f9fc;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
            }
        """)

        # 背景内部布局
        content_layout = QVBoxLayout(self.background_frame)
        content_layout.setContentsMargins(30, 20, 30, 30)

        # 1. 顶部控制栏 (最大化、关闭按钮)
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addStretch()

        # 按钮通用样式
        btn_style = """
            QPushButton {
                color: #555;
                background: transparent;
                border: none;
                font-size: 18px;
                font-family: Arial;
                font-weight: bold;
                border-radius: 15px;
            }
            QPushButton:hover { background-color: rgba(0, 0, 0, 0.05); }
        """

        # 关闭按钮样式比较特殊（红色背景）
        close_btn_style = """
            QPushButton {
                color: #555;
                background: transparent;
                border: none;
                font-size: 18px;
                font-weight: bold;
                border-radius: 15px;
            }
            QPushButton:hover { background-color: #ff5c5c; color: white; }
        """

        # 最大化/还原按钮


        # 2. 标题 "欢迎使用"
        title_label = QLabel("智能识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        # 深色字体
        title_label.setStyleSheet("color: #333333; font-weight: bold; letter-spacing: 2px;")
        font = QFont("Microsoft YaHei", 36, QFont.Bold)
        title_label.setFont(font)
        content_layout.addWidget(title_label)

        content_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 3. 中间三个卡片区域
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(40)
        cards_layout.setContentsMargins(20, 20, 20, 20)

        # 定义卡片图标
        cards_data = ["image/pic1.png", "image/pic2.png", "image/pic3.png"] # 假设有图片，这里还是用符号
        cards_symbols = ["⊞", "⚡", "☁"]

        self.card_widgets = []

        for i, icon_text in enumerate(cards_symbols):
            card = QFrame()
            # 卡片样式：纯白背景，轻微阴影效果
            card.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 20px;
                    border: 1px solid #eaeff5;
                }
                QFrame:hover {
                    border: 1px solid #d0d7de;
                    background-color: #ffffff;
                }
            """)
            # card.setFixedHeight(220)  <-- Delete or comment this out to allow flexibility
            card.setMinimumHeight(220) # Use minimum height instead
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow expansion

            # 阴影效果可以通过 QGraphicsDropShadowEffect 实现，但这里用 CSS 模拟简单的层次感

            card_layout = QVBoxLayout(card)

            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            # 图片显示时的样式：保留圆角
            image_label.setStyleSheet("border: none; background: transparent;")

            icon_label = QLabel(icon_text)
            icon_label.setAlignment(Qt.AlignCenter)
            # 图标样式：淡色背景深色图标
            icon_label.setStyleSheet("""
                color: #5c6bc0;
                background-color: #f0f2ff;
                border-radius: 12px;
                border: none;
            """)
            icon_label.setFont(QFont("Segoe UI Symbol", 28))
            icon_label.setFixedSize(64, 64)

            inner_layout = QVBoxLayout()
            inner_layout.addWidget(image_label)
            inner_layout.addWidget(icon_label, 0, Qt.AlignCenter)

            image_label.hide()
            image_label.setScaledContents(True) # 让图片自适应 Label 大小

            card_layout.addLayout(inner_layout)
            cards_layout.addWidget(card)

            self.card_widgets.append((card, icon_label, image_label, inner_layout))

        content_layout.addLayout(cards_layout)
        content_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 4. 底部菜单项
        menu_items = ["选择图片", "开始裁剪", "信息识别"]
        menu_layout = QHBoxLayout() # 改为横向布局可能更美观，或者保持纵向但增加样式
        menu_layout.setSpacing(30)
        menu_layout.setContentsMargins(50, 0, 50, 20)

        # 创建按钮
        self.btn_select_img = QPushButton(menu_items[0])
        self.btn_crop = QPushButton(menu_items[1])
        self.btn_ocr = QPushButton(menu_items[2])

        buttons = [self.btn_select_img, self.btn_crop, self.btn_ocr]

        # 按钮样式：实心胶囊按钮
        btn_css = """
            QPushButton {
                background-color: #5c6bc0;
                color: white;
                border-radius: 20px;
                padding: 10px 30px;
                font-family: "Microsoft YaHei";
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f5cb3;
            }
            QPushButton:pressed {
                background-color: #3e4c9b;
            }
        """

        for btn in buttons:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(btn_css)
            menu_layout.addWidget(btn)

        # 绑定点击事件
        self.btn_select_img.clicked.connect(self.choose_image)
        self.btn_crop.clicked.connect(self.start_crop)

        content_layout.addLayout(menu_layout)
        content_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        main_layout.addWidget(self.background_frame)

    def toggle_maximize(self):
        if self.is_maximized:
            self.showNormal()
            self.max_btn.setText("□")
            self.is_maximized = False
            self.background_frame.setStyleSheet("""
                QFrame#backgroundFrame {
                    background-color: #f7f9fc;
                    border-radius: 15px;
                    border: 1px solid #e0e0e0;
                }
            """)
            self.layout().setContentsMargins(10, 10, 10, 10)
        else:
            self.showMaximized()
            self.max_btn.setText("❐")
            self.is_maximized = True
            # 全屏时去掉圆角和边距
            self.background_frame.setStyleSheet("""
                QFrame#backgroundFrame {
                    background-color: #f7f9fc;
                    border-radius: 0px;
                    border: none;
                }
            """)
            self.layout().setContentsMargins(0, 0, 0, 0)

    def choose_image(self):
        """选择图片并显示在第一个框中"""
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', "Image files (*.jpg *.gif *.png *.jpeg)")

        if fname:
            self.current_image_path = fname
            card_frame, icon_label, image_label, layout = self.card_widgets[0]

            pixmap = QPixmap(fname)
            if not pixmap.isNull():
                # 设置图片到 Label，由 setScaledContents 处理缩放
                # scaled_pixmap = pixmap.scaled(...) # 不再需要手动缩放
                image_label.setPixmap(pixmap)

                image_label.show()
                icon_label.hide()

            print(f"Loaded image: {fname}")

    def start_crop(self):
        """开始裁剪（预测）并显示在第二个框中"""
        if hasattr(self, 'current_image_path') and self.current_image_path:
            try:
                model_path = "yolo11n.pt"
                results = predict_image(self.current_image_path, model_path=model_path, save=False, show=False)

                # Plot results on the original image
                res_plotted = results[0].plot()

                # Convert BGR (OpenCV) to RGB (Qt)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                height, width, channel = res_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(res_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                card_frame, icon_label, image_label, layout = self.card_widgets[1]

                # Display the image
                pixmap = QPixmap.fromImage(q_img)
                if not pixmap.isNull():
                    # Set pixmap directly, let scaledContents handle resizing
                    image_label.setPixmap(pixmap)

                    image_label.show()
                    icon_label.hide()
                    print("Prediction displayed in second card.")

            except Exception as e:
                print(f"Error during prediction: {e}")
        else:
            print("Please select an image first.")

    # 允许拖动无边框窗口 (仅在非全屏模式下) 以及调整大小
    def mousePressEvent(self, event):
        if self.is_maximized:
            return

        if event.button() == Qt.LeftButton:
            self.m_drag = True
            self.m_DragPosition = event.globalPos() - self.pos()

            # Check if we are in a resize area
            self.resize_area = self.get_resize_area(event.pos())
            if self.resize_area:
                self.m_drag = False # We are resizing, not moving
                self.resize_start_pos = event.globalPos()
                self.resize_start_geometry = self.geometry()

            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_maximized:
            return

        # Update cursor shape based on position
        if not event.buttons() & Qt.LeftButton:
            area = self.get_resize_area(event.pos())
            self.set_cursor_shape(area)
            return

        # Handle resizing
        if hasattr(self, 'resize_area') and self.resize_area:
            delta = event.globalPos() - self.resize_start_pos
            geo = self.resize_start_geometry
            new_geo = self.calculate_new_geometry(geo, delta, self.resize_area)
            self.setGeometry(new_geo)
            event.accept()

        # Handle moving
        elif hasattr(self, 'm_drag') and self.m_drag and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.m_DragPosition)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.m_drag = False
        self.resize_area = None
        self.setCursor(Qt.ArrowCursor)

    def get_resize_area(self, pos):
        """Determine which edge/corner the mouse is on"""
        margin = 10
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()

        left = x < margin
        right = x > w - margin
        top = y < margin
        bottom = y > h - margin

        if top and left: return "top_left"
        if top and right: return "top_right"
        if bottom and left: return "bottom_left"
        if bottom and right: return "bottom_right"
        if top: return "top"
        if bottom: return "bottom"
        if left: return "left"
        if right: return "right"
        return None

    def set_cursor_shape(self, area):
        if area == "top_left" or area == "bottom_right":
            self.setCursor(Qt.SizeFDiagCursor)
        elif area == "top_right" or area == "bottom_left":
            self.setCursor(Qt.SizeBDiagCursor)
        elif area == "top" or area == "bottom":
            self.setCursor(Qt.SizeVerCursor)
        elif area == "left" or area == "right":
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def calculate_new_geometry(self, geo, delta, area):
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        dx, dy = delta.x(), delta.y()

        min_w, min_h = 600, 400 # Minimum size

        if "top" in area:
            new_y = y + dy
            new_h = h - dy
            if new_h > min_h:
                y = new_y
                h = new_h
        elif "bottom" in area:
            h += dy

        if "left" in area:
            new_x = x + dx
            new_w = w - dx
            if new_w > min_w:
                x = new_x
                w = new_w
        elif "right" in area:
            w += dx

        return QRect(x, y, max(w, min_w), max(h, min_h))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

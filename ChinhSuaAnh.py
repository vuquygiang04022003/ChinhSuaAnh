import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QSlider, QPushButton,
    QWidget, QGridLayout, QFileDialog, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Điều chỉnh ảnh")
        self.setGeometry(100, 100, 1200, 800)

        # Biến lưu ảnh gốc và ảnh đã chỉnh sửa
        self.original_image = None
        self.processed_image = None
        self.edge_image = None

        # Giao diện chính
        self.init_ui()

    def init_ui(self):
        main_layout = QGridLayout()

        # Thiết lập các thành phần giao diện
        self.original_image_label = QLabel("Ảnh gốc")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setFixedSize(400, 300)

        self.processed_image_label = QLabel("Ảnh đã chỉnh sửa")
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setFixedSize(400, 300)

        self.edge_image_label = QLabel("Ảnh phát hiện biên")
        self.edge_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edge_image_label.setFixedSize(400, 300)

        # Slider điều chỉnh độ tương phản
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 50)
        self.contrast_slider.setValue(20)  # Giá trị mặc định
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)

        # Nút chọn ảnh
        self.load_button = QPushButton("Chọn ảnh")
        self.load_button.clicked.connect(self.load_image)

        # Nút lưu ảnh
        self.save_button = QPushButton("Lưu ảnh")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        # Slider điều chỉnh ngưỡng dưới và ngưỡng trên cho Canny
        self.lower_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.lower_threshold_slider.setRange(0, 255)
        self.lower_threshold_slider.setValue(50)
        self.lower_threshold_slider.valueChanged.connect(self.detect_edges)

        self.upper_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.upper_threshold_slider.setRange(0, 255)
        self.upper_threshold_slider.setValue(150)
        self.upper_threshold_slider.valueChanged.connect(self.detect_edges)

        # Nhập góc để xoay ảnh
        self.rotate_angle_input = QLineEdit()
        self.rotate_angle_input.setPlaceholderText("Nhập góc xoay")

        # Nút xoay ảnh
        self.rotate_button = QPushButton("Xoay ảnh")
        self.rotate_button.clicked.connect(self.rotate_image)

        # Biểu đồ histogram
        self.histogram_canvas = FigureCanvas(plt.figure(figsize=(4, 2)))

        # Slider điều chỉnh độ nhiễu
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)  # Mặc định không có nhiễu
        self.noise_slider.valueChanged.connect(self.add_noise)

        # Slider điều chỉnh độ sắc nét
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 10)
        self.sharpness_slider.setValue(1)  # Mặc định độ sắc nét là 1 (không thay đổi)
        self.sharpness_slider.valueChanged.connect(self.sharpen_image)

        # Slider điều chỉnh độ sáng
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)  # Giá trị từ -100 đến 100
        self.brightness_slider.setValue(0)  # Giá trị mặc định
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)

        # Bố cục giao diện
        main_layout.addWidget(self.original_image_label, 0, 0, 1, 1)
        main_layout.addWidget(self.processed_image_label, 0, 1, 1, 1)
        main_layout.addWidget(self.edge_image_label, 0, 2, 1, 1)

        main_layout.addWidget(self.load_button, 1, 0, 1, 1)
        main_layout.addWidget(self.save_button, 1, 1, 1, 1)

        main_layout.addWidget(QLabel("Điều chỉnh độ tương phản"), 2, 0, 1, 1)
        main_layout.addWidget(self.contrast_slider, 2, 1, 1, 2)

        # Các slider điều chỉnh ngưỡng Canny
        main_layout.addWidget(QLabel("Ngưỡng dưới Canny"), 3, 0)
        main_layout.addWidget(self.lower_threshold_slider, 3, 1, 1, 2)

        main_layout.addWidget(QLabel("Ngưỡng trên Canny"), 4, 0)
        main_layout.addWidget(self.upper_threshold_slider, 4, 1, 1, 2)

        # Nhập thông số cho xoay ảnh
        main_layout.addWidget(QLabel("Xoay ảnh (Góc xoay)"), 5, 0, 1, 1)
        main_layout.addWidget(self.rotate_angle_input, 5, 1, 1, 2)

        main_layout.addWidget(self.rotate_button, 6, 0, 1, 3)

        # Các slider điều chỉnh độ nhiễu và độ sắc nét
        main_layout.addWidget(QLabel("Điều chỉnh độ nhiễu"), 7, 0)
        main_layout.addWidget(self.noise_slider, 7, 1, 1, 2)

        main_layout.addWidget(QLabel("Điều chỉnh độ sắc nét"), 8, 0)
        main_layout.addWidget(self.sharpness_slider, 8, 1, 1, 2)

        # Slider điều chỉnh độ sáng
        main_layout.addWidget(QLabel("Điều chỉnh độ sáng"), 9, 0)
        main_layout.addWidget(self.brightness_slider, 9, 1, 1, 2)

        # Biểu đồ histogram
        main_layout.addWidget(self.histogram_canvas, 10, 0, 1, 3)

        # Chèn layout vào cửa sổ chính
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Đặt màu sắc và phong cách
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QLabel {
                border: 1px solid black;
                background-color: #F8F8F8;
                color: #333333;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: #ddd;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -2px 0;
                border-radius: 7px;
            }
            QLineEdit {
                border: 1px solid #888;
                padding: 4px;
                font-size: 14px;
                border-radius: 3px;
            }
        """)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()

            self.display_image(self.original_image, self.original_image_label)
            self.display_image(self.processed_image, self.processed_image_label)

            self.update_histogram(self.original_image, self.processed_image)
            self.save_button.setEnabled(True)

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def adjust_contrast(self):
        if self.original_image is not None:
            contrast = self.contrast_slider.value() / 10
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=contrast, beta=0)
            self.display_image(self.processed_image, self.processed_image_label)

    def detect_edges(self):
        if self.processed_image is not None:
            lower_threshold = self.lower_threshold_slider.value()
            upper_threshold = self.upper_threshold_slider.value()
            gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            edge_overlay = self.processed_image.copy()
            edge_overlay[edges > 0] = [255, 0, 0]
            self.display_image(edge_overlay, self.edge_image_label)

    def add_noise(self):
        if self.original_image is not None:
            noise_level = self.noise_slider.value() / 100
            noise = np.random.normal(0, 25 * noise_level, self.original_image.shape).astype(np.uint8)
            self.processed_image = cv2.add(self.original_image, noise)
            self.display_image(self.processed_image, self.processed_image_label)

    def sharpen_image(self):
        if self.original_image is not None:
            strength = self.sharpness_slider.value()
            kernel = np.array([[0, -1, 0], [-1, 5 + strength, -1], [0, -1, 0]])
            self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
            self.display_image(self.processed_image, self.processed_image_label)

    def adjust_brightness(self):
        if self.original_image is not None:
            brightness = self.brightness_slider.value()
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=1, beta=brightness)
            self.display_image(self.processed_image, self.processed_image_label)

    def rotate_image(self):
        if self.processed_image is not None:
            try:
                angle = float(self.rotate_angle_input.text())
                (h, w) = self.processed_image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(self.processed_image, rotation_matrix, (w, h))
                self.processed_image = rotated
                self.display_image(self.processed_image, self.processed_image_label)
            except ValueError:
                self.rotate_angle_input.setPlaceholderText("Vui lòng nhập số hợp lệ!")

    def display_image(self, image, label):
        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_histogram(self, original, processed):
        if original is not None and processed is not None:
            self.histogram_canvas.figure.clear()

            # Histogram ảnh xám
            gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            gray_processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

            ax = self.histogram_canvas.figure.add_subplot(121)
            ax.hist(gray_original.ravel(), bins=256, color='blue', alpha=0.5, label='Grayscale Original')
            ax.hist(gray_processed.ravel(), bins=256, color='green', alpha=0.5, label='Grayscale Processed')
            ax.set_title("Grayscale Histogram")
            ax.legend()

            # Histogram ảnh màu
            bx = self.histogram_canvas.figure.add_subplot(122)
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                bx.hist(original[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=f'Original {color.upper()}')
                bx.hist(processed[:, :, i].ravel(), bins=256, color=color, alpha=0.3, linestyle='dashed', label=f'Processed {color.upper()}')
            bx.set_title("RGB Histogram")
            bx.legend()

            self.histogram_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())

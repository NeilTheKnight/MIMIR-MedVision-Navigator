# main_window_final.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QListWidget, QTextBrowser, QFrame, QListWidgetItem,
    QPushButton, QMenu
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPainter, QPen, QIcon, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QSize, QPoint

import numpy as np
import random

# 确保项目根目录在 sys.path，便于导入 back 包和 config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import PATHS

# 引入后端处理线程（back/），缩放控件仍来自前端的 main_window
try:
    from back.processor_thread import ImageProcessorThread as TumorImageProcessorThread
except Exception:
    TumorImageProcessorThread = None

# 在本文件内定义缩放控件 ZoomableImageLabel


# --- 后端模拟（保持不变）---
class ImageProcessorThread(QThread):
    images_processed = pyqtSignal(dict)
    analysis_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        image_data = self._simulate_image_loading()
        analysis_text = self._simulate_analysis(image_data)
        self.images_processed.emit(image_data)
        self.analysis_ready.emit(analysis_text)

    def _simulate_image_loading(self):
        dummy_images = {
            "A": np.random.normal(128, 40, (300, 300)).astype(np.uint8),
            "B": np.random.normal(128, 40, (300, 300)).astype(np.uint8),
            "C": np.random.normal(128, 40, (300, 300)).astype(np.uint8),
            "D": np.random.normal(128, 40, (300, 300)).astype(np.uint8),
        }
        if random.random() > 0.5:
            center_x, center_y = random.randint(100, 200), random.randint(100, 200)
            radius = random.randint(20, 40)
            Y, X = np.ogrid[:300, :300]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)** 2)
            mask = dist_from_center < radius
            dummy_images["B"][mask] = np.clip(dummy_images["B"][mask] + 80, 0, 255)
        return dummy_images

    def _simulate_analysis(self, image_data):
        analysis = "Image Analysis & Statistics\n\n"
        analysis += "No tumors detected.\nConfidence: 99.8%\n\n"
        if random.random() > 0.3:
            tumor1_x = random.randint(50, 200)
            tumor1_y = random.randint(50, 200)
            tumor1_size = round(random.uniform(20.0, 50.0), 1)
            analysis += f"Tumor 1:\nCoordinates: x={tumor1_x}, y={tumor1_y}\n" \
                        f"Size: {tumor1_size} mm\nType: Lung Nodule\nConfidence: {round(random.uniform(90.0, 98.0), 1)}%\n\n"
        else:
             analysis += "No abnormalities in Chest CT.\nConfidence: 95.0%\n\n"
        if random.random() > 0.5:
            analysis += "Tumor 2:\nSize: Liver Lesion\nConfidence: {round(random.uniform(85.0, 95.0), 1)}%\n\n"
        else:
            analysis += "No abnormalities in Whole Body CT.\nConfidence: 96.2%\n\n"
        analysis += "No findings in Lung Windows.\nConfidence: 99.0%\n"
        return analysis


class QwenAnalyzerThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path: str, question: str = None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.question = question

    def run(self):
        try:
            # 运行时导入，避免启动失败时影响主界面
            from back.llm_analyzer import analyze_image_with_qwen
            result = analyze_image_with_qwen(self.image_path, self.question)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

# --- 缩放与拖拽控件（加入到本文件）---
class ZoomableImageLabel(QLabel):
    """支持放大缩小、拖拽移动和坐标显示的图像标签"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.base_scale_factor = 1.0  # 基础缩放因子（适应标签大小）
        self.user_scale_factor = 1.0  # 用户缩放因子（滚轮控制）
        self.min_scale = 0.1
        self.max_scale = 10.0

        # 拖拽相关
        self.offset_x = 0  # 图像偏移X
        self.offset_y = 0  # 图像偏移Y
        self.is_dragging = False
        self.drag_start_pos = QPoint()
        self.last_offset = QPoint(0, 0)

        # 坐标显示
        self.current_mouse_pos = QPoint()
        self.show_coordinates = True

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.setScaledContents(False)  # 不使用自动缩放，手动控制
        self.setMouseTracking(True)  # 启用鼠标跟踪，即使不按下也能跟踪

    def setPixmap(self, pixmap):
        """设置原始图像"""
        self.original_pixmap = pixmap
        self.user_scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_offset = QPoint(0, 0)

        # 计算基础缩放因子，使图像适应标签大小
        if pixmap and not pixmap.isNull():
            label_size = self.size()
            pixmap_size = pixmap.size()

            # 计算适应标签大小的缩放比例
            scale_w = label_size.width() / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_h = label_size.height() / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            self.base_scale_factor = min(scale_w, scale_h) * 0.95  # 留一点边距

        self._update_display()

    def _update_display(self):
        """更新显示的图像"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        # 计算总缩放因子
        total_scale = self.base_scale_factor * self.user_scale_factor

        # 计算缩放后的尺寸
        original_size = self.original_pixmap.size()
        scaled_width = int(original_size.width() * total_scale)
        scaled_height = int(original_size.height() * total_scale)

        # 缩放图像
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 触发重绘
        self.update()

    def paintEvent(self, event):
        """重写绘制事件，支持偏移绘制"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), Qt.black)

        if self.scaled_pixmap is None or self.scaled_pixmap.isNull():
            return

        # 计算绘制位置（居中 + 偏移）
        label_size = self.size()
        pixmap_size = self.scaled_pixmap.size()

        # 居中位置
        center_x = (label_size.width() - pixmap_size.width()) // 2
        center_y = (label_size.height() - pixmap_size.height()) // 2

        # 应用偏移
        draw_x = center_x + self.offset_x
        draw_y = center_y + self.offset_y

        # 绘制图像
        painter.drawPixmap(draw_x, draw_y, self.scaled_pixmap)

        # 绘制坐标信息（如果鼠标在图像区域内）
        if self.show_coordinates and self.current_mouse_pos:
            self._draw_coordinates(painter)

    def _draw_coordinates(self, painter):
        """在图像上绘制坐标信息"""
        if self.scaled_pixmap is None or self.scaled_pixmap.isNull():
            return

        label_size = self.size()
        pixmap_size = self.scaled_pixmap.size()
        center_x = (label_size.width() - pixmap_size.width()) // 2
        center_y = (label_size.height() - pixmap_size.height()) // 2

        # 检查鼠标是否在图像区域内
        mouse_x = self.current_mouse_pos.x()
        mouse_y = self.current_mouse_pos.y()

        image_left = center_x + self.offset_x
        image_right = image_left + pixmap_size.width()
        image_top = center_y + self.offset_y
        image_bottom = image_top + pixmap_size.height()

        if image_left <= mouse_x <= image_right and image_top <= mouse_y <= image_bottom:
            # 计算在原始图像中的坐标
            total_scale = self.base_scale_factor * self.user_scale_factor
            original_x = int((mouse_x - image_left) / total_scale)
            original_y = int((mouse_y - image_top) / total_scale)

            # 限制坐标范围
            if self.original_pixmap:
                original_size = self.original_pixmap.size()
                original_x = max(0, min(original_x, original_size.width() - 1))
                original_y = max(0, min(original_y, original_size.height() - 1))

            # 绘制坐标文本
            coord_text = f"X: {original_x}, Y: {original_y}"
            painter.setPen(Qt.yellow)
            painter.setFont(self.font())

            # 在鼠标位置附近绘制，避免超出边界
            text_x = min(mouse_x + 10, label_size.width() - 100)
            text_y = max(mouse_y - 10, 20)

            # 绘制背景矩形以提高可读性
            text_rect = painter.fontMetrics().boundingRect(coord_text)
            text_rect.moveTo(text_x, text_y - text_rect.height())
            bg_rect = text_rect.adjusted(-2, -2, 2, 2)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 180))  # 半透明黑色背景
            painter.drawText(text_x, text_y, coord_text)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件：缩放图像"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        # 获取滚轮增量
        delta = event.angleDelta().y()

        # 计算缩放因子（滚轮向上放大，向下缩小）
        if delta > 0:
            self.user_scale_factor *= 1.15
        else:
            self.user_scale_factor *= 0.85

        # 限制缩放范围
        self.user_scale_factor = max(self.min_scale, min(self.max_scale, self.user_scale_factor))

        # 更新显示
        self._update_display()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件：开始拖拽"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            self.last_offset = QPoint(self.offset_x, self.offset_y)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件：拖拽图像或显示坐标"""
        self.current_mouse_pos = event.pos()

        if self.is_dragging and event.buttons() & Qt.LeftButton:
            # 计算拖拽偏移
            delta = event.pos() - self.drag_start_pos
            self.offset_x = self.last_offset.x() + delta.x()
            self.offset_y = self.last_offset.y() + delta.y()
            self.update()  # 触发重绘
        else:
            # 仅更新坐标显示
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件：结束拖拽"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """双击事件：重置缩放和位置"""
        if event.button() == Qt.LeftButton:
            self.reset_view()

    def contextMenuEvent(self, event):
        """右键菜单事件：重置视图"""
        menu = QMenu(self)
        reset_action = menu.addAction("重置视图 (Reset View)")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == reset_action:
            self.reset_view()

    def reset_view(self):
        """重置缩放和位置到初始状态"""
        self.user_scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_offset = QPoint(0, 0)
        self._update_display()

    def leaveEvent(self, event):
        """鼠标离开事件：清除坐标显示"""
        self.current_mouse_pos = QPoint()
        self.update()

    def resizeEvent(self, event):
        """窗口大小改变时，重新计算基础缩放因子"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            label_size = self.size()
            pixmap_size = self.original_pixmap.size()

            scale_w = label_size.width() / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_h = label_size.height() / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            self.base_scale_factor = min(scale_w, scale_h) * 0.95

            self._update_display()

        super().resizeEvent(event)

# --- 仪表盘组件（适配右侧宽度）---
class GaugeWidget(QWidget):
    def __init__(self, title, unit, value, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.value = value
        self.setFixedSize(90, 90)
        self.setStyleSheet("background-color: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setPen(QPen(QColor(40, 60, 90), 3))
        painter.drawEllipse(rect)
        start_angle = 90 * 16
        span_angle = int((self.value / 200.0) * 360 * 16)
        painter.setPen(QPen(QColor(0, 150, 255), 5, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, start_angle, -span_angle)
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(self.value))
        font.setPointSize(7)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(150, 180, 255))
        painter.drawText(rect.adjusted(0, 20, 0, 0), Qt.AlignHCenter | Qt.AlignTop, self.unit)
        painter.drawText(rect.adjusted(0, -30, 0, 0), Qt.AlignHCenter | Qt.AlignBottom, self.title)

# --- 主窗口 ---
class MedicalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医疗健康可视化 - Medical health visualization")
        self.setGeometry(100, 100, 1440, 900)  # 调整为1440×900

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        # 列宽比例保持：左侧2份，中间3份，右侧1份（适配1440宽度）
        self.main_layout.setColumnStretch(0, 2)
        self.main_layout.setColumnStretch(1, 3)
        self.main_layout.setColumnStretch(2, 1)

        self._init_styles()
        self._init_ui()
        self._load_initial_data()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)
        self._update_time()

    def _init_styles(self):
        self.bg_color = "#0c1524"
        self.panel_bg_color = "#1a253a"
        self.accent_color_blue = "#00c0ff"
        self.accent_color_green = "#00ff80"
        self.text_color_light = "#e0e0e0"
        self.text_color_dim = "#90a0b0"

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {self.bg_color}; }}
            QWidget {{
                background-color: {self.bg_color};
                color: {self.text_color_light};
                font-family: 'Segoe UI', sans-serif;
            }}
            .PanelFrame {{
                background-color: {self.panel_bg_color};
                border-radius: 8px;
                padding: 10px;
            }}
            QLabel.HeaderTitle {{ font-size: 16pt; font-weight: bold; color: {self.accent_color_blue}; }}
            QLabel {{ color: {self.text_color_light}; }}
            QTextBrowser {{
                background-color: #0d1a2f;
                border: 1px solid #3a4a60;
                border-radius: 5px;
                color: {self.accent_color_green};
                font-family: 'Consolas';
                font-size: 9pt;
                padding: 8px;
            }}
            QListWidget {{
                background-color: transparent;
                border: 1px solid #3a4a60;
                border-radius: 5px;
                color: {self.text_color_light};
                font-size: 10pt;
            }}
            QListWidget::item {{ padding: 6px; border-bottom: 1px solid #3a4a60; }}
            QListWidget::item:last-child {{ border-bottom: none; }}
        """)

    def _create_panel(self):
        panel = QFrame()
        panel.setProperty("class", "PanelFrame")
        return panel

    def _init_ui(self):
        # --- 1. 顶部标题栏（横跨所有列）---
        header_panel = QWidget()
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(0,0,0,0)
        title_label = QLabel("医疗健康可视化  Medical health visualization")
        title_label.setProperty("class", "HeaderTitle")
        header_layout.addWidget(title_label, alignment=Qt.AlignLeft)
        header_layout.addStretch()
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {self.accent_color_blue}; font-size: 12pt; font-weight: bold;")
        header_layout.addWidget(self.time_label, alignment=Qt.AlignRight)
        self.main_layout.addWidget(header_panel, 0, 0, 1, 3)  # 行0，跨3列


        # --- 2. 左侧列（2/5宽度）---
        # 2.1 患者基本信息 + 过往病史面板（上半部分）
        patient_info_panel = self._create_panel()
        patient_info_layout = QVBoxLayout(patient_info_panel)
        # 顶部左侧占据两行（共四行：1-4），与CT面板各占一半
        self.main_layout.addWidget(patient_info_panel, 1, 0, 2, 1)

        # 患者基本信息
        patient_name = QLabel("患者姓名: Hoppe, Melany")
        patient_name.setStyleSheet("font-size: 16pt; font-weight: bold; color: white;")
        patient_id = QLabel("病例号: TCGA-50-5072")
        patient_id.setStyleSheet(f"font-size: 10pt; color: {self.text_color_dim};")
        patient_info_layout.addWidget(patient_name)
        patient_info_layout.addWidget(patient_id)
        patient_info_layout.addSpacing(15)

        basic_info_label = QLabel("基本信息")
        basic_info_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        basic_info_grid = QGridLayout()
        self._add_info_row(basic_info_grid, "性别", "男", 0)
        self._add_info_row(basic_info_grid, "民族", "汉族", 1)
        self._add_info_row(basic_info_grid, "年龄", "28", 2)
        self._add_info_row(basic_info_grid, "体重", "65KG", 3)
        patient_info_layout.addWidget(basic_info_label)
        patient_info_layout.addLayout(basic_info_grid)
        patient_info_layout.addSpacing(20)

        # 过往病史
        history_label = QLabel("过往病史")
        history_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        history_list = QListWidget()
        histories = [
            "2020-03: 高血压（轻度）",
            "2021-07: 肺炎（已治愈）",
            "2023-01: 体检发现肺结节（3mm）"
        ]
        for h in histories:
            item = QListWidgetItem(h)
            item.setSizeHint(item.sizeHint() + QSize(0, 10))
            history_list.addItem(item)
        patient_info_layout.addWidget(history_list)
        patient_info_layout.addStretch()


        # 2.2 CT Image面板（左下区域：改为可选择的图片列表，读取test_images）
        ct_panel = self._create_panel()
        ct_layout = QVBoxLayout(ct_panel)
        # 底部左侧占据两行
        self.main_layout.addWidget(ct_panel, 3, 0, 2, 1)

        ct_title = QLabel("MRI Image")
        ct_title.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 15px;")
        ct_layout.addWidget(ct_title)

        self.ct_image_list = QListWidget()
        self.ct_image_list.setStyleSheet(f"background-color: transparent; border: 1px solid #3a4a60; color: {self.text_color_light};")
        self.ct_image_list.setIconSize(QSize(160, 100))
        self.ct_image_list.itemClicked.connect(self._on_ct_image_selected)
        ct_layout.addWidget(self.ct_image_list)
        ct_layout.addStretch()


        # --- 3. 中间列（3/5宽度，核心图像区域，缩小尺寸解决过长问题）---
        self.image_grid_panel = self._create_panel()
        image_grid_layout = QGridLayout(self.image_grid_panel)
        self.main_layout.addWidget(self.image_grid_panel, 1, 1, 4, 1)  # 从顶到底

        self.image_labels = {}
        # 图像标题顺序保持不变，调整尺寸
        self.image_titles = {
            "A": "  ",
            "B": "  ",
            "C": "  ",
            "D": "  "
        }
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2×2网格
        for i, (key, title) in enumerate(self.image_titles.items()):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5,5,5,5)
            
            title_label = QLabel(f"{key} {title}")
            title_label.setStyleSheet(f"color: {self.text_color_dim}; font-size: 9pt;")
            container_layout.addWidget(title_label, alignment=Qt.AlignLeft)
            
            if ZoomableImageLabel is not None:
                img_label = ZoomableImageLabel()
            else:
                img_label = QLabel("Loading image...")
            img_label.setFixedSize(450, 450)
            img_label.setAlignment(Qt.AlignCenter)
            # 统一使用第二界面的样式
            img_label.setStyleSheet(f"background-color: #0d1a2f; border: 1px solid #3a4a60; color: {self.text_color_dim};")
            self.image_labels[key] = img_label
            container_layout.addWidget(img_label)
            
            image_grid_layout.addWidget(container, positions[i][0], positions[i][1])


        # --- 4. 右侧列（1/5宽度，包含上下两个面板）---
        right_panel = self._create_panel()
        right_layout = QVBoxLayout(right_panel)
        self.main_layout.addWidget(right_panel, 1, 2, 4, 1)

        # 4.1 上方：图像分析与统计（占右侧上半部分）
        stats_panel = QFrame()
        stats_layout = QVBoxLayout(stats_panel)
        stats_panel.setLayout(stats_layout)

        stats_title = QLabel("Image Analysis & Statistics")
        stats_title.setStyleSheet("font-size: 11pt; font-weight: bold; margin-bottom: 8px;")
        self.report_browser = QTextBrowser()

        stats_layout.addWidget(stats_title)
        stats_layout.addWidget(self.report_browser)

        # 4.2 下方：AI语义分析报告（占右侧下半部分）
        ai_report_panel = QFrame()
        ai_report_layout = QVBoxLayout(ai_report_panel)
        ai_report_panel.setLayout(ai_report_layout)

        ai_title = QLabel("AI Detailed Report")
        ai_title.setStyleSheet("font-size: 11pt; font-weight: bold; margin-bottom: 8px;")
        self.analysis_text_browser = QTextBrowser()
        self.analysis_text_browser.setText("点击下方按钮，对当前叠加图进行语义分析。")
        self.analyze_button = QPushButton("Analyze Overlay (Qwen-VL)")
        self.analyze_button.setStyleSheet("padding: 6px 10px; font-weight: bold;")
        self.analyze_button.clicked.connect(self._analyze_current_overlay)

        ai_report_layout.addWidget(ai_title)
        ai_report_layout.addWidget(self.analysis_text_browser)
        ai_report_layout.addWidget(self.analyze_button)

        # 将两个面板添加到右侧主布局，并设置等比伸缩（各占一半高度）
        right_layout.addWidget(stats_panel)
        right_layout.addWidget(ai_report_panel)
        right_layout.setStretch(0, 1)
        right_layout.setStretch(1, 1)

    def _add_info_row(self, layout, label_text, value_text, row):
        label = QLabel(label_text + ":")
        label.setStyleSheet(f"font-size: 10pt; color: {self.text_color_dim};")
        value = QLabel(value_text)
        value.setStyleSheet("font-size: 10pt; font-weight: bold; color: white;")
        layout.addWidget(label, row, 0)
        layout.addWidget(value, row, 1)

    def _update_time(self):
        self.time_label.setText(QDateTime.currentDateTime().toString("hh:mm:ss"))

    def _load_initial_data(self):
        # 加载左侧CT图片列表并默认选中第一张触发推理；若无图片则回退模拟
        self._load_ct_image_list()
        if hasattr(self, 'ct_image_list') and self.ct_image_list.count() > 0:
            self.ct_image_list.setCurrentRow(0)
            first_item = self.ct_image_list.item(0)
            self._on_ct_image_selected(first_item)
        else:
            # 如果没有图片，可以保留一个初始状态或提示
            test_image_dir = PATHS["test_images"]
            self.report_browser.setText(f"在 {test_image_dir} 目录中未找到任何图像。")

    def _update_images(self, image_data):
        """更新中间核心网格图像，兼容模拟numpy数组和后端PIL图像。
        支持两种键：A/B/C/D 或 model_input/probability_map/mask/overlay。
        """
        from PIL import Image as PILImage

        # 映射键名到四宫格
        key_map = None
        if set(image_data.keys()) >= {"model_input", "probability_map", "mask", "overlay"}:
            key_map = {
                "model_input": "A",
                "probability_map": "B",
                "mask": "C",
                "overlay": "D",
            }

        for k, v in image_data.items():
            # 选择目标位置键
            target_key = key_map[k] if key_map and k in key_map else k
            if target_key not in self.image_labels:
                continue

            q_img = None
            # numpy灰度或RGB
            if hasattr(v, 'ndim') and hasattr(v, 'shape'):
                arr = v
                if arr.ndim == 2:
                    h, w = arr.shape
                    q_img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    h, w, c = arr.shape
                    q_img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            # PIL图像
            elif isinstance(v, PILImage.Image):
                img = v
                if img.mode == 'L':
                    w, h = img.size
                    q_img = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    w, h = img.size
                    q_img = QImage(img.tobytes(), w, h, w * 3, QImage.Format_RGB888)

            if q_img is None:
                continue

            pixmap = QPixmap.fromImage(q_img)
            self.image_labels[target_key].setPixmap(pixmap)
            self.image_labels[target_key].setText("")
            # 若为叠加图（D），保存临时文件以供语义分析
            if target_key == "D":
                try:
                    tmp_dir = PATHS["test_results"]
                    os.makedirs(tmp_dir, exist_ok=True)
                    tmp_path = os.path.join(tmp_dir, "tmp_overlay.jpg")
                    if isinstance(v, PILImage.Image):
                        v.convert("RGB").save(tmp_path, format="JPEG", quality=92)
                    else:
                        pixmap.save(tmp_path, "JPG")
                    self._current_overlay_path = tmp_path
                except Exception:
                    self._current_overlay_path = None

        # 左侧列表不再使用随机CT图

    def _update_analysis_text(self, text):
        """更新右侧上方的图像分析统计报告。"""
        self.report_browser.setText(text)

    def _load_report(self, img_basename):
        # 从后端目录读取详细报告
        info_dir = PATHS["inference_info"]
        info_file = os.path.join(info_dir, f"{img_basename}_info.txt")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    self.report_browser.setText(f.read())
            except Exception as e:
                self.report_browser.setText(f"错误：无法读取报告文件\nError reading report file: {str(e)}")
        else:
            self.report_browser.setText(
                f"报告文件不存在\nReport file not found:\n{info_file}\n\n图像可能正在处理中...\nImage may still be processing..."
            )

    def _load_ct_image_list(self):
        """读取 test_images，按文件名排序并加载为可选择列表"""
        self.ct_image_list.clear()
        test_image_dir = PATHS["test_images"]
        if not os.path.isdir(test_image_dir):
            return
        files = [f for f in os.listdir(test_image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for name in sorted(files):
            img_path = os.path.join(test_image_dir, name)
            # 创建缩略图图标
            try:
                pix = QPixmap(img_path).scaled(160, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(pix)
            except Exception:
                icon = QIcon()
            item = QListWidgetItem(icon, name)
            item.setData(Qt.UserRole, img_path)
            item.setSizeHint(QSize(180, 110))
            self.ct_image_list.addItem(item)

    def _on_ct_image_selected(self, item):
        # 启动后端处理线程，并传递图像路径（优先真实推理，回退模拟）
        img_path = item.data(Qt.UserRole)
        if not img_path or not os.path.exists(img_path):
            return

        # 如有正在运行的线程，尝试停止
        if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
            try:
                if hasattr(self.processor_thread, 'stop'):
                    self.processor_thread.stop()
                else:
                    self.processor_thread.quit()
                    self.processor_thread.wait(200)
            except Exception:
                pass

        if TumorImageProcessorThread is not None:
            # 使用后端真实处理线程
            self.processor_thread = TumorImageProcessorThread(selected_image_path=img_path)
            self.processor_thread.images_processed.connect(self._update_images)
            # 使用后端信号在报告生成后加载
            try:
                self.processor_thread.image_name_ready.connect(self._load_report)
            except Exception:
                pass
            self.processor_thread.error_occurred.connect(self._show_error)
            self.processor_thread.start()
            # 上方面板显示处理中提示
            self.report_browser.setText("正在处理图像...\nProcessing image...")
        else:
            # 回退到前端模拟线程（不传入字符串参数）
            self.processor_thread = ImageProcessorThread()
            self.processor_thread.images_processed.connect(self._update_images)
            # 模拟线程的分析文本直接显示到上方统计面板
            try:
                self.processor_thread.analysis_ready.connect(self._update_analysis_text)
            except Exception:
                pass
            self.processor_thread.start()

    def _show_error(self, error_message):
        self.report_browser.setText(error_message)

    def closeEvent(self, event):
        if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
            # 优先调用真实线程的stop方法以安全结束
            if hasattr(self.processor_thread, 'stop'):
                try:
                    self.processor_thread.stop()
                except Exception:
                    self.processor_thread.quit()
                    self.processor_thread.wait()
            else:
                self.processor_thread.quit()
                self.processor_thread.wait()
        event.accept()

    def _analyze_current_overlay(self):
        """调用 Qwen-VL 对当前叠加图进行语义分析。"""
        if not hasattr(self, "_current_overlay_path") or not self._current_overlay_path:
            self.analysis_text_browser.setText("尚未生成叠加图，无法进行语义分析。请先选择图像并完成推理。")
            return

        try:
            self.analysis_text_browser.setText("正在调用 Qwen-VL API 分析叠加图...\n请稍候。")
            self._qwen_thread = QwenAnalyzerThread(self._current_overlay_path)
            self._qwen_thread.result_ready.connect(lambda text: self.analysis_text_browser.setText(text))
            self._qwen_thread.error_occurred.connect(lambda err: self.analysis_text_browser.setText(f"分析失败：{err}"))
            self._qwen_thread.start()
        except Exception as e:
            self.analysis_text_browser.setText(f"启动分析线程失败：{e}")

class QwenAnalyzerThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, question=None):
        super().__init__()
        self.image_path = image_path
        self.question = question

    def run(self):
        try:
            # 动态导入，避免在主线程加载
            from back.llm_analyzer import analyze_image_with_qwen
            analysis_result = analyze_image_with_qwen(self.image_path, self.question)
            self.result_ready.emit(analysis_result)
        except Exception as e:
            self.error_occurred.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MedicalViewer()
    viewer.show()
    sys.exit(app.exec_())
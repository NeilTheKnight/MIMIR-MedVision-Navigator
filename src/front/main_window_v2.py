import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QPushButton, QSizePolicy, QSpacerItem,
    QStyle, QToolButton
)
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPainter, QColor, QPainterPath
from PyQt5.QtCore import Qt, QSize, QPoint, pyqtSignal

# 引入分类与分割页面
from front.main_window1 import MedicalViewer as ClassificationViewer
from front.main_window2 import MedicalViewer as SegmentationViewer
# 安全获取标准图标的辅助函数（避免不存在的枚举导致崩溃）
def get_standard_icon(icon_name: str, fallback: QStyle.StandardPixmap = QStyle.SP_FileIcon) -> QIcon:
    try:
        std_pixmap = getattr(QStyle, icon_name)
        return QApplication.instance().style().standardIcon(std_pixmap)
    except Exception:
        return QApplication.instance().style().standardIcon(fallback)

def make_round_pixmap(src: QPixmap, diameter: int) -> QPixmap:
    """将方形/任意图像裁剪为圆形头像，并缩放到直径指定大小"""
    size = QSize(diameter, diameter)
    # 先等比缩放以覆盖圆区域
    scaled = src.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    # 在透明底上绘制圆形裁剪
    rounded = QPixmap(size)
    rounded.fill(Qt.transparent)
    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    path = QPainterPath()
    path.addEllipse(0, 0, diameter, diameter)
    painter.setClipPath(path)
    # 将缩放后的图像居中绘制到圆形区域
    x = (diameter - scaled.width()) // 2
    y = (diameter - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return rounded

# --- 1. Custom Sidebar Item Widget ---
# 用于实现侧边栏导航项，可以控制选中状态的样式
class NavItem(QWidget):
    clicked = pyqtSignal(object)
    def __init__(self, icon_name, text, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)

        # 去除图标，仅保留文字并加粗
        self.text_label = QLabel(text)
        self.text_label.setFont(QFont("Inter", 10, QFont.Bold))

        self.layout.addWidget(self.text_label)
        self.layout.addStretch()

        # 使用面板样式隔开每一项
        self.setObjectName("NavItemPanel")
        self._is_active = False

    def set_active(self, active):
        self._is_active = active
        # 使用属性以便样式表生效
        self.setProperty("active", True if active else False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # 绘制左侧的蓝色指示条
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#4B7BEC")) # 蓝色
            # 加粗左侧指示条
            painter.drawRect(0, 5, 6, self.height() - 10)
        
    def mousePressEvent(self, event):
        # 发射点击信号，外部统一处理选中态
        self.clicked.emit(self)
        super().mousePressEvent(event)


# --- 2. Custom Dashboard Card Widgets ---
class FunctionalCard(QFrame):
    def __init__(self, title, description, button_text, icon_name, parent=None):
        super().__init__(parent)
        self.setObjectName("FunctionalCard")
        self.setFixedSize(400, 200)
        self.setCursor(Qt.PointingHandCursor) # 设置鼠标手型

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # 标题和图标
        header_layout = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setFont(QFont("Inter", 14, QFont.Bold))
        title_label.setObjectName("CardTitle")
        
        # 使用 QToolButton 模仿图标按钮
        icon_btn = QToolButton()
        icon = get_standard_icon(icon_name, fallback=QStyle.SP_FileIcon)
        icon_btn.setIcon(icon)
        icon_btn.setObjectName("IconBtn")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(icon_btn)
        
        # 描述
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("CardDescription")
        
        # 按钮
        btn = QPushButton(button_text)
        btn.setObjectName("CardButton")
        btn.setFixedSize(120, 40)
        self.button = btn  # 暴露按钮以便外部连接点击事件
        
        layout.addLayout(header_layout)
        layout.addWidget(desc_label)
        layout.addStretch()
        layout.addWidget(btn)
        
        self.setLayout(layout)

class StatusCard(QFrame):
    def __init__(self, title, metrics, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusCard")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Inter", 12))
        title_label.setObjectName("StatusTitle")
        
        # 指标内容 (可以是数字或百分比)
        metrics_layout = QHBoxLayout()
        for metric in metrics:
            value_label = QLabel(metric['value'])
            value_label.setFont(QFont("Inter", 24, QFont.Bold))
            value_label.setObjectName("StatusValue")
            
            unit_label = QLabel(metric['unit'])
            unit_label.setFont(QFont("Inter", 10))
            unit_label.setObjectName("StatusUnit")
            
            # 创建一个垂直布局来放置数值和单位
            metric_col = QVBoxLayout()
            metric_col.setSpacing(0)
            metric_col.addWidget(value_label)
            metric_col.addWidget(unit_label)
            metric_col.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            
            metrics_layout.addLayout(metric_col)
            metrics_layout.addSpacing(40) # 间距
        
        metrics_layout.addStretch()
            
        layout.addWidget(title_label)
        layout.addLayout(metrics_layout)
        layout.addStretch()
        
        self.setLayout(layout)

# --- 3. Main Application Window ---
class AIAssistedDiagnosisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("脑肿瘤 AI 辅助诊断系统")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self._apply_stylesheet()
        self._setup_main_layout()

    def _apply_stylesheet(self):
        # 模仿截图中的深色主题和圆角样式
        style_sheet = """
        /* 整体窗口和背景 */
        QMainWindow {
            background-color: #1A1A2E; /* 主背景深蓝紫 */
            color: #E0E0E0;
            font-family: "Inter", sans-serif;
        }
        /* 全局文字默认颜色，避免黑色文字与深色背景重合 */
        QLabel { color: #FFFFFF; }
        QPushButton { color: #FFFFFF; }
        
        /* 侧边栏样式 */
        #Sidebar {
            background-color: #1A1A2E;
            border-right: 1px solid #33334A;
        }

        /* 侧边栏导航项面板样式 */
        #NavItemPanel {
            background-color: #2C2C40;
            border-radius: 8px;
            padding: 10px 14px;
            margin: 6px 10px;
            color: #FFFFFF;
        }
        #NavItemPanel:hover {
            background-color: #383850;
        }
        /* 选中高亮背景（与左侧粗指示条叠加） */
        #NavItemPanel[active="true"] {
            background-color: #3A4AA0;
        }
        
        /* 核心功能卡片 (FunctionalCard) */
        #FunctionalCard, #StatusCard {
            background-color: #2C2C40; /* 略浅的卡片背景 */
            border-radius: 12px;
            padding: 15px;
        }
        #FunctionalCard:hover {
            background-color: #383850;
        }
        
        /* 按钮样式 (例如 "开始分类") */
        #CardButton {
            background-color: #4B7BEC; /* 蓝色主色 */
            color: white;
            border-radius: 6px;
            padding: 8px;
        }
        #CardButton:hover {
            background-color: #6A9BFB;
        }
        
        /* 头部和文字颜色 */
        #Header {
            background-color: #1A1A2E;
            border-bottom: 1px solid #33334A;
        }
        #AppTitle {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: bold;
        }
        #WelcomeText {
            color: #FFFFFF;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        /* 卡片文字样式 */
        #CardTitle {
            color: #FFFFFF;
        }
        #CardDescription {
            color: #A0A0B0;
            font-size: 10px;
        }
        
        /* 状态卡片数值 */
        #StatusValue {
            color: #FFFFFF;
        }
        #StatusUnit {
            color: #A0A0B0;
        }
        /* 模型状态的百分比 */
        #ModelAccuracy {
            color: #4B7BEC; /* 蓝色强调 */
            font-size: 32px;
        }
        
        /* 头部用户头像/图标按钮 */
        QToolButton {
            background: none;
            border: none;
            color: #A0A0B0;
        }
        QToolButton:hover {
            color: #FFFFFF;
        }
        
        """
        QApplication.instance().setStyleSheet(style_sheet)

    def _setup_main_layout(self):
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. 侧边栏 (Left Sidebar)
        sidebar_widget = self._create_sidebar()
        main_layout.addWidget(sidebar_widget, 0) # 0 权重，固定宽度
        
        # 2. 右侧主内容区 (Header + Dashboard)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header
        header_widget = self._create_header()
        content_layout.addWidget(header_widget, 0)
        
        # Dashboard Content
        dashboard_widget = self._create_dashboard_content()
        content_layout.addWidget(dashboard_widget, 1) # 1 权重，占据剩余空间
        
        main_layout.addWidget(content_widget, 1) # 1 权重，占据剩余宽度

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(200)
        
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(15, 20, 5, 10)
        vbox.setSpacing(5)
        
        # 顶部标题/Logo
        app_title = QLabel("脑肿瘤 AI 辅助诊断系统")
        app_title.setObjectName("AppTitle")
        app_title.setWordWrap(True)
        app_title.setFixedHeight(50)
        vbox.addWidget(app_title)
        vbox.addSpacing(30)
        
        # 导航菜单
        
        # 核心功能标题
        core_label = QLabel("核心功能")
        core_label.setFont(QFont("Inter", 8))
        core_label.setStyleSheet("color: #707080; margin-top: 10px;")
        vbox.addWidget(core_label)
        
        # 导航项目
        nav_items = [
            ("SP_ComputerIcon", "首页"), # 首页 active（有效）
            ("SP_DirIcon", "肿瘤分类"),   # 有效
            ("SP_DriveHDIcon", "肿瘤分割"), # 有效
            ("SP_FileIcon", "历史记录"),   # 有效
            ("SP_DesktopIcon", "个人中心"), # 有效
        ]
        
        # 添加导航项
        self._sidebar_items = []
        for i, (icon, text) in enumerate(nav_items):
            item = NavItem(icon, text)
            item.clicked.connect(self._on_nav_item_clicked)
            if i == 0:
                item.set_active(True) # 默认选中首页
            vbox.addWidget(item)
            self._sidebar_items.append(item)
            
        vbox.addStretch(1) # 占据中部空白

        # 底部帮助中心
        help_item = NavItem("SP_MessageBoxQuestion", "帮助中心")
        help_item.clicked.connect(self._on_nav_item_clicked)
        vbox.addWidget(help_item)
        self._sidebar_items.append(help_item)
        
        return sidebar

    def _on_nav_item_clicked(self, clicked_item: 'NavItem'):
        # 切换选中高亮
        for item in getattr(self, "_sidebar_items", []):
            item.set_active(item is clicked_item)

    def _create_header(self):
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(60)
        
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(30, 0, 30, 0)
        
        # 留空，主内容区处理欢迎文本
        hbox.addStretch() 
        
        # 用户信息
        user_info_label = QLabel("张医生 (神经外科)")
        user_info_label.setFont(QFont("Inter", 10))
        user_info_label.setStyleSheet("color: #FFFFFF;")
        
        # 头像（圆形 40x40）
        avatar_icon = QToolButton()
        diameter = 40
        avatar_icon.setFixedSize(diameter, diameter)
        avatar_path = os.path.join(os.path.dirname(__file__), "img.jpg")
        if os.path.exists(avatar_path):
            raw = QPixmap(avatar_path)
            rounded = make_round_pixmap(raw, diameter)
            avatar_icon.setIcon(QIcon(rounded))
        else:
            # 回退到一个可用的标准图标
            fallback = QApplication.instance().style().standardIcon(QStyle.SP_ComputerIcon)
            avatar_icon.setIcon(fallback)
        avatar_icon.setIconSize(QSize(diameter, diameter))
        
        # 通知和设置图标
        notification_btn = QToolButton()
        notification_btn.setIcon(QApplication.instance().style().standardIcon(QStyle.SP_MessageBoxInformation))
        notification_btn.setFixedSize(30, 30)

        settings_btn = QToolButton()
        settings_btn.setIcon(QApplication.instance().style().standardIcon(QStyle.SP_TitleBarContextHelpButton))
        settings_btn.setFixedSize(30, 30)

        hbox.addWidget(avatar_icon)
        hbox.addWidget(user_info_label)
        hbox.addSpacing(20)
        hbox.addWidget(notification_btn)
        hbox.addSpacing(10)
        hbox.addWidget(settings_btn)
        
        return header

    def _create_dashboard_content(self):
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 欢迎文本
        welcome_label = QLabel("欢迎回来，张医生!")
        welcome_label.setObjectName("WelcomeText")
        layout.addWidget(welcome_label)
        
        # 卡片网格布局
        grid = QGridLayout()
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(30)
        
        # 1. 肿瘤类型识别卡片
        card1 = FunctionalCard(
            title="肿瘤类型识别",
            description="上传图像，快速分类 4 类脑肿瘤",
            button_text="开始分类",
            icon_name="SP_DialogOpenButton" # 使用有效图标
        )
        # 按下后打开分类页面
        card1.button.clicked.connect(self._open_classification_page)
        grid.addWidget(card1, 0, 0)
        
        # 2. 肿瘤位置定位卡片
        card2 = FunctionalCard(
            title="肿瘤位置定位",
            description="精准分割肿瘤区域",
            button_text="开始分割",
            icon_name="SP_DialogApplyButton" # 使用有效图标
        )
        # 按下后打开分割页面
        card2.button.clicked.connect(self._open_segmentation_page)
        grid.addWidget(card2, 0, 1)
        
        # 3. 今日诊断卡片 (使用 StatusCard)
        card3 = StatusCard(
            title="今日诊断",
            metrics=[
                {'value': '23', 'unit': '分类'},
                {'value': '15', 'unit': '分割'}
            ]
        )
        card3.setFixedSize(400, 200)
        grid.addWidget(card3, 1, 0)
        
        # 4. 模型状态卡片
        model_status_card = QFrame()
        model_status_card.setObjectName("StatusCard")
        model_status_card.setFixedSize(400, 200)
        
        status_layout = QVBoxLayout(model_status_card)
        status_layout.setContentsMargins(20, 20, 20, 20)
        
        status_title = QLabel("模型状态")
        status_title.setObjectName("StatusTitle")
        
        accuracy_label = QLabel("99.1%")
        accuracy_label.setObjectName("ModelAccuracy")
        
        desc_label = QLabel("当前模型准确率")
        desc_label.setObjectName("StatusUnit")
        
        status_layout.addWidget(status_title)
        status_layout.addSpacing(10)
        status_layout.addWidget(accuracy_label)
        status_layout.addWidget(desc_label)
        status_layout.addStretch()
        
        grid.addWidget(model_status_card, 1, 1)
        
        # 将网格布局添加到主布局中
        layout.addLayout(grid)
        layout.addStretch(1) # 占据底部空白
        
        return dashboard

    def _open_classification_page(self):
        try:
            viewer = ClassificationViewer()
            if not hasattr(self, "_child_windows"):
                self._child_windows = []
            self._child_windows.append(viewer)
            viewer.show()
        except Exception as e:
            print(f"打开分类页面失败: {e}")

    def _open_segmentation_page(self):
        try:
            viewer = SegmentationViewer()
            if not hasattr(self, "_child_windows"):
                self._child_windows = []
            self._child_windows.append(viewer)
            viewer.show()
        except Exception as e:
            print(f"打开分割页面失败: {e}")

# --- 4. Application Entry Point ---
if __name__ == '__main__':
    # 为了让 QSS 中的 "Inter" 字体生效，需要确保系统中有该字体，
    # 或者可以替换为 Arial/Microsoft YaHei 等常见字体。
    # 这里保持 "Inter" 不变。
    
    app = QApplication(sys.argv)
    window = AIAssistedDiagnosisApp()
    window.show()
    sys.exit(app.exec_())
    
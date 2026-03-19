import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QLineEdit, 
    QStackedWidget, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
from front.main_window_v2 import AIAssistedDiagnosisApp


# --- 登录页面 (LoginWidget) ---
class LoginWidget(QWidget):
    login_successful = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._init_styles()
        self._init_ui()

    def _init_styles(self):
        self.setStyleSheet("""
            LoginWidget {
                background-color: #0a1028;
            }
            #LoginFormBox {
                background-color: #141b38;
                border-radius: 8px;
                padding: 30px;
            }
            QLabel {
                color: #e0e5f0;
                font-size: 11pt;
            }
            QLabel#LoginTitle {
                color: white;
                font-size: 24pt;
                font-weight: bold;
            }
            QLabel#LoginSubtitle {
                color: #a0a5b8;
                font-size: 12pt;
            }
            QLineEdit {
                background-color: #2d3756;
                border: 1px solid #3a4568;
                border-radius: 4px;
                padding: 10px;
                color: #e0e5f0;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border-color: #4a90e2;
            }
            QPushButton#LoginButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton#LoginButton:hover {
                background-color: #5a9ff2;
            }
            QCheckBox {
                color: #a0a5b8;
                font-size: 10pt;
            }
            QPushButton.LinkButton {
                background-color: transparent;
                border: none;
                color: #4a90e2;
                font-size: 10pt;
                text-align: right;
            }
            QPushButton.LinkButton:hover {
                text-decoration: underline;
            }
        """)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setAlignment(Qt.AlignCenter)
        
        # 标题区域：使用 MIMIR 图标替换原文本标题
        logo_label = QLabel(objectName="LoginTitle")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(__file__), "MIMIR.png")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
        else:
            logo_label.setText("")
        subtitle = QLabel("脑肿瘤 AI 辅助诊断系统", objectName="LoginSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        content_layout.addWidget(logo_label)
        content_layout.addWidget(subtitle)
        content_layout.addSpacing(30)

        # 登录表单
        form_frame = QFrame(objectName="LoginFormBox")
        form_frame.setFixedWidth(420)
        form_layout = QVBoxLayout(form_frame)
        form_layout.setSpacing(15)

        # 账号
        form_layout.addWidget(QLabel("医生工号 / 账号"))
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入医生工号 / 账号")
        form_layout.addWidget(self.username_input)

        # 密码
        form_layout.addWidget(QLabel("密码"))
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(self.password_input)

        # 验证码
        form_layout.addWidget(QLabel("验证码"))
        captcha_layout = QHBoxLayout()
        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")
        captcha_layout.addWidget(self.captcha_input, 3)
        captcha_label = QLabel("[ 8xZ2 ]")
        captcha_label.setFixedSize(100, 36)
        captcha_label.setAlignment(Qt.AlignCenter)
        captcha_label.setStyleSheet("background-color: #2d3756; border: 1px solid #3a4568; color: #a0a5b8; font-weight: bold;")
        captcha_layout.addWidget(captcha_label, 1)
        form_layout.addLayout(captcha_layout)
        
        # 记住我 / 忘记密码
        options_layout = QHBoxLayout()
        options_layout.addWidget(QCheckBox("记住我 (7天内免登录)"))
        options_layout.addStretch()
        options_layout.addWidget(QPushButton("忘记密码?", objectName="LinkButton"))
        form_layout.addLayout(options_layout)

        # 登录按钮
        login_button = QPushButton("登 录", objectName="LoginButton")
        login_button.clicked.connect(self._check_credentials)
        form_layout.addWidget(login_button)
        
        # 注册链接
        register_button = QPushButton("无账号? 联系管理员开通", objectName="LinkButton")
        register_button.setStyleSheet("text-align: center;")
        form_layout.addWidget(register_button)

        content_layout.addWidget(form_frame)
        main_layout.addWidget(content_widget)

        # 页脚
        footer = QLabel("© 2025 医疗 AI 辅助平台 | 技术支持")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #5e6a8d; font-size: 9pt; margin-top: 50px;")
        main_layout.addWidget(footer)

    def _check_credentials(self):
        if self.username_input.text() == "admin" and self.password_input.text() == "12345":
            self.login_successful.emit()
        else:
            QMessageBox.warning(self, "登录失败", "账号或密码不正确，请重试。")
            self.password_input.clear()


# --- 仪表板页面 (DashboardWidget) ---
class DashboardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        # 核心颜色配置（与截图匹配）
        self.main_bg = "#0a1028"         # 主背景色
        self.sidebar_bg = "#141b38"      # 侧边栏背景
        self.sidebar_active = "#2d3756"  # 侧边栏选中项背景
        self.text_primary = "#ffffff"    # 主要文字
        self.text_secondary = "#a0a5b8"  # 次要文字
        self.btn_bg = "#2d3756"          # 按钮背景
        self.blue = "#4a90e2"            # 蓝色图标/文字
        self.purple = "#9d4edd"          # 紫色图标/文字
        self.green = "#00ff80"           # 模型准确率绿色
        
        self._init_styles()
        self._init_layout()

    def _init_styles(self):
        self.setStyleSheet(f"""
            DashboardWidget {{
                background-color: {self.main_bg};
            }}
            #Sidebar {{
                background-color: {self.sidebar_bg};
                border-right: 1px solid #1f2a48;
            }}
            QPushButton.SidebarBtn {{
                background-color: transparent;
                color: {self.text_secondary};
                font-size: 11pt;
                padding: 12px 15px;
                text-align: left;
                border: none;
                border-radius: 0;
            }}
            QPushButton.SidebarBtn:checked {{
                background-color: {self.sidebar_active};
                color: {self.text_primary};
                font-weight: 500;
            }}
            QPushButton.SidebarBtn:hover {{
                background-color: #252f4d;
            }}
            QLabel.SidebarHeader {{
                color: {self.text_secondary};
                font-size: 9pt;
                font-weight: bold;
                padding: 15px 15px 5px 15px;
                text-transform: uppercase;
            }}
            #MainContent {{
                background-color: {self.main_bg};
                padding: 20px 30px;
            }}
            QLabel.WelcomeText {{
                color: {self.text_primary};
                font-size: 20pt;
                font-weight: bold;
                margin: 10px 0 25px 0;
            }}
            .Card {{
                background-color: transparent;
                padding: 20px;
                border-radius: 6px;
            }}
            QLabel.CardTitle {{
                color: {self.text_primary};
                font-size: 13pt;
                margin-bottom: 8px;
            }}
            QLabel.CardDesc {{
                color: {self.text_secondary};
                font-size: 10pt;
                margin-bottom: 15px;
            }}
            QPushButton.ActionBtn {{
                background-color: {self.btn_bg};
                color: {self.text_primary};
                border: none;
                padding: 8px 20px;
                font-size: 10pt;
                border-radius: 4px;
            }}
            QPushButton.ActionBtn:hover {{
                background-color: #3a4768;
            }}
            QLabel.StatNum {{
                color: {self.text_primary};
                font-size: 26pt;
                font-weight: bold;
                margin: 10px 0;
            }}
            QLabel.StatText {{
                color: {self.text_secondary};
                font-size: 10pt;
            }}
            QLabel.ModelAccuracy {{
                color: {self.green};
                font-size: 28pt;
                font-weight: bold;
                margin: 10px 0;
            }}
            QLabel.ModelText {{
                color: {self.text_secondary};
                font-size: 10pt;
            }}
            #TopBar {{
                padding: 10px 0;
                border-bottom: 1px solid #1f2a48;
                margin-bottom: 10px;
            }}
            QLabel.TitleText {{
                color: {self.text_primary};
                font-size: 13pt;
                font-weight: bold;
            }}
            QLabel.UserInfo {{
                color: {self.text_primary};
                font-size: 10pt;
            }}
            QPushButton.IconBtn {{
                background-color: transparent;
                color: {self.text_secondary};
                border: none;
                font-size: 14pt;
                width: 30px;
                height: 30px;
            }}
            QPushButton.IconBtn:hover {{
                color: {self.text_primary};
            }}
            QLabel.IconLabel {{
                font-size: 20pt;
                font-weight: bold;
            }}
        """)

    def _init_layout(self):
        # 主布局（侧边栏+主内容）
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 侧边栏
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar, 1)

        # 2. 主内容区
        main_content = self._create_main_content()
        main_layout.addWidget(main_content, 5)

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(180)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 首页按钮（默认选中）
        btn_home = QPushButton("首页", objectName="SidebarBtn")
        btn_home.setCheckable(True)
        btn_home.setChecked(True)
        btn_home.setText(" [首页]")  # 模拟图标位置
        
        # 核心功能标题
        lbl_core = QLabel("核心功能", objectName="SidebarHeader")
        
        # 功能按钮
        btn_classify = QPushButton(" [肿瘤分类]", objectName="SidebarBtn")
        btn_classify.setCheckable(True)
        btn_segment = QPushButton(" [肿瘤分割]", objectName="SidebarBtn")
        btn_segment.setCheckable(True)
        
        # 其他功能
        btn_history = QPushButton(" [历史记录]", objectName="SidebarBtn")
        btn_history.setCheckable(True)
        btn_profile = QPushButton(" [个人中心]", objectName="SidebarBtn")
        btn_profile.setCheckable(True)
        
        # 帮助中心（底部）
        btn_help = QPushButton(" [帮助中心]", objectName="SidebarBtn")
        btn_help.setCheckable(True)
        
        # 添加到布局
        layout.addWidget(btn_home)
        layout.addWidget(lbl_core)
        layout.addWidget(btn_classify)
        layout.addWidget(btn_segment)
        layout.addSpacing(10)
        layout.addWidget(btn_history)
        layout.addWidget(btn_profile)
        layout.addStretch()
        layout.addWidget(btn_help)
        
        return sidebar

    def _create_main_content(self):
        main_content = QFrame()
        main_content.setObjectName("MainContent")
        
        layout = QVBoxLayout(main_content)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 顶部栏
        top_bar = self._create_top_bar()
        layout.addWidget(top_bar)
        
        # 欢迎语
        welcome = QLabel("欢迎回来，张医生！", objectName="WelcomeText")
        layout.addWidget(welcome)
        
        # 内容网格
        grid = QGridLayout()
        grid.setSpacing(20)
        grid.setContentsMargins(0, 0, 0, 20)
        
        # 卡片1：肿瘤类型识别
        card1 = self._create_function_card(
            title="肿瘤类型识别",
            desc="上传图像，快速分类 4 类肿瘤",
            btn_text="开始分类",
            icon="[C]",
            icon_color=self.blue
        )
        
        # 卡片2：肿瘤位置定位
        card2 = self._create_function_card(
            title="肿瘤位置定位",
            desc="精准分割肿瘤区域",
            btn_text="开始分割",
            icon="[S]",
            icon_color=self.purple
        )
        
        # 卡片3：今日诊断
        card3 = self._create_stats_card()
        
        # 卡片4：模型状态
        card4 = self._create_model_card()
        
        grid.addWidget(card1, 0, 0)
        grid.addWidget(card2, 0, 1)
        grid.addWidget(card3, 1, 0)
        grid.addWidget(card4, 1, 1)
        
        layout.addLayout(grid)
        layout.addStretch()
        
        return main_content

    def _create_top_bar(self):
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 系统标题
        title = QLabel("智脑鉴瘤 - 脑肿瘤 AI 辅助诊断系统", objectName="TitleText")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # 用户信息
        user_info = QLabel("张医生 (神经外科)", objectName="UserInfo")
        
        # 通知和退出图标
        btn_notify = QPushButton("🔔", objectName="IconBtn")
        btn_logout = QPushButton("↗", objectName="IconBtn")
        
        layout.addWidget(user_info)
        layout.addSpacing(15)
        layout.addWidget(btn_notify)
        layout.addWidget(btn_logout)
        
        return top_bar

    def _create_function_card(self, title, desc, btn_text, icon, icon_color):
        """创建功能卡片（肿瘤分类/分割）"""
        card = QFrame()
        card.setObjectName("Card")
        
        layout = QVBoxLayout(card)
        
        # 标题+图标
        top = QHBoxLayout()
        text_area = QVBoxLayout()
        text_area.addWidget(QLabel(title, objectName="CardTitle"))
        text_area.addWidget(QLabel(desc, objectName="CardDesc"))
        top.addLayout(text_area)
        
        # 图标
        icon_label = QLabel(icon, objectName="IconLabel")
        icon_label.setStyleSheet(f"color: {icon_color};")
        top.addWidget(icon_label)
        layout.addLayout(top)
        
        # 按钮
        btn = QPushButton(btn_text, objectName="ActionBtn")
        layout.addWidget(btn)
        
        return card

    def _create_stats_card(self):
        """创建今日诊断统计卡片"""
        card = QFrame()
        card.setObjectName("Card")
        
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("今日诊断", objectName="CardTitle"))
        
        # 统计数字
        stats = QHBoxLayout()
        stats.setSpacing(30)
        
        # 分类统计
        classify = QVBoxLayout()
        classify.addWidget(QLabel("23", objectName="StatNum"))
        classify.addWidget(QLabel("分类", objectName="StatText"))
        
        # 分割统计
        segment = QVBoxLayout()
        segment.addWidget(QLabel("15", objectName="StatNum"))
        segment.addWidget(QLabel("分割", objectName="StatText"))
        
        stats.addLayout(classify)
        stats.addLayout(segment)
        stats.addStretch()
        
        layout.addLayout(stats)
        return card

    def _create_model_card(self):
        """创建模型状态卡片"""
        card = QFrame()
        card.setObjectName("Card")
        
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("模型状态", objectName="CardTitle"))
        layout.addWidget(QLabel("99.1%", objectName="ModelAccuracy"))
        layout.addWidget(QLabel("当前模型准确率", objectName="ModelText"))
        
        return card


# --- 主窗口 (AppWindow) ---
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医疗 AI 辅助诊断系统")
        self.setGeometry(100, 100, 1200, 800)

        # 仅登录页
        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)
        self.login_page = LoginWidget()
        self.login_page.login_successful.connect(self.show_dashboard)
        self.stacked.addWidget(self.login_page)

        # 默认显示登录页
        self.stacked.setCurrentIndex(0)

    def show_dashboard(self):
        # 登录成功后切换到 v2 界面（新窗口），并关闭当前窗口
        self.v2_window = AIAssistedDiagnosisApp()
        self.v2_window.show()
        self.close()

    def closeEvent(self, event):
        event.accept()


# --- 程序入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
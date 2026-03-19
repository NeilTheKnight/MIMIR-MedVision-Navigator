import sys
from PyQt5.QtWidgets import QApplication

# 统一入口：从登录页（v1）启动，后续在 v1 内部跳转到 v2；
# v2 中按钮可进一步打开 分类（main_window1） 与 分割（main_window2） 页面
from front.main_window_v1 import AppWindow


def main():
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
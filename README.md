# 医影慧导 MIMIR - 脑肿瘤 AI 辅助诊断系统

**Multimodal Imaging & Medical Intelligent Robotics (MIMIR)**  
*以多模态融合 · 开启智慧诊疗时代！*

---

## 🌟 项目简介

MIMIR-MedVision-Navigator 是一款专为神经外科医生设计的脑肿瘤 AI 辅助诊断系统。系统集成了深度学习分割模型（U-Net）、分类模型（CNN）以及先进的多模态大模型（Qwen-VL），实现了从 MRI 影像自动分割、肿瘤分类识别到 AI 临床分析的全流程辅助。

## 🚀 核心功能

- **精准肿瘤分割**：基于 U-Net 架构，实现 MRI 影像中肿瘤区域的像素级分割，自动计算肿瘤占比。
- **高效类型识别**：支持垂体瘤、脑膜瘤、胶质瘤等四类脑部病变的自动分类，准确率达 99.1%。
- **AI 影像辅助分析**：集成 Qwen-VL 多模态大模型，自动生成包含肿瘤阶段、位置风险及临床建议的分析报告。
- **现代化交互界面**：采用 PyQt5 构建，支持影像实时缩放、平移、坐标跟踪及可视化结果对比。
- **高性能架构**：内置模型缓存机制与异步处理线程，确保流畅的临床操作体验。

## 🛠️ 安装教程

### 1. 环境要求
- Windows 10/11
- Python 3.8 或 3.9 (推荐使用 Anaconda 或 venv)
- 显存 4GB 以上的 NVIDIA GPU (推荐，用于加速推理)

### 2. 克隆仓库
```bash
git clone https://github.com/YourUsername/MIMIR-MedVision-Navigator.git
cd MIMIR-MedVision-Navigator
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 环境变量配置
若需使用 AI 语义分析功能，请配置阿里云 DashScope API Key：
- **Windows (PowerShell)**:
  ```powershell
  $env:DASHSCOPE_API_KEY="您的API_KEY"
  ```
- **Windows (CMD)**:
  ```cmd
  set DASHSCOPE_API_KEY=您的API_KEY
  ```

## 📖 使用指南

### 1. 启动系统
在项目根目录下运行：
```bash
python src/main.py
```

### 2. 登录与导航
- 使用默认演示账号登录（账号: `admin` / 密码: `12345`）。
- 在仪表盘界面，通过侧边栏选择“肿瘤分类”或“肿瘤分割”模块。

### 3. 执行诊断
- **肿瘤分类**：选择左侧列表中的 MRI 文件夹，系统将自动批量识别并生成置信度报告。
- **肿瘤分割**：点击感兴趣的 MRI 切片，系统将实时显示分割掩码、热力图，并可点击“AI 分析”获取大模型解读。

## 📂 项目结构

```text
├── datasets/           # 外部化数据集（需手动放置测试影像）
├── src/                # 核心源代码
│   ├── config.py       # 全局路径配置中心
│   ├── back/           # 后端算法与模型推理
│   └── front/          # PyQt5 界面实现
└── docs/               # 项目详细结构报告
```

## 📜 声明
本系统仅作为 AI 辅助诊断工具，所有自动生成的结论仅供医疗专业人员参考，不作为最终临床诊断依据。

---
© 2025 医影慧导 MIMIR 团队 | 赋能智慧医疗

import os

# 获取项目根目录 (D:\claude_code\MIMIR-MedVision-Navigator)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集路径
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

PATHS = {
    "test_images": os.path.join(DATASETS_DIR, "test_images"),
    "test_masks": os.path.join(DATASETS_DIR, "test_masks"),
    "test_results": os.path.join(DATASETS_DIR, "test_results"),
    "cnn_images": os.path.join(DATASETS_DIR, "cnn_images"),
    
    # 补充子目录路径
    "inference_info": os.path.join(DATASETS_DIR, "test_results", "inference_info"),
    "masks": os.path.join(DATASETS_DIR, "test_results", "masks"),
    "visualizations": os.path.join(DATASETS_DIR, "test_results", "visualizations"),
    
    # 模型路径
    "unet_weights": os.path.join(BASE_DIR, "src", "back", "unet_best_weights.h5"),
    "cnn_model": os.path.join(BASE_DIR, "src", "back", "CNN", "brain_tumor_classification_best.h5"),
}

# 确保结果目录存在
os.makedirs(PATHS["test_results"], exist_ok=True)
os.makedirs(os.path.join(PATHS["test_results"], "inference_info"), exist_ok=True)
os.makedirs(os.path.join(PATHS["test_results"], "masks"), exist_ok=True)
os.makedirs(os.path.join(PATHS["test_results"], "visualizations"), exist_ok=True)
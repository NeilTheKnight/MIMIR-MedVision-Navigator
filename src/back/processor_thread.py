import os
import sys
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image

# 添加项目根目录和 src 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/back
src_dir = os.path.dirname(current_dir) # src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import PATHS

# 采用 pre_size 的统计与报告写入，实现更详细的推理信息
from .pre_size import (
    load_tumor_model,
    predict_tumor_with_stats,
    save_inference_info,
)

# 使用现有工具生成概率图、掩码图和叠加图
from .model_utils import (
    create_probability_image,
    create_mask_image,
    create_overlay_image,
)

class ImageProcessorThread(QThread):
    images_processed = pyqtSignal(dict)
    image_name_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    # 类静态变量，用于缓存已加载的模型
    _cached_model = None
    _cached_model_path = None

    def __init__(self, selected_image_path: str = None,
                 model_path: str = None,
                 test_image_dir: str = None,
                 results_dir: str = None,
                 parent=None):
        super().__init__(parent)
        self._stop_flag = False

        self.model_path = model_path or PATHS["unet_weights"]
        self.test_image_dir = test_image_dir or PATHS["test_images"]
        self.results_dir = results_dir or PATHS["test_results"]
        self.selected_image_path = selected_image_path

    def run(self):
        try:
            # choose image
            img_path = self.selected_image_path
            if not img_path:
                # pick first image
                candidates = []
                if not os.path.exists(self.test_image_dir):
                    raise FileNotFoundError(f'测试目录不存在: {self.test_image_dir}')
                for fn in os.listdir(self.test_image_dir):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        candidates.append(os.path.join(self.test_image_dir, fn))
                candidates.sort()
                if not candidates:
                    raise FileNotFoundError(f'在目录 {self.test_image_dir} 中未找到图片。')
                img_path = candidates[0]

            # 加载/获取缓存的模型
            if ImageProcessorThread._cached_model_path != self.model_path or ImageProcessorThread._cached_model is None:
                ImageProcessorThread._cached_model = load_tumor_model(self.model_path)
                ImageProcessorThread._cached_model_path = self.model_path
            
            model = ImageProcessorThread._cached_model

            # 预测并获取详细统计（使用 pre_size）
            # 统一尺寸与阈值，保持与 pre_size 中默认一致
            img_size = 256
            threshold = 0.5
            original_resized, pred_prob, pred_mask, stats_info = predict_tumor_with_stats(
                model, img_path, img_size, threshold
            )

            # 生成可视化图片（四宫格使用）
            prob2d = pred_prob.squeeze()
            mask2d = (pred_mask.squeeze() * 255).astype('uint8')
            prob_img = create_probability_image(prob2d)
            mask_img = create_mask_image(mask2d)
            overlay_img = create_overlay_image(original_resized.convert('RGB'), mask2d)

            # save inference info
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            save_inference_info(image_name, stats_info, self.results_dir)

            # emit image name (for info loading)
            self.image_name_ready.emit(image_name)

            # emit images
            images = {
                'model_input': original_resized,  # 模型输入尺寸的缩放原图
                'probability_map': prob_img,
                'mask': mask_img,
                'overlay': overlay_img,
            }
            if not self._stop_flag:
                self.images_processed.emit(images)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._stop_flag = True
        try:
            self.quit()
            self.wait(200)
        except Exception:
            pass
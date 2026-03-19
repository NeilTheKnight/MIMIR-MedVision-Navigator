import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from .base_model import load_tumor_model
import sys
# 添加项目根目录和 src 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/back
src_dir = os.path.dirname(current_dir) # src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import PATHS

# -------------------------- 配置参数 --------------------------
class TestConfig:
    model_path = PATHS["unet_weights"]
    test_image_dir = PATHS["test_images"]
    test_mask_dir = PATHS["test_masks"]
    output_dir = PATHS["test_results"]
    img_size = 256
    threshold = 0.5
    save_visualization = True
    save_info_file = True
    show_only_last = True

# -------------------------- 图像预处理 + 提取基础信息 --------------------------
def preprocess_image_with_info(img_path, img_size):
    img_original = Image.open(img_path).convert('RGB')
    original_width, original_height = img_original.size
    original_total_pixels = original_width * original_height
    
    img_resized = img_original.resize((img_size, img_size))
    resized_total_pixels = img_size * img_size  # 直接计算缩放后总像素
    
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    return {
        "img_input": img_input,
        "img_resized": img_resized,
        "original_info": {
            "width": original_width,
            "height": original_height,
            "total_pixels": original_total_pixels,
            "scale": (original_width/img_size, original_height/img_size)  # 宽高缩放比例
        },
        "resized_info": {
            "width": img_size,
            "height": img_size,
            "total_pixels": resized_total_pixels  # 存储缩放后总像素
        }
    }

# -------------------------- 模型预测 + 计算肿瘤分割统计信息 --------------------------
def predict_tumor_with_stats(model, img_path, img_size, threshold):
    preprocess_result = preprocess_image_with_info(img_path, img_size)
    img_input = preprocess_result["img_input"]
    img_resized = preprocess_result["img_resized"]
    original_info = preprocess_result["original_info"]
    resized_info = preprocess_result["resized_info"]
    
    pred_prob = model.predict(img_input, verbose=0)[0]
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    
    # 肿瘤核心统计
    tumor_pixels_resized = np.sum(pred_mask)
    tumor_pixel_ratio = (tumor_pixels_resized / resized_info["total_pixels"]) * 100
    scale_w, scale_h = original_info["scale"]
    tumor_pixels_original = int(tumor_pixels_resized * (scale_w * scale_h))
    
    # 计算肿瘤近似宽高（基于掩码外接矩形）
    mask_squeezed = pred_mask.squeeze()
    coords = np.argwhere(mask_squeezed == 1)
    tumor_resized_w, tumor_resized_h = 0, 0
    tumor_original_w, tumor_original_h = 0, 0
    
    if len(coords) > 0:
        min_h, max_h = coords[:, 0].min(), coords[:, 0].max()
        min_w, max_w = coords[:, 1].min(), coords[:, 1].max()
        tumor_resized_h = max_h - min_h + 1
        tumor_resized_w = max_w - min_w + 1
        tumor_original_w = int(tumor_resized_w * scale_w)
        tumor_original_h = int(tumor_resized_h * scale_h)
    
    # 概率图关键信息
    prob_mean = round(np.mean(pred_prob), 4)
    high_conf_pixels = np.sum(pred_prob > 0.8)
    
    stats_info = {
        "image_info": {
            "original_size": f"{original_info['width']}×{original_info['height']}",
            "original_total_pixels": original_info["total_pixels"],
            "model_input_size": f"{resized_info['width']}×{resized_info['height']}",
            "model_input_total_pixels": resized_info["total_pixels"]  # 新增：缩放后总像素
        },
        "tumor_stats": {
            "resized": {
                "pixels": tumor_pixels_resized,
                "ratio": round(tumor_pixel_ratio, 2),
                "approx_size": f"{tumor_resized_w}×{tumor_resized_h}"
            },
            "original_est": {
                "pixels": tumor_pixels_original,
                "approx_size": f"{tumor_original_w}×{tumor_original_h}"
            }
        },
        "prob_info": {
            "mean_prob": prob_mean,
            "high_conf_pixels": high_conf_pixels,
            "threshold": threshold
        }
    }
    
    return img_resized, pred_prob, pred_mask, stats_info

# -------------------------- 可视化结果 --------------------------
def visualize_prediction(img, pred_mask, pred_prob, img_name, stats_info, save_dir, is_last, show=True):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    
    # 1. 缩放后图像
    axes[0].imshow(img)
    axes[0].set_title(
        f"Model Input Image\nSize: {stats_info['image_info']['model_input_size']}",
        fontsize=10
    )
    axes[0].axis('off')
    
    # 2. 预测概率图
    im2 = axes[1].imshow(pred_prob.squeeze(), cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(
        f"Prediction Probability\nMean: {stats_info['prob_info']['mean_prob']}",
        fontsize=10
    )
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Probability", fontsize=8)
    
    # 3. 预测掩码
    axes[2].imshow(pred_mask.squeeze(), cmap='gray')
    axes[2].set_title(
        f"Predicted Tumor Mask\nPixels: {stats_info['tumor_stats']['resized']['pixels']} ({stats_info['tumor_stats']['resized']['ratio']}%)\nApprox Size: {stats_info['tumor_stats']['resized']['approx_size']}",
        fontsize=10
    )
    axes[2].axis('off')
    
    # 4. 叠加效果
    axes[3].imshow(img, alpha=0.7)
    axes[3].imshow(pred_mask.squeeze(), cmap='jet', alpha=0.3)
    axes[3].set_title(
        f"Overlay (Tumor Region)\nOriginal Size Est: {stats_info['image_info']['original_size']}\nTumor Est Size: {stats_info['tumor_stats']['original_est']['approx_size']}",
        fontsize=10
    )
    axes[3].axis('off')
    
    save_path = os.path.join(save_dir, f"{img_name}_result.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if is_last and show:
        plt.show()
    plt.close()
    return save_path

# -------------------------- 保存详细信息到文件（修正错误） --------------------------
def save_inference_info(img_name, stats_info, save_dir):
    info_dir = os.path.join(save_dir, "inference_info")
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"{img_name}_info.txt")
    
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"=== 脑部肿瘤分割推理报告-图像：{img_name} ===\n")
        f.write(f"推理时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分割阈值：{stats_info['prob_info']['threshold']}\n\n")
        
        f.write("1.原始图像信息\n")
        f.write(f" - 尺寸(宽x高)：{stats_info['image_info']['original_size']} 像素\n")
        f.write(f" - 总像素格数：{stats_info['image_info']['original_total_pixels']} 像素\n\n")
        
        f.write("2.模型输入图像信息(缩放后)\n")
        f.write(f" - 尺寸(宽x高)：{stats_info['image_info']['model_input_size']} 像素\n")
        # 修正：直接使用预计算的总像素值，避免字符串运算
        f.write(f" - 总像素格数：{stats_info['image_info']['model_input_total_pixels']} 像素\n\n")
        
        f.write("3.肿瘤分割统计信息\n")
        f.write(f" - 缩放后肿瘤区域像素数：{stats_info['tumor_stats']['resized']['pixels']} 像素\n")
        f.write(f" - 缩放后肿瘤像素占比：{stats_info['tumor_stats']['resized']['ratio']}%\n")
        f.write(f" - 原始尺寸肿瘤像素数(估算)：{stats_info['tumor_stats']['original_est']['pixels']} 像素\n\n")
        
        f.write("4.肿瘤分割结果\n")
        f.write("   【缩放后（模型输入尺寸）】\n")
        f.write(f"   - 肿瘤像素数：{stats_info['tumor_stats']['resized']['pixels']} | 占比：{stats_info['tumor_stats']['resized']['ratio']}%\n")
        f.write(f"   - 肿瘤近似宽高：{stats_info['tumor_stats']['resized']['approx_size']} 像素\n")
        f.write("   【原始尺寸估算】\n")
        f.write(f"   - 肿瘤像素数：{stats_info['tumor_stats']['original_est']['pixels']}\n")
        f.write(f"   - 肿瘤近似宽高：{stats_info['tumor_stats']['original_est']['approx_size']} 像素\n\n")
        
        f.write("5.预测概率信息\n")
        f.write(f"   平均概率：{stats_info['prob_info']['mean_prob']} | 高置信度肿瘤像素（>0.8）：{stats_info['prob_info']['high_conf_pixels']}\n")
    
    return info_path

# -------------------------- 主测试流程 --------------------------
def main():
    cfg = TestConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_tumor_model(cfg.model_path)
    if model is None:
        return
    
    # 获取测试图像
    image_extensions = ('.jpg', '.jpeg', '.png', '.nii', '.nii.gz')
    test_images = [f for f in os.listdir(cfg.test_image_dir) if f.lower().endswith(image_extensions)]
    if not test_images:
        print(f"测试文件夹 {cfg.test_image_dir} 中未找到图像文件！")
        return
    print(f"找到 {len(test_images)} 张测试图像，开始推理...\n")
    
    # 批量测试
    for idx, img_name in enumerate(tqdm(test_images, desc="推理进度")):
        img_path = os.path.join(cfg.test_image_dir, img_name)
        img_basename = os.path.splitext(img_name)[0]
        
        # 预测肿瘤区域 + 获取详细统计信息
        img_resized, pred_prob, pred_mask, stats_info = predict_tumor_with_stats(
            model, img_path, cfg.img_size, cfg.threshold
        )
        
        # 控制台打印关键信息（仅最后一张）
        if idx == len(test_images) - 1:
            print(f"\n=== 最后一张图像：{img_basename} ===")
            print(f"1.原始图像：{stats_info['image_info']['original_size']} 像素 | 总像素 {stats_info['image_info']['original_total_pixels']}")
            print(f"2.模型输入：{stats_info['image_info']['model_input_size']} 像素 | 总像素 {stats_info['image_info']['model_input_total_pixels']}")
            print(f"3.肿瘤分割：缩放后{stats_info['tumor_stats']['resized']['pixels']}像素({stats_info['tumor_stats']['resized']['ratio']}%) | 原始估算{stats_info['tumor_stats']['original_est']['pixels']}像素")
            print(f"4.肿瘤尺寸：缩放后{stats_info['tumor_stats']['resized']['approx_size']} | 原始估算{stats_info['tumor_stats']['original_est']['approx_size']}")
            print(f"5.概率信息：平均{stats_info['prob_info']['mean_prob']} | 高置信像素{stats_info['prob_info']['high_conf_pixels']}")
        
        # 可视化并保存结果
        if cfg.save_visualization:
            is_last = (idx == len(test_images) - 1)
            vis_path = visualize_prediction(
                img=img_resized,
                pred_mask=pred_mask,
                pred_prob=pred_prob,
                img_name=img_basename,
                stats_info=stats_info,
                save_dir=os.path.join(cfg.output_dir, "visualizations"),
                is_last=is_last,
                show=cfg.show_only_last
            )
            if idx == len(test_images) - 1:
                print(f"可视化结果保存路径：{vis_path}")
        
        # 保存预测掩码
        mask_save_path = os.path.join(cfg.output_dir, "masks", f"{img_basename}_pred_mask.png")
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        pred_mask_img = Image.fromarray((pred_mask.squeeze() * 255).astype(np.uint8))
        pred_mask_img.save(mask_save_path)
        
        # 保存详细信息到文件
        if cfg.save_info_file:
            info_path = save_inference_info(img_basename, stats_info, cfg.output_dir)
            if idx == len(test_images) - 1:
                print(f"详细信息文件保存路径：{info_path}")
    
    print(f"\n所有图像推理完成！结果总目录：{cfg.output_dir}")

if __name__ == "__main__":
    main()
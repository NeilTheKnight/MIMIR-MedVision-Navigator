import os
import sys
import time
import datetime
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 添加 src 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/back/CNN
src_dir = os.path.dirname(os.path.dirname(current_dir)) # src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import PATHS

# 配置路径
MODEL_PATH = PATHS["cnn_model"]
INPUT_DIR = PATHS["cnn_images"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 类别与映射（需与训练一致）
CLASS_TYPES = ['pituitary', 'notumor', 'meningioma', 'glioma']
CLASS_NAMES_CN = {
    'pituitary': '垂体瘤',
    'notumor': '无肿瘤',
    'meningioma': '脑膜瘤',
    'glioma': '胶质瘤'
}

# 图像预处理配置（需与训练一致）
IMAGE_SIZE = (150, 150)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_trained_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    return load_model(model_path)


def preprocess_image(image_path: str) -> np.ndarray:
    image = load_img(image_path, target_size=IMAGE_SIZE)
    array = img_to_array(image) / 255.0
    array = np.expand_dims(array, axis=0)  # (1, H, W, 3)
    return array


def infer_image(model, image_path: str) -> Tuple[str, float, Dict[str, float]]:
    start = time.perf_counter()
    x = preprocess_image(image_path)
    preds = model.predict(x, verbose=0)[0]
    elapsed = time.perf_counter() - start

    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_TYPES[predicted_idx]
    confidence = float(preds[predicted_idx] * 100.0)
    all_probs = {CLASS_TYPES[i]: float(preds[i] * 100.0) for i in range(len(CLASS_TYPES))}
    return predicted_class, confidence, all_probs, elapsed


def try_extract_true_label_from_filename(filename: str) -> str:
    name = os.path.basename(filename).lower()
    for cls in CLASS_TYPES:
        if cls in name:
            return cls
    return ""


def format_distribution_lines(probabilities: Dict[str, float]) -> List[str]:
    # 固定顺序输出四类
    lines = []
    for cls in ['glioma', 'notumor', 'meningioma', 'pituitary']:
        cn = CLASS_NAMES_CN.get(cls, cls)
        prob = probabilities.get(cls, 0.0)
        # 标注“核心置信度”的语义：仅当该类为最大概率类时在外层描述
        lines.append(f"{cn}（{cls}）：{prob:.1f}%")
    return lines


def build_report_header(avg_sec_per_image: float) -> List[str]:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = []
    header.append("脑部肿瘤分类推理报告")
    header.append("一、推理基础信息")
    header.append(f"推理时间：{now_str}")
    header.append("模型名称：脑肿瘤分类 CNN 模型（4 分类）")
    header.append(f"单张图平均推理耗时：≈ {avg_sec_per_image:.3f} s")
    return header


def build_single_image_block(index: int, predicted_class: str, confidence: float, probabilities: Dict[str, float], true_label: str = "") -> List[str]:
    lines = []
    idx_str = f"{index}. 图像 {index}"
    lines.append(idx_str)
    cn_pred = CLASS_NAMES_CN.get(predicted_class, predicted_class)
    lines.append(f"肿瘤类别：{cn_pred}（{predicted_class}）")

    # 置信度分布
    lines.append("预测置信度分布：")
    distribution_lines = format_distribution_lines(probabilities)
    # 将最大类说明为“核心置信度（对应原‘准确率’）”
    max_cls = max(probabilities, key=lambda k: probabilities[k])
    for i, line in enumerate(distribution_lines):
        if max_cls in line:
            distribution_lines[i] = line.replace(
                f"{CLASS_NAMES_CN.get(max_cls, max_cls)}（{max_cls}）：",
                f"{CLASS_NAMES_CN.get(max_cls, max_cls)}（{max_cls}）："
            ) + "（核心置信度）"
            break
    lines.extend(distribution_lines)

    # 结果解读
    if confidence >= 99.0:
        lines.append("结果解读：模型对该图像的肿瘤类型判断高度明确，置信度接近满分。")
    elif confidence >= 95.0:
        lines.append("结果解读：模型对该图像的判断非常可靠，置信度很高。")
    elif confidence >= 80.0:
        lines.append("结果解读：模型对该图像的判断较为可靠，建议结合临床进一步确认。")
    else:
        lines.append("结果解读：模型置信度一般，建议谨慎参考并结合更多证据评估。")

    # 可选真实标签（若从文件名中可解析）
    if true_label:
        lines.insert(2, f"真实类别：{CLASS_NAMES_CN.get(true_label, true_label)}（{true_label}）")

    return lines


def write_report_for_folder(model, folder_path: str, output_dir: str) -> str:
    # 收集图片（常见格式）
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(image_exts)]
    if not image_files:
        return ""

    per_image_times = []
    image_blocks: List[List[str]] = []

    for idx, image_path in enumerate(image_files, start=1):
        predicted_class, confidence, probs, elapsed = infer_image(model, image_path)
        per_image_times.append(elapsed)
        true_label = try_extract_true_label_from_filename(image_path)
        block = build_single_image_block(idx, predicted_class, confidence, probs, true_label)
        image_blocks.append(block)

    avg_time = sum(per_image_times) / max(len(per_image_times), 1)

    # 组装报告
    lines: List[str] = []
    lines.extend(build_report_header(avg_time))
    lines.append("二、单图推理详情（按图像顺序）")
    for block in image_blocks:
        lines.extend(block)
        lines.append("")  # 空行分隔

    # 写出
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    ensure_output_dir(output_dir)
    out_path = os.path.join(output_dir, f"report_{folder_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return out_path


def main():
    ensure_output_dir(OUTPUT_DIR)
    model = load_trained_model(MODEL_PATH)

    # 期望的子目录为 1..10
    subdirs = [d for d in sorted(os.listdir(INPUT_DIR)) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    # 若需严格限定 1..10，可按以下筛选：
    # subdirs = [d for d in subdirs if d.isdigit() and 1 <= int(d) <= 10]

    written_reports: List[str] = []
    for sub in subdirs:
        folder_path = os.path.join(INPUT_DIR, sub)
        out_path = write_report_for_folder(model, folder_path, OUTPUT_DIR)
        if out_path:
            print(f"已生成报告: {out_path}")
            written_reports.append(out_path)
        else:
            print(f"跳过（无图像）: {folder_path}")

    if not written_reports:
        print("未生成任何报告，请检查输入目录与文件结构。")
    else:
        print(f"报告已全部生成到目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()




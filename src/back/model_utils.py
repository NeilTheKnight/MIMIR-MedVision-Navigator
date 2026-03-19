import os
import numpy as np
from PIL import Image
from .base_model import load_tumor_model


def preprocess_image_with_info(image_path: str, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    img_resized = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return img, arr, original_size


def predict_tumor_with_stats(model, img_arr):
    prob = model.predict(img_arr, verbose=0)[0]  # (H, W, 1) or (H, W)
    if prob.ndim == 3:
        prob2d = prob[..., 0]
    else:
        prob2d = prob
    # simple threshold for mask
    mask = (prob2d > 0.5).astype(np.uint8) * 255
    # stats
    tumor_pixels = int(mask.sum() // 255)
    total_pixels = mask.size
    tumor_ratio = tumor_pixels / total_pixels
    return prob2d, mask, {
        'tumor_pixels': tumor_pixels,
        'total_pixels': total_pixels,
        'tumor_ratio': tumor_ratio,
    }


def create_probability_image(prob2d: np.ndarray):
    # map to [0,255]
    prob_norm = (prob2d - prob2d.min()) / (prob2d.max() - prob2d.min() + 1e-8)
    prob_img = (prob_norm * 255).astype(np.uint8)
    # apply jet colormap (red high probability)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    colored = (cmap(prob_norm) * 255).astype(np.uint8)  # RGBA
    colored_rgb = colored[..., :3]
    return Image.fromarray(colored_rgb)


def create_mask_image(mask2d: np.ndarray):
    return Image.fromarray(mask2d, mode='L')


def create_overlay_image(original_rgb: Image.Image, mask2d: np.ndarray, alpha=0.35):
    base = original_rgb.convert('RGBA')
    # color mask: red channel for tumor
    mask_rgb = np.zeros((*mask2d.shape, 4), dtype=np.uint8)
    mask_rgb[..., 0] = (mask2d > 0).astype(np.uint8) * 255  # R
    mask_rgb[..., 3] = (mask2d > 0).astype(np.uint8) * int(alpha * 255)  # A
    mask_img = Image.fromarray(mask_rgb, mode='RGBA')
    composed = Image.alpha_composite(base, mask_img)
    return composed.convert('RGB')


def save_inference_info(info_dir: str, image_name_noext: str, stats: dict):
    os.makedirs(info_dir, exist_ok=True)
    out_path = os.path.join(info_dir, f"{image_name_noext}_info.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {image_name_noext}\n")
        f.write(f"Tumor Pixels: {stats['tumor_pixels']}\n")
        f.write(f"Total Pixels: {stats['total_pixels']}\n")
        f.write(f"Tumor Ratio: {stats['tumor_ratio']:.6f}\n")
    return out_path
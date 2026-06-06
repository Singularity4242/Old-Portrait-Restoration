"""
Nicole

degradation_pipeline.py
=======================
为“老照片人像修复”构建 LQ-HQ 成对训练数据。

设计边界
--------
本脚本只模拟轻中度复合退化：
  - 非结构退化：模糊、噪声、JPEG 压缩、重采样、褪色、泛黄、饱和度下降
  - 轻微结构退化：细划痕、尘点、轻微污渍

本脚本不在主训练集中模拟大面积内容缺失：
  - 大块遮挡、照片撕裂、粗划痕遮挡五官、大片缺失区域

原因是 MambaIRv2 属于监督回归式图像复原模型，更适合恢复“信息仍然存在但质量下降”的图像。
大面积缺失更接近 inpainting 或生成式补全任务，应作为 limitation 单独分析。
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. 全局配置
# ============================================================

# 训练集退化强度分布。
# severe_controlled 只表示较强噪声、压缩、色偏和更多小缺陷，不包含大面积内容缺失。
LEVEL_PROBS = {
    'mild': 0.25,
    'medium': 0.60,
    'severe_controlled': 0.15,
}


# 每张训练样本的最低退化约束。
# 这样可以避免 LQ 与 HQ 几乎一致，导致模型过多学习 identity mapping。
MIN_TOTAL_DEGRADATIONS = 3


# 实体老化退化与数字化退化分开统计。
# 每张图至少应包含一类实体老化退化和一类数字化退化。
PHYSICAL_DEGRADATIONS = (
    'fading',
    'yellowing',
    'desaturation',
    'dust_spots',
    'thin_scratch',
    'mild_stain',
)
DIGITAL_DEGRADATIONS = (
    'blur',
    'resize',
    'noise',
    'jpeg',
)


# 各强度下的退化触发概率。
# 这些概率控制“是否出现某类退化”，不是具体强度参数。
PROBS = {
    'mild': dict(
        fading=0.35,
        yellowing=0.30,
        desaturation=0.25,
        dust=0.25,
        scratch=0.12,
        stain=0.10,
        blur=0.55,
        resize=0.35,
        noise=0.55,
        jpeg=0.45,
    ),
    'medium': dict(
        fading=0.60,
        yellowing=0.50,
        desaturation=0.40,
        dust=0.45,
        scratch=0.25,
        stain=0.20,
        blur=0.75,
        resize=0.55,
        noise=0.70,
        jpeg=0.65,
    ),
    'severe_controlled': dict(
        fading=0.75,
        yellowing=0.65,
        desaturation=0.55,
        dust=0.60,
        scratch=0.35,
        stain=0.30,
        blur=0.85,
        resize=0.70,
        noise=0.80,
        jpeg=0.75,
    ),
}


# ============================================================
# 2. 通用工具函数
# ============================================================

def clip_uint8(img):
    """把 float 图像裁剪到 [0, 255] 并转回 uint8。"""
    return np.clip(img, 0, 255).astype(np.uint8)


def sample_level(level_probs=None):
    """按给定概率采样退化强度。"""
    probs = level_probs or LEVEL_PROBS
    levels = list(probs.keys())
    weights = list(probs.values())
    return random.choices(levels, weights=weights, k=1)[0]


def append_meta(meta, op_type, **kwargs):
    """向单张图像的退化记录中追加一个操作及其参数。"""
    record = {'type': op_type}
    record.update(kwargs)
    meta.append(record)


# ============================================================
# 3. 实体老化退化
# ============================================================

def apply_fading(img, level='medium'):
    """
    模拟老照片整体褪色。

    做法：把图像向中性灰色 128 拉近。
    factor 越小，图像越灰、对比越弱。
    """
    ranges = {
        'mild': (0.78, 0.95),
        'medium': (0.55, 0.85),
        'severe_controlled': (0.40, 0.70),
    }
    factor = random.uniform(*ranges[level])
    faded = img.astype(np.float32) * factor + 128.0 * (1.0 - factor)
    return clip_uint8(faded), {'factor': round(factor, 4)}


def apply_yellowing(img, level='medium'):
    """
    模拟纸质照片老化后的泛黄和暖色偏。

    OpenCV 默认使用 BGR，因此这里降低 B 通道，轻微提升 R 通道。
    """
    ranges = {
        'mild': ((0.86, 0.96), (0.98, 1.03), (1.02, 1.08)),
        'medium': ((0.75, 0.92), (0.95, 1.02), (1.05, 1.14)),
        'severe_controlled': ((0.65, 0.86), (0.92, 1.00), (1.08, 1.20)),
    }
    b_range, g_range, r_range = ranges[level]
    b_scale = random.uniform(*b_range)
    g_scale = random.uniform(*g_range)
    r_scale = random.uniform(*r_range)

    out = img.astype(np.float32)
    out[:, :, 0] *= b_scale
    out[:, :, 1] *= g_scale
    out[:, :, 2] *= r_scale

    return clip_uint8(out), {
        'b_scale': round(b_scale, 4),
        'g_scale': round(g_scale, 4),
        'r_scale': round(r_scale, 4),
    }


def apply_desaturation(img, level='medium'):
    """
    模拟旧照颜色饱和度下降。

    做法：在原图和彩色转灰度图之间插值。
    strength 越大，颜色越接近灰度。
    """
    ranges = {
        'mild': (0.05, 0.18),
        'medium': (0.12, 0.35),
        'severe_controlled': (0.25, 0.50),
    }
    strength = random.uniform(*ranges[level])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = img.astype(np.float32) * (1.0 - strength) + gray_bgr * strength
    return clip_uint8(out), {'strength': round(strength, 4)}


def apply_dust_spots(img, level='medium'):
    """
    模拟尘点和小霉点。

    只生成小尺度、半透明圆点，不生成大面积遮挡。
    """
    ranges = {
        'mild': (8, 25, 1, 2, 0.10, 0.22),
        'medium': (20, 55, 1, 3, 0.14, 0.30),
        'severe_controlled': (45, 90, 1, 4, 0.18, 0.36),
    }
    min_num, max_num, min_radius, max_radius, min_alpha, max_alpha = ranges[level]
    h, w = img.shape[:2]
    overlay = img.copy().astype(np.float32)
    num_spots = random.randint(min_num, max_num)

    for _ in range(num_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(min_radius, max_radius)
        alpha = random.uniform(min_alpha, max_alpha)
        color = random.randint(170, 235)
        cv2.circle(overlay, (x, y), radius, (color, color, color), thickness=-1)
        img = clip_uint8(img.astype(np.float32) * (1.0 - alpha) + overlay * alpha)
        overlay = img.copy().astype(np.float32)

    return img, {'num_spots': num_spots, 'radius_range': [min_radius, max_radius]}


def apply_thin_scratch(img, level='medium'):
    """
    模拟轻中度细划痕。

    限制：
      - 线条较细；
      - 透明度有限；
      - 长度受控；
      - 不模拟撕裂或大面积缺失。
    """
    ranges = {
        'mild': (0, 2, 0.15, 0.32, 0.15, 0.45),
        'medium': (1, 4, 0.18, 0.38, 0.20, 0.60),
        'severe_controlled': (2, 6, 0.22, 0.45, 0.25, 0.70),
    }
    min_num, max_num, min_alpha, max_alpha, min_len, max_len = ranges[level]
    h, w = img.shape[:2]
    num_lines = random.randint(min_num, max_num)
    out = img.copy().astype(np.float32)

    for _ in range(num_lines):
        length = int(random.uniform(min_len, max_len) * min(h, w))
        angle = random.uniform(-np.pi, np.pi)
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        thickness = random.choice([1, 1, 1, 2])
        color = random.randint(175, 245)
        alpha = random.uniform(min_alpha, max_alpha)

        line_layer = out.copy()
        cv2.line(line_layer, (x1, y1), (x2, y2), (color, color, color), thickness)
        out = out * (1.0 - alpha) + line_layer * alpha

    return clip_uint8(out), {'num_lines': num_lines}


def apply_mild_stain(img, level='medium'):
    """
    模拟轻微局部污渍或纸张老化斑。

    只使用低透明度椭圆区域，避免形成内容缺失型大遮挡。
    """
    ranges = {
        'mild': (0, 1, 0.05, 0.12, 0.08, 0.16),
        'medium': (0, 2, 0.08, 0.18, 0.10, 0.22),
        'severe_controlled': (1, 3, 0.12, 0.24, 0.14, 0.28),
    }
    min_num, max_num, min_scale, max_scale, min_alpha, max_alpha = ranges[level]
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    num_stains = random.randint(min_num, max_num)

    for _ in range(num_stains):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        axis_x = max(3, int(random.uniform(min_scale, max_scale) * w))
        axis_y = max(3, int(random.uniform(min_scale, max_scale) * h))
        angle = random.randint(0, 180)
        alpha = random.uniform(min_alpha, max_alpha)

        # 先生成局部椭圆 mask，再对 mask 做模糊，避免整张图被额外模糊。
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, axis_x / 3))
        mask = np.clip(mask[..., None] * alpha, 0.0, 1.0)

        color = (
            random.randint(115, 150),
            random.randint(130, 165),
            random.randint(155, 195),
        )
        stain_color = np.zeros_like(out)
        stain_color[:, :] = color
        out = out * (1.0 - mask) + stain_color * mask

    return clip_uint8(out), {'num_stains': num_stains}


def apply_physical_aging(img, level, meta):
    """
    实体老化阶段。

    这一步模拟照片在纸质介质上发生的老化问题。
    """
    p = PROBS[level]

    if random.random() < p['fading']:
        img, params = apply_fading(img, level)
        append_meta(meta, 'fading', **params)

    if random.random() < p['yellowing']:
        img, params = apply_yellowing(img, level)
        append_meta(meta, 'yellowing', **params)

    if random.random() < p['desaturation']:
        img, params = apply_desaturation(img, level)
        append_meta(meta, 'desaturation', **params)

    if random.random() < p['dust']:
        img, params = apply_dust_spots(img, level)
        append_meta(meta, 'dust_spots', **params)

    if random.random() < p['scratch']:
        img, params = apply_thin_scratch(img, level)
        append_meta(meta, 'thin_scratch', **params)

    if random.random() < p['stain']:
        img, params = apply_mild_stain(img, level)
        append_meta(meta, 'mild_stain', **params)

    return img


# ============================================================
# 4. 数字化退化
# ============================================================

def motion_blur_kernel(kernel_size):
    """生成简单运动模糊核。"""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    angle = random.uniform(-30, 30)
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= kernel.sum() + 1e-8
    return kernel, angle


def apply_blur(img, level='medium'):
    """
    模拟扫描、拍摄或失焦造成的模糊。

    包含 Gaussian blur 和轻微 motion blur。
    """
    ranges = {
        'mild': ((3, 7), (0.4, 1.2), 0.15),
        'medium': ((3, 13), (0.7, 2.2), 0.25),
        'severe_controlled': ((5, 17), (1.2, 3.2), 0.35),
    }
    kernel_range, sigma_range, motion_prob = ranges[level]
    kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))

    if random.random() < motion_prob:
        kernel, angle = motion_blur_kernel(kernel_size)
        out = cv2.filter2D(img, -1, kernel)
        return out, {'blur_type': 'motion', 'kernel_size': kernel_size, 'angle': round(angle, 4)}

    sigma = random.uniform(*sigma_range)
    out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return out, {'blur_type': 'gaussian', 'kernel_size': kernel_size, 'sigma': round(sigma, 4)}


def apply_resize_degradation(img, level='medium'):
    """
    模拟数字化过程中的重采样退化。

    先随机缩小，再放回原尺寸。这样保持 LQ/HQ 尺寸一致，适配 scale=1 的复原任务。
    """
    ranges = {
        'mild': (0.75, 1.00),
        'medium': (0.55, 0.95),
        'severe_controlled': (0.42, 0.85),
    }
    h, w = img.shape[:2]
    scale = random.uniform(*ranges[level])
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    interp_down = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    interp_up = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])

    small = cv2.resize(img, (new_w, new_h), interpolation=interp_down)
    out = cv2.resize(small, (w, h), interpolation=interp_up)

    return out, {
        'scale': round(scale, 4),
        'down_size': [new_h, new_w],
        'interp_down': int(interp_down),
        'interp_up': int(interp_up),
    }


def apply_noise(img, level='medium'):
    """
    模拟扫描噪声和数字图像噪声。

    以 Gaussian noise 为主，少量使用 Poisson-like noise。
    """
    gaussian_ranges = {
        'mild': (2, 8),
        'medium': (5, 20),
        'severe_controlled': (12, 35),
    }
    poisson_ranges = {
        'mild': (0.006, 0.015),
        'medium': (0.012, 0.030),
        'severe_controlled': (0.020, 0.050),
    }

    if random.random() < 0.80:
        sigma = random.uniform(*gaussian_ranges[level])
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return clip_uint8(out), {'noise_type': 'gaussian', 'sigma': round(sigma, 4)}

    strength = random.uniform(*poisson_ranges[level])
    img_float = img.astype(np.float32) / 255.0
    noise = np.random.poisson(np.maximum(img_float, 0) / strength) * strength
    out = np.clip(noise * 255.0, 0, 255)
    return clip_uint8(out), {'noise_type': 'poisson_like', 'strength': round(strength, 5)}


def apply_jpeg_compression(img, level='medium'):
    """
    模拟 JPEG 保存或网络传播导致的压缩伪影。
    """
    ranges = {
        'mild': (68, 95),
        'medium': (38, 85),
        'severe_controlled': (22, 65),
    }
    quality = random.randint(*ranges[level])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', img, encode_param)
    if not success:
        return img, {'quality': quality, 'encode_success': False}
    out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out, {'quality': quality, 'encode_success': True}


def apply_digitization(img, level, meta):
    """
    数字化阶段。

    这一步模拟照片被扫描、拍摄、保存和传播时产生的退化。
    """
    p = PROBS[level]

    if random.random() < p['blur']:
        img, params = apply_blur(img, level)
        append_meta(meta, 'blur', **params)

    if random.random() < p['resize']:
        img, params = apply_resize_degradation(img, level)
        append_meta(meta, 'resize', **params)

    if random.random() < p['noise']:
        img, params = apply_noise(img, level)
        append_meta(meta, 'noise', **params)

    if random.random() < p['jpeg']:
        img, params = apply_jpeg_compression(img, level)
        append_meta(meta, 'jpeg', **params)

    return img


def meta_has_any(meta, op_types):
    """检查当前图像是否已经触发某一类退化。"""
    return any(op['type'] in op_types for op in meta)


def choose_missing_op(meta, candidates):
    """
    从候选退化中选择一个尚未触发过的操作。

    如果候选退化都已经出现，则允许重复选择。这样可以保证最低退化数量。
    """
    used = {op['type'] for op in meta}
    available = [op for op in candidates if op not in used]
    return random.choice(available or list(candidates))


def apply_specific_degradation(img, level, meta, op_type, forced=False):
    """
    直接应用指定退化操作。

    forced=True 表示该操作由最低退化约束补充触发。
    记录这个标记有助于后续检查每张图的生成原因。
    """
    if op_type == 'fading':
        img, params = apply_fading(img, level)
    elif op_type == 'yellowing':
        img, params = apply_yellowing(img, level)
    elif op_type == 'desaturation':
        img, params = apply_desaturation(img, level)
    elif op_type == 'dust_spots':
        img, params = apply_dust_spots(img, level)
    elif op_type == 'thin_scratch':
        img, params = apply_thin_scratch(img, level)
    elif op_type == 'mild_stain':
        img, params = apply_mild_stain(img, level)
    elif op_type == 'blur':
        img, params = apply_blur(img, level)
    elif op_type == 'resize':
        img, params = apply_resize_degradation(img, level)
    elif op_type == 'noise':
        img, params = apply_noise(img, level)
    elif op_type == 'jpeg':
        img, params = apply_jpeg_compression(img, level)
    else:
        raise ValueError(f'Unsupported degradation type: {op_type}')

    if forced:
        params['forced_by_minimum_constraint'] = True
    append_meta(meta, op_type, **params)
    return img


def enforce_minimum_degradation(img, level, meta):
    """
    保证每张 LQ 图像具有足够的训练信号。

    约束：
      1. 至少包含 1 个实体老化退化；
      2. 至少包含 1 个数字化退化；
      3. 总退化操作数不少于 MIN_TOTAL_DEGRADATIONS。

    这样可以减少“肉眼几乎看不出退化”的样本比例，同时保留少量轻度样本的正则化价值。
    """
    if not meta_has_any(meta, PHYSICAL_DEGRADATIONS):
        op_type = choose_missing_op(meta, PHYSICAL_DEGRADATIONS)
        img = apply_specific_degradation(img, level, meta, op_type, forced=True)

    if not meta_has_any(meta, DIGITAL_DEGRADATIONS):
        op_type = choose_missing_op(meta, DIGITAL_DEGRADATIONS)
        img = apply_specific_degradation(img, level, meta, op_type, forced=True)

    while len(meta) < MIN_TOTAL_DEGRADATIONS:
        op_type = choose_missing_op(meta, PHYSICAL_DEGRADATIONS + DIGITAL_DEGRADATIONS)
        img = apply_specific_degradation(img, level, meta, op_type, forced=True)

    return img


# ============================================================
# 5. 单张图像退化入口
# ============================================================

def degrade(img, level=None, level_probs=None):
    """
    对单张 HQ 图像生成 LQ 图像。

    返回：
      lq   : 退化后的图像
      meta : 本图像触发的退化操作和具体参数
    """
    if level is None:
        level = sample_level(level_probs)

    meta = []

    # 先模拟实体照片老化，再模拟数字化过程。
    img = apply_physical_aging(img, level, meta)
    img = apply_digitization(img, level, meta)
    img = enforce_minimum_degradation(img, level, meta)

    return img, {
        'level': level,
        'ops': meta,
    }


# ============================================================
# 6. 数据集构建
# ============================================================

def collect_images(src_dir):
    """收集源目录中的图像路径。"""
    src = Path(src_dir)
    suffixes = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']
    paths = []
    for suffix in suffixes:
        paths.extend(src.rglob(suffix))
    return sorted(
        set(path for path in paths if not path.name.startswith('._') and not path.name.startswith('.'))
    )


def prepare_hq_image(img, target_size):
    """
    生成统一尺寸的 HQ 图像。

    FFHQ 已经是对齐人脸图像，因此这里直接 resize 到 target_size。
    """
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)


def update_stats(stats, split_name, level, meta):
    """更新数据集统计信息。"""
    stats[split_name]['count'] += 1
    stats[split_name]['levels'][level] = stats[split_name]['levels'].get(level, 0) + 1

    for op in meta['ops']:
        op_type = op['type']
        stats[split_name]['ops_count'][op_type] = stats[split_name]['ops_count'].get(op_type, 0) + 1


def build_dataset(
    src_dir,
    out_dir,
    num_images=20000,
    target_size=256,
    split=(0.85, 0.05, 0.10),
    seed=42,
    sample_mode='first',
    level_probs=None,
):
    """
    构建老照片人像 LQ-HQ 成对数据集。

    Parameters
    ----------
    src_dir : str
        FFHQ 1024x1024 源图像目录。
    out_dir : str
        输出根目录。
    num_images : int
        使用多少张源图像。默认使用前 20000 张。
    target_size : int
        输出 HQ/LQ 的统一尺寸。建议 256。
    split : tuple[float, float, float]
        train/val/test 比例。
    seed : int
        随机种子，用于退化参数和可选随机采样。
    sample_mode : str
        'first' 表示使用排序后的前 num_images 张；
        'random' 表示固定随机种子后随机采样 num_images 张。
    level_probs : dict | None
        每张图采样 mild/medium/severe_controlled 的概率。
    """
    assert abs(sum(split) - 1.0) < 1e-6, 'split 比例之和必须为 1'
    assert sample_mode in {'first', 'random'}, "sample_mode 必须是 'first' 或 'random'"

    random.seed(seed)
    np.random.seed(seed)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_imgs = collect_images(src_dir)
    if not all_imgs:
        raise FileNotFoundError(f'没有在源目录中找到图像: {src_dir}')

    if sample_mode == 'random':
        random.shuffle(all_imgs)

    selected_imgs = all_imgs[:num_images]
    if len(selected_imgs) < num_images:
        print(f'[warn] 仅找到 {len(selected_imgs)} 张图像，少于请求的 {num_images} 张。')

    n = len(selected_imgs)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    splits = {
        'train': selected_imgs[:n_train],
        'val': selected_imgs[n_train:n_train + n_val],
        'test': selected_imgs[n_train + n_val:],
    }

    # dataset_stats.json 记录整体统计，degradation_meta.jsonl 记录逐图参数。
    stats = {
        split_name: {
            'count': 0,
            'levels': {},
            'ops_count': {},
        }
        for split_name in splits
    }
    config = {
        'src_dir': str(src_dir),
        'out_dir': str(out_dir),
        'num_images': n,
        'target_size': target_size,
        'split': split,
        'seed': seed,
        'sample_mode': sample_mode,
        'level_probs': level_probs or LEVEL_PROBS,
        'minimum_constraints': {
            'min_total_degradations': MIN_TOTAL_DEGRADATIONS,
            'require_physical_degradation': True,
            'require_digital_degradation': True,
        },
        'note': 'Main training degradation excludes large missing regions and severe content-loss defects.',
    }

    meta_path = out_root / 'degradation_meta.jsonl'
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        for split_name, img_list in splits.items():
            hq_dir = out_root / split_name / 'HQ'
            lq_dir = out_root / split_name / 'LQ'
            hq_dir.mkdir(parents=True, exist_ok=True)
            lq_dir.mkdir(parents=True, exist_ok=True)

            print(f'\n[{split_name}] 处理 {len(img_list)} 张图...')

            for i, img_path in enumerate(img_list):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f'  [skip] 读取失败: {img_path.name}')
                    continue

                hq = prepare_hq_image(img, target_size)
                lq, meta = degrade(hq.copy(), level=None, level_probs=level_probs)

                out_name = f'{img_path.stem}.png'
                cv2.imwrite(str(hq_dir / out_name), hq)
                cv2.imwrite(str(lq_dir / out_name), lq)

                update_stats(stats, split_name, meta['level'], meta)

                meta_record = {
                    'filename': out_name,
                    'source': str(img_path),
                    'split': split_name,
                    'target_size': target_size,
                    'level': meta['level'],
                    'ops': meta['ops'],
                }
                meta_file.write(json.dumps(meta_record, ensure_ascii=False) + '\n')

                if (i + 1) % 500 == 0:
                    print(f'  {i + 1}/{len(img_list)}')

    stats_path = out_root / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'config': config, 'stats': stats}, f, indent=2, ensure_ascii=False)

    print('\n完成！数据集已生成。')
    print(f'  输出目录: {out_root}')
    print(f'  统计文件: {stats_path}')
    print(f'  逐图退化记录: {meta_path}')
    for split_name in ['train', 'val', 'test']:
        print(f"  {split_name}: {stats[split_name]['count']} 张")


# ============================================================
# 7. 命令行入口
# ============================================================

def parse_args():
    """解析命令行参数，便于在服务器上直接修改路径和规模。"""
    parser = argparse.ArgumentParser(description='Build old portrait restoration paired dataset.')
    parser.add_argument(
        '--src_dir',
        default='datasets/example',
        #default='/root/autodl-tmp/MambaIR/FFHQ/images1024x1024',
        help='FFHQ 1024x1024 图像目录。',
    )
    parser.add_argument(
        '--out_dir',
        default='datasets/example/paired',
        #default='/root/autodl-tmp/MambaIR/my-test/data/old_portrait',
        help='输出 LQ-HQ paired 数据集目录。',
    )
    parser.add_argument('--num_images', type=int, default=5, help='使用图像数量。')
    parser.add_argument('--target_size', type=int, default=256, help='输出图像尺寸。')
    parser.add_argument('--seed', type=int, default=42, help='随机种子。')
    parser.add_argument(
        '--sample_mode',
        choices=['first', 'random'],
        default='first',
        help='first 使用排序后的前 num_images 张；random 使用固定随机种子随机采样。',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_dataset(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        num_images=args.num_images,
        target_size=args.target_size,
        split=(0.85, 0.05, 0.10),
        seed=args.seed,
        sample_mode=args.sample_mode,
        level_probs=LEVEL_PROBS,
    )

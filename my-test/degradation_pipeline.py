"""
degradation_pipeline.py  (v2)
==============================
改进点 vs 原版：
  1. 自动划分 train / val / test 三个子集
  2. 增加"退化强度等级"参数（mild / medium / severe）
  3. 128x128 缩略图过小 -> 生成时自动 resize 到 target_size
  4. 统计并打印退化参数分布，便于论文描述
  5. 保留原始文件名，方便对照检查
"""

import cv2
import numpy as np
import random
import json
from pathlib import Path


# ─────────────────────────────────────────────
# 各种退化函数
# ─────────────────────────────────────────────

def apply_gaussian_blur(img, level='medium'):
    ranges = {
        'mild':   ((3,  7), (0.5, 1.5)),
        'medium': ((3, 15), (0.5, 3.0)),
        'severe': ((7, 21), (1.5, 5.0)),
    }
    kr, sr = ranges[level]
    kernel_size = random.choice(range(kr[0], kr[1] + 1, 2))
    sigma = random.uniform(*sr)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_noise(img, level='medium'):
    ranges = {'mild': (2, 10), 'medium': (5, 30), 'severe': (20, 60)}
    sigma = random.uniform(*ranges[level])
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img, level='medium'):
    ranges = {'mild': (60, 90), 'medium': (30, 80), 'severe': (10, 50)}
    quality = random.randint(*ranges[level])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_fading(img, level='medium'):
    ranges = {'mild': (0.7, 0.95), 'medium': (0.4, 0.8), 'severe': (0.2, 0.6)}
    factor = random.uniform(*ranges[level])
    faded = img.astype(np.float32) * factor + 128 * (1 - factor)
    return np.clip(faded, 0, 255).astype(np.uint8)


def apply_yellowing(img, level='medium'):
    strengths = {'mild': 0.5, 'medium': 1.0, 'severe': 1.5}
    s = strengths[level]
    img = img.astype(np.float32)
    img[:, :, 0] *= random.uniform(0.7, 0.7 + 0.2 * (1 - s / 1.5))  # B 降低
    img[:, :, 1] *= random.uniform(0.9, 1.0)                          # G 轻微
    img[:, :, 2] *= random.uniform(1.0, 1.0 + 0.1 * s)               # R 提升
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_scratch(img, level='medium'):
    ranges = {'mild': (0, 2), 'medium': (1, 5), 'severe': (3, 10)}
    img = img.copy()
    h, w = img.shape[:2]
    num = random.randint(*ranges[level])
    for _ in range(num):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        color = random.randint(160, 255)
        thickness = random.randint(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), (color, color, color), thickness)
    return img


# ─────────────────────────────────────────────
# 退化概率表
# ─────────────────────────────────────────────

PROBS = {
    'mild':   dict(blur=0.6, noise=0.6, jpeg=0.5, fading=0.3, yellowing=0.3, scratch=0.2),
    'medium': dict(blur=0.8, noise=0.8, jpeg=0.7, fading=0.6, yellowing=0.5, scratch=0.5),
    'severe': dict(blur=0.9, noise=0.9, jpeg=0.8, fading=0.8, yellowing=0.7, scratch=0.7),
}


def degrade(img, level='medium'):
    p = PROBS[level]
    ops = []
    if random.random() < p['blur']:
        img = apply_gaussian_blur(img, level); ops.append('blur')
    if random.random() < p['noise']:
        img = apply_noise(img, level); ops.append('noise')
    if random.random() < p['jpeg']:
        img = apply_jpeg_compression(img, level); ops.append('jpeg')
    if random.random() < p['fading']:
        img = apply_fading(img, level); ops.append('fading')
    if random.random() < p['yellowing']:
        img = apply_yellowing(img, level); ops.append('yellowing')
    if random.random() < p['scratch']:
        img = apply_scratch(img, level); ops.append('scratch')
    return img, ops


# ─────────────────────────────────────────────
# 数据集构建
# ─────────────────────────────────────────────

def build_dataset(
    src_dir,
    out_dir,
    num_images=5000,
    target_size=256,
    level='medium',
    split=(0.85, 0.05, 0.10),
    seed=42,
):
    """
    Parameters
    ----------
    src_dir     : 原始 FFHQ 目录（png 文件）
    out_dir     : 输出根目录
    num_images  : 最多使用多少张原图
    target_size : HQ 图 resize 到此分辨率
    level       : 退化强度 'mild' | 'medium' | 'severe'
    split       : (train, val, test) 比例
    seed        : 随机种子
    """
    assert abs(sum(split) - 1.0) < 1e-6, "split 比例之和必须为 1"
    random.seed(seed)
    np.random.seed(seed)

    all_imgs = sorted(Path(src_dir).glob('*.png'))
    if not all_imgs:
        all_imgs = sorted(Path(src_dir).glob('*.jpg'))
    all_imgs = all_imgs[:num_images]
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    splits  = {
        'train': all_imgs[:n_train],
        'val':   all_imgs[n_train:n_train + n_val],
        'test':  all_imgs[n_train + n_val:],
    }

    stats = {s: {'ops_count': {}, 'count': 0} for s in splits}

    for split_name, img_list in splits.items():
        hq_dir = Path(out_dir) / split_name / 'HQ'
        lq_dir = Path(out_dir) / split_name / 'LQ'
        hq_dir.mkdir(parents=True, exist_ok=True)
        lq_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split_name}] 处理 {len(img_list)} 张图...")

        for i, img_path in enumerate(img_list):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  跳过（读取失败）: {img_path.name}")
                continue

            if img.shape[0] != target_size or img.shape[1] != target_size:
                img = cv2.resize(img, (target_size, target_size),
                                 interpolation=cv2.INTER_LANCZOS4)

            lq, ops = degrade(img.copy(), level)

            stem = img_path.stem
            cv2.imwrite(str(hq_dir / f"{stem}.png"), img)
            cv2.imwrite(str(lq_dir / f"{stem}.png"), lq)

            stats[split_name]['count'] += 1
            for op in ops:
                stats[split_name]['ops_count'][op] = \
                    stats[split_name]['ops_count'].get(op, 0) + 1

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(img_list)}")

    stats_path = Path(out_dir) / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n完成！数据保存在 {out_dir}")
    print(f"  train : {stats['train']['count']}")
    print(f"  val   : {stats['val']['count']}")
    print(f"  test  : {stats['test']['count']}")
    print(f"\n退化操作触发率（train）:")
    n_tr = stats['train']['count'] or 1
    for op, cnt in sorted(stats['train']['ops_count'].items()):
        print(f"  {op:<12}: {cnt / n_tr * 100:.1f}%")
    print(f"\n统计文件: {stats_path}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    build_dataset(
        src_dir     = '/root/autodl-tmp/MambaIR/FFHQ/thumbnails128x128',
        out_dir     = '/root/autodl-tmp/MambaIR/datasets/old_portrait',
        num_images  = 5000,
        target_size = 256,   # 资源不足可改为 128
        level       = 'medium',
        split       = (0.85, 0.05, 0.10),
        seed        = 42,
    )
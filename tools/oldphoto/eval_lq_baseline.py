# 【NICOLE2026】
"""测试集 LQ-vs-HQ 基线指标(修复前的输入质量),作为论文消融表的参照行。

复用 basicsr 注册的同一套指标函数(calculate_psnr/ssim/face_psnr/face_ssim/lpips),
确保和 basicsr/test.py 跑各臂时的口径完全一致 —— 这样 "LQ 基线" 这一行可以和
A0/A1/A2/v2 的数字直接同表对比,看每个方法相对退化输入提升了多少。

服务器用法(默认就是 test 集路径，net=vgg 与测试配置一致):
    cd /root/autodl-tmp/MambaIR
    python tools/oldphoto/eval_lq_baseline.py
首次跑前先缓存 vgg：python -c "import lpips; lpips.LPIPS(net='vgg')"
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from basicsr.metrics import (calculate_face_psnr, calculate_face_ssim, calculate_lpips,
                             calculate_psnr, calculate_ssim)

METRIC_KEYS = ['psnr', 'ssim', 'face_psnr', 'face_ssim', 'lpips']


def main():
    p = argparse.ArgumentParser(description='LQ-vs-HQ baseline metrics on the test set.')
    p.add_argument('--hq_dir', default='/root/autodl-tmp/MambaIR/datasets/FFHQ_pair/test/HQ')
    p.add_argument('--lq_dir', default='/root/autodl-tmp/MambaIR/datasets/FFHQ_pair/test/LQ')
    p.add_argument('--face_weight_dir', default='/root/autodl-tmp/MambaIR/datasets/FFHQ_pair/test/face_weight')
    p.add_argument('--net', default='vgg', help='LPIPS 主干，与测试配置保持一致(vgg)。')
    p.add_argument('--crop_border', type=int, default=0)
    p.add_argument('--mask_threshold', type=int, default=1)
    args = p.parse_args()

    hq_dir, lq_dir, fw_dir = Path(args.hq_dir), Path(args.lq_dir), Path(args.face_weight_dir)
    names = sorted(q.name for q in hq_dir.glob('*.png'))
    if not names:
        raise FileNotFoundError(f'{hq_dir} 下没有 png。')

    sums = {k: 0.0 for k in METRIC_KEYS}
    n = 0
    for name in names:
        hq = cv2.imread(str(hq_dir / name))                          # GT  -> img2
        lq = cv2.imread(str(lq_dir / name))                          # 退化输入 -> img
        fw = cv2.imread(str(fw_dir / name), cv2.IMREAD_GRAYSCALE)    # 人脸掩码 (H,W) 0/76/255
        if hq is None or lq is None or fw is None:
            print(f'[skip] 缺文件: {name}')
            continue
        if lq.shape != hq.shape:  # scale=1 下本应同尺寸，防御性对齐
            lq = cv2.resize(lq, (hq.shape[1], hq.shape[0]), interpolation=cv2.INTER_LINEAR)

        sums['psnr'] += calculate_psnr(lq, hq, args.crop_border, test_y_channel=False)
        sums['ssim'] += calculate_ssim(lq, hq, args.crop_border, test_y_channel=False)
        sums['face_psnr'] += calculate_face_psnr(lq, hq, fw, args.crop_border, mask_threshold=args.mask_threshold)
        sums['face_ssim'] += calculate_face_ssim(lq, hq, fw, args.crop_border, mask_threshold=args.mask_threshold)
        sums['lpips'] += calculate_lpips(lq, hq, crop_border=args.crop_border, net=args.net)
        n += 1
        if n % 200 == 0:
            print(f'  processed {n}/{len(names)}')

    if n == 0:
        raise RuntimeError('没有成功处理任何图像。')

    print(f'\nLQ-vs-HQ baseline over {n} images (LPIPS net={args.net}):')
    for k in METRIC_KEYS:
        print(f'  {k:<10s} {sums[k] / n:.4f}')


if __name__ == '__main__':
    main()
# 【NICOLE2026】

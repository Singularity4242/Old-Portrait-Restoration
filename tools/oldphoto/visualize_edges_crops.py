# 【NICOLE2026】
"""可视化对比：整图 / 眼-嘴 crop / Sobel 边缘图。

用途：
  1. 看 Sobel 损失是否真的让边缘更接近 GT —— 对比 Sobel(A0) vs Sobel(A1) vs Sobel(GT)。
  2. 在对齐人脸（FFHQ 256）上裁固定的眼/嘴区域做锐度目检。

本脚本读取已保存的 PNG（用 basicsr/test.py + save_img:true 产出修复图），不依赖 torch。
Sobel 核与 FaceRegionWeightedSobelCharbonnierLoss 中的完全一致（/4.0 + reflect pad）。

示例（A0 vs A1 对照，看 8 张）：
    python tools/oldphoto/visualize_edges_crops.py \
        --gt_dir datasets/FFHQ_pair/val_200/HQ \
        --lq_dir datasets/FFHQ_pair/val_200/LQ \
        --restored_dirs experiments/.../A0_vis experiments/.../A1_vis \
        --labels A0 A1 \
        --out_dir viz_A0_vs_A1 --max_n 8

只看 A1 单个也可以：--restored_dirs <A1_vis> --labels A1

训练中途目检（读 BasicSR 训练时 save_img 存的嵌套目录，不中断训练）：
    python tools/oldphoto/visualize_edges_crops.py \
        --gt_dir datasets/FFHQ_pair/val_200/HQ \
        --restored_dirs experiments/MambaIRv2_OldPhoto_NICOLE_A1_charb_sobel_30k/visualization \
        --labels A1 --iter 5000 \
        --out_dir viz_A1_iter5000 --max_n 8
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# 与损失一致的 Sobel 核
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0


def sobel_magnitude(img_bgr):
    """逐通道 Sobel 后取幅值，再对通道求均值，返回单通道 float 边缘强度图。"""
    img = img_bgr.astype(np.float32)
    mags = []
    for c in range(img.shape[2]):
        gx = cv2.filter2D(img[:, :, c], -1, SOBEL_X, borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(img[:, :, c], -1, SOBEL_Y, borderType=cv2.BORDER_REFLECT)
        mags.append(np.sqrt(gx * gx + gy * gy))
    return np.mean(mags, axis=0)


def edge_to_bgr(mag, scale):
    """边缘强度图按给定 scale 归一化到 0-255 灰度（共享 scale 保证多图亮度可比）。"""
    vis = np.clip(mag / (scale + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def label_tile(tile, text, tile_size):
    """把内容缩放到 tile_size 方块并在左上角标注文字。"""
    tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(tile, (0, 0), (tile_size, 18), (0, 0, 0), -1)
    cv2.putText(tile, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def crop_box(img, box):
    """box = (x0, y0, x1, y1)，越界自动夹紧。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    return img[y0:y1, x0:x1]


def parse_box(s):
    return tuple(int(v) for v in s.split(','))


def resolve_restored(d, name, iteration):
    """定位单张修复图路径。

    iteration 给定 -> BasicSR 训练时布局 <dir>/<stem>/<stem>_<iter>.png。
    iteration None -> 依次尝试：
        1. 扁平 <dir>/<name>.png
        2. BasicSR test 布局 <dir>/<stem>_<configname>.png
           （test.py 在 suffix 为空时会把 yml 的 name 追加到文件名）。
    """
    if iteration is not None:
        stem = Path(name).stem
        return d / stem / f'{stem}_{iteration}.png'
    if (d / name).exists():
        return d / name
    stem = Path(name).stem
    matches = sorted(d.glob(f'{stem}_*.png'))
    if matches:
        return matches[0]
    return d / name  # 找不到则原样返回，交由上层 imread 失败并跳过


def main():
    p = argparse.ArgumentParser(description='Edge / eye-mouth crop comparison.')
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--restored_dirs', nargs='+', required=True, help='1 或 2 个修复结果目录。')
    p.add_argument('--labels', nargs='+', required=True, help='与 restored_dirs 一一对应的列名。')
    p.add_argument('--lq_dir', default=None, help='可选，加一列退化输入。')
    p.add_argument('--out_dir', default='viz_out')
    p.add_argument('--iter', type=int, default=None,
                   help='给定则读 BasicSR 训练嵌套布局 <dir>/<stem>/<stem>_<iter>.png（训练中途目检用）；'
                        '不给则读扁平 <dir>/<name>.png。')
    p.add_argument('--max_n', type=int, default=8, help='最多可视化多少张。')
    p.add_argument('--tile', type=int, default=256, help='每个方块显示尺寸。')
    # 默认框针对对齐 FFHQ-256；先跑一张目检后按需调整
    p.add_argument('--eye_box', type=parse_box, default=(55, 95, 200, 150), help='x0,y0,x1,y1')
    p.add_argument('--mouth_box', type=parse_box, default=(85, 160, 170, 212), help='x0,y0,x1,y1')
    args = p.parse_args()

    if len(args.labels) != len(args.restored_dirs):
        raise ValueError('labels 数量必须与 restored_dirs 一致。')

    gt_dir = Path(args.gt_dir)
    restored_dirs = [Path(d) for d in args.restored_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = args.tile

    names = sorted(pp.name for pp in gt_dir.glob('*.png'))[:args.max_n]
    if not names:
        raise FileNotFoundError(f'{gt_dir} 下没有 png。')

    for name in names:
        gt = cv2.imread(str(gt_dir / name))
        if gt is None:
            print(f'[skip] GT 缺失: {name}')
            continue
        restored = []
        ok = True
        for d in restored_dirs:
            r = cv2.imread(str(resolve_restored(d, name, args.iter)))
            if r is None:
                print(f'[skip] {d} 缺 {name}（iter={args.iter}）')
                ok = False
                break
            restored.append(r)
        if not ok:
            continue

        # 列顺序：[LQ?] + 各修复结果 + GT
        cols, col_labels = [], []
        if args.lq_dir:
            lq = cv2.imread(str(Path(args.lq_dir) / name))
            if lq is not None:
                cols.append(lq)
                col_labels.append('LQ')
        for img, lab in zip(restored, args.labels):
            cols.append(img)
            col_labels.append(lab)
        cols.append(gt)
        col_labels.append('GT')

        # Sobel 行用 GT 边缘最大值作共享 scale，保证各列亮度可比
        edge_scale = float(sobel_magnitude(gt).max())

        rows = []
        for row_name, transform in [
            ('full', lambda im: im),
            ('eye', lambda im: crop_box(im, args.eye_box)),
            ('mouth', lambda im: crop_box(im, args.mouth_box)),
            ('sobel', lambda im: edge_to_bgr(sobel_magnitude(im), edge_scale)),
        ]:
            tiles = []
            for img, lab in zip(cols, col_labels):
                content = transform(img)
                tiles.append(label_tile(content, f'{lab}-{row_name}', T))
            rows.append(cv2.hconcat(tiles))
        panel = cv2.vconcat(rows)

        out_path = out_dir / f'{Path(name).stem}_panel.png'
        cv2.imwrite(str(out_path), panel)
        print(f'[ok] {out_path}')

    print(f'\n完成，共写入 {out_dir}/。眼/嘴框默认针对对齐 FFHQ-256，'
          f'若位置不准用 --eye_box / --mouth_box 调整。')


if __name__ == '__main__':
    main()
# 【NICOLE2026】

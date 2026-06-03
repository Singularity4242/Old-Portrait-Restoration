"""
baseline_eval.py
=================
在 old_portrait 测试集上评估 MambaIRv2，输出：
  1. 每张图的 PSNR / SSIM / LPIPS
  2. 汇总均值（console + CSV）
  3. LQ | Restored | HQ 横排对比图

使用方式
--------
# 验证流程（随机权重，不需要训练好的模型）
python my-test/baseline_eval.py --mode random --max_imgs 20

# 评估训练好的 checkpoint
python my-test/baseline_eval.py \
    --checkpoint experiments/MambaIRv2_OldPortrait/models/net_g_200000.pth \
    --mode trained
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basicsr.archs.mambairv2_arch import MambaIRv2


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """输入 uint8 HWC RGB"""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def _ssim_channel(pred: np.ndarray, gt: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(pred, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gt,   (11, 11), 1.5)
    mu1_sq = mu1 ** 2; mu2_sq = mu2 ** 2; mu12 = mu1 * mu2
    s1  = cv2.GaussianBlur(pred ** 2, (11, 11), 1.5) - mu1_sq
    s2  = cv2.GaussianBlur(gt ** 2,   (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(pred * gt, (11, 11), 1.5) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-12)))


def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(np.float64); g = gt.astype(np.float64)
    return float(np.mean([_ssim_channel(p[..., c], g[..., c]) for c in range(3)]))


def calc_lpips(pred_t: torch.Tensor, gt_t: torch.Tensor, vgg=None) -> float:
    """LPIPS 近似；vgg=None 时退化为像素 L2（仅流程验证用）"""
    if vgg is None:
        return float((pred_t - gt_t).pow(2).mean().sqrt().item())
    with torch.no_grad():
        f1 = vgg(pred_t); f2 = vgg(gt_t)
    return float(F.mse_loss(f1, f2).item())


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def build_model(checkpoint: str = None) -> torch.nn.Module:
    net = MambaIRv2(
        upscale=1,
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=60,
        mlp_ratio=2.0,
        upsampler='',
        resi_connection='1conv',
    )

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location='cpu')
        if 'params_ema' in state:
            state = state['params_ema']
        elif 'params' in state:
            state = state['params']
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[model] 加载权重: {checkpoint}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected  : {len(unexpected)}")
    else:
        print("[model] 未指定权重，使用随机初始化（仅用于流程验证）")

    return net.eval()


# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────

@torch.no_grad()
def restore_image(net, lq_bgr: np.ndarray, device, tile=None) -> np.ndarray:
    """lq_bgr: uint8 HWC BGR  →  返回 uint8 HWC BGR"""
    lq_rgb = cv2.cvtColor(lq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(lq_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, H, W = inp.shape

    if tile is None:
        out = net(inp)
    else:
        stride = tile - tile // 8
        out   = torch.zeros_like(inp)
        count = torch.zeros_like(inp)
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                ye = min(y + tile, H); xe = min(x + tile, W)
                patch = inp[:, :, y:ye, x:xe]
                pred  = net(patch)
                out[:, :, y:ye, x:xe]   += pred
                count[:, :, y:ye, x:xe] += 1
        out = out / count.clamp(min=1)

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def make_visual(lq, restored, hq):
    h, w = hq.shape[:2]
    lq_r  = cv2.resize(lq,       (w, h))
    res_r = cv2.resize(restored, (w, h))
    gap   = np.full((h, 4, 3), 200, dtype=np.uint8)
    row   = np.concatenate([lq_r, gap, res_r, gap, hq], axis=1)
    for i, txt in enumerate(['LQ (input)', 'Restored', 'HQ (GT)']):
        x = i * (w + 4) + 4
        cv2.putText(row, txt, (x, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 200, 50), 1)
    return row


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lq_dir',     default='datasets/old_portrait/test/LQ')
    parser.add_argument('--hq_dir',     default='datasets/old_portrait/test/HQ')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out_dir',    default='experiments/baseline_eval')
    parser.add_argument('--mode',       choices=['random', 'pretrained', 'trained'],
                        default='random')
    parser.add_argument('--tile',       type=int, default=None,
                        help='显存不足时分块推理，如 --tile 128')
    parser.add_argument('--max_imgs',   type=int, default=200)
    parser.add_argument('--save_vis',   action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    net = build_model(args.checkpoint).to(device)

    # 尝试加载 VGG 用于 LPIPS 近似
    vgg_feat = None
    try:
        import torchvision.models as tv
        vgg = tv.vgg16(pretrained=True).features[:16].eval().to(device)
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std_t  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        class VGGFeat(torch.nn.Module):
            def __init__(self, v, m, s):
                super().__init__()
                self.vgg = v; self.mean = m; self.std = s
            def forward(self, x):
                return self.vgg((x - self.mean) / self.std)

        vgg_feat = VGGFeat(vgg, mean_t, std_t)
        print("[lpips] VGG16 加载成功")
    except Exception as e:
        print(f"[lpips] VGG 不可用（{e}），使用像素 L2 代替")

    # 收集测试图
    lq_dir   = Path(args.lq_dir)
    hq_dir   = Path(args.hq_dir)
    lq_paths = sorted(lq_dir.glob('*.png'))[:args.max_imgs]
    if not lq_paths:
        lq_paths = sorted(lq_dir.glob('*.jpg'))[:args.max_imgs]
    print(f"[data] {len(lq_paths)} 张测试图")

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / 'vis'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    records = []
    t0 = time.time()

    for i, lq_p in enumerate(lq_paths):
        hq_p = hq_dir / lq_p.name
        if not hq_p.exists():
            print(f"  [warn] GT 不存在: {lq_p.name}，跳过")
            continue

        lq_bgr = cv2.imread(str(lq_p))
        hq_bgr = cv2.imread(str(hq_p))
        if lq_bgr is None or hq_bgr is None:
            continue

        restored = restore_image(net, lq_bgr, device, tile=args.tile)

        hq_rgb  = cv2.cvtColor(hq_bgr,  cv2.COLOR_BGR2RGB)
        res_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        lq_rgb  = cv2.cvtColor(lq_bgr,  cv2.COLOR_BGR2RGB)

        p_score  = calc_psnr(res_rgb, hq_rgb)
        s_score  = calc_ssim(res_rgb, hq_rgb)
        p_lq     = calc_psnr(lq_rgb,  hq_rgb)

        def to_t(arr):
            return torch.from_numpy(
                arr.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

        lp_score = calc_lpips(to_t(res_rgb), to_t(hq_rgb), vgg_feat)

        records.append({
            'name':    lq_p.name,
            'psnr_lq': round(p_lq,    4),
            'psnr':    round(p_score,  4),
            'ssim':    round(s_score,  4),
            'lpips':   round(lp_score, 6),
        })

        if args.save_vis:
            vis = make_visual(lq_bgr, restored, hq_bgr)
            cv2.imwrite(str(vis_dir / lq_p.name), vis)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(lq_paths)}]  "
                  f"PSNR={p_score:.2f}  SSIM={s_score:.4f}  "
                  f"LPIPS={lp_score:.4f}  ({time.time()-t0:.0f}s)")

    if not records:
        print("没有结果，请检查数据路径。")
        return

    avg_psnr_lq = np.mean([r['psnr_lq'] for r in records])
    avg_psnr    = np.mean([r['psnr']     for r in records])
    avg_ssim    = np.mean([r['ssim']     for r in records])
    avg_lpips   = np.mean([r['lpips']    for r in records])

    print("\n" + "=" * 52)
    print(f"  评估图片数       : {len(records)}")
    print(f"  PSNR (LQ 输入)  : {avg_psnr_lq:.2f} dB  <- LQ 基准")
    print(f"  PSNR (restored) : {avg_psnr:.2f} dB")
    print(f"  SSIM (restored) : {avg_ssim:.4f}")
    print(f"  LPIPS(restored) : {avg_lpips:.4f}")
    print("=" * 52)

    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['name', 'psnr_lq', 'psnr', 'ssim', 'lpips'])
        writer.writeheader()
        writer.writerows(records)
        writer.writerow({
            'name':    'MEAN',
            'psnr_lq': round(avg_psnr_lq, 4),
            'psnr':    round(avg_psnr,    4),
            'ssim':    round(avg_ssim,    4),
            'lpips':   round(avg_lpips,   6),
        })

    print(f"\n  CSV -> {csv_path}")
    if args.save_vis:
        print(f"  Vis -> {vis_dir}/")


if __name__ == '__main__':
    main()

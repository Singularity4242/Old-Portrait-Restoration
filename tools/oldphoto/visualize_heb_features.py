# 【NICOLE2026】
"""HEB（高频增强块）中间响应可视化：进 HEB 前 / HEB 注入的高频 / 出 HEB 后。

目的（回应导师"证明新增模块有效且 face-aware"）：
  1. 空间证据：跨通道特征能量热度图，看 HEB 把高频注入到了哪里（应集中在边缘/五官）。
  2. 频域证据：进/出 HEB 的 2D FFT 对数幅度谱 + 径向平均功率谱(RAPSD)曲线，
     证明出 HEB 后高频尾部被抬升（HEB=高频增强、FFL=频域损失，频域证据最对味）。
  3. 数字证据：给定 face_weight 掩码时，统计 HEB 注入能量在"人脸内 vs 人脸外"的比值(>1)。

关键原理：HEB.forward 为 x + alpha*high，故 after_HEB - before_HEB == alpha*high，
即"HEB 注入的高频"由前向 hook 的输入/输出之差精确得到，无需改 arch。
（HEB 实现见 basicsr/archs/mambairv2_arch.py:828-831；调用见 :1092）

依赖 torch / matplotlib，只能在训练服务器跑（本地 Mac 仅 py_compile）。

示例：
    python tools/oldphoto/visualize_heb_features.py \
        --config options/train/mambairv2/OldPhoto/MambaIRv2_OldPhoto_NICOLE_v2_30k.yml \
        --ckpt experiments/MambaIRv2_OldPhoto_NICOLE_v2_30k/models/net_g_latest.pth \
        --param_key params_ema \
        --lq_dir datasets/FFHQ_pair/val_200/LQ \
        --gt_dir datasets/FFHQ_pair/val_200/HQ \
        --face_weight_dir datasets/FFHQ_pair/val_200/face_weight \
        --out_dir visualization/viz_heb --max_n 8

注意：--config 必须是 use_heb:true 的配置（A0/A1/A2 无 HEB，hook 不到）。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import torch
import yaml

from basicsr.archs import build_network

TURBO = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)


# ----------------------------- 通用小工具 ----------------------------- #
def label_tile(tile, text, tile_size):
    """缩放到 tile_size 方块并在左上角标注文字（与 visualize_edges_crops.py 一致）。"""
    tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(tile, (0, 0), (tile_size, 18), (0, 0, 0), -1)
    cv2.putText(tile, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def feature_energy(feat):
    """[1,C,H,W] -> [H,W] 跨通道均方根能量（numpy float32）。"""
    e = feat.pow(2).mean(dim=1).sqrt()  # [1,H,W]
    return e[0].detach().cpu().numpy().astype(np.float32)


def norm01(arr, scale=None, pct=99.0):
    """按给定 scale（或 pct 分位数）归一化到 [0,1]，抑制离群点。返回 (vis01, scale)。"""
    if scale is None:
        scale = float(np.percentile(arr, pct)) + 1e-8
    return np.clip(arr / scale, 0.0, 1.0), scale


def overlay_heatmap(base_bgr, heat01, alpha=0.5):
    """把 [0,1] 热度图上色后叠加到底图。base_bgr 与 heat01 尺寸需一致。"""
    heat_u8 = (heat01 * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, TURBO)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0.0)


def fft_logamp(gray):
    """单通道图 -> fftshift 后的对数幅度谱（uint8 BGR，便于拼图）。"""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    vis, _ = norm01(mag, pct=100.0)
    return cv2.applyColorMap((vis * 255).astype(np.uint8), TURBO)


def rapsd(gray):
    """径向平均功率谱：返回 (freq[0,1], power) 一维曲线。"""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin[:rmax] / np.maximum(nr[:rmax], 1)
    freq = np.arange(rmax) / float(rmax)
    return freq, radial


# ----------------------------- 模型构建 / 加载 ----------------------------- #
def build_and_load(config_path, ckpt_path, param_key, device, require_heb=True):
    """构建并加载 net_g。require_heb=True 时强制 use_heb（HEB 可视化用）；
    特征 stage 走查/臂间对比可传 False 以加载 A0/A2 等无 HEB 的臂。"""
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt = opt['network_g']
    if require_heb and not net_opt.get('use_heb', False):
        raise ValueError('--config 的 network_g.use_heb 必须为 true，否则没有 HEB 可 hook。')
    net = build_network(net_opt)
    sd = torch.load(ckpt_path, map_location='cpu')
    if param_key and param_key in sd:
        sd = sd[param_key]
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # blur 是 persistent=False 的 buffer，不在 checkpoint 里也不在 state_dict 里，strict=False 兜底
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if unexpected:
        print(f'[warn] unexpected keys: {unexpected[:5]}{" ..." if len(unexpected) > 5 else ""}')
    net.eval().to(device)
    if require_heb:
        if isinstance(net.heb, torch.nn.Identity):
            raise ValueError('net.heb 是 Identity（该权重对应 use_heb:false），无法可视化。')
        a = net.heb.alpha.detach().abs().mean().item()
        print(f'[info] HEB alpha |mean|={a:.4f}（应明显 >0，否则 HEB 近似 no-op）')
    return net


# ----------------------------- 主流程 ----------------------------- #
def img_to_tensor(path, device):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t, bgr


def tensor_to_bgr(t):
    arr = t[0].clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description='HEB 中间响应（进/注入/出）可视化。')
    p.add_argument('--config', required=True, help='use_heb:true 的 yml（取 network_g）。')
    p.add_argument('--ckpt', required=True, help='net_g .pth。')
    p.add_argument('--param_key', default='params_ema', help='checkpoint 内权重键（默认 params_ema）。')
    p.add_argument('--lq_dir', required=True)
    p.add_argument('--gt_dir', default=None)
    p.add_argument('--face_weight_dir', default=None, help='给定则做人脸内/外注入能量量化。')
    p.add_argument('--out_dir', default='viz_heb')
    p.add_argument('--max_n', type=int, default=8)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--shared_scale', action='store_true',
                   help='before/after 共享归一化 scale（亮度可比，默认各自归一化）。')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    net = build_and_load(args.config, args.ckpt, args.param_key, device)

    lq_dir = Path(args.lq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = args.tile

    names = sorted(pp.name for pp in lq_dir.glob('*.png'))[:args.max_n]
    if not names:
        raise FileNotFoundError(f'{lq_dir} 下没有 png。')

    # hook：pre_hook 取 before，hook 取 after
    cap = {}
    h1 = net.heb.register_forward_pre_hook(lambda m, inp: cap.__setitem__('before', inp[0].detach()))
    h2 = net.heb.register_forward_hook(lambda m, inp, out: cap.__setitem__('after', out.detach()))

    rapsd_acc = []   # (freq, before_power, after_power) 用于平均曲线
    focus_rows = []  # face_focus csv
    rel_acc = []     # 每图的相对注入强度 E_inc/E_before

    try:
        for name in names:
            lq_t, lq_bgr = img_to_tensor(lq_dir / name, device)
            if lq_t is None:
                print(f'[skip] 读不到 LQ: {name}')
                continue
            cap.clear()
            with torch.no_grad():
                out_t = net(lq_t)
            out_bgr = tensor_to_bgr(out_t)
            H, W = lq_bgr.shape[:2]

            # 立刻取走真 alpha 那次前向的特征（裁回原图尺寸）：
            # 下面的 alpha=0 前向会再次触发 hook 覆盖 cap，必须先存。
            f_before = cap['before'][:, :, :H, :W].clone()
            f_after = cap['after'][:, :, :H, :W].clone()

            # HEB 开/关输出差：临时把 alpha 置零再前向一次（不用重训），
            # |out_on - out_off| 即 HEB 对成品图的净贡献，比看 latent 直观。
            orig_alpha = net.heb.alpha.data.clone()
            net.heb.alpha.data.zero_()
            with torch.no_grad():
                out_off_t = net(lq_t)
            net.heb.alpha.data.copy_(orig_alpha)
            out_off_bgr = tensor_to_bgr(out_off_t)
            out_diff = (out_t - out_off_t)[0].abs().mean(0).cpu().numpy()[:H, :W]

            E_before = feature_energy(f_before)
            E_after = feature_energy(f_after)
            E_inc = feature_energy(f_after - f_before)  # == alpha*high 的能量
            # 相对注入强度：HEB 增量能量 / 进 HEB 信号能量（诊断 HEB 贡献大小）
            rel = float(E_inc.mean() / (E_before.mean() + 1e-8))
            rel_acc.append(rel)

            # --- 空间面板 ---
            common = max(E_before.max(), E_after.max()) if args.shared_scale else None
            vb, _ = norm01(E_before, scale=common)
            va, _ = norm01(E_after, scale=common)
            vi, _ = norm01(E_inc)
            vdiff, _ = norm01(out_diff)
            cols = [
                (lq_bgr, 'LQ'),
                (overlay_heatmap(lq_bgr, vb), 'before-HEB'),
                (overlay_heatmap(lq_bgr, vi), f'HEB-inject(rel{rel:.2f})'),
                (overlay_heatmap(lq_bgr, va), 'after-HEB'),
                (overlay_heatmap(out_bgr, vdiff), 'dOut(HEB on-off)'),
                (out_bgr, 'output'),
            ]
            if args.gt_dir:
                gt = cv2.imread(str(Path(args.gt_dir) / name))
                if gt is not None:
                    cols.append((gt, 'GT'))
            panel = cv2.hconcat([label_tile(c, lab, T) for c, lab in cols])
            cv2.imwrite(str(out_dir / f'{Path(name).stem}_heb_panel.png'), panel)

            # --- 2D FFT 谱（用特征通道均值图）---
            g_before = f_before[0].mean(0).cpu().numpy()
            g_after = f_after[0].mean(0).cpu().numpy()
            spec = cv2.hconcat([
                label_tile(fft_logamp(g_before), 'FFT before', T),
                label_tile(fft_logamp(g_after), 'FFT after', T),
            ])
            cv2.imwrite(str(out_dir / f'{Path(name).stem}_heb_spectrum.png'), spec)

            # --- RAPSD 1D 曲线 ---
            fr, pb = rapsd(g_before)
            _, pa = rapsd(g_after)
            rapsd_acc.append((fr, pb, pa))
            save_rapsd_plot(out_dir / f'{Path(name).stem}_rapsd.png', fr, pb, pa, name)

            # --- 人脸内/外注入能量量化 ---
            if args.face_weight_dir:
                mw = cv2.imread(str(Path(args.face_weight_dir) / name), cv2.IMREAD_GRAYSCALE)
                if mw is not None:
                    if mw.shape[:2] != (H, W):
                        mw = cv2.resize(mw, (W, H), interpolation=cv2.INTER_NEAREST)
                    inside = mw > 0
                    organ = mw >= 200          # tri-level 中 255 档≈五官
                    e_in = float(E_inc[inside].mean()) if inside.any() else 0.0
                    e_out = float(E_inc[~inside].mean()) if (~inside).any() else 0.0
                    e_org = float(E_inc[organ].mean()) if organ.any() else 0.0
                    focus_rows.append((name, e_in, e_out, e_in / (e_out + 1e-8),
                                       e_org, e_org / (e_out + 1e-8)))

            print(f'[ok] {name}')
    finally:
        h1.remove()
        h2.remove()

    # 平均 RAPSD
    if rapsd_acc:
        fr = rapsd_acc[0][0]
        pb = np.mean([r[1] for r in rapsd_acc], axis=0)
        pa = np.mean([r[2] for r in rapsd_acc], axis=0)
        save_rapsd_plot(out_dir / 'rapsd_mean.png', fr, pb, pa, f'mean of {len(rapsd_acc)}')

    # face focus csv + 汇总
    if focus_rows:
        import csv
        csv_path = out_dir / 'heb_face_focus.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['name', 'E_inc_face', 'E_inc_bg', 'ratio_face/bg',
                        'E_inc_organ', 'ratio_organ/bg'])
            w.writerows(focus_rows)
        ratios = np.array([r[3] for r in focus_rows])
        org_ratios = np.array([r[5] for r in focus_rows])
        print(f'\n[face-focus] 人脸内/外注入能量比 均值={ratios.mean():.3f}（>1 即偏向人脸），'
              f'五官/背景={org_ratios.mean():.3f} -> {csv_path}')

    if rel_acc:
        print(f'\n[heb-strength] 相对注入强度 E_inc/E_before 均值={np.mean(rel_acc):.3f} '
              f'(min={min(rel_acc):.3f}, max={max(rel_acc):.3f})。'
              f'此值很小说明 HEB 对特征改动有限 -> 看 dOut 列与 rapsd_mean 判断是否值得保留。')

    print(f'\n完成，结果在 {out_dir}/。')


def save_rapsd_plot(path, freq, p_before, p_after, title):
    """画进/出 HEB 的径向平均功率谱（log-y）。matplotlib 不可用时静默跳过曲线。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('[warn] 无 matplotlib，跳过 RAPSD 曲线（谱图/面板仍会输出）。')
        return
    plt.figure(figsize=(5, 4))
    plt.semilogy(freq, p_before + 1e-12, label='before HEB')
    plt.semilogy(freq, p_after + 1e-12, label='after HEB')
    plt.xlabel('normalized spatial frequency')
    plt.ylabel('radially averaged power')
    plt.title(f'RAPSD: {title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(path), dpi=120)
    plt.close()


if __name__ == '__main__':
    main()
# 【NICOLE2026】

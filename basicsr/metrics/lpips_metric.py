# 【NICOLE2026】
"""LPIPS 验证指标：训练过程中跟踪感知质量（越低越好）。

依赖 `lpips` 包（服务器上 tools/oldphoto/calculate_lpips.py 已在使用，无新依赖）。
网络惰性加载并全局缓存，只在第一次验证时初始化一次。
yaml 配置示例：
    lpips:
      type: calculate_lpips
      better: lower
      crop_border: 0
      net: alex
"""
import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

_lpips_net_cache = {}


def _get_lpips_net(net):
    """惰性加载并缓存 LPIPS 网络（避免每张验证图重复初始化）。"""
    if net not in _lpips_net_cache:
        import lpips  # 惰性导入：不装 lpips 包不影响其他指标
        model = lpips.LPIPS(net=net, verbose=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            model = model.cuda()
        _lpips_net_cache[net] = model
    return _lpips_net_cache[net]


def _img_to_lpips_tensor(img, device):
    """tensor2img 输出的 BGR uint8 (H, W, C) -> LPIPS 所需的 RGB float [-1, 1] (1, C, H, W)。"""
    img = img.astype(np.float32) / 255.0
    img = img[..., ::-1].copy()  # BGR -> RGB
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return (t * 2.0 - 1.0).to(device)


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, net='alex', **kwargs):
    """计算 LPIPS（Learned Perceptual Image Patch Similarity），越低越好。

    Args:
        img (ndarray): 待评估图，BGR uint8 (H, W, C)，tensor2img 的输出格式。
        img2 (ndarray): 参考 GT 图，格式同上。
        crop_border (int): 计算前裁掉的边界像素数，与 PSNR/SSIM 口径一致。
        net (str): LPIPS 主干，'alex'（快，监控趋势够用）或 'vgg'（论文常用）。
        **kwargs: 吸收验证管线透传的额外键（face_weight、better 等）。

    Returns:
        float: LPIPS 距离。
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    model = _get_lpips_net(net)
    device = next(model.parameters()).device
    with torch.no_grad():
        dist = model(_img_to_lpips_tensor(img, device), _img_to_lpips_tensor(img2, device))
    return float(dist.item())
# 【NICOLE2026】

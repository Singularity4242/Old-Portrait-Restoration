# 【NICOLE2026】
"""Identity similarity 指标：测试时衡量修复结果与 GT 的人脸身份一致性。
该指标用 ArcFace 提取 SR 与 GT 的人脸嵌入，计算余弦相似度。
依赖 `facexlib`。网络惰性加载并全局缓存，只在第一次调用时初始化一次

yaml 配置示例：
    identity:
      type: calculate_identity
      better: higher
      crop_border: 0
"""
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY

_arcface_cache = {}


def _get_arcface_net(model_path=None):
    """惰性加载并缓存 ArcFace 网络（避免每张测试图重复初始化与下载权重）。

    Args:
        model_path (str | None): 本地 ir_se50 权重路径（recognition_arcface_ir_se50.pth）。
            给定则直接从本地加载，绕开 facexlib 的自动下载（服务器下载慢时用）；
            为 None 时回退到 facexlib 自动下载/缓存。
    """
    if 'model' not in _arcface_cache:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_path:
            # 惰性导入：不装 facexlib 不影响其他指标
            from facexlib.recognition.arcface_arch import Backbone
            model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
            sd = torch.load(model_path, map_location='cpu')
            if isinstance(sd, dict) and 'params' in sd:  # facexlib release 权重包了一层 'params'
                sd = sd['params']
            model.load_state_dict(sd)
            model = model.to(device).eval()
        else:
            from facexlib.recognition import init_recognition_model
            model = init_recognition_model('arcface', device=device).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _arcface_cache['model'] = model
        _arcface_cache['device'] = device
    return _arcface_cache['model'], _arcface_cache['device']


def _img_to_arcface_tensor(img, device):
    """tensor2img 输出的 BGR uint8 (H, W, C) -> ArcFace 所需的 RGB float [-1, 1] (1, 3, 112, 112)。"""
    import cv2
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img[..., ::-1].copy()  # BGR -> RGB
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    t = (t - 0.5) / 0.5  # [0, 1] -> [-1, 1]，匹配 ir_se50 的输入归一化
    return t.to(device)


@METRIC_REGISTRY.register()
def calculate_identity(img, img2, crop_border=0, model_path=None, **kwargs):
    """
    Args:
        img (ndarray): 修复结果，BGR uint8 (H, W, C)，tensor2img 的输出格式。
        img2 (ndarray): 参考 GT 图，格式同上。
        crop_border (int): 计算前裁掉的边界像素数，与 PSNR/SSIM 口径一致。
        model_path (str | None): 本地 ir_se50 权重路径，在 yaml 里用 `model_path:` 指定，
            可绕开 facexlib 自动下载（服务器下载慢时用）。
        **kwargs: 吸收验证管线透传的额外键（face_weight、better、test_y_channel 等）。
    Returns:
        float: 余弦相似度，范围 [-1, 1]。
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    model, device = _get_arcface_net(model_path)
    with torch.no_grad():
        emb1 = F.normalize(model(_img_to_arcface_tensor(img, device)), dim=1)
        emb2 = F.normalize(model(_img_to_arcface_tensor(img2, device)), dim=1)
        cos = (emb1 * emb2).sum(dim=1)
    return float(cos.item())
# 【NICOLE2026】

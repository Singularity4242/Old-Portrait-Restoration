"""
face_losses.py
==============
两个用于人脸感知的 loss，注册到 BasicSR 的 LOSS_REGISTRY：

  ROIWeightedL1Loss  : 用 BiSeNet 人脸解析生成 soft mask，对人脸区域赋予更高权重的 L1 loss
  IdentityLoss       : 用冻结的 ArcFace 计算余弦距离，约束修复前后的人脸身份一致性

在 basicsr/losses/__init__.py 末尾加一行即可使用：
    from .face_losses import ROIWeightedL1Loss, IdentityLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


# ─────────────────────────────────────────────────────────────
# 辅助：BiSeNet 人脸解析网络（冻结，仅推理）
# ─────────────────────────────────────────────────────────────

class _BiSeNet(nn.Module):
    """
    加载预训练 BiSeNet-ResNet18 人脸解析模型。
    支持两种 checkpoint 格式：
      1. state_dict（来自 face-parsing.PyTorch 仓库的 .pth）
      2. TorchScript（torch.jit.save 保存的 .pt）
    """

    def __init__(self, checkpoint_path: str, n_classes: int = 19):
        super().__init__()
        try:
            from face_parsing.model import BiSeNet  # type: ignore
            net = BiSeNet(n_classes=n_classes)
            net.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu'), strict=False
            )
        except (ModuleNotFoundError, Exception):
            # fallback：尝试作为 TorchScript 加载
            net = torch.jit.load(checkpoint_path, map_location='cpu')
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)，值域 [0, 1]
        返回 : (B, n_classes, H, W) logits
        """
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        out = self.net((x - mean) / std)
        # 部分实现返回 tuple，取第一项
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


# ─────────────────────────────────────────────────────────────
# 辅助：ArcFace 人脸识别网络（冻结，仅推理）
# ─────────────────────────────────────────────────────────────

class _ArcFace(nn.Module):
    """
    加载预训练 ArcFace（iresnet50）用于身份特征提取。
    支持 InsightFace state_dict 和 TorchScript 两种格式。
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        try:
            from insightface.recognition.arcface_torch.backbones import iresnet50  # type: ignore
            net = iresnet50(pretrained=False)
            net.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu'), strict=False
            )
        except (ModuleNotFoundError, Exception):
            net = torch.jit.load(checkpoint_path, map_location='cpu')
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)，值域 [0, 1]，内部 resize 到 112x112
        返回 : (B, 512) L2 归一化的身份嵌入向量
        """
        mean = x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std  = x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        x_112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        emb = self.net((x_112 - mean) / std)
        return F.normalize(emb, dim=1)


# ─────────────────────────────────────────────────────────────
# ROI-weighted L1 Loss
# ─────────────────────────────────────────────────────────────

@LOSS_REGISTRY.register()
class ROIWeightedL1Loss(nn.Module):
    """
    空间加权 L1 loss。

    对每个像素 p：
        L = mean( soft_mask(p) * |pred(p) - gt(p)| )

    soft_mask 由冻结的 BiSeNet 生成：
        - 人脸关键区域（眼、鼻、嘴等）→ 权重接近 1.0
        - 背景区域 → 权重 = background_weight（默认 0.5）
        - 边缘用高斯模糊平滑过渡

    Parameters
    ----------
    loss_weight : float
        loss 整体缩放系数。
    face_parser_path : str
        BiSeNet checkpoint 路径。为空时退化为普通 L1。
    mask_blur_kernel : int
        高斯模糊核大小（必须为奇数），用于平滑 mask 边缘。
    face_class_ids : list[int]
        BiSeNet 中属于人脸的 class id。
        默认：skin(1), l-brow(2), r-brow(3), l-eye(4), r-eye(5),
              nose(10), upper-lip(11), inner-mouth(12), lower-lip(13)
    background_weight : float
        背景像素的最低权重。0=完全忽略背景，1=和人脸同等对待。
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        face_parser_path: str = '',
        mask_blur_kernel: int = 51,
        face_class_ids: list = None,
        background_weight: float = 0.5,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.blur_k = mask_blur_kernel if mask_blur_kernel % 2 == 1 else mask_blur_kernel + 1
        self.face_class_ids = face_class_ids or [1, 2, 3, 4, 5, 10, 11, 12, 13]
        self.bg_weight = background_weight

        self.parser = None
        if face_parser_path:
            self.parser = _BiSeNet(face_parser_path)
            for p in self.parser.parameters():
                p.requires_grad_(False)

    def _make_soft_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)，值域 [0, 1]
        返回 soft_mask : (B, 1, H, W)，值域 [bg_weight, 1.0]
        """
        if self.parser is None:
            return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3],
                              device=x.device, dtype=x.dtype)

        with torch.no_grad():
            logits = self.parser(x)                          # (B, 19, H, W)
            pred   = logits.argmax(dim=1, keepdim=True)     # (B, 1, H, W)

        # 二值人脸 mask
        face_mask = torch.zeros_like(pred, dtype=x.dtype)
        for cid in self.face_class_ids:
            face_mask = face_mask + (pred == cid).to(x.dtype)
        face_mask = face_mask.clamp(0, 1)

        # 可分离高斯模糊，平滑边缘
        k = self.blur_k
        sigma = k / 6.0
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        kernel = g1d.unsqueeze(0).unsqueeze(0)   # (1, 1, k)

        B, _, H, W = face_mask.shape
        m = face_mask.view(B * H, 1, W)
        m = F.conv1d(m, kernel, padding=k // 2).view(B, 1, H, W)
        m = m.permute(0, 1, 3, 2)
        m = F.conv1d(
            m.reshape(B * W, 1, H), kernel, padding=k // 2
        ).view(B, 1, W, H).permute(0, 1, 3, 2)

        # 线性映射：背景→bg_weight，人脸→1.0
        return self.bg_weight + (1.0 - self.bg_weight) * m

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lq: torch.Tensor = None,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        pred   : (B, 3, H, W)，值域 [0, 1]，模型输出
        target : (B, 3, H, W)，值域 [0, 1]，GT
        lq     : (B, 3, H, W)，值域 [0, 1]，低质量输入（可选，用于生成 mask）
        weight : 保留参数，与 BasicSR 接口兼容，不使用
        """
        src = lq if lq is not None else target
        soft_mask = self._make_soft_mask(src.detach())
        loss = (soft_mask * (pred - target).abs()).mean()
        return loss * self.loss_weight


# ─────────────────────────────────────────────────────────────
# Identity Loss
# ─────────────────────────────────────────────────────────────

@LOSS_REGISTRY.register()
class IdentityLoss(nn.Module):
    """
    基于 ArcFace 余弦距离的身份保持 loss。

        L_id = 1 - cosine_similarity( f(pred), f(target) )

    f(·) 为冻结的 ArcFace 编码器，输出 512 维 L2 归一化向量。

    Parameters
    ----------
    loss_weight : float
        loss 整体缩放系数。
    arcface_path : str
        ArcFace checkpoint 路径。为空时 loss 恒为 0。
    """

    def __init__(self, loss_weight: float = 0.1, arcface_path: str = ''):
        super().__init__()
        self.loss_weight = loss_weight

        self.arcface = None
        if arcface_path:
            self.arcface = _ArcFace(arcface_path)
            for p in self.arcface.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        pred   : (B, 3, H, W)，值域 [0, 1]
        target : (B, 3, H, W)，值域 [0, 1]
        """
        if self.arcface is None:
            return pred.new_tensor(0.0)

        with torch.no_grad():
            emb_gt = self.arcface(target.detach())   # (B, 512)，不传梯度

        emb_pred = self.arcface(pred)                # (B, 512)，梯度流过 pred

        cos_sim = (emb_pred * emb_gt).sum(dim=1)     # (B,)，值域 [-1, 1]
        loss = (1.0 - cos_sim).mean()
        return loss * self.loss_weight

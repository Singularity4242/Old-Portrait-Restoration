from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
# 【NICOLE 2026】
from .psnr_ssim import calculate_face_psnr, calculate_face_ssim, calculate_psnr, calculate_ssim
# 【NICOLE 2026】
# 【NICOLE2026】LPIPS 验证指标
from .lpips_metric import calculate_lpips
# 【NICOLE2026】

# 【NICOLE 2026】
# __all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_face_psnr', 'calculate_face_ssim']
# 【NICOLE 2026】
# 【NICOLE2026】
__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_face_psnr', 'calculate_face_ssim', 'calculate_lpips']
# 【NICOLE2026】


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric

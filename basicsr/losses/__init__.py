from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
# 【NICOLE 2026】
from .losses import (CharbonnierLoss, FaceRegionWeightedCharbonnierLoss, FaceRegionWeightedSobelCharbonnierLoss,
                     GANLoss, L1Loss, MSELoss, WeightedTVLoss, g_path_regularize, gradient_penalty_loss, r1_penalty)
# 【NICOLE 2026】

# 【NICOLE 2026】
__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'FaceRegionWeightedCharbonnierLoss',
    'FaceRegionWeightedSobelCharbonnierLoss', 'WeightedTVLoss', 'GANLoss', 'gradient_penalty_loss', 'r1_penalty',
    'g_path_regularize'
]
# 【NICOLE 2026】

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss

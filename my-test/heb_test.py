import torch

ckpt = torch.load("experiments/MambaIRv2_OldPhoto_NICOLE_Stage2_70k/models/net_g_70000.pth", map_location="cpu")
# BasicSR 存的权重通常在 'params' 或 'params_ema' 键下；若直接是 state_dict 就去掉这层
sd = ckpt.get("params_ema", ckpt.get("params", ckpt))

alpha = None
for k, v in sd.items():
    if k.endswith("heb.alpha"):
        alpha = v.flatten().float()
        print("found:", k, tuple(v.shape))
        break

assert alpha is not None, "没找到 heb.alpha，确认这次权重确实是开了 use_heb=True 训练的"

print(f"通道数 : {alpha.numel()}")
print(f"min    : {alpha.min().item():.6f}")
print(f"max    : {alpha.max().item():.6f}")
print(f"mean   : {alpha.mean().item():.6f}")
print(f"std    : {alpha.std().item():.6f}")
print(f"|alpha| 均值 : {alpha.abs().mean().item():.6f}")
print(f"接近0(|a|<1e-3)的通道占比 : {(alpha.abs() < 1e-3).float().mean().item():.2%}")
# #仅用于测试环境的训练
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# import cv2
# import numpy as np
# import sys
# sys.path.insert(0, '/root/autodl-tmp/MambaIR')
#
# from basicsr.archs.mambairv2_arch import MambaIRv2
#
# # ─────────────────────────────────────────
# # 1. Dataset
# # ─────────────────────────────────────────
# class OldPhotoDataset(Dataset):
#     def __init__(self, data_dir, patch_size=128):
#         self.hq_dir = Path(data_dir) / 'hq'
#         self.lq_dir = Path(data_dir) / 'lq'
#         self.files = sorted(self.hq_dir.glob('*.png'))
#         self.patch_size = patch_size
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         name = self.files[idx].name
#         hq = cv2.imread(str(self.hq_dir / name))
#         lq = cv2.imread(str(self.lq_dir / name))
#
#         # BGR → RGB，归一化到 [0, 1]
#         hq = cv2.cvtColor(hq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#         lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#
#         # HWC → CHW
#         hq = torch.from_numpy(hq.transpose(2, 0, 1))
#         lq = torch.from_numpy(lq.transpose(2, 0, 1))
#         return lq, hq
#
# # ─────────────────────────────────────────
# # 2. 训练配置
# # ─────────────────────────────────────────
# DATA_DIR    = '/root/autodl-tmp/dataset'
# BATCH_SIZE  = 4
# NUM_EPOCHS  = 5        # 先跑5个epoch验证流程
# LR          = 1e-4
# SAVE_DIR    = '/root/autodl-tmp/checkpoints'
# Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
#
# # ─────────────────────────────────────────
# # 3. 模型
# # ─────────────────────────────────────────
# model = MambaIRv2(
#     upscale=1,
#     img_size=128,
#     embed_dim=48,
#     d_state=8,
#     depths=[5, 5, 5, 5],
#     num_heads=[4, 4, 4, 4],
#     window_size=16,
#     inner_rank=32,
#     num_tokens=64,
#     convffn_kernel_size=5,
#     img_range=1.,
#     mlp_ratio=1.,
#     upsampler=''
# ).cuda()
#
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
# criterion = nn.L1Loss()
#
# # ─────────────────────────────────────────
# # 4. DataLoader
# # ─────────────────────────────────────────
# dataset = OldPhotoDataset(DATA_DIR)
# loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
#                      shuffle=True, num_workers=4, pin_memory=True)
# print(f"数据集大小: {len(dataset)} 张，每个epoch {len(loader)} 个batch")
#
# # ─────────────────────────────────────────
# # 5. 训练循环
# # ─────────────────────────────────────────
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     total_loss = 0.0
#
#     for i, (lq, hq) in enumerate(loader):
#         lq, hq = lq.cuda(), hq.cuda()
#
#         pred = model(lq)
#         loss = criterion(pred, hq)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         if (i + 1) % 50 == 0:
#             print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
#                   f"Step [{i+1}/{len(loader)}] "
#                   f"Loss: {loss.item():.4f}")
#
#     avg_loss = total_loss / len(loader)
#     print(f"── Epoch {epoch+1} 平均Loss: {avg_loss:.4f} ──")
#     scheduler.step()
#
#     # 每个epoch保存一次checkpoint
#     ckpt_path = f"{SAVE_DIR}/epoch_{epoch+1}.pth"
#     torch.save({
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': avg_loss,
#     }, ckpt_path)
#     print(f"已保存 checkpoint: {ckpt_path}")
#
# print("训练完成！")
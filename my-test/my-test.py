# import torch
# import sys
# sys.path.insert(0, '/root/autodl-tmp/MambaIR')
#
# from basicsr.archs.mambairv2_arch import MambaIRv2
#
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
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
# lq = torch.randn(2, 3, 128, 128).cuda()
# gt = torch.randn(2, 3, 128, 128).cuda()
#
# pred = model(lq)
# print("✅ 前向传播成功，输出尺寸:", pred.shape)
#
# loss = torch.nn.functional.l1_loss(pred, gt)
# loss.backward()
# optimizer.step()
#
# print("✅ 训练环境完全验证通过，loss:", loss.item())
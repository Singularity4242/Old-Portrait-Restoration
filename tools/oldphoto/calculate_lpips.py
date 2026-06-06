import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def load_rgb_tensor(path, device, torch):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def find_restored(restored_dir, gt_path, result_suffix):
    if result_suffix:
        candidate = restored_dir / f'{gt_path.stem}_{result_suffix}{gt_path.suffix}'
        if candidate.exists():
            return candidate

    exact = restored_dir / gt_path.name
    if exact.exists():
        return exact

    matches = []
    for ext in IMG_EXTS:
        matches.extend(restored_dir.glob(f'{gt_path.stem}_*{ext}'))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple restored images match {gt_path.name}. '
            f'Use --result_suffix to disambiguate.'
        )
    raise FileNotFoundError(f'No restored image found for {gt_path.name}')


def main():
    parser = argparse.ArgumentParser(description='Calculate average LPIPS for old photo restoration results.')
    parser.add_argument('--restored_dir', required=True, help='Directory containing restored result images.')
    parser.add_argument('--gt_dir', required=True, help='Directory containing HQ/GT images.')
    parser.add_argument('--result_suffix', default='', help='Suffix appended by BasicSR test, without image extension.')
    parser.add_argument('--net', default='alex', choices=['alex', 'vgg', 'squeeze'], help='LPIPS backbone.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for LPIPS calculation.')
    args = parser.parse_args()

    import torch

    try:
        import lpips
    except ImportError as exc:
        raise ImportError('Please install lpips first: pip install lpips') from exc

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    restored_dir = Path(args.restored_dir)
    gt_dir = Path(args.gt_dir)

    gt_paths = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not gt_paths:
        raise FileNotFoundError(f'No images found in {gt_dir}')

    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    scores = []
    with torch.no_grad():
        for gt_path in gt_paths:
            restored_path = find_restored(restored_dir, gt_path, args.result_suffix)
            restored = load_rgb_tensor(restored_path, device, torch)
            gt = load_rgb_tensor(gt_path, device, torch)
            if restored.shape != gt.shape:
                raise ValueError(f'Shape mismatch: {restored_path.name} {restored.shape} vs {gt_path.name} {gt.shape}')
            score = loss_fn(restored, gt).item()
            scores.append(score)
            print(f'{gt_path.name}: {score:.6f}')

    print(f'\nAverage LPIPS ({args.net}): {float(np.mean(scores)):.6f}')
    print(f'Number of images: {len(scores)}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate offline face-region weight masks for old-portrait restoration.

This script uses the BiSeNet face-parsing checkpoint from
zllrunning/face-parsing.PyTorch, such as ``79999_iter.pth``, to parse each
HQ/GT portrait image. It then converts the 19-class parsing result into a
single-channel PNG weight mask.

By default, the generated mask is intended to be loaded by the training dataset
later:

    0   -> non-prioritized region, e.g. background, cloth, hat
    76  -> normal face-related region, e.g. skin, hair, ear, neck, glasses
    255 -> key facial structure, e.g. eyes, brows, nose, mouth, lips

The training loss can map the saved mask to actual pixel weights, for example:

    W = 1.0 + mask / 255.0

This keeps training independent from BiSeNet. BiSeNet is used only once in this
offline preprocessing step.

If ``--output-kind labelmap`` is used, the script saves the raw 19-class parsing
label map instead:

    0-18 -> CelebAMask-HQ-style face parsing class ids used by
            zllrunning/face-parsing.PyTorch
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# Supported HQ/GT image extensions.
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# zllrunning/face-parsing.PyTorch 19-class label convention:
# 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye,
# 6 eye_g, 7 l_ear, 8 r_ear, 9 ear_r, 10 nose, 11 mouth,
# 12 u_lip, 13 l_lip, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat.
NORMAL_FACE_LABELS = (1, 6, 7, 8, 9, 14, 15, 17)
KEY_FACE_LABELS = (2, 3, 4, 5, 10, 11, 12, 13)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate offline face weight masks with BiSeNet 79999_iter.pth.'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing HQ/GT images, e.g. datasets/FFHQ_pair/train/HQ.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where generated single-channel uint8 PNGs are saved.',
    )
    parser.add_argument(
        '--output-kind',
        default='weight',
        choices=['weight', 'labelmap'],
        help='Save mapped face weights 0/76/255 or raw parsing labels 0-18. Default: weight.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to BiSeNet checkpoint, e.g. pretrained_models/79999_iter.pth.',
    )
    parser.add_argument(
        '--bisenet-repo',
        required=True,
        help='Path to cloned zllrunning/face-parsing.PyTorch repository.',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Inference device. Use cuda, cuda:0, or cpu. Default: cuda.',
    )
    parser.add_argument(
        '--parser-size',
        type=int,
        default=512,
        help='BiSeNet input size. The common 79999_iter.pth inference size is 512.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate masks even when the output PNG already exists.',
    )
    parser.add_argument(
        '--preview-dir',
        default='',
        help='Optional directory for overlay previews used to visually check masks.',
    )
    parser.add_argument(
        '--preview-max',
        type=int,
        default=30,
        help='Maximum number of preview images to save. Default: 30.',
    )
    return parser.parse_args()


def resolve_device(device_name):
    """Return a valid torch.device and fall back to CPU if CUDA is unavailable."""
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_name)


def import_bisenet(bisenet_repo):
    """Import BiSeNet from a local zllrunning/face-parsing.PyTorch clone.

    The original repository is not packaged as an installable module. Its
    ``model.py`` imports ``resnet.py`` by filename, so we add the repository root
    to ``sys.path`` and then load ``model.py`` explicitly.
    """
    repo = Path(bisenet_repo).expanduser().resolve()
    model_py = repo / 'model.py'
    resnet_py = repo / 'resnet.py'

    if not model_py.is_file():
        raise FileNotFoundError(f'Cannot find model.py in BiSeNet repo: {model_py}')
    if not resnet_py.is_file():
        raise FileNotFoundError(f'Cannot find resnet.py in BiSeNet repo: {resnet_py}')

    # Needed because zllrunning's model.py imports "resnet" as a local module.
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Load model.py from the user-provided path to avoid ambiguity with any other
    # module named "model" in the current Python environment.
    spec = importlib.util.spec_from_file_location('zllrunning_face_parsing_model', model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'BiSeNet'):
        raise AttributeError(f'BiSeNet class is not found in {model_py}')
    return module.BiSeNet


def load_bisenet(checkpoint_path, bisenet_repo, device):
    """Build BiSeNet, load 79999_iter.pth, and switch it to eval mode."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Cannot find checkpoint: {checkpoint_path}')

    BiSeNet = import_bisenet(bisenet_repo)
    net = BiSeNet(n_classes=19)

    # The public 79999_iter.pth is normally a raw state_dict. This branch also
    # supports checkpoints wrapped as {"state_dict": ...}.
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # Some checkpoints trained with DataParallel prefix every key with "module.".
    # Removing this prefix makes loading robust without changing actual weights.
    clean_state = {}
    for key, value in state.items():
        clean_state[key.replace('module.', '')] = value

    net.load_state_dict(clean_state, strict=True)
    net.to(device)
    net.eval()
    return net


def list_images(input_dir):
    """List supported image files under input_dir recursively."""
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')

    image_paths = sorted(p for p in input_dir.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f'No supported images found in: {input_dir}')
    return input_dir, image_paths


def preprocess_for_bisenet(img_bgr, parser_size, device):
    """Convert an OpenCV BGR image to normalized BiSeNet input tensor.

    BiSeNet 79999_iter.pth is commonly evaluated with 512 x 512 RGB input and
    ImageNet normalization. If the HQ/GT images are 256 x 256, this temporary
    resize is used only for parsing and does not change the training images.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (parser_size, parser_size), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def logits_to_label(logits):
    """Convert BiSeNet logits to a 2D uint8 parsing label map."""
    # Some BiSeNet implementations return (main_out, aux16, aux32). We use the
    # main output only, which is the first element.
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    label = logits.argmax(dim=1).squeeze(0)
    return label.detach().cpu().numpy().astype(np.uint8)


def label_to_weight_map(label):
    """Convert a 19-class face parsing label map to a single-channel mask."""
    weight = np.zeros(label.shape, dtype=np.uint8)

    # Normal face-related areas receive a medium priority.
    for class_id in NORMAL_FACE_LABELS:
        weight[label == class_id] = 76

    # Key facial structures receive the highest priority.
    for class_id in KEY_FACE_LABELS:
        weight[label == class_id] = 255

    return weight


def parse_one_image(net, img_bgr, parser_size, device):
    """Generate one label map and one weight map with the input image size."""
    height, width = img_bgr.shape[:2]
    tensor = preprocess_for_bisenet(img_bgr, parser_size, device)

    with torch.no_grad():
        label_parser_size = logits_to_label(net(tensor))

    # Parsing labels are discrete class ids. Nearest-neighbor resize is required
    # so that interpolation does not create invalid intermediate class values.
    label = cv2.resize(label_parser_size, (width, height), interpolation=cv2.INTER_NEAREST)
    weight = label_to_weight_map(label)
    return label, weight


def save_preview(img_bgr, weight, preview_path):
    """Save an overlay preview for manual quality control.

    Yellow marks normal face-related regions. Red marks key facial structures.
    These colors are only for visualization and are not used by training.
    """
    color = np.zeros_like(img_bgr)
    color[weight == 76] = (0, 180, 255)
    color[weight == 255] = (0, 0, 255)

    overlay = cv2.addWeighted(img_bgr, 0.65, color, 0.35, 0.0)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), overlay)


def main():
    """Run mask generation for all images in the input directory."""
    args = parse_args()
    device = resolve_device(args.device)

    input_dir, image_paths = list_images(args.input_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    preview_dir = Path(args.preview_dir).expanduser().resolve() if args.preview_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    net = load_bisenet(args.checkpoint, args.bisenet_repo, device)

    generated = 0
    skipped = 0
    failed = 0
    all_zero = 0
    preview_count = 0

    for img_path in tqdm(image_paths, desc='Generating face weight masks'):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix('.png')

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            failed += 1
            print(f'[warn] Failed to read image, skipped: {img_path}')
            continue

        label, weight = parse_one_image(net, img_bgr, args.parser_size, device)

        # A fully zero mask may mean no face was parsed, or the face is heavily
        # misaligned. It is not fatal, but these samples should be inspected.
        saved_mask = weight if args.output_kind == 'weight' else label
        if args.output_kind == 'weight' and int(weight.max()) == 0:
            all_zero += 1
            print(f'[warn] All-zero mask generated: {img_path}')

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), saved_mask.astype(np.uint8))
        if not ok:
            failed += 1
            print(f'[warn] Failed to write mask: {out_path}')
            continue

        generated += 1

        if preview_dir is not None and preview_count < args.preview_max:
            preview_path = preview_dir / rel_path.with_suffix('.png')
            save_preview(img_bgr, weight, preview_path)
            preview_count += 1

    print('Done.')
    print(f'Input directory: {input_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Total images: {len(image_paths)}')
    print(f'Generated masks: {generated}')
    print(f'Skipped existing masks: {skipped}')
    print(f'Failed images: {failed}')
    print(f'All-zero masks: {all_zero}')
    if preview_dir is not None:
        print(f'Preview directory: {preview_dir}')
        print(f'Preview images: {preview_count}')


if __name__ == '__main__':
    main()

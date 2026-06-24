"""
Direct whole-image inference for MambaIRv2.

This script is only for debugging partition artifacts. It builds net_g from the
YAML network_g section and calls net(lq) directly, so it bypasses
MambaIRv2Model.test() and its partition-and-merge logic.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicsr.archs import build_network  # noqa: E402
from basicsr.utils.img_util import img2tensor, tensor2img  # noqa: E402
from basicsr.utils.options import ordered_yaml  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        required=True,
        help="Path to the same test YAML used by basicsr/test.py.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one LQ test image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: experiments/direct_forward_debug/<name>_direct.png",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override path:path:pretrain_network_g in the YAML.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Use cuda when available; use cpu for a slow but memory-safe sanity check.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_yaml(opt_path):
    opt_path = resolve_path(opt_path)
    with opt_path.open("r") as f:
        return yaml.load(f, Loader=ordered_yaml()[0])


def load_network(opt, checkpoint, device):
    net = build_network(opt["network_g"]).to(device)

    ckpt_path = checkpoint or opt["path"].get("pretrain_network_g")
    if ckpt_path is None:
        raise ValueError("No checkpoint provided. Set path:pretrain_network_g or pass --checkpoint.")

    ckpt_path = resolve_path(ckpt_path)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    strict = opt.get("path", {}).get("strict_load_g", True)
    missing, unexpected = net.load_state_dict(state, strict=strict)
    print(f"[model] checkpoint: {ckpt_path}")
    if missing:
        print(f"[model] missing keys: {len(missing)}")
    if unexpected:
        print(f"[model] unexpected keys: {len(unexpected)}")

    net.eval()
    return net


@torch.no_grad()
def direct_forward(net, image_path, device):
    image_path = resolve_path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read input image: {image_path}")

    img = img.astype(np.float32) / 255.0
    lq = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    print(f"[input] {image_path} shape={tuple(lq.shape)}")

    output = net(lq)
    return tensor2img(output, rgb2bgr=True, out_type=np.uint8)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[warn] CUDA is not available. Falling back to CPU.")

    opt = load_yaml(args.opt)
    net = load_network(opt, args.checkpoint, device)
    result = direct_forward(net, args.input, device)

    if args.output is None:
        out_dir = ROOT / "experiments" / "direct_forward_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(args.input).stem}_direct.png"
    else:
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), result)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()

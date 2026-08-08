"""Command-line entry point for ACDC training.

Training uses 2-D slices from ``data/ACDC/train``.  Checkpoint selection uses
the complete ``valid`` volumes and mean foreground Dice; ``test`` is reserved
for the separate final evaluation script.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from acdc_common import ensure_dataset_files, resolve_device, seed_everything
from lib.networks import EMCADNet
from trainer_acdc import trainer_acdc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMCAD on ACDC")
    parser.add_argument("--root_path", default="../data/ACDC", type=str)
    parser.add_argument("--list_dir", default="../data/ACDC/lists/lists_ACDC", type=str)
    parser.add_argument("--output_dir", default="./model_pth", type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--encoder", default="pvt_v2_b2", type=str)
    parser.add_argument("--kernel_sizes", default=[1, 3, 5], nargs="+", type=int)
    parser.add_argument("--expansion_factor", default=2, type=int)
    parser.add_argument("--lgag_ks", default=3, type=int)
    parser.add_argument("--activation_mscb", default="relu6", type=str)
    parser.add_argument("--no_dw_parallel", action="store_true")
    parser.add_argument("--concatenation", action="store_true")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--pretrained_dir", default="./pretrained_pth/pvt/", type=str)
    parser.add_argument(
        "--supervision",
        default="deep_supervision",
        choices=["deep_supervision", "mutation", "last_layer"],
    )
    parser.add_argument("--output_aggregation", default="sum", choices=["sum", "last"])
    parser.add_argument("--max_epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--inference_batch_size", default=8, type=int)
    parser.add_argument("--validate_every", default=1, type=int)
    parser.add_argument("--max_train_batches", default=0, type=int, help="smoke-test limit; 0 means all")
    parser.add_argument("--max_valid_volumes", default=0, type=int, help="smoke-test limit; 0 means all")
    parser.add_argument("--seed", default=2222, type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes != 4:
        raise ValueError("ACDC labels are background/RV/Myo/LV, so --num_classes must be 4")
    if args.max_epochs < 1 or args.batch_size < 1:
        raise ValueError("--max_epochs and --batch_size must be positive")
    ensure_dataset_files(args.root_path, args.list_dir)
    args.device = str(resolve_device(args.device))
    seed_everything(args.seed, deterministic=args.deterministic)

    if args.run_name is None:
        args.run_name = "acdc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = Path(args.output_dir) / args.run_name
    snapshot_path.mkdir(parents=True, exist_ok=True)
    torch.save(vars(args), snapshot_path / "config.pt")

    model = EMCADNet(
        num_classes=4,
        kernel_sizes=args.kernel_sizes,
        expansion_factor=args.expansion_factor,
        dw_parallel=not args.no_dw_parallel,
        add=not args.concatenation,
        lgag_ks=args.lgag_ks,
        activation=args.activation_mscb,
        encoder=args.encoder,
        pretrain=not args.no_pretrain,
        pretrained_dir=args.pretrained_dir,
    )
    trainer_acdc(args, model, snapshot_path)


if __name__ == "__main__":
    main()

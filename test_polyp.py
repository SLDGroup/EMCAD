import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch

from utils.dataloader_polyp import (
    SUPPORTED_EXTENSIONS,
    get_loader,
)
from utils.polyp_utils import (
    build_model,
    evaluate_loader,
    load_checkpoint,
    resolve_device,
    seed_everything,
    write_metrics_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EMCAD on one Polyp split"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data_root",
        default="../data/polyp/target",
    )
    parser.add_argument(
        "--dataset_name",
        default="ClinicDB",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--output_csv",
        default=None,
    )

    parser.add_argument(
        "--encoder",
        default="pvt_v2_b2",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lgag_ks",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--activation_mscb",
        default="relu6",
    )
    parser.add_argument(
        "--no_dw_parallel",
        action="store_true",
    )
    parser.add_argument(
        "--concatenation",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_dir",
        default="./pretrained_pth/pvt/",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2222,
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
    )

    return parser.parse_args()


def _split_stems(dataset_root, split):
    image_root = Path(dataset_root) / split / "images"

    if not image_root.is_dir():
        return set()

    return {
        path.stem.casefold()
        for path in image_root.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            "Checkpoint not found: {}".format(
                args.checkpoint
            )
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )

    dataset_root = os.path.join(
        args.data_root,
        args.dataset_name,
    )
    split_root = os.path.join(
        dataset_root,
        args.split,
    )

    required = [
        os.path.join(split_root, "images"),
        os.path.join(split_root, "masks"),
    ]

    missing = [
        path
        for path in required
        if not os.path.isdir(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Polyp evaluation directories:\n"
            + "\n".join(missing)
        )

    selected_stems = _split_stems(
        dataset_root,
        args.split,
    )

    for other_split in ("train", "val", "test"):
        if other_split == args.split:
            continue

        overlap = (
            selected_stems
            & _split_stems(
                dataset_root,
                other_split,
            )
        )

        if overlap:
            raise RuntimeError(
                "{} and {} overlap: {}".format(
                    args.split,
                    other_split,
                    sorted(overlap)[:10],
                )
            )

    seed_everything(
        args.seed,
        bool(args.deterministic),
    )

    device = resolve_device(args.device)

    loader = get_loader(
        image_root=os.path.join(
            split_root,
            "images",
        ),
        gt_root=os.path.join(
            split_root,
            "masks",
        ),
        batchsize=args.inference_batch_size,
        trainsize=args.img_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        augmentation=False,
        split=args.split,
        color_image=not args.grayscale,
        seed=args.seed,
    )

    checkpoint = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint)

    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            checkpoint_dir,
            "{}_{}_outputs".format(
                args.split,
                args.dataset_name,
            ),
        )
    )

    output_csv = os.path.abspath(
        args.output_csv
        or os.path.join(
            output_dir,
            "test_metrics.csv",
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(
            output_dir,
            "test.log",
        ),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout)
    )

    logging.info("args=%s", args)
    logging.info("device=%s", device)
    logging.info(
        "images=%d",
        len(loader.dataset),
    )

    model = build_model(
        args,
        pretrain=False,
    )

    load_checkpoint(
        model,
        checkpoint,
    )

    model.to(device).eval()

    rows, mean_row, std_row = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
        max_cases=args.max_cases,
        output_dir=(
            None
            if args.no_save_predictions
            else output_dir
        ),
        save_probabilities=args.save_probabilities,
        compute_surface=True,
        description="Polyp {}".format(args.split),
    )

    write_metrics_csv(
        output_csv,
        rows,
        mean_row,
        std_row,
    )

    report = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "output_csv": output_csv,
        "evaluated_images": len(rows),
        "dataset_images": len(loader.dataset),
        "manifest_sha256": loader.dataset.manifest_sha256,
        "threshold": args.threshold,
        "macro_mean": mean_row,
        "macro_std": std_row,
        "metric_policy": {
            "aggregation": (
                "per-image macro mean and population std"
            ),
            "dice_and_iou": (
                "binary masks at fixed sigmoid threshold"
            ),
            "hd95_and_assd_unit": "pixels",
            "surface_one_empty": (
                "NaN and surface_distance_defined=0"
            ),
            "surface_both_empty": "0 pixels",
        },
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    with open(
        os.path.join(
            output_dir,
            "test_summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            _json_safe(report),
            stream,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        os.path.join(
            output_dir,
            "test_config.json",
        ),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            _json_safe(
                {
                    **vars(args),
                    "checkpoint": checkpoint,
                    "data_root": os.path.abspath(
                        args.data_root
                    ),
                    "output_dir": output_dir,
                    "output_csv": output_csv,
                }
            ),
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print("metric          MEAN          STD")

    for name in (
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "hd95",
        "assd",
    ):
        print(
            "{:<12} {:>12.6f} {:>12.6f}".format(
                name,
                mean_row[name],
                std_row[name],
            )
        )

    print(
        "SURFACE_VALID={}/{}".format(
            mean_row["surface_distance_defined"],
            len(rows),
        )
    )
    print("CSV=" + output_csv)
    print("OUTPUT_DIR=" + output_dir)


if __name__ == "__main__":
    main()
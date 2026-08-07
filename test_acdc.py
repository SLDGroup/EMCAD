"""Final ACDC volume evaluation with Dice, HD95, Jaccard and ASSD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from acdc_common import (
    ACDC_CLASS_NAMES,
    ACDC_NUM_CLASSES,
    checkpoint_model_args,
    ensure_dataset_files,
    load_model_checkpoint,
    predict_volume,
    resolve_device,
    seed_everything,
)
from lib.networks import EMCADNet
from utils.dataset_ACDC import ACDCVolumeDataset
from utils.metrics_acdc import METRIC_NAMES, metric_means, safe_mean, volume_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EMCAD ACDC checkpoint")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--root_path", default="../data/ACDC", type=str)
    parser.add_argument("--list_dir", default="../data/ACDC/lists/lists_ACDC", type=str)
    parser.add_argument("--output_csv", default=None, type=str)
    parser.add_argument("--save_predictions_dir", default=None, type=str)
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
    parser.add_argument("--inference_batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--max_cases", default=0, type=int, help="smoke-test limit; 0 means all")
    parser.add_argument("--voxel_spacing", default=[1.0, 1.0, 1.0], nargs=3, type=float)
    parser.add_argument("--seed", default=2222, type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _load_saved_model_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload = torch.load(args.checkpoint, map_location="cpu")
    metadata = payload if isinstance(payload, dict) else {}
    saved = checkpoint_model_args(metadata)
    for key, value in saved.items():
        setattr(args, key, value)
    return metadata


def _write_csv(rows: list[Dict[str, Any]], output_csv: Path) -> None:
    fieldnames = ["case_name"]
    for metric in METRIC_NAMES:
        fieldnames.extend(f"{class_name}_{metric}" for class_name in ACDC_CLASS_NAMES)
        fieldnames.append(f"mean_{metric}")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_dataset_files(args.root_path, args.list_dir)
    metadata = _load_saved_model_args(args)
    saved_args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
    saved_num_classes = saved_args.get("num_classes", 4) if isinstance(saved_args, dict) else 4
    if int(saved_num_classes) != 4:
        raise ValueError("ACDC checkpoint must have four output classes")
    args.device = str(resolve_device(args.device))
    seed_everything(args.seed)

    model = EMCADNet(
        num_classes=ACDC_NUM_CLASSES,
        kernel_sizes=args.kernel_sizes,
        expansion_factor=args.expansion_factor,
        dw_parallel=not args.no_dw_parallel,
        add=not args.concatenation,
        lgag_ks=args.lgag_ks,
        activation=args.activation_mscb,
        encoder=args.encoder,
        # A full checkpoint is loaded immediately below; loading ImageNet
        # weights first is unnecessary and would make testing depend on them.
        pretrain=False,
        pretrained_dir=args.pretrained_dir,
    )
    load_model_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model.to(device)

    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_csv = Path(args.output_csv) if args.output_csv else Path(args.checkpoint).with_name("test_metrics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_predictions_dir) if args.save_predictions_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    all_per_class = {str(class_index): {metric: [] for metric in METRIC_NAMES} for class_index in range(1, 4)}
    for case_index, sampled in enumerate(tqdm(loader, desc="ACDC test")):
        if args.max_cases > 0 and case_index >= args.max_cases:
            break
        image = sampled["image"][0]
        label = sampled["label"][0].numpy()
        case_name = sampled["case_name"][0]
        prediction = predict_volume(
            model,
            image,
            device=device,
            img_size=args.img_size,
            inference_batch_size=args.inference_batch_size,
        )
        per_class = volume_metrics(
            prediction,
            label,
            num_classes=ACDC_NUM_CLASSES,
            voxel_spacing=args.voxel_spacing,
        )
        row: Dict[str, Any] = {"case_name": case_name}
        for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
            for metric in METRIC_NAMES:
                value = per_class[str(class_index)][metric]
                row[f"{class_name}_{metric}"] = value
                all_per_class[str(class_index)][metric].append(value)
        means = metric_means(per_class)
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = means[metric]
        rows.append(row)
        if save_dir is not None:
            np.savez_compressed(save_dir / f"{case_name}_prediction.npz", prediction=prediction)

    mean_row: Dict[str, Any] = {"case_name": "MEAN"}
    for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
        for metric in METRIC_NAMES:
            mean_row[f"{class_name}_{metric}"] = safe_mean(all_per_class[str(class_index)][metric])
    for metric in METRIC_NAMES:
        mean_row[f"mean_{metric}"] = safe_mean(
            row[f"mean_{metric}"] for row in rows
        )
    rows.append(mean_row)
    _write_csv(rows, output_csv)

    print("class       Dice       HD95       Jaccard       ASSD")
    for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
        print(
            f"{class_name:<8} "
            f"{mean_row[f'{class_name}_dice']:.6f} "
            f"{mean_row[f'{class_name}_hd95']:.6f} "
            f"{mean_row[f'{class_name}_jaccard']:.6f} "
            f"{mean_row[f'{class_name}_asd']:.6f}"
        )
    print(
        f"MEAN      {mean_row['mean_dice']:.6f} {mean_row['mean_hd95']:.6f} "
        f"{mean_row['mean_jaccard']:.6f} {mean_row['mean_asd']:.6f}"
    )
    print("CSV saved to:", output_csv.resolve())
    if metadata.get("args"):
        print("Model arguments loaded from checkpoint")


if __name__ == "__main__":
    main()

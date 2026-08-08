import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.acdc_utils import (
    ACDC_CLASS_NAMES,
    ACDC_NUM_CLASSES,
    METRIC_NAMES,
    build_model,
    load_checkpoint,
    mean_metrics,
    predict_volume,
    save_nifti_triplet,
    seed_everything,
    volume_metrics,
)
from utils.dataset_ACDC import ACDCVolumeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test EMCAD on ACDC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", default="./data/ACDC")
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_csv", default=None)

    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--expansion_factor", type=int, default=2)
    parser.add_argument("--lgag_ks", type=int, default=3)
    parser.add_argument("--activation_mscb", default="relu6")
    parser.add_argument("--no_dw_parallel", action="store_true")
    parser.add_argument("--concatenation", action="store_true")
    parser.add_argument("--pretrained_dir", default="./pretrained_pth/pvt/")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--z_spacing", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--save_nii", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def write_csv(path, rows):
    fieldnames = ["case_name"]
    for class_name in ACDC_CLASS_NAMES:
        fieldnames.extend(
            "{}_{}".format(class_name, metric_name)
            for metric_name in METRIC_NAMES
        )
    fieldnames.extend("mean_" + name for name in METRIC_NAMES)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    required = [
        args.checkpoint,
        os.path.join(args.root_path, "test"),
        os.path.join(args.list_dir, "test.txt"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing ACDC paths:\n" + "\n".join(missing))

    seed_everything(args.seed, deterministic=True)
    device = resolve_device(args.device)
    model = build_model(args, pretrain=False)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    output_dir = args.output_dir or os.path.join(checkpoint_dir, "predictions")
    output_csv = args.output_csv or os.path.join(checkpoint_dir, "test_metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_dir, "test.log"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("args=%s", args)

    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    class_accumulator = {
        class_index: {name: [] for name in METRIC_NAMES}
        for class_index in range(1, ACDC_NUM_CLASSES)
    }
    voxelspacing = (float(args.z_spacing), 1.0, 1.0)

    for case_index, sampled in enumerate(tqdm(loader, desc="ACDC test")):
        if args.max_cases and case_index >= args.max_cases:
            break
        image = sampled["image"][0].numpy()
        label = sampled["label"][0].numpy()
        case_name = sampled["case_name"][0]
        prediction = predict_volume(
            model,
            image,
            device=device,
            img_size=args.img_size,
            batch_size=args.inference_batch_size,
        )
        per_class = volume_metrics(
            prediction,
            label,
            num_classes=ACDC_NUM_CLASSES,
            voxelspacing=voxelspacing,
        )
        means = mean_metrics(per_class)
        row = {"case_name": case_name}
        for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
            for metric_name in METRIC_NAMES:
                value = per_class[class_index][metric_name]
                row["{}_{}".format(class_name, metric_name)] = value
                class_accumulator[class_index][metric_name].append(value)
        for metric_name in METRIC_NAMES:
            row["mean_" + metric_name] = means[metric_name]
        rows.append(row)
        logging.info(
            "case=%s dice=%.6f hd95=%.6f jaccard=%.6f asd=%.6f",
            case_name,
            means["dice"],
            means["hd95"],
            means["jaccard"],
            means["asd"],
        )

        if args.save_npz:
            np.savez_compressed(
                os.path.join(output_dir, case_name + "_prediction.npz"),
                prediction=prediction,
            )
        if args.save_nii:
            save_nifti_triplet(
                image,
                prediction,
                label,
                output_dir,
                case_name,
                args.z_spacing,
            )

    if not rows:
        raise RuntimeError("ACDC test produced no cases")

    summary = {"case_name": "MEAN"}
    for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
        for metric_name in METRIC_NAMES:
            summary["{}_{}".format(class_name, metric_name)] = float(
                np.mean(class_accumulator[class_index][metric_name])
            )
    for metric_name in METRIC_NAMES:
        summary["mean_" + metric_name] = float(
            np.mean([row["mean_" + metric_name] for row in rows])
        )
    rows.append(summary)
    write_csv(output_csv, rows)

    print("class       Dice       HD95       Jaccard       ASD")
    for class_name in ACDC_CLASS_NAMES:
        print(
            "{:<8} {:>10.6f} {:>10.6f} {:>13.6f} {:>10.6f}".format(
                class_name,
                summary[class_name + "_dice"],
                summary[class_name + "_hd95"],
                summary[class_name + "_jaccard"],
                summary[class_name + "_asd"],
            )
        )
    print(
        "{:<8} {:>10.6f} {:>10.6f} {:>13.6f} {:>10.6f}".format(
            "MEAN",
            summary["mean_dice"],
            summary["mean_hd95"],
            summary["mean_jaccard"],
            summary["mean_asd"],
        )
    )
    print("CSV=" + os.path.abspath(output_csv))
    print("OUTPUT_DIR=" + os.path.abspath(output_dir))


if __name__ == "__main__":
    main()

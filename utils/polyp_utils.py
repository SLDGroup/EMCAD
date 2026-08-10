import csv
import itertools
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

from lib.networks import EMCADNet


TEST_METRIC_NAMES = (
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
    "hd95",
    "assd",
)


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return device


def build_model(args, pretrain):
    return EMCADNet(
        num_classes=1,
        kernel_sizes=args.kernel_sizes,
        expansion_factor=args.expansion_factor,
        dw_parallel=not args.no_dw_parallel,
        add=not args.concatenation,
        lgag_ks=args.lgag_ks,
        activation=args.activation_mscb,
        encoder=args.encoder,
        pretrain=pretrain,
        pretrained_dir=args.pretrained_dir,
    )


def model_outputs(model, images, mode="test"):
    outputs = model(images, mode=mode)

    if isinstance(outputs, (list, tuple)):
        return list(outputs)

    return [outputs]


def _model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()

    return model.state_dict()


def save_checkpoint(model, path):
    torch.save(_model_state_dict(model), path)


def load_checkpoint(model, path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: {}".format(path)
        )

    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    target = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )

    target.load_state_dict(
        state_dict,
        strict=True,
    )


def structure_loss(logits, mask):
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(
            mask,
            kernel_size=31,
            stride=1,
            padding=15,
        )
        - mask
    )

    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        mask,
        reduction="none",
    )

    weighted_bce = (
        weight * weighted_bce
    ).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = torch.sigmoid(logits)

    intersection = (
        probability * mask * weight
    ).sum(dim=(2, 3))

    union = (
        (probability + mask) * weight
    ).sum(dim=(2, 3))

    weighted_iou = 1.0 - (
        (intersection + 1.0)
        / (union - intersection + 1.0)
    )

    return (weighted_bce + weighted_iou).mean()


def supervised_structure_loss(outputs, mask, supervision):
    count = len(outputs)
    indices = list(range(count))

    if supervision == "paper":
        # EMCAD 二分类论文设置：
        # 4 个单输出损失 + 4 个输出相加后的损失。
        groups = [[index] for index in indices]
        groups.append(indices)
    elif supervision == "deep_supervision":
        groups = [[index] for index in indices]
    elif supervision == "last_layer":
        groups = [[count - 1]]
    elif supervision == "mutation":
        groups = [
            list(group)
            for length in range(1, count + 1)
            for group in itertools.combinations(
                indices,
                length,
            )
        ]
    else:
        raise ValueError(
            "Unknown supervision: {}".format(supervision)
        )

    loss = mask.new_tensor(0.0)

    for group in groups:
        logits = sum(outputs[index] for index in group)
        loss = loss + structure_loss(logits, mask)

    return loss


def binary_metrics(prediction, target, compute_surface=True):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction/target shape mismatch: {} vs {}".format(
                prediction.shape,
                target.shape,
            )
        )

    tp = int(
        np.logical_and(prediction, target).sum()
    )
    tn = int(
        np.logical_and(~prediction, ~target).sum()
    )
    fp = int(
        np.logical_and(prediction, ~target).sum()
    )
    fn = int(
        np.logical_and(~prediction, target).sum()
    )

    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    positive_target = tp + fn
    negative_target = tn + fp
    positive_prediction = tp + fp
    total = tp + tn + fp + fn

    dice = (
        1.0
        if dice_denominator == 0
        else (2.0 * tp) / dice_denominator
    )

    iou = (
        1.0
        if union == 0
        else tp / union
    )

    if positive_target == 0:
        sensitivity = (
            1.0
            if positive_prediction == 0
            else 0.0
        )
    else:
        sensitivity = tp / positive_target

    specificity = (
        1.0
        if negative_target == 0
        else tn / negative_target
    )

    if positive_prediction == 0:
        precision = (
            1.0
            if positive_target == 0
            else 0.0
        )
    else:
        precision = tp / positive_prediction

    accuracy = (
        1.0
        if total == 0
        else (tp + tn) / total
    )

    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())

    surface_defined = int(
        (
            prediction_nonempty
            and target_nonempty
        )
        or (
            not prediction_nonempty
            and not target_nonempty
        )
    )

    hd95 = float("nan")
    assd = float("nan")

    if (
        compute_surface
        and prediction_nonempty
        and target_nonempty
    ):
        hd95 = float(
            metric.binary.hd95(
                prediction,
                target,
            )
        )
        assd = float(
            metric.binary.assd(
                prediction,
                target,
            )
        )
    elif (
        compute_surface
        and not prediction_nonempty
        and not target_nonempty
    ):
        hd95 = 0.0
        assd = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "hd95": hd95,
        "assd": assd,
        "surface_distance_defined": (
            surface_defined
            if compute_surface
            else 0
        ),
        "pred_foreground_pixels": int(
            prediction.sum()
        ),
        "gt_foreground_pixels": int(
            target.sum()
        ),
    }


def _finite_stat(rows, name, reducer):
    values = np.asarray(
        [row[name] for row in rows],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(reducer(values))


def summarize_rows(rows):
    if not rows:
        raise RuntimeError(
            "Evaluation produced no images"
        )

    mean_row = {"case_name": "MEAN"}
    std_row = {"case_name": "STD"}

    for name in TEST_METRIC_NAMES:
        mean_row[name] = _finite_stat(
            rows,
            name,
            np.mean,
        )
        std_row[name] = _finite_stat(
            rows,
            name,
            np.std,
        )

    mean_row["surface_distance_defined"] = int(
        sum(
            row["surface_distance_defined"]
            for row in rows
        )
    )
    std_row["surface_distance_defined"] = ""

    for name in (
        "pred_foreground_pixels",
        "gt_foreground_pixels",
    ):
        mean_row[name] = _finite_stat(
            rows,
            name,
            np.mean,
        )
        std_row[name] = _finite_stat(
            rows,
            name,
            np.std,
        )

    return mean_row, std_row


def evaluate_loader(
    model,
    loader,
    device,
    threshold=0.5,
    max_cases=0,
    output_dir=None,
    save_probabilities=False,
    compute_surface=True,
    description="Polyp evaluation",
):
    if output_dir:
        mask_dir = os.path.join(
            output_dir,
            "predictions",
        )
        probability_dir = os.path.join(
            output_dir,
            "probabilities",
        )

        os.makedirs(mask_dir, exist_ok=True)

        if save_probabilities:
            os.makedirs(
                probability_dir,
                exist_ok=True,
            )

    rows = []
    model.eval()

    with torch.no_grad():
        for (
            images,
            targets,
            original_sizes,
            names,
        ) in tqdm(loader, desc=description):
            images = images.to(
                device=device,
                dtype=torch.float32,
            )

            logits = model_outputs(
                model,
                images,
                mode="test",
            )[-1]

            for index, name in enumerate(names):
                if max_cases and len(rows) >= max_cases:
                    break

                height = int(
                    original_sizes[index, 0]
                )
                width = int(
                    original_sizes[index, 1]
                )

                # probability = torch.sigmoid(
                #     F.interpolate(
                #         logits[index:index + 1],
                #         size=(height, width),
                #         mode="bilinear",
                #         align_corners=False,
                #     )
                # )[0, 0].cpu().numpy()

                # target = (
                #     targets[index]
                #     .squeeze(0)
                #     .cpu()
                #     .numpy()
                #     >= 0.5
                # )
                # prediction = probability >= float(
                #     threshold
                # )
                probability = torch.sigmoid(
                    F.interpolate(
                        logits[index:index + 1],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )[0, 0].cpu().numpy()

                # Match the official EMCAD Polyp evaluation:
                # per-image min-max normalization before thresholding.
                probability = (
                    probability - probability.min()
                ) / (
                    probability.max()
                    - probability.min()
                    + 1e-8
                )

                target = (
                    targets[index]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    >= 0.5
                )

                prediction = probability >= float(
                    threshold
                )






                metrics = binary_metrics(
                    prediction,
                    target,
                    compute_surface=compute_surface,
                )

                rows.append(
                    {
                        "case_name": name,
                        **metrics,
                    }
                )

                if output_dir:
                    prediction_image = (
                        prediction.astype(np.uint8) * 255
                    )

                    saved = cv2.imwrite(
                        os.path.join(
                            mask_dir,
                            name,
                        ),
                        prediction_image,
                    )

                    if not saved:
                        raise RuntimeError(
                            "Failed to save prediction: "
                            + name
                        )

                    if save_probabilities:
                        probability_image = np.clip(
                            probability * 255.0,
                            0,
                            255,
                        ).astype(np.uint8)

                        saved = cv2.imwrite(
                            os.path.join(
                                probability_dir,
                                name,
                            ),
                            probability_image,
                        )

                        if not saved:
                            raise RuntimeError(
                                "Failed to save probability: "
                                + name
                            )

            if max_cases and len(rows) >= max_cases:
                break

    mean_row, std_row = summarize_rows(rows)
    return rows, mean_row, std_row


def write_metrics_csv(
    path,
    rows,
    mean_row,
    std_row,
):
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )

    fieldnames = [
        "case_name",
        *TEST_METRIC_NAMES,
        "surface_distance_defined",
        "pred_foreground_pixels",
        "gt_foreground_pixels",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)
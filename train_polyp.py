import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.dataloader_polyp import get_loader
from utils.polyp_utils import (
    build_model,
    evaluate_loader,
    load_checkpoint,
    model_outputs,
    resolve_device,
    save_checkpoint,
    seed_everything,
    supervised_structure_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMCAD on one Polyp dataset"
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
        "--output_dir",
        default="./model_pth/Polyp",
    )
    parser.add_argument(
        "--run_name",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
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
        "--no_pretrain",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_dir",
        default="./pretrained_pth/pvt/",
    )

    parser.add_argument(
        "--supervision",
        choices=[
            "paper",
            "deep_supervision",
            "last_layer",
            "mutation",
        ],
        default="paper",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scheduler",
        choices=["constant", "cosine"],
        default="constant",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--scale_rates",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
    )
    parser.add_argument(
        "--no_multi_scale",
        action="store_true",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
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
        "--validate_every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_valid_cases",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def append_history(path, row):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_dice",
        "val_iou",
        "learning_rate",
        "elapsed_seconds",
    ]

    exists = os.path.isfile(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def append_validation_rows(path, epoch, rows):
    fieldnames = [
        "epoch",
        "case_name",
        "dice",
        "iou",
    ]

    exists = os.path.isfile(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "epoch": epoch,
                    "case_name": row["case_name"],
                    "dice": row["dice"],
                    "iou": row["iou"],
                }
            )


def resized_batch(images, masks, image_size, rate):
    if rate == 1.0:
        return images, masks

    scaled = int(
        round(image_size * rate / 32.0) * 32
    )

    # images = F.interpolate(
    #     images,
    #     size=(scaled, scaled),
    #     mode="bilinear",
    #     align_corners=False,
    # ) 为修复BKAI数据集指标差了8.4个点而修改
    images = F.interpolate(
        images,
        size=(scaled, scaled),
        mode="bilinear",
        align_corners=True,
    )

    masks = F.interpolate(
        masks,
        size=(scaled, scaled),
        mode="nearest",
    )

    return images, masks


def main():
    args = parse_args()

    if args.validate_every < 1:
        raise ValueError(
            "--validate_every must be at least 1"
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )

    dataset_root = os.path.join(
        args.data_root,
        args.dataset_name,
    )
    train_root = os.path.join(
        dataset_root,
        "train",
    )
    val_root = os.path.join(
        dataset_root,
        "val",
    )

    required = [
        os.path.join(train_root, "images"),
        os.path.join(train_root, "masks"),
        os.path.join(val_root, "images"),
        os.path.join(val_root, "masks"),
    ]

    missing = [
        path
        for path in required
        if not os.path.isdir(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Polyp train/val directories:\n"
            + "\n".join(missing)
        )

    seed_everything(
        args.seed,
        bool(args.deterministic),
    )

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader = get_loader(
        image_root=os.path.join(
            train_root,
            "images",
        ),
        gt_root=os.path.join(
            train_root,
            "masks",
        ),
        batchsize=args.batch_size,
        trainsize=args.img_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        augmentation=not args.no_augmentation,
        split="train",
        color_image=not args.grayscale,
        seed=args.seed,
    )

    val_loader = get_loader(
        image_root=os.path.join(
            val_root,
            "images",
        ),
        gt_root=os.path.join(
            val_root,
            "masks",
        ),
        batchsize=args.val_batch_size,
        trainsize=args.img_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        augmentation=False,
        split="val",
        color_image=not args.grayscale,
        seed=args.seed,
    )

    overlap = (
        set(train_loader.dataset.stems)
        & set(val_loader.dataset.stems)
    )

    if overlap:
        raise RuntimeError(
            "Train/val leakage detected: {}".format(
                sorted(overlap)[:10]
            )
        )

    if args.run_name is None:
        args.run_name = (
            "train_Polyp_{}_{}".format(
                args.dataset_name,
                datetime.now().strftime(
                    "%Y-%m-%d_%H%M%S"
                ),
            )
        )

    run_dir = os.path.abspath(
        os.path.join(
            args.output_dir,
            args.dataset_name,
            args.run_name,
        )
    )

    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(
            run_dir,
            "train.log",
        ),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout)
    )

    configuration = {
        **vars(args),
        "data_root": os.path.abspath(
            args.data_root
        ),
        "dataset_root": os.path.abspath(
            dataset_root
        ),
        "run_dir": run_dir,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "train_manifest_sha256": (
            train_loader.dataset.manifest_sha256
        ),
        "val_manifest_sha256": (
            val_loader.dataset.manifest_sha256
        ),
        "command": " ".join(sys.argv),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    with open(
        os.path.join(run_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            configuration,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("args=%s", args)
    logging.info("device=%s", device)
    logging.info(
        "train_images=%d val_images=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    model = build_model(
        args,
        pretrain=not args.no_pretrain,
    )

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(
                "Checkpoint not found: {}".format(
                    args.checkpoint
                )
            )
        load_checkpoint(
            model,
            args.checkpoint,
        )

    model.to(device)

    if device.type == "cuda" and args.n_gpu > 1:
        available = torch.cuda.device_count()

        if args.n_gpu > available:
            raise RuntimeError(
                "Requested {} GPUs, but only {} are visible".format(
                    args.n_gpu,
                    available,
                )
            )

        model = nn.DataParallel(
            model,
            device_ids=list(
                range(args.n_gpu)
            ),
        )

    logging.info(
        "model_parameters=%d",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=args.min_lr,
        )

    scaler = GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    writer = SummaryWriter(
        os.path.join(run_dir, "tensorboard")
    )

    history_path = os.path.join(
        run_dir,
        "train_history.csv",
    )
    validation_path = os.path.join(
        run_dir,
        "validation_metrics.csv",
    )

    best_dice = float("-inf")
    best_epoch = 0
    global_step = 0
    scale_rates = (
        [1.0]
        if args.no_multi_scale
        else args.scale_rates
    )
    started = time.time()

    for epoch in range(
        1,
        args.max_epochs + 1,
    ):
        model.train()
        epoch_losses = []

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="epoch {}/{}".format(
                epoch,
                args.max_epochs,
            ),
        )

        for batch_index, (images, masks) in progress:
            if (
                args.max_train_batches
                and batch_index >= args.max_train_batches
            ):
                break

            images = images.to(
                device=device,
                dtype=torch.float32,
            )
            masks = masks.to(
                device=device,
                dtype=torch.float32,
            )

            for rate in scale_rates:
                scaled_images, scaled_masks = resized_batch(
                    images,
                    masks,
                    args.img_size,
                    float(rate),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with autocast(
                    enabled=scaler.is_enabled()
                ):
                    outputs = model_outputs(
                        model,
                        scaled_images,
                        mode="train",
                    )

                    loss = supervised_structure_loss(
                        outputs,
                        scaled_masks,
                        args.supervision,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    args.clip,
                )

                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                loss_value = float(
                    loss.item()
                )
                epoch_losses.append(loss_value)

                writer.add_scalar(
                    "train/loss",
                    loss_value,
                    global_step,
                )
                writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

            progress.set_postfix(
                loss="{:.4f}".format(
                    float(
                        np.mean(
                            epoch_losses[
                                -len(scale_rates):
                            ]
                        )
                    )
                )
            )

        if not epoch_losses:
            raise RuntimeError(
                "No Polyp training batches were processed"
            )

        train_loss = float(
            np.mean(epoch_losses)
        )
        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        save_checkpoint(
            model,
            os.path.join(
                run_dir,
                "last.pth",
            ),
        )

        val_dice = ""
        val_iou = ""

        if (
            epoch % args.validate_every == 0
            or epoch == args.max_epochs
        ):
            val_rows, val_mean, _ = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                threshold=args.threshold,
                max_cases=args.max_valid_cases,
                output_dir=None,
                compute_surface=False,
                description="Polyp val",
            )

            val_dice = val_mean["dice"]
            val_iou = val_mean["iou"]

            append_validation_rows(
                validation_path,
                epoch,
                val_rows,
            )

            writer.add_scalar(
                "val/dice",
                val_dice,
                epoch,
            )
            writer.add_scalar(
                "val/iou",
                val_iou,
                epoch,
            )

            logging.info(
                "epoch=%d train_loss=%.6f val_dice=%.6f val_iou=%.6f",
                epoch,
                train_loss,
                val_dice,
                val_iou,
            )

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch

                save_checkpoint(
                    model,
                    os.path.join(
                        run_dir,
                        "best.pth",
                    ),
                )

                with open(
                    os.path.join(
                        run_dir,
                        "best_validation.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(
                        {
                            "epoch": best_epoch,
                            "val_dice": best_dice,
                            "val_iou": val_iou,
                        },
                        stream,
                        indent=2,
                    )

                logging.info(
                    "saved best.pth epoch=%d val_dice=%.6f",
                    best_epoch,
                    best_dice,
                )
        else:
            logging.info(
                "epoch=%d train_loss=%.6f",
                epoch,
                train_loss,
            )

        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.time() - started,
            },
        )

        writer.add_scalar(
            "train/epoch_loss",
            train_loss,
            epoch,
        )

        if args.save_every and (
            epoch % args.save_every == 0
            or epoch == args.max_epochs
        ):
            save_checkpoint(
                model,
                os.path.join(
                    run_dir,
                    "epoch_{}.pth".format(epoch),
                ),
            )

        if scheduler is not None:
            scheduler.step()

    writer.close()

    logging.info(
        "training finished best_epoch=%d best_val_dice=%.6f elapsed=%.2fs",
        best_epoch,
        best_dice,
        time.time() - started,
    )

    print("RUN_DIR=" + run_dir)
    print(
        "BEST_CHECKPOINT="
        + os.path.join(run_dir, "best.pth")
    )


if __name__ == "__main__":
    main()
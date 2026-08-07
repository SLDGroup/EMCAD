"""Training loop for ACDC (slice training, volume validation)."""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from acdc_common import (
    ACDC_CLASS_NAMES,
    ACDC_NUM_CLASSES,
    load_model_checkpoint,
    model_outputs,
    predict_volume,
    supervised_loss,
)
from utils.dataset_ACDC import ACDCVolumeDataset, ACDCdataset, RandomGenerator
from utils.metrics_acdc import dice_only, safe_mean


def _logger(snapshot_path: Path) -> logging.Logger:
    logger = logging.getLogger("acdc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(snapshot_path / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_dice: float,
    args: Any,
) -> None:
    state_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state": state_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": vars(args),
        },
        path,
    )


def _validate(args: Any, model: torch.nn.Module, device: torch.device, csv_path: Path) -> float:
    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="valid")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    rows = []
    all_dice = []

    for case_index, sampled in enumerate(tqdm(loader, desc="ACDC validation", leave=False)):
        if args.max_valid_volumes > 0 and case_index >= args.max_valid_volumes:
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
        per_class = dice_only(prediction, label, num_classes=ACDC_NUM_CLASSES)
        row = {"case_name": case_name}
        for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
            value = per_class[str(class_index)]
            row[f"{class_name}_dice"] = value
        row["mean_dice"] = safe_mean(per_class.values())
        rows.append(row)
        all_dice.append(row["mean_dice"])

    fieldnames = ["case_name"] + [f"{name}_dice" for name in ACDC_CLASS_NAMES] + ["mean_dice"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return safe_mean(all_dice)


def trainer_acdc(args: Any, model: torch.nn.Module, snapshot_path: str | os.PathLike[str]) -> str:
    """Run ACDC training and select ``best.pth`` using validation Dice only."""

    snapshot = Path(snapshot_path)
    snapshot.mkdir(parents=True, exist_ok=True)
    logger = _logger(snapshot)
    device = torch.device(args.device)
    model.to(device)

    train_dataset = ACDCdataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    logger.info("train slices=%d, iterations/epoch=%d, device=%s", len(train_dataset), len(train_loader), device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_dice = -float("inf")

    if args.resume:
        metadata = load_model_checkpoint(model, args.resume)
        if "optimizer_state" in metadata:
            optimizer.load_state_dict(metadata["optimizer_state"])
        if "scaler_state" in metadata:
            scaler.load_state_dict(metadata["scaler_state"])
        start_epoch = int(metadata.get("epoch", -1)) + 1
        best_dice = float(metadata.get("best_dice", best_dice))
        logger.info("resumed checkpoint=%s from epoch=%d", args.resume, start_epoch)

    writer = None
    try:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(str(snapshot / "tensorboard"))
    except Exception as error:  # TensorBoard is useful but not required to train.
        logger.warning("TensorBoard disabled: %s", error)

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"ACDC train {epoch + 1}/{args.max_epochs}", leave=False)
        for batch_index, sampled in enumerate(progress):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            images = sampled["image"].to(device, non_blocking=True)
            labels = sampled["label"].to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                outputs = model_outputs(model, images, mode="train")
                loss = supervised_loss(
                    outputs,
                    labels,
                    supervision=args.supervision,
                    output_aggregation=args.output_aggregation,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, batch={batch_index}: {loss.item()}")
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if writer is not None:
                writer.add_scalar("train/loss", epoch_losses[-1], global_step)
            progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

        mean_loss = safe_mean(epoch_losses)
        logger.info("epoch=%d loss=%.6f", epoch + 1, mean_loss)
        _save_checkpoint(snapshot / "last.pth", model, optimizer, scaler, epoch, best_dice, args)

        if (epoch + 1) % args.validate_every == 0 or epoch == args.max_epochs - 1:
            valid_dice = _validate(args, model, device, snapshot / "valid_metrics.csv")
            logger.info("epoch=%d validation_mean_dice=%.6f", epoch + 1, valid_dice)
            if writer is not None:
                writer.add_scalar("valid/mean_dice", valid_dice, epoch + 1)
            if valid_dice >= best_dice:
                best_dice = valid_dice
                _save_checkpoint(snapshot / "best.pth", model, optimizer, scaler, epoch, best_dice, args)
                logger.info("saved best.pth (validation_mean_dice=%.6f)", best_dice)

    if not (snapshot / "best.pth").exists():
        _save_checkpoint(snapshot / "best.pth", model, optimizer, scaler, args.max_epochs - 1, best_dice, args)
    if writer is not None:
        writer.close()
    logger.info("training finished: %s", snapshot.resolve())
    return str(snapshot)

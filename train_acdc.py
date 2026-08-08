import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.acdc_utils import (
    ACDC_CLASS_NAMES,
    ACDC_NUM_CLASSES,
    DiceLoss,
    build_model,
    load_checkpoint,
    model_outputs,
    predict_volume,
    seed_everything,
    supervised_loss,
    validation_dice,
)
from utils.dataset_ACDC import ACDCVolumeDataset, ACDCdataset, RandomGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train EMCAD on ACDC")
    parser.add_argument("--root_path", default="./data/ACDC")
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    parser.add_argument("--output_dir", default="./model_pth/ACDC")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--expansion_factor", type=int, default=2)
    parser.add_argument("--lgag_ks", type=int, default=3)
    parser.add_argument("--activation_mscb", default="relu6")
    parser.add_argument("--no_dw_parallel", action="store_true")
    parser.add_argument("--concatenation", action="store_true")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--pretrained_dir", default="./pretrained_pth/pvt/")

    parser.add_argument(
        "--supervision",
        choices=["mutation", "deep_supervision", "last_layer"],
        default="mutation",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_valid_volumes", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def save_state_dict(model, path):
    torch.save(get_state_dict(model), path)


def append_validation_csv(path, epoch, rows):
    fieldnames = ["epoch", "case_name", "RV_dice", "MYO_dice", "LV_dice", "mean_dice"]
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"epoch": epoch, **row})


def validate(args, model, device, csv_path, epoch):
    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="valid")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    for index, sampled in enumerate(tqdm(loader, desc="ACDC valid", leave=False)):
        if args.max_valid_volumes and index >= args.max_valid_volumes:
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
        per_class = validation_dice(prediction, label)
        mean_dice = float(np.mean(list(per_class.values())))
        rows.append(
            {
                "case_name": case_name,
                "RV_dice": per_class[1],
                "MYO_dice": per_class[2],
                "LV_dice": per_class[3],
                "mean_dice": mean_dice,
            }
        )
    if not rows:
        raise RuntimeError("ACDC validation produced no volumes")
    append_validation_csv(csv_path, epoch, rows)
    return float(np.mean([row["mean_dice"] for row in rows]))


def main():
    args = parse_args()
    if ACDC_NUM_CLASSES != 4:
        raise RuntimeError("ACDC must use four classes including background")

    required = [
        os.path.join(args.root_path, "train"),
        os.path.join(args.root_path, "valid"),
        os.path.join(args.list_dir, "train.txt"),
        os.path.join(args.list_dir, "valid.txt"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing ACDC paths:\n" + "\n".join(missing))

    seed_everything(args.seed, bool(args.deterministic))
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if args.run_name is None:
        args.run_name = "acdc_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    snapshot_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "train.log"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("args=%s", args)
    logging.info("device=%s", device)
    with open(os.path.join(snapshot_path, "config.json"), "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2)

    train_dataset = ACDCdataset(
        args.root_path,
        args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )

    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    logging.info("train slices=%d batches=%d", len(train_dataset), len(trainloader))

    model = build_model(args, pretrain=not args.no_pretrain)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.to(device)
    if device.type == "cuda" and args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpu)))

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(ACDC_NUM_CLASSES)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))

    best_dice = -1.0
    global_step = 0
    valid_csv = os.path.join(snapshot_path, "validation_metrics.csv")

    for epoch in range(args.max_epochs):
        model.train()
        epoch_losses = []
        progress = tqdm(trainloader, desc="epoch {}/{}".format(epoch + 1, args.max_epochs))
        for batch_index, sampled in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images = sampled["image"].to(device=device, dtype=torch.float32)
            labels = sampled["label"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                outputs = model_outputs(model, images, mode="train")
                loss = supervised_loss(
                    outputs,
                    labels,
                    supervision=args.supervision,
                    ce_loss=ce_loss,
                    dice_loss=dice_loss,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            epoch_losses.append(float(loss.item()))
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", args.base_lr, global_step)
            progress.set_postfix(loss="{:.4f}".format(loss.item()))

        if not epoch_losses:
            raise RuntimeError("No ACDC training batches were processed")
        mean_loss = float(np.mean(epoch_losses))
        logging.info("epoch=%d train_loss=%.6f", epoch + 1, mean_loss)
        writer.add_scalar("train/epoch_loss", mean_loss, epoch + 1)
        save_state_dict(model, os.path.join(snapshot_path, "last.pth"))

        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.max_epochs:
            mean_dice = validate(args, model, device, valid_csv, epoch + 1)
            logging.info("epoch=%d validation_mean_dice=%.6f", epoch + 1, mean_dice)
            writer.add_scalar("valid/mean_dice", mean_dice, epoch + 1)
            if mean_dice >= best_dice:
                best_dice = mean_dice
                save_state_dict(model, os.path.join(snapshot_path, "best.pth"))
                logging.info("saved best.pth validation_mean_dice=%.6f", best_dice)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.max_epochs:
            save_state_dict(
                model,
                os.path.join(snapshot_path, "epoch_{}.pth".format(epoch + 1)),
            )

    writer.close()
    logging.info("training finished best_dice=%.6f", best_dice)
    print("BEST_CHECKPOINT=" + os.path.abspath(os.path.join(snapshot_path, "best.pth")))


if __name__ == "__main__":
    main()

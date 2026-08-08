import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric

from lib.networks import EMCADNet


ACDC_NUM_CLASSES = 4
ACDC_CLASS_NAMES = ("RV", "MYO", "LV")
METRIC_NAMES = ("dice", "hd95", "jaccard", "asd")


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def build_model(args, pretrain):
    return EMCADNet(
        num_classes=ACDC_NUM_CLASSES,
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
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(
            target.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * target_one_hot, dim=dims)
        denominator = torch.sum(
            probabilities * probabilities + target_one_hot * target_one_hot,
            dim=dims,
        )
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        return 1.0 - dice.mean()


def _supervision_groups(output_count, supervision):
    indices = list(range(output_count))
    if supervision == "last_layer":
        return [[output_count - 1]]
    if supervision == "deep_supervision":
        return [[index] for index in indices]
    if supervision == "mutation":
        return [
            list(group)
            for length in range(1, output_count + 1)
            for group in itertools.combinations(indices, length)
        ]
    raise ValueError("Unknown supervision: " + supervision)


def supervised_loss(outputs, target, supervision, ce_loss, dice_loss):
    total = target.new_tensor(0.0, dtype=torch.float32)
    for group in _supervision_groups(len(outputs), supervision):
        logits = sum(outputs[index] for index in group)
        total = total + 0.3 * ce_loss(logits, target.long())
        total = total + 0.7 * dice_loss(logits, target)
    return total


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def predict_volume(model, image, device, img_size, batch_size=8):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("Expected [D,H,W] volume, got {}".format(image.shape))
    depth, height, width = image.shape
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, depth, batch_size):
            batch = torch.from_numpy(image[start:start + batch_size]).unsqueeze(1)
            batch = batch.to(device=device, dtype=torch.float32)
            if (height, width) != (img_size, img_size):
                batch = F.interpolate(
                    batch,
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                )
            logits = model_outputs(model, batch, mode="test")[-1]
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.int16)


def calculate_metric_percase(prediction, target, voxelspacing=None):
    """Local copy of the existing Synapse metric policy; Synapse stays untouched."""
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    if prediction.any() and target.any():
        return {
            "dice": float(metric.binary.dc(prediction, target)),
            "hd95": float(
                metric.binary.hd95(
                    prediction, target, voxelspacing=voxelspacing
                )
            ),
            "jaccard": float(metric.binary.jc(prediction, target)),
            "asd": float(
                metric.binary.assd(
                    prediction, target, voxelspacing=voxelspacing
                )
            ),
        }
    if prediction.any() and not target.any():
        return {"dice": 1.0, "hd95": 0.0, "jaccard": 1.0, "asd": 0.0}
    return {"dice": 0.0, "hd95": 0.0, "jaccard": 0.0, "asd": 0.0}


def volume_metrics(prediction, target, num_classes=ACDC_NUM_CLASSES, voxelspacing=None):
    return {
        class_index: calculate_metric_percase(
            prediction == class_index,
            target == class_index,
            voxelspacing=voxelspacing,
        )
        for class_index in range(1, num_classes)
    }


def mean_metrics(per_class):
    return {
        name: float(np.mean([values[name] for values in per_class.values()]))
        for name in METRIC_NAMES
    }


def validation_dice(prediction, target, num_classes=ACDC_NUM_CLASSES):
    values = {}
    for class_index in range(1, num_classes):
        pred_mask = np.asarray(prediction == class_index).astype(bool)
        target_mask = np.asarray(target == class_index).astype(bool)
        if pred_mask.any() and target_mask.any():
            values[class_index] = float(metric.binary.dc(pred_mask, target_mask))
        elif pred_mask.any() and not target_mask.any():
            values[class_index] = 1.0
        else:
            values[class_index] = 0.0
    return values


def save_nifti_triplet(image, prediction, target, output_dir, case_name, z_spacing):
    import SimpleITK as sitk

    os.makedirs(output_dir, exist_ok=True)
    for suffix, array in (
        ("img", image),
        ("pred", prediction),
        ("gt", target),
    ):
        itk_image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        itk_image.SetSpacing((1.0, 1.0, float(z_spacing)))
        sitk.WriteImage(
            itk_image,
            os.path.join(output_dir, "{}_{}.nii.gz".format(case_name, suffix)),
        )

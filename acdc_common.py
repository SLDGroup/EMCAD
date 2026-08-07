"""Shared helpers for the standalone ACDC train/test entry points.

The existing Synapse trainer is intentionally left untouched.  ACDC uses a
separate trainer because its validation/test samples are complete volumes,
while its training samples are 2-D NPZ slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from lib.networks import EMCADNet


ACDC_NUM_CLASSES = 4
ACDC_CLASS_NAMES = ("RV", "Myo", "LV")
MODEL_ARG_NAMES = (
    "num_classes",
    "img_size",
    "encoder",
    "kernel_sizes",
    "expansion_factor",
    "lgag_ks",
    "activation_mscb",
    "no_dw_parallel",
    "concatenation",
    "no_pretrain",
    "pretrained_dir",
)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for a reproducible run."""

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``/``cuda``/``cpu`` and fail clearly when CUDA is absent."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(args: Any) -> EMCADNet:
    """Build EMCAD with the ACDC four-class output head."""

    return EMCADNet(
        num_classes=ACDC_NUM_CLASSES,
        kernel_sizes=list(args.kernel_sizes),
        expansion_factor=args.expansion_factor,
        dw_parallel=not args.no_dw_parallel,
        add=not args.concatenation,
        lgag_ks=args.lgag_ks,
        activation=args.activation_mscb,
        encoder=args.encoder,
        pretrain=not args.no_pretrain,
        pretrained_dir=args.pretrained_dir,
    )


def model_outputs(model: torch.nn.Module, images: torch.Tensor, mode: str = "train") -> List[torch.Tensor]:
    """Normalize EMCAD's list output and tolerate a single-output model."""

    outputs = model(images, mode=mode)
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = ACDC_NUM_CLASSES) -> torch.Tensor:
    """Multi-class soft Dice loss, including background like the baseline."""

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long()
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    reduce_dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 1.0 - dice.mean()


def _non_empty_subsets(items: Sequence[int]) -> Iterable[List[int]]:
    """Yield every non-empty output subset for mutation supervision."""

    for mask in range(1, 1 << len(items)):
        yield [items[index] for index in range(len(items)) if mask & (1 << index)]


def supervised_loss(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    supervision: str,
    output_aggregation: str = "sum",
) -> torch.Tensor:
    """Compute CE+Dice for last-layer, deep-supervision or mutation training."""

    if supervision == "last_layer":
        groups: Iterable[Sequence[int]] = [[len(outputs) - 1]]
    elif supervision == "deep_supervision":
        groups = ([index] for index in range(len(outputs)))
    elif supervision == "mutation":
        groups = _non_empty_subsets(list(range(len(outputs))))
    else:
        raise ValueError("supervision must be last_layer, deep_supervision or mutation")

    losses: List[torch.Tensor] = []
    for group in groups:
        if output_aggregation == "last":
            logits = outputs[group[-1]]
        else:
            logits = torch.stack([outputs[index] for index in group], dim=0).mean(dim=0)
        ce = F.cross_entropy(logits, target.long())
        losses.append(0.3 * ce + 0.7 * dice_loss(logits, target))
    return torch.stack(losses).mean()


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    img_size: int,
    inference_batch_size: int = 8,
) -> np.ndarray:
    """Predict a ``[D,H,W]`` volume in chunks and return integer labels."""

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected volume [D,H,W], got {tuple(image.shape)}")

    image = image.float()
    depth, height, width = image.shape
    slices = image.unsqueeze(1)
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, depth, max(1, inference_batch_size)):
            batch = slices[start : start + max(1, inference_batch_size)].to(device, non_blocking=True)
            if batch.shape[-2:] != (img_size, img_size):
                batch = F.interpolate(batch, size=(img_size, img_size), mode="bilinear", align_corners=False)
            logits = model_outputs(model, batch, mode="test")[-1]
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int16))
    return np.concatenate(predictions, axis=0)


def load_model_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> Dict[str, Any]:
    """Load checkpoints produced by this trainer and plain state dictionaries."""

    payload = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(payload, Mapping) and "model_state" in payload:
        state = payload["model_state"]
        metadata = dict(payload)
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
        metadata = dict(payload)
    else:
        state = payload
        metadata = {}
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint does not contain a state dictionary: {checkpoint}")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return metadata


def checkpoint_model_args(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only architecture arguments stored in a trainer checkpoint."""

    saved = metadata.get("args", {})
    if not isinstance(saved, Mapping):
        return {}
    return {name: saved[name] for name in MODEL_ARG_NAMES if name in saved}


def ensure_dataset_files(root_path: str | Path, list_dir: str | Path) -> None:
    """Fail before model construction when the expected ACDC layout is absent."""

    root = Path(root_path)
    lists = Path(list_dir)
    required = (
        root / "train",
        root / "valid",
        root / "test",
        lists / "train.txt",
        lists / "valid.txt",
        lists / "test.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ACDC paths:\n" + "\n".join(missing))

"""ACDC segmentation metrics.

``asd`` in the output is MedPy's symmetric average surface distance
(``assd``).  ACDC files in this checkout do not store physical spacing, so
the default unit is voxel.  Pass a three-value spacing tuple only when the
spacing is known for every volume.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from medpy.metric import binary as medpy_binary


METRIC_NAMES = ("dice", "hd95", "jaccard", "asd")


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Return Dice, HD95, Jaccard and symmetric ASD for one foreground mask.

    Distance metrics are undefined for a one-sided empty mask and are stored
    as NaN; aggregate means use ``nanmean``.  Two empty masks are a correct
    negative and receive Dice/Jaccard 1 and distance 0.
    """

    prediction = np.asarray(prediction).astype(bool, copy=False)
    target = np.asarray(target).astype(bool, copy=False)
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())

    if not prediction_nonempty and not target_nonempty:
        return {"dice": 1.0, "hd95": 0.0, "jaccard": 1.0, "asd": 0.0}
    if prediction_nonempty != target_nonempty:
        return {"dice": 0.0, "hd95": float("nan"), "jaccard": 0.0, "asd": float("nan")}

    spacing = tuple(float(value) for value in voxel_spacing)
    return {
        "dice": float(medpy_binary.dc(prediction, target)),
        "hd95": float(medpy_binary.hd95(prediction, target, voxelspacing=spacing)),
        "jaccard": float(medpy_binary.jc(prediction, target)),
        "asd": float(medpy_binary.assd(prediction, target, voxelspacing=spacing)),
    }


def volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for classes 1..``num_classes-1`` in one volume."""

    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    return {
        str(class_index): binary_metrics(
            prediction == class_index,
            target == class_index,
            voxel_spacing=voxel_spacing,
        )
        for class_index in range(1, num_classes)
    }


def safe_mean(values: Iterable[float]) -> float:
    """NaN-aware mean that does not emit a RuntimeWarning for all-NaN input."""

    array = np.asarray(list(values), dtype=np.float64)
    finite = array[~np.isnan(array)]
    return float(finite.mean()) if finite.size else float("nan")


def metric_means(per_class: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Average each metric over foreground classes."""

    return {name: safe_mean(per_class[key][name] for key in sorted(per_class)) for name in METRIC_NAMES}


def dice_only(prediction: np.ndarray, target: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    """Fast validation metric used for checkpoint selection."""

    result = {}
    for class_index in range(1, num_classes):
        result[str(class_index)] = binary_metrics(
            prediction == class_index,
            target == class_index,
        )["dice"]
    return result

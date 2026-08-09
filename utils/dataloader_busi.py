import csv
import hashlib
from collections import Counter
from pathlib import Path

from utils.dataloader_polyp import (
    get_loader as _get_polyp_loader,
)


SUPPORTED_EXTENSIONS = {".png"}
VALID_CLASSES = {
    "benign",
    "malignant",
}
VALID_SPLITS = {
    "train",
    "val",
    "test",
}

EXPECTED_SPLIT_CLASS_COUNTS = {
    "train": {
        "benign": 349,
        "malignant": 168,
    },
    "val": {
        "benign": 44,
        "malignant": 21,
    },
    "test": {
        "benign": 44,
        "malignant": 21,
    },
}


def _sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _class_from_id(sample_id):
    class_name = (
        sample_id
        .split("_", 1)[0]
        .casefold()
    )

    if class_name not in VALID_CLASSES:
        raise RuntimeError(
            "BUSI sample ID must start "
            "with benign_ or malignant_: "
            "{}".format(sample_id)
        )

    return class_name


def _read_manifest(dataset_root):
    path = (
        Path(dataset_root)
        / "manifest.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "BUSI manifest not found: "
            "{}".format(path)
        )

    rows = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)

        required = {
            "sample_id",
            "class_name",
            "split",
        }

        if (
            reader.fieldnames is None
            or not required.issubset(
                reader.fieldnames
            )
        ):
            raise RuntimeError(
                "BUSI manifest requires "
                "columns: {}".format(
                    sorted(required)
                )
            )

        for row in reader:
            sample_id = (
                row["sample_id"]
                .casefold()
            )
            class_name = (
                row["class_name"]
                .casefold()
            )
            split = (
                row["split"]
                .casefold()
            )

            if sample_id in rows:
                raise RuntimeError(
                    "Duplicate BUSI manifest "
                    "ID: {}".format(sample_id)
                )

            if (
                class_name
                not in VALID_CLASSES
                or split
                not in VALID_SPLITS
            ):
                raise RuntimeError(
                    "Invalid BUSI manifest "
                    "row: {}".format(row)
                )

            if (
                _class_from_id(sample_id)
                != class_name
            ):
                raise RuntimeError(
                    "BUSI manifest class "
                    "disagrees with ID: "
                    "{}".format(row)
                )

            rows[sample_id] = {
                "class_name": class_name,
                "split": split,
            }

    if len(rows) != 647:
        raise RuntimeError(
            "BUSI manifest must contain "
            "647 rows, found {}".format(
                len(rows)
            )
        )

    return rows, _sha256_file(path)


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    augmentation=False,
    split="train",
    color_image=True,
    seed=2222,
):
    if split not in VALID_SPLITS:
        raise ValueError(
            "split must be train, val, "
            "or test"
        )

    if not color_image:
        raise ValueError(
            "BUSI is supplied to the "
            "ImageNet-pretrained encoder "
            "as 3-channel input"
        )

    loader = _get_polyp_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=batchsize,
        trainsize=trainsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        augmentation=augmentation,
        split=split,
        color_image=True,
        seed=seed,
    )

    invalid = [
        str(path)
        for (
            _,
            image_path,
            mask_path,
        ) in loader.dataset.samples
        for path in (
            image_path,
            mask_path,
        )
        if path.suffix.lower()
        != ".png"
    ]

    if invalid:
        raise RuntimeError(
            "Prepared BUSI files must "
            "be PNG: {}".format(
                invalid[:10]
            )
        )

    dataset_root = (
        Path(image_root)
        .resolve()
        .parent
        .parent
    )

    (
        manifest,
        manifest_sha256,
    ) = _read_manifest(dataset_root)

    expected = {
        sample_id: row["class_name"]
        for sample_id, row
        in manifest.items()
        if row["split"] == split
    }

    actual_ids = set(
        loader.dataset.stems
    )

    if set(expected) != actual_ids:
        raise RuntimeError(
            "BUSI {} files disagree with "
            "manifest. missing={} "
            "unexpected={}".format(
                split,
                sorted(
                    set(expected)
                    - actual_ids
                )[:10],
                sorted(
                    actual_ids
                    - set(expected)
                )[:10],
            )
        )

    class_counts = dict(
        Counter(
            expected[sample_id]
            for sample_id
            in actual_ids
        )
    )

    if (
        class_counts
        != EXPECTED_SPLIT_CLASS_COUNTS[
            split
        ]
    ):
        raise RuntimeError(
            "BUSI {} class counts must "
            "be {}, found {}".format(
                split,
                EXPECTED_SPLIT_CLASS_COUNTS[
                    split
                ],
                class_counts,
            )
        )

    loader.dataset.class_counts = (
        class_counts
    )
    loader.dataset.prepared_manifest_sha256 = (
        manifest_sha256
    )
    loader.dataset.sample_ids = (
        loader.dataset.stems
    )

    return loader
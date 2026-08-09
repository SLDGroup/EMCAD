import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("benign", "malignant")
SPLITS = ("train", "val", "test")

EXPECTED_SOURCE_COUNTS = {
    "benign": 437,
    "malignant": 210,
}

EXPECTED_SPLIT_CLASS_COUNTS = {
    "train": {"benign": 349, "malignant": 168},
    "val": {"benign": 44, "malignant": 21},
    "test": {"benign": 44, "malignant": 21},
}

PROTOCOL = "emcad_80_10_10_stratified_image_level"

MASK_PATTERN = re.compile(
    r"^(?P<base>.+)_mask(?:_(?P<index>[0-9]+))?$",
    re.IGNORECASE,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def decoded_image_sha256(image):
    digest = hashlib.sha256()
    digest.update(
        str(image.shape).encode("ascii")
    )
    digest.update(
        str(image.dtype).encode("ascii")
    )
    digest.update(image.tobytes())
    return digest.hexdigest()


def resolve_source_root(requested):
    requested = (
        Path(requested)
        .expanduser()
        .resolve()
    )

    if not requested.is_dir():
        raise FileNotFoundError(
            "BUSI source directory not found: {}".format(
                requested
            )
        )

    candidates = [
        requested,
        requested / "Dataset_BUSI_with_GT",
    ]
    candidates.extend(
        requested.rglob("Dataset_BUSI_with_GT")
    )

    valid = []
    seen = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in seen:
            continue

        seen.add(candidate)

        if all(
            (candidate / name).is_dir()
            for name in CLASSES
        ):
            valid.append(candidate)

    if len(valid) != 1:
        raise RuntimeError(
            "Expected exactly one Dataset_BUSI_with_GT "
            "directory under {}, found: {}".format(
                requested,
                [str(path) for path in valid],
            )
        )

    return valid[0]


def count_original_images(class_dir):
    class_dir = Path(class_dir)

    if not class_dir.is_dir():
        return 0

    return sum(
        1
        for path in class_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
            and MASK_PATTERN.fullmatch(
                path.stem
            )
            is None
        )
    )


def collect_class_samples(
    dataset_root,
    class_name,
):
    class_dir = (
        Path(dataset_root) / class_name
    )

    images = {}
    masks = defaultdict(list)

    for path in sorted(class_dir.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() != ".png"
        ):
            continue

        mask_match = MASK_PATTERN.fullmatch(
            path.stem
        )

        if mask_match is None:
            key = path.stem.casefold()

            if key in images:
                raise RuntimeError(
                    "Duplicate BUSI image ID in {}: "
                    "{} and {}".format(
                        class_dir,
                        images[key].name,
                        path.name,
                    )
                )

            images[key] = path.resolve()

        else:
            key = (
                mask_match
                .group("base")
                .casefold()
            )
            masks[key].append(path.resolve())

    if set(images) != set(masks):
        raise RuntimeError(
            "BUSI image/mask IDs do not match "
            "in {}. missing_masks={} "
            "orphan_masks={}".format(
                class_dir,
                sorted(
                    set(images) - set(masks)
                )[:20],
                sorted(
                    set(masks) - set(images)
                )[:20],
            )
        )

    samples = []
    sample_ids = set()

    name_pattern = re.compile(
        r"^{}\s*\(([0-9]+)\)$".format(
            re.escape(class_name)
        ),
        re.IGNORECASE,
    )

    for key in sorted(images):
        image_path = images[key]

        match = name_pattern.fullmatch(
            image_path.stem
        )

        if match is None:
            raise RuntimeError(
                "Unexpected BUSI image name: {}. "
                "Expected '{} (number).png'.".format(
                    image_path.name,
                    class_name,
                )
            )

        sample_id = "{}_{:04d}".format(
            class_name,
            int(match.group(1)),
        )

        if sample_id.casefold() in sample_ids:
            raise RuntimeError(
                "Duplicate normalized sample ID: "
                + sample_id
            )

        sample_ids.add(
            sample_id.casefold()
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                "Cannot read BUSI image: "
                + str(image_path)
            )

        mask_paths = tuple(
            sorted(
                masks[key],
                key=lambda path: path.name,
            )
        )

        for mask_path in mask_paths:
            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                raise RuntimeError(
                    "Cannot read BUSI mask: "
                    + str(mask_path)
                )

            if mask.shape != image.shape[:2]:
                raise RuntimeError(
                    "Image/mask size mismatch for {}: "
                    "image={} mask={} ({})".format(
                        sample_id,
                        image.shape[:2],
                        mask.shape,
                        mask_path.name,
                    )
                )

        samples.append(
            {
                "sample_id": sample_id,
                "class_name": class_name,
                "image_source": image_path,
                "mask_sources": mask_paths,
                "image_file_sha256": (
                    sha256_file(image_path)
                ),
                "image_pixel_sha256": (
                    decoded_image_sha256(image)
                ),
            }
        )

    return samples


def build_duplicate_groups(samples):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[
            sample["image_pixel_sha256"]
        ].append(sample)

    groups = []

    for key, members in grouped.items():
        groups.append(
            {
                "key": key,
                "samples": sorted(
                    members,
                    key=lambda sample: (
                        sample["sample_id"]
                    ),
                ),
                "counts": {
                    class_name: sum(
                        member["class_name"]
                        == class_name
                        for member in members
                    )
                    for class_name in CLASSES
                },
            }
        )

    return groups


def select_exact_groups(
    groups,
    target_counts,
    seed,
    phase,
):
    target = tuple(
        target_counts[name]
        for name in CLASSES
    )

    def score(group):
        value = "{}\t{}\t{}".format(
            seed,
            phase,
            group["key"],
        ).encode("utf-8")

        return hashlib.sha256(
            value
        ).hexdigest()

    ordered = sorted(
        groups,
        key=lambda group: (
            score(group),
            group["key"],
        ),
    )

    selections = {
        (0, 0): (),
    }

    for group in ordered:
        increment = tuple(
            group["counts"][name]
            for name in CLASSES
        )

        previous_states = list(
            selections.items()
        )

        for (
            state,
            selected_keys,
        ) in previous_states:
            next_state = (
                state[0] + increment[0],
                state[1] + increment[1],
            )

            if (
                next_state[0] > target[0]
                or next_state[1] > target[1]
                or next_state in selections
            ):
                continue

            selections[next_state] = (
                selected_keys
                + (group["key"],)
            )

    if target not in selections:
        raise RuntimeError(
            "Cannot create an exact {} split "
            "with class target {} while keeping "
            "decoded-pixel duplicate groups "
            "together. Verify that you downloaded "
            "the original 780-image BUSI "
            "package.".format(
                phase,
                target_counts,
            )
        )

    return set(selections[target])


def make_partitions(groups, seed):
    val_keys = select_exact_groups(
        groups,
        EXPECTED_SPLIT_CLASS_COUNTS[
            "val"
        ],
        seed,
        "val",
    )

    remaining = [
        group
        for group in groups
        if group["key"] not in val_keys
    ]

    test_keys = select_exact_groups(
        remaining,
        EXPECTED_SPLIT_CLASS_COUNTS[
            "test"
        ],
        seed,
        "test",
    )

    partitions = {
        split: []
        for split in SPLITS
    }

    for group in groups:
        if group["key"] in val_keys:
            split = "val"
        elif group["key"] in test_keys:
            split = "test"
        else:
            split = "train"

        for sample in group["samples"]:
            copied = dict(sample)
            copied[
                "duplicate_group_size"
            ] = len(group["samples"])

            partitions[split].append(
                copied
            )

    for split in SPLITS:
        partitions[split].sort(
            key=lambda sample: (
                sample["sample_id"]
            )
        )

    actual = {
        split: {
            class_name: sum(
                sample["class_name"]
                == class_name
                for sample
                in partitions[split]
            )
            for class_name in CLASSES
        }
        for split in SPLITS
    }

    if (
        actual
        != EXPECTED_SPLIT_CLASS_COUNTS
    ):
        raise RuntimeError(
            "Internal split-count error: "
            "expected={} actual={}".format(
                EXPECTED_SPLIT_CLASS_COUNTS,
                actual,
            )
        )

    return partitions


def merge_masks(sample):
    image = cv2.imread(
        str(sample["image_source"]),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "Cannot read BUSI image: {}".format(
                sample["image_source"]
            )
        )

    merged = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    for mask_path in sample[
        "mask_sources"
    ]:
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                "Cannot read BUSI mask: "
                + str(mask_path)
            )

        if mask.shape != merged.shape:
            raise RuntimeError(
                "Mask shape changed during "
                "preparation: {}".format(
                    mask_path
                )
            )

        merged = np.maximum(
            merged,
            (mask > 0).astype(np.uint8),
        )

    if int(merged.sum()) == 0:
        raise RuntimeError(
            "Lesion mask is empty for {}".format(
                sample["sample_id"]
            )
        )

    return merged * 255


def write_target(
    source_root,
    output_root,
    partitions,
    groups,
    seed,
    normal_images,
):
    output_root = (
        Path(output_root)
        .expanduser()
        .resolve()
    )

    if (
        output_root.exists()
        or output_root.is_symlink()
    ):
        raise FileExistsError(
            "Target already exists; it was "
            "not modified: {}. Use a new "
            "--output_root or archive the old "
            "target first.".format(
                output_root
            )
        )

    if (
        output_root == source_root
        or source_root
        in output_root.parents
    ):
        raise ValueError(
            "Output root must be outside "
            "the raw BUSI directory"
        )

    output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage = Path(
        tempfile.mkdtemp(
            prefix=".BUSI.prepare.",
            dir=str(output_root.parent),
        )
    )

    try:
        rows = []

        for split in SPLITS:
            image_dir = (
                stage / split / "images"
            )
            mask_dir = (
                stage / split / "masks"
            )

            image_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            mask_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for sample in partitions[split]:
                image_target = image_dir / (
                    sample["sample_id"]
                    + ".png"
                )
                mask_target = mask_dir / (
                    sample["sample_id"]
                    + ".png"
                )

                shutil.copy2(
                    sample["image_source"],
                    image_target,
                )

                merged_mask = merge_masks(
                    sample
                )

                if not cv2.imwrite(
                    str(mask_target),
                    merged_mask,
                ):
                    raise RuntimeError(
                        "Failed to save merged "
                        "mask: "
                        + str(mask_target)
                    )

                rows.append(
                    {
                        "split": split,
                        "sample_id": (
                            sample["sample_id"]
                        ),
                        "class_name": (
                            sample["class_name"]
                        ),
                        "image": str(
                            image_target.relative_to(
                                stage
                            )
                        ),
                        "mask": str(
                            mask_target.relative_to(
                                stage
                            )
                        ),
                        "source_image": str(
                            sample["image_source"]
                        ),
                        "source_masks": (
                            json.dumps(
                                [
                                    str(path)
                                    for path
                                    in sample[
                                        "mask_sources"
                                    ]
                                ],
                                ensure_ascii=True,
                            )
                        ),
                        "source_mask_count": len(
                            sample["mask_sources"]
                        ),
                        "image_file_sha256": (
                            sample[
                                "image_file_sha256"
                            ]
                        ),
                        "image_pixel_sha256": (
                            sample[
                                "image_pixel_sha256"
                            ]
                        ),
                        "merged_mask_sha256": (
                            sha256_file(
                                mask_target
                            )
                        ),
                        "duplicate_group_size": (
                            sample[
                                "duplicate_group_size"
                            ]
                        ),
                    }
                )

        order = {
            split: index
            for index, split
            in enumerate(SPLITS)
        }

        rows.sort(
            key=lambda row: (
                order[row["split"]],
                row["sample_id"],
            )
        )

        manifest_path = (
            stage / "manifest.csv"
        )

        fieldnames = list(
            rows[0].keys()
        )

        with manifest_path.open(
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

        logical_digest = (
            hashlib.sha256()
        )

        for row in rows:
            logical_digest.update(
                (
                    "{split}\t{sample_id}\t"
                    "{class_name}\t"
                    "{image_file_sha256}\t"
                    "{image_pixel_sha256}\t"
                    "{merged_mask_sha256}\n"
                )
                .format(**row)
                .encode("utf-8")
            )

        counts = {
            split: len(partitions[split])
            for split in SPLITS
        }

        class_counts = {
            split: {
                class_name: sum(
                    sample["class_name"]
                    == class_name
                    for sample
                    in partitions[split]
                )
                for class_name in CLASSES
            }
            for split in SPLITS
        }

        duplicate_groups = [
            group
            for group in groups
            if len(group["samples"]) > 1
        ]

        cross_class_groups = [
            group
            for group in duplicate_groups
            if sum(
                group["counts"][name] > 0
                for name in CLASSES
            )
            > 1
        ]

        all_samples = [
            sample
            for split in SPLITS
            for sample
            in partitions[split]
        ]

        summary = {
            "dataset_name": "BUSI",
            "protocol": PROTOCOL,
            "split_unit": "image",
            "patient_level_split": False,
            "seed": int(seed),
            "counts": counts,
            "class_counts": class_counts,
            "total": sum(
                counts.values()
            ),
            "source_class_counts": (
                EXPECTED_SOURCE_COUNTS
            ),
            "normal_images_excluded": (
                int(normal_images)
            ),
            "multi_mask_images": sum(
                len(
                    sample["mask_sources"]
                )
                > 1
                for sample in all_samples
            ),
            "exact_pixel_duplicate_groups": (
                len(duplicate_groups)
            ),
            "images_in_exact_pixel_duplicate_groups": (
                sum(
                    len(group["samples"])
                    for group
                    in duplicate_groups
                )
            ),
            "cross_class_exact_pixel_duplicate_groups": (
                len(cross_class_groups)
            ),
            "duplicate_policy": (
                "All decoded-pixel-identical "
                "images are retained but forced "
                "into the same split."
            ),
            "near_duplicate_policy": (
                "Near-duplicate grouping was not "
                "possible from the official "
                "metadata and was not performed."
            ),
            "manifest_sha256": (
                logical_digest.hexdigest()
            ),
            "manifest_file_sha256": (
                sha256_file(manifest_path)
            ),
            "source_root": str(
                source_root
            ),
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "notes": [
                (
                    "Only benign and malignant "
                    "lesion images are used."
                ),
                (
                    "All masks belonging to one "
                    "image are merged with "
                    "logical OR."
                ),
                (
                    "The public BUSI package has "
                    "no patient IDs, so this is "
                    "not a patient-level split."
                ),
                (
                    "Train and validation are "
                    "used during training; test "
                    "remains isolated for final "
                    "evaluation."
                ),
            ],
        }

        with (
            stage / "split_summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                summary,
                stream,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            stage,
            output_root,
        )

    except Exception:
        shutil.rmtree(
            stage,
            ignore_errors=True,
        )
        raise

    print(
        "SOURCE_ROOT={}".format(
            source_root
        )
    )
    print(
        "TARGET_ROOT={}".format(
            output_root
        )
    )
    print(
        "COUNTS=train:517 val:65 test:65"
    )
    print(
        "CLASS_COUNTS={}".format(
            EXPECTED_SPLIT_CLASS_COUNTS
        )
    )
    print(
        "MANIFEST={}".format(
            output_root / "manifest.csv"
        )
    )
    print(
        "SUMMARY={}".format(
            output_root
            / "split_summary.json"
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the 647-image BUSI "
            "lesion subset for EMCAD using "
            "a fixed, auditable 80/10/10 "
            "image-level split."
        )
    )

    parser.add_argument(
        "--source_root",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        default=(
            "../data/busi/target/BUSI"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2222,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source_root = resolve_source_root(
        args.source_root
    )

    samples = []
    source_counts = {}

    for class_name in CLASSES:
        class_samples = (
            collect_class_samples(
                source_root,
                class_name,
            )
        )

        samples.extend(class_samples)
        source_counts[class_name] = len(
            class_samples
        )

    if (
        source_counts
        != EXPECTED_SOURCE_COUNTS
    ):
        raise RuntimeError(
            "Wrong BUSI lesion counts. "
            "Expected {} but found {}. "
            "Do not use the paper's erroneous "
            "487-benign figure, BUSI_WHU, or "
            "another breast-ultrasound "
            "dataset.".format(
                EXPECTED_SOURCE_COUNTS,
                source_counts,
            )
        )

    normal_images = (
        count_original_images(
            source_root / "normal"
        )
    )

    groups = build_duplicate_groups(
        samples
    )

    partitions = make_partitions(
        groups,
        args.seed,
    )

    write_target(
        source_root=source_root,
        output_root=args.output_root,
        partitions=partitions,
        groups=groups,
        seed=args.seed,
        normal_images=normal_images,
    )


if __name__ == "__main__":
    main()
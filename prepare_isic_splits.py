import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
MASK_EXTENSIONS = {".png"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | MASK_EXTENSIONS
MASK_SUFFIX = "_segmentation"
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def canonical_id(path, is_mask):
    stem = Path(path).stem
    if is_mask and stem.casefold().endswith(MASK_SUFFIX):
        stem = stem[:-len(MASK_SUFFIX)]
    return stem.casefold()


def index_directory(root, is_mask):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(root))

    extensions = MASK_EXTENSIONS if is_mask else IMAGE_EXTENSIONS
    indexed = {}

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        key = canonical_id(path, is_mask=is_mask)

        if not key:
            raise RuntimeError("Empty ISIC ID for: {}".format(path))

        if key in indexed:
            raise RuntimeError(
                "Duplicate canonical ID in {}: {} and {}".format(
                    root,
                    indexed[key].name,
                    path.name,
                )
            )

        indexed[key] = path.resolve()

    if not indexed:
        raise RuntimeError("No supported files found in: {}".format(root))

    return indexed


def collect_pairs(image_root, mask_root):
    images = index_directory(image_root, is_mask=False)
    masks = index_directory(mask_root, is_mask=True)

    image_ids = set(images)
    mask_ids = set(masks)

    if image_ids != mask_ids:
        raise RuntimeError(
            "Image/mask IDs do not match. "
            "missing_masks={} missing_images={}".format(
                sorted(image_ids - mask_ids)[:20],
                sorted(mask_ids - image_ids)[:20],
            )
        )

    return [
        {
            "key": key,
            "sample_id": images[key].stem,
            "image_source": images[key],
            "mask_source": masks[key],
        }
        for key in sorted(image_ids)
    ]


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def destination_matches(source, destination):
    source = Path(source)
    destination = Path(destination)

    try:
        if os.path.samefile(source, destination):
            return True
    except (FileNotFoundError, OSError):
        pass

    if source.stat().st_size != destination.stat().st_size:
        return False

    return sha256_file(source) == sha256_file(destination)


def materialize(source, destination, mode):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if destination.exists() and destination_matches(source, destination):
            return

        raise FileExistsError(
            "Destination exists with different content: {}".format(
                destination
            )
        )

    if mode == "hardlink":
        try:
            os.link(str(source), str(destination))
        except OSError as error:
            raise RuntimeError(
                "Hardlink failed for {}. Use --mode copy if source and "
                "target are on different filesystems. Error: {}".format(
                    source,
                    error,
                )
            ) from error

    elif mode == "symlink":
        os.symlink(str(source), str(destination))

    elif mode == "copy":
        shutil.copy2(source, destination)

    else:
        raise ValueError("Unknown materialization mode: {}".format(mode))


def deterministic_order(samples, seed):
    def score(sample):
        payload = "{}\t{}".format(seed, sample["key"]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return sorted(
        samples,
        key=lambda sample: (
            score(sample),
            sample["key"],
        ),
    )


def assert_disjoint(partitions):
    seen = {}

    for split, samples in partitions.items():
        for sample in samples:
            key = sample["key"]

            if key in seen:
                raise RuntimeError(
                    "Split leakage: {} is in both {} and {}".format(
                        key,
                        seen[key],
                        split,
                    )
                )

            seen[key] = split


def expected_names(samples):
    image_names = {
        "{}{}".format(
            sample["sample_id"],
            sample["image_source"].suffix.lower(),
        )
        for sample in samples
    }

    mask_names = {
        "{}.png".format(sample["sample_id"])
        for sample in samples
    }

    return image_names, mask_names


def reject_extra_files(root, expected, label):
    root = Path(root)

    if not root.exists():
        return

    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in ALL_EXTENSIONS
    }

    extra = sorted(actual - expected)

    if extra:
        raise RuntimeError(
            "Unexpected stale files in {} {}: {}".format(
                label,
                root,
                extra[:20],
            )
        )


def write_dataset(
    output_root,
    dataset_name,
    protocol,
    partitions,
    mode,
    seed,
    sources,
):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    assert_disjoint(partitions)

    directories = {}

    for split in ("train", "val", "test"):
        samples = partitions[split]
        image_names, mask_names = expected_names(samples)

        image_dir = output_root / split / "images"
        mask_dir = output_root / split / "masks"

        reject_extra_files(
            image_dir,
            image_names,
            "{} images".format(split),
        )
        reject_extra_files(
            mask_dir,
            mask_names,
            "{} masks".format(split),
        )

        directories[split] = (image_dir, mask_dir)

    all_rows = []

    for split in ("train", "val", "test"):
        image_dir, mask_dir = directories[split]

        for sample in partitions[split]:
            image_name = "{}{}".format(
                sample["sample_id"],
                sample["image_source"].suffix.lower(),
            )
            mask_name = "{}.png".format(sample["sample_id"])

            image_target = image_dir / image_name
            mask_target = mask_dir / mask_name

            materialize(
                sample["image_source"],
                image_target,
                mode,
            )
            materialize(
                sample["mask_source"],
                mask_target,
                mode,
            )

            all_rows.append(
                {
                    "split": split,
                    "sample_id": sample["sample_id"],
                    "image": str(
                        image_target.relative_to(output_root)
                    ),
                    "mask": str(
                        mask_target.relative_to(output_root)
                    ),
                }
            )

    all_rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["sample_id"].casefold(),
        )
    )

    manifest_path = output_root / "split_manifest.csv"
    manifest_tmp = output_root / "split_manifest.csv.tmp"

    with manifest_tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "sample_id",
                "image",
                "mask",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    os.replace(manifest_tmp, manifest_path)

    manifest_digest = hashlib.sha256()

    for row in all_rows:
        manifest_digest.update(
            (
                "{split}\t{sample_id}\t"
                "{image}\t{mask}\n"
            ).format(**row).encode("utf-8")
        )

    counts = {
        split: len(partitions[split])
        for split in ("train", "val", "test")
    }

    summary = {
        "dataset_name": dataset_name,
        "protocol": protocol,
        "split_unit": "image",
        "seed": seed,
        "counts": counts,
        "total": sum(counts.values()),
        "manifest_sha256": manifest_digest.hexdigest(),
        "materialization_mode": mode,
        "source_directories": {
            key: str(Path(value).resolve())
            for key, value in sources.items()
        },
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "notes": (
            "Canonical IDs are disjoint across splits. "
            "This does not prove patient-level or "
            "lesion-level separation."
        ),
    }

    summary_path = output_root / "split_summary.json"
    summary_tmp = output_root / "split_summary.json.tmp"

    with summary_tmp.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(summary_tmp, summary_path)

    print("DATASET={}".format(dataset_name))
    print("PROTOCOL={}".format(protocol))
    print(
        "COUNTS=train:{} val:{} test:{}".format(
            counts["train"],
            counts["val"],
            counts["test"],
        )
    )
    print("MANIFEST={}".format(manifest_path))
    print("SUMMARY={}".format(summary_path))


def prepare_2017(args):
    partitions = {
        "train": collect_pairs(
            args.train_images,
            args.train_masks,
        ),
        "val": collect_pairs(
            args.val_images,
            args.val_masks,
        ),
        "test": collect_pairs(
            args.test_images,
            args.test_masks,
        ),
    }

    expected = {
        "train": 2000,
        "val": 150,
        "test": 600,
    }

    actual = {
        split: len(samples)
        for split, samples in partitions.items()
    }

    if (
        not args.allow_nonstandard_count
        and actual != expected
    ):
        raise RuntimeError(
            "ISIC2017 official counts must be {} "
            "but found {}. Use "
            "--allow_nonstandard_count only for a "
            "documented nonstandard study.".format(
                expected,
                actual,
            )
        )

    write_dataset(
        output_root=args.output_root,
        dataset_name="ISIC2017",
        protocol="official_train_val_test",
        partitions=partitions,
        mode=args.mode,
        seed=None,
        sources={
            "train_images": args.train_images,
            "train_masks": args.train_masks,
            "val_images": args.val_images,
            "val_masks": args.val_masks,
            "test_images": args.test_images,
            "test_masks": args.test_masks,
        },
    )


def prepare_2018(args):
    samples = collect_pairs(
        args.images,
        args.masks,
    )

    if (
        not args.allow_nonstandard_count
        and len(samples) != 2594
    ):
        raise RuntimeError(
            "ISIC2018 EMCAD pool must contain "
            "2594 paired images, found {}. Use "
            "--allow_nonstandard_count only for a "
            "documented nonstandard study.".format(
                len(samples)
            )
        )

    ordered = deterministic_order(
        samples,
        args.seed,
    )

    train_count = int(len(ordered) * 0.80)
    val_count = int(len(ordered) * 0.10)

    partitions = {
        "train": ordered[:train_count],
        "val": ordered[
            train_count:train_count + val_count
        ],
        "test": ordered[
            train_count + val_count:
        ],
    }

    write_dataset(
        output_root=args.output_root,
        dataset_name="ISIC2018",
        protocol="emcad_80_10_10_image_level",
        partitions=partitions,
        mode=args.mode,
        seed=args.seed,
        sources={
            "images": args.images,
            "masks": args.masks,
        },
    )


def add_common_arguments(parser):
    parser.add_argument(
        "--output_root",
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "hardlink",
            "symlink",
            "copy",
        ],
        default="hardlink",
    )
    parser.add_argument(
        "--allow_nonstandard_count",
        action="store_true",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed, auditable ISIC "
            "splits for EMCAD"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    parser_2017 = subparsers.add_parser(
        "isic2017",
        help="Preserve the official 2000/150/600 split",
    )
    parser_2017.add_argument(
        "--train_images",
        required=True,
    )
    parser_2017.add_argument(
        "--train_masks",
        required=True,
    )
    parser_2017.add_argument(
        "--val_images",
        required=True,
    )
    parser_2017.add_argument(
        "--val_masks",
        required=True,
    )
    parser_2017.add_argument(
        "--test_images",
        required=True,
    )
    parser_2017.add_argument(
        "--test_masks",
        required=True,
    )
    add_common_arguments(parser_2017)
    parser_2017.set_defaults(
        function=prepare_2017
    )

    parser_2018 = subparsers.add_parser(
        "isic2018",
        help=(
            "Split the 2594 labeled images "
            "using fixed 80/10/10"
        ),
    )
    parser_2018.add_argument(
        "--images",
        required=True,
    )
    parser_2018.add_argument(
        "--masks",
        required=True,
    )
    parser_2018.add_argument(
        "--seed",
        type=int,
        default=2222,
    )
    add_common_arguments(parser_2018)
    parser_2018.set_defaults(
        function=prepare_2018
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
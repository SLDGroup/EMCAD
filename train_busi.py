import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import train_polyp as _base
from utils.dataloader_busi import (
    get_loader,
)


EXPECTED_PROTOCOL = (
    "emcad_80_10_10_stratified_image_level"
)

EXPECTED_COUNTS = {
    "train": 517,
    "val": 65,
    "test": 65,
}

EXPECTED_CLASS_COUNTS = {
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

_original_parse_args = (
    _base.parse_args
)
_original_evaluate_loader = (
    _base.evaluate_loader
)


def _option_was_given(name):
    return any(
        argument == name
        or argument.startswith(
            name + "="
        )
        for argument in sys.argv[1:]
    )


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


def _parse_args():
    args = _original_parse_args()

    if not _option_was_given(
        "--data_root"
    ):
        args.data_root = (
            "../data/busi/target"
        )

    if not _option_was_given(
        "--dataset_name"
    ):
        args.dataset_name = "BUSI"

    if not _option_was_given(
        "--output_dir"
    ):
        args.output_dir = (
            "./model_pth"
        )

    if not _option_was_given(
        "--img_size"
    ):
        args.img_size = 256

    if args.dataset_name != "BUSI":
        raise ValueError(
            "--dataset_name must be BUSI"
        )

    if args.grayscale:
        raise ValueError(
            "BUSI uses 3-channel input "
            "for the ImageNet-pretrained "
            "PVT encoder; remove "
            "--grayscale"
        )

    # EMCAD specifies fixed 256x256
    # input for BUSI. Polyp/ISIC
    # multi-scale training is disabled.
    args.no_multi_scale = True

    if args.run_name is None:
        args.run_name = (
            "train_BUSI_{}".format(
                datetime.now().strftime(
                    "%Y-%m-%d_%H%M%S"
                )
            )
        )

    dataset_root = (
        Path(args.data_root).resolve()
        / "BUSI"
    )

    manifest_path = (
        dataset_root / "manifest.csv"
    )
    summary_path = (
        dataset_root
        / "split_summary.json"
    )

    if (
        not manifest_path.is_file()
        or not summary_path.is_file()
    ):
        raise FileNotFoundError(
            "Prepared BUSI metadata is "
            "missing. Run "
            "prepare_busi_splits.py first:\n"
            "{}\n{}".format(
                manifest_path,
                summary_path,
            )
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        summary = json.load(stream)

    if (
        summary.get("dataset_name")
        != "BUSI"
    ):
        raise RuntimeError(
            "split_summary.json is not "
            "for BUSI"
        )

    if (
        summary.get("protocol")
        != EXPECTED_PROTOCOL
    ):
        raise RuntimeError(
            "BUSI split protocol mismatch: "
            "expected={} actual={}".format(
                EXPECTED_PROTOCOL,
                summary.get("protocol"),
            )
        )

    if (
        summary.get("counts")
        != EXPECTED_COUNTS
    ):
        raise RuntimeError(
            "BUSI split counts mismatch: "
            "expected={} actual={}".format(
                EXPECTED_COUNTS,
                summary.get("counts"),
            )
        )

    if (
        summary.get("class_counts")
        != EXPECTED_CLASS_COUNTS
    ):
        raise RuntimeError(
            "BUSI class counts mismatch: "
            "expected={} actual={}".format(
                EXPECTED_CLASS_COUNTS,
                summary.get(
                    "class_counts"
                ),
            )
        )

    manifest_file_sha256 = (
        _sha256_file(manifest_path)
    )

    if (
        summary.get(
            "manifest_file_sha256"
        )
        != manifest_file_sha256
    ):
        raise RuntimeError(
            "manifest.csv hash differs "
            "from split_summary.json"
        )

    manifest_sha256 = summary.get(
        "manifest_sha256"
    )

    if (
        not isinstance(
            manifest_sha256,
            str,
        )
        or len(manifest_sha256) != 64
    ):
        raise RuntimeError(
            "Invalid BUSI manifest "
            "SHA-256 in split_summary.json"
        )

    args.split_protocol = (
        EXPECTED_PROTOCOL
    )
    args.split_unit = summary.get(
        "split_unit"
    )
    args.split_manifest_sha256 = (
        manifest_sha256
    )
    args.patient_level_split = bool(
        summary.get(
            "patient_level_split",
            False,
        )
    )
    args.duplicate_policy = (
        summary.get(
            "duplicate_policy"
        )
    )

    run_dir = (
        Path(args.output_dir).resolve()
        / args.dataset_name
        / args.run_name
    )

    if (
        run_dir.exists()
        and any(run_dir.iterdir())
    ):
        raise FileExistsError(
            "BUSI run directory is not "
            "empty; refusing to overwrite: "
            "{}".format(run_dir)
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        manifest_path,
        run_dir
        / "data_split_manifest.csv",
    )
    shutil.copy2(
        summary_path,
        run_dir
        / "data_split_summary.json",
    )

    return args


def _evaluate_loader(
    *args,
    **kwargs,
):
    description = kwargs.get(
        "description"
    )

    if isinstance(description, str):
        kwargs["description"] = (
            description.replace(
                "Polyp",
                "BUSI",
            )
        )

    return _original_evaluate_loader(
        *args,
        **kwargs,
    )


_base.parse_args = _parse_args
_base.get_loader = get_loader
_base.evaluate_loader = (
    _evaluate_loader
)


if __name__ == "__main__":
    _base.main()
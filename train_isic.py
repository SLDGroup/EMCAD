import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import train_polyp as _base
from utils.dataloader_isic import get_loader


ALLOWED_DATASETS = {
    "ISIC2017",
    "ISIC2018",
}

EXPECTED_PROTOCOLS = {
    "ISIC2017": "official_train_val_test",
    "ISIC2018": "emcad_80_10_10_image_level",
}

_original_parse_args = _base.parse_args
_original_evaluate_loader = (
    _base.evaluate_loader
)


def _option_was_given(name):
    return any(
        argument == name
        or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def _parse_args():
    args = _original_parse_args()

    if not _option_was_given("--data_root"):
        args.data_root = "../data/isic/target"

    if not _option_was_given("--dataset_name"):
        args.dataset_name = "ISIC2018"

    if not _option_was_given("--output_dir"):
        args.output_dir = "./model_pth/ISIC"

    if args.dataset_name not in ALLOWED_DATASETS:
        raise ValueError(
            "--dataset_name must be "
            "ISIC2017 or ISIC2018"
        )

    if args.grayscale:
        raise ValueError(
            "ISIC training requires RGB input; "
            "remove --grayscale"
        )

    if args.run_name is None:
        args.run_name = (
            "train_ISIC_{}_{}".format(
                args.dataset_name,
                datetime.now().strftime(
                    "%Y-%m-%d_%H%M%S"
                ),
            )
        )

    dataset_root = (
        Path(args.data_root).resolve()
        / args.dataset_name
    )

    manifest_path = (
        dataset_root
        / "split_manifest.csv"
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
            "ISIC split metadata is missing. "
            "Run prepare_isic_splits.py first:\n"
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
        != args.dataset_name
    ):
        raise RuntimeError(
            "split_summary dataset mismatch: "
            "expected {}, got {}".format(
                args.dataset_name,
                summary.get("dataset_name"),
            )
        )

    expected_protocol = EXPECTED_PROTOCOLS[
        args.dataset_name
    ]

    if (
        summary.get("protocol")
        != expected_protocol
    ):
        raise RuntimeError(
            "split protocol mismatch: "
            "expected {}, got {}".format(
                expected_protocol,
                summary.get("protocol"),
            )
        )

    run_dir = (
        Path(args.output_dir).resolve()
        / args.dataset_name
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        manifest_path,
        run_dir / "data_split_manifest.csv",
    )
    shutil.copy2(
        summary_path,
        run_dir / "data_split_summary.json",
    )

    return args


def _evaluate_loader(*args, **kwargs):
    description = kwargs.get("description")

    if isinstance(description, str):
        kwargs["description"] = (
            description.replace(
                "Polyp",
                "ISIC",
            )
        )

    return _original_evaluate_loader(
        *args,
        **kwargs,
    )


_base.parse_args = _parse_args
_base.get_loader = get_loader
_base.evaluate_loader = _evaluate_loader


if __name__ == "__main__":
    _base.main()
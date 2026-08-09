import json
import shutil
import sys
from pathlib import Path

import test_polyp as _base
from utils.dataloader_isic import (
    SUPPORTED_EXTENSIONS,
    get_loader,
)


ALLOWED_DATASETS = {
    "ISIC2017",
    "ISIC2018",
}

EXPECTED_PROTOCOLS = {
    "ISIC2017": "official_train_val_test",
    "ISIC2018": "emcad_80_10_10_image_level",
}

ARCHITECTURE_OPTIONS = {
    "encoder": "--encoder",
    "kernel_sizes": "--kernel_sizes",
    "expansion_factor": "--expansion_factor",
    "lgag_ks": "--lgag_ks",
    "activation_mscb": "--activation_mscb",
    "no_dw_parallel": "--no_dw_parallel",
    "concatenation": "--concatenation",
    "img_size": "--img_size",
    "grayscale": "--grayscale",
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


def _normalized(value):
    if isinstance(value, tuple):
        return list(value)

    return value


def _parse_args():
    args = _original_parse_args()

    if not _option_was_given("--data_root"):
        args.data_root = "../data/isic/target"

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    config_path = (
        checkpoint.parent
        / "config.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            "Checkpoint config is required "
            "for safe ISIC testing: {}".format(
                config_path
            )
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = json.load(stream)

    config_dataset = config.get(
        "dataset_name"
    )

    if not _option_was_given(
        "--dataset_name"
    ):
        args.dataset_name = config_dataset

    if (
        args.dataset_name
        not in ALLOWED_DATASETS
    ):
        raise ValueError(
            "--dataset_name must be "
            "ISIC2017 or ISIC2018"
        )

    if config_dataset != args.dataset_name:
        raise RuntimeError(
            "Checkpoint/data mismatch: "
            "checkpoint={} requested={}".format(
                config_dataset,
                args.dataset_name,
            )
        )

    for (
        field,
        option,
    ) in ARCHITECTURE_OPTIONS.items():
        if field not in config:
            raise RuntimeError(
                "Missing architecture field "
                "in config.json: {}".format(
                    field
                )
            )

        configured = _normalized(
            config[field]
        )
        current = _normalized(
            getattr(args, field)
        )

        if (
            _option_was_given(option)
            and current != configured
        ):
            raise RuntimeError(
                "{} conflicts with checkpoint "
                "config: requested={} saved={}".format(
                    option,
                    current,
                    configured,
                )
            )

        setattr(
            args,
            field,
            config[field],
        )

    if args.grayscale:
        raise RuntimeError(
            "ISIC checkpoint unexpectedly "
            "uses grayscale input"
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
            "ISIC split metadata is missing:\n"
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
            "Dataset split summary does "
            "not match checkpoint"
        )

    if (
        summary.get("protocol")
        != EXPECTED_PROTOCOLS[
            args.dataset_name
        ]
    ):
        raise RuntimeError(
            "Dataset split protocol does "
            "not match EMCAD setup"
        )

    output_dir = Path(
        args.output_dir
        or (
            checkpoint.parent
            / "{}_{}_outputs".format(
                args.split,
                args.dataset_name,
            )
        )
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        manifest_path,
        output_dir / "data_split_manifest.csv",
    )
    shutil.copy2(
        summary_path,
        output_dir / "data_split_summary.json",
    )

    print(
        "CHECKPOINT_CONFIG={}".format(
            config_path
        )
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
_base.SUPPORTED_EXTENSIONS = (
    SUPPORTED_EXTENSIONS
)
_base.evaluate_loader = _evaluate_loader


if __name__ == "__main__":
    _base.main()
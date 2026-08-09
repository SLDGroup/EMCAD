import hashlib
import json
import shutil
import sys
from pathlib import Path

import test_polyp as _base
from utils.dataloader_busi import (
    SUPPORTED_EXTENSIONS,
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

ARCHITECTURE_OPTIONS = {
    "encoder": "--encoder",
    "kernel_sizes": "--kernel_sizes",
    "expansion_factor": (
        "--expansion_factor"
    ),
    "lgag_ks": "--lgag_ks",
    "activation_mscb": (
        "--activation_mscb"
    ),
    "no_dw_parallel": (
        "--no_dw_parallel"
    ),
    "concatenation": (
        "--concatenation"
    ),
    "img_size": "--img_size",
    "grayscale": "--grayscale",
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


def _normalized(value):
    if isinstance(value, tuple):
        return list(value)

    return value


def _parse_args():
    args = _original_parse_args()

    if not _option_was_given(
        "--data_root"
    ):
        args.data_root = (
            "../data/busi/target"
        )

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
            "for safe BUSI testing: "
            "{}".format(config_path)
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
        args.dataset_name = (
            config_dataset
        )

    if (
        args.dataset_name != "BUSI"
        or config_dataset != "BUSI"
    ):
        raise RuntimeError(
            "Checkpoint and requested "
            "dataset must both be BUSI: "
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
                "config: requested={} "
                "saved={}".format(
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
            "BUSI checkpoint unexpectedly "
            "uses grayscale model input"
        )

    saved_threshold = config.get(
        "threshold"
    )

    if saved_threshold is None:
        raise RuntimeError(
            "Training config has no "
            "threshold"
        )

    if (
        _option_was_given(
            "--threshold"
        )
        and abs(
            float(args.threshold)
            - float(saved_threshold)
        )
        > 1e-12
    ):
        raise RuntimeError(
            "--threshold conflicts with "
            "checkpoint config: requested={} "
            "saved={}".format(
                args.threshold,
                saved_threshold,
            )
        )

    args.threshold = float(
        saved_threshold
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
            "missing:\n{}\n{}".format(
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
            "Current split_summary.json "
            "is not BUSI"
        )

    if (
        summary.get("protocol")
        != EXPECTED_PROTOCOL
    ):
        raise RuntimeError(
            "Current BUSI split protocol "
            "is not the training protocol"
        )

    if (
        summary.get("counts")
        != EXPECTED_COUNTS
    ):
        raise RuntimeError(
            "Current BUSI split counts "
            "are invalid"
        )

    if (
        summary.get("class_counts")
        != EXPECTED_CLASS_COUNTS
    ):
        raise RuntimeError(
            "Current BUSI class counts "
            "are invalid"
        )

    current_manifest_sha256 = (
        _sha256_file(manifest_path)
    )

    if (
        summary.get(
            "manifest_file_sha256"
        )
        != current_manifest_sha256
    ):
        raise RuntimeError(
            "Current manifest.csv hash "
            "differs from "
            "split_summary.json"
        )

    if (
        config.get(
            "split_manifest_sha256"
        )
        != summary.get(
            "manifest_sha256"
        )
    ):
        raise RuntimeError(
            "BUSI split manifest differs "
            "from the one used for training"
        )

    output_dir = Path(
        args.output_dir
        or (
            checkpoint.parent
            / "{}_BUSI_outputs".format(
                args.split
            )
        )
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        manifest_path,
        output_dir
        / "data_split_manifest.csv",
    )
    shutil.copy2(
        summary_path,
        output_dir
        / "data_split_summary.json",
    )

    print(
        "CHECKPOINT_CONFIG={}".format(
            config_path
        )
    )
    print(
        "SPLIT_MANIFEST_SHA256={}".format(
            summary.get(
                "manifest_sha256"
            )
        )
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
_base.SUPPORTED_EXTENSIONS = (
    SUPPORTED_EXTENSIONS
)
_base.evaluate_loader = (
    _evaluate_loader
)


if __name__ == "__main__":
    _base.main()
import argparse
import os
import re
from collections import defaultdict

import numpy as np


VALID_PATTERN = re.compile(
    r"^(case_?\d+)_slice(ED|ES)_(\d+)(?:\.npz)?$",
    re.IGNORECASE,
)


def read_list(list_dir, split):
    path = os.path.join(list_dir, split + ".txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def resolve(directory, name):
    paths = [os.path.join(directory, name)]
    if not name.lower().endswith(".npz"):
        paths.append(os.path.join(directory, name + ".npz"))
    for path in paths:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(" or ".join(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./data/ACDC")
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    args = parser.parse_args()

    labels_seen = set()
    valid_groups = defaultdict(int)
    counts = {}

    for split in ("train", "valid", "test"):
        names = read_list(args.list_dir, split)
        counts[split] = len(names)
        for name in names:
            path = resolve(os.path.join(args.root_path, split), name)
            with np.load(path, allow_pickle=False) as data:
                if "img" not in data or "label" not in data:
                    raise KeyError(path + " lacks img/label")
                image = data["img"]
                label = data["label"]
            if image.shape != label.shape:
                raise ValueError(path + " image/label shape mismatch")
            expected_ndim = 3 if split == "test" else 2
            if image.ndim != expected_ndim:
                raise ValueError(
                    "{} expected {}D but got {}".format(path, expected_ndim, image.shape)
                )
            labels_seen.update(int(value) for value in np.unique(label))
            if split == "valid":
                match = VALID_PATTERN.fullmatch(os.path.basename(name))
                if match is None:
                    raise ValueError("Cannot group valid filename: " + name)
                key = "{}_{}".format(match.group(1), match.group(2).upper())
                valid_groups[key] += 1

    if not labels_seen.issubset({0, 1, 2, 3}):
        raise ValueError("Unexpected ACDC labels: {}".format(sorted(labels_seen)))

    print("ACDC_DATA_OK")
    print("train_slices={}".format(counts["train"]))
    print("valid_slices={}".format(counts["valid"]))
    print("valid_volumes={}".format(len(valid_groups)))
    print("test_volumes={}".format(counts["test"]))
    print("labels={}".format(sorted(labels_seen)))


if __name__ == "__main__":
    main()

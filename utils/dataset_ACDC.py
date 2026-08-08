import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        choice = random.random()
        if choice > 0.5:
            image, label = random_rot_flip(image, label)
        elif choice > 0.25:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if [x, y] != list(self.output_size):
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0,
            )

        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.int64)),
        }


def _read_list(list_dir, split):
    list_path = os.path.join(list_dir, split + ".txt")
    if not os.path.isfile(list_path):
        raise FileNotFoundError("Missing ACDC list: " + list_path)
    with open(list_path, "r", encoding="utf-8") as stream:
        names = [line.strip() for line in stream if line.strip()]
    if not names:
        raise RuntimeError("Empty ACDC list: " + list_path)
    return names


def _resolve_npz(directory, name):
    candidates = [os.path.join(directory, name)]
    if not name.lower().endswith(".npz"):
        candidates.append(os.path.join(directory, name + ".npz"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Missing ACDC sample: " + " or ".join(candidates))


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "img" not in data or "label" not in data:
            raise KeyError(path + " must contain 'img' and 'label'")
        image = data["img"].astype(np.float32)
        label = data["label"].astype(np.int64)
    if image.shape != label.shape:
        raise ValueError(
            "Image/label shape mismatch in {}: {} vs {}".format(
                path, image.shape, label.shape
            )
        )
    return image, label


class ACDCdataset(Dataset):
    """Two-dimensional slice dataset used only for ACDC training."""

    def __init__(self, base_dir, list_dir, split="train", transform=None):
        if split not in {"train", "valid"}:
            raise ValueError("ACDCdataset supports only train/valid slices")
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = _read_list(list_dir, split)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        name = self.sample_list[index]
        path = _resolve_npz(os.path.join(self.base_dir, self.split), name)
        image, label = _load_npz(path)
        if image.ndim != 2:
            raise ValueError("Training slice must be 2D: {} -> {}".format(path, image.shape))
        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = {
                "image": torch.from_numpy(image).unsqueeze(0),
                "label": torch.from_numpy(label),
            }
        sample["case_name"] = os.path.splitext(os.path.basename(name))[0]
        return sample


class ACDCVolumeDataset(Dataset):
    """ACDC volume dataset: regroup valid slices and read test volumes."""

    valid_pattern = re.compile(
        r"^(case_?\d+)_slice(ED|ES)_(\d+)(?:\.npz)?$",
        re.IGNORECASE,
    )

    def __init__(self, base_dir, list_dir, split):
        if split not in {"valid", "test"}:
            raise ValueError("split must be 'valid' or 'test'")
        self.base_dir = base_dir
        self.split = split
        names = _read_list(list_dir, split)

        if split == "valid":
            grouped = defaultdict(list)
            for name in names:
                base_name = os.path.basename(name)
                match = self.valid_pattern.fullmatch(base_name)
                if match is None:
                    raise ValueError(
                        "Cannot group ACDC valid slice filename: " + base_name
                    )
                case_name = "{}_volume_{}".format(
                    match.group(1), match.group(2).upper()
                )
                grouped[case_name].append((int(match.group(3)), name))
            self.samples = [
                (case_name, [name for _, name in sorted(indexed)])
                for case_name, indexed in sorted(grouped.items())
            ]
        else:
            self.samples = [
                (os.path.splitext(os.path.basename(name))[0], [name])
                for name in names
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        case_name, names = self.samples[index]
        split_dir = os.path.join(self.base_dir, self.split)

        if self.split == "valid":
            images, labels = [], []
            for name in names:
                image, label = _load_npz(_resolve_npz(split_dir, name))
                if image.ndim != 2:
                    raise ValueError("Valid item is not a 2D slice: " + name)
                images.append(image)
                labels.append(label)
            image = np.stack(images, axis=0)
            label = np.stack(labels, axis=0)
        else:
            image, label = _load_npz(_resolve_npz(split_dir, names[0]))
            if image.ndim != 3:
                raise ValueError(
                    "Test item must be a 3D volume: {} -> {}".format(
                        names[0], image.shape
                    )
                )

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "case_name": case_name,
        }

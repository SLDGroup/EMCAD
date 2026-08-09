import hashlib
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
MASK_EXTENSIONS = {".png"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS
MASK_SUFFIX = "_segmentation"


def _canonical_id(path, is_mask):
    stem = Path(path).stem

    if (
        is_mask
        and stem.casefold().endswith(MASK_SUFFIX)
    ):
        stem = stem[:-len(MASK_SUFFIX)]

    return stem.casefold()


def _index_directory(root, is_mask):
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(
            "Directory not found: {}".format(root)
        )

    extensions = (
        MASK_EXTENSIONS
        if is_mask
        else IMAGE_EXTENSIONS
    )

    indexed = {}

    for path in sorted(root.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() not in extensions
        ):
            continue

        sample_id = _canonical_id(
            path,
            is_mask=is_mask,
        )

        if not sample_id:
            raise RuntimeError(
                "Empty sample ID for: {}".format(path)
            )

        if sample_id in indexed:
            raise RuntimeError(
                "Duplicate canonical ISIC ID in {}: "
                "{} and {}".format(
                    root,
                    indexed[sample_id].name,
                    path.name,
                )
            )

        indexed[sample_id] = path

    if not indexed:
        raise RuntimeError(
            "No supported files found in: {}".format(
                root
            )
        )

    return indexed


def _seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed() % (2 ** 32)
    )
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def isic_eval_collate(batch):
    images = torch.stack(
        [item[0] for item in batch],
        dim=0,
    )
    masks = [item[1] for item in batch]
    original_sizes = torch.stack(
        [item[2] for item in batch],
        dim=0,
    )
    names = [item[3] for item in batch]

    return (
        images,
        masks,
        original_sizes,
        names,
    )


class ISICDataset(data.Dataset):
    def __init__(
        self,
        image_root,
        gt_root,
        trainsize,
        augmentation=False,
        split="train",
        color_image=True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be train, val, or test"
            )

        if not color_image:
            raise ValueError(
                "ISIC must be loaded as RGB; "
                "grayscale is unsupported"
            )

        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.split = split
        self.color_image = True

        images = _index_directory(
            image_root,
            is_mask=False,
        )
        masks = _index_directory(
            gt_root,
            is_mask=True,
        )

        image_ids = set(images)
        mask_ids = set(masks)

        if image_ids != mask_ids:
            missing_masks = sorted(
                image_ids - mask_ids
            )[:20]
            missing_images = sorted(
                mask_ids - image_ids
            )[:20]

            raise RuntimeError(
                "ISIC image/mask IDs do not match. "
                "missing_masks={} missing_images={}".format(
                    missing_masks,
                    missing_images,
                )
            )

        self.samples = [
            (
                sample_id,
                images[sample_id],
                masks[sample_id],
            )
            for sample_id in sorted(image_ids)
        ]

        self.stems = tuple(
            sample_id
            for sample_id, _, _ in self.samples
        )
        self.sample_ids = self.stems

        digest = hashlib.sha256()

        for (
            sample_id,
            image_path,
            mask_path,
        ) in self.samples:
            digest.update(
                (
                    "{}\t{}\t{}\t{}\t{}\n"
                ).format(
                    sample_id,
                    image_path.name,
                    image_path.stat().st_size,
                    mask_path.name,
                    mask_path.stat().st_size,
                ).encode("utf-8")
            )

        self.manifest_sha256 = digest.hexdigest()

        transforms = []

        if (
            self.split == "train"
            and self.augmentation
        ):
            transforms.extend(
                [
                    A.Rotate(
                        limit=90,
                        p=0.5,
                    ),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                ]
            )

        transforms.extend(
            [
                A.Resize(
                    height=self.trainsize,
                    width=self.trainsize,
                ),
                A.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
                ToTensorV2(),
            ]
        )

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (
            sample_id,
            image_path,
            mask_path,
        ) = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                "Cannot read ISIC image: {}".format(
                    image_path
                )
            )

        if mask is None:
            raise RuntimeError(
                "Cannot read ISIC mask: {}".format(
                    mask_path
                )
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                "Image/mask size mismatch for {}: "
                "image={} mask={}".format(
                    sample_id,
                    image.shape[:2],
                    mask.shape[:2],
                )
            )

        mask = (mask >= 128).astype(np.uint8)

        if self.split == "train":
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image_tensor = transformed[
                "image"
            ].float()
            mask_tensor = transformed[
                "mask"
            ].float()

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            return image_tensor, mask_tensor

        transformed = self.transform(
            image=image
        )

        image_tensor = transformed[
            "image"
        ].float()

        mask_tensor = (
            torch.from_numpy(mask)
            .unsqueeze(0)
            .float()
        )

        original_size = torch.tensor(
            mask.shape,
            dtype=torch.int64,
        )

        output_name = "{}.png".format(
            Path(image_path).stem
        )

        return (
            image_tensor,
            mask_tensor,
            original_size,
            output_name,
        )


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
    dataset = ISICDataset(
        image_root=image_root,
        gt_root=gt_root,
        trainsize=trainsize,
        augmentation=augmentation,
        split=split,
        color_image=color_image,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return data.DataLoader(
        dataset=dataset,
        batch_size=int(batchsize),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=(
            None
            if split == "train"
            else isic_eval_collate
        ),
        persistent_workers=(
            int(num_workers) > 0
        ),
    )
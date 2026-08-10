import hashlib
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def _index_by_stem(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(root))

    indexed = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            key = path.stem.casefold()
            if key in indexed:
                raise RuntimeError(
                    "Duplicate file stem in {}: {} and {}".format(
                        root, indexed[key].name, path.name
                    )
                )
            indexed[key] = path

    if not indexed:
        raise RuntimeError("No supported image files found in: {}".format(root))

    return indexed


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def polyp_eval_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = [item[1] for item in batch]
    original_sizes = torch.stack([item[2] for item in batch], dim=0)
    names = [item[3] for item in batch]
    return images, masks, original_sizes, names


class PolypDataset(data.Dataset):
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
            raise ValueError("split must be train, val, or test")

        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.split = split
        self.color_image = bool(color_image)

        images = _index_by_stem(image_root)
        masks = _index_by_stem(gt_root)

        image_keys = set(images)
        mask_keys = set(masks)

        if image_keys != mask_keys:
            missing_masks = sorted(image_keys - mask_keys)[:10]
            missing_images = sorted(mask_keys - image_keys)[:10]
            raise RuntimeError(
                "Image/mask stems do not match. "
                "missing_masks={} missing_images={}".format(
                    missing_masks, missing_images
                )
            )

        self.samples = [
            (key, images[key], masks[key])
            for key in sorted(image_keys)
        ]
        self.stems = tuple(key for key, _, _ in self.samples)

        digest = hashlib.sha256()
        for key, image_path, mask_path in self.samples:
            digest.update(
                "{}\t{}\t{}\n".format(
                    key, image_path.name, mask_path.name
                ).encode("utf-8")
            )
        self.manifest_sha256 = digest.hexdigest()

        mean = (
            (0.485, 0.456, 0.406)
            if self.color_image
            else (0.5,)
        )
        std = (
            (0.229, 0.224, 0.225)
            if self.color_image
            else (0.229,)
        )

        transforms = []

        if self.split == "train" and self.augmentation:
            transforms.extend(
                [
                    A.Rotate(limit=90, p=0.5),
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
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, image_path, mask_path = self.samples[index]

        image_flag = (
            cv2.IMREAD_COLOR
            if self.color_image
            else cv2.IMREAD_GRAYSCALE
        )

        # image = cv2.imread(str(image_path), image_flag)
        # mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # if image is None:
        #     raise RuntimeError("Cannot read image: {}".format(image_path))
        # if mask is None:
        #     raise RuntimeError("Cannot read mask: {}".format(mask_path))

        # if self.color_image:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if image.shape[:2] != mask.shape[:2]:
        #     raise RuntimeError(
        #         "Image/mask size mismatch for {}: image={} mask={}".format(
        #             key, image.shape[:2], mask.shape[:2]
        #         )
        #     )

        # if int(mask.max()) > 10:
        #     mask = (mask >= 128).astype(np.uint8)
        # else:
        #     mask = (mask > 0).astype(np.uint8)

        image = cv2.imread(
            str(image_path),
            image_flag,
        )
        mask_raw = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                "Cannot read image: {}".format(image_path)
            )

        if mask_raw is None:
            raise RuntimeError(
                "Cannot read mask: {}".format(mask_path)
            )

        if self.color_image:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        if mask_raw.ndim == 2:
            # ClinicDB, Kvasir, ColonDB, ETIS:
            # preserve the original grayscale-mask behavior.
            mask = mask_raw

            if int(mask.max()) > 10:
                mask = (
                    mask >= 128
                ).astype(np.uint8)
            else:
                mask = (
                    mask > 0
                ).astype(np.uint8)

        elif (
            mask_raw.ndim == 3
            and mask_raw.shape[2] >= 3
        ):
            # BKAI uses red and green foreground labels.
            # Merge all foreground colors into one binary mask.
            mask_signal = np.max(
                mask_raw[:, :, :3],
                axis=2,
            )

            mask = (
                mask_signal >= 128
            ).astype(np.uint8)

            if not np.any(mask):
                raise RuntimeError(
                    "Empty color mask after binarization: "
                    "{}".format(mask_path)
                )

        else:
            raise RuntimeError(
                "Unsupported mask shape for {}: {}".format(
                    mask_path,
                    mask_raw.shape,
                )
            )

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                "Image/mask size mismatch for {}: "
                "image={} mask={}".format(
                    key,
                    image.shape[:2],
                    mask.shape[:2],
                )
            )


        if self.split == "train":
            transformed = self.transform(
                image=image,
                mask=mask,
            )
            image_tensor = transformed["image"].float()
            mask_tensor = transformed["mask"].float()

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            return image_tensor, mask_tensor

        transformed = self.transform(image=image)
        image_tensor = transformed["image"].float()

        # Validation/test 保留原始分辨率 GT，避免先缩放后再放大。
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        original_size = torch.tensor(
            mask.shape,
            dtype=torch.int64,
        )
        output_name = "{}.png".format(Path(image_path).stem)

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
    dataset = PolypDataset(
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
            else polyp_eval_collate
        ),
        persistent_workers=int(num_workers) > 0,
    )
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import re  # 用于解析 case_019_sliceED_10.npz 这类文件名。2026.8.5 19:38新增
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


# class ACDCVolumeDataset(Dataset):  # 定义按完整三维体读取验证集和测试集的数据集。
#     """把 valid 二维切片重组成 volume；test 直接读取三维 NPZ。"""  # 说明类的用途。

#     _VALID_PATTERN = re.compile(  # 预编译验证集切片文件名匹配规则。
#         r"^(case_\d+)_slice(ED|ES)_(\d+)\.npz$"  # 提取病例、ED/ES 和切片编号。
#     )

#     def __init__(self, base_dir, list_dir, split):  # 接收 ACDC 根目录、列表目录和数据划分。
#         if split not in {"valid", "test"}:  # 该类只允许读取验证 volume 或测试 volume。
#             raise ValueError("split must be 'valid' or 'test'")  # 拒绝错误的数据划分。

#         self.base_dir = base_dir  # 保存 ./data/ACDC 根目录。
#         self.split = split  # 保存当前划分名称。
#         list_path = os.path.join(list_dir, split + ".txt")  # 构造 valid.txt 或 test.txt 路径。

#         if not os.path.isfile(list_path):  # 检查列表文件是否存在。
#             raise FileNotFoundError(list_path)  # 列表不存在时立即报告准确路径。

#         with open(list_path, "r", encoding="utf-8") as stream:  # 使用 UTF-8 打开列表文件。
#             file_names = [line.strip() for line in stream if line.strip()]  # 删除空行和换行符。

#         if split == "valid":  # 验证集当前保存的是二维切片。
#             grouped_files = {}  # 建立 volume 名称到切片列表的映射。

#             for file_name in file_names:  # 遍历 valid.txt 中的全部切片文件。
#                 match = self._VALID_PATTERN.fullmatch(file_name)  # 解析当前文件名。

#                 if match is None:  # 文件名不符合预期格式时不能继续。
#                     raise ValueError("Invalid ACDC valid filename: " + file_name)  # 报告错误文件名。

#                 patient_name = match.group(1)  # 取得 case_019 形式的病例名。
#                 cardiac_phase = match.group(2)  # 取得 ED 或 ES。
#                 slice_index = int(match.group(3))  # 将切片编号转换成整数。
#                 case_name = f"{patient_name}_volume_{cardiac_phase}"  # 生成 volume 名称。

#                 grouped_files.setdefault(case_name, []).append(  # 将切片加入对应 volume。
#                     (slice_index, file_name)  # 同时保存整数切片编号，防止 slice10 排在 slice2 前。
#                 )

#             self.samples = []  # 保存最终的验证 volume 索引。

#             for case_name, indexed_files in sorted(grouped_files.items()):  # 按病例名稳定排序。
#                 indexed_files.sort(key=lambda item: item[0])  # 按整数切片编号排序。
#                 ordered_files = [item[1] for item in indexed_files]  # 只保留排序后的文件名。
#                 self.samples.append((case_name, ordered_files))  # 保存一个完整验证 volume。
#         else:  # 测试集中的每个 NPZ 本身已经是三维 volume。
#             self.samples = [  # 直接建立测试 volume 索引。
#                 (os.path.splitext(file_name)[0], [file_name])  # 去掉扩展名作为 case_name。
#                 for file_name in file_names  # 遍历 test.txt。
#             ]

#         if not self.samples:  # 防止空列表导致训练后期才出错。
#             raise RuntimeError(f"No ACDC {split} samples were found")  # 立即报告空数据集。

#     def __len__(self):  # 返回验证或测试 volume 数量。
#         return len(self.samples)  # valid 应为 20，test 应为 40。

#     def __getitem__(self, index):  # 读取一个完整 ED 或 ES volume。
#         case_name, file_names = self.samples[index]  # 取得 volume 名称和所属文件。

#         if self.split == "valid":  # 验证集需要把二维切片堆叠成三维体。
#             image_slices = []  # 保存图像切片。
#             label_slices = []  # 保存标签切片。

#             for file_name in file_names:  # 按数字切片顺序逐个读取。
#                 file_path = os.path.join(  # 构造验证切片路径。
#                     self.base_dir, "valid", file_name  # 路径为 data/ACDC/valid/xxx.npz。
#                 )

#                 if not os.path.isfile(file_path):  # 检查切片文件。
#                     raise FileNotFoundError(file_path)  # 精确报告缺失路径。

#                 with np.load(file_path, allow_pickle=False) as data:  # 安全打开 NPZ。
#                     image_slices.append(data["img"].astype(np.float32))  # 图像统一为 float32。
#                     label_slices.append(data["label"].astype(np.int64))  # 标签统一为 int64。

#             image = np.stack(image_slices, axis=0)  # 得到 [D,H,W] 验证图像。
#             label = np.stack(label_slices, axis=0)  # 得到 [D,H,W] 验证标签。
#         else:  # 测试文件已经是完整三维体。
#             file_path = os.path.join(  # 构造测试 volume 路径。
#                 self.base_dir, "test", file_names[0]  # 路径为 data/ACDC/test/xxx.npz。
#             )

#             if not os.path.isfile(file_path):  # 检查测试文件。
#                 raise FileNotFoundError(file_path)  # 精确报告缺失路径。

#             with np.load(file_path, allow_pickle=False) as data:  # 安全打开三维 NPZ。
#                 image = data["img"].astype(np.float32)  # 图像统一为 float32。
#                 label = data["label"].astype(np.int64)  # 标签统一为 int64。

#         if image.ndim != 3 or label.ndim != 3:  # 检查 volume 必须为 [D,H,W]。
#             raise ValueError(  # 报告实际错误形状。
#                 f"{case_name}: image={image.shape}, label={label.shape}"
#             )

#         if image.shape != label.shape:  # 图像和标签必须逐体素对应。
#             raise ValueError(  # 报告形状不匹配。
#                 f"{case_name}: image and label shapes do not match"
#             )

#         return {  # 返回 DataLoader 能自动拼接的字典。
#             "image": torch.from_numpy(np.ascontiguousarray(image)),  # float32 [D,H,W]。
#             "label": torch.from_numpy(np.ascontiguousarray(label)),  # int64 [D,H,W]。
#             "case_name": case_name,  # 返回不带扩展名的 volume 名称。
#         }

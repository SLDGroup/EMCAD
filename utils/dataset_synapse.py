import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

"""
该函数接收一张图像（image）和对应的标签（label），对它们同步进行随机的旋转和翻转操作，最后返回处理后的图像和标签。
此代码假设标签也是类似图像的数组形式（如掩膜 Mask）。如果标签是边界框坐标或分类标签，则需要完全不同的处理方式。
"""


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
            """
                order=3 表示使用三次样条插值。
                原因：图像通常是连续的信号，包含丰富的灰度/颜色信息。三次样条插值能够产生更平滑、更高质量的缩放结果，保留更多的细节，视觉效果更好。
                为什么不是 0 或 1：order=0 是最近邻插值，会导致图像出现明显的锯齿（马赛克）；order=1 是双线性插值，比最近邻平滑，但不如三次插值精细。
             """
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            """
            原因：在分割任务中，label（标签）通常是整数掩码，代表不同的类别（例如：0=背景，1=肝脏，2=肾脏）。
            如果对标签使用 order=3（三次插值），插值过程会产生非整数的浮点数（例如 0.5, 1.2 等）。这会导致标签中出现了不存在的“中间类别”，破坏了标签的语义含义。
            必须使用 order=0：最近邻插值选择最近的像素值，能保证缩放后的标签依然是整数，不会产生新的类别。
            """
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # print(image.shape)
            # image = np.reshape(image, (512, 512))
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # label = np.reshape(label, (512, 512))


        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            # image = np.reshape(image, (image.shape[2], 512, 512))
            # label = np.reshape(label, (label.shape[2], 512, 512))
            # label[label==5]= 0
            # label[label==9]= 0
            # label[label==10]= 0
            # label[label==12]= 0
            # label[label==13]= 0
            # label[label==11]= 5

        # if self.nclass == 9:
        #     label[label==5]= 0
        #     label[label==9]= 0
        #     label[label==10]= 0
        #     label[label==12]= 0
        #     label[label==13]= 0
        #     label[label==11]= 5

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

# SLDGroup_EMCAD 项目彻底梳理

这份文档按当前仓库源码和本地文件状态重新整理，不只复述旧文档。核心目标是让你能回答三个问题：

1. 这个项目到底在做什么。
2. 数据、模型、训练、测试是怎么串起来的。
3. 本地现在什么能直接用，什么会踩坑。

---

## 1. 一句话理解这个项目

这是一个医学图像分割项目，主干思想是：

`医学图像 -> 编码器(PVTv2/ResNet)提取四层特征 -> EMCAD解码器逐级融合 -> 多尺度预测头输出分割图 -> 用多输出监督训练`

它不是从零写一个完整网络，而是把成熟编码器接上论文里的 EMCAD 解码器：

- Encoder：默认 `PVTv2-B2`，也支持 `PVTv2-B0/B1/B3/B4/B5` 和 `ResNet18/34/50/101/152`。
- Decoder：`EMCAD`，核心模块在 `lib/decoders.py`。
- 任务：
  - Synapse：腹部 CT 多器官分割，9 类，当前主线最完整。
  - Polyp：结肠息肉二分类分割，有训练/测试脚本，但本地没有 polyp 数据。
  - ACDC：心脏 MRI 分割，本地有数据和 loader，但没有接好的训练/测试入口。

---

## 2. 当前本地仓库状态

### 2.1 关键文件和目录

```text
SLDGroup_EMCAD/
├── train_synapse.py              # Synapse 训练入口
├── trainer.py                    # Synapse 训练循环和 epoch 验证
├── test_synapse.py               # Synapse 测试入口
├── train_polyp.py                # Polyp 训练入口
├── test_polyp.py                 # Polyp 测试入口
├── networks.py                   # EMCADNet 副本/注释版，不是主脚本导入对象
├── lib/
│   ├── networks.py               # 主网络 EMCADNet，训练/测试实际导入这里
│   ├── decoders.py               # EMCAD 解码器和核心模块
│   ├── pvtv2.py                  # PVTv2 编码器实现
│   └── resnet.py                 # ResNet 编码器实现
├── utils/
│   ├── dataset_synapse.py        # Synapse 数据集
│   ├── dataset_ACDC.py           # ACDC 数据集
│   ├── dataloader_polyp.py       # 当前 Polyp loader
│   ├── dataloader.py             # 旧版 Polyp loader
│   ├── utils.py                  # loss、metrics、体数据推理、保存结果
│   ├── preprocess_synapse_data.py
│   └── preprocess_synapse_data_3d.py
├── data/
│   ├── Synapse/
│   └── ACDC/
├── pretrained_pth/
│   └── pvt/pvt_v2_b2.pth         # 本地已有 PVTv2-B2 预训练权重
├── model_pth/                    # 本地没有 best.pth/last.pth
├── requirements.txt
└── LICENSE
```

### 2.2 本地数据情况

Synapse：

- `data/Synapse/train_npz`：2211 个 `.npz` 训练切片。
- `data/Synapse/test_vol_h5`：12 个 `.npy.h5` 测试体数据。
- `data/Synapse/lists/lists_Synapse/train.txt`：2211 行。
- `data/Synapse/lists/lists_Synapse/test_vol.txt`：12 行。
- 抽样训练切片 key：`image`, `label`。
- 抽样训练切片 shape：`image=(512,512)`, `label=(512,512)`。

ACDC：

- `data/ACDC/train`：1304 个 `.npz`。
- `data/ACDC/valid`：182 个 `.npz`。
- `data/ACDC/test`：40 个 `.npz`。
- 抽样 key：`img`, `label`。
- 训练/验证 slice shape 多为 `(224,224)`，测试 volume shape 如 `(10,224,224)`。
- 标签值抽样为 `0,1,2,3`。

Polyp：

- 当前本地没有 `data/polyp` 目录。
- `train_polyp.py` 默认期待路径形如：

```text
data/polyp/target/ClinicDB/train/images/
data/polyp/target/ClinicDB/train/masks/
data/polyp/target/ClinicDB/val/images/
data/polyp/target/ClinicDB/val/masks/
data/polyp/target/ClinicDB/test/images/
data/polyp/target/ClinicDB/test/masks/
```

### 2.3 权重和环境状态

- 本地有 `pretrained_pth/pvt/pvt_v2_b2.pth`，大小约 101 MB。
- 本地 `model_pth` 里没有可测试的 `best.pth` 或 `last.pth`。
- `model_pth/run_seed2222/log.txt` 显示曾经启动过训练，至少跑到 epoch 0 的 iteration 50，loss 约 `9.851843`。
- 当前系统 Python 缺少 `torch/timm/cv2/h5py` 等深度学习依赖。
- 仓库里的 `.venv` 也不是完整实验环境，连 `numpy` 都未装。
- 所以当前状态是：源码能编译，数据和预训练权重在，但不能直接训练/推理，除非先配置好 Python/PyTorch 环境。

---

## 3. Synapse 主线：从命令到结果

Synapse 是这个仓库最完整、最值得优先理解的一条线。

### 3.1 数据任务

任务是腹部 CT 多器官分割。默认 `num_classes=9`：

```text
0 background
1 spleen
2 right kidney
3 left kidney
4 gallbladder
5 pancreas
6 liver
7 stomach
8 aorta
```

注意：代码里保留了 14 类转 9 类的注释逻辑，但在当前 `utils/dataset_synapse.py` 中没有启用。也就是说，当前 loader 默认认为数据已经是 0-8 的 9 类标签。如果你换成 TransUNet 风格 14 类预处理数据，需要重新核查标签映射。

### 3.2 预处理

文件：`utils/preprocess_synapse_data.py`

处理流程：

1. 读取原始 NIfTI：`img/*.nii.gz` 和 `label/*.nii.gz`。
2. CT 值裁剪：`[-125, 275]`。
3. 归一化到 `[0,1]`。
4. 维度转置：原始 `(H,W,D)` 转为 `(D,H,W)`。
5. 训练集：每个 slice 保存一个 `.npz`，key 为 `image`, `label`。
6. 测试集：每个 volume 保存一个 `.npy.h5`，key 为 `image`, `label`。

`utils/preprocess_synapse_data_3d.py` 是多帧版本，会把连续 3 张切片拼成 3 通道输入；但当前主训练脚本和主网络默认仍按单通道 CT slice 使用。

### 3.3 数据加载

文件：`utils/dataset_synapse.py`

训练：

- 读取 `list_dir/train.txt`。
- 每行是一个 slice 名，比如 `case0031_slice000`。
- 拼出路径：`base_dir/case0031_slice000.npz`。
- 读取 `image`, `label`。
- 做随机增强和 resize。

测试/验证：

- 读取 `list_dir/test_vol.txt`。
- 每行是一个 case 名，比如 `case0008`。
- 拼出路径：`base_dir/case0008.npy.h5`。
- 读取完整 3D volume。

训练增强 `RandomGenerator`：

- 50% 概率随机 90 度旋转 + 翻转。
- 否则另有概率做 `-20` 到 `20` 度随机旋转。
- image 用 `order=3` 插值。
- label 用 `order=0` 最近邻，避免标签出现小数类别。
- 输出 image shape 为 `(1,H,W)`，label 为 `(H,W)`。

### 3.4 训练入口

文件：`train_synapse.py`

主要参数：

```text
--root_path        训练 npz 目录，默认 ./data/synapse/train_npz
--volume_path      验证 volume 目录，默认 ./data/synapse/test_vol_h5
--list_dir         列表目录，源码默认 ./lists/lists_Synapse
--num_classes      默认 9
--encoder          默认 pvt_v2_b2
--img_size         默认 224
--batch_size       默认 6
--max_epochs       默认 300
--base_lr          默认 0.0001
--supervision      mutation / deep_supervision / last_layer
```

训练入口做的事情：

1. 设置 deterministic、随机种子。
2. 按参数拼一个很长的 `snapshot_path`。
3. 创建 `EMCADNet`。
4. `model.cuda()`。
5. 调用 `trainer_synapse(args, model, snapshot_path)`。

本地运行时建议显式传路径：

```bash
python train_synapse.py ^
  --root_path ./data/Synapse/train_npz ^
  --volume_path ./data/Synapse/test_vol_h5 ^
  --list_dir ./data/Synapse/lists/lists_Synapse ^
  --pretrained_dir ./pretrained_pth/pvt
```

原因：当前源码默认 `--list_dir ./lists/lists_Synapse`，但本地实际没有根目录 `lists/`。

### 3.5 训练循环

文件：`trainer.py`

`trainer_synapse` 的核心流程：

1. 创建训练集 `Synapse_dataset(... split="train")`。
2. `DataLoader(... batch_size=batch_size, shuffle=True, num_workers=8)`。
3. 创建 loss：
   - `CrossEntropyLoss`
   - `DiceLoss(num_classes)`
4. 优化器：
   - `AdamW(lr=base_lr, weight_decay=0.0001)`
5. 每个 batch：
   - `P = model(image_batch, mode='train')`
   - `P` 是 `[p4,p3,p2,p1]` 四个输出，每个都已经上采样到输入分辨率。
   - 根据 `args.supervision` 组合输出。
   - 对组合输出计算 `0.3 * CE + 0.7 * Dice`。
   - 反向传播和 `optimizer.step()`。
6. 每个 epoch 后：
   - 保存 `last.pth`。
   - 调用 `inference()` 在测试 volume 上做验证。
   - 如果 Dice 更好，保存 `best.pth`。
   - 每 50 epoch 保存一次 `epoch_x.pth`。

### 3.6 Mutation supervision 到底是什么

`utils.utils.powerset` 会生成四个输出索引 `[0,1,2,3]` 的所有子集。

四个输出有 15 个非空组合：

```text
p4
p3
p2
p1
p4+p3
p4+p2
p4+p1
p3+p2
p3+p1
p2+p1
p4+p3+p2
p4+p3+p1
p4+p2+p1
p3+p2+p1
p4+p3+p2+p1
```

每个组合都算一次 CE + Dice，再累加。所以 mutation supervision 的 loss 数值会比只监督一个输出大很多。日志里 epoch 0 iteration 50 的 loss 约 9.85，并不一定异常，因为它是很多项 loss 的总和。

如果设置：

- `--supervision mutation`：监督所有非空组合。
- `--supervision deep_supervision`：只分别监督四个输出。
- 其他值：只监督 `P[-1]`，即最后最高分辨率输出。

### 3.7 Synapse 验证和测试

训练时验证：

- `trainer.py` 里的 `inference()`。
- 调用 `utils.utils.val_single_volume()`。
- 只计算每类 Dice，再平均。
- 不保存 PNG/NIfTI。

正式测试：

- `test_synapse.py`。
- 调用 `utils.utils.test_single_volume()`。
- 逐 volume、逐 slice 推理。
- 使用 `P[-1]` 作为最终输出。
- 计算 Dice、HD95、Jaccard、ASD。
- 可保存：
  - `case_xxx_pred.nii.gz`
  - `case_xxx_img.nii.gz`
  - `case_xxx_gt.nii.gz`
  - 每个 slice 的 overlay PNG。

测试命令也建议显式传路径，并注意 `max_iterations`：

```bash
python test_synapse.py ^
  --volume_path ./data/Synapse/test_vol_h5 ^
  --list_dir ./data/Synapse/lists/lists_Synapse ^
  --max_iterations 50000
```

原因：

- `test_synapse.py` 默认 `--volume_path ./data/synapse/test_vol_h5_new`，本地实际是 `test_vol_h5`。
- `test_synapse.py` 默认 `--max_iterations 30000`，而 `train_synapse.py` 默认是 `50000`。这个参数会参与拼接 `snapshot_path`。如果训练和测试默认值不同，测试会去错误目录找权重。
- 当前本地没有 `best.pth`，所以即使参数正确，也不能直接完成测试。

---

## 4. 模型结构：EMCADNet

主文件：`lib/networks.py`

### 4.1 总结构

`EMCADNet` 由四部分组成：

1. 输入通道转换：
   - 如果输入是 1 通道，先用 `1x1 Conv + BN + ReLU` 转成 3 通道。
   - 这是为了兼容 ImageNet 预训练的 PVTv2/ResNet。
2. Encoder：
   - 默认 PVTv2-B2。
   - 输出四层特征 `x1,x2,x3,x4`。
3. Decoder：
   - `EMCAD(x4, [x3,x2,x1])`。
   - 自顶向下逐级上采样和 skip fusion。
4. 输出头：
   - `out_head4/3/2/1` 都是 `1x1 Conv`。
   - 分别把 decoder 的四层输出变成类别 logits。
   - 再用固定 scale factor 上采样到原图尺寸。

### 4.2 PVTv2-B2 的特征尺寸

输入 `224x224` 时：

```text
x1: 64 channels,  56x56   1/4
x2: 128 channels, 28x28   1/8
x3: 320 channels, 14x14   1/16
x4: 512 channels, 7x7     1/32
```

输入 `352x352` 时，即 Polyp 默认尺寸：

```text
x1: 64 channels,  88x88
x2: 128 channels, 44x44
x3: 320 channels, 22x22
x4: 512 channels, 11x11
```

这解释了为什么项目喜欢用 `224`、`352` 这种能被 32 整除的尺寸。因为输出头使用固定倍数：

```text
p4 scale_factor=32
p3 scale_factor=16
p2 scale_factor=8
p1 scale_factor=4
```

如果输入不是 32 的倍数，固定倍数上采样可能和 label 尺寸对不上。

### 4.3 PVTv2 编码器

文件：`lib/pvtv2.py`

PVTv2 不是普通 CNN，而是分层 Transformer：

- `OverlapPatchEmbed`：用卷积方式切 patch，有重叠。
- `Attention`：支持 spatial reduction，`sr_ratio=[8,4,2,1]`。
- `Mlp`：里面有 depthwise convolution，给 token 混入局部空间信息。
- 四个 stage，每个 stage 输出一个 feature map。

PVTv2-B2 配置：

```text
embed_dims = [64, 128, 320, 512]
num_heads  = [1, 2, 5, 8]
depths     = [3, 4, 6, 3]
sr_ratios  = [8, 4, 2, 1]
```

### 4.4 ResNet 编码器

文件：`lib/resnet.py`

ResNet forward 返回四层 features：

```text
layer1 -> x1
layer2 -> x2
layer3 -> x3
layer4 -> x4
```

如果选择 ResNet：

- `resnet18/34` channels 为 `[512,256,128,64]`。
- `resnet50/101/152` channels 为 `[2048,1024,512,256]`。

注意：ResNet 预训练会通过 `model_zoo.load_url()` 下载 PyTorch 官方权重。如果环境没有网络或缓存，可能失败。

---

## 5. EMCAD 解码器

主文件：`lib/decoders.py`

EMCAD 的全称是 Efficient Multi-scale Convolutional Attention Decoding。这个项目真正的创新集中在 decoder。

### 5.1 解码流程

输入：

```text
x4: 最深层 encoder feature
skips = [x3, x2, x1]
```

流程：

```text
x4
 -> CAB4 -> SAB -> MSCB4 = d4
 -> EUCB3 上采样到 x3 尺寸
 -> LGAG3 门控 x3
 -> d3 + gated_x3
 -> CAB3 -> SAB -> MSCB3 = d3
 -> EUCB2 上采样到 x2 尺寸
 -> LGAG2 门控 x2
 -> d2 + gated_x2
 -> CAB2 -> SAB -> MSCB2 = d2
 -> EUCB1 上采样到 x1 尺寸
 -> LGAG1 门控 x1
 -> d1 + gated_x1
 -> CAB1 -> SAB -> MSCB1 = d1

return [d4, d3, d2, d1]
```

### 5.2 CAB：Channel Attention Block

作用：判断哪些通道重要。

实现：

- 对 feature 做 `AdaptiveAvgPool2d(1)`。
- 再做 `AdaptiveMaxPool2d(1)`。
- 两条路径共享 `fc1/fc2`。
- 相加后 sigmoid。
- 输出 shape 是 `(B,C,1,1)`，和原特征相乘。

### 5.3 SAB：Spatial Attention Block

作用：判断哪些空间位置重要。

实现：

- 沿 channel 维求 mean，得到 `(B,1,H,W)`。
- 沿 channel 维求 max，得到 `(B,1,H,W)`。
- 拼成 `(B,2,H,W)`。
- 经过一个 `kernel_size=7` 的卷积。
- sigmoid 后和原特征相乘。

### 5.4 MSDC：Multi-Scale Depth-wise Convolution

作用：用多个 depthwise 卷积核捕获不同尺度上下文。

默认 kernel sizes：

```text
[1, 3, 5]
```

每个分支：

```text
depthwise conv -> BatchNorm -> activation
```

`dw_parallel=True`：

- 每个 kernel 都看同一个输入。

`dw_parallel=False`：

- 分支串行，后续分支输入会加上前一个分支输出。

### 5.5 MSCB：Multi-Scale Convolution Block

MSCB 是 MSDC 的完整块：

```text
1x1 pointwise expand
 -> MSDC 多尺度 depthwise
 -> add 或 concat 聚合
 -> channel_shuffle
 -> 1x1 pointwise project
 -> residual skip
```

关键参数：

- `expansion_factor=2`：先把通道扩到 2 倍。
- `add=True`：多个 kernel 的输出相加。
- `add=False`：多个 kernel 的输出 concat。
- `activation='relu6'`：默认激活。

### 5.6 EUCB：Efficient Up-Convolution Block

作用：轻量上采样。

结构：

```text
Upsample(scale_factor=2)
 -> depthwise conv
 -> BatchNorm
 -> activation
 -> channel_shuffle
 -> 1x1 pointwise conv
```

它比普通反卷积更轻，也更符合 efficient 的设计。

### 5.7 LGAG：Large-kernel Grouped Attention Gate

作用：用 decoder 当前层特征 `g` 去门控 encoder skip 特征 `x`。

流程：

```text
g -> grouped conv -> BN
x -> grouped conv -> BN
g1 + x1 -> activation -> 1x1 conv -> BN -> sigmoid = psi
return x * psi
```

直觉：不是把 skip connection 原样加回来，而是先判断哪些 skip 信息该保留。

---

## 6. 输出和监督逻辑

### 6.1 四个输出头

`EMCADNet.forward()` 总是返回 list：

```python
[p4, p3, p2, p1]
```

虽然参数里有 `mode='test'`，但当前实现里 train/test 都返回同样的四个输出。

每个输出：

- channel 数为 `num_classes`。
- 对 Synapse 是 9。
- 对 Polyp 是 1。
- 都被上采样到输入尺寸。

### 6.2 为什么测试用 `P[-1]`

`P[-1]` 是 `p1`，来自最高分辨率 decoder 输出 `d1`，通常空间细节最好。

训练时可以监督所有层，但推理时只用最终层：

```python
outputs = P[-1]
```

Synapse：

- 对 `outputs` 做 softmax。
- `argmax` 得到类别图。

Polyp：

- 对 `outputs` 做 sigmoid。
- 阈值 0.5 得到二值 mask。

---

## 7. Polyp 分支

文件：

- `train_polyp.py`
- `test_polyp.py`
- `utils/dataloader_polyp.py`

### 7.1 任务定义

Polyp 是二分类分割：

- foreground：息肉。
- background：背景。
- `num_classes=1`，输出单通道 logits。

### 7.2 数据加载

`utils/dataloader_polyp.py` 使用 Albumentations：

训练增强：

- rotate up to 90。
- vertical flip。
- horizontal flip。
- resize 到 `img_size`。
- normalize。
- `ToTensorV2`。

mask 自适应二值化：

- 如果 mask 最大值大于 127，认为是 0/255 图，`mask > 20` 为前景。
- 否则认为是 0/1/2/3 这类标签，`mask >= 1` 为前景。

测试时返回：

```text
image, mask, original_shape, name
```

注意：`PIL.Image.size` 是 `(width, height)`。当前 `train_polyp.py` 和 `test_polyp.py` 里把 `original_shapes[0]` 命名成 `h_orig`，把 `original_shapes[1]` 命名成 `w_orig`。如果原图不是正方形，这里需要重点核查是否发生高宽互换。

### 7.3 Polyp loss

`structure_loss(pred, mask)`：

- 先用 `avg_pool2d` 生成边界/结构权重 `weit`。
- 算 weighted BCE。
- 再算 weighted IoU。
- 返回二者之和。

训练时对五项求和：

```text
loss(P[0])
loss(P[1])
loss(P[2])
loss(P[3])
loss(P[0]+P[1]+P[2]+P[3])
```

这类似 Synapse 的多输出思想，但不是 powerset mutation。

### 7.4 Polyp 训练策略

默认参数：

```text
dataset_name = ClinicDB
epoch = 200
lr = 0.0005
batchsize = 8
test_batchsize = 8
img_size = 352
clip = 0.5
augmentation = True
```

训练还用了 multi-scale：

```text
size_rates = [0.75, 1, 1.25]
```

每个 batch 会在三个尺度上各 forward/backward 一次。尺度会被 round 到 32 的倍数，以适配 PVTv2/输出上采样结构。

默认会跑 5 次：

```python
for run in [1,2,3,4,5]:
```

每次创建一个带 timestamp 的 `run_id`，保存到 `model_pth/{run_id}/`。

### 7.5 Polyp 测试输出

`test_polyp.py` 需要指定 `--run_id`。

它会读取：

```text
model_pth/{run_id}/{run_id}-best.pth
```

输出：

- `predictions_polyp/{run_id}/{dataset_name}/{split}/` 下的预测 PNG。
- `results_polyp/Results_{run_id}_{dataset_name}_{split}.xlsx`。
- `All_Runs_Summary_Polyp.xlsx` 汇总表。

指标：

- Dice
- IoU
- Sensitivity
- Specificity
- Precision
- HD95

---

## 8. ACDC 分支

文件：`utils/dataset_ACDC.py`

当前状态：

- 有本地数据。
- 有 dataset 类。
- 没有 `train_acdc.py` / `test_acdc.py`。
- 没有 trainer 接 ACDC。

数据格式：

训练/验证：

```text
data/ACDC/train/{slice_name}.npz
data/ACDC/valid/{slice_name}.npz
keys: img, label
shape: (224,224)
```

测试：

```text
data/ACDC/test/{volume_name}.npz
keys: img, label
shape: (D,224,224)
```

如果要跑 ACDC，需要做的事情：

1. 新建 `train_acdc.py`，参考 `train_synapse.py`。
2. 新建/改造 trainer，使用 `ACDCdataset`。
3. 设置 `num_classes=4`。
4. 设置 `list_dir=./data/ACDC/lists/lists_ACDC`。
5. 评估函数可复用 `val_single_volume/test_single_volume`，但 class name 和 z spacing 要重配。

---

## 9. 指标和评估细节

### 9.1 Synapse 多类指标

`utils/utils.py` 中：

- `calculate_metric_percase`：
  - Dice
  - HD95
  - Jaccard
  - ASD
- `calculate_dice_percase`：
  - 只算 Dice。

正式测试每类器官都算：

```python
for i in range(1, classes):
    metric_list.append(calculate_metric_percase(prediction == i, label == i))
```

背景类 0 不参与平均。

### 9.2 一个值得核查的指标边界情况

当前 `calculate_metric_percase` 的逻辑是：

```python
if pred.sum() > 0 and gt.sum() > 0:
    正常计算
elif pred.sum() > 0 and gt.sum() == 0:
    return 1, 0, 1, 0
else:
    return 0, 0, 0, 0
```

这在“某类 GT 不存在但模型预测了该类”时返回 Dice=1、Jaccard=1，直觉上不合理。标准定义下 false positive 通常不应记满分。

如果 Synapse 每个 case 每个器官都存在，这个问题影响小；如果有缺失类，会污染指标。建议后续严肃实验前核查。

### 9.3 可视化保存成本

`test_single_volume()` 会对每个 volume 的每个 slice 保存两张 300 dpi PNG：

- GT overlay。
- Pred overlay。

这很容易产生大量文件和内存压力。当前代码已经加了：

```python
plt.close(fig_gt)
plt.close(fig_pred)
```

个人部署笔记中提到的未关闭 figure 问题，当前源码已修复。

---

## 10. 依赖和环境

`requirements.txt` 包含：

- numpy
- loguru
- tqdm
- pyyaml
- pandas
- matplotlib
- scikit-learn
- scikit-image
- scipy
- opencv-python
- seaborn
- albumentations==1.1.0
- transformers==4.21.3
- timm==0.6.12
- h5py
- simpleitk
- nibabel
- medpy
- tensorboardx
- thop
- ptflops
- 等等

但它没有包含 `torch` 和 `torchvision`。这类项目通常需要单独按 CUDA 版本安装 PyTorch。

旧文档建议：

```bash
conda create -n emcadenv python=3.8
conda activate emcadenv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

是否必须用这个版本，取决于你的 CUDA、显卡驱动和系统环境。当前仓库源码使用的是老一代 `timm==0.6.12` 和 `transformers==4.21.3`，Python 3.8/3.9 会比太新的 Python 更稳。

---

## 11. 运行前必须知道的坑

### 11.1 `list_dir` 默认路径不对

源码默认：

```text
./lists/lists_Synapse
```

本地实际：

```text
./data/Synapse/lists/lists_Synapse
```

训练和测试都建议显式传 `--list_dir`。

### 11.2 `test_synapse.py` 默认 volume 路径不对

源码默认：

```text
./data/synapse/test_vol_h5_new
```

本地实际：

```text
./data/Synapse/test_vol_h5
```

### 11.3 训练和测试的 `max_iterations` 默认值不一致

`train_synapse.py`：

```text
max_iterations = 50000
```

`test_synapse.py`：

```text
max_iterations = 30000
```

这个参数会参与模型目录名生成。测试时如果不传一致值，会找错目录。

### 11.4 当前本地没有训练好的 Synapse 权重

只有 PVTv2 预训练权重，没有分割模型权重：

```text
pretrained_pth/pvt/pvt_v2_b2.pth   存在
model_pth/**/best.pth              不存在
model_pth/**/last.pth              不存在
```

所以不能直接做最终测试。

### 11.5 脚本里有无条件 `.cuda()`

`train_synapse.py` 和 `test_synapse.py` 都有 `model.cuda()`。如果机器没有 CUDA，直接运行会失败。虽然 `trainer.py` 内部又写了 device 判断，但入口处已经先 `.cuda()` 了。

### 11.6 Windows DataLoader worker

`trainer.py` 默认：

```python
num_workers=8
```

代码注释里写了 Windows 可能需要改成 0。实际是否需要取决于环境，但如果训练启动卡住、子进程报错或内存暴涨，优先试 `num_workers=0`。

### 11.7 `networks.py` 有两个

根目录：

```text
networks.py
```

主实现：

```text
lib/networks.py
```

训练/测试脚本导入的是：

```python
from lib.networks import EMCADNet
```

所以调试时不要改错文件。根目录 `networks.py` 更像注释/调试副本。

### 11.8 Polyp 原图高宽可能互换

`dataloader_polyp.py` 返回的 `original_shape = img.size` 是 `(width,height)`。测试代码里命名为：

```python
h_orig = original_shapes[0]
w_orig = original_shapes[1]
```

如果数据图片不是正方形，要核查 resize 回原图时是否宽高反了。

### 11.9 Polyp 同时用了 adjust_lr 和 CosineAnnealingLR

`train_polyp.py` 每个 epoch：

```python
adjust_lr(...)
train(...)
scheduler.step()
```

这等于手动阶梯衰减和 cosine scheduler 同时作用。默认 `decay_epoch=300`，而 epoch=200，所以 `adjust_lr` 实际影响不大；但如果改小 `decay_epoch`，学习率策略会变得不清晰。

### 11.10 License 不是宽松开源许可

`LICENSE` 是 UT Austin Research License，核心含义：

- 可用于 academic/research/experimental/personal use。
- 明确排除 Commercial Use。
- 不允许随意分发、再授权或转让。
- derivative products 也受同样条款约束，并有向 licensor 提供副本/授权的条款。

如果只是课程、论文复现、个人实验，一般问题不大；如果商业使用或公开分发衍生版本，需要认真处理许可。

---

## 12. 推荐阅读顺序

如果你想彻底搞懂，不要从 `pvtv2.py` 开始。那个文件细节多但不是项目主线。

建议顺序：

1. `train_synapse.py`
   - 看参数、路径、模型创建、snapshot_path。
2. `trainer.py`
   - 看训练循环、loss、保存权重、epoch 验证。
3. `utils/dataset_synapse.py`
   - 看数据从 list 到 npz/h5 的读取方式。
4. `lib/networks.py`
   - 看 EMCADNet 怎么把 encoder、decoder、heads 串起来。
5. `lib/decoders.py`
   - 重点读 EMCAD、MSCB、EUCB、LGAG、CAB、SAB。
6. `utils/utils.py`
   - 看 DiceLoss、val/test single volume、指标保存。
7. `test_synapse.py`
   - 看如何找到权重、如何调用正式测试。
8. `train_polyp.py` / `test_polyp.py`
   - 对比二分类分割和多类分割的差异。

---

## 13. 可以怎么向别人解释这个项目

最短版本：

> 这个项目复现/改造了 EMCAD 医学图像分割网络。它用 PVTv2 或 ResNet 作为编码器提取多尺度特征，用 EMCAD 解码器通过通道注意力、空间注意力、多尺度深度卷积和注意力门控 skip connection 来恢复分割图。Synapse 分支做 9 类腹部器官分割，训练时使用 CE+Dice，并对四个输出头做 mutation supervision；测试时逐 slice 推理 3D volume，用最高分辨率输出计算 Dice/HD95/Jaccard/ASD。

稍详细版本：

> 输入 CT slice 如果是单通道，会先被 1x1 卷积映射成 3 通道。PVTv2-B2 产生 1/4、1/8、1/16、1/32 四个尺度的特征。EMCAD 从最深层开始，先用 CAB/SAB 做注意力，再用 MSCB 做多尺度卷积增强，然后通过 EUCB 上采样，并用 LGAG 对对应 encoder skip feature 做门控融合。最终四个尺度各接一个 1x1 输出头，上采样回原图大小。训练时可以监督所有输出组合，推理时用最后的 p1 输出。

---

## 14. 当前项目最应该优先修的点

如果你接下来要让项目稳定可复现，我建议优先顺序是：

1. 统一路径默认值：
   - `--list_dir` 改成 `./data/Synapse/lists/lists_Synapse`。
   - `test_synapse.py --volume_path` 改成 `./data/Synapse/test_vol_h5`。
2. 统一 `train_synapse.py` 和 `test_synapse.py` 的 `max_iterations` 默认值，避免找错模型目录。
3. 把 snapshot path 生成逻辑封成一个公共函数，训练和测试共用。
4. 在入口脚本里不要无条件 `.cuda()`，改成 device 判断。
5. 给 Synapse/Polyp 写最小 smoke test：
   - dataset 能读一条样本。
   - model forward 输出 shape 正确。
   - loss 能算一次。
6. 核查 `calculate_metric_percase` 对 absent class 的处理。
7. 如果要做 ACDC，新增 `train_acdc.py` 和 `test_acdc.py`，不要混在 Synapse 脚本里硬改。

---

## 15. 你脑子里应该留下的项目地图

```text
数据层
  Synapse_dataset / ACDCdataset / PolypDataset
        |
        v
模型层
  EMCADNet
    |
    +-- input conv: 1ch -> 3ch
    +-- encoder: PVTv2 or ResNet
    +-- decoder: EMCAD
    +-- heads: p4,p3,p2,p1
        |
        v
训练层
  Synapse: CE + Dice + mutation supervision
  Polyp: structure_loss + multi-scale training
        |
        v
评估层
  Synapse: per-volume slice inference, Dice/HD95/Jaccard/ASD
  Polyp: binary threshold, Dice/IoU/Sens/Spec/Precision/HD95
        |
        v
输出层
  best.pth / last.pth / logs / TensorBoard / predictions / Excel results
```

你真正需要抓住的是两条线：

1. 数据线：`list txt -> npz/h5/image -> tensor -> model input`
2. 模型线：`encoder features -> EMCAD fusion -> four outputs -> loss/eval`

把这两条线想清楚，这个项目就不再散。

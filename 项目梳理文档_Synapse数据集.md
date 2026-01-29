# EMCAD 医学图像分割项目 - Synapse数据集完整梳理

## 📋 项目概述

**EMCAD (Efficient Multi-scale Convolutional Attention Decoding)** 是一个用于医学图像分割的高效网络架构，发表在CVPR 2024。

- **论文**: [EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation](https://arxiv.org/abs/2405.06880)
- **数据集**: Synapse Multi-organ Dataset（多器官分割）
- **任务**: 医学CT图像的9类器官分割

---

## 🗂️ 项目结构

```
SLDGroup_EMCAD/
├── train_synapse.py          # 🚀 训练入口文件
├── test_synapse.py           # 🧪 测试入口文件
├── trainer.py                # 训练器（包含trainer_synapse函数）
├── lib/
│   ├── networks.py           # EMCADNet主网络定义
│   ├── decoders.py           # EMCAD解码器（核心模块）
│   ├── pvtv2.py              # PVTv2编码器（Pyramid Vision Transformer）
│   └── resnet.py             # ResNet编码器（备选）
├── utils/
│   ├── dataset_synapse.py    # Synapse数据集加载器
│   ├── utils.py              # 工具函数（损失、评估指标等）
│   ├── preprocess_synapse_data.py  # 数据预处理脚本
│   └── transforms.py         # 数据增强
├── data/
│   └── Synapse/
│       ├── train_npz/        # 训练数据（切片，.npz格式）
│       ├── test_vol_h5/      # 测试数据（体积，.h5格式）
│       └── lists/
│           └── lists_Synapse/
│               ├── train.txt  # 训练集列表
│               └── test_vol.txt  # 测试集列表
├── pretrained_pth/
│   └── pvt/
│       └── pvt_v2_b2.pth     # 预训练编码器权重
└── model_pth/                # 训练保存的模型权重
```

---

## 🚀 入口文件详解

### 1. 训练入口：`train_synapse.py`

**功能**: 训练EMCAD模型在Synapse数据集上

**关键参数**:
```python
--root_path          # 训练数据路径: ./data/synapse/train_npz
--volume_path        # 测试数据路径: ./data/synapse/test_vol_h5
--encoder            # 编码器: pvt_v2_b2 (默认)
--num_classes        # 类别数: 9 (背景+8个器官)
--img_size           # 输入图像大小: 224x224
--batch_size         # 批次大小: 6
--max_epochs         # 最大epoch: 300
--base_lr            # 学习率: 0.0001
```

**执行流程**:
1. 解析命令行参数
2. 设置随机种子（确保可复现）
3. 创建模型：`EMCADNet`
4. 调用训练器：`trainer_synapse()`
5. 保存模型到 `model_pth/` 目录

**运行命令**:
```bash
python train_synapse.py --root_path ./data/synapse/train_npz --volume_path ./data/synapse/test_vol_h5 --encoder pvt_v2_b2
```

---

### 2. 测试入口：`test_synapse.py`

**功能**: 在测试集上评估训练好的模型

**关键参数**:
```python
--volume_path        # 测试数据路径
--encoder            # 必须与训练时一致
--is_savenii         # 是否保存预测结果为.nii.gz文件
--test_save_dir      # 预测结果保存目录
```

**执行流程**:
1. 加载训练好的模型权重（`best.pth`）
2. 对每个测试体积进行逐切片推理
3. 计算评估指标（Dice, HD95, Jaccard, ASD）
4. 保存预测结果（可选）

**运行命令**:
```bash
python test_synapse.py --volume_path ./data/synapse/test_vol_h5 --encoder pvt_v2_b2
```

---

## 📊 数据流程

### 数据预处理流程

1. **原始数据** → **预处理** (`utils/preprocess_synapse_data.py`)
   - 读取NIfTI格式的CT图像和标签
   - 窗宽窗位调整：[-125, 275] HU
   - 归一化到 [0, 1]
   - 转置维度：(H, W, D) → (D, H, W)

2. **训练数据** (`train_npz/`)
   - 格式：`.npz` 文件（每个切片一个文件）
   - 命名：`caseXXXX_sliceXXX.npz`
   - 内容：`{'image': 2D数组, 'label': 2D数组}`

3. **测试数据** (`test_vol_h5/`)
   - 格式：`.h5` 文件（每个体积一个文件）
   - 命名：`caseXXXX.npy.h5`
   - 内容：`{'image': 3D数组, 'label': 3D数组}`

### 数据加载流程 (`utils/dataset_synapse.py`)

**训练时**:
```python
Synapse_dataset(
    base_dir='./data/synapse/train_npz',
    split='train',
    transform=RandomGenerator(output_size=[224, 224])  # 数据增强
)
```

**数据增强** (`RandomGenerator`):
- 随机旋转90°（0, 90, 180, 270）
- 随机翻转（水平/垂直）
- 随机旋转（-20°到20°）
- 调整大小到224x224

**测试时**:
```python
Synapse_dataset(
    base_dir='./data/synapse/test_vol_h5',
    split='test_vol',
    transform=None  # 无增强
)
```

---

## 🏗️ 网络架构

### EMCADNet 整体架构

```
输入图像 (1通道, 224x224)
    ↓
[Conv 1→3通道]  # 将灰度图转为3通道
    ↓
[PVTv2-B2 编码器]  # 提取多尺度特征
    ↓
x1, x2, x3, x4  # 4个不同尺度的特征图
    ↓
[EMCAD 解码器]  # 高效多尺度卷积注意力解码
    ↓
d4, d3, d2, d1  # 4个解码器输出
    ↓
[预测头]  # 4个1x1卷积
    ↓
p4, p3, p2, p1  # 4个预测结果
    ↓
[上采样到原图大小]  # 双线性插值
    ↓
最终预测 (9类, 224x224)
```

### 编码器：PVTv2-B2

- **输入**: 3通道图像 (224×224)
- **输出**: 4个多尺度特征图
  - x1: 64通道, 56×56 (1/4分辨率)
  - x2: 128通道, 28×28 (1/8分辨率)
  - x3: 320通道, 14×14 (1/16分辨率)
  - x4: 512通道, 7×7 (1/32分辨率)

### 解码器：EMCAD (核心创新)

**主要组件**:

1. **MSCB (Multi-Scale Convolution Block)**
   - 多尺度深度卷积（kernel sizes: 1, 3, 5）
   - 并行或串行执行
   - 扩展因子：2

2. **EUCB (Efficient Up-Convolution Block)**
   - 高效上采样模块
   - 深度可分离卷积 + 上采样

3. **LGAG (Large-kernel Grouped Attention Gate)**
   - 大核分组注意力门控
   - 用于特征融合

4. **CAB (Channel Attention Block)**
   - 通道注意力机制

5. **SAB (Spatial Attention Block)**
   - 空间注意力机制

**解码流程**:
```
x4 (512, 7×7)
    ↓ [CAB + SAB + MSCB4]
d4 (512, 7×7)
    ↓ [EUCB3]
d3_up (320, 14×14)
    ↓ [LGAG3] + x3 (skip connection)
d3 (320, 14×14)
    ↓ [CAB + SAB + MSCB3]
d3 (320, 14×14)
    ↓ [EUCB2]
d2_up (128, 28×28)
    ↓ [LGAG2] + x2 (skip connection)
d2 (128, 28×28)
    ↓ [CAB + SAB + MSCB2]
d2 (128, 28×28)
    ↓ [EUCB1]
d1_up (64, 56×56)
    ↓ [LGAG1] + x1 (skip connection)
d1 (64, 56×56)
    ↓ [CAB + SAB + MSCB1]
d1 (64, 56×56)
```

---

## 🎯 训练流程 (`trainer.py`)

### trainer_synapse 函数

**训练循环**:
```python
for epoch in range(max_epochs):  # 300 epochs
    for batch in trainloader:
        # 1. 前向传播
        P = model(image_batch, mode='train')  # 返回4个预测
        
        # 2. 计算损失（Mutation Supervision）
        loss = 0.0
        for subset in powerset([0,1,2,3]):  # 所有子集组合
            combined_output = sum(P[i] for i in subset)
            loss_ce = CrossEntropyLoss(combined_output, label)
            loss_dice = DiceLoss(combined_output, label)
            loss += 0.3 * loss_ce + 0.7 * loss_dice
        
        # 3. 反向传播
        loss.backward()
        optimizer.step()
    
    # 4. 验证（每个epoch）
    performance = inference(args, model, best_performance)
    
    # 5. 保存最佳模型
    if performance > best_performance:
        save_model('best.pth')
```

**损失函数**:
- **CrossEntropy Loss**: 权重 0.3
- **Dice Loss**: 权重 0.7
- **Mutation Supervision**: 使用所有输出子集的组合进行监督

**优化器**:
- **AdamW**: lr=0.0001, weight_decay=0.0001

**评估指标**:
- **Dice Score**: 主要指标
- **HD95**: 95% Hausdorff距离
- **Jaccard**: IoU
- **ASD**: 平均表面距离

---

## 🧪 测试流程

### test_synapse.py 执行流程

1. **加载模型**: 从 `model_pth/.../best.pth` 加载权重
2. **逐体积推理**:
   ```python
   for volume in testloader:
       for slice in volume:  # 逐切片处理
           pred = model(slice)  # 推理
           # 保存预测结果（PNG + NIfTI）
   ```
3. **计算指标**: 对每个器官类别计算Dice、HD95等
4. **保存结果**: 
   - PNG图像（带掩码叠加）
   - NIfTI文件（3D体积）

---

## 📈 Synapse数据集信息

**数据集**: Synapse Multi-organ Dataset

**类别** (9类):
- 0: 背景
- 1: 脾脏 (Spleen)
- 2: 右肾 (Right Kidney)
- 3: 左肾 (Left Kidney)
- 4: 胆囊 (Gallbladder)
- 5: 胰腺 (Pancreas)
- 6: 肝脏 (Liver)
- 7: 胃 (Stomach)
- 8: 主动脉 (Aorta)

**数据划分**:
- **训练集**: 18个CT扫描（切片级别，约2200+切片）
- **测试集**: 12个CT扫描（体积级别）

**数据格式**:
- 原始: NIfTI (.nii.gz)
- 预处理后: 
  - 训练: NPZ (每个切片一个文件)
  - 测试: H5 (每个体积一个文件)

---

## 🔧 关键配置参数

### 网络参数
- `encoder`: `pvt_v2_b2` (默认)
- `kernel_sizes`: `[1, 3, 5]` (多尺度卷积核)
- `expansion_factor`: `2` (MSCB扩展因子)
- `lgag_ks`: `3` (LGAG核大小)
- `activation_mscb`: `relu6` (激活函数)

### 训练参数
- `batch_size`: `6`
- `max_epochs`: `300`
- `base_lr`: `0.0001`
- `img_size`: `224`
- `supervision`: `mutation` (损失监督方式)

### 数据参数
- `num_classes`: `9`
- `z_spacing`: `1` (切片间距)

---

## 📝 使用步骤

### 1. 环境配置
```bash
conda create -n emcadenv python=3.8
conda activate emcadenv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 2. 数据准备
- 从[Synapse官网](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)下载数据
- 按照TransUNet的方式划分训练/测试集
- 运行预处理脚本：
  ```bash
  python utils/preprocess_synapse_data.py
  ```

### 3. 下载预训练权重
- 下载PVTv2-B2预训练权重到 `pretrained_pth/pvt/pvt_v2_b2.pth`

### 4. 训练
```bash
python train_synapse.py \
    --root_path ./data/synapse/train_npz \
    --volume_path ./data/synapse/test_vol_h5 \
    --encoder pvt_v2_b2 \
    --batch_size 6 \
    --max_epochs 300
```

### 5. 测试
```bash
python test_synapse.py \
    --volume_path ./data/synapse/test_vol_h5 \
    --encoder pvt_v2_b2 \
    --is_savenii
```

---

## 🎨 输出结果

### 训练输出
- `model_pth/.../best.pth`: 最佳模型权重
- `model_pth/.../last.pth`: 最后一个epoch的权重
- `model_pth/.../log.txt`: 训练日志
- `model_pth/.../log/`: TensorBoard日志

### 测试输出
- `predictions/.../caseXXXX_pred.nii.gz`: 预测结果（3D）
- `predictions/.../caseXXXX_gt.nii.gz`: 真实标签（3D）
- `predictions/.../caseXXXX_img.nii.gz`: 原始图像（3D）
- `predictions/.../caseXXXX_sliceXXX_pred.png`: 预测可视化（2D）
- `predictions/.../caseXXXX_sliceXXX_gt.png`: 真实标签可视化（2D）

---

## 🔍 关键代码位置

| 功能 | 文件位置 | 关键函数/类 |
|------|---------|------------|
| 训练入口 | `train_synapse.py` | `main()` |
| 测试入口 | `test_synapse.py` | `inference()` |
| 训练器 | `trainer.py` | `trainer_synapse()` |
| 网络定义 | `lib/networks.py` | `EMCADNet` |
| 解码器 | `lib/decoders.py` | `EMCAD` |
| 数据加载 | `utils/dataset_synapse.py` | `Synapse_dataset` |
| 损失函数 | `utils/utils.py` | `DiceLoss` |
| 评估指标 | `utils/utils.py` | `test_single_volume()` |

---

## 💡 核心创新点

1. **EMCAD解码器**: 高效的多尺度卷积注意力解码
2. **MSCB模块**: 多尺度深度卷积块，捕获不同尺度的特征
3. **LGAG**: 大核分组注意力门控，用于特征融合
4. **Mutation Supervision**: 使用所有输出子集的组合进行监督学习

---

## 📚 参考资料

- 论文: [EMCAD: Efficient Multi-scale Convolutional Attention Decoding](https://arxiv.org/abs/2405.06880)
- 代码: [GitHub Repository](https://github.com/SLDGroup/EMCAD/)
- 数据集: [Synapse Multi-organ Dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)

---

**最后更新**: 2024年


# Synapse数据集9个类别详解

## 📋 概述

Synapse Multi-organ Dataset（多器官分割数据集）是一个用于腹部CT图像多器官分割的医学图像数据集。在这个EMCAD项目中，**使用9个类别**进行分割任务。

---

## 🏷️ 9个类别的详细说明

### 类别列表

| 类别ID | 类别名称 | 英文名称 | 说明 |
|--------|---------|---------|------|
| **0** | 背景 | Background | 所有非器官区域 |
| **1** | 脾脏 | Spleen | 位于左上腹部的免疫器官 |
| **2** | 右肾 | Right Kidney | 右侧肾脏 |
| **3** | 左肾 | Left Kidney | 左侧肾脏 |
| **4** | 胆囊 | Gallbladder | 储存胆汁的器官 |
| **5** | 胰腺 | Pancreas | 分泌消化酶和胰岛素的器官 |
| **6** | 肝脏 | Liver | 人体最大的内脏器官 |
| **7** | 胃 | Stomach | 消化器官 |
| **8** | 主动脉 | Aorta | 人体最大的动脉血管 |

---

## 🔄 从14类到9类的转换

### 原始Synapse数据集

Synapse数据集**原始标注包含14个类别**：
1. 背景 (Background)
2. 脾脏 (Spleen)
3. 右肾 (Right Kidney)
4. 左肾 (Left Kidney)
5. 胆囊 (Gallbladder)
6. 食管 (Esophagus) - **被移除**
7. 肝脏 (Liver)
8. 胃 (Stomach)
9. 主动脉 (Aorta)
10. 下腔静脉 (Inferior Vena Cava) - **被移除**
11. 门静脉和脾静脉 (Portal Vein and Splenic Vein) - **被移除**
12. 胰腺 (Pancreas)
13. 右肾上腺 (Right Adrenal Gland) - **被移除**
14. 左肾上腺 (Left Adrenal Gland) - **被移除**

### 转换逻辑

在 `utils/dataset_synapse.py` 中，通过以下代码将14类转换为9类：

```python
if self.nclass == 9:
    label[label==5]= 0      # 食管 → 背景
    label[label==9]= 0      # 下腔静脉 → 背景
    label[label==10]= 0     # 门静脉和脾静脉 → 背景
    label[label==12]= 0     # 右肾上腺 → 背景
    label[label==13]= 0     # 左肾上腺 → 背景
    label[label==11]= 5     # 胰腺从11 → 5（重新编号）
```

**转换规则**：
- **保留的8个器官**：脾脏、右肾、左肾、胆囊、胰腺、肝脏、胃、主动脉
- **移除的5个类别**：食管、下腔静脉、门静脉和脾静脉、右肾上腺、左肾上腺 → 全部归为背景(0)
- **重新编号**：胰腺从原始标签11重新编号为5

---

## 🎯 与项目的关系

### 1. 网络输出维度

```python
# lib/networks.py
self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)  # num_classes = 9
self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
```

**网络输出**：每个像素点输出9个类别的概率分布
- 输出形状：`(batch_size, 9, 224, 224)`
- 经过softmax后，每个像素的9个值之和为1
- 取argmax得到最终预测类别：`(batch_size, 224, 224)`

### 2. 损失函数计算

```python
# trainer.py
ce_loss = CrossEntropyLoss()  # 需要9个类别的one-hot编码
dice_loss = DiceLoss(num_classes=9)  # 对9个类别分别计算Dice损失
```

**损失计算**：
- **CrossEntropy Loss**：计算9类分类的交叉熵
- **Dice Loss**：对每个器官类别（1-8）分别计算Dice系数，然后求平均

### 3. 评估指标

```python
# utils/utils.py - test_single_volume()
for i in range(1, classes):  # 从1到8，跳过背景
    metric_list.append(calculate_metric_percase(prediction == i, label == i))
```

**评估指标**（对每个器官类别分别计算）：
- **Dice Score**：主要评估指标
- **HD95**：95% Hausdorff距离
- **Jaccard (IoU)**：交并比
- **ASD**：平均表面距离

**注意**：评估时只计算8个器官类别（1-8），**不包括背景(0)**

### 4. 可视化输出

```python
# test_synapse.py
classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 
           'pancreas', 'liver', 'stomach', 'aorta']
```

在测试时，会为每个器官类别生成：
- **彩色掩码叠加图**：不同器官用不同颜色标注
- **NIfTI文件**：保存3D分割结果

---

## 📊 类别在CT图像中的位置

### 腹部CT横断面典型位置

```
┌─────────────────────────────────────┐
│        腹部CT横断面示意图            │
├─────────────────────────────────────┤
│                                     │
│   [胃]        [肝脏]                │
│                                     │
│   [脾脏]      [胆囊]                │
│                                     │
│   [左肾]      [右肾]                │
│                                     │
│         [胰腺]                      │
│                                     │
│         [主动脉]                    │
│                                     │
└─────────────────────────────────────┘
```

### 各器官的CT特征

1. **脾脏 (Spleen)**
   - 位置：左上腹部
   - CT值：约40-60 HU（软组织密度）
   - 形状：新月形或椭圆形

2. **肾脏 (Kidneys)**
   - 位置：腹膜后，脊柱两侧
   - CT值：约30-50 HU
   - 形状：豆形，有肾门结构

3. **胆囊 (Gallbladder)**
   - 位置：肝脏下方
   - CT值：胆汁约0-20 HU
   - 形状：梨形囊状结构

4. **胰腺 (Pancreas)**
   - 位置：上腹部，横跨脊柱前方
   - CT值：约40-50 HU
   - 形状：长条形，分头、体、尾

5. **肝脏 (Liver)**
   - 位置：右上腹部
   - CT值：约50-70 HU
   - 形状：最大内脏器官，分左右叶

6. **胃 (Stomach)**
   - 位置：左上腹部
   - CT值：内容物变化大（空气、食物、液体）
   - 形状：囊状，有胃壁结构

7. **主动脉 (Aorta)**
   - 位置：脊柱前方
   - CT值：血液约30-50 HU，增强后明显
   - 形状：管状血管结构

---

## 🔍 为什么选择9个类别？

### 1. 临床重要性
这8个器官是腹部CT中最常需要分割和评估的器官，具有重要的临床意义：
- **诊断**：器官大小、形态异常
- **手术规划**：术前评估器官位置
- **疾病监测**：肿瘤、炎症等病变

### 2. 数据标注质量
这8个器官在CT图像中：
- **边界清晰**：容易标注
- **对比度好**：与周围组织区分明显
- **标注一致性强**：不同标注者之间差异小

### 3. 计算效率
- **减少类别数**：从14类减少到9类，降低计算复杂度
- **提高精度**：专注于重要器官，避免小器官带来的噪声

### 4. 与TransUNet等基准方法一致
为了与现有方法（如TransUNet）进行公平对比，使用相同的9类设置。

---

## 💻 代码中的使用示例

### 训练时的类别处理

```python
# train_synapse.py
parser.add_argument('--num_classes', type=int, default=9, 
                    help='output channel of network')

# 创建模型
model = EMCADNet(num_classes=9, ...)  # 输出9个类别
```

### 数据加载时的标签转换

```python
# utils/dataset_synapse.py
def __getitem__(self, idx):
    image, label = data['image'], data['label']
    
    # 14类 → 9类转换
    if self.nclass == 9:
        label[label==5]= 0   # 食管 → 背景
        label[label==9]= 0   # 下腔静脉 → 背景
        label[label==10]= 0  # 门静脉 → 背景
        label[label==12]= 0  # 右肾上腺 → 背景
        label[label==13]= 0  # 左肾上腺 → 背景
        label[label==11]= 5  # 胰腺重新编号
```

### 测试时的类别名称

```python
# test_synapse.py
classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 
           'pancreas', 'liver', 'stomach', 'aorta']

# 计算每个类别的指标
for i in range(1, args.num_classes):  # 1-8，跳过背景
    logging.info('Mean class (%d) %s mean_dice %f' % 
                (i, classes[i-1], metric_list[i-1][0]))
```

---

## 📈 评估结果示例

训练完成后，测试输出类似：

```
Mean class (1) spleen mean_dice 0.923456 mean_hd95 2.345
Mean class (2) right kidney mean_dice 0.912345 mean_hd95 1.876
Mean class (3) left kidney mean_dice 0.934567 mean_hd95 1.654
Mean class (4) gallbladder mean_dice 0.789012 mean_hd95 3.456
Mean class (5) pancreas mean_dice 0.756789 mean_hd95 4.123
Mean class (6) liver mean_dice 0.945678 mean_hd95 2.789
Mean class (7) stomach mean_dice 0.823456 mean_hd95 3.234
Mean class (8) aorta mean_dice 0.912345 mean_hd95 2.567

Testing performance: mean_dice : 0.874567
```

---

## 🎨 可视化示例

测试时会生成每个切片的可视化结果：

```
predictions/
├── case0001_0_pred.png    # 第0个切片，预测结果
├── case0001_0_gt.png       # 第0个切片，真实标签
├── case0001_1_pred.png
├── case0001_1_gt.png
...
```

每个PNG图像中：
- **不同颜色**代表不同的器官类别
- **半透明叠加**在原始CT图像上
- **便于直观查看**分割效果

---

## 🔗 总结

**9个类别在项目中的作用**：

1. **定义网络输出维度**：`num_classes=9`
2. **指导损失函数计算**：CE Loss + Dice Loss（9类）
3. **评估模型性能**：每个器官类别的Dice、HD95等指标
4. **生成可视化结果**：不同颜色标注不同器官
5. **保存分割结果**：NIfTI格式，每个像素值为0-8

**这9个类别是项目的核心**，整个训练、测试、评估流程都围绕这9个类别展开！




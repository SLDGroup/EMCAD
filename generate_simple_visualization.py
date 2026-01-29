"""
生成简化的项目可视化图表（不依赖中文字体）
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 创建图表
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ========== 1. 项目结构图 ==========
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Project Structure', fontsize=14, fontweight='bold', pad=20)

# 根目录
root = FancyBboxPatch((1, 8), 8, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(root)
ax1.text(5, 8.75, 'SLDGroup_EMCAD/', ha='center', va='center', fontsize=12, fontweight='bold')

# 主要文件
files = [
    ('train_synapse.py', 2, 6.5, 'lightgreen'),
    ('test_synapse.py', 5, 6.5, 'lightgreen'),
    ('trainer.py', 8, 6.5, 'lightyellow'),
    ('lib/', 2, 5, 'lightcoral'),
    ('utils/', 5, 5, 'lightcoral'),
    ('data/Synapse/', 8, 5, 'lightcoral'),
]

for name, x, y, color in files:
    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(x, y, name, ha='center', va='center', fontsize=9)

# lib子目录
lib_files = ['networks.py', 'decoders.py', 'pvtv2.py']
for i, f in enumerate(lib_files):
    box = FancyBboxPatch((1.5, 3.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(2.25, 3.75-i*0.8, f, ha='center', va='center', fontsize=8)

# utils子目录
utils_files = ['dataset_synapse.py', 'utils.py', 'preprocess.py']
for i, f in enumerate(utils_files):
    box = FancyBboxPatch((4.5, 3.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(5.25, 3.75-i*0.8, f, ha='center', va='center', fontsize=8)

# data子目录
data_files = ['train_npz/', 'test_vol_h5/', 'lists/']
for i, f in enumerate(data_files):
    box = FancyBboxPatch((7.5, 3.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(8.25, 3.75-i*0.8, f, ha='center', va='center', fontsize=8)

# 箭头
for x in [2, 5, 8]:
    ax1.arrow(x, 4.7, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# ========== 2. 数据流程图 ==========
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Data Flow', fontsize=14, fontweight='bold', pad=20)

# 流程步骤
steps = [
    ('Raw NIfTI\n.nii.gz', 1.5, 8.5, 'lightblue'),
    ('Preprocess\npreprocess_synapse_data.py', 5, 8.5, 'lightgreen'),
    ('Train Data\n(train_npz/)', 1.5, 6, 'lightyellow'),
    ('Test Data\n(test_vol_h5/)', 8.5, 6, 'lightyellow'),
    ('DataLoader\nSynapse_dataset', 1.5, 3.5, 'lightcoral'),
    ('Augmentation\nRandomGenerator', 5, 3.5, 'wheat'),
    ('Model Input\n(224x224)', 8.5, 3.5, 'lightblue'),
]

for text, x, y, color in steps:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax2.add_patch(box)
    ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
arrows = [
    (1.5, 8.1, 5, 8.9),
    (5, 8.1, 1.5, 6.4),
    (5, 8.1, 8.5, 6.4),
    (1.5, 5.6, 1.5, 3.9),
    (1.5, 3.1, 5, 3.9),
    (5, 3.1, 8.5, 3.9),
]

for x1, y1, x2, y2 in arrows:
    ax2.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.15, 
             fc='red', ec='red', linewidth=2)

# ========== 3. 网络架构图 ==========
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 8)
ax3.axis('off')
ax3.set_title('EMCAD Network Architecture', fontsize=16, fontweight='bold', pad=20)

# 输入
input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                           facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(input_box)
ax3.text(1.5, 7, 'Input\n1x224x224', ha='center', va='center', fontsize=10, fontweight='bold')

# Conv 1->3
conv_box = FancyBboxPatch((3.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
ax3.add_patch(conv_box)
ax3.text(4.5, 7, 'Conv 1->3\nChannel Convert', ha='center', va='center', fontsize=10, fontweight='bold')

# 编码器
encoder_box = FancyBboxPatch((6.5, 5), 3, 3, boxstyle="round,pad=0.1",
                            facecolor='lightyellow', edgecolor='black', linewidth=2)
ax3.add_patch(encoder_box)
ax3.text(8, 7.5, 'PVTv2-B2 Encoder', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(8, 6.8, 'x1: 64ch, 56x56', ha='center', va='center', fontsize=9)
ax3.text(8, 6.3, 'x2: 128ch, 28x28', ha='center', va='center', fontsize=9)
ax3.text(8, 5.8, 'x3: 320ch, 14x14', ha='center', va='center', fontsize=9)
ax3.text(8, 5.3, 'x4: 512ch, 7x7', ha='center', va='center', fontsize=9)

# 解码器
decoder_box = FancyBboxPatch((10.5, 1), 6, 5, boxstyle="round,pad=0.1",
                            facecolor='lightcoral', edgecolor='black', linewidth=2)
ax3.add_patch(decoder_box)
ax3.text(13.5, 5.5, 'EMCAD Decoder', ha='center', va='center', fontsize=12, fontweight='bold')

# 解码器内部
decoder_steps = [
    ('MSCAM4', 11.5, 4, 'wheat'),
    ('EUCB3', 13.5, 4, 'wheat'),
    ('LGAG3', 15.5, 4, 'wheat'),
    ('MSCAM3', 11.5, 2.5, 'wheat'),
    ('EUCB2', 13.5, 2.5, 'wheat'),
    ('LGAG2', 15.5, 2.5, 'wheat'),
    ('MSCAM2', 11.5, 1.5, 'wheat'),
    ('EUCB1', 13.5, 1.5, 'wheat'),
    ('LGAG1', 15.5, 1.5, 'wheat'),
    ('MSCAM1', 13.5, 1, 'wheat'),
]

for text, x, y, color in decoder_steps:
    box = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1)
    ax3.add_patch(box)
    ax3.text(x, y, text, ha='center', va='center', fontsize=7)

# 预测头
head_box = FancyBboxPatch((17.5, 1), 2, 5, boxstyle="round,pad=0.1",
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
ax3.add_patch(head_box)
ax3.text(18.5, 5.5, 'Prediction\nHeads', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(18.5, 4.5, 'p4, p3, p2, p1', ha='center', va='center', fontsize=9)
ax3.text(18.5, 3.5, 'Upsample to', ha='center', va='center', fontsize=9)
ax3.text(18.5, 3, '224x224', ha='center', va='center', fontsize=9)

# 输出
output_box = FancyBboxPatch((17.5, 0.2), 2, 0.6, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(output_box)
ax3.text(18.5, 0.5, 'Output\n9 classes', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
main_arrows = [
    (2.5, 7, 3.5, 7),
    (5.5, 7, 6.5, 6.5),
    (9.5, 6.5, 10.5, 3.5),
    (16.5, 3.5, 17.5, 4),
    (18.5, 1.6, 18.5, 0.8),
]

for x1, y1, x2, y2 in main_arrows:
    ax3.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.15,
             fc='red', ec='red', linewidth=2)

# ========== 4. 训练流程图 ==========
ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 20)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Training Flow', fontsize=14, fontweight='bold', pad=20)

# 训练步骤
train_steps = [
    ('Init Model\nLoad Pretrain', 2, 5, 'lightblue'),
    ('Load Data\nDataLoader', 5, 5, 'lightgreen'),
    ('Forward\nmodel(x)', 8, 5, 'lightyellow'),
    ('Loss\nCE + Dice', 11, 5, 'lightcoral'),
    ('Backward\nloss.backward()', 14, 5, 'wheat'),
    ('Update\noptimizer.step()', 17, 5, 'lightblue'),
    ('Validation\ninference()', 5, 2.5, 'lightgreen'),
    ('Save Model\nbest.pth', 8, 2.5, 'lightyellow'),
    ('Metrics\nDice, HD95', 11, 2.5, 'lightcoral'),
]

for text, x, y, color in train_steps:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 训练循环箭头
for i in range(5):
    x1 = 2 + i * 3
    x2 = 5 + i * 3
    ax4.arrow(x1, 5, x2-x1-0.2, 0, head_width=0.15, head_length=0.1,
             fc='red', ec='red', linewidth=2)

# epoch循环
ax4.arrow(17, 4.6, -12, 0, head_width=0.15, head_length=0.1,
         fc='blue', ec='blue', linewidth=2, linestyle='--')
ax4.text(11, 4.3, 'Each epoch loop', ha='center', va='center', fontsize=9, 
        color='blue', fontweight='bold')

# 验证箭头
ax4.arrow(8, 4.6, -3, -1.7, head_width=0.15, head_length=0.1,
         fc='green', ec='green', linewidth=2)
ax4.arrow(11, 2.9, -3, 0, head_width=0.15, head_length=0.1,
         fc='green', ec='green', linewidth=2)
ax4.arrow(8, 2.9, 3, 0, head_width=0.15, head_length=0.1,
         fc='green', ec='green', linewidth=2)
ax4.arrow(11, 2.9, 0, 2.1, head_width=0.15, head_length=0.1,
         fc='green', ec='green', linewidth=2)

# 保存图片
plt.savefig('project_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Visualization saved as: project_visualization.png")





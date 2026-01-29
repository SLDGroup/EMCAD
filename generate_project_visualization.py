"""
生成项目可视化图表
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# ========== 1. 项目结构图 ==========
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('项目文件结构', fontsize=14, fontweight='bold', pad=20)

# 根目录
root = FancyBboxPatch((1, 10), 8, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(root)
ax1.text(5, 10.75, 'SLDGroup_EMCAD/', ha='center', va='center', fontsize=12, fontweight='bold')

# 主要文件
files = [
    ('train_synapse.py', 2, 8.5, 'lightgreen'),
    ('test_synapse.py', 5, 8.5, 'lightgreen'),
    ('trainer.py', 8, 8.5, 'lightyellow'),
    ('lib/', 2, 7, 'lightcoral'),
    ('utils/', 5, 7, 'lightcoral'),
    ('data/Synapse/', 8, 7, 'lightcoral'),
]

for name, x, y, color in files:
    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(x, y, name, ha='center', va='center', fontsize=9)

# lib子目录
lib_files = ['networks.py', 'decoders.py', 'pvtv2.py']
for i, f in enumerate(lib_files):
    box = FancyBboxPatch((1.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(2.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# utils子目录
utils_files = ['dataset_synapse.py', 'utils.py', 'preprocess_synapse_data.py']
for i, f in enumerate(utils_files):
    box = FancyBboxPatch((4.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(5.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# data子目录
data_files = ['train_npz/', 'test_vol_h5/', 'lists/']
for i, f in enumerate(data_files):
    box = FancyBboxPatch((7.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         facecolor='wheat', edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(8.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# 箭头
for x in [2, 5, 8]:
    ax1.arrow(x, 7.3, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# ========== 2. 数据流程图 ==========
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('数据流程', fontsize=14, fontweight='bold', pad=20)

# 流程步骤
steps = [
    ('原始NIfTI\n(.nii.gz)', 1.5, 8.5, 'lightblue'),
    ('预处理\npreprocess_synapse_data.py', 5, 8.5, 'lightgreen'),
    ('训练数据\n(train_npz/)', 1.5, 6, 'lightyellow'),
    ('测试数据\n(test_vol_h5/)', 8.5, 6, 'lightyellow'),
    ('数据加载\nSynapse_dataset', 1.5, 3.5, 'lightcoral'),
    ('数据增强\nRandomGenerator', 5, 3.5, 'wheat'),
    ('模型输入\n(224×224)', 8.5, 3.5, 'lightblue'),
]

for text, x, y, color in steps:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax2.add_patch(box)
    ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
arrows = [
    (1.5, 8.1, 5, 8.9),  # 原始 -> 预处理
    (5, 8.1, 1.5, 6.4),  # 预处理 -> 训练数据
    (5, 8.1, 8.5, 6.4),  # 预处理 -> 测试数据
    (1.5, 5.6, 1.5, 3.9),  # 训练数据 -> 数据加载
    (1.5, 3.1, 5, 3.9),  # 数据加载 -> 数据增强
    (5, 3.1, 8.5, 3.9),  # 数据增强 -> 模型输入
]

for x1, y1, x2, y2 in arrows:
    ax2.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.15, 
             fc='red', ec='red', linewidth=2)

# ========== 3. 网络架构图 ==========
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 8)
ax3.axis('off')
ax3.set_title('EMCAD网络架构', fontsize=16, fontweight='bold', pad=20)

# 输入
input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                           facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(input_box)
ax3.text(1.5, 7, '输入图像\n1×224×224', ha='center', va='center', fontsize=10, fontweight='bold')

# Conv 1->3
conv_box = FancyBboxPatch((3.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
ax3.add_patch(conv_box)
ax3.text(4.5, 7, 'Conv 1→3\n通道转换', ha='center', va='center', fontsize=10, fontweight='bold')

# 编码器
encoder_box = FancyBboxPatch((6.5, 5), 3, 3, boxstyle="round,pad=0.1",
                            facecolor='lightyellow', edgecolor='black', linewidth=2)
ax3.add_patch(encoder_box)
ax3.text(8, 7.5, 'PVTv2-B2 编码器', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(8, 6.8, 'x1: 64ch, 56×56', ha='center', va='center', fontsize=9)
ax3.text(8, 6.3, 'x2: 128ch, 28×28', ha='center', va='center', fontsize=9)
ax3.text(8, 5.8, 'x3: 320ch, 14×14', ha='center', va='center', fontsize=9)
ax3.text(8, 5.3, 'x4: 512ch, 7×7', ha='center', va='center', fontsize=9)

# 解码器
decoder_box = FancyBboxPatch((10.5, 1), 6, 5, boxstyle="round,pad=0.1",
                            facecolor='lightcoral', edgecolor='black', linewidth=2)
ax3.add_patch(decoder_box)
ax3.text(13.5, 5.5, 'EMCAD 解码器', ha='center', va='center', fontsize=12, fontweight='bold')

# 解码器内部
decoder_steps = [
    ('MSCAM4\n(CAB+SAB+MSCB)', 11.5, 4, 'wheat'),
    ('EUCB3\n上采样', 13.5, 4, 'wheat'),
    ('LGAG3\n注意力门控', 15.5, 4, 'wheat'),
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
ax3.text(18.5, 5.5, '预测头\n(4个1×1 Conv)', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(18.5, 4.5, 'p4, p3, p2, p1', ha='center', va='center', fontsize=9)
ax3.text(18.5, 3.5, '上采样到', ha='center', va='center', fontsize=9)
ax3.text(18.5, 3, '224×224', ha='center', va='center', fontsize=9)

# 输出
output_box = FancyBboxPatch((17.5, 0.2), 2, 0.6, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(output_box)
ax3.text(18.5, 0.5, '输出\n9类分割', ha='center', va='center', fontsize=10, fontweight='bold')

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

# Skip connections
skip_arrows = [
    (8, 5.3, 11.5, 4.3),  # x4 -> MSCAM4
    (8, 5.8, 15.5, 4.3),  # x3 -> LGAG3
    (8, 6.3, 15.5, 2.8),  # x2 -> LGAG2
    (8, 6.8, 15.5, 1.8),  # x1 -> LGAG1
]

for x1, y1, x2, y2 in skip_arrows:
    ax3.plot([x1, x2], [y1, y2], 'b--', linewidth=1.5, alpha=0.6)
    ax3.arrow(x2-0.3, y2, 0.2, 0, head_width=0.1, head_length=0.1,
             fc='blue', ec='blue', linewidth=1)

# ========== 4. 训练流程图 ==========
ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 20)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('训练流程', fontsize=14, fontweight='bold', pad=20)

# 训练步骤
train_steps = [
    ('初始化模型\n加载预训练权重', 2, 5, 'lightblue'),
    ('加载数据\nDataLoader', 5, 5, 'lightgreen'),
    ('前向传播\nmodel(x)', 8, 5, 'lightyellow'),
    ('计算损失\nCE + Dice', 11, 5, 'lightcoral'),
    ('反向传播\nloss.backward()', 14, 5, 'wheat'),
    ('更新参数\noptimizer.step()', 17, 5, 'lightblue'),
    ('验证\ninference()', 5, 2.5, 'lightgreen'),
    ('保存模型\nbest.pth', 8, 2.5, 'lightyellow'),
    ('评估指标\nDice, HD95', 11, 2.5, 'lightcoral'),
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
ax4.text(11, 4.3, '每个epoch循环', ha='center', va='center', fontsize=9, 
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

# ========== 5. 数据格式说明 ==========
ax5 = fig.add_subplot(gs[3, 0])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('数据格式说明', fontsize=14, fontweight='bold', pad=20)

# 训练数据
train_data_box = FancyBboxPatch((0.5, 7), 4, 2.5, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
ax5.add_patch(train_data_box)
ax5.text(2.5, 8.5, '训练数据 (train_npz/)', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax5.text(2.5, 7.8, '格式: .npz文件', ha='center', va='center', fontsize=9)
ax5.text(2.5, 7.3, '命名: caseXXXX_sliceXXX.npz', ha='center', va='center', fontsize=9)
ax5.text(2.5, 6.8, '内容: {\'image\': 2D数组, \'label\': 2D数组}', ha='center', va='center', fontsize=9)
ax5.text(2.5, 6.3, '数量: ~2200+切片', ha='center', va='center', fontsize=9)

# 测试数据
test_data_box = FancyBboxPatch((5.5, 7), 4, 2.5, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
ax5.add_patch(test_data_box)
ax5.text(7.5, 8.5, '测试数据 (test_vol_h5/)', ha='center', va='center',
        fontsize=11, fontweight='bold')
ax5.text(7.5, 7.8, '格式: .h5文件', ha='center', va='center', fontsize=9)
ax5.text(7.5, 7.3, '命名: caseXXXX.npy.h5', ha='center', va='center', fontsize=9)
ax5.text(7.5, 6.8, '内容: {\'image\': 3D数组, \'label\': 3D数组}', ha='center', va='center', fontsize=9)
ax5.text(7.5, 6.3, '数量: 12个体积', ha='center', va='center', fontsize=9)

# 类别信息
class_box = FancyBboxPatch((0.5, 3.5), 9, 2.5, boxstyle="round,pad=0.1",
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
ax5.add_patch(class_box)
ax5.text(5, 5.5, 'Synapse数据集 - 9类器官分割', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax5.text(5, 4.8, '0: 背景 | 1: 脾脏 | 2: 右肾 | 3: 左肾 | 4: 胆囊', ha='center', va='center', fontsize=9)
ax5.text(5, 4.3, '5: 胰腺 | 6: 肝脏 | 7: 胃 | 8: 主动脉', ha='center', va='center', fontsize=9)

# ========== 6. 关键模块说明 ==========
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('关键模块说明', fontsize=14, fontweight='bold', pad=20)

modules = [
    ('MSCB\n多尺度卷积块', 2, 8.5, 'lightblue'),
    ('EUCB\n高效上采样', 5, 8.5, 'lightgreen'),
    ('LGAG\n大核注意力门控', 8, 8.5, 'lightyellow'),
    ('CAB\n通道注意力', 2, 6, 'lightcoral'),
    ('SAB\n空间注意力', 5, 6, 'wheat'),
    ('MSCAM\n多尺度注意力模块', 8, 6, 'lightblue'),
    ('Mutation\nSupervision', 2, 3.5, 'lightgreen'),
    ('Dice Loss\n+ CE Loss', 5, 3.5, 'lightyellow'),
    ('评估指标\nDice, HD95', 8, 3.5, 'lightcoral'),
]

for text, x, y, color in modules:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax6.add_patch(box)
    ax6.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 保存图片
plt.savefig('项目可视化图表.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 可视化图表已保存为: 项目可视化图表.png")

# 也保存一个简化版本
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('EMCAD项目核心流程图', fontsize=16, fontweight='bold', y=0.98)

# 简化版：执行流程图
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('训练执行流程', fontsize=12, fontweight='bold')

flow = [
    ('python train_synapse.py', 2, 5),
    ('加载数据', 5, 5),
    ('训练模型', 8, 5),
    ('保存权重', 5, 2.5),
]

for text, x, y in flow:
    box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

for i in range(len(flow)-1):
    if i < 2:
        ax.arrow(flow[i][1]+1, flow[i][2], flow[i+1][1]-flow[i][1]-0.2, 0,
                head_width=0.2, head_length=0.15, fc='red', ec='red', linewidth=2)
    else:
        ax.arrow(flow[i][1], flow[i][2]-0.5, -3, -2, head_width=0.2, head_length=0.15,
                fc='red', ec='red', linewidth=2)

# 简化版：测试流程
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('测试执行流程', fontsize=12, fontweight='bold')

test_flow = [
    ('python test_synapse.py', 2, 5),
    ('加载模型', 5, 5),
    ('逐切片推理', 8, 5),
    ('计算指标', 5, 2.5),
    ('保存结果', 8, 2.5),
]

for text, x, y in test_flow:
    box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 简化版：数据流向
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('数据流向', fontsize=12, fontweight='bold')

data_flow = [
    ('原始数据\n.nii.gz', 1.5, 4.5),
    ('预处理', 4.5, 4.5),
    ('训练\n.npz', 1.5, 2),
    ('测试\n.h5', 7.5, 2),
    ('模型', 4.5, 2),
]

for text, x, y in data_flow:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                         facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 简化版：网络输入输出
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('网络输入输出', fontsize=12, fontweight='bold')

io_flow = [
    ('输入\n1×224×224', 2, 4),
    ('编码器\nPVTv2-B2', 5, 4),
    ('解码器\nEMCAD', 8, 4),
    ('输出\n9×224×224', 5, 1.5),
]

for text, x, y in io_flow:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                         facecolor='lightcoral', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

for i in range(len(io_flow)-1):
    if i < 2:
        ax.arrow(io_flow[i][1]+0.8, io_flow[i][2], io_flow[i+1][1]-io_flow[i][1]-0.2, 0,
                head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    else:
        ax.arrow(io_flow[i][1], io_flow[i][2]-0.4, -3, -2.1,
                head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)

plt.tight_layout()
plt.savefig('项目核心流程图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 核心流程图已保存为: 项目核心流程图.png")





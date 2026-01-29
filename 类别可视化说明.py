"""
生成Synapse数据集9个类别的可视化说明图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ========== 1. 9个类别列表 ==========
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Synapse数据集 - 9个类别', fontsize=14, fontweight='bold', pad=20)

# 类别信息
classes_info = [
    (0, '背景', 'Background', 'lightgray'),
    (1, '脾脏', 'Spleen', '#FF6B6B'),
    (2, '右肾', 'Right Kidney', '#4ECDC4'),
    (3, '左肾', 'Left Kidney', '#45B7D1'),
    (4, '胆囊', 'Gallbladder', '#FFA07A'),
    (5, '胰腺', 'Pancreas', '#98D8C8'),
    (6, '肝脏', 'Liver', '#F7DC6F'),
    (7, '胃', 'Stomach', '#BB8FCE'),
    (8, '主动脉', 'Aorta', '#85C1E2'),
]

y_start = 9
for i, (cid, cname_cn, cname_en, color) in enumerate(classes_info):
    y_pos = y_start - i * 1.0
    
    # 类别框
    box = FancyBboxPatch((0.5, y_pos-0.4), 9, 0.8, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.add_patch(box)
    
    # 类别ID
    ax1.text(1.2, y_pos, f'类别 {cid}', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    
    # 中文名称
    ax1.text(3.5, y_pos, cname_cn, ha='left', va='center', 
            fontsize=11, fontweight='bold')
    
    # 英文名称
    ax1.text(6, y_pos, cname_en, ha='left', va='center', 
            fontsize=10, style='italic')

# ========== 2. 14类到9类的转换 ==========
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('14类 → 9类 转换规则', fontsize=14, fontweight='bold', pad=20)

# 原始14类（左侧）
ax2.text(2, 9, '原始14类', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='blue')
original_classes = [
    '0: 背景', '1: 脾脏', '2: 右肾', '3: 左肾', '4: 胆囊',
    '5: 食管 ✗', '6: 肝脏', '7: 胃', '8: 主动脉',
    '9: 下腔静脉 ✗', '10: 门静脉 ✗', '11: 胰腺',
    '12: 右肾上腺 ✗', '13: 左肾上腺 ✗'
]

for i, cls in enumerate(original_classes):
    y_pos = 8 - i * 0.5
    color = 'lightcoral' if '✗' in cls else 'lightgreen'
    box = FancyBboxPatch((0.5, y_pos-0.2), 3.5, 0.4, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1, alpha=0.6)
    ax2.add_patch(box)
    ax2.text(2.25, y_pos, cls, ha='center', va='center', fontsize=8)

# 箭头
ax2.arrow(4.5, 4, 1, 0, head_width=0.3, head_length=0.2,
         fc='red', ec='red', linewidth=3)

# 转换后9类（右侧）
ax2.text(7.5, 9, '转换后9类', ha='center', va='center',
        fontsize=12, fontweight='bold', color='green')
converted_classes = [
    '0: 背景', '1: 脾脏', '2: 右肾', '3: 左肾', 
    '4: 胆囊', '5: 胰腺', '6: 肝脏', '7: 胃', '8: 主动脉'
]

for i, cls in enumerate(converted_classes):
    y_pos = 8 - i * 0.5
    box = FancyBboxPatch((6, y_pos-0.2), 3, 0.4, boxstyle="round,pad=0.05",
                         facecolor='lightgreen', edgecolor='black', linewidth=1, alpha=0.6)
    ax2.add_patch(box)
    ax2.text(7.5, y_pos, cls, ha='center', va='center', fontsize=8)

# 转换说明
ax2.text(5, 1.5, '转换规则:\n• 食管、下腔静脉、门静脉、\n  肾上腺 → 背景(0)\n• 胰腺: 11 → 5', 
        ha='center', va='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ========== 3. 在项目中的使用 ==========
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('9个类别在项目中的作用', fontsize=14, fontweight='bold', pad=20)

# 使用场景
usage_scenarios = [
    ('网络输出', 1.5, 8.5, 'num_classes=9\n输出: (B, 9, 224, 224)', 'lightblue'),
    ('损失函数', 5.5, 8.5, 'CE Loss + Dice Loss\n9类分类', 'lightgreen'),
    ('评估指标', 8.5, 8.5, '每个器官的\nDice, HD95', 'lightyellow'),
    ('数据转换', 1.5, 5.5, '14类 → 9类\n标签映射', 'lightcoral'),
    ('可视化', 5.5, 5.5, '不同颜色\n标注器官', 'wheat'),
    ('保存结果', 8.5, 5.5, 'NIfTI文件\n像素值0-8', 'lightblue'),
]

for text, x, y, desc, color in usage_scenarios:
    # 主框
    box = FancyBboxPatch((x-0.7, y-0.8), 1.4, 1.6, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax3.add_patch(box)
    ax3.text(x, y+0.3, text, ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax3.text(x, y-0.3, desc, ha='center', va='center', 
            fontsize=8)

# ========== 4. 腹部CT示意图 ==========
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('腹部CT横断面 - 器官位置示意', fontsize=14, fontweight='bold', pad=20)

# 绘制简化的腹部CT示意图
# 背景（身体轮廓）
body = plt.Circle((5, 5), 3.5, fill=False, edgecolor='black', linewidth=2)
ax4.add_patch(body)

# 器官位置（简化示意）
organs_pos = [
    (2.5, 6, '胃', '#BB8FCE', 0.4),
    (7, 5.5, '肝脏', '#F7DC6F', 0.8),
    (2, 4, '脾脏', '#FF6B6B', 0.5),
    (7.5, 4.5, '胆囊', '#FFA07A', 0.3),
    (2.5, 3, '左肾', '#45B7D1', 0.5),
    (7.5, 3, '右肾', '#4ECDC4', 0.5),
    (5, 4, '胰腺', '#98D8C8', 0.4),
    (5, 2.5, '主动脉', '#85C1E2', 0.3),
]

for x, y, name, color, size in organs_pos:
    # 器官圆形
    organ = plt.Circle((x, y), size, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.add_patch(organ)
    # 器官名称
    ax4.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

# 添加说明
ax4.text(5, 0.5, '注: 这是简化的位置示意图\n实际CT中器官位置和大小会变化', 
        ha='center', va='center', fontsize=8, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 保存图片
plt.savefig('Synapse_9类类别说明图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 类别说明图已保存为: Synapse_9类类别说明图.png")

# 创建第二个图：代码使用示例
fig2, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('9个类别在代码中的使用示例', fontsize=16, fontweight='bold', pad=20)

# 代码示例框
code_examples = [
    ('模型定义', 2, 8.5, 
     'model = EMCADNet(\n    num_classes=9,  # 9个类别\n    ...\n)', 
     'lightblue'),
    ('数据加载', 2, 5.5,
     'if self.nclass == 9:\n    label[label==5] = 0   # 食管→背景\n    label[label==11] = 5  # 胰腺重编号',
     'lightgreen'),
    ('损失计算', 6, 8.5,
     'ce_loss = CrossEntropyLoss()\ndice_loss = DiceLoss(num_classes=9)\nloss = 0.3*ce + 0.7*dice',
     'lightyellow'),
    ('评估指标', 6, 5.5,
     'for i in range(1, 9):  # 1-8器官\n    dice = calculate_dice(\n        pred==i, gt==i\n    )',
     'lightcoral'),
]

for title, x, y, code, color in code_examples:
    # 标题
    ax.text(x, y+1.2, title, ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # 代码框
    code_box = FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(code_box)
    
    # 代码文本
    ax.text(x, y, code, ha='center', va='center',
           fontsize=9, family='monospace')

plt.savefig('Synapse_代码使用示例.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 代码使用示例图已保存为: Synapse_代码使用示例.png")




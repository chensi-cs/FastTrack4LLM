import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = [ "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def lr_lambda(epoch):
    lr= d_model ** -0.5 * min(epoch ** -0.5, epoch * warmup_steps ** -1.5)
    return lr


d_model=512
warmup_steps=4000
peak_step = warmup_steps
peak_lr = lr_lambda(peak_step)
total_steps = 100000
steps = np.arange(1, total_steps + 1)
lrs = [lr_lambda(step) for step in steps]

# 绘制学习率曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, lrs, label='学习率', color='blue', linewidth=2.5)

# 标记峰值点
plt.scatter([peak_step], [peak_lr], color='red', s=50, zorder=5)
plt.annotate(f'峰值: {peak_lr:.6f}', 
             xy=(peak_step, peak_lr), 
             xytext=(peak_step + 500, peak_lr + 0.0001),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12)

# 标记预热阶段和衰减阶段
plt.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5)
plt.text(warmup_steps/2, max(lrs)*0.9, '预热阶段\n(线性增长)', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.text((warmup_steps + total_steps)/2, max(lrs)*0.9, '衰减阶段\n(平方根衰减)', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 设置图表属性
plt.title('Transformer学习率调度曲线', fontsize=16, pad=15)
plt.xlabel('训练步数', fontsize=14, labelpad=10)
plt.ylabel('学习率', fontsize=14, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 显示图表
plt.show()
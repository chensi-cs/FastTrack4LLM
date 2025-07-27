#!/bin/bash

# 获取脚本所在目录的绝对路径，定义为根目录
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 验证根目录（用于调试）
echo "项目根目录: $ROOT_DIR"

# 1. 删除旧目录（强制删除，避免提示）
echo "清理旧目录..."
rm -rf "$ROOT_DIR/logs"        # 强制删除日志目录
rm -rf "$ROOT_DIR/checkpoints" # 强制删除检查点目录
rm -rf "$ROOT_DIR/saved_models" # 强制删除模型保存目录

# 2. 重新创建目录（确保权限正确）
echo "创建新目录..."
mkdir -p "$ROOT_DIR/logs"        # -p 确保父目录存在，无报错
mkdir -p "$ROOT_DIR/checkpoints"
mkdir -p "$ROOT_DIR/saved_models"

# 3. 确保目录权限（当前用户可读写）
chmod -R 755 "$ROOT_DIR/logs"        # 递归设置权限
chmod -R 755 "$ROOT_DIR/checkpoints"
chmod -R 755 "$ROOT_DIR/saved_models"

# 4. 启动TensorBoard（指定绝对路径，避免歧义）
echo "启动TensorBoard..."
tensorboard --logdir="$ROOT_DIR/logs" &  # 使用绝对路径

# 5. 运行训练脚本（确保在根目录执行）
echo "开始训练..."
python -m trainner.train_pretrain

# 6. 训练结束后提示
echo "训练结束，TensorBoard日志路径: $ROOT_DIR/logs"
#!/bin/bash
# 启动TensorBoard并在后台运行
tensorboard --logdir=runs &

# 打开浏览器访问TensorBoard (适用于Linux/macOS)
echo "TensorBoard已启动，访问 http://localhost:6006"
echo "按Ctrl+C停止训练脚本，但TensorBoard会继续在后台运行"
echo "若要停止TensorBoard，请运行: pkill tensorboard"

# 运行训练脚本
python -m trainner.train_pretrain
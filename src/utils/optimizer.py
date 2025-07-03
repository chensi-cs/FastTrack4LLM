"""
自己设计各种优化器
"""
import torch
import torch.nn as nn

def bgd(model,train_data,labels,lr=0.01,num_epochs=100,criterion):
    """
    批量梯度下降
    Args:
        model: 模型
        train_data: 全部数据
        labels: 标签
        lr: 学习率
        num_epochs: 迭代次数
        criterion: 损失函数
    """
    for epoch in range(num_epochs):
        # 整体预测
        prediction = model(train_data)
        loss = criterion(prediction,labels)
        # 反向传播
        loss.backward()
        # 更新参数
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
                param.grad.zero_()

def sgd(model,data_loader,lr=0.01,num_epochs=100,criterion):
    """
    随机(小批量)梯度下降
    Args:
        model: 模型
        data_loader: 数据加载器
        lr: 学习率
        num_epochs: 迭代次数
        criterion: 损失函数
    """
    for epoch in range(num_epochs):
        for idx,batch_data in enumerate(data_loader):
            # 随机梯度下降batch_data的size是1
            # 小批量梯度下降batch_data的size是设定的batch_size
            inputs = batch_data['input']
            labels = batch_data['label']
            prediction = model(inputs)
            loss = criterion(prediction,labels)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr*param.grad
                    param.grad.zero_()

def momentum(model,data_loader,lr=0.01,num_epochs=100,beta=0.9,criterion):
    """
    动量梯度下降
    Args:
        model: 模型
        data_loader: 数据加载器
        lr: 学习率
        num_epochs: 迭代次数
        beta: 动量系数
        criterion: 损失函数
    """
    for epoch in range(num_epochs):
        v = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for idx,batch_data in enumerate(data_loader):
            inputs = batch_data['input']
            labels = batch_data['label']
            prediction = model(inputs)
            loss = criterion(prediction,labels)
            loss.backward()
            with torch.no_grad():
                for name,param in  model.named_parameters():
                    v[name] = beta * v[name] + (1-beta) * param.grad
                    param = param - lr * v[name]
                    param.grad.zero_()

def 

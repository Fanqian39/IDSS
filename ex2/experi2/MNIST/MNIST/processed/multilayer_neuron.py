import torch
import torch.nn as nn

# 输入
x = torch.randn(1, 4)
print('\n来自其它神经元的输入形式:', x.shape)
print('\n来自其它神经元的输入:', x)

# 多层全连接网络模型
class MultiLayerModel(nn.Module):
    def __init__(self, input_features, hidden_features1, hidden_features2, output_features):
        super(MultiLayerModel, self).__init__()
        # 第一层神经元
        self.neuron1 = nn.Linear(input_features, hidden_features1)
        # 第一层激活函数
        self.activation1 = nn.ReLU()
        # 第二层神经元
        self.neuron2 = nn.Linear(hidden_features1, hidden_features2)
        # 第二层激活函数
        self.activation2 = nn.ReLU()
        # 输出层神经元
        self.output_neuron = nn.Linear(hidden_features2, output_features)
    
    def forward(self, x):
        x = self.neuron1(x)
        x = self.activation1(x)
        x = self.neuron2(x)
        x = self.activation2(x)
        x = self.output_neuron(x)
        return x

# 实例化多层全连接网络模型
input_features = 4
hidden_features1 = 8  # 第一隐藏层的神经元数量
hidden_features2 = 4  # 第二隐藏层的神经元数量
output_features = 1   # 输出层的神经元数量
net = MultiLayerModel(input_features, hidden_features1, hidden_features2, output_features)

print('\n第一层神经元输入的加权权重形式:', net.neuron1.weight.shape)
print('\n第一层神经元输入的加权权重:', net.neuron1.weight.data)
print('\n第二层神经元输入的加权权重形式:', net.neuron2.weight.shape)
print('\n第二层神经元输入的加权权重:', net.neuron2.weight.data)
print('\n输出层神经元输入的加权权重形式:', net.output_neuron.weight.shape)
print('\n输出层神经元输入的加权权重:', net.output_neuron.weight.data)

# 输出
y = net(x)
print('\n输出形式:', y.shape)
print('\n输出值:', y)

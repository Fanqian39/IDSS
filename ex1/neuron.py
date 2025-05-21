import torch
import torch.nn as nn

# 输入
x = torch.randn(3,4)
print('\n来自其它神经元的输入形式:', x.shape)
print('\n来自其它神经元的输入:', x)

# 神经元模型
class model(nn.Module):
    def __init__(self, class_=4):
        super(model, self).__init__()
        # 神经元
        self.neuron = nn.Linear(class_, 1)
		# 激活函数
        self.activation = nn.Sigmoid()
	
    def forward(self, x):
        neuron = self.neuron(x)
        output = self.activation(neuron)
        return output

# No need to change
net = model()
print('\n神经元输入的加权权重形式:', net.neuron.weight.shape)
print('\n神经元输入的加权权重:', net.neuron.weight.data)

# 输出
y = net(x)
print('\n输出形式:', y.shape)
print('\n输出值:', y)
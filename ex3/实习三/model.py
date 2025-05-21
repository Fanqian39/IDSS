import torch
import torch.nn as nn


class net1(nn.Module):
    def __init__(self, class_=2):
        super(net1, self).__init__()
        # 多个全连接层
        # 1
        in_features = 32 * 32 * 3
        hidden_features = 128
        self.layer_1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=False)
        self.activation = nn.ReLU()
        # 2
        self.layer_2 = nn.Linear(in_features=hidden_features, out_features=class_, bias=False)

    def forward(self, x):
        print(x.shape)
        x = x.view(x.size(0),-1)
        layer1 = self.activation(self.layer_1(x))
        output = self.layer_2(layer1)
        return output


# class net2(nn.Module):
#     def __init__(self, class_=):
#         super(net2, self).__init__()
#         # 更多的全连接层
#         # 1
#         self.layer_1 = nn.Linear(in_features=, out_features=, bias=False)
#         self.activation = nn.ReLU()
#         # 3
#         self.layer_2 = nn.Linear(in_features=, out_features=class_, bias=False)
#         # ···
#
#     def forward(self, x):
#         x = x.view(x.size(0),-1)
#         layer1 = self.activation(self.layer_1(x))
#         layer2 = self.activation(self.layer_1(layer1))
#         # ···
#         return output

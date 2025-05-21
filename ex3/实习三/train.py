import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import numpy
from utils import progress_bar
from model import *

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--max_epoch', default=300, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
gpu = "0"           # Choice:0 or 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss = 1000
best_acc = 0
start_epoch = 0
flod = args.flod
work = True
weight_decay = 0.000


# 定义自己想分类的类别名称与个数
NAME = ['麻雀', '灰嘴喜鹊']
#当前程序运行使用机器学习（Train），还是测试（Test）
CHOICE = 'Train'



# Data
def data_prepare():
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 强制统一为32x32
        transforms.ToTensor()
    ])
    train_dataset0 = ImageFolder(root='D:/IDSS/ex3/实习三/data/data_1', transform=data_transform)
    train_loader0 = DataLoader(train_dataset0, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_dataset1 = ImageFolder(root='D:/IDSS/ex3/实习三/data/data_2', transform=data_transform)
    train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return train_loader0, train_loader1



# Model
def model_prepare(work):
    print('==> Building model..')
    global best_acc
    global start_epoch
    if work == True:
        net = net1()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    return net, optimizer, scheduler, criterion



def train(epoch, dataloader0, dataloader1, net, optimizer, criterion, vali=True):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    print('Train')
    global tr_loss
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_id = 0
    for batch_id0, (inputs0, targets0) in enumerate(dataloader0):
        for batch_id1, (inputs1, targets1) in enumerate(dataloader1):
            if batch_id1 == batch_id0:
                if inputs0.size(0) != inputs1.size(0):
                    print(f"Warning: Batch sizes do not match: {inputs0.size(0)} vs {inputs1.size(0)}")
                    continue  # Skip this batch
                batch_id = batch_id0 + batch_id1
                optimizer.zero_grad()

                # 制作标签
                targets0 = torch.zeros(inputs0.size(0), dtype=torch.long)
                targets1 = torch.ones(inputs1.size(0), dtype=torch.long)
                inputs = torch.cat((inputs0, inputs1), dim=0)
                targets = torch.cat((targets0, targets1), dim=0)
                inputs, targets = inputs.to(device), targets.to(device)

                # 图片输入到网络
                outputs = net(inputs)
                # 计算网络输出与图片实际标签间的误差
                loss = criterion(outputs, targets.long())
                # 基于该误差进行网络权重的求导
                loss.backward()
                # 基于求导结果更新网络权重
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(batch_id, 'Loss: %.3f | Acc: %.3f (%d/%d)' % (
                    train_loss / (batch_id + 1), 100. * correct / total, correct, total))
            else:
                pass
    if vali is True:
        tr_loss = train_loss / (batch_id + 1)
    return train_loss / (batch_id + 1), 100. * correct / total




def predict_model():
    print('Test')
    image_path = "./image.jpg"
    image = Image.open(image_path)
    data_transform = transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor()
    ])
    image = data_transform(image)
    image = image.view(-1, 3, 32, 32)
    image = image.to(device)

    net = net1()
    net = net.to(device)
    net.load_state_dict(torch.load('./params/flod.t7', map_location='cpu'), strict=False)
    output = net(image)
    softmax = nn.Softmax(dim=1)
    out = softmax(output)
    print(out)
    pred = out.max(1, keepdim=True)[1]
    return pred.item()



if __name__ == '__main__':
    if CHOICE == 'Train':
        train_loader0, train_loader1 = data_prepare()
        net, optimizer, scheduler, criterion = model_prepare(work)
        train_list0, train_list1 = [], []
        for epoch in range(start_epoch, start_epoch+args.max_epoch):
            train_loss, train_acc = train(epoch, train_loader0, train_loader1, net, optimizer, criterion)
            scheduler.step(tr_loss)
            lr = optimizer.param_groups[0]['lr']
            train_list0.append(train_loss)
            train_list1.append(train_acc)
            if lr < 1e-6 or epoch == args.max_epoch - 1:
                print('Saving:')
                train_array0 = numpy.array(train_list0)
                train_array1 = numpy.array(train_list1)
                plt.figure(1)
                plt.subplot(1, 2, 1)
                plt.xlabel('epoch')
                plt.ylabel('train loss')
                plt.plot([i for i in range(epoch+1)], train_array0, '-')
                plt.subplot(1, 2, 2)
                plt.xlabel('epoch')
                plt.ylabel('train acc')
                plt.plot([i for i in range(epoch+1)], train_array1, '-')
                plt.savefig("network.jpg")
                plt.show()                
                print('Saving:')
                state = {
                    'net': net.state_dict(),
                    'acc': train_acc,
                }
                if not os.path.isdir('./params'):
                    os.makedirs('./params')
                torch.save(state, './params/flod''.t7')
                break
            else:
                pass


    elif CHOICE == 'Test':
        label = predict_model()
        print(NAME[label])







    
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
import numpy
from utils import progress_bar
from model import *

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--max_epoch', default=10, type=int, help='max epoch')
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


# Data
def data_prepare():
    transf = [transforms.ToTensor()]
    transform_train = transforms.Compose(transf)
    trainset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./MNIST', train=False, download=False, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# Model
def model_prepare(work):
    print('==> Building model..')
    global best_acc
    global start_epoch
    if work == True:
        net = net1()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    return net, optimizer, scheduler, criterion


def train(epoch, dataloader, net, optimizer, criterion, vali=True):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    global tr_loss
    net.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        num_id += 1
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)

        # 图片输入到网络，在下面补充（）中内容
        outputs = net(inputs)
        # 计算网络输出与图片实际标签间的误差，在下面补充 .long()的 . 前内容
        loss = criterion(outputs, targets.long())
        # 基于该误差进行网络权重的求导，在下面补充误差求导函数
        loss.backward()
        # 基于求导结果更新网络权重，在下面补充优化器对权重的更新函数
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                     % (train_loss / (batch_id + 1), 100. * correct / total, correct, total))
    if vali is True:
        tr_loss = train_loss / num_id
    return train_loss / num_id, 100. * correct / total


def test(epoch, dataloader, net, criterion):
    """Validation and the test."""
    global best_acc
    net.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            num_id += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_id + 1), 100. * correct / total, correct, total))
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    trainloader, testloader = data_prepare()
    net, optimizer, scheduler, criterion = model_prepare(work)
    train_list0, train_list1 = [], []
    test_list0, test_list1 = [], []
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        train_loss, train_acc = train(epoch, trainloader, net, optimizer, criterion)
        test_loss, test_acc = test(epoch, testloader, net, criterion)
        scheduler.step(tr_loss)
        lr = optimizer.param_groups[0]['lr']
        train_list0.append(train_loss)
        train_list1.append(train_acc)
        test_list0.append(test_loss)
        test_list1.append(test_acc)
        if lr < 5e-3 or epoch == args.max_epoch - 1:
            print('Saving:')
            train_array0 = numpy.array(train_list0)
            train_array1 = numpy.array(train_list1)
            test_array0 = numpy.array(test_list0)
            test_array1 = numpy.array(test_list1)
            plt.figure(1)
            plt.subplot(2, 2, 1)
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.plot([i for i in range(epoch+1)], train_array0, '-')
            plt.subplot(2, 2, 2)
            plt.xlabel('epoch')
            plt.ylabel('train acc')
            plt.plot([i for i in range(epoch+1)], train_array1, '-')
            plt.subplot(2, 2, 3)
            plt.xlabel('epoch')
            plt.ylabel('test loss')
            plt.plot([i for i in range(epoch+1)], test_array0, '-')
            plt.subplot(2, 2, 4)
            plt.xlabel('epoch')
            plt.ylabel('test acc')
            plt.plot([i for i in range(epoch+1)], test_array1, '-')
            plt.savefig("network.jpg")
            plt.show()
            print('OVER')
            break
        else:
            pass
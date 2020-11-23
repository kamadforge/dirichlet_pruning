# it contains the module for computing the the accuracy of the network when we remove combinations of filters.
#

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys

print(sys.path)
print("newh2")
sys.path.append(
    "/home/kamil/Dropbox/Current_research/python_tests/results_networktest/external_codes/pytorch-cifar-master/models")
sys.path.append("/home/kamil/Dropbox/Current_research/python_tests/results_compression")
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as f
import logging
import matplotlib.pyplot as plt

# import magnitude_pruning


# file_dir = os.path.dirname("utlis.p")
# sys.path.append(file_dir)

# from models import *

# from utils import progress_bar

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

#####################################
# DATA

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

########## rainval

trainval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainval_perc = 0.85
train_size = int(trainval_perc * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
torch.manual_seed(0)
print(torch.rand(2))
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
# val_dataset=torch.load("val_dataset")

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
# for batch_idx, (inputs, targets) in enumerate(val_loader):
#    inputs, targets = inputs.to(device), targets.to(device)

################## test

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# with more workers there may be an error in debug mode: RuntimeError: DataLoader worker (pid 29274) is killed by signal: Terminated.


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#
criterion = nn.CrossEntropyLoss()


def test(epoch, net):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


def test_val(epoch, net):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(targets)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # if (predicted.eq(targets).sum().item())!=128:
            #     print(predicted.eq(targets))
            #     print(predicted)
            #     print(targets)
            # else:
            #     print(predicted)
            # print("----------------------------------------------")

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


###########################################################
# copied from network pruning important but commented for clarity

# def compute_combinations_random(file_write):
#     for name, param in net.named_parameters():
#         print(name)
#         print(param.shape)
#         layer = "module.features.1.weight"
#         if layer in name:
#             layerbias = layer[:17] + ".bias"
#             params_bias = net.state_dict()[layerbias]
#             while (True):
#
#                 all_results = {}
#                 # s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
#                 # for r in range(1,50): #produces the combinations of the elements in s
#                 #    results=[]
#                 randperm = np.random.permutation(param.shape[0])
#                 randint = 0
#                 while (randint == 0):
#                     randint = np.random.randint(param.shape[0])
#                 randint_indextoremove = np.random.randint(randint)
#                 combination = randperm[:randint]
#                 combination2 = np.delete(combination, randint_indextoremove)
#                 print(combination[randint_indextoremove])
#
#                 #if file_write:
#                 print("in")
#                 with open("results_running/combinations_pruning_cifar_vgg_%s.txt" % (layer),
#                           "a+") as textfile:
#                     textfile.write("%d\n" % randint_indextoremove)
#
#                 for combination in [combination, combination2]:
#                     # for combination in list(combinations(s, r)):
#
#                     combination = torch.LongTensor(combination)
#
#                     print(combination)
#                     params_saved = param[combination].clone()
#                     param_bias_saved = params_bias[combination].clone()
#
#                     # param[torch.LongTensor([1, 4])] = 0
#                     # workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
#                     if len(combination) != 0:
#                         param[combination[0]] = 0
#                         # param[combination]=0
#                         params_bias[combination] = 0
#
#                     accuracy = test_val(-1)
#                     param[combination] = params_saved
#                     params_bias[combination] = param_bias_saved
#
#                     #if file_write:
#                     print("out")
#                     with open("results_running/combinations_pruning_cifar_vgg_%s.txt" % (layer),
#                               "a+") as textfile:
#                         textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))


##########################

from itertools import chain, combinations


def compute_combinations(file_write, net, testval, channel="c13"):
    for name, param in net.named_parameters():
        print(name)
        print(param.shape)

        channel_name = "module." + channel

        if (channel_name in name and "weight" in name):
            layer = name
            print(layer)

            # if layer in name:
            if ("module.c1." not in name) and ("module.c1" in name):
                layerbias = layer[:11] + "bias"
            else:
                layerbias = layer[:10] + "bias"

            params_bias = net.state_dict()[layerbias]

            all_results = {}
            s = torch.range(0, param.shape[0] - 1)  # list from 0 to 19 as these are the indices of the data tensor
            # for r in range(1,param.shape[0]): #produces the combinations of the elements in s
            for r in range(1, 4):  # produces the combinations of the elements in s
                results = []
                for combination in list(combinations(s, r)):
                    combination = torch.LongTensor(combination)

                    print(combination)
                    params_saved = param[combination].clone();
                    param_bias_saved = params_bias[combination].clone()
                    param[combination[0]] = 0
                    param[combination] = 0;
                    params_bias[combination] = 0
                    accuracy = testval()
                    # accuracy = test(-1, net)
                    param[combination] = params_saved;
                    params_bias[combination] = param_bias_saved

                    results.append((combination, accuracy))

                    if file_write:
                        with open("combinations/combinations_pruning_cifar_vgg16_%s.txt" % (layer), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))
                        print("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))
                        logging.info("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r] = results
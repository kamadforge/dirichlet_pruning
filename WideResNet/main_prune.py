from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import numpy as np

from networks import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_false', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--prune', default=True)
parser.add_argument('--arch', default="75,85,80,80,159,159,154,159,315,315,314,316")
args = parser.parse_args()

thresh = [int(n) for n in args.arch.split(",")]
print(thresh)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    params_num_all=0
    for name, param in net.named_parameters():
        print (name, param.shape)
        if "weight" in name:
            param_num=1
            for i in range(len(param.shape)):
                param_num*=param.shape[i]
            params_num_all+=param_num
    print(params_num_all) #exactly 36.5



    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7', map_location=lambda storage, loc: storage)
    net = checkpoint['net']

    #for name, param in net:
    #    print (name,param.shape)


    best_acc = 0#checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if args.prune:

        ranks=np.load("ranks.npy", allow_pickle=True)


        reverse=True

        unimportant_channels={}
        prune_rates={"layer1.0" : thresh[0], "layer1.1" : thresh[1], "layer1.2" : thresh[2], "layer1.3" : thresh[3], "layer2.0" : thresh[4], "layer2.1" : thresh[5], "layer2.2" : thresh[6], "layer2.3" : thresh[7], "layer3.0" : thresh[8], "layer3.1" : thresh[9], "layer3.2" : thresh[10], "layer3.3" : thresh[11] }
        for r in ranks[()].keys():
            layer=r[7:15]
            rank=ranks[()][r]
            if reverse==True:
                # create inverted indices
                # idx = [i for i in range(rank.size(0) - 1, -1, -1)]
                # idx = torch.LongTensor(idx)
                # rank_rev = rank.index_select(0, idx)
                # rank=rank_rev

                rNpArr = np.flip(rank.detach().cpu().numpy(), 0).copy()  # Reverse of copy of numpy array of given tensor
                rank = torch.from_numpy(rNpArr)

            unimportant_channels[layer]=channels_to_remove=rank[prune_rates[layer]:]

            for name, param in net.named_parameters():
                if layer in name and ("bn" not in name) and ("parameter" not in name) and ("shortcut" not in name):
                    channels_to_remove=unimportant_channels[layer]
                    param.data[channels_to_remove]=0


else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()





# Training
def train(epoch):

    def gradi10(module):
        module[unimportant_channels["layer1.0"]] = 0

    def gradi11(module):
        module[unimportant_channels["layer1.1"]] = 0

    def gradi12(module):
        module[unimportant_channels["layer1.2"]] = 0

    def gradi13(module):
        module[unimportant_channels["layer1.3"]] = 0

    def gradi20(module):
        module[unimportant_channels["layer2.0"]] = 0

    def gradi21(module):
        module[unimportant_channels["layer2.1"]] = 0

    def gradi22(module):
        module[unimportant_channels["layer2.2"]] = 0

    def gradi23(module):
        #print("23",module.shape)
        module[unimportant_channels["layer2.3"]] = 0

    def gradi30(module):
        #print("30",module.shape)
        module[unimportant_channels["layer3.0"]] = 0

    def gradi31(module):
        #print("31", module.shape)
        module[unimportant_channels["layer3.1"]] = 0

    def gradi32(module):
        module[unimportant_channels["layer3.2"]] = 0

    def gradi33(module):
        module[unimportant_channels["layer3.3"]] = 0

    if use_cuda:
        net.module.layer1[0].conv1.weight.register_hook(gradi10)
        net.module.layer1[0].conv1.bias.register_hook(gradi10)
        net.module.layer1[0].bn2.weight.register_hook(gradi10)
        net.module.layer1[0].bn2.bias.register_hook(gradi10)

        net.module.layer1[1].conv1.weight.register_hook(gradi11)
        net.module.layer1[1].conv1.bias.register_hook(gradi11)
        net.module.layer1[1].bn2.weight.register_hook(gradi11)
        net.module.layer1[1].bn2.bias.register_hook(gradi11)

        net.module.layer1[2].conv1.weight.register_hook(gradi12)
        net.module.layer1[2].conv1.bias.register_hook(gradi12)
        net.module.layer1[2].bn2.weight.register_hook(gradi12)
        net.module.layer1[2].bn2.bias.register_hook(gradi12)

        net.module.layer1[3].conv1.weight.register_hook(gradi13)
        net.module.layer1[3].conv1.bias.register_hook(gradi13)
        net.module.layer1[3].bn2.weight.register_hook(gradi13)
        net.module.layer1[3].bn2.bias.register_hook(gradi13)

        net.module.layer2[0].conv1.weight.register_hook(gradi20)
        net.module.layer2[0].conv1.bias.register_hook(gradi20)
        net.module.layer2[0].bn2.weight.register_hook(gradi20)
        net.module.layer2[0].bn2.bias.register_hook(gradi20)

        net.module.layer2[1].conv1.weight.register_hook(gradi21)
        net.module.layer2[1].conv1.bias.register_hook(gradi21)
        net.module.layer2[1].bn2.weight.register_hook(gradi21)
        net.module.layer2[1].bn2.bias.register_hook(gradi21)

        net.module.layer2[2].conv1.weight.register_hook(gradi22)
        net.module.layer2[2].conv1.bias.register_hook(gradi22)
        net.module.layer2[2].bn2.weight.register_hook(gradi22)
        net.module.layer2[2].bn2.bias.register_hook(gradi22)

        net.module.layer2[3].conv1.weight.register_hook(gradi23)
        net.module.layer2[3].conv1.bias.register_hook(gradi23)
        net.module.layer2[3].bn2.weight.register_hook(gradi23)
        net.module.layer2[3].bn2.bias.register_hook(gradi23)

        net.module.layer3[0].conv1.weight.register_hook(gradi30)
        net.module.layer3[0].conv1.bias.register_hook(gradi30)
        net.module.layer3[0].bn2.weight.register_hook(gradi30)
        net.module.layer3[0].bn2.bias.register_hook(gradi30)

        net.module.layer3[1].conv1.weight.register_hook(gradi31)
        net.module.layer3[1].conv1.bias.register_hook(gradi31)
        net.module.layer3[1].bn2.weight.register_hook(gradi31)
        net.module.layer3[1].bn2.bias.register_hook(gradi31)

        net.module.layer3[2].conv1.weight.register_hook(gradi32)
        net.module.layer3[2].conv1.bias.register_hook(gradi32)
        net.module.layer3[2].bn2.weight.register_hook(gradi32)
        net.module.layer3[2].bn2.bias.register_hook(gradi32)

        net.module.layer3[3].conv1.weight.register_hook(gradi33)
        net.module.layer3[3].conv1.bias.register_hook(gradi33)
        net.module.layer3[3].bn2.weight.register_hook(gradi33)
        net.module.layer3[3].bn2.bias.register_hook(gradi33)
    else:
        net.layer1[0].conv1.weight.register_hook(gradi10)
        net.layer1[0].conv1.bias.register_hook(gradi10)
        net.layer1[0].bn2.weight.register_hook(gradi10)
        net.layer1[0].bn2.bias.register_hook(gradi10)

        net.layer1[1].conv1.weight.register_hook(gradi11)
        net.layer1[1].conv1.bias.register_hook(gradi11)
        net.layer1[1].bn2.weight.register_hook(gradi11)
        net.layer1[1].bn2.bias.register_hook(gradi11)

        net.layer1[2].conv1.weight.register_hook(gradi12)
        net.layer1[2].conv1.bias.register_hook(gradi12)
        net.layer1[2].bn2.weight.register_hook(gradi12)
        net.layer1[2].bn2.bias.register_hook(gradi12)

        net.layer1[3].conv1.weight.register_hook(gradi13)
        net.layer1[3].conv1.bias.register_hook(gradi13)
        net.layer1[3].bn2.weight.register_hook(gradi13)
        net.layer1[3].bn2.bias.register_hook(gradi13)

        net.layer2[0].conv1.weight.register_hook(gradi20)
        net.layer2[0].conv1.bias.register_hook(gradi20)
        net.layer2[0].bn2.weight.register_hook(gradi20)
        net.layer2[0].bn2.bias.register_hook(gradi20)

        net.layer2[1].conv1.weight.register_hook(gradi21)
        net.layer2[1].conv1.bias.register_hook(gradi21)
        net.layer2[1].bn2.weight.register_hook(gradi21)
        net.layer2[1].bn2.bias.register_hook(gradi21)

        net.layer2[2].conv1.weight.register_hook(gradi22)
        net.layer2[2].conv1.bias.register_hook(gradi22)
        net.layer2[2].bn2.weight.register_hook(gradi22)
        net.layer2[2].bn2.bias.register_hook(gradi22)

        net.layer2[3].conv1.weight.register_hook(gradi23)
        net.layer2[3].conv1.bias.register_hook(gradi23)
        net.layer2[3].bn2.weight.register_hook(gradi23)
        net.layer2[3].bn2.bias.register_hook(gradi23)

        net.layer3[0].conv1.weight.register_hook(gradi30)
        net.layer3[0].conv1.bias.register_hook(gradi30)
        net.layer3[0].bn2.weight.register_hook(gradi30)
        net.layer3[0].bn2.bias.register_hook(gradi30)

        net.layer3[1].conv1.weight.register_hook(gradi31)
        net.layer3[1].conv1.bias.register_hook(gradi31)
        net.layer3[1].bn2.weight.register_hook(gradi31)
        net.layer3[1].bn2.bias.register_hook(gradi31)

        net.layer3[2].conv1.weight.register_hook(gradi32)
        net.layer3[2].conv1.bias.register_hook(gradi32)
        net.layer3[2].bn2.weight.register_hook(gradi32)
        net.layer3[2].bn2.bias.register_hook(gradi32)

        net.layer3[3].conv1.weight.register_hook(gradi33)
        net.layer3[3].conv1.bias.register_hook(gradi33)
        net.layer3[3].bn2.weight.register_hook(gradi33)
        net.layer3[3].bn2.bias.register_hook(gradi33)




    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        #print(net.layer1[0].conv1.weight)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+'_pruned_'+args.arch+'_acc_'+str(acc)+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            best_acc = acc



print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

torch.cuda.empty_cache()

# made from main2vgg_switch_integral
#and integral was made from main2vgg_switch


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
import numpy as np
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import torch.nn.functional as f
import argparse
import socket
import scipy
import scipy.io

#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch"))
    path_switch = os.path.join(cwd, "results_switch")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_switch = cwd
    path_main= parent_path




#file_dir = os.path.dirname("utlis.p")
#sys.path.append(file_dir)

#from models import *

#from utils import progress_bar


##########################################
resume=True


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

##############################3
# CHANGE

#hideen_dum
#recopy the position of switches in class VGG
#name at the bottom, to check if the correct version is there


#############################
# PARAMS

architecture="vgg"
#annealing
#how_many_epochs=200
#annealing_steps = float(30000 * how_many_epochs)
annealing_steps = float(6000000)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
#alpha and switch_init
# arguments=argparse.ArgumentParser()
# arguments.add_argument("--alpha", default=0.5, type=float)#2 # below 1 so that we encourage sparsity
# arguments.add_argument("--switch_init", default=0.05, type=float)#-1
# arguments.add_argument("--layer", default='conv1')
# arguments.add_argument("--epochs_num", default=10)
# arguments.add_argument("--switch_samps", default=3)
# arguments.add_argument("--path_switch_checkpoint", default="None")
# args=arguments.parse_args()
#alpha = float(sys.argv[2]) if len (sys.argv)>2 else 0.5#2  # below 1 so that we encourage sparsity
#switch_init=float(sys.argv[3]) if len (sys.argv)>3 else 0.05#-1
#switch_layer= sys.argv[1] if len(sys.argv)>1 else 'conv14' # only name, need to change the position of the switches manually separately
# alpha=args.alpha
# switch_init=args.switch_init
# switch_layer=args.layer
# epochs_num=args.epochs_num
# switch_samps=args.switch_samps
dataset='cifar'


BATCH_SIZE=100
model_parameters = '94.34'
print(len(sys.argv))
lr = 0.1

epoch_to_save=1

# print(args.layer)
#
# #saving
# save_path=path_switch+"/results/cifar/vgg_%s/switch_init_%.2f_alpha_%.2f_annealing_%d" % (model_parameters, alpha, switch_init, annealing_steps)
# os.makedirs(save_path, exist_ok=True)
# save_textfile="%s/switch_init_%.2f, alpha_%.2f.txt" % (save_path, alpha, switch_init)
# save_switches_params=True
# save_switches_text=True

############################################################
# NETWORK


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGGKAM': [0, 39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62],
    'VGGKAMFULL': [0, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'VGGBC': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_structure = cfg['VGGKAMFULL']
# hidden_dim2 = model_structure[int(switch_layer[4:])] #it's a number of parameters we want to estimate, e.g. # conv1 filters


class VGG(nn.Module):
    def __init__(self, vgg_name, switch_samps, switch_init, hidden_dim):
        super(VGG, self).__init__()
        # self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)

        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp1 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp2 = nn.MaxPool2d(2)

        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp3 = nn.MaxPool2d(2)

        self.c8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp4 = nn.MaxPool2d(2)

        self.c11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp5 = nn.MaxPool2d(2)

        self.c14 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c15 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp5 = nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
        # self.ap = nn.AvgPool2d(1, stride=1)

        self.l1 = nn.Linear(512, 512)
        # self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout()
        self.d2 = nn.Dropout()

        # self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

        # print("switch param %.1f" % switch_init)
        self.parameter_switch = Parameter(switch_init * torch.ones(hidden_dim),
                                          requires_grad=True)  # this parameter lies #S

    def switch_multiplication(self, output, Sprime):
        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])
        return output

    def forward(self, x, switch_layer):
        phi = f.softplus(self.parameter_switch)
        S = phi / torch.sum(phi)
        # Smax = torch.max(S)
        # Sprime = S/Smax
        Sprime = S

        output = self.c1(x)
        if (switch_layer == 'conv1'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn1(output))
        output = self.c2(output)
        if (switch_layer == 'conv2'):
            output = self.switch_multiplication(output, Sprime)
        output = self.mp1(output)

        output = self.c3(output)
        if (switch_layer == 'conv3'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn3(output))
        output = self.c4(output)
        if (switch_layer == 'conv4'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn4(output))
        output = self.mp2(output)

        output = self.c5(output)
        if (switch_layer == 'conv5'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn5(output))
        output = self.c6(output)
        if (switch_layer == 'conv6'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn6(output))
        output = self.c7(output)
        if (switch_layer == 'conv7'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn7(output))
        output = self.mp3(output)

        output = self.c8(output)
        if (switch_layer == 'conv8'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn8(output))
        output = self.c9(output)
        if (switch_layer == 'conv9'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn9(output))
        output = self.c10(output)
        if (switch_layer == 'conv10'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn10(output))
        output = self.mp4(output)

        output = self.c11(output)
        if (switch_layer == 'conv11'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn11(output))
        output = self.c12(output)
        if (switch_layer == 'conv12'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn12(output))
        output = self.c13(output)
        if (switch_layer == 'conv13'):
            output = self.switch_multiplication(output, Sprime)
        output = f.relu(self.bn13(output))
        output = self.mp5(output)

        output = output.view(-1, 512)
        output = self.l1(output)
        if (switch_layer == 'conv14'):
            output = self.switch_multiplication(output, Sprime)
        output = self.l3(output)

        # output = f.relu(self.l1(output))
        # output = self.d2(output)
        # output = f.relu(self.l2(output))

        # out = self.features(x)
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return output, Sprime


def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)





# def test():
#     net = VGG('VGG11')
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())

# test()

#####################################
# DATA

#parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
#args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if dataset=="cifar":

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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')








###################################################
# MAKE AN INSTANCE OD NETWORK AND (POSSIBLY) LOAD THE MODEL
# if we uncomment this and load weights to that net, then the wronf (uniform) switches appear

# Model
#print('==> Building model..')
#net = VGG('VGG16', switch_samps)

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)

#net = net.to(device)

# for name, param in net.named_parameters():
#     print (name, param.shape)


# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
#     print(device)

#if args.resume:
def load_weights(net_all, path_switch_checkpoint=""):
    if (resume):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        if dataset=='cifar':
            if path_switch_checkpoint=="":
                path_switch_checkpoint=path_switch + '/checkpoint/ckpt_vgg16_%s.t7' % model_parameters
            else:
                path_switch_checkpoint=path_switch_checkpoint
            checkpoint = torch.load(path_switch_checkpoint, map_location=lambda storage, loc: storage)
            net_all.load_state_dict(checkpoint['net'], strict=False)



        #best_acc = checkpoint['acc']
        #start_epoch = checkpoint['epoch']

#load_weights(net)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


#######################s
# LOSS



def loss_function(prediction, true_y, S, alpha, hidden_dim, how_many_samps):
    # BCE = f.binary_cross_entropy(prediction, true_y, reduction='sum')
    BCE = criterion(prediction, true_y)

    return BCE


def loss_functionKL(prediction, true_y, S, alpha, hidden_dim, batch_size, annealing_rate):
    # BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')
    BCE = criterion(prediction, true_y)

    # KLD term
    alpha = torch.Tensor([alpha]).to(device)
    hidden_dim = torch.Tensor([hidden_dim]).to(device)
    trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim * alpha)
    trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim * torch.lgamma(alpha)
    trm3 = torch.sum((S - alpha) * (torch.digamma(S) - torch.digamma(torch.sum(S))))
    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better
    return BCE, BCE, KLD, annealing_rate * KLD / batch_size

    #return BCE + annealing_rate * KLD / batch_size, BCE, KLD, annealing_rate * KLD / batch_size
    #return BCE + annealing_rate * KLD / batch_size, BCE, KLD, annealing_rate * KLD / batch_size

########################################################
# Training


def train(epoch, net_all, optimizer, hidden_dim, switch_layer):
    print('\nEpoch: %d' % epoch)
    net_all.train()
    train_loss = 0
    correct = 0
    total = 0
    annealing_rate = beta_func(epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs, S = net_all(inputs, switch_layer)
        #loss = criterion(outputs, targets)
        #print("alpha: %.1f" % alpha)
        loss, BCE, KLD, KLD_discounted = loss_functionKL(outputs, targets, S, alpha, hidden_dim, BATCH_SIZE, annealing_rate)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx % 1000 ==0):
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print('BCE: %.3f, KLD: %.3f, KLD_discounted: %.3f' % (BCE, KLD, KLD_discounted))
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("S")
    print(S)
    print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
    # print(torch.argsort(S))
    ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
    # if epoch == epoch_to_save and save_switches_params:
    #     torch.save(S, '%s/%s_new_alpha%.2f_switchinit%.2f_%s_ep%d.pt' % (save_path, model_parameters, alpha, switch_init, switch_layer, epoch))
    #     if save_switches_text:
    #         with open(save_textfile, "a+") as file:
    #             file.write(switch_layer+"\n\n")
    #             file.write('\nEpoch: %d\n' % epoch)
    #             file.write("alpha:%.2f switchinit:%.2f\n" % (alpha, switch_init))
    #             file.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #             file.write('BCE: %.3f, KLD: %.3f, KLD_discounted: %.3f\n' % (BCE, KLD, KLD_discounted))
    #             file.write(" ".join(str(item) for item in S.cpu().detach().numpy()))
    #             file.write("\nmax: %.4f, min: %.4f\n" % (torch.max(S), torch.min(S)))
    #             file.write(",".join(map(str, ranks_sorted)))
    #             file.write("\n\n\n")


    print(",".join(map(str, ranks_sorted)))
    return ranks_sorted, S


#################################################################
# TEST



def test(epoch, net_all, switch_layer):
    global best_acc
    net_all.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
            outputs = net_all(inputs, switch_layer)[0] #[0] added because of the tuple output,S
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net_all.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_%.2f.t7' % acc)
        best_acc = acc

###########################################################
#copied from network pruning
#
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
#                     accuracy = test(-1)
#                     param[combination] = params_saved
#                     params_bias[combination] = param_bias_saved
#
#                     #if file_write:
#                     print("out")
#                     with open("results_running/combinations_pruning_cifar_vgg_%s.txt" % (layer),
#                               "a+") as textfile:
#                         textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))


##########################################################################3
# RUN EXPERIMENT

#file_write=True
#compute_combinations_random(file_write)

def main(switch_layer, epochs_num, switch_samps, alpha_param, switch_init_param, file_path):


    global alpha, switch_init
    alpha=alpha_param
    switch_init=switch_init_param

    training=True
    hidden_dim = model_structure[int(switch_layer[4:])]  # it's a number of parameters we want to estimate, e.g. # conv1 filters

    # Model
    print('==> Building model..')
    net2 = VGG('VGG16', switch_samps, switch_init, hidden_dim)
    net2 = net2.to(device)

    if device == 'cuda':
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True
        print(device)


    for name, param in net2.named_parameters():
        #print(name, param.shape)
        if (name != "module.parameter_switch"):
            h = param.register_hook(lambda grad: grad * 0)  # double the gradient

    if training:
        optimizer = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        if 1:
            if 1:


                print("switch_init %.2f, alpha %.2f" %(switch_init, alpha))
                load_weights(net2)

                for epoch in range(0, epochs_num):
                    print(switch_layer)
                    print(hidden_dim)

                    ranks, switches = train(epoch, net2, optimizer, hidden_dim, switch_layer)
                    test(epoch, net2, switch_layer)

    return ranks, switches


if __name__=="__main__":
    main("conv1", epochs_num, switch_samps)
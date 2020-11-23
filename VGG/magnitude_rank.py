import torch
from torch import nn, optim

import torch
import torch.optim as optim
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb
import os
import sys
import socket

from torch.nn.parameter import Parameter
#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch"))
    path_compression = os.path.join(cwd, "results_compression")
    path_networktest = os.path.join(cwd, "results_networktest")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_compression = cwd
    path_networktest = os.path.join(parent_path, "results_networktest")
    path_main= parent_path

print(cwd)
print(sys.path)

print("newh2")
sys.path.append(os.path.join(path_networktest, "external_codes/pytorch-cifar-master/models"))
sys.path.append(path_compression)


############################

#######################
# takes the network parameters from the conv layer and clusters them (with the purpose of removing some of them)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainval_perc = 1
BATCH_SIZE = 100


def setup(network_arg='vgg_cifar', dataset_arg='cifar'):
    global dataset;
    dataset = dataset_arg
    global network;
    network = network_arg

    if network == 'vgg':

        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGGKAM': [0, 39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62],
            'VGGBC': [0, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        model_structure = cfg['VGGBC']

        class VGG(nn.Module):
            def __init__(self, vgg_name):
                super(VGG, self).__init__()

                # self.features = self._make_layers(cfg[vgg_name])
                # self.classifier = nn.Linear(512, 10)
                # model_structure={'c1_num':39, 'c2_num'=39, 'c3_num'=63; 'c4_num'=48, 'c5_num'=55, 'c6_num'=98, 'c7_num'=97, 'c8_num'=52, 'c9_num'=62,
                # 'c10_num'=22, 'c11_num'=42, 'c12_num'=47 ; 'c13_num'=47 ; 'c14_num'=42 ; 'c15_num'=62}

                self.c1 = nn.Conv2d(3, model_structure[1], 3, padding=1)
                self.bn1 = nn.BatchNorm2d(model_structure[1], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c2 = nn.Conv2d(model_structure[1], model_structure[2], 3, padding=1)
                self.bn2 = nn.BatchNorm2d(model_structure[2], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.mp1 = nn.MaxPool2d(2)

                self.c3 = nn.Conv2d(model_structure[2], model_structure[3], 3, padding=1)
                self.bn3 = nn.BatchNorm2d(model_structure[3], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c4 = nn.Conv2d(model_structure[3], model_structure[4], 3, padding=1)
                self.bn4 = nn.BatchNorm2d(model_structure[4], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.mp2 = nn.MaxPool2d(2)

                self.c5 = nn.Conv2d(model_structure[4], model_structure[5], 3, padding=1)
                self.bn5 = nn.BatchNorm2d(model_structure[5], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c6 = nn.Conv2d(model_structure[5], model_structure[6], 3, padding=1)
                self.bn6 = nn.BatchNorm2d(model_structure[6], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c7 = nn.Conv2d(model_structure[6], model_structure[7], 3, padding=1)
                self.bn7 = nn.BatchNorm2d(model_structure[7], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.mp3 = nn.MaxPool2d(2)

                self.c8 = nn.Conv2d(model_structure[7], model_structure[8], 3, padding=1)
                self.bn8 = nn.BatchNorm2d(model_structure[8], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c9 = nn.Conv2d(model_structure[8], model_structure[9], 3, padding=1)
                self.bn9 = nn.BatchNorm2d(model_structure[9], eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)
                self.c10 = nn.Conv2d(model_structure[9], model_structure[10], 3, padding=1)
                self.bn10 = nn.BatchNorm2d(model_structure[10], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.c11 = nn.Conv2d(model_structure[10], model_structure[11], 3, padding=1)
                self.bn11 = nn.BatchNorm2d(model_structure[11], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.mp4 = nn.MaxPool2d(2)

                self.c12 = nn.Conv2d(model_structure[11], model_structure[12], 3, padding=1)
                self.bn12 = nn.BatchNorm2d(model_structure[12], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.c13 = nn.Conv2d(model_structure[12], model_structure[13], 3, padding=1)
                self.bn13 = nn.BatchNorm2d(model_structure[13], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.c14 = nn.Conv2d(model_structure[13], model_structure[14], 3, padding=1)
                self.bn14 = nn.BatchNorm2d(model_structure[14], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.c15 = nn.Conv2d(model_structure[14], model_structure[15], 3, padding=1)
                self.bn15 = nn.BatchNorm2d(model_structure[15], eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True)
                self.mp5 = nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
                self.ap = nn.AvgPool2d(1, stride=1)

                self.l3 = nn.Linear(model_structure[15], 10)
                self.d1 = nn.Dropout()
                self.d2 = nn.Dropout()

                self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

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

    elif network == 'vgg_cifar':

        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGGBC': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }

        class VGG(nn.Module):
            def __init__(self, vgg_name):
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

                self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

            # def forward(self, x, i):  # VISU
            def forward(self, x):
                phi = f.softplus(self.parameter)
                S = phi / torch.sum(phi)
                # Smax = torch.max(S)
                # Sprime = S/Smax
                Sprime = S

                output = f.relu(self.bn1(self.c1(x)))
                output = f.relu(self.bn2(self.c2(output)))
                output = self.mp1(output)

                output = f.relu(self.bn3(self.c3(output)))
                output = f.relu(self.bn4(self.c4(output)))
                output = self.mp2(output)

                output = f.relu(self.bn5(self.c5(output)))
                output = f.relu(self.bn6(self.c6(output)))
                output = f.relu(self.bn7(self.c7(output)))
                output = self.mp3(output)

                output = f.relu(self.bn8(self.c8(output)))
                output = f.relu(self.bn9(self.c9(output)))
                output = f.relu(self.bn10(self.c10(output)))
                output = self.mp4(output)

                output = f.relu(self.bn11(self.c11(output)))
                output = f.relu(self.bn12(self.c12(output)))
                output = f.relu(self.bn13(self.c13(output)))
                output = self.mp5(output)

                # output = f.relu(self.bn14(self.c14(output)))
                # output = f.relu(self.bn15(self.c15(output)))
                # output = self.mp5(output)
                # output = self.ap(output)

                output = output.view(-1, 512)
                output = self.l1(output)
                output = self.l3(output)

                # output = f.relu(self.l1(output))
                # output = self.d2(output)
                # output = f.relu(self.l2(output))

                # out = self.features(x)
                # out = out.view(out.size(0), -1)
                # out = self.classifier(out)
                return output

            # def forward(self, x):
            #     out = self.features(x)
            #     out = out.view(out.size(0), -1)
            #     out = self.classifier(out)
            #     return out

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


    elif network == 'lenet':

        ####################################
        # NETWORK (conv-conv-fc-fc)

        class Lenet(nn.Module):
            def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
                super(Lenet, self).__init__()

                self.nodesNum2 = nodesNum2

                self.c1 = nn.Conv2d(1, nodesNum1, 5)
                self.s2 = nn.MaxPool2d(2)
                self.bn1 = nn.BatchNorm2d(nodesNum1)
                self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
                self.s4 = nn.MaxPool2d(2)
                self.bn2 = nn.BatchNorm2d(nodesNum2)
                self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
                self.f6 = nn.Linear(nodesFc1, nodesFc2)
                self.output = nn.Linear(nodesFc2, 10)

                self.parameter = Parameter(-1e-10 * torch.ones(nodesNum1), requires_grad=True)  # this parameter lies #S

            def forward(self, x):
                # output=f.relu(self.fc1(x))
                # output=self.bn1(output)
                # output=f.relu(self.fc2(output))
                # output=self.bn2(output)
                # output=self.fc3(output)
                # return output

                # #x=x.view(-1,784)
                output = self.c1(x)
                output = f.relu(self.s2(output))
                output = self.bn1(output)
                output = self.c3(output)
                output = f.relu(self.s4(output))
                output = self.bn2(output)
                output = output.view(-1, self.nodesNum2 * 4 * 4)
                output = self.c5(output)
                output = self.f6(output)
                return output

    global net

    ################# MODEL CHECKPOINTS

    if network == 'vgg':
        net = VGG('VGG16')

        path = "checkpoint/ckpt_93.92.t7"

        net = net.to(device)

        net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)
        # print("c1")
        # print(net.c1.weight)

    if network == 'vgg_cifar':

        net = VGG('VGG16')

        if torch.cuda.is_available():
            net = torch.nn.DataParallel(net)
            # cudnn.benchmark = True
            # print(device)

        # for name, param in net.named_parameters():
        #     print (name, param.shape)

        checkpoint = torch.load(path_compression+'/checkpoint/ckpt_vgg16_94.34.t7')
        # checkpoint = torch.load('./checkpoint/ckpt_vgg16_prunedto[39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62]_64.55.t7')
        net.load_state_dict(checkpoint['net'])
        # print(net.module.c1.module.weight)

        # path='./checkpoint/ckpt_vgg16_94.06.t7'

        # path="checkpoint/ckpt_93.92.t7"

        # net=net.to(device)

        # net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['net'], strict=False)
        # print("c1")
        # print(net.c1.weight)


    elif network == 'lenet':
        nodesNum1, nodesNum2, nodesFc1, nodesFc2 = 10, 20, 100, 25
        net = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2)

        if dataset == "mnist":
            path = "models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
            # path="models/mnist_trainval0.9_epo461_acc99.06"
        elif dataset == "fashionmnist":
            path = "models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"

        net = net.to(device)

        # net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
        # net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)

    # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
    # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"


###################################################################
#####################################3


def get_ranks(method, net):
    combinationss = [];
    combinationss_weights = []

    # print(net.named_parameters()['c1.weight'])
    for name, param in net.named_parameters():

        if (("c" in name) or ("f" in name) or ("l1" in name)) and ("weight" in name): #i think f and l are both fullt connected
            # print(name)
            print (name, param.shape)
            m = torch.flatten(param, start_dim=1)

            l2 = torch.norm(m, p=2, dim=1)
            l1 = torch.norm(m, p=1, dim=1)
            # print(l1)
            # print(l2)

            # m = torch.flatten(param, start_dim=1)
            # m = torch.abs(m)
            # sum = m.sum(1)#
            # print(sum) #SAME AS L1

            # l1rank=torch.argsort(l1, descending=True)
            # l2rank=torch.argsort(l2, descending=True)

            l1 = l1.detach().cpu().numpy()
            l2 = l2.detach().cpu().numpy()

            # for lenet change the order
            # l1rank = np.argsort(l1)[::-1]
            # l2rank = np.argsort(l2)[::-1]

            l1rank = np.argsort(l1)[::-1]
            l2rank = np.argsort(l2)[::-1]
            l1weightrank = np.sort(l1)[::-1]
            l2weightrank = np.sort(l2)[::-1]

            # print(l1rank)

            if method == 'l1':
                # print(l1rank)
                combinationss.append(l1rank)
                combinationss_weights.append(l1weightrank)
            elif method == 'l2':
                combinationss.append(l2rank)
                combinationss_weights.append(l2weightrank)

            # if method == 'l1':
            #     combinationss.append(torch.LongTensor(l1rank))
            # elif method == 'l2':
            #     combinationss.append(torch.LongTensor(l2rank))

    return combinationss
    # sum=np.sum(param[0, :].sum()
    # print(sum)


if __name__ == '__main__':
    setup()
    comb = get_ranks('l2')

    # print('\n', comb[0])
    # print('\n', comb_weights[0])

    # for i in range(len(comb)):
    #    print('\n',comb[i])
    #    print('\n', comb_weights[i])
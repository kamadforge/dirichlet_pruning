import torch.nn.functional as f
import matplotlib.pyplot as plt
#import parent module
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import argparse
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


cfg = {
    'VGG15': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, arch=cfg['VGG15']):
        super(VGG, self).__init__()

        self.cfg_arch=arch
        cfg_arch=self.cfg_arch

        self.c1 = nn.Conv2d(3, cfg_arch[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cfg_arch[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c2 = nn.Conv2d(cfg_arch[0], cfg_arch[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cfg_arch[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp1 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(cfg_arch[1], cfg_arch[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(cfg_arch[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c4 = nn.Conv2d(cfg_arch[2], cfg_arch[3], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(cfg_arch[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp2 = nn.MaxPool2d(2)

        self.c5 = nn.Conv2d(cfg_arch[3], cfg_arch[4], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(cfg_arch[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c6 = nn.Conv2d(cfg_arch[4], cfg_arch[5], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(cfg_arch[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c7 = nn.Conv2d(cfg_arch[5], cfg_arch[6], 3, padding=1)
        self.bn7 = nn.BatchNorm2d(cfg_arch[6], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp3 = nn.MaxPool2d(2)

        self.c8 = nn.Conv2d(cfg_arch[6], cfg_arch[7], 3, padding=1)
        self.bn8 = nn.BatchNorm2d(cfg_arch[7], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c9 = nn.Conv2d(cfg_arch[7], cfg_arch[8], 3, padding=1)
        self.bn9 = nn.BatchNorm2d(cfg_arch[8], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c10 = nn.Conv2d(cfg_arch[8], cfg_arch[9], 3, padding=1)
        self.bn10 = nn.BatchNorm2d(cfg_arch[9], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp4 = nn.MaxPool2d(2)

        self.c11 = nn.Conv2d(cfg_arch[9], cfg_arch[10], 3, padding=1)
        self.bn11 = nn.BatchNorm2d(cfg_arch[10], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c12 = nn.Conv2d(cfg_arch[10], cfg_arch[11], 3, padding=1)
        self.bn12 = nn.BatchNorm2d(cfg_arch[11], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c13 = nn.Conv2d(cfg_arch[11], cfg_arch[12], 3, padding=1)
        self.bn13 = nn.BatchNorm2d(cfg_arch[12], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp5 = nn.MaxPool2d(2)

        self.l1 = nn.Linear(cfg_arch[12], cfg_arch[13])
        self.l3 = nn.Linear(cfg_arch[13], 10)
        self.d1 = nn.Dropout()
        self.d2 = nn.Dropout()

        self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S




    # def forward(self, x, i):  # VISU
    def forward(self, x, i=-1):
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

        output = output.view(-1, self.cfg_arch[12])
        output = self.l1(output)
        output = self.l3(output)

        return output

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

#####################################
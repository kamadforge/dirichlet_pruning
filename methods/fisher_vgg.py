import torch.nn as nn
import torch
import torch.nn.functional as f
device = 'cuda' if torch.cuda.is_available() else 'cpu'



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG15': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    #'VGG15_comp': [39, 39, 63, 48, 55, 88, 87, 52, 62, 22, 42, 47, 47, 47],
    'VGG15_comp': [34, 34, 68, 68, 75, 106, 101, 92, 102, 92, 67, 67, 62, 62],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG_fisher(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_fisher, self).__init__()

        cfg_arch=cfg['VGG15']

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

        #self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

        # Fisher method is called on backward passes
        self.running_fisher = []
        # for i in range(3):
        #     self.running_fisher.append(torch.Tensor(64).to(device))
        # for i in range(2):
        #     self.running_fisher.append(torch.Tensor(128).to(device))
        # for i in range(3):
        #     self.running_fisher.append(torch.Tensor(256).to(device))
        # for i in range(7):
        #     self.running_fisher.append(torch.Tensor(512).to(device))

        self.running_fisher.append(torch.Tensor(64).to(device)) #first dummy for 0
        for i in range(len(cfg_arch)):
            self.running_fisher.append(torch.Tensor(cfg_arch[i]).to(device))


        self.act = [0] * 15

        self.activation1 = Identity()
        self.activation2 = Identity()
        self.activation3 = Identity()
        self.activation4 = Identity()
        self.activation5 = Identity()
        self.activation6 = Identity()
        self.activation7 = Identity()
        self.activation8 = Identity()
        self.activation9 = Identity()
        self.activation10 = Identity()
        self.activation11 = Identity()
        self.activation12 = Identity()
        self.activation13 = Identity()
        self.activation14 = Identity()

        self.activation1.register_backward_hook(self._fisher1)
        self.activation2.register_backward_hook(self._fisher2)
        self.activation3.register_backward_hook(self._fisher3)
        self.activation4.register_backward_hook(self._fisher4)
        self.activation5.register_backward_hook(self._fisher5)
        self.activation6.register_backward_hook(self._fisher6)
        self.activation7.register_backward_hook(self._fisher7)
        self.activation8.register_backward_hook(self._fisher8)
        self.activation9.register_backward_hook(self._fisher9)
        self.activation10.register_backward_hook(self._fisher10)
        self.activation11.register_backward_hook(self._fisher11)
        self.activation12.register_backward_hook(self._fisher12)
        self.activation13.register_backward_hook(self._fisher13)
        self.activation14.register_backward_hook(self._fisher14)

    # def forward(self, x, i):  # VISU
    def forward(self, x, i=-1):
        #phi = f.softplus(self.parameter)
        #S = phi / torch.sum(phi)
        # Smax = torch.max(S)
        # Sprime = S/Smax
        #Sprime = S

        # if vis:
        #     for filter_num in range(3):
        #         mm = x.cpu().detach().numpy()
        #         # Split
        #         img = mm[1, filter_num, :, :]
        #         if filter_num == 0:
        #             cmap_col = 'Reds'
        #         elif filter_num == 1:
        #             cmap_col = 'Greens'
        #         elif filter_num == 2:
        #             cmap_col = 'Blues'
        #
        #         # plt.imshow(matrix)  # showing 2nd channel (example of a channel)
        #
        #         plt.gca().set_axis_off()
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                             hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #         plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #         cax_00 = plt.imshow(img, cmap=cmap_col)
        #         plt.show()

        output = f.relu(self.bn1(self.c1(x)))

        # if vis:
        #     for filter_num in range(64):
        #         mm = output.cpu().detach().numpy()
        #
        #         matrix = mm[1, filter_num, :, :]
        #         print(filter_num)
        #         # print(matrix[0:20, 0])
        #         # ave=0
        #         ave = np.average(matrix[0:20, 0])
        #         matrix = matrix - ave
        #
        #         plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)  # showing 2nd channel (example of a channel)
        #
        #         plt.gca().set_axis_off()
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                             hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #         plt.gca().yaxis.set_major_locator(plt.NullLocator())


        # out = self.activation1(output)
        self.act[1] = self.activation1(output)
        output = f.relu(self.bn2(self.c2(output)))
        self.act[2] = self.activation2(output)
        output = self.mp1(output)

        output = f.relu(self.bn3(self.c3(output)))
        self.act[3] = self.activation3(output)
        output = f.relu(self.bn4(self.c4(output)))
        self.act[4] = self.activation4(output)
        output = self.mp2(output)

        output = f.relu(self.bn5(self.c5(output)))
        self.act[5] = self.activation5(output)
        output = f.relu(self.bn6(self.c6(output)))
        self.act[6] = self.activation6(output)
        output = f.relu(self.bn7(self.c7(output)))
        self.act[7] = self.activation7(output)
        output = self.mp3(output)

        output = f.relu(self.bn8(self.c8(output)))
        self.act[8] = self.activation8(output)
        output = f.relu(self.bn9(self.c9(output)))
        self.act[9] = self.activation9(output)
        output = f.relu(self.bn10(self.c10(output)))
        self.act[10] = self.activation10(output)
        output = self.mp4(output)

        output = f.relu(self.bn11(self.c11(output)))
        self.act[11] = self.activation11(output)
        output = f.relu(self.bn12(self.c12(output)))
        self.act[12] = self.activation12(output)
        output = f.relu(self.bn13(self.c13(output)))
        self.act[13] = self.activation13(output)
        output = self.mp5(output)

        output = output.view(-1, cfg['VGG15'][13])
        output = self.l1(output)
        self.act[14] = self.activation14(output)
        output = self.l3(output)

        return output

    def _fisher1(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 1)

    def _fisher2(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 2)

    def _fisher3(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 3)

    def _fisher4(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 4)

    def _fisher5(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 5)

    def _fisher6(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 6)

    def _fisher7(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 7)

    def _fisher8(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 8)

    def _fisher9(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 9)

    def _fisher10(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 10)

    def _fisher11(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 11)

    def _fisher12(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 12)

    def _fisher13(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 13)

    def _fisher14(self, notused1, notused2, grad_output):
        self._fisher_fc(grad_output, 14)

    def _fisher(self, grad_output, i):
        act = self.act[i].detach()
        grad = grad_output[0].detach()
        #
        # print("Grad: ",grad_output[0].shape)
        # print("Act: ", act.shape, '\n')

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        # print(del_k.shape)
        # print(i)
        self.running_fisher[i] += del_k

    def _fisher_fc(self, grad_output, i):
        act = self.act[i].detach()
        grad = grad_output[0].detach()
        #
        # print("Grad: ",grad_output[0].shape)
        # print("Act: ", act.shape, '\n')

        g_nk = (act * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        # print(del_k.shape)
        # print(i)
        self.running_fisher[i] += del_k

    def reset_fisher(self):
        for i in range(len(self.running_fisher)):
            self.running_fisher[i] = torch.Tensor(len(self.running_fisher[i])).to(device)

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


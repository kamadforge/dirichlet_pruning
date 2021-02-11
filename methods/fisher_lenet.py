import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        self.c1 = nn.Conv2d(1, nodesNum1, 5)
        self.s2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nodesNum1)
        self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
        self.s4 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(nodesNum2)
        self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
        self.f6 = nn.Linear(nodesFc1, nodesFc2)
        self.out7 = nn.Linear(nodesFc2, 10)

        #self.parameter = Parameter(-1e-10*torch.ones(nodesNum1),requires_grad=True) # this parameter lies #S

        # Fisher method is called on backward passes
        self.running_fisher = [0] * 4

        self.act1=0
        self.activation1 = Identity()
        self.activation1.register_backward_hook(self._fisher1)

        self.act2=0
        self.activation2 = Identity()
        self.activation2.register_backward_hook(self._fisher2)

        self.act3=0
        self.activation3 = Identity()
        self.activation3.register_backward_hook(self._fisher3)

        self.act4=0
        self.activation4 = Identity()
        self.activation4.register_backward_hook(self._fisher4)





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
        out = self.activation1(output)
        self.act1 = out
        output = self.bn1(output)

        output = self.c3(output)
        output = f.relu(self.s4(output))
        out = self.activation2(output)
        self.act2 = out
        output = self.bn2(output)

        output = output.view(-1, self.nodesNum2 * 4 * 4)
        output = self.c5(output)
        out = self.activation3(output)
        self.act3 = out


        output = self.f6(output)
        out = self.activation4(output)
        self.act4 = out
        output = self.out7(output) #remove for 99.27 and 90.04 models

        return output

    def _fisher1(self, notused1, notused2, grad_output):
        act1 = self.act1.detach()
        grad = grad_output[0].detach()
        #print(grad_output[0].shape)

        g_nk = (act1 * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        #print(del_k.shape)
        self.running_fisher[0] += del_k

    def _fisher2(self, notused1, notused2, grad_output):
        act2 = self.act2.detach()
        grad = grad_output[0].detach()

        g_nk = (act2 * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher[1] += del_k

    def _fisher3(self, notused1, notused2, grad_output):
        act3 = self.act3.detach()
        grad = grad_output[0].detach()

        g_nk = (act3 * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher[2] += del_k

    def _fisher4(self, notused1, notused2, grad_output):
        act4 = self.act4.detach()
        grad = grad_output[0].detach()

        g_nk = (act4 * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher[3] += del_k

    def reset_fisher(self):
        self.running_fisher[3] = 0 * self.running_fisher

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()

        # convs
        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[
            3] + middle_channels * out_channels * \
                      conv2_size[2] * conv2_size[3]

        # batchnorms, assuming running stats are absorbed
        self.params += 2 * in_channels + 2 * middle_channels

        # skip
        if not self.equalInOut:
            self.params += in_channels * out_channels
        else:
            self.params += 0



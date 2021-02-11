#in forward we added a layer as the second argument
#in evaluate call function we added "c3 in forward and suddenly we get an error that the output is tuple (which it is as it returns outptu and Sprime)
#why then before we didn't get the error when we didn't have forward as a second argument
###for this version invesrtiagtre

#feb 27
#we puit the net creation inside run experiment and make sure it is evaluated and upadted on that net



#it's the same test as for mnist.#L.py but with conv layers (con lenet)
#it's also a gpu version which add extra gpu support to the previous version of mnist.3L.conv.py (which wa deleted and this version was named after this)

#transforms the input data

# the difference between this file nad mnist.#L.conv.gpu (without switch is
#1. changing the loss function to cross entropy plus KL
#2. addding loading the weights (could be added there too)
#3. adding require_grad = False option for network layers

import torch.utils.data

import torch
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb
import os

from torch.nn.parameter import Parameter
from torch.distributions import Gamma


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"


#############################
# PARAMS

sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
layer="c1"
how_many_epochs=200
annealing_steps = float(8000. * how_many_epochs)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
alpha_0 = 2  # below 1 so that we encourage sparsity
hidden_dim = 10 #it's a number of parameters we want to estimate, e.g. # conv1 filters
hidden_dims={'c1': conv1, 'c3': conv2, 'c5': fc1, 'f6' : fc2}
hidden_dim = hidden_dims[layer] #it's a number of parameters we want to estimate, e.g. # conv1 filters
num_samps_for_switch = 5

###################################################
# DATA


dataset="mnist"
trainval_perc=1
BATCH_SIZE = 100

trainval_dataset=datasets.MNIST('data', train=True, download=True,
                    #transform=transforms.Compose([transforms.ToTensor(),
                    #transforms.Normalize((0.1307,), (0.3081,))]),
                    transform=transforms.ToTensor())

train_size = int(trainval_perc * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)

dataset="mnist"

##################

##############################################################################
# NETWORK (conv-conv-fc-fc)

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer, num_samps_for_switch):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        self.c1=nn.Conv2d(1, nodesNum1, 5)
        self.s2=nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(nodesNum1)


        self.c3=nn.Conv2d(nodesNum1,nodesNum2,5)
        self.s4=nn.MaxPool2d(2)
        self.bn2=nn.BatchNorm2d(nodesNum2)
        self.c5=nn.Linear(nodesNum2*4*4, nodesFc1)
        self.f6=nn.Linear(nodesFc1,nodesFc2)
        self.f7=nn.Linear(nodesFc2,10)

        self.drop_layer = nn.Dropout(p=0.5)

        self.parameter = Parameter(-1*torch.ones(hidden_dims[layer]),requires_grad=True) # this parameter lies #S
        self.num_samps_for_switch = num_samps_for_switch

    def switch_func(self, output, SstackT):
        rep = SstackT.unsqueeze(2).unsqueeze(2).repeat(1, 1, output.shape[2], output.shape[3])  # (150,10,24,24)
        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output = torch.einsum('ijkl, mjkl -> imjkl', (rep, output))
        output = output.view(output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4])
        return output, SstackT

    def switch_func_fc(self, output, SstackT):

        #rep = SstackT.unsqueeze(2).unsqueeze(2).repeat(1, 1,)  # (150,10,24,24)
        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output=torch.einsum('ij, mj -> imj', (SstackT, output))
        #output = torch.einsum('ijkl, mjkl -> imjkl', (rep, output))
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        return output, SstackT

    def forward(self, x, layer):

        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        #x=x.view(-1,784)
        output=self.c1(x)


        #############S
        phi = f.softplus(self.parameter)

        """ draw Gamma RVs using phi and 1 """
        num_samps = self.num_samps_for_switch
        concentration_param = phi.view(-1, 1).repeat(1, num_samps).to(device) #[feat x samp]
        beta_param = torch.ones(concentration_param.size()).to(device)
        # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample()  # 200, 150, hidden_dim x samples_num

        if any(torch.sum(gamma_samps, 0) == 0):
            print("sum of gamma samps are zero!")
        else:
            Sstack = gamma_samps / torch.sum(gamma_samps, 0)  # 1dim - number of neurons (200), 2dim - samples (150)

        # Sstack -switch, for output of the network (say 200) we used to have switch 200, now we have samples (150 of them), sowe have switch which is (200, 150)        #

        # output.shape
        # Out[2]: torch.Size([100, 10, 24, 24])
        # Sstack.shape
        # Out[3]: torch.Size([10, 150])

        SstackT=Sstack.t()



        #x_samps = torch.einsum("ij,jk -> ijk", (output, Sstack))
        #x_samps = F.relu(x_samps)
        #x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W2))) + self.b2
        #output=labelstack = torch.sigmoid(x_out)  # 100,200 100- sa

        #"""directly use mean of Dir RV."""
        #S = phi / torch.sum(phi)

        #Smax = torch.max(S)
        #Sprime = S/Smax
        #Sprime = Sstack

        # for i in range(len(Sprime)):
        #     output[:, i] *= Sprime[i].expand_as(output[:, i]) #13.28 deteministic, acc increases

        if layer == 'c1': #SstackT [samp, feat], output [batch x feat x convkernsize x convkernsize]
             output, Sprime = self.switch_func(output, SstackT) #13.28 deteministic, acc increases
            #output - [batch*samp x feat x covkernsize x covkernsize]

        #for i in range(len(S)):
        #output[:, i] = output[:, i] * S[i]


        #output = output[1] * S
        ##############

        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)

        if layer == 'c3':
            output, SstackT = self.switch_func(output, SstackT)  # 13.28 deteministic, acc increases

        # for i in range(len(Sprime)):
        #     output[:, i] *= Sprime[i].expand_as(output[:, i])

        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)



        output=self.c5(output)

        if layer == 'c5':
            output, SstackT = self.switch_func_fc(output, SstackT)  # 13.28 deteministic, acc increases

        output=self.f6(output)

        if layer == 'f6':
            output, SstackT = self.switch_func_fc(output, SstackT)  # 13.28 deteministic, acc increases

        output = self.f7(output)

        output = output.reshape(BATCH_SIZE, self.num_samps_for_switch, -1)
        output = torch.mean(output, 1)


        return output, phi



# class Lenet(nn.Module):
#     def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
#         super(Lenet, self).__init__()
#
#         self.nodesNum2=nodesNum2
#
#         self.c1 = nn.Conv2d(1, nodesNum1, 5)
#         self.s2 = nn.MaxPool2d(2)
#         self.bn1 = nn.BatchNorm2d(nodesNum1)
#         self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
#         self.s4 = nn.MaxPool2d(2)
#         self.bn2 = nn.BatchNorm2d(nodesNum2)
#         self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
#         self.f6 = nn.Linear(nodesFc1, nodesFc2)
#         self.output = nn.Linear(nodesFc2, 10)
#
#         self.parameter = Parameter(-1e-10*torch.ones(nodesNum1),requires_grad=True) # this parameter lies #S
#
#
#
#     def forward(self, x):
#
#         # output=f.relu(self.fc1(x))
#         # output=self.bn1(output)
#         # output=f.relu(self.fc2(output))
#         # output=self.bn2(output)
#         # output=self.fc3(output)
#         # return output
#
#         # #x=x.view(-1,784)
#         output = self.c1(x)
#         output = f.relu(self.s2(output))
#         output = self.bn1(output)
#         output = self.c3(output)
#         output = f.relu(self.s4(output))
#         output = self.bn2(output)
#         output = output.view(-1, self.nodesNum2 * 4 * 4)
#         output = self.c5(output)
#         output = self.f6(output)
#         return output

####################

nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
# net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2, layer).to(device)
criterion = nn.CrossEntropyLoss()
#
# optimizer=optim.Adam(net.parameters(), lr=0.001)

###############################################################################
# LOAD MODEL (optionally)

package_directory = os.path.dirname(os.path.abspath(__file__))

font_file = os.path.join(package_directory, 'fonts', 'myfont.ttf')

#path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
#path="models/MNIST_conv_10_conv_20_fc_100_fc_25_rel_bn_drop_trainval_modelopt1.0_epo_231_acc_99.19"
#path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
#path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

path_full=os.path.join(package_directory, path)

#net.load_state_dict(torch.load(path_full)['model_state_dict'], strict=False)

#net.load_state_dict(torch.load(path)['model_state_dict'], strict=False)
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"


########################################################
# EVALUATE

def evaluate(net2, layer):
    # print('Prediction when network is forced to predict')
    net2.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        images = images.to(device)
        #dummy works as it should, if we don't execute switch function in forward the accuracy should be original, 99.27
        #predicted_prob = net2.forward(images, "dummy")[0]  # if using switches
        #predicted_prob = net2.forward(images, "c1")[0] #13.68 for 99.27
        #predicted_prob = net2.forward(images, "c3")[0] #11.35
        predicted_prob = net2.forward(images, layer)[0]
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    accuracy = 100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

#evaluate(net2, layer)


#######################s
# LOSS



def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps):
    # BCE = f.binary_cross_entropy(prediction, true_y, reduction='sum')
    BCE = criterion(prediction, true_y)

    return BCE


###########################

def loss_functionKL(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
    # BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')
    BCE = criterion(prediction, true_y)

    # KLD term
    alpha_0 = torch.Tensor([alpha_0]).to(device)
    hidden_dim = torch.Tensor([hidden_dim]).to(device)
    trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim * alpha_0)
    trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim * torch.lgamma(alpha_0)
    trm3 = torch.sum((S - alpha_0) * (torch.digamma(S) - torch.digamma(torch.sum(S))))
    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    return BCE + annealing_rate * KLD / how_many_samps





###################################################
# RUN TRAINING

def run_experiment(epochs_num, layer, nodesNum1, nodesNum2, nodesFc1, nodesFc2, num_samps_for_switch, path):
    print("\nRunning experiment\n")
    print("Switches samples: ", num_samps_for_switch)

    #CHECK WHY THSI CHANGES SO MUCH
    net2 = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer, num_samps_for_switch).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net2.parameters(), lr=0.001)

    print(path)
    net2.load_state_dict(torch.load(path)['model_state_dict'], strict=False)


    print("Evaluate:\n")
    evaluate(net2, layer)
    # for name, param in net.named_parameters():
    #     print(name)
    #     print(param[1])
    # for name, param in net.named_parameters():
    #     print(name)
    #     #print (name, param.shape)
    #     #print("/n")
    #     if name!="parameter":
    #         param.requires_grad=False
    #     print(param.requires_grad)
    accuracy=evaluate(net2, layer)


    h = net2.c1.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.c3.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.c5.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.f6.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.c1.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.c3.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.c5.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.f6.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.bn1.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.bn1.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.bn2.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.bn2.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.f7.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    h = net2.f7.bias.register_hook(lambda grad: grad * 0)  # double the gradient

    accuracy = evaluate(net2, layer)

    #print("Retraining\n")
    net2.train()
    stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3); best_model=-1
    # while (stop<early_stopping):
    for epochs in range(epochs_num):
        epoch=epoch+1
        annealing_rate = beta_func(epoch)
        net2.train()
        evaluate(net2, layer)
        for i, data in enumerate(train_loader):
            inputs, labels=data
            inputs, labels=inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, S=net2(inputs, layer) #when switc hes
            #outputs=net2(inputs)
            #loss=criterion(outputs, labels)
            loss = loss_functionKL(outputs, labels, S, alpha_0, hidden_dim, BATCH_SIZE, annealing_rate)
            #loss=loss_function(outputs, labels, 1, 1, 1, 1)
            loss.backward()
            #print(net2.c1.weight.grad[1, :])
            #print(net2.c1.weight[1, :])
            optimizer.step()
            # if i % 100==0:
            #    print (i)
            #    print (loss.item())
            #    evaluate()
        #print (i)
        print (loss.item())
        accuracy=evaluate(net2, layer)
        print ("Epoch " +str(epoch)+ " ended.")
        # for name, param in net2.named_parameters():
        #     print(name)
        #     print(param[1])


        print("S")
        print(S)
        print(torch.argsort(S))
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))

        if (accuracy<=best_accuracy):
            stop=stop+1
            entry[2]=0
        else:
            best_accuracy=accuracy
            print("Best updated")
            stop=0
            entry[2]=1
            best_model=net2.state_dict()
            best_optim=optimizer.state_dict()
            torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
        #write
        # entry[0]=accuracy; entry[1]=loss
        # with open(filename, "a+") as file:
        #     file.write(",".join(map(str, entry))+"\n")
    return best_accuracy, epoch, best_model, S




########################################################
# PARAMS
epochs_num=10
sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)
filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)


######################################################
#single run  avergaed pver n iterations  

if __name__=='__main__':
    for i in range(1):
        with open(filename, "a+") as file:
            file.write("\nInteration: "+ str(i)+"\n")
            print("\nIteration: "+str(i))
        best_accuracy, num_epochs, best_model=run_experiment(epochs_num, layer, conv1, conv2, fc1, fc2, num_samps_for_switch, path_full)
        sum_average+=best_accuracy
        average_accuracy=sum_average/(i+1)

        with open(filename, "a+") as file:
            file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
        print("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
        #torch.save(best_model, filename_model)

    #multiple runs

    # for i1 in range(1,20):
    #     for i2 in range(1,20):
    #         with open(filename, "a+") as file:
    #             file.write("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
    #             print("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")

    #         best_accuracy, num_epochs=run_experiment(i1, i2)
    #         with open(filename, "a+") as file:
    #             file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
    #             print("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
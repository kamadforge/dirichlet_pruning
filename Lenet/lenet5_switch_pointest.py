#in forward we added a layer as the second argument
#in evaluate call function we added "c3 in forward and suddenyl we get an error that the output is tuple (which it is as it returns outptu and Sprime)
#why then before we didn't get the error when we didn't have forward as a second argument
###for this version invesrtiagtre

#feb 27
#we put the net creation inside run experiment and make sure it is evaluated and upadted on that net



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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)
print("Drop")


#############################
# PARAMS

sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
layer="c5"
how_many_epochs=200
annealing_steps = float(8000. * how_many_epochs)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
alpha_0 = 2  # below 1 so that we encourage sparsity
hidden_dim = 10 #it's a number of parameters we want to estimate, e.g. # conv1 filters
hidden_dims={'c1': conv1, 'c3': conv2, 'c5': fc1, 'f6' : fc2}
hidden_dim = hidden_dims[layer] #it's a number of parameters we want to estimate, e.g. # conv1 filters

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

##############################################################################
# NETWORK (conv-conv-fc-fc)

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer):
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

    def switch_func(self, output, Sprime):
        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])
        return output, Sprime

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

        phi = f.softplus(self.parameter)
        S = phi / torch.sum(phi) #"""directly use mean of Dir RV, which is {E} [X_{i}]={\frac {\alpha _{i}} {\sum _{k=1}^{K}\alpha _{k}}}}

        #Smax = torch.max(S)
        #Sprime = S/Smax
        Sprime = S
        if layer == 'c1':
            output, Sprime = self.switch_func(output, Sprime) #13.28 deteministic, acc increases
        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)
        if layer == 'c3':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)
        output=self.c5(output)
        if layer == 'c5':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output=self.f6(output)
        if layer == 'f6':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output = self.f7(output) #remove for 99.27
        return output, Sprime


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
criterion = nn.CrossEntropyLoss()
#
# optimizer=optim.Adam(net.parameters(), lr=0.001)

###############################################################################
# LOAD MODEL (optionally)

package_directory = os.path.dirname(os.path.abspath(__file__))

font_file = os.path.join(package_directory, 'fonts', 'myfont.ttf')

#path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
#path="models/MNIST_conv_10_conv_20_fc_100_fc_25_rel_bn_drop_trainval_modelopt1.0_epo_231_acc_99.19"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
#path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
#path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

path_full=os.path.join(package_directory, path)


########################################################
# EVALUATE

def evaluate(net2, layer):
    #print('Evaluating switches with layer:', layer )
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
        #print(correct)
    # print(str(correct) +" "+ str(total))
    # pdb.set_trace()
    accuracy = 100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

print("Loaded model:")
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


#################################


###################################################
# RUN TRAINING

def run_experiment(epochs_num, layer, nodesNum1, nodesNum2, nodesFc1, nodesFc2, path):
    print("\nRunning experiment\n")

    net2 = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net2.parameters(), lr=0.001)
    print(path)
    net2.load_state_dict(torch.load(path)['model_state_dict'], strict=False)

    print("Evaluate:\n")
    evaluate(net2, layer)
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
    h = net2.f7.bias.register_hook(lambda grad: grad * 0)  # double the gradient #change to output for 99.27

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
            loss = loss_functionKL(outputs, labels, S, alpha_0, hidden_dim, BATCH_SIZE, annealing_rate)
            loss.backward()
            #print(net2.c1.weight.grad[1, :])
            optimizer.step()
        print (loss.item())
        accuracy=evaluate(net2, layer)
        print ("Epoch " +str(epoch)+ " ended.")


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
            #torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
        #write
        # entry[0]=accuracy; entry[1]=loss
        # with open(filename, "a+") as file:
        #     file.write(",".join(map(str, entry))+"\n")
    return best_accuracy, epoch, best_model, S
        

print("\n\n NEW EXPERIMENT:\n")




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
        best_accuracy, num_epochs, best_model=run_experiment(epochs_num, layer, conv1, conv2, fc1, fc2, path_full)
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
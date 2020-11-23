# for four layer lenet:
# -loads the models
# -evaluates the model
#- prunes the model
# -retrains the model
# - computes the combinations with the pruned model


#IMPORTANT
#for 99.27 and 90.04 models remove output = self.out7(output)


import torch
from torch import nn, optim
import sys



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
import socket

from sklearn.model_selection import ParameterGrid

import argparse


#############
# params

arguments=argparse.ArgumentParser()
arguments.add_argument("--arch", default="7,10,40,20")
arguments.add_argument("--folder")
arguments.add_argument("--method", default="switch_point") #switch_itegral, swithc_point, fisher, l1, l2, random
arguments.add_argument("--switch_samps", default=150, type=int)
arguments.add_argument("--switch_comb", default='train') #train, load
arguments.add_argument("--dataset", default="mnist")
arguments.add_argument("--early_stopping", default=500, type=int)

arguments.add_argument("--resume", default=False)
arguments.add_argument("--prune_bool", default=False)
arguments.add_argument("--retrain", default=False)

arguments.add_argument("--path", default="None")

args=arguments.parse_args()
path=args.path

#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch"))
    path_compression = os.path.join(cwd, "results_compression")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_compression = cwd
    path_main= parent_path

print(cwd)
print(sys.path)


######################


from torch.nn.parameter import Parameter
import magnitude_rank
#from lenet_network_pruning_withcombinations import compute_combinations_lenet
#from lenet_network_pruning_withcombinations import get_data
from lenet5_switch_integral import run_experiment as run_experiment_integral
from lenet5_switch_pointest import run_experiment as run_experiment_pointest



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainval_perc=0.8
BATCH_SIZE = 105

dataset=args.dataset
evaluation="test"
adversarial_dataset=False

###################################################
# DATA


if dataset=="fashionmnist":

    trainval_dataset=datasets.FashionMNIST('data/FashionMNIST', train=True, download=True,
                        #transform=transforms.Compose([transforms.ToTensor(),
                        #transforms.Normalize((0.1307,), (0.3081,))]),
                        transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset=datasets.FashionMNIST('data/FashionMNIST', train=False, transform=transforms.ToTensor())

    if adversarial_dataset:
        tensor_x = torch.load('../results_adversarial/data/FashionMNIST_adversarial/tensor_x.pt')
        tensor_y = torch.load('../results_adversarial/data/FashionMNIST_adversarial/tensor_y.pt')
        tensor_x = tensor_x.unsqueeze(1)
        tensor_y=tensor_y.squeeze(1)

        test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset

    print("Loading,", dataset)

elif dataset=="mnist":

    trainval_dataset = datasets.MNIST('data/MNIST', train=True, download=True,
                                             # transform=transforms.Compose([transforms.ToTensor(),
                                             # transforms.Normalize((0.1307,), (0.3081,))]),
                                             transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transforms.ToTensor())

    print("Loading,", dataset)


# Load datasets

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    test_dataset,
    batch_size=BATCH_SIZE, shuffle=False)


##############################################################################################3####33
# ###############################################################################3##########
# NETWORK (conv-conv-fc-fc)

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



##################################


nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
criterion = nn.CrossEntropyLoss()

#optimizer=optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


######################################
#LOADING MODEL/RESUME

def load_model():
    global path
    path=args.path
    if path=="None":
        print("Please provide checkpoint path")
        sys.exit()
    else:
        net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)

    print(dataset, "loaded.")
    print(os.path.split(path)[1])
    return net



########################################################
# EVALUATE

def evaluate():
    # print('Prediction when network is forced to predict')
    net.eval()
    correct = 0
    total = 0
    if evaluation=="test":
        eval_loader=test_loader
    elif evaluation=="val":
        eval_loader=val_loader

    for j, data in enumerate(eval_loader):
        images, labels = data
        images = images.to(device)
        predicted_prob = net.forward(images)  # images.view(-1,28*28)
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    accuracy = 100 * float(correct) / total
    print("test accuracy: %.2f %%" % (accuracy))
    return accuracy




############################3
#

def train(thresh=[-1,-1,-1,-1]):
    # here retraining works
    net.train()
    stop = 0;
    epoch = 0;
    best_accuracy = 0;
    entry = np.zeros(3);
    best_model = -1;
    early_stopping = args.early_stopping
    while (stop < early_stopping):
    #for i in range(5):
        epoch = epoch + 1
        # for epoch in range(30):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # net.c1.weight.data[1] = 0  # instead of hook
            # net.c1.bias.data[1] = 0  # instead of hook

        print(loss.item())
        accuracy = evaluate()
        print("Epoch " + str(epoch) + " ended.")

        if (accuracy <= best_accuracy):
            stop = stop + 1
            entry[2] = 0
        else:
            best_accuracy = accuracy
            print("Best updated")
            stop = 0
            entry[2] = 1
            best_model = net.state_dict()
            best_optim = optimizer.state_dict()
            if save:
                if retrain:
                    if best_accuracy > save_accuracy:
                        torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim}, "%s_retrained_epo_%d_prunedto_%d_%d_%d_%d_acc_%.2f" % (
                        path, epoch, thresh[0], thresh[1], thresh[2], thresh[3], best_accuracy))
                else:
                    if best_accuracy > save_accuracy:
                        torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim}, "%s_trainval%.1f_epo%d_acc%.2f" % (
                            dataset, trainval_perc,epoch, best_accuracy))

            entry[0] = accuracy;
            entry[1] = loss
            if write_training:
                with open(filename, "a+") as file:
                    file.write("\n Epoch: %d\n" % epoch)
                    file.write(",".join(map(str, entry)) + "\n")
                    if (accuracy > 98.9):
                        file.write("Yes\n")
                    elif (accuracy > 98.8):
                        file.write("Ok\n")

    print(loss.item())
    print("Final: " + str(best_accuracy))
    accuracy = evaluate()



def finetune():

    # switch to train mode
    net.train()


    dataiter = iter(train_loader)

    for i in range(0, 100):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(train_loader)
            input, target = dataiter.next()


        input, target = input.to(device), target.to(device)

        # compute output
        output = net(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





####################################################3
# get ranks

def get_ranks(method):
    #### GET RANKS



    if method == 'random':
        combinationss = [np.random.permutation(nodesNum1), np.random.permutation(nodesNum2),
                         np.random.permutation(nodesFc1), np.random.permutation(nodesFc2)]

    elif method == 'fisher':
        finetune()
        combinationss=[]
        for i in range(4):
            fisher_rank=np.argsort(net.running_fisher[i].detach().cpu().numpy())[::-1]
            combinationss.append(fisher_rank)


    elif method=="switch_integral":

        #train or load
        getranks_method = args.switch_comb
        switch_data={}; switch_data['combinationss'] = []; switch_data['switches']=[]
        num_samps_for_switch=args.switch_samps

        print("integral evaluation")
        epochs_num = 3
        file_path=os.path.join(path_main, 'results_switch/results/switch_data_%s_9927_integral_samps_%s_epochs_%i.npy' % (dataset, str(num_samps_for_switch), epochs_num))

        if getranks_method=='train':

            for layer in ["c1", "c3", "c5", "f6"]:
                best_accuracy, epoch, best_model, S= run_experiment_integral(epochs_num, layer, 10, 20, 100, 25, num_samps_for_switch, path)
                print("Rank for switches from most important/largest to smallest after %s " %  str(epochs_num))
                print(S)
                print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
                ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
                print(",".join(map(str, ranks_sorted)))
                switch_data['combinationss'].append(ranks_sorted); switch_data['switches'].append(S.cpu().detach().numpy())



            print('*'*30)
            print(switch_data['combinationss'])
            combinationss=switch_data['combinationss']
            np.save(file_path, switch_data)

        elif getranks_method=='load':
            combinationss=list(np.load(file_path,  allow_pickle=True).item()['combinationss'])

    elif method == "switch_point":

        getranks_method = args.switch_comb
        switch_data={}; switch_data['combinationss'] = []; switch_data['switches']=[]


        print("switch mean")

        epochs_num = 3
#        file_path=os.path.join(path_main, 'results_switch/results/switch_data_%s_9927_pointest_epochs_%i.npy' % (dataset, epochs_num))

        import os
        if not os.path.exists("switches"):
            os.makedirs("switches")
        file_path='switches/switch_data_%s_9927_pointest_epochs_%i.npy' % (dataset, epochs_num)

        if getranks_method == 'train':

            for layer in ["c1", "c3", "c5", "f6"]:
                best_accuracy, epoch, best_model, S = run_experiment_pointest(epochs_num, layer, 10, 20, 100, 25, path)
                print("Rank for switches from most important/largest to smallest after %s " % str(epochs_num))
                print(S)
                print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
                ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
                print(",".join(map(str, ranks_sorted)))
                switch_data['combinationss'].append(ranks_sorted)
                switch_data['switches'].append(S.cpu().detach().numpy())

            print('*' * 30)
            print(switch_data['combinationss'])
            combinationss=switch_data['combinationss']
            np.save(file_path, switch_data)

        elif getranks_method == 'load':
            combinationss=list(np.load(file_path,  allow_pickle=True).item()['combinationss'])




    elif method == "switch_point_multiple":
        file_path=os.path.join(path_main, 'results_switch/results/combinations_multiple_9032.npy')

        combinationss =list(np.load(file_path,  allow_pickle=True))


    else:

        combinationss = magnitude_rank.get_ranks(method, net)



    return  combinationss


##################################################################################
# RETRAIN

def threshold_prune_and_retrain(combinationss, thresh):


    ##### THRESHOLD
    # the ranks are sorted from best to worst
    # thresh is what we keep, combinationss is what we discard

    for i in range(len(combinationss)):
        combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())


    #filename = "%s_retrained_paramsearch1.txt" % path
    #
    # if write:
    #     with open(filename, "a+") as file:
    #         file.write("\n\nprunedto:%d_%d_%d_%d\n\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))
    print("\n\nprunedto:%d_%d_%d_%d\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))

    print("Channels pruned: ")
    print(combinationss)

    #################################################################################################################3
    ########## PRUNE/ ZERO OUT THE WEIGHTS

    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
    #net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)

    if prune_bool:

        it = 0
        for name, param in net.named_parameters():
            print(name)
            if (("c" in name) or ("f" in name)) and ("weight" in name):
                it += 1
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if (("c" in name) or ("f" in name)) and ("bias" in name):
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if ("bn" in name) and ("weight" in name):
                param.data[combinationss[it - 1]] = 0
            if ("bn" in name) and ("bias" in name):
                param.data[combinationss[it - 1]] = 0



        # net.c1.weight.data[combination]=0; net.c1.bias.data[combination] = 0
        # net.c3.weight.data[combination2] = 0; net.c3.bias.data[combination2] = 0
        # net.c5.weight.data[combination3] = 0;net.c5.bias.data[combination3] = 0
        # net.f6.weight.data[combination4] = 0;net.f6.bias.data[combination4] = 0

        print("After pruning")
        acc=evaluate()

    ##################################################################### RETRAIN


    if retrain:
        def gradi(module):
            module[combinationss[0]]=0
        net.c1.weight.register_hook(gradi)
        net.c1.bias.register_hook(gradi)
        net.bn1.weight.register_hook(gradi)
        net.bn1.bias.register_hook(gradi)

        def gradi2(module):
            module[combinationss[1]]=0

        net.c3.weight.register_hook(gradi2)
        net.c3.bias.register_hook(gradi2)
        net.bn2.weight.register_hook(gradi2)
        net.bn2.bias.register_hook(gradi2)

        def gradi3(module):
            module[combinationss[2]] = 0
            # print(module[1])

        net.c5.weight.register_hook(gradi3)
        net.c5.bias.register_hook(gradi3)

        def gradi4(module):
            module[combinationss[3]] = 0
            # print(module[1])

        net.f6.weight.register_hook(gradi4)
        net.f6.bias.register_hook(gradi4)


        print("Retraining")

        train(thresh)

    return acc



#######################

#SAVING MODEL
#the models are saved in the savedirectory as the original model
if dataset=="mnist":
    save_accuracy=98.6
if dataset=="fashionmnist":
    save_accuracy=89.50

save=True
#WRITING
# the output text file will be saved also in the same directory as the original model
write_training=False
#################################

resume=args.resume
prune_bool=args.prune_bool
retrain=args.retrain


##############################

file_write=False
comp_combinations=False

#################################3################
################################################



if resume:
    load_model()
    evaluate()
    #if comp_combinations:
    #    compute_combinations_lenet(False, net, evaluate, dataset, "zeroing") #can be "additive noise instead of zeroing

    methods=[args.method]

    if prune_bool:

        pruned_arch_layer=[int(n) for n in args.arch.split(",")]
        pruned_arch={}
        pruned_arch['c1']=pruned_arch_layer[0]; pruned_arch['c3']=pruned_arch_layer[1]; pruned_arch['c5']=pruned_arch_layer[2];pruned_arch['f6']=pruned_arch_layer[3];

        if 1:
            accs={}
            for method in methods:
                print("\n\n %s \n" % method)
                combinationss = get_ranks(method); print(combinationss)
                acc=threshold_prune_and_retrain(combinationss, [pruned_arch['c1'], pruned_arch['c3'], pruned_arch['c5'], pruned_arch['f6']])
                accs[method]=acc
                #prune(False, i1, i2, i3, i4, write, save)

            print("\n*********************************\n\n")

if resume==False and prune_bool==False and retrain==False:
    train()


print("\n\nEND")

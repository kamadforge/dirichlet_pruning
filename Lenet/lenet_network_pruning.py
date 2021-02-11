# for four layer lenet:
# -loads the models
# -evaluates the model
#- prunes the model
# -retrains the model
# - computes the combinations with the pruned model

#IMPORTANT
#for 99.27 and 90.04 models remove output = self.out7(output)


import sys
import torch
from torch import nn, optim
import torch.utils.data
import numpy as np
import os
import socket
import argparse

#############
# PARAMS

arguments=argparse.ArgumentParser()
arguments.add_argument("--arch", default="7,10,40,20")
arguments.add_argument("--folder")
arguments.add_argument("--method", default="switch_point") #switch_itegral, swithc_point, fisher, l1, l2, random
arguments.add_argument("--switch_samps", default=150, type=int)
arguments.add_argument("--switch_comb", default='train') #train, load
arguments.add_argument("--dataset", default="mnist")
arguments.add_argument("--early_stopping", default=500, type=int)
arguments.add_argument("--batch_size", default=105, type=int)
arguments.add_argument("--trainval_perc", default=0.8, type=float)

arguments.add_argument("--resume", default=False)
arguments.add_argument("--prune_bool", default=False)
arguments.add_argument("--retrain", default=False)

arguments.add_argument("--path_checkpoint_load")
arguments.add_argument("--path_checkpoint_save", default="checkpoint")

args=arguments.parse_args()
print(args)

evaluation="test"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset=args.dataset
os.makedirs(args.path_checkpoint_save, exist_ok=True)
path_checkpoint_save_scratch= os.path.join(args.path_checkpoint_save, "scratch", dataset)
os.makedirs(path_checkpoint_save_scratch, exist_ok=True)
path_checkpoint_save_retrain = os.path.join(args.path_checkpoint_save, "retrain", dataset)
os.makedirs(path_checkpoint_save_retrain, exist_ok=True)

######################### p

################
# path stuff

cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch_new"))
    path_compression = os.path.join(cwd, "results_compression")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch_new"))
    path_compression = cwd
    path_main= parent_path

print(f"Current working directory is: {cwd}")

#import parent module
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from methods import magnitude_rank
#from lenet_network_pruning_withcombinations import compute_combinations_lenet
#from lenet_network_pruning_withcombinations import get_data
from methods.lenet5_switch_integral import run_experiment as run_experiment_integral
from methods.lenet5_switch_pointest import run_experiment as run_experiment_pointest
from importlib.machinery import SourceFileLoader
dataset_mnist = SourceFileLoader("module_mnist", "../dataloaders/dataset_mnist.py").load_module()
dataset_fashionmnist = SourceFileLoader("module_fashionmnist", "../dataloaders/dataset_fashionmnist.py").load_module()
model_lenet5 = SourceFileLoader("module_lenet", "../models/lenet5.py").load_module()
from module_fashionmnist import load_fashionmnist
from module_mnist import load_mnist
from module_lenet import Lenet

###################################################
# DATA

if dataset=="fashionmnist":
    train_loader, test_loader, val_loader = load_fashionmnist(args.batch_size, args.trainval_perc)
elif dataset=="mnist":
    train_loader, test_loader, val_loader = load_mnist(args.batch_size, args.trainval_perc)

###################################################################
# NETWORK (conv-conv-fc-fc)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
criterion = nn.CrossEntropyLoss()
#optimizer=optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

######################################
#LOADING MODEL/RESUME

def load_model(path_checkpoint_load):

    print("Loading from... ", path_checkpoint_load)
    if path_checkpoint_load is None:
        print("The checkpoint not indicated, looking for the existing checkpoints.")
        checkpoints = os.listdir(path_checkpoint_save_scratch)
        # if there are no checkpoints from previous training
        if len(checkpoints)==0:
            print("No checkpoints found.")
            sys.exit()
        else:
            # take the checkpoint with the highest accoracy
            checkpoint_accuracies = []
            for element in checkpoints:
                checkpoint_accuracies.append(float(element[-5:]))
            argsorted = np.argsort(checkpoint_accuracies)
            path_checkpoint_load = os.path.join(path_checkpoint_save_scratch, checkpoints[argsorted[-1]])

    net.load_state_dict(torch.load(path_checkpoint_load, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)

    print("\nModel loaded:")
    print(os.path.split(path_checkpoint_load)[1])
    return net, path_checkpoint_load



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
    net, path_checkpoint_load_ret = load_model(args.path_checkpoint_load)
    acc = evaluate()
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

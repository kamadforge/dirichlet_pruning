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
arguments.add_argument("--arch", default="6,7,34,16")
arguments.add_argument("--folder")
arguments.add_argument("--method", default="shapley") #switch_itegral, swithc_point, fisher, l1, l2, random
arguments.add_argument("--switch_samps", default=150, type=int)
arguments.add_argument("--switch_comb", default='train') #train, load
#shapley
arguments.add_argument("--shap_method", default="combin")
arguments.add_argument("--load_file", default=1, type=int)
arguments.add_argument("--k_num", default=None)
arguments.add_argument("--shap_sample_num", default=2, type=int)

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
print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
from methods import shapley_rank
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

def evaluate(net=net, evaluation="test"):
    print(f'\nEvaluating model on {evaluation} dataset')
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
    old_checkpoint=""
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
            # print(net.c1.weight.data[1])
            # print(net.c1.weight.grad[1])

        print(f"Loss: {loss.item()}")
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
                if best_accuracy > save_accuracy:
                    if retrain:
                        save_path = f"{path_checkpoint_save_retrain}_retrained_epo_{epoch}_prunedto_{thresh[0]}_{thresh[1]}_{thresh[2]}_{thresh[3]}_acc_{best_accuracy}"
                        torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim}, save_path)
                        print(f"Saving checkpoint to {save_path}")
                    else:
                        save_path = f"{path_checkpoint_save_scratch}/{dataset}_trainval_{args.trainval_perc}_epo_{epoch}_acc_{best_accuracy}"
                        torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim}, save_path)
                        print(f"Saving checkpoint to {save_path}")

                    if os.path.isfile(old_checkpoint):
                        os.remove(old_checkpoint)
                    old_checkpoint = save_path


            entry[0] = accuracy;
            entry[1] = loss
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

#input : method, path to the model we want to rank
#output : rank (list of four numpy arrays, one for each layer)


def get_ranks(method, path_checkpoint):
    print(f"Ranking method {method}")

    if method == 'random':
        combinationss = [np.random.permutation(nodesNum1), np.random.permutation(nodesNum2),
                         np.random.permutation(nodesFc1), np.random.permutation(nodesFc2)]

    elif method == 'fisher':
        finetune()
        combinationss=[]
        for i in range(4):
            fisher_rank=np.argsort(net.running_fisher[i].detach().cpu().numpy())[::-1]
            combinationss.append(fisher_rank)

    elif method == 'shapley':
        load_file = args.load_file

        try:
            combinationss = shapley_rank.shapley_rank(evaluate, net, "Lenet", os.path.split(path_checkpoint)[1], dataset, load_file, args.k_num, args.shap_method, args.shap_sample_num)
        except KeyboardInterrupt:
            print('Interrupted')
            shapley_rank.file_check()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


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
        epochs_num = 1
        path_switches = "../methods/switches/Lenet"
        if getranks_method == 'train':
            for layer in ["c1", "c3", "c5", "f6"]:
                print(f"\nLayer: {layer}")
                best_accuracy, epoch, best_model, S = run_experiment_pointest(epochs_num, layer, 10, 20, 100, 25, path_checkpoint, args)
                print("Rank for switches from most important/largest to smallest after %s " % str(epochs_num))
                print(S)
                print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
                ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
                print(",".join(map(str, ranks_sorted)))
                switch_data['combinationss'].append(ranks_sorted)
                switch_data['switches'].append(S.cpu().detach().numpy())
            print(switch_data['combinationss'])
            combinationss = switch_data['combinationss']
            # save switches
            if not os.path.exists(path_switches):
                os.makedirs(path_switches)
            file_path = os.path.join(path_switches, f"switches_{dataset}_{epochs_num}_{path_checkpoint[-5:]}.npy")
            np.save(file_path, switch_data)
        elif getranks_method == 'load':
            switches_files = os.listdir(path_switches)
            for file in switches_files:
                if (file[-9:-4] == path_checkpoint[-5:]):
                    path_switches_file = os.path.join(path_switches, file)
                    combinationss = list(np.load(path_switches_file, allow_pickle=True).item()['combinationss'])

    elif method == "switch_point_multiple":
        file_path=os.path.join(path_main, 'results_switch/results/combinations_multiple_9032.npy')
        combinationss =list(np.load(file_path,  allow_pickle=True))

    else:
        combinationss = magnitude_rank.get_ranks(method, net)

    return  combinationss


##################################################################################
# RETRAIN
def threshold_prune_and_retrain(combinationss, thresh):
    '''
    PRUNE/ ZERO OUT THE WEIGHTS
    the ranks are sorted from best to worst
    thresh is what we keep, combinationss is what we discard
    '''
    for i in range(len(combinationss)):
        combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())
    print("\n\nPrunedto:%d_%d_%d_%d\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))
    print("Channels pruned: ")
    #print(combinationss)

    ######################################################################################
    # PRUNE
    if prune_bool:
        def zero_param():
            it = 0
            for name, param in net.named_parameters():
                #print(name)
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
                if ("bn" in name) and ("running_mean" in name):
                    param.data[combinationss[it - 1]] = 0
                if ("bn" in name) and ("running_var" in name):
                    param.data[combinationss[it - 1]] = 0

        zero_param()
        print("After pruning")
        acc=evaluate()

    #####################################################################
    # RETRAIN

    print("Gradient")
    if retrain:
        def gradi_new(combs_num):
            def hook(module):
                module[combinationss[combs_num]] = 0
            return hook
        net.c1.weight.register_hook(gradi_new(0))
        net.c1.bias.register_hook(gradi_new(0))
        net.bn1.weight.register_hook(gradi_new(0))
        net.bn1.bias.register_hook(gradi_new(0))
        net.c3.weight.register_hook(gradi_new(1))
        net.c3.bias.register_hook(gradi_new(1))
        net.bn2.weight.register_hook(gradi_new(1))
        net.bn2.bias.register_hook(gradi_new(1))
        net.c5.weight.register_hook(gradi_new(2))
        net.c5.bias.register_hook(gradi_new(2))
        net.f6.weight.register_hook(gradi_new(3))
        net.f6.bias.register_hook(gradi_new(3))
        print("Retraining")
        train(thresh)

    return acc

#######################
# SAVING MODEL
# the models are saved in the savedirectory as the original model
if dataset == "mnist":
    save_accuracy = 98.0
if dataset == "fashionmnist":
    save_accuracy = 80.50

print(f"Checkpoint saving accuracy: {save_accuracy}")

save = True

resume = args.resume
prune_bool = args.prune_bool
retrain = args.retrain

#######


if resume:
    net, path_checkpoint_load_ret = load_model(args.path_checkpoint_load)
    acc = evaluate()

    if prune_bool:

        # parse number of channels to prune at each layer
        pruned_arch_layer = [int(n) for n in args.arch.split(",")]
        pruned_arch = {}
        pruned_arch['c1'] = pruned_arch_layer[0];
        pruned_arch['c3'] = pruned_arch_layer[1];
        pruned_arch['c5'] = pruned_arch_layer[2];
        pruned_arch['f6'] = pruned_arch_layer[3];

        accs = {}
        methods = [args.method]
        for method in methods:
            print("\n\n %s \n" % method)
            combinationss, combinationss_dic = get_ranks(method, path_checkpoint_load_ret);
            for comb in combinationss:
                print(comb)
            acc = threshold_prune_and_retrain(combinationss, [pruned_arch['c1'], pruned_arch['c3'], pruned_arch['c5'],pruned_arch['f6']])
            accs[method] = acc

if resume == False and prune_bool == False and retrain == False:
    print("\nTraining from scratch\n")
    train()

print("\n\nEND")

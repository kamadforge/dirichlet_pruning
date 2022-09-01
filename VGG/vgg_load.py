# 1. It trains the network from scratch (if "resume" is off)
# 2. if resume is on, it loads the pretrained model,
#     if prune_ bool , it prunes the network
#     if retrain_ bool is on , retrains it
# e.g. it can retrain, but not prune
# 3. it can be used for visualizing, if uncomment the comments #VISU

# other features:
# it loads the ranks from shapley or switches (ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/')

from __future__ import print_function
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import socket
import numpy as np
import torch.nn.functional as f
import matplotlib.pyplot as plt
#import parent module
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from methods import magnitude_rank
#from vgg_computeComb2 import compute_combinations
import argparse
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
dataset_cifar = SourceFileLoader("module_cifar", "../dataloaders/dataset_cifar.py").load_module()
model_lenet5 = SourceFileLoader("module_vgg", "../models/vgg.py").load_module()
from module_cifar import load_cifar
from methods.script_vgg_vggswitch import switch_run as script_vgg
from methods import shapley_rank
from models import vgg



#######
# PATH

cwd = os.getcwd()
#if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
#    #the cwd is where the sub file is so ranking/
#    sys.path.append(os.path.join(cwd, "results_switch"))
#    path_compression = cwd
#    #path_compression = os.path.join(cwd, "results_compression")
#    path_networktest = os.path.join(cwd, "results_networktest")
#    path_switch = os.path.join(cwd, "results_switch")
#    path_main= cwd
#    print("path_main: ", path_main)
if 1:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_compression = cwd
    path_networktest = os.path.join(parent_path, "results_networktest")
    path_switch = os.path.join(parent_path, "results_switch")
    path_main= parent_path
    print("path_main: ", path_main)

sys.path.append(os.path.join(path_networktest, "external_codes/pytorch-cifar-master/models"))
sys.path.append(path_compression)


##############################
# PARAMETERS

parser = argparse.ArgumentParser()
# parser.add_argument("--arch", default='25,25,65,80,201,158,159,460,450,490,470,465,465,450')
parser.add_argument("--pruned_arch", default='34,34,60,60,70,101,97,88,95,85,86,67,61,55,55')
#parser.add_argument("--pruned_arch", default='')

#parser.add_argument("--arch", default='25,25,65,80,201,158,159,460,450,490,470,465,465,450')
# ar.add_argument("-arch", default=[21,20,65,80,201,147,148,458,436,477,454,448,445,467,441])
parser.add_argument('--layer', help="layer to prune", default="None")
parser.add_argument("--method", default='shapley') #switch, l1, l2
parser.add_argument("--dataset", default="cifar")
parser.add_argument("--trainval_perc", default=0.8, type=float)

#Dirichlet
parser.add_argument("--switch_samps", default=3, type=int)
parser.add_argument("--switch_epochs", default=1, type=int)
parser.add_argument("--ranks_method", default='point') #point, integral
parser.add_argument("--switch_trainranks", default=1, type=int)
#shapley
parser.add_argument("--shap_method", default="kernel")
parser.add_argument("--load_file", default=0, type=int)
parser.add_argument("--k_num", default=10)
parser.add_argument("--shap_sample_num", default=2, type=int)
parser.add_argument("--adding", default=0, type=int)
#general
parser.add_argument("--resume", default=1, type=int)
parser.add_argument("--prune_bool", default=1, type=int)
parser.add_argument("--retrain_bool", default=False, type=int)
parser.add_argument("--model", default="None")
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
save_accuracy=65.0
print(f"Save accuracy: {save_accuracy}\n")
os.makedirs("checkpoint", exist_ok=True)
args = parser.parse_args()
print(args)
#print(args.layer)
print(torch.cuda.get_device_name(torch.cuda.current_device()))


# DATA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
trainloader, testloader, valloader = load_cifar(args.trainval_perc)

###################################################
# MAKE AN INSTANCE OF A NETWORK AND (POSSIBLY) LOAD THE MODEL

arch = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]

print('==> Building model..')
net = vgg.VGG(arch)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(device)


#######################################################
#TRAIN


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # outputs = net(inputs, batch_idx) #VISU
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if (batch_idx % 1000 ==0):
    print('Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100. * correct / total, best_acc


###############################################################

def finetune(net=net):
    # switch to train mode
    net.train()

    net.to(device)

    dataiter = iter(trainloader)

    for i in range(0, 100):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        input, target = input.to(device), target.to(device)

        # compute output
        output = net(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#################################################################
# TEST

def test(dataset):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs, batch_idx)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


def testval(net=net, mode="val"):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if mode == "val":
        evalloader = valloader
    else:
        evalloader = testloader
    print(f"Evaluating on {mode} dataset")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


########################################
## just LOAD MODEL AND SAVE




def load_model(test_bool=True, net=net):
    # Load checkpoint.
    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model2load)
    # checkpoint = torch.load('./checkpoint/ckpt_vgg16_prunedto[39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62]_64.55.t7')
    net.load_state_dict(checkpoint['net'], strict=False)
    # print(net.module.c1.weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if test_bool:
        print("Accuracy of the tested model: ")

        test(-1)
    print("----")
    # testval(-1)
    # test_val(-1, net)


######################################################
# SAVE experiment
global old_checkpoint
old_checkpoint=""

def save_checkpoint(epoch, acc, best_acc, remaining=0):
    global old_checkpoint
    # Save checkpoint.
    # acc = test(epoch)
    if acc > best_acc:
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        print("acc: ", acc)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if acc > save_accuracy:
            print(f'Saving.. to {path_compression}/checkpoint/')
            if args.pruned_arch == 0:  # regular training
                file_path_save = path_compression+'/checkpoint/ckpt_vgg16_trainval_{}_acc_{}.t7'.format(args.trainval_perc, acc)
                torch.save(state, file_path_save)
            else:
                file_loaded = os.path.split(model2load)[1]
                file_path_save = path_compression+f'/checkpoint/{file_loaded}_pruned{args.pruned_arch}_acc_{acc}.t7'
                torch.save(state, file_path_save)

            if os.path.isfile(old_checkpoint):
                os.remove(old_checkpoint)
            old_checkpoint = file_path_save

        best_acc = acc
    return best_acc


##############################################
# PRUNE and RETRAIN
def prune_and_retrain(thresh):
    load_model(False)
    if prune_bool:
        ############################3
        # READ THE RANKS
        print("\nPruning the model\n")
        print("architecture for pruning: ", args.pruned_arch)
        if method == 'switch':
            epochs_num=args.switch_epochs
            num_samps_for_switch = args.switch_samps
            ranks_method=args.ranks_method
            #path =
            ########################################################
            # if ranks_method == 'shapley':
            #     combinationss = []
            #     shapley_file = open(
            #         "/home/user/Dropbox/Current_research/python_tests/results_shapley/combinations/94.34/zeroing_0.2val/shapley.txt")
            #     for line in shapley_file:
            #         line = line.strip()[1:-2]
            #         nums = line.split(",")
            #         nums_int = [int(i) for i in nums]
            #         combinationss.append(nums_int)
            #######################################################
            if ranks_method == 'integral':
                print(ranks_method)
                if args.switch_trainranks:
                    print("\nTraining switches\n")
                    ranks = script_vgg("switch_" + ranks_method, epochs_num, path_main, args.dataset, num_samps_for_switch)
                    combinationss = ranks['combinationss']
                else:
                    print("\nLoading switches\n")
                    ranks_path = path_main+"/methods/switches/VGG/integral/switch_data_cifar_integral_samps_%i_epochs_%i.npy" % (args.switch_samps, args.switch_epochs)
                    combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])
            #######################################################
            elif ranks_method == 'point':
                print(ranks_method)
                if args.switch_trainranks:
                    print("Training switches\n")
                    ranks = script_vgg("switch_"+ranks_method, epochs_num, path_main, args.dataset)
                    combinationss = ranks['combinationss']
                else:
                    print("Loading switches")
                    ranks_path = path_main+'/methods/switches/VGG/point/switch_data_cifar_point_epochs_%i.npy' % (args.switch_epochs)
                    combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])
                # these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())

        #################################################################
        elif method == 'l1' or method == 'l2':
            magnitude_rank.setup()
            combinationss = magnitude_rank.get_ranks(method, net)
            # the numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())
            print(combinationss[1])

        ###############
        elif method == 'shapley':
            try:
                combinationss, rank_dic = shapley_rank.shapley_rank(testval, net, "VGG", os.path.split(model2load)[1], args.dataset, args.load_file, args.k_num, args.shap_method, args.shap_sample_num, args.adding)
            except KeyboardInterrupt:
                print('Interrupted')
                shapley_rank.file_check(args.shap_method)
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
            combinationss.pop()

            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())

        ####################################################################
        elif method == 'fisher':
            # in the process of finetuning we accumulate the gradient information that w eadd for each batch. We use this gradient info for constructing a ranking.
            from methods.fisher_vgg import VGG_fisher
            net_fisher = VGG_fisher(arch)
            load_model(False, net_fisher)

            net_fisher.reset_fisher() #.net_fisher.module.reset_fisher
            finetune(net_fisher)
            combinationss = []
            for i in range(14):
                fisher_rank = torch.argsort(net_fisher.running_fisher[i], descending=True)
                combinationss.append(fisher_rank.detach().cpu())
            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:])
            print(f"E.g., cut off {len(combinationss[1])} channels from the 2nd layer: {combinationss[1]}")

        # # PRINT THE PRUNED ARCHITECTURE
        # remaining = []
        # for i in range(len(combinationss)):
        #     print(arch[i], len(combinationss[i]))
        #     remaining.append(int(arch[i]) - len(combinationss[i]))
        # print(remaining)

        #############
        # PRUNE

        def zero_params():
            it = 0
            for name, param in net.state_dict().items():
                #print(name, param.shape)
                if ("module.c" in name or "module.l1" in name) and "weight" in name:
                    it += 1
                    #print(len(combinationss[it-1]))
                    bad_channels = combinationss[it - 1]
                    param.data[bad_channels] = 0
                    # print(param.data)
                if ("module.c" in name or "module.l1" in name) and "bias" in name:
                    param.data[bad_channels] = 0
                    # print(param.data)
                if ("bn" in name) and ("weight" in name):
                    param.data[bad_channels] = 0
                if ("bn" in name) and ("bias" in name):
                    param.data[bad_channels] = 0
                if ("bn" in name) and ("running_mean" in name):
                    param.data[bad_channels] = 0
                if ("bn" in name) and ("running_var" in name):
                    param.data[bad_channels] = 0

        zero_params()

        print("After pruning")
        test(-1)

        ######################
        # GRAD
        print("Gradients for retraining")
        # def gradi1(module):
        #     module[combinationss[0]] = 0
        #     # print(module[21])
        def gradi_new(combs_num):
            def hook(module):
                module[combinationss[combs_num]] = 0
            return hook
        net.module.c1.weight.register_hook(gradi_new(0))
        net.module.c1.bias.register_hook(gradi_new(0))
        net.module.bn1.weight.register_hook(gradi_new(0))
        net.module.bn1.bias.register_hook(gradi_new(0))
        net.module.c2.weight.register_hook(gradi_new(1))
        net.module.c2.bias.register_hook(gradi_new(1))
        net.module.bn2.weight.register_hook(gradi_new(1))
        net.module.bn2.bias.register_hook(gradi_new(1))
        net.module.c3.weight.register_hook(gradi_new(2))
        net.module.c3.bias.register_hook(gradi_new(2))
        net.module.bn3.weight.register_hook(gradi_new(2))
        net.module.bn3.bias.register_hook(gradi_new(2))
        net.module.c4.weight.register_hook(gradi_new(3))
        net.module.c4.bias.register_hook(gradi_new(3))
        net.module.bn4.weight.register_hook(gradi_new(3))
        net.module.bn4.bias.register_hook(gradi_new(3))
        h1 = net.module.c5.weight.register_hook(gradi_new(4))
        h1 = net.module.c5.bias.register_hook(gradi_new(4))
        h12 = net.module.bn5.weight.register_hook(gradi_new(4))
        h13 = net.module.bn5.bias.register_hook(gradi_new(4))
        h1 = net.module.c6.weight.register_hook(gradi_new(5))
        h1 = net.module.c6.bias.register_hook(gradi_new(5))
        h12 = net.module.bn6.weight.register_hook(gradi_new(5))
        h13 = net.module.bn6.bias.register_hook(gradi_new(5))
        h1 = net.module.c7.weight.register_hook(gradi_new(6))
        h1 = net.module.c7.bias.register_hook(gradi_new(6))
        h12 = net.module.bn7.weight.register_hook(gradi_new(6))
        h13 = net.module.bn7.bias.register_hook(gradi_new(6))
        h1 = net.module.c8.weight.register_hook(gradi_new(7))
        h1 = net.module.c8.bias.register_hook(gradi_new(7))
        h12 = net.module.bn8.weight.register_hook(gradi_new(7))
        h13 = net.module.bn8.bias.register_hook(gradi_new(7))
        h1 = net.module.c9.weight.register_hook(gradi_new(8))
        h1 = net.module.c9.bias.register_hook(gradi_new(8))
        h12 = net.module.bn9.weight.register_hook(gradi_new(8))
        h13 = net.module.bn9.bias.register_hook(gradi_new(8))
        h1 = net.module.c10.weight.register_hook(gradi_new(9))
        h1 = net.module.c10.bias.register_hook(gradi_new(9))
        h12 = net.module.bn10.weight.register_hook(gradi_new(9))
        h13 = net.module.bn10.bias.register_hook(gradi_new(9))
        h1 = net.module.c11.weight.register_hook(gradi_new(10))
        h1 = net.module.c11.bias.register_hook(gradi_new(10))
        h12 = net.module.bn11.weight.register_hook(gradi_new(10))
        h13 = net.module.bn11.bias.register_hook(gradi_new(10))
        h1 = net.module.c12.weight.register_hook(gradi_new(11))
        h1 = net.module.c12.bias.register_hook(gradi_new(11))
        h12 = net.module.bn12.weight.register_hook(gradi_new(11))
        h13 = net.module.bn12.bias.register_hook(gradi_new(11))
        h1 = net.module.c13.weight.register_hook(gradi_new(12))
        h1 = net.module.c13.bias.register_hook(gradi_new(12))
        h12 = net.module.bn13.weight.register_hook(gradi_new(12))
        h13 = net.module.bn13.bias.register_hook(gradi_new(12))
        h1 = net.module.l1.weight.register_hook(gradi_new(13))
        h1 = net.module.l1.bias.register_hook(gradi_new(13))
        h12 = net.module.l1.weight.register_hook(gradi_new(13))
        h13 = net.module.l1.bias.register_hook(gradi_new(13))

    #######################################################
    # RETRAIN
    if retrain_bool:
        print("\nRetraining\n")
        net.train()
        stop = 0;
        epoch = 0;
        best_accuracy = 0;
        early_stopping = 500
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        while (stop < early_stopping):
            epoch = epoch + 1
            for i, data in enumerate(trainloader):
                inputs, labels = data;
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                #net.module.c2.weight.grad  #  just check the gradient
                optimizer.step()
                # net.c1.weight.data[1] = 0  # instead of hook
                # net.c1.bias.data[1] = 0  # instead of hook

                if args.prune_bool:
                    zero_params()

            print(loss.item())
            accuracy = test(-1)
            # print(net.module.c2.weight.data)
            print("Epoch " + str(epoch) + " ended.")
            if (accuracy <= best_accuracy):
                stop = stop + 1
            else:
                if accuracy > 50.5:
                    # compares accuracy and best_accuracy by itself again
                    best_accuracy = save_checkpoint(epoch, accuracy, best_accuracy, args.pruned_arch)
                print("Best updated")
                stop = 0
        print(loss.item())
        accuracy = test(-1)

#################################################################
# MAIN
os.makedirs('checkpoint', exist_ok=True)
if args.model=="None":
    model2load = path_compression+'/checkpoint/ckpt_vgg16_94.34.t7'
else:
    model2load = args.model
orig_accuracy = 94.34
# if all False just train the network
resume = args.resume
prune_bool = args.prune_bool
retrain_bool = args.retrain_bool  # whether we retrain the model or just evaluate
comp_combinations = False  # must be with resume #with retrain if we want to retrain combinations
vis = False
if resume:
    load_model()
# training from scratch
else:
    print("\nTraining from scratch\n")
    best_accuracy = 0
    session1end = start_epoch + 10;
    session2end = start_epoch + 250;
    session3end = start_epoch + 3250;  # was til 550
    for epoch in range(start_epoch, session1end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session1end, session2end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session2end, session3end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)


# loading a pretrained model
# if prune_bool:
    # thresh=[15,15,10,10,10,110,210,490,490,497,505,505,504,503,495]
    # thresh=[15,15,10,10,10,110,21,49,490,497,505,505,504,503,495]
    # thresh=[30,30,70,80,201,170,175,420,430,440,440,445,445,450,450]
    # thresh=[30,30,60,60,181,150,155,420,410,420,420,445,445,450,450]
    # thresh=[20,20,30,90,181,150,155,320,310,320,320,445,445,450,50]
    # thresh=[15,15,24,10,141,150,195,220,210,220,220,345,345,350,350]
    # thresh=args['arch']
    # thresh=[20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 160, 160, 160, 80]
    # thresh = [20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 80, 80, 80, 80]
    # thresh = [5, 5, 40, 40, 20, 40, 120, 230, 250, 300, 300, 160, 250, 250, 160]  # 10 #0.3
    # thresh=[5, 5, 40, 40, 20, 40, 80, 130, 190, 260, 260, 160, 250, 250, 160] #11 #0.4
    # thresh=[5, 5, 40, 40, 20, 40, 80, 80, 160, 40, 40, 160, 80, 160, 160] #12 #0.5 %17.81
    # thresh=[5, 5, 10, 10, 40, 20, 20, 40, 40, 160, 160, 40, 160, 80, 80] # 13
    # thresh=[5, 5, 20, 10, 20, 80, 40, 40, 40, 80, 160, 80, 80, 40, 80] #14 #0.6 #10.74 (94.34)
    # thresh = [5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 80, 160, 80] #15
    # thresh = [5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 40, 40,160]  # 16 55.86. 69.81, 58.49, 57.11 (fish, filt, l1, l2) (94.34
    # thresh=[5, 5, 10, 10, 20, 10, 20, 20, 40, 20, 20, 40, 40, 20, 80] #17 # 87.86. 71.44, 58.49, 57.21 (94.34)
    # thresh=[5, 5, 10, 10, 10, 10, 10, 20, 20, 20, 10, 10, 10, 10, 10] #~18 #0.95

if retrain_bool or prune_bool:
    print('\n****************\n')
    for method in [args.method]:
        # for method in ['fisher']:
        print('\n\n' + method + "\n")
        thresh = [int(n) for n in args.pruned_arch.split(",")]
        #[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        print(thresh)
        prune_and_retrain(thresh)

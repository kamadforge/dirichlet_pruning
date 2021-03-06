from models.lenet5 import Lenet
import torch
import numpy as np
from itertools import chain, combinations
import os
from sklearn.linear_model import LinearRegression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shapley_rank(evaluate, net, net_name, checkpoint_name, dataset, file_load, k_num, method, sample_num, criterion="dummy"):
    path_file = "sv/Lenet/combinations"
    print("Computing Shapley rank in two stages")
    acc = evaluate(net, "test")
    # compute combinations/ characteristic function

    os.makedirs(f"../methods/sv/{net_name}/combinations", exist_ok=True)
    os.makedirs(f"../methods/sv/{net_name}/randomsvs", exist_ok=True)
    shap_ranks=[]
    for layer_name, param in net.named_parameters():
        if "weight" in layer_name and "bn" not in layer_name and "out" not in layer_name:
            if not net_name == "Resnet" or (net_name == "Resnet" and "layer" in layer_name):
                global file_name, file_name_new
                file_name = f"../methods/sv/{net_name}/combinations/combinations_pruning_{checkpoint_name}_{layer_name}"
                file_name_new = file_name + "_new.txt"
                file_old = file_name + ".txt"

                if method == "kern":
                    shap_arr = kernshap(True, net, net_name, layer_name, evaluate, dataset, k_num, param, sample_num, "zeroing")
                    print("shaps\n", shap_arr)

                if method == "random":
                    if not file_load:
                        shap_arr = randomshap(True, net, net_name, layer_name, evaluate, dataset, k_num, param, sample_num, "zeroing")
                    else:
                        shap_arr = file_read("random", net_name, layer_name)

                if method == "compcomb":
                    if not file_load:
                        compute_combinations_lenet(True, net, net_name, layer_name, evaluate, dataset, k_num, "zeroing")

                        sample_num = 50
                        dic, nodes_num = readdata_notsampled(file_old, acc)
                        #compute the shapley value from the combinations
                        shap_arr = shapley_samp(dic, nodes_num, 2000)

                if method == "exact_partial":
                    if not file_load:
                        compute_combinations_lenet(True, net, net_name, layer_name, evaluate, dataset, k_num, "zeroing")
                    shap_arr = exact_partial(dic, nodes_num, acc)


                shap_rank = np.argsort(shap_arr)[::-1]
                shap_ranks.append(shap_rank)
    return shap_ranks


def file_read(meth, net_name, layer):
    samples_most=0
    for fname in os.listdir(f'../methods/sv/{net_name}/randomsvs'):
        core_name = f"{meth}shap_{layer}_samp_"
        if core_name in fname:
            samp_num_temp = fname.replace(core_name, "")
            samp_num = samp_num_temp.replace(".npy", "")
            samples_num = int(samp_num)
            if samples_num>samples_most:
                samples_most = samples_num
    randsvs = np.load(f"../methods/sv/{net_name}/randomsvs/{meth}shap_{layer}_samp_{samples_most}.npy")
    return randsvs


def file_check():
    # check if new results have more lines than the previous one
    file_old = file_name + ".txt"
    file_new = file_name + "_new.txt"
    if os.path.exists(file_old):
        num_lines_old = sum(1 for line in open(file_old, "r"))
        num_lines_new = sum(1 for line in open(file_new, "r"))
        if num_lines_old > num_lines_new:
            os.remove(file_new)
        else:
            os.remove(file_old)
            os.rename(file_new, file_old)
    else:
        os.rename(file_new, file_old)


# taken form ranking/results_compression/lenet_network_pruning_withcombinations.py
def compute_combinations_lenet(file_write, net, net_name, layer, evaluate, dataset, k_num, perturbation_method):
    print("1. Computing combinations")

    acc = evaluate(net, "test")
    print("from other")
    # for name, param in net.named_parameters():
    #     print(name)
    for name, param in net.named_parameters():
        print("Working on the layer: ", layer)
        # find a layer (weight and bias) where we compute rank

        if layer in name:
            if file_write:
                with open(file_name_new, "a+") as textfile:
                    textfile.write(str(param.shape[0])+"\n")
            if net_name is not "Resnet":
                layerbias = layer[:-6] + "bias" #:3 for lenet
                params_bias = net.state_dict()[layerbias]
            all_results = {}
            # get s and r to compute the (s choose r)
            s = torch.arange(0, param.shape[0])  # list from 0 to 19 as these are the indices of the data tensor
            # get the alternating elements in the channel list to have the most combinations from the beginning and end first
            a = np.arange(1, param.shape[0])
            channel_list = [a[-i // 2] if i % 2 else a[i // 2] for i in range(len(a))]
            channel_list=channel_list[:] if k_num==None else channel_list[:]
            #for r in range(1, param.shape[0]):  # produces the combinations of the elements in s
            for r in channel_list:
                print(r)
                results = []
                for combination in list(combinations(s, r)):
                    combination = torch.LongTensor(combination)
                    print(combination)
                    # save current values in a placeholder
                    params_saved = param[combination].clone();
                    if net_name is not "Resnet":
                        param_bias_saved = params_bias[combination].clone()
                    # zero out a subset of the channels
                    if perturbation_method == "zeroing":
                        param[combination[0]] = 0
                        if net_name is not "Resnet":
                            params_bias[combination] = 0
                        accuracy = evaluate(net, "val")
                    # add noise to subset of channels (experimental feature)
                    # elif perturbation_method == "additive_noise":
                        # # norm_dist=torch.distributions.Normal(0,0.1)
                        # # param[combination[0]] += norm_dist.sample(param[combination[0]].shape).to(device)
                        # # multiplying by noise
                        # # norm_dist = torch.distributions.Normal(1, 0.1)
                        # # param[combination[0]] *= norm_dist.sample(param[combination[0]].shape)
                        # # adding noise
                        # accuracies = []
                        # for i in range(5):
                        #     norm_dist = torch.distributions.Normal(0, 0.1)
                        #     param[combination[0]] += norm_dist.sample(param[combination[0]].shape)
                        #     accuracies.append(evaluate())
                        # accuracy = np.mean(accuracies)
                        # print("Averaged accuracy: ", accuracy)
                    ########################################333
                    # accuracy = evaluate(net)
                    param.data[combination] = params_saved
                    if net_name is not "Resnet":
                        params_bias.data[combination] = param_bias_saved
                    results.append((combination, accuracy))
                    # write the combinations to the file
                    if file_write:
                        with open(file_name_new, "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r] = results
            file_check()


def exact_partial(dic, nodesNum, original_acc):

    dic[tuple(np.arange(nodesNum))] = original_acc #if no nodes, zero accuracy
    dic[()] = 0
    m = list(dic.keys())
    m.sort(key=lambda t: len(t), reverse=True)

    shaps = np.zeros(nodesNum)
    shaps_samps = np.zeros(nodesNum)
    for elem in m:
        #print("el", elem)
        val1 = dic[elem]
        for i in elem:
            elem_minus = tuple(np.delete(elem, np.where(elem == i)))
            if len(elem_minus) >1 and tuple(elem_minus) in m:
                val2 = dic[elem_minus] - val1
            elif len(elem_minus)==1:
                val2 = dic[elem_minus]

                shaps[i]+=val2
                shaps_samps[i]+=1

    svs = np.divide(shaps, nodesNum)
    print("svs", svs)
    return svs



def check_combination(net, net_name, combination, param, evaluate, params_bias):
    combination = torch.LongTensor(combination)
    print(combination)
    params_saved = param[combination].clone()
    param_bias_saved = params_bias[combination].clone()

    #param[combination[0]] = 0
    param.data[combination] = 0
    #print("Sum:\n ", torch.sum(param, axis=(1, 2, 3)))
    if net_name is not "Resnet":
        params_bias[combination] = 0
    accuracy = evaluate(net, "val")

    param.data[combination] = params_saved
    if net_name is not "Resnet":
        params_bias.data[combination] = param_bias_saved

    return accuracy


def kernshap(file_write, net, net_name, layer, evaluate, dataset, k_num, param, samples_num=10, perturbation_method=None):

            if net_name is not "Resnet":
                layerbias = layer[:-6] + "bias"  #:3 for lenet
                params_bias = net.state_dict()[layerbias]

            combinations_bin = np.zeros((samples_num, param.shape[0]))
            accuracies = np.zeros(samples_num)
            for i in range(samples_num):
                randperm = np.random.permutation(param.shape[0])
                randint = 0
                while (randint == 0):
                    randint = np.random.randint(param.shape[0])
                randint_indextoremove = np.random.randint(randint)
                combination = randperm[:randint]
                combination2 = np.delete(combination, randint_indextoremove)
                print(combination[randint_indextoremove])

                acc = check_combination(net, net_name, combination, param, evaluate, params_bias)

                combinations_bin[i, combination] = 1
                accuracies[i]=acc

            reg = LinearRegression().fit(combinations_bin, accuracies)

            dumm=1
            return reg.coef_


def randomshap(file_write, net, net_name, layer, evaluate, dataset, k_num, param, samples_num=10,
             perturbation_method=None):
    if net_name is not "Resnet":
        layerbias = layer[:-6] + "bias"  #:3 for lenet
        params_bias = net.state_dict()[layerbias]

    acc_val = evaluate(net, "val")

    shaps = np.zeros(param.shape[0])
    combinations_bin = np.zeros((samples_num, param.shape[0]))
    accuracies = np.zeros(samples_num)
    for i in range(samples_num):
        randperm = np.random.permutation(param.shape[0])
        last_acc  = acc_val
        for j in range(param.shape[0]):
            elem = randperm[j]
            print(f"\n\nChannel: {elem}")
            combination = randperm[:j+1]
            acc = check_combination(net, net_name, combination, param, evaluate, params_bias)
            marginal = last_acc - acc
            last_acc = acc
            shaps[elem]+= marginal

        print(shaps)
        randsvs = shaps/samples_num
        print(randsvs)
        print(np.argsort(randsvs)[::-1])
        np.save(f"../methods/sv/{net_name}/randomsvs/randomshap_{layer}_samp_{samples_num}.npy", randsvs)
    return randsvs



# CHOOSES RANDOM COMBINATION and then removed one of the random nodes and computes accuracy for that node
# from ranking/results_compression/network_pruning_withcombinstions.py
def compute_combinations_random(file_write, net, evaluate):
    for name, param in net.named_parameters():
        print(name)
        print(param.shape)
        layer = "c5.weight"
        # find a layer (weight and bias) where we compute rank

        if layer in name:
            layerbias = layer[:3] + "bias"
            params_bias = net.state_dict()[layerbias]
            while (True):

                all_results = {}
                # s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
                # for r in range(1,50): #produces the combinations of the elements in s
                #    results=[]
                randperm = np.random.permutation(param.shape[0])
                randint = 0
                while (randint == 0):
                    randint = np.random.randint(param.shape[0])
                randint_indextoremove = np.random.randint(randint)
                combination = randperm[:randint]
                combination2 = np.delete(combination, randint_indextoremove)
                print(combination[randint_indextoremove])

                if file_write:
                    with open("results_running/combinations_pruning_mnist_%s_%s.txt" % (path[7:], layer), "a+") as textfile:
                        textfile.write("%d\n" % randint_indextoremove)
                for combination in [combination, combination2]:
                    # for combination in list(combinations(s, r)):
                    combination = torch.LongTensor(combination)
                    print(combination)
                    params_saved = param[combination].clone()
                    param_bias_saved = params_bias[combination].clone()
                    # param[torch.LongTensor([1, 4])] = 0
                    # workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
                    if len(combination) != 0:
                        param[combination[0]] = 0
                        # param[combination]=0
                        params_bias[combination] = 0
                    accuracy = evaluate()
                    param.data[combination] = params_saved
                    params_bias.data[combination] = param_bias_saved
                    if file_write:
                        with open("results_running/combinations_pruning_fashionmnist_%s_%s.txt" % (path[7:], layer),
                                  "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                # all_results[r]=results

                # import pickle
                # filename='combinations_all_results_rel_bn_%d.pkl' % r
                # file=open(filename, 'wb')
                # pickle.dump(all_results, file)
                # file.close()


#############################################3
# copied from ranking/results_shapley/shapley.py

# READ ONLY DATA
# not sampled, we take all the combinations of size 1, then all the combinations of size 2, etc.

# reads into dic 0,6 : 98.51
# 6: 98.82
# 7: 98.17
# 8: 98.57
# 9: 99.02
# 0,1: 97.65
# 0,2: 98.83
# 0,3: 98.63
# 0,4: 98.80


def readdata_notsampled(file, original_accuracy):
    f = open(file)
    dict = {(): 0}
    nodes_num = next(f)[:-1] # number of points, first line of the file only
    for line in f:
        linesplit = line.strip().split(":")
        tup = tuple(int(i) for i in linesplit[0].split(","))
        acc = float(linesplit[1])
        dict[tup] = original_accuracy - acc
        print(tup, acc)
    f.close()
    return dict, int(nodes_num)


#######################

# SHAPLEY VALUE

###########################################################
# copied from ranking/results_shapley/shapley.py

# sampled shapley, "full" perms
# (in quotes because we may not have computed all the perms, but we compute them sequentially
# to get all of them, e.g. all perms of size 1, all perms of size 2, etc

# for each node we want to compute Shapley value:
# we get a random permutation and find that node (we count the subset from the beginning up to that node)
# remove it and chceck the difference if both the subsets are present

# works on such dics
# 8: 98.57
# 9: 99.02
# 0,1: 97.65
# 0,2: 98.83
# 0,3: 98.63

def shapley_samp(dict_passed, nodesnum, samples_num):
    print("Partial Random Shapley")
    dict = dict_passed

    # permutations = list(itertools.permutations(elements))
    shap_array = []
    elements_num = nodesnum
    for elem in range(elements_num):  # for each element we want to compute SV of
        sum = 0
        dict_elems = 0
        print(elem)
        for i in range(samples_num):
            perm = np.random.permutation(elements_num).tolist()
            # print(perm)
            # we look at all the permutations
            ind = perm.index(elem)
            del perm[ind + 1:]
            perm.sort()
            perm_tuple = tuple(perm)
            perm.remove(elem)
            removed_perm_tuple = tuple(perm)
            if perm_tuple in dict and removed_perm_tuple in dict:
                val = dict[perm_tuple] - dict[removed_perm_tuple]
                sum += val
                # print(val)
                dict_elems += 1
        # print("sum: %.2f, perms: %d" % (sum,dict_elems))
        shap = sum / dict_elems
        print("shap: %.2f" % shap)
        shap_array.append(shap)

    return shap_array

#from models.lenet5 import Lenet
import torch
import numpy as np
from itertools import chain, combinations
import os
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from operator import itemgetter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def oracle(dic, size, nodenum, adding):

    #smaller values means this set (which is zeroed out) bring highest loss
    keys = list(dic.keys())
    if adding:
        mm = [i for i in keys if len(i) == nodenum - size]
    else:
        mm = [i for i in keys if len(i) == size]

    lal = {k: dic[k] for k in mm}
    d_rev = OrderedDict(sorted(lal.items(), key=itemgetter(1), reverse=True))

    d_keys = list(d_rev)
    d_values = list(d_rev.values())

    # we take the complement of the set
    # adding. we take biggest
    if adding:
        d_keys_comp =[]
        for key in range(len(d_keys)):
            comp = set(np.arange(nodenum)) - set(d_keys[key])
            d_keys_comp.append(comp)

        return d_keys_comp[:5], d_values[:5]

    # removing, we take smallest, removing hurts the accuracy
    else: #best to remove
        return d_keys[:5], d_values[:5]


def oracle_get(dic, param, rank):
    # oracle
    print("\nOracle adding\n") # adding means its loss is the biggest when removing just one
    good=0; all=0
    for o in range(1, 6):
        # get the best set of size o
        set_oracle, val_oracle = oracle(dic, o, param.shape[0], True)
        print(f"\nOracle best to add for {param.shape[0] - o}:")
        print(set_oracle, val_oracle)
        ora = set(list(set_oracle[:o][0]))
        ran = set(rank[:o])
        inter = ora.intersection(ran)
        good+=len(inter); all+=len(ran)
        print(f"Acc add: {good/float(all)}")
    print("\nOracle removing\n")
    good = 0; all = 0
    for o in range(1, 6):
        set_oracle, val_oracle = oracle(dic, o, param.shape[0], False)
        print(f"\nOracle vest to remove for {o}:")
        print(set_oracle, val_oracle)
        ora = set(list(set_oracle[:o][0]))
        ran = set(rank[-o:])
        inter = ora.intersection(ran)
        good += len(inter);
        all += len(ran)
        print(f"Acc remov: {good / float(all)}")


def shapley_rank(evaluate, net, net_name, checkpoint_name, dataset, file_load, k_num, method, sample_num, adding, layer="None", criterion="dummy", args=None, path=None):
    path_file = "sv/Lenet/combin"
    print("Computing Shapley rank in two stages")
    print(f"Shapley method: {method}")
    # just to check the original accuracy without any pruning
    if net_name=="Resnet50":
        acc = 76.13
        #acc = evaluate(dataset, net, criterion, args) # datset is val_laoder
    elif net_name=="resnet": #from grad_drop
        acc = evaluate(net, dataset)
    else:
        acc = evaluate(net, "test")
    # compute combinations/ characteristic function
    if path is None:
        path=f"../methods/sv/{net_name}/{method}"
    os.makedirs(path, exist_ok=True)
    shap_ranks=[]; shap_ranks_dic = {}

    for layer_name, param in net.named_parameters():
        # for a particular layer indicated in args
        if layer != "None":
            if layer==layer_name:
                pass
            else:
                continue
        print(layer_name)

        if "weight" in layer_name and "bn" not in layer_name and "out" not in layer_name:
            print("Lay")
        #if "weight" in layer_name and "bn" not in layer_name and "out" not in layer_name and "fc" in layer_name: #remove after cvpr2022
            if not "Resnet" in net_name or ("Resnet" in net_name and ("layer" in layer_name or "fc" in layer_name or layer_name=="module.conv1.weight")):
            #if not "Resnet" in net_name or ("Resnet" in net_name and "fc" in layer_name):

                print("Layer: ", layer_name)
                global file_name, file_name_new, file_name_old
                file_name = f"{path}/{method}_pruning_{checkpoint_name}_{layer_name}"
                file_name_new = file_name + "_new.txt"
                file_old = file_name + ".txt"
                file_name_old = file_name + ".txt"
                if not os.path.isfile(file_name_old):
                    with open(file_name_old, "a+") as f:
                        f.write((str(param.shape[0])+"\n"))

                if method == "kernel":
                    if not file_load: # kernshap writes the results to file
                        shap_arr = kernshap(True, net, net_name, layer_name, evaluate, dataset, k_num, param, sample_num, "zeroing", args, criterion)
                    dic, nodes_num = readdata_notsampled(file_old, acc)
                    print(f"Read from {file_old}")
                    print(f"Number of samples: {len(dic.keys())}")

                    reg = LinearRegression().fit(list(dic.keys())[1:], list(dic.values())[1:])
                    shap_arr = reg.coef_
                    shap_arr=-1*shap_arr
                    print("shaps\n", shap_arr)

                if method == "random":
                    if not file_load:
                        randomshap(True, net, net_name, checkpoint_name, layer_name, evaluate, dataset, k_num, param, sample_num, "zeroing")
                    shap_arr = readdata_notsampled_random(file_old, acc)
                    print("shaps\n", shap_arr)
                    #shap_arr = file_read("random", net_name, checkpoint_name, layer_name)

                if method == "combin":
                    if not file_load:
                        compute_combinations_lenet(True, net, net_name, layer_name, evaluate, dataset, k_num, "zeroing")
                    dic, nodes_num = readdata_notsampled_combin(file_name_new, acc)
                    #k_num=1; adding=False
                    print(f"\nExact partial for {k_num} and adding: {adding}")
                    shap_arr = exact_partial(dic, nodes_num, acc, adding, k_num)
                    la=np.argsort(shap_arr)[::-1]
                    print(",".join(map(str, la)))

                shap_rank = np.argsort(shap_arr)[::-1]
                print(shap_rank)
                shap_ranks.append(shap_rank)
                shap_ranks_dic[layer_name]=shap_rank

                #get oracle
                file_name = f"../methods/sv/{net_name}/combin/combin_pruning_{checkpoint_name}_{layer_name}"
                file_name_new = file_name + "_new.txt"


                # compute the intersection of the rank selected above and the oracle set
                # maybe commented on the server
                if os.path.isfile(file_name_new):
                    dic, nodes_num = readdata_notsampled_combin(file_name_new, acc)
                    oracle_get(dic, param, shap_rank)


    return shap_ranks, shap_ranks_dic


# def file_read_npy(meth, net_name, checkpoint_name, layer):
#     if meth=="random":
#         samples_most=0
#         for fname in os.listdir(f'../methods/sv/{net_name}/{meth}'):
#             core_name = f"{meth}shap_{checkpoint_name}_{layer}_samp_"
#             if core_name in fname:
#                 samp_num_temp = fname.replace(core_name, "")
#                 samp_num = samp_num_temp.replace(".npy", "")
#                 samples_num = int(samp_num)
#                 if samples_num>samples_most:
#                     samples_most = samples_num
#         #loading file
#         path_meth = f"../methods/sv/{net_name}/{meth}/{meth}shap_{checkpoint_name}_{layer}_samp_{samples_most}.npy"
#         randsvs = np.load(path_meth)
#         print(f"Loaded {meth} Shapley file from {path_meth}")
#     return randsvs


def file_check(method):
    if method=="combin":
        # check if new results have more lines than the previous one
        file_old = file_name + ".txt"
        file_new = file_name + "_new.txt"
        if os.path.exists(file_old):
            num_lines_old = sum(1 for line in open(file_old, "r"))
            num_lines_new = sum(1 for line in open(file_new, "r"))
            # if num_lines_old > num_lines_new:
            #     os.remove(file_new)
            # else:
            #     os.remove(file_old)
            #     os.rename(file_new, file_old)
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
            if "Resnet" not in net_name:
                layerbias = layer[:-6] + "bias" #:3 for lenet
                params_bias = net.state_dict()[layerbias]
            all_results = {}
            # get s and r to compute the (s choose r)
            s = torch.arange(0, param.shape[0])  # list from 0 to 19 as these are the indices of the data tensor
            # get the alternating elements in the channel list to have the most combinations from the beginning and end first
            a = np.arange(0, param.shape[0]+1)
            channel_list = [a[-i // 2] if i % 2 else a[i // 2] for i in range(len(a))]
            channel_list=channel_list[:] if k_num==None else channel_list[:k_num]
            #for r in range(1, param.shape[0]):  # produces the combinations of the elements in s
            for r in channel_list:
                print(r)
                results = []
                for combination in list(combinations(s, r)):
                    combination = torch.LongTensor(combination)
                    #print(combination)
                    # save current values in a placeholder
                    params_saved = param[combination].clone();
                    if "Resnet" not in net_name:
                        param_bias_saved = params_bias[combination].clone()
                    # zero out a subset of the channels
                    if perturbation_method == "zeroing":


                        ## param[combination] = 0
                        ## if net_name is not "Resnet":
                        ##     params_bias[combination] = 0
                        ## accuracy = evaluate(net, "val")

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

                    ##param.data[combination] = params_saved
                    ##if net_name is not "Resnet":
                    ##    params_bias.data[combination] = param_bias_saved

                        accuracy = check_combination(net, net_name, combination, param, evaluate, params_bias)


                    results.append((combination, accuracy))
                    # write the combinations to the file
                    if file_write:
                        with open(file_name_new, "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r] = results
            file_check("combin")


def exact_partial(dic, nodesNum, original_acc, adding, K_param=0):
    #minus means actually plus because we remove it from the list of 0s, so add to the list of non-zeros
    #dic[tuple(np.arange(nodesNum))] = 10 #random accuracy of no all zeros
    dic[()] = original_acc
    m = list(dic.keys())
    m.sort(key=lambda t: len(t), reverse=True)

    shaps = np.zeros(nodesNum)
    shaps_samps = np.zeros(nodesNum)
    N = nodesNum
    for elem in m:
        val1 = dic[elem]
        #print("el: ", elem, "val: ", val1)
        if len(elem) == 1:
            mama = 0
        elem_set = set(elem)
        if adding:
            thresh = (nodesNum - K_param)
        else:
            thresh = K_param
        for i in elem:
            elem_set.remove(i)
            elem_plus = tuple(elem_set)
            K = len(elem_plus)
            if (K>=thresh and adding) or (K+1<=thresh and not adding):
                #if K >1 and tuple(elem_plus) in m:
                if tuple(elem_plus) in m:
                    val2 = dic[elem_plus]-val1
                # elif len(elem_plus)==1:
                #     val2 = dic[elem_plus]
                #print(print("i: ", i, "val2: ", val2, "el: ", elem_plus, "val: ", dic[elem_plus]))

                coeff = np.math.factorial(N-K-1)*np.math.factorial(K)
                shaps[i]+=val2*coeff
                shaps_samps[i]+=1
            elem_set.add(i)

    svs = np.divide(shaps, np.math.factorial(N))
    print("svs", svs)
    return svs



def check_combination(net, net_name, combination, param, evaluate, params_bias, args=None, criterion=None, loader=None):
    combination = torch.LongTensor(combination)
    print(combination)
    params_saved = param[combination].clone()
    if "Resnet" not in net_name and "resnet" not in net_name:
        param_bias_saved = params_bias[combination].clone()

    #param[combination[0]] = 0
    param.data[combination] = 0
    #print("Sum:\n ", torch.sum(param, axis=(1, 2, 3)))
    if "Resnet" not in net_name and "resnet" not in net_name:
        params_bias[combination] = 0

    if net_name is not "Resnet50" and "resnet" not in net_name: #resnet50
        accuracy = evaluate(net, "val")
    elif "resnet" in net_name:
        accuracy = evaluate(net, loader)
    else:
        accuracy = evaluate(loader, net, criterion, args)

    param.data[combination] = params_saved
    if "Resnet" not in net_name and "resnet" not in net_name:
        params_bias.data[combination] = param_bias_saved

    return accuracy

def write_file(file_write, comb, acc):
    if file_write:
        with open(file_name_old, "a+") as textfile:
            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in comb), acc))
            print(f"Saved in {file_name_old}")

def kernshap(file_write, net, net_name, layer, evaluate, dataset, k_num, param, samples_num=10, perturbation_method=None, args=None, criterion=None):

            if "Resnet" not in net_name and "resnet" not in net_name:
                layerbias = layer[:-6] + "bias"  #:3 for lenet
                params_bias = net.state_dict()[layerbias]
            else:
                params_bias = None

            # if file_write:
            #     with open(file_name, "a+") as textfile:
            #         textfile.write(str(param.shape[0])+"\n")

            combinations_bin = np.zeros((samples_num, param.shape[0]))
            accuracies = np.zeros(samples_num)
            for i in range(samples_num):
                print(f"samp: {i}")
                randperm = np.random.permutation(param.shape[0])
                randint = 0
                while (randint == 0):
                    randint = np.random.randint(param.shape[0])
                randint_indextoremove = np.random.randint(randint)
                combination = randperm[:randint]
                combination2 = np.delete(combination, randint_indextoremove)
                print(combination[randint_indextoremove])

                acc = check_combination(net, net_name, combination, param, evaluate, params_bias, args, criterion, dataset)


                combinations_bin[i, combination] = 1
                accuracies[i]=acc

                write_file(file_write, combinations_bin[i], accuracies[i])

            #file_check()

            dumm=1
            return


def randomshap(file_write, net, net_name, checkpoint_name, layer, evaluate, dataset, k_num, param, samples_num=10,
             perturbation_method=None):
    if "Resnet" not in net_name :
        layerbias = layer[:-6] + "bias"  #:3 for lenet
        params_bias = net.state_dict()[layerbias]
    else:
        params_bias = None

    acc_val = evaluate(net, "val")

    shaps = np.zeros(param.shape[0])
    combinations_bin = np.zeros((samples_num, param.shape[0]))
    accuracies = np.zeros(samples_num)
    for i in range(samples_num):
        print(f"\nSample num: {i}")
        randperm = np.random.permutation(param.shape[0])
        last_acc  = acc_val
        nums = []; marginals = [];
        for j in range(param.shape[0]):
            elem = randperm[j]
            print(f"\n\nChannel marginal check: {elem}")
            combination = randperm[:j+1]
            acc = check_combination(net, net_name, combination, param, evaluate, params_bias)
            marginal = last_acc - acc
            last_acc = acc
            shaps[elem]+= marginal

            nums.append(combination); marginals.append(acc)
        for k in range(len(nums)):
            write_file(file_write, nums[k], marginals[k])

        if i % 10 == 0 or i==samples_num-1:
            print(shaps)
            randsvs = shaps/(i+1)
            print(randsvs)
            print(np.argsort(randsvs)[::-1])
            #np.save(f"../methods/sv/{net_name}/random/randomshap_{checkpoint_name}_{layer}_samp_{(i+1)}.npy", randsvs)
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


def readdata_notsampled_marginals(file, original_accuracy):
    f = open(file)
    dict = {(): 0}
    nodes_num = int(next(f)[:-1]) # number of points, first line of the file only
    shap=np.zeros(nodes_num)
    for i in range(nodes_num):
        dict[i]=[]
    for line in f:
        linesplit = line.strip().split(":")
        tup = int(linesplit[0])
        acc = float(linesplit[1])
        #dict[tup] = original_accuracy - acc
        dict[tup].append(acc)
        #print(tup, acc)
    f.close()
    for m in range(nodes_num):
        shap[m]=np.average(dict[m])
    return shap


def readdata_notsampled(file, original_accuracy):
    f = open(file)
    nodes_num = next(f)[:-1] # number of points, first line of the file only
    #line = next(f)
    #linesplit = line.strip().split(":")
    #original_accuracy2 = float(linesplit[1])
    dict = {(): original_accuracy}
    for line in f:



        #print(line)
        linesplit = line.strip().split(":")
        #try:
        tup = tuple(int(float(i) )for i in linesplit[0].split(","))
        #except:
        #    lala=7
        acc = float(linesplit[1])
        #dict[tup] = original_accuracy - acc
        dict[tup]=acc
        #print(tup, acc)
    f.close()
    return dict, int(nodes_num)

def readdata_notsampled_combin(file, original_accuracy):
    f = open(file)
    nodes_num = next(f)[:-1] # number of points, first line of the file only
    line = next(f)
    linesplit = line.strip().split(":")
    original_accuracy2 = float(linesplit[1])
    dict = {(): original_accuracy2}
    for line in f:
        #print(line)

        linesplit = line.strip().split(":")
        #try:
        tup = tuple(int(float(i) )for i in linesplit[0].split(","))
        #except:
        #    lala=7
        acc = float(linesplit[1])
        #dict[tup] = original_accuracy - acc
        dict[tup]=acc
        #print(tup, acc)
    f.close()
    return dict, int(nodes_num)


def get_svs(dict, original_accuracy, nodes_num):
    shaps=np.zeros(nodes_num)
    for key in dict.keys():
        keyl = list(key)
        if len(keyl)==0:
            #shaps[keyl[-1]]=original_accuracy-dict[key]
            old_key_val=dict[key]
        else:
            shaps[keyl[-1]]=old_key_val-dict[key]
            old_key_val=dict[key]
    return shaps


def readdata_notsampled_random(file, original_accuracy):
    f = open(file)
    nodes_num = int(next(f)[:-1]) # number of points, first line of the file only
    #line = next(f)
    #linesplit = line.strip().split(":")
    #original_accuracy2 = float(linesplit[1])
    i=0
    dict = {(): original_accuracy}
    shaps=np.zeros(nodes_num)
    samps=0
    for line in f:
        i+=1
        linesplit = line.strip().split(":")
        tup = tuple(int(float(i) )for i in linesplit[0].split(","))
        acc = float(linesplit[1])
        #dict[tup] = original_accuracy - acc
        dict[tup]=acc
        #print(tup, acc)
        if i == int(nodes_num):
            i=0
            samps+=1
            # compute the difference in a permuattion from removing more and mroe nodes
            shaps_part = get_svs(dict, original_accuracy, nodes_num)
            shaps+=shaps_part
            dict = {(): original_accuracy}

    shaps=shaps/samps
    f.close()
    return shaps


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

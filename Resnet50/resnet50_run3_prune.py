import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import socket
import datetime

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#from models.resnet50 import resnet50
from models.resnet_im_ex import resnet50
from methods import shapley_rank

from dataloaders.dataset_google import load_google
from dataloaders.dataset_imagenet import load_imagenet, load_imagenet_tar


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dir_data', metavar='DIR',
                    help='path to dataset', default="/home/kamil/Dropbox/Current_research/data/imagenet/imagenet")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=40, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=1, type=int,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=1, type=int,
                    help='use pre-trained model')
parser.add_argument("--train_bool", default=1, type=int)
parser.add_argument("--restart", default=0, type=int)
parser.add_argument("--restart_name", default="")

parser.add_argument("--rank_method", default="shapley")
parser.add_argument("--layer", default="None") #module.layer1.0.conv1.weight

#shapley
parser.add_argument("--shap_method", default="kernel")
parser.add_argument("--load_file", default=1, type=int) #loads texfile with shapley coalitions
parser.add_argument("--k_num", default=None)
parser.add_argument("--shap_sample_num", default=1, type=int)
parser.add_argument("--adding", default=0, type=int) #for combin/oracle


parser.add_argument("--prune", default=0, type=int)
parser.add_argument("--pruned_arch_ins", default="42,80,130,250") #remaining, not what we prune
parser.add_argument("--pruned_arch_out", default="50,240,390,704,1648") #remaining, not what we prune
#parser.add_argument("--pruned_arch", default="23,67,130,260")
parser.add_argument("--dataset", default="imagenet_tar")
parser.add_argument("--train_len", default=None)
parser.add_argument("--test_len", default=None)


best_acc1 = 0


def main():



    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Device count: ", torch.cuda.device_count())

    args = parser.parse_args()
    args.pruned_arch_ins = args.pruned_arch_ins.replace("_", ",")
    args.pruned_arch_out = args.pruned_arch_out.replace("_", ",")

    restart_list = os.listdir("restart")
    random.shuffle(restart_list)
    create_new=1
    time.sleep(np.random.uniform(1,20))
    for r in restart_list:
        with open("restart/"+r, "r") as f:
            lines = f.readlines()
        if len(lines)>0:
            print("line:", lines[-1].strip())
            if lines[-1]=="taken\n":
                continue
            else:
                if "checkpoint" in lines[-1]: #extra check
                    with open("restart/"+r, "a+") as f:
                        f.write("taken\n")
                        print(r)
                        print("taken old")
                        create_new=0
                        args.restart_name = r
                    args.resume = lines[-1].strip()
                    break
        #if create_new==0:
        #    break
    print("create", create_new)
    if create_new:
        print("in")
        args.restart_name = str(time.time())+".txt"
        with open("restart/" + args.restart_name, "a+") as f:
            f.write("taken\n")


    if socket.gethostname() == 'kamilblade':
            args.dir_data = "/home/kamil/Dropbox/Current_research/data/imagenet/imagenet"
    print(f"Imagenet data at {args.dir_data}")
        #args.batch_size = 512 #128 1 gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        with open(args.restart_name, "a+") as f:
            f.write("taken\n")


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
        model = resnet50(pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

#model is in /anaconda3/lib/python3.7/site-packages/torchvision/models/resnet.py

    params=0
    for name, param in model.named_parameters():
        #if "weight" in name and "bn" not in name and "linear" not in name:
        #print(name)
        #print(param.shape)
        if "weight" in name:
            if "bn" in name:
                params+=param.shape[0]
            elif "downsample" in name and ".0.wei" in name:
                params=param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3]
            elif "downsample" in name:
                params+=param.shape[0]
            elif "fc" in name:
                params += param.shape[0] * param.shape[1]
            else:
                params+=param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3]
        if "bias" in name:
            params+=param.shape[0]
    print("All params: ", params)



    #
    # flops: should be 3.8 billion FLOPs.


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("Dataparallel")
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.dir_data, 'train')
    valdir = os.path.join(args.dir_data, 'val')

    ####
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    #######

    if args.dataset == "imagenet":
        train_loader, val_loader, testval_loader = load_imagenet(args)
    elif args.dataset =="imagenet_tar":
        train_loader, val_loader, testval_loader = load_imagenet_tar(args)
    elif args.dataset == "google":
        train_loader, val_loader = load_google(args)

    # before Oct 22
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    #
    # torch.manual_seed(100)
    # trainval_perc = 0.9
    #
    # trainset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform_train)
    # valset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform_test)
    # testset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform_test)
    # n_train = len(trainset)
    # indices = list(range(n_train))
    # np.random.shuffle(indices)
    # trainval_size = int(trainval_perc * len(trainset))
    # val_size = len(trainset) - trainval_size
    #
    # assert val_size < n_train
    # train_idx, val_idx = indices[val_size:], indices[:val_size]
    #
    # # to have a subset of train samples
    # train_len = len(indices[val_size:])
    # fixed_train_idx = indices[:int(0.05 * train_len)]
    #
    # train_sampler = SubsetRandomSampler(train_idx)  # train_idx
    # val_sampler = SubsetRandomSampler(val_idx)
    #
    #
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
    #                                            sampler=train_sampler, num_workers=args.workers)
    #
    # trainall_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers)
    #
    # valtrain_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
    #                                          sampler=val_sampler, num_workers=args.workers)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers)
    #
    # print(f"Train all: {len(trainall_loader)}, train sampler: {len(train_loader.sampler)}, valtrain sampler: {len(valtrain_loader.sampler)}, val sampler: {len(val_loader.sampler)}")

    ########

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    #
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    ##############

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)

    if args.prune:

        print("\n   Pruning")
        prune_func(model, args, val_loader, criterion) # should be testval
        if args.evaluate:
            prec1 = validate(val_loader, model, criterion, args)


    if args.train_bool:
        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            #     train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                filename=args.dataset+"_"+str(epoch+1)+"_"+str(best_acc1.item())+".pth.tar"
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args, filename)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        int(args.train_len/args.batch_size),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        #print("bef", datetime.datetime.now())

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # debug
        #model.module.layer1[0].conv1.weight
        #print("Debug sum")
        #rank_layer = ranks["module.layer3.1.conv2.weight"]
        #output channel (first) that is worst according to the rank, should be 0
        #print(torch.sum(model.module.layer3[1].conv2.weight[rank_layer[-1]]))

        # compute output

        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if args.prune:
            zero_params(model, ranks, thresholds_ins, thresholds_out, args)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        #print("aft", datetime.datetime.now())



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        int(args.test_len/args.batch_size),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = "checkpoint/"+ filename
    torch.save(state, filename)

    with open("restart/"+args.restart_name, "a+") as file:
        file.write(filename+"\n")

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    print("Saved")


def get_ranks(model, args, val_loader, criterion):

    if args.rank_method == 'switches':
        to_do=1
        # switch_train = args.switch_train
        # if switch_train:
        #     ranks = resnet_switch_main(args)
        # else:
        #     ranks = np.load("../methods/switches/Resnet56/ranks_epi_4.npy", allow_pickle=True)
        #     ranks = ranks[()]
        #     #renaming keys
        #
        # new_ranks={}
        # for key in ranks.keys():
        #     if "parameter2" in key:
        #         new_key = key[:15]+".conv2.weight"
        #         new_ranks[new_key] = ranks[key]
        #     elif "parameter1" in key:
        #         new_key = key[:15] + ".conv1.weight"
        #         new_ranks[new_key] = ranks[key]
        #
        # ranks = new_ranks

    elif args.rank_method == 'shapley':
        try:
            #validate(val_loader, model, criterion)
            ranks_list, ranks = shapley_rank.shapley_rank(validate, model, "Resnet50", "", val_loader, args.load_file, args.k_num, args.shap_method, args.shap_sample_num, args.adding, args.layer, criterion, args)
        except KeyboardInterrupt:
            print('Interrupted')
            shapley_rank.file_check()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


    return ranks


#in case of switches we will prune the first conv in block accou=rding to the rankings in the second

def zero_params(model, ranks, thresholds_ins, thresholds_out, args):

    # for bottlenext input/output
    global name_conv1, name_conv3
    name_conv1 = -1
    name_conv3 = -1  # name of the bottleneck module )in case we want to prune
    param_conv3 = -1  # sides)

    for name, param in model.state_dict().items():
        #print(name)
        #print(param)



        if "layer" in name:
        #if "layer3.8" in name and ("conv2" in name or "bn2" in name):
            # print(f"pruning layer {name}")
            core_name=name[:15]
            # we only look at conv1 because we prune two layer modules and only the output of the 1st and 2nd conv in the module, that is inside bottleneck
            if "conv1.weight" in name or "conv2.weight" in name:
                #param1_name=core_name+".parameter1"
                #rank1 = ranks[()][param1_name]
                if args.rank_method == "switches":
                    name_ch = name.replace("conv1", "conv2")
                else:
                    name_ch = name
                rank1 = ranks[name_ch]
                channels_bad= rank1[thresholds_ins[core_name]:] #to be removed
                channels_bad = channels_bad if torch.is_tensor(channels_bad) else torch.Tensor(channels_bad.copy()).long()
                # param.data[:, channels_bad]=0
                param.data[channels_bad, :] = 0

                # for bottlneck input/output pruning
                if "conv1.weight" in name:
                    param_conv1 = param
                    name_conv1 = name
                    #print(name_conv1)

            elif "conv1.bias" in name or "bn1.bias" in name or "bn1.weight" in name or "bn1.running_var" in name or "bn1.running_mean" in name or "conv2.bias" in name or "bn2.bias" in name or "bn2.weight" in name or "bn2.running_var" in name or "bn2.running_mean" in name:
                #rank1 = ranks[()][param1_name]
                name_orig = core_name +".conv1.weight"
                if args.rank_method == "switches":
                    name_ch = name_orig.replace("conv1", "conv2")
                else:
                    name_ch = name_orig
                rank1 = ranks[name_ch]
                channels_bad=rank1[thresholds_ins[core_name]:]
                channels_bad = channels_bad if torch.is_tensor(channels_bad) else torch.Tensor(channels_bad.copy()).long()
                param.data[channels_bad]=0
            # if

            ################# side channels
            if "conv3.weight" in name:

                # param1_name=core_name+".parameter1"
                # rank1 = ranks[()][param1_name]
                if args.rank_method == "switches":
                    name_ch = name.replace("conv1", "conv2")
                else:
                    name_ch = name
                rank1 = ranks[name_ch]
                # now keeps the same number as in bottleneck, so removed more in the last layer because of dilation
                channels_bad = rank1[thresholds_out[core_name]:]  # to be removed
                channels_bad = channels_bad if torch.is_tensor(channels_bad) else torch.Tensor(channels_bad.copy()).long()
                # param.data[:, channels_bad]=0
                param.data[channels_bad, :] = 0

                name_conv3 = name
                channels_bad_conv3 = channels_bad

                # after pruning output to conv3 we prune input to conv1
                # we do not prune the in of the first con in first layer
                new_conv1_name = name_conv3.replace("conv3", "conv1")
                if new_conv1_name == name_conv1 and ".0." not in new_conv1_name:  # saninty check
                    param_conv1.data[:, channels_bad_conv3] = 0
                    name_conv3 = -1

            elif "conv3.bias" in name or "bn3.bias" in name or "bn3.weight" in name or "bn3.running_var" in name or "bn3.running_mean" in name:
                # rank1 = ranks[()][param1_name]
                name_orig = core_name + ".conv1.weight"
                if args.rank_method == "switches":
                    name_ch = name_orig.replace("conv1", "conv2")
                else:
                    name_ch = name_orig
                rank1 = ranks[name_ch]
                channels_bad = rank1[thresholds_out[core_name]:]
                channels_bad = channels_bad if torch.is_tensor(channels_bad) else torch.Tensor(channels_bad.copy()).long()
                param.data[channels_bad] = 0










# module.layer3.8.bn2.weight torch.Size([64])
# module.layer3.8.bn2.bias torch.Size([64])
# module.linear.weight torch.Size([10, 64])
# module.linear.bias torch.Size([10])
# Test: [0/79]	Time 0.120 (0.120)	Loss 4.0260 (4.0260)	Prec@1 9.375 (9.375)

def prune_func(model, args, val_loader, criterion):

    # get threshold
    global ranks, thresholds_ins, thresholds_out
    thresholds_ins ={}; thresholds_out={}
    preserved_ins = [int(n) for n in args.pruned_arch_ins.split(",")]
    preserved_ins = np.insert(preserved_ins, 0, 0) #adding dummy value at the 0th position
    preserved_out = [int(n) for n in args.pruned_arch_out.split(",")]
    preserved_out = np.insert(preserved_out, 0, 0)  # adding dummy value at the 0th position
    for i1 in range(1,5):
        for i2 in range(0,9):
            thresholds_ins[f"module.layer{i1}.{i2}"]=int(preserved_ins[i1])
            thresholds_out[f"module.layer{i1}.{i2}"] = int(preserved_out[i1])
            #thresholds[f"layer{i1}.{i2}.2"] = preserved[i1]
    thresholds_ins[f"module.layer2.0"] = int(preserved_ins[2])
    thresholds_ins[f"module.layer3.0"] = int(preserved_ins[3])
    thresholds_ins[f"module.layer4.0"] = int(preserved_ins[4])
    thresholds_out[f"module.layer2.0"] = int(preserved_out[2])
    thresholds_out[f"module.layer3.0"] = int(preserved_out[3])
    thresholds_out[f"module.layer4.0"] = int(preserved_out[4])

    # get ranks
    print("Getting ranks")
    ranks = get_ranks(model, args, val_loader, criterion)
    #print(ranks)

    # zero params
    zero_params(model, ranks, thresholds_ins, thresholds_out, args)
    print(f"\nThresholds_ins: {thresholds_ins}, thresholds_out: {thresholds_out}")



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

#https://github.com/DeepLearnPhysics/pytorch-resnet-example/blob/master/resnet_example.py

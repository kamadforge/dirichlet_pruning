import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
from pathlib import Path
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
#sys.path.append(str(Path(sys.path[0]).resolve().parent))
#sys.path.append(str(Path(sys.path[0]).resolve().parent / "folder1"))

from methods.resnet_trainer_switch import main as resnet_switch_main

from models import resnet
#from results_switch_v3.models import resnet
import numpy as np

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet56)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='pretrained_models/resnet56-4bfd9763.th', type=str, metavar='PATH',
#parser.add_argument('--resume', default='save_temp/checkpoint_pruned_76.26.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)


parser.add_argument("--prune", default=True)
parser.add_argument("--pruned_arch", default="13,31,45")


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    #data
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 0#checkpoint['epoch']
            best_prec1 = 0# checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, 0))
            validate(val_loader, model, criterion)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.prune:

        print("\n   Pruning")
        prune_func(model)
        prec1 = validate(val_loader, model, criterion)

    if args.evaluate:
        # for name, param in model.named_parameters():
        #     print(name, param.shape)
        print("Evaluating:")
        validate(val_loader, model, criterion)
        return

    print("\nTraining\n")
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # check if the connections are actually pruned
        # for name, param in model.named_parameters():
        #     print(name)
        #     if "layer" in name and ("conv1.weight" in name or "conv2.weight" in name):
        #         channels = param.shape[1]
        #         for ch in range(channels):
        #             ch_sum = torch.sum(param[:,ch])
        #             if ch_sum == 0:
        #                 print(ch)
        #     else:
        #         channels = param.shape[0]
        #         for ch in range(channels):
        #             ch_sum = torch.sum(param[ch])
        #             if ch_sum == 0:
        #                 print(ch)


        if args.prune:
            name_checkpoint = f"checkpoint_pruned_{prec1}.th"
        else:
            name_checkpoint = f"checkpoint_{prec1}.th"

        #if epoch > 0 and epoch % args.save_every == 0 and is_best:
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, name_checkpoint))

        # save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        # }, is_best, filename=os.path.join(args.save_dir, f'model_{prec1}.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     if "param" not in name:
        #         #param.grad = None
        #         print(param.grad.shape)
        #print(model.module.layer1[0].conv1.weight.grad[0])
        #model.module.layer1[0].conv1.weight.grad[0] = 0
        optimizer.step()

        zero_params(model, ranks, thresholds)

        # print(model.module.layer1[0].conv1.weight[0])
        # print('*'*100)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    print("\nValidating:\n")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print('Validation * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def get_ranks():

    switch_train = False
    if switch_train:
        ranks = resnet_switch_main()
    else:
        ranks = np.load("../methods/switches/Resnet56/ranks.npy", allow_pickle=True)

    return ranks

def zero_params(model, ranks, thresholds):
    for name, param in model.named_parameters():
        #print(name)
        if "layer" in name:
            core_name=name[:15]
            if "conv1.weight" in name:
                param1_name=core_name+".parameter1"
                rank1 = ranks[()][param1_name]
                channels_bad= rank1[thresholds[core_name]:] #to be removed
                param.data[:, channels_bad]=0
            elif "conv2.weight" in name:
                param2_name = core_name+".parameter2"
                rank2 = ranks[()][param2_name]
                channels_bad=rank2[thresholds[core_name]:]
                param.data[:, channels_bad]=0
            elif "conv1.bias" in name or "bn1.bias" in name or "bn1.weight" in name:
                rank1 = ranks[()][param1_name]
                channels_bad=rank1[thresholds[core_name]:]
                param.data[channels_bad] = 0
            elif "conv2.bias" in name or "bn2.bias" in name or "bn2.weight" in name:
                rank2 = ranks[()][param2_name]
                channels_bad=rank2[thresholds[core_name]:]
                param.data[channels_bad] = 0


def prune_func(model):

    # get threshold
    global ranks, thresholds
    thresholds ={}
    preserved = [int(n) for n in args.pruned_arch.split(",")]
    preserved = np.insert(preserved, 0, 0) #adding dummy value at the 0th position
    for i1 in range(1,4):
        for i2 in range(0,9):
            thresholds[f"module.layer{i1}.{i2}"]=int(preserved[i1])
            #thresholds[f"layer{i1}.{i2}.2"] = preserved[i1]
    thresholds[f"module.layer2.0"] = int(preserved[1])
    thresholds[f"module.layer3.0"] = int(preserved[2])

    # get ranks
    print("Getting ranks")
    ranks = get_ranks()
    #print(ranks)

    # zero params
    zero_params(model, ranks, thresholds)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

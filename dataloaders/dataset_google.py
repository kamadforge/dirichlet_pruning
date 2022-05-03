from torchvision import datasets, transforms
import torch
import torchvision
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import torch.nn.functional as F
from torchvision.utils import save_image


def load_google(args, trainval_perc=0.9):
    # Data
    print('==> Preparing data..')
    root_dir = args.dir_data
    train_batch_size = args.batch_size
    val_batch_size = 128
    num_workers = args.workers #args.n_threads
    print('Number of workers {}'.format(num_workers))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print("Root dir: ", root_dir)
    trainset = datasets.ImageFolder(root_dir, transform_train)
    valset = datasets.ImageFolder(root_dir, transform_test)

    # extract val dataset from train dataset
    n_train = len(trainset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    train_size = int(trainval_perc * len(trainset))
    val_size = len(trainset) - train_size

    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    # to have a subset of train samples
    fixed_indices = indices[:2000]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                             sampler=val_sampler, num_workers=num_workers)

    # from IPython import embed; embed()
    return train_loader, val_loader

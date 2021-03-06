from torchvision import datasets, transforms
import torch
import torchvision
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def load_imagenet(trainval_perc):
    # Data
    print('==> Preparing data..')
    root_dir = './data/imagenet'
    train_batch_size = 256
    val_batch_size = 128
    num_workers = 8

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

    trainset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform_train)
    valset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform_test)
    testset = datasets.ImageFolder(os.path.join(root_dir, 'val'), transform_test)
    n_train = len(trainset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    train_size = int(trainval_perc * len(trainset))
    val_size = len(trainset) - train_size

    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                             sampler=val_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, num_workers=num_workers)


    return train_loader, test_loader, val_loader

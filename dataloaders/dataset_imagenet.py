from torchvision import datasets, transforms
import torch
import torchvision
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import torch.nn.functional as F
from torchvision.utils import save_image


def load_imagenet(args, trainval_perc=1.0):
    # Data
    print('==> Preparing data..')
    root_dir = args.dir_data
    train_batch_size = args.batch_size
    val_batch_size = 128
    num_workers = args.workers
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

    trainset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform_train)
    valset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform_test)
    testset = datasets.ImageFolder(os.path.join(root_dir, 'val'), transform_test)

    # extract val dataset from train dataset
    n_train = len(trainset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    train_size = int(trainval_perc * len(trainset))
    val_size = len(trainset) - train_size

    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    # to have a subset of train samples
    fixed_train_idx = indices[:100000]
    # train_len = len(indices[val_size:])
    # fixed_train_idx = indices[:int(0.05 * train_len)]

    train_sampler = SubsetRandomSampler(fixed_train_idx) #train_idx
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                             sampler=val_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, num_workers=num_workers)

    # from IPython import embed; embed()
    return train_loader, test_loader, val_loader


class Data:
    def __init__(self, args):
        self.loader_train, self.loader_test, self.loader_val = load_imagenet(args)

    #def downsample(self):

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__=="__main__":
    print("imagenet")
    args = {}
    args['dir_data'] = "/home/kamil/Dropbox/Current_research/data/imagenet/imagenet"
    args['n_threads'] = 0
    args['batch_size'] = 1
    args = dotdict(args)
    print(f"Imagenet data at {args.data}")
    imagenet = Data(args)

    for i, (images, target) in enumerate(imagenet.loader_train):
        print(i, target)
        if i<2000:
            img = images[0]
            torch_img_scaled = F.interpolate(img.unsqueeze(0), (32,32), mode='bilinear').squeeze(0)
            #new_img = torch.nn.Upsample(images, size_new=(32, 32), mode="bilinear")
            #save_image(img, f"../data/{i}_orig.jpeg")
            channels=3
            if channels==1:
                os.makedirs(f"../data/data1/{target.item()}", exist_ok=True)
                save_image(torch_img_scaled[1].unsqueeze(0), f"../data/data1/{target.item()}/{i}.jpeg")
            elif channels==3:
                os.makedirs(f"../data/data3_32/{target.item()}", exist_ok=True)
                save_image(torch_img_scaled, f"../data/data3_32/{target.item()}/{i}.jpeg")




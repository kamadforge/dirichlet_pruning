from torchvision import datasets, transforms
import torch
import torchvision
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import torch.nn.functional as F
from torchvision.utils import save_image
import webdataset as wds
import socket


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
    print(n_train)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    train_size = int(trainval_perc * len(trainset))
    val_size = len(trainset) - train_size

    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    # to have a subset of train samples
    # fixed_train_idx = indices[:100000]
    # train_len = len(indices[val_size:])
    # fixed_train_idx = indices[:int(0.05 * train_len)]

    train_sampler = SubsetRandomSampler(train_idx) #train_idx
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                             sampler=val_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, num_workers=num_workers)

    args.train_len = len(trainset)
    args.test_len = len(testset)
    
    print("train: ", args.train_len, "test: ", args.train_len)

    # from IPython import embed; embed()
    return train_loader, test_loader, val_loader


def load_imagenet_tar(args, trainval_perc=1.0):
    # Data
    print('==> Preparing data..')
    if socket.gethostname() != 'kamilblade':
        root_dir = "/home/kamil/Dropbox/Current_research/data/imagenet_tar/imagenet_sample_train.tar" #args.dir_data
        root_dir = "/home/kamil/Dropbox/Current_research/data/imagenet_tar/dataset-%06d.tar"
        root_dir = "/home/kamil/Dropbox/Current_research/data/imagenet_tar/dataset-{000000..000003}.tar"
        root_dir_test = "/is/cluster/scratch/kamil_old/imagenet/imagenet/imnet_test/imagenet_test-{000000..000049}.tar"
        root_dir_train = "/is/cluster/scratch/kamil_old/imagenet/imagenet/imnet_train/imagenet_train-{000000..001281}.tar" # all train 1281
        #root_dir_test = "/is/cluster/scratch/kamil_old/imagenet/imagenet/imagenet_test-sample.tar" #all test dataset
        #root_dir_train = "/is/cluster/scratch/kamil_old/imagenet/imagenet/train_random/imagenet_train-{000001..001281}.tar" # randomized imagenet to have all the class representatives in the first few tar files
    else:
        root_dir_train = "/home/kamil/Dropbox/Current_research/data/imagenet_tar/imagenet_sample_train.tar"
        root_dir_test = "/home/kamil/Dropbox/Current_research/data/imagenet_tar/imagenet_sample_test.tar"

#    import tarfile
#    test_tar_num = int(root_dir_test[-11:-5])
#    train_tar_num = int(root_dir_train[-11:-5])
#    print(train_tar_num)
#    print(test_tar_num)
#    args.train_len=0; args.test_len=0
#    for t in range(1,train_tar_num+1):
#        print(t)
#        root_dir_train_single = root_dir_train[:-20]+str(t).zfill(6)+".tar"
#        train_tar = tarfile.open(root_dir_train_single)
#        args.train_len += int(len(train_tar.getmembers())/2) #input, output
#    for t in range(1,test_tar_num+1):
#        root_dir_test_single = root_dir_test[:-20]+str(t).zfill(6)+".tar"
#        test_tar = tarfile.open(root_dir_test_single)
#        args.test_len += int(len(test_tar.getmembers())/2) #input, output
    
    args.train_len = 1281166
    args.test_len = 50000
    
    #test_tar = tarfile.open(root_dir_test)
    #args.test_len = int(len(test_tar.getmembers())/2)
    print("Train: ", args.train_len, "test: ", args.test_len)

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

    train_dataset_tar = (wds.WebDataset(root_dir_train).shuffle(1000).decode("pil").rename(image="input.pyd", i="output.pyd").map_dict(image=transform_train).to_tuple("image", "i"))

    #x, y = next(iter(train_dataset_tar))
    #print(x.shape, str(y)[:50])


######

    test_dataset_tar = (
        wds.WebDataset(root_dir_test).shuffle(1000).decode("pil").rename(image="input.pyd", i="output.pyd").map_dict(
            image=transform_test).to_tuple("image", "i"))

    train_sampler = None

    ###################
    root_dir_t = args.dir_data

#    test_dataset_tar = datasets.ImageFolder(os.path.join(root_dir_t, 'val'), transform_test)

    #    train_dataset_tar = datasets.ImageFolder(os.path.join(root_dir_t, 'train'), transform_train)
    #    # to have a subset of train samples
    #    n_train = len(train_dataset_tar)
    #    indices = list(range(n_train))
    #    np.random.shuffle(indices)
    #    fixed_train_idx = indices[:100000]
    #    train_sampler = SubsetRandomSampler(fixed_train_idx) #train_idx
    #    print(fixed_train_idx[:10])

    #######################

    train_loader = torch.utils.data.DataLoader(train_dataset_tar, batch_size=train_batch_size, sampler=train_sampler,num_workers=num_workers)
    # val_loader = torch.utils.data.DataLoader(train_dataset_tar, batch_size=val_batch_size, sampler=val_sampler, num_workers=num_workers)
    val_loader = None
    test_loader = torch.utils.data.DataLoader(test_dataset_tar, batch_size=val_batch_size, num_workers=num_workers)

    #######################

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
    print(f"Imagenet data at {args.dir_data}")
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




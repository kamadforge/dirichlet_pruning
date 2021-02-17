from torchvision import datasets, transforms
import torch
import torchvision
from torchvision.transforms import ToTensor



def load_svhn():
    # Data
    print('==> Preparing data..')
    trainval_dataset = torchvision.datasets.SVHN(root='./data', download=True, transform=ToTensor())
    trainval_perc = 0.8
    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    #return trainloader, testloader, valloader
    return trainloader, valloader #valloader is a testloader because SVHN does not have a separate test dataset

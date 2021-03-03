from torchvision import datasets, transforms
import torch
import torchvision


def load_cifar(trainval_perc):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    trainval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # with more workers there may be an error in debug mode: RuntimeError: DataLoader worker (pid 29274) is killed by signal: Terminated.
    print(f"Training on CIFAR on the {trainval_perc} of the training set.\n")
    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, valloader

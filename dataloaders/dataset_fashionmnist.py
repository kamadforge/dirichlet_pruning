from torchvision import datasets, transforms
import torch


def load_fashionmnist(BATCH_SIZE, trainval_perc=1):
    trainval_dataset = datasets.FashionMNIST('data/FashionMNIST', train=True, download=True,
                                                 # transform=transforms.Compose([transforms.ToTensor(),
                                                 # transforms.Normalize((0.1307,), (0.3081,))]),
                                                 transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset = datasets.FashionMNIST('data/FashionMNIST', train=False, transform=transforms.ToTensor())

    print("Loading FashionMNIST")

    # Load datasets

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if len(val_dataset) >0:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        val_loader = None
    test_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        test_dataset,
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, val_loader

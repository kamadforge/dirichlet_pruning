import torch.utils.data
import torch
from torch import nn, optim
import torch.nn.functional as f
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from torch.nn.parameter import Parameter

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"

from importlib.machinery import SourceFileLoader
dataset_mnist = SourceFileLoader("module_mnist", "../dataloaders/dataset_mnist.py").load_module()
dataset_fashionmnist = SourceFileLoader("module_fashionmnist", "../dataloaders/dataset_fashionmnist.py").load_module()
model_lenet5 = SourceFileLoader("module_lenet", "../models/lenet5.py").load_module()
from module_fashionmnist import load_fashionmnist
from module_mnist import load_mnist
from module_lenet import Lenet

#############################
# PARAMS

sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
layer="c5"
how_many_epochs=200
annealing_steps = float(8000. * how_many_epochs)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
alpha_0 = 2  # below 1 so that we encourage sparsity
hidden_dim = 10 #it's a number of parameters we want to estimate, e.g. # conv1 filters
hidden_dims={'c1': conv1, 'c3': conv2, 'c5': fc1, 'f6' : fc2}
hidden_dim = hidden_dims[layer] #it's a number of parameters we want to estimate, e.g. # conv1 filters


##############################################################################
# NETWORK (conv-conv-fc-fc)

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        self.c1=nn.Conv2d(1, nodesNum1, 5)
        self.s2=nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(nodesNum1)
        self.c3=nn.Conv2d(nodesNum1,nodesNum2,5)
        self.s4=nn.MaxPool2d(2)
        self.bn2=nn.BatchNorm2d(nodesNum2)
        self.c5=nn.Linear(nodesNum2*4*4, nodesFc1)
        self.f6=nn.Linear(nodesFc1,nodesFc2)
        self.f7=nn.Linear(nodesFc2,10)
        self.drop_layer = nn.Dropout(p=0.5)

        self.parameter = Parameter(-1*torch.ones(hidden_dims[layer]),requires_grad=True) # this parameter lies #S

    def switch_func(self, output, Sprime):
        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])
        return output, Sprime

    def forward(self, x, layer):

        phi = f.softplus(self.parameter)
        S = phi / torch.sum(phi)  # """directly use mean of Dir RV, which is {E} [X_{i}]={\frac {\alpha _{i}} {\sum _{k=1}^{K}\alpha _{k}}}}
        Sprime = S
        #print('*'*30)
        #print(S)

        output=self.c1(x)
        if layer == 'c1':
            output, Sprime = self.switch_func(output, Sprime) #13.28 deteministic, acc increases
        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)
        if layer == 'c3':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)
        output=self.c5(output)
        if layer == 'c5':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output=self.f6(output)
        if layer == 'f6':
            output, Sprime = self.switch_func(output, Sprime)  # 13.28 deteministic, acc increases
        output = self.f7(output) #remove for 99.27
        return output, Sprime

nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
criterion = nn.CrossEntropyLoss()
#
# optimizer=optim.Adam(net.parameters(), lr=0.001)

########################################################
# EVALUATE

def evaluate(net2, layer, test_loader):
    #print('Evaluating switches with layer:', layer )
    net2.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        images = images.to(device)
        #dummy works as it should, if we don't execute switch function in forward the accuracy should be original, 99.27
        #predicted_prob = net2.forward(images, "dummy")[0]  # if using switches
        #predicted_prob = net2.forward(images, "c1")[0] #13.68 for 99.27
        #predicted_prob = net2.forward(images, "c3")[0] #11.35
        predicted_prob = net2.forward(images, layer)[0]
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        #print(correct)
    # print(str(correct) +" "+ str(total))
    accuracy = 100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

#evaluate(net2, layer)


#######################s
# LOSS

def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps):
    # BCE = f.binary_cross_entropy(prediction, true_y, reduction='sum')
    BCE = criterion(prediction, true_y)
    return BCE

###########################

def loss_functionKL(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
    # BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')
    BCE = criterion(prediction, true_y)

    # KLD term
    alpha_0 = torch.Tensor([alpha_0]).to(device)
    hidden_dim = torch.Tensor([hidden_dim]).to(device)
    trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim * alpha_0)
    trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim * torch.lgamma(alpha_0)
    trm3 = torch.sum((S - alpha_0) * (torch.digamma(S) - torch.digamma(torch.sum(S))))
    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    return BCE + annealing_rate * KLD / how_many_samps


###################################################
# RUN TRAINING

def run_experiment(epochs_num, layer, nodesNum1, nodesNum2, nodesFc1, nodesFc2, path, args):
    print("\nRunning Dirichlet switch training\n")

    ###################################################
    # DATA
    if args.dataset == "fashionmnist":
        train_loader, test_loader, val_loader = load_fashionmnist(args.batch_size, args.trainval_perc)
    elif args.dataset == "mnist":
        train_loader, test_loader, val_loader = load_mnist(args.batch_size, args.trainval_perc)

    net2 = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net2.parameters(), lr=0.001)
    net2.load_state_dict(torch.load(path)['model_state_dict'], strict=False)

    print("Evaluate with Dirichlet switches:\n")
    evaluate(net2, layer, test_loader)

    for name, param in net2.named_parameters():
        if not (name == "parameter"):
            param.register_hook(lambda grad: grad * 0)

    print("Dirichlet switch training:\n")
    net2.train()
    stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3); best_model=-1
    # while (stop<early_stopping):
    for epochs in range(epochs_num):
        epoch=epoch+1
        annealing_rate = beta_func(epoch)
        net2.train()
        for i, data in enumerate(train_loader):
            inputs, labels=data
            inputs, labels=inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, S=net2(inputs, layer) #when switc hes
            loss = loss_functionKL(outputs, labels, S, alpha_0, hidden_dim, args.batch_size, annealing_rate)
            loss.backward()
            #print(net2.c1.weight.grad[1, :])
            optimizer.step()
        print (loss.item())
        print ("Epoch " +str(epoch)+ " ended.")
        accuracy=evaluate(net2, layer, test_loader)

        print("S")
        print(S)
        print(torch.argsort(S))
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))

        if (accuracy<=best_accuracy):
            stop=stop+1
            entry[2]=0
        else:
            best_accuracy=accuracy
            print("Best updated")
            stop=0
            entry[2]=1
            best_model=net2.state_dict()
            best_optim=optimizer.state_dict()
            #torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
        #write
        # entry[0]=accuracy; entry[1]=loss
        # with open(filename, "a+") as file:
        #     file.write(",".join(map(str, entry))+"\n")
    return best_accuracy, epoch, best_model, S


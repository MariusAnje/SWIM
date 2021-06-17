import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, FakeSCrossEntropyLoss
from modules import SConv2d, SLinear
from tqdm import tqdm
import time
import argparse
import os
import numpy as np

def eval():
    total = 0
    correct = 0
    model.clear_noise()
    model.clear_mask()
    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return correct/total

def Seval():
    total = 0
    correct = 0
    with torch.no_grad():
        model.clear_noise()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs[0].argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def Seval_noise(var):
    total = 0
    correct = 0
    model.clear_noise()

    with torch.no_grad():
        model.set_noise(var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs[0].argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def STrain(epochs, header, verbose=False):
    best_acc = 0.0
    for i in range(epochs):
        running_loss = 0.
        running_l = 0.
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs, outputsS = model(images)
            loss = criteria(outputs, outputsS,labels)
            loss.backward()
            l = loss + model.fetch_S_grad()
            optimizer.step()
            running_loss += loss.item()
            running_l += l.item()
        test_acc = Seval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}, s: {(running_l - running_loss) / len(trainloader):-5.4f}")
        scheduler.step()

def GetSecond():
    model.clear_noise()
    optimizer.zero_grad()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1, 784)
        outputs, outputsS = model(images)
        loss = criteria(outputs, outputsS,labels)
        loss.backward()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

def load_state_and_S(filename):
    state_dict, gradS_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    for name in gradS_dict.keys():
        model._modules[name].weightS.grad = gradS_dict[name]

def tensor_information(tensor, name=""):
    _mean = tensor.view(-1).mean()
    _std = tensor.view(-1).std()
    _max = tensor.view(-1).max()
    print(f"tensor {name:5s} mean {_mean:E}, std {_std:E}, max {_max:E}")

def model_S_information():
    for name, m in model.named_modules():
        if isinstance(m, SConv2d) or isinstance(m, SLinear):
            tensor_information(m.weightS.grad.data, name)

def find_zeros(x:torch.Tensor):
    print((x == 0).view(-1).nonzero())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_mask', action='store',type=int, default=5,
            help='number of items being masked out')
    parser.add_argument('--num_noise', action='store',type=int, default=100,
            help='number of noise inference')
    args = parser.parse_args()

    print(args)



    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BS = 128

    trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=2)
    # model = SLeNet()
    model = SMLP4()

    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    criteria = SCrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])

    parent_path = "./pretrained"
    header = "1_MLP4"
    # no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}.pt"))
    # print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")

    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    model.clear_noise()
    load_state_and_S("tmp_gradS_MLP4.pt")
    model.clear_mask()
    print(f"S grad before masking: {model.fetch_S_grad().item():E}")
    # model_S_information()
    ori_acc = Seval()
    print(ori_acc)
    mask_acc_list = []
    for _ in range(args.num_noise):
        acc = Seval_noise(0.2)
        mask_acc_list.append(acc)
    
    print(f"mean: {np.mean(mask_acc_list)}")
    print(f"max: {np.max(mask_acc_list)}")
    print(f"min: {np.min(mask_acc_list)}")
    print(f"std: {np.std(mask_acc_list)}")
    # layer = model.fc2
    # indexes = torch.randint(0,1024,[args.num_mask]).numpy()
    # print(indexes)
    # P = layer.mask.view(-1)
    # P[indexes] *= 0
    # layer.mask = P.view(layer.mask.shape)
    # new_acc = Seval()
    # print(new_acc, new_acc - ori_acc)
    # print(f"S grad after  masking: {model.fetch_S_grad().item():E}")
    # GetSecond()
    # print(f"S grad after  masking: {model.fetch_S_grad().item():E}")
    # model_S_information()
    # print(layer.weightS.grad.data.view(-1)[indexes].cpu().numpy())
    # print(layer.op.weight.data.view(-1)[indexes].cpu().numpy())
    # mask_acc_list = []
    # for _ in range(args.num_noise):
    #     acc = Seval_noise(0.2)
    #     mask_acc_list.append(acc)
    
    # print(f"mean: {np.mean(mask_acc_list)}")
    # print(f"max: {np.max(mask_acc_list)}")
    # print(f"min: {np.min(mask_acc_list)}")
    # print(f"std: {np.std(mask_acc_list)}")
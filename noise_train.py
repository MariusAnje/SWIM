import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss
from modules import SModule
from tqdm import tqdm
import time
import argparse
import os

def CEval():
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs[0].argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEval(var):
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

def NEachEval(var):
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs[0].argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NTrain(epochs, header, var, verbose=False):
    best_acc = 0.0
    for i in range(epochs):
        running_loss = 0.
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs, outputsS = model(images)
            loss = criteria(outputs, outputsS,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = NEachEval(var)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--fine_epoch', action='store', type=int, default=20,
            help='# of epochs of finetuning')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--noise_var', action='store', type=float, default=0.1,
            help='noise variation')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet", "CIFAR"],
            help='model to use')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    args = parser.parse_args()

    print(args)
    header = time.time()
    header_timer = header
    parent_path = "./"
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128

    if args.model != "CIFAR":
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=2)
    else:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)


    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()
    elif args.model == "CIFAR":
        model = CIFAR()

    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    model.to_fake(device)
    # criteria = SCrossEntropyLoss()
    criteria = FakeSCrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    if not args.pretrained:
        NTrain(args.train_epoch, header, args.noise_var, args.verbose)
        state_dict = torch.load(f"tmp_best_{header}.pt")
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), f"saved_B_{header}.pt")

        no_mask_acc_list = []
        state_dict = torch.load(f"saved_B_{header}.pt")
        print(f"No mask no noise: {CEval():.4f}")
        model.load_state_dict(state_dict)
        model.clear_mask()
        model.back_real(device)
        model.push_S_device()
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(args.noise_var)
            no_mask_acc_list.append(acc)
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.noise_var}.pt")

        exit()
    else:
        parent_path = args.model_path
        header = args.header
        no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.noise_var}.pt"))
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        model.back_real(device)
        model.push_S_device()

    
    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    if args.model == "CIFAR":
        model.to_fake(device)
    model.load_state_dict(state_dict)
    model.back_real(device)
    model.push_S_device()
    criteria = SCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    NTrain(args.fine_epoch, header_timer, args.noise_var, args.verbose)
    print(f"Finetune no noise: {CEval():.4f}")
    fine_mask_acc_list = []
    loader = range(args.noise_epoch)
    for _ in loader:
        acc = NEval(args.noise_var)
        fine_mask_acc_list.append(acc)
    print(f"Finetune noise average acc: {np.mean(fine_mask_acc_list):.4f}, std: {np.std(fine_mask_acc_list):.4f}")
    model.clear_noise()

import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4
from tqdm import tqdm
import time
import argparse

def eval():
    total = 0
    correct = 0
    model.clear_noise()
    model.clear_mask()
    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return correct/total

def Seval(is_clear_mask=True):
    total = 0
    correct = 0
    with torch.no_grad():
        model.clear_noise()
        if is_clear_mask:
            model.clear_mask()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 784)
            outputs = model(images)
            predictions = outputs[0].argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def Seval_noise(var, is_clear_mask=True):
    total = 0
    correct = 0
    model.clear_noise()
    if is_clear_mask:
        model.clear_mask()
    with torch.no_grad():
        model.set_noise(var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 784)
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
            images = images.view(-1, 784)
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

def GetSecond():
    optimizer.zero_grad()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1, 784)
        outputs, outputsS = model(images)
        loss = criteria(outputs, outputsS,labels)
        loss.backward()


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
    parser.add_argument('--mask_p', action='store', type=float, default=0.01,
            help='portion of the mask')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=bool, default=False,
            help='see training process')
    args = parser.parse_args()

    print(args)
    header = time.time()
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

    model = SMLP4()
    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    criteria = SCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    STrain(args.train_epoch, header, args.verbose)

    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), f"saved_{header}.pt")

    no_mask_acc_list = []
    state_dict = torch.load(f"saved_{header}.pt")
    print(f"No mask no noise: {Seval(False):.4f}")
    model.load_state_dict(state_dict)
    model.clear_mask()
    loader = range(args.noise_epoch)
    for _ in loader:
        acc = Seval_noise(args.noise_var, False)
        no_mask_acc_list.append(acc)
    print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")

    state_dict = torch.load(f"saved_{header}.pt")
    model.load_state_dict(state_dict)
    GetSecond()
    mask_acc_list = []
    model.set_mask(args.mask_p)
    print(f"with mask no noise: {Seval(False):.4f}")
    loader = range(args.noise_epoch)
    for _ in loader:
        acc = Seval_noise(args.noise_var, False)
        mask_acc_list.append(acc)
    print(f"With mask noise average acc: {np.mean(mask_acc_list):.4f}, std: {np.std(mask_acc_list):.4f}")
    
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    STrain(args.fine_epoch, header, args.verbose)
    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), f"saved_{header}.pt")
    fine_mask_acc_list = []
    print(f"Finetune no noise: {Seval(False):.4f}")
    loader = range(args.noise_epoch)
    for _ in loader:
        acc = Seval_noise(args.noise_var, False)
        fine_mask_acc_list.append(acc)
    print(f"Finetune noise average acc: {np.mean(fine_mask_acc_list):.4f}, std: {np.std(fine_mask_acc_list):.4f}")
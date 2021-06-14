import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, FakeSCrossEntropyLoss
from tqdm import tqdm
import time
import argparse
import os

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
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet"],
            help='model to use')
    parser.add_argument('--header', action='store',type=int, default=0,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
    parser.add_argument('--use_mask', action='store',type=str2bool, default=True,
            help='if to do the masking experiment')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
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
    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()

    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    model.to_fake(device)
    # criteria = SCrossEntropyLoss()
    criteria = FakeSCrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    if not args.pretrained:
        STrain(args.train_epoch, header, args.verbose)

        # optimizer = optim.SGD(model.parameters(), lr=0.001)
        # STrain(args.train_epoch - 20, header, args.verbose)
        
        state_dict = torch.load(f"tmp_best_{header}.pt")
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), f"saved_B_{header}.pt")

        no_mask_acc_list = []
        state_dict = torch.load(f"saved_B_{header}.pt")
        print(f"No mask no noise: {Seval():.4f}")
        model.load_state_dict(state_dict)
        model.clear_mask()
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = Seval_noise(args.noise_var)
            no_mask_acc_list.append(acc)
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}.pt")
    
    else:
        parent_path = args.model_path
        header = args.header
        no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}.pt"))
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        model.back_real(device)
        model.push_S_device()

    
    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.back_real(device)
    model.push_S_device()
    criteria = SCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    model.clear_noise()
    GetSecond()
    print(f"S grad before masking: {model.fetch_S_grad().item():E}")
    
    if args.use_mask:
        mask_acc_list = []
        th = model.calc_S_grad_th(args.mask_p)
        model.set_mask(th, mode="th")
        print(f"with mask no noise: {Seval():.4f}")
        # GetSecond()
        print(f"S grad after  masking: {model.fetch_S_grad().item():E}")
        GetSecond()
        print(f"S grad after  masking: {model.fetch_S_grad().item():E}")
        # loader = range(args.noise_epoch)
        # for _ in loader:
        #     acc = Seval_noise(args.noise_var)
        #     mask_acc_list.append(acc)
        # print(f"With mask noise average acc: {np.mean(mask_acc_list):.4f}, std: {np.std(mask_acc_list):.4f}")
        
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
        STrain(args.fine_epoch, header_timer, args.verbose)

        if args.save_file:
            torch.save(model.state_dict(), f"saved_A_{header}_{header_timer}.pt")
        fine_mask_acc_list = []
        print(f"Finetune no noise: {Seval():.4f}")
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = Seval_noise(args.noise_var)
            fine_mask_acc_list.append(acc)
        print(f"Finetune noise average acc: {np.mean(fine_mask_acc_list):.4f}, std: {np.std(fine_mask_acc_list):.4f}")
        model.clear_noise()
        GetSecond()
        print(f"S grad after finetune: {model.fetch_S_grad().item():E}")
        os.system(f"rm tmp_best_{header_timer}.pt")

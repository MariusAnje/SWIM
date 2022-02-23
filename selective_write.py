import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from model_zoo.models import SMLP3, SMLP4, SLeNet, CIFAR 
import modules
from model_zoo.qmodels import QSLeNet, QCIFAR
from model_zoo import resnet
from model_zoo import qresnet
from model_zoo import qvgg
from model_zoo import qdensnet
from modules import SModule, SCrossEntropyLoss, FakeSCrossEntropyLoss
from tqdm import tqdm
import time
import argparse
import os

def CEval():
    """
    Function used for model inference on test dataset w/o device variation
    Very standard PyTorch fashion
    """
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEval(dev_var, write_var):
    """
    Function used for model inference on test dataset w/ device variation
    Used model.set_noise once for each epoch
    One run of this function offers one accuracy
    Needs to be run hundreds of times to collect an average accuracy
    """
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var, write_var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEachEval(dev_var, write_var):
    """
    Function used for model inference on test dataset w/ device variation
    Used model.set_noise once for each min-batch
    Fast evaluation of model average accuracy, runs only once and the results is an estimation of the average accuracy
    """
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NTrain(epochs, header, dev_var, write_var, verbose=False):
    """
    Function used for model training
    Very standard PyTorch fashion
    """
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def GetSecond():
    """
    Function used for collecting the second derivative of weights
    The second derivative value is stored in SModule.weightS.grad
    """
    model.eval()
    model.clear_noise()
    optimizer.zero_grad()
    for images, labels in secondloader:
        images, labels = images.to(device), labels.to(device)
        outputs, outputsS = model(images)
        loss = criteria(outputs, outputsS,labels)
        loss.backward()

def str2bool(a):
    """
    Transferring strings to boolean. Used in argparse.
    """
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
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.0,
            help='device variation [std] when training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--write_var', action='store', type=float, default=0.03,
            help='device variation [std] after write and verify')
    parser.add_argument('--mask_p', action='store', type=float, default=0.01,
            help='portion of the mask')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG"],
            help='model to use')
    parser.add_argument('--method', action='store', default="SM", choices=["second", "magnitude", "saliency", "random", "SM"],
            help='method used to calculate saliency')
    parser.add_argument('--alpha', action='store', type=float, default=1e6,
            help='weight used in saliency - substract')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=False,
            help='if to use pretrained model')
    parser.add_argument('--use_mask', action='store',type=str2bool, default=True,
            help='if to do the masking experiment')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    parser.add_argument('--calc_S', action='store',type=str2bool, default=True,
            help='if calculated S grad if not necessary')
    parser.add_argument('--div', action='store', type=int, default=1,
            help='division points for second')
    parser.add_argument('--layerwise', action='store',type=str2bool, default=False,
            help='if do it layer by layer')
    args = parser.parse_args()

    print("Experimental Setup: ", args)
    header = time.time()
    header_timer = header
    parent_path = "./"
    
    # set the device this experiments is running on
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128 # batch size in both training and test dataset

    # Preparing dataset for each model
    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        # dataloader used to calculate second derivatives, usually with smaller batch sizes
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                ])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/train', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=8)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/val',  transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=8)
    else:
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div,
                                                shuffle=False, num_workers=2)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=2)

    # Loading selected models
    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()
    elif args.model == "CIFAR":
        model = CIFAR()
    elif args.model == "Res18":
        model = resnet.resnet18(num_classes = 10)
    elif args.model == "TIN":
        model = resnet.resnet18(num_classes = 200)
    elif args.model == "QLeNet":
        model = QSLeNet()
    elif args.model == "QCIFAR":
        model = QCIFAR()
    elif args.model == "QRes18":
        model = qresnet.resnet18(num_classes = 10)
    elif args.model == "QDENSE":
        model = qdensnet.densenet121(num_classes = 10)
    elif args.model == "QTIN":
        model = qresnet.resnet18(num_classes = 200)
    elif args.model == "QVGG":
        model = qvgg.vgg11(num_classes = 200)
    else:
        NotImplementedError

    # Pushing models and other parameters into target device [similar to model.cuda()]
    # Note that noise and mask is not nn.Parameter so they should be pushed to the target device using a different functions shown below.
    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()

    # Selecting optimizers and schedulers for different models
    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])

    if not args.pretrained:
        # Training the selected model
        print("Start training models.")
        model.to_first_only() # You don't need to calculate second derivatives during training.
        NTrain(args.train_epoch, header, args.train_var, 0.0, args.verbose)
        if args.train_var > 0:
            state_dict = torch.load(f"tmp_best_{header}.pt")
            model.load_state_dict(state_dict)
        model.from_first_back_second()
        torch.save(model.state_dict(), f"saved_B_{header}.pt")
        state_dict = torch.load(f"saved_B_{header}.pt")
        print(f"Accuracy before masking w/o noise:: {CEval():.4f}")
        model.load_state_dict(state_dict)
        model.clear_mask()

        # Evaluate average model performance under device variations
        print("Performance of the trained model under variations:")
        no_mask_acc_list = []
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(args.dev_var, 0.0)
            no_mask_acc_list.append(acc)
        print(f"[{args.dev_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.dev_var}.pt")

        # Evaluate average model performance when all the weights are protected by write-verify
        no_mask_acc_list = []
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(args.write_var, 0.0)
            no_mask_acc_list.append(acc)
        print(f"[{args.write_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.write_var}.pt")

    else:
        print("Use pretrained model.")
        print("Performance of the pretrained model under variations:")
        parent_path = args.model_path
        header = args.header
        # Loading the experimental results for the pretrained model
        # Average accuracy under device variation w/o and w/ write-verify
        # This code still works if these results are not present
        try:
            no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.dev_var}.pt"))
            print(f"[{args.dev_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        except:
            print(f"[{args.dev_var}] Not Found")
        try:
            no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.write_var}.pt"))
            print(f"[{args.write_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        except:
            print(f"[{args.write_var}] Not Found")
        model.back_real(device)
        model.push_S_device()

    # Loading the model trained before or the pretrained model
    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.back_real(device)
    model.push_S_device()

    criteria = SCrossEntropyLoss() # A special lost function module that can calculate second derivatives
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    model.clear_noise()

    model.normalize() # Normalize model weights to [-1, +1] to fit the quantization scheme
    GetSecond() # Calculate the second derivative
    model.fine_S_grad()
    
    if args.use_mask:
        model.clear_mask()
        th = model.calc_sail_th(args.mask_p, args.method, args.alpha) # Calculate threshold of weight sensitivity according to the portion
        model.set_mask_sail(th, "th", args.method, args.alpha) # Mask out weights with sensitivities greater than the threshold

        total, RM_new = model.get_mask_info() # Calculate exactly how many weights are masked out
        print(f"Weights protected by write-verify: {RM_new/total * 100:.2f}%")
        model.de_normalize()
        
        print(f"Accuracy w/o noise: {CEval():.4f}")
        # Evaluate the final accuracy
        fine_mask_acc_list = []
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(args.dev_var, args.write_var)
            fine_mask_acc_list.append(acc)
        print(f"Accuracy after masking w/ noise, average: {np.mean(fine_mask_acc_list):.4f}, std: {np.std(fine_mask_acc_list):.4f}")

# SWIM

Code for paper _Zheyu Yan, X. Sharon Hu and Yiyu Shi, "SWIM: Selective Write-Verify for Computing-in-Memory Neural Accelerator," in Proc. of IEEE/ACM Design Automation Conference (DAC), 2022_  
Paper link: https://arxiv.org/abs/2202.08395  
This repo is implemented in Python3 using PyTorch framework.

## Requirements

1. PyTorch == 1.4.1 or >= 1.10
2. torchvision
3. numpy
4. Others stated in requirements.txt

## Usage

The major file of experiment is selective_write.py

### Training model from scratch

```shell
python selective_write.py --model QLeNet --train_epoch 20 --noise_epoch 100 --mask_p 0.1
```

Useful arguments:
```shell
--mask_p      # Portion of the weights protected by write-verify
--method      # "SM": SWIM method, also includes "random" and "magnitude"
--noise_epoch # Number of instances used to calculate average accuracy under variation
--train_epoch # Number of epochs used for training the original model
--train_var   # Device variation magnitude used in training
--dev_var     # Device variation magnitude for device w/o write-verify
--write_var   # Device variation magnitude for device w/  write-verify
--model       # DNN models for evaluations, includes LeNet for MNIST, ConvNet and ResNet-18 for CIFAR-10 and ResNet-18 for Tiny ImageNet. Also includes their quantized version. 
              # Naming: LeNet/QLeNet, CIFAR/QCIFAR (ConvNet), Res18/QRes18 (ResNet-18 for CIFAR-10), TIN/QTIN (ResNet-18 for Tiny ImageNet)
--div         # Divide the batch size by an integer when calculating second derivatives
```

### Using pretrained model
Useful arguments:
```shell
--pretrained  # A True value indicates we are using a pretrained model
--model_path  # The directory that stores the state dict of this model. Can also contain files that stores the results for model average accuracy before using SWIM
--header      # The file loaded to get the model should be named saved_B_{header}.pt
```

## Extending this repo
This repo is quite friendly for those who are familar with PyTorch.

### Implemented modules
All modules are implemented for efficient second derivative calculation.  
For each computational module (e.g., fully connected and convolutional), there is a wrapper module called "SModule". SModule.op is the wrapped operation (layer), SModule.weightS is a dummy placeholder to host the second derivative of its weight, SModule.noise is an instance of noise caused by device variations and SModule.mask is a mask that shows which weight is protected by write-verify.  
Note that the syntax for each layer wrapped by SModule is exactly the same as the PyTorch version, except that they takes two inputs: "input" and "inputS". "input" is the traditional input and inputS is a dummy placeholder to host the second derivative of "input".   
A list of modules implemented is:
1. SLinear
2. SConv2d
3. SBatchNorm2d
4. SCrossEntropyLoss
5. SReLU
6. SMaxPool2d
7. SAdaptiveAvgPool2d
8. SAvgPool2d

### Looking for second derivative
Second derivatives for each weight is stored in SModule.weightS.grad after running "GetSecond" function in selective_write.py. The first order derivatives are in SModule.op.weight.grad

### Wrting new models
Please refer to model_zoo/models.py to the syntax of writing basic convolutional models.  
Please refer to model_zoo/resnet.py to the syntax of writing skip connections.  
Please refer to model_zoo/qmodels.py to the syntax of writing quantized models.  


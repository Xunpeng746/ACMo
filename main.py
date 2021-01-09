'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import copy
import json

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from acutum_variants import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='one of SGD, Adam, Acutum')
parser.add_argument('--optimizer', default='Acutum', type=str, help='one of SGD, Adam, Acutum')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=300, type=float, help='epoch num')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

if args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

if args.dataset == 'CIFAR100':
    testset = testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=transform_test)
else:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

if args.dataset == 'CIFAR100':
    hackdataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
else:
    hackdataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
hackloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.dataset == 'CIFAR100':
    num_classes = 100
else:
    num_classes =10

# Model
print('==> Building model..')
# net = VGG('VGG16', num_classes=num_classes)
# net = ResNet50(num_classes=num_classes)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121(num_classes=num_classes)
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

net_adagrad = VGG('VGG16', num_classes=num_classes).to(device)
net_acutum = VGG('VGG16', num_classes=num_classes).to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


def optimizer_select(opt_name, net):
    opt_set = {
        'SGD': optim.SGD(net.parameters(), lr=0.1, momentum=0, weight_decay=args.wd),
        # optimal weight decay = 5e-4
        'Adam': optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd),
        # optimal weight decay = 1e-4
        'Amsgrad': optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd, amsgrad=True),
        'Adagrad': optim.Adagrad(net.parameters(), lr=0.001, weight_decay=args.wd),
        'AdamW': AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd),
        'Acutum': Acutum_Original(net.parameters(), lr=0.1, weight_decay=args.wd),
        'Acutum_OT': Acutum_OT(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Adabound': AdaBound(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Padam': Padam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd),
        'Acutum_MAMPV3': Acutum_MAMPV3(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_MAMPV4': Acutum_MAMPV4(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Lipschitz': Acutum_Lipschitz(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Adaptive': Acutum_Adaptive(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Truncated': Acutum_Truncated(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Truncated_V2': Acutum_Truncated_V2(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Truncated_V3': Acutum_Truncated_V3(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Acutum_Truncated_V4': Acutum_Truncated_V4(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Adam_SMT': Adam_SMT(net.parameters(), lr=args.lr, weight_decay=args.wd),
        'Yogi': Yogi(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd),
        'RAdam': RAdam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    }
    if opt_name not in opt_set:
        print("SGD go")
        return opt_set['SGD']
    return opt_set[opt_name]


optimizer_adagrad = optimizer_select(opt_name="Adagrad", net=net_adagrad)
optimizer_acutum = optimizer_select(opt_name="Acutum", net=net_acutum)
adagrad_f = open('result_adagrad', 'w')
acutum_f = open('result_acutum', 'w')

'''
{
    'nsgd': (更新后cur batch loss下降, 更新后其他batch loss上升)
    'sgd': (cur, other)
}
'''
adagrad_result = []
acutum_result = []

def hack_calculate(model, loader, inputs, targets, optimizer=None):
    if optimizer is None:
        model = copy.deepcopy(model)
        optimizer = optimizer_select(opt_name="SGD", net=model)
    
    # 先算更新前 loss_cur loss_all
    outputs = model(inputs)
    loss_cur = criterion(outputs, targets)
    loss_all = 0
    for batch_idx, (h_inputs, h_targets) in enumerate(loader):
        h_inputs, h_targets = h_inputs.to(device), h_targets.to(device)
        h_outputs = model(h_inputs)
        loss_all += criterion(h_outputs, h_targets).item()
    
    # 更新后的
    optimizer.zero_grad()
    loss_cur.backward()
    optimizer.step()
    
    outputs = model(inputs)
    new_loss_cur = criterion(outputs, targets)
    new_loss_all = 0
    for batch_idx, (h_inputs, h_targets) in enumerate(loader):
        h_inputs, h_targets = h_inputs.to(device), h_targets.to(device)
        h_outputs = model(h_inputs)
        new_loss_all += criterion(h_outputs, h_targets).item()
        
    return loss_cur.item()-new_loss_cur.item(), (new_loss_all-new_loss_cur.item())-(loss_all-loss_cur.item())
    

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_adagrad.train()
    net_acutum.train()
    train_loss = 0
    correct = 0
    total = 0
    loss = torch.tensor([0])
    if epoch % 50 == 0:
        print("Learning rate decay begin")
        lr = args.lr * (0.1 ** (epoch // 50))
        print("Learning rate = {} now".format(lr))
        for param_group in optimizer_adagrad.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_acutum.param_groups:
            param_group['lr'] = lr
        print("Learning rate decay done")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # hack to calculate sgd
        flag = random.random() < 0.005
        if flag:
            adagrad_result_sgd = hack_calculate(net_adagrad, hackloader, inputs, targets, None)
            acutum_result_sgd = hack_calculate(net_acutum, hackloader, inputs, targets, None)
            adagrad_result.append({'sgd': adagrad_result_sgd})
            acutum_result.append({'sgd': acutum_result_sgd})
            
            # train net_adagrad
            adagrad_result_nsgd = hack_calculate(net_adagrad, hackloader, inputs, targets, optimizer_adagrad)
            adagrad_result[-1]['nsgd'] = adagrad_result_nsgd
            
            # train net_acutum
            acutum_result_nsgd = hack_calculate(net_acutum, hackloader, inputs, targets, optimizer_acutum)
            acutum_result[-1]['nsgd'] = acutum_result_nsgd
            
            # print
            print('new sample')
            print(adagrad_result[-1])
            print(acutum_result[-1])
            adagrad_f.write(json.dumps(adagrad_result[-1])+'\n')
            acutum_f.write(json.dumps(acutum_result[-1])+'\n')
            adagrad_f.flush()
            acutum_f.flush()
            
        else:
            # train net_adagrad
            optimizer_adagrad.zero_grad()
            outputs = net_adagrad(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_adagrad.step()

            # train net_acutum
            optimizer_acutum.zero_grad()
            outputs = net_acutum(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_acutum.step()

        # note
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net_acutum.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_acutum(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net_acutum.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + int(args.epoch)):
    train(epoch)
    test(epoch)
adagrad_f.close()
acutum_f.close()

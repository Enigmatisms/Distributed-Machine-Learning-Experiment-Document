import os
from urllib import parse
import torch
from torch.distributed.distributed_c10d import get_rank, get_world_size 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

import argparse
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync
import dist_utils

class SubNetConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

    def forward(self, x_rref):
        x = x_rref
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        out = x.flatten(1)
        return out
    
    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

class SubNetFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x_rref):
        x = x_rref
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
    
    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

class ParallelNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # 分别远程声明SubNetConv和SubNetFC
        super().__init__()
        self.subnet_conv = rpc.remote("worker1",SubNetConv,args=(in_channels,))
        self.subnet_fc = rpc.remote("worker2",SubNetFC,args=(num_classes,))

    def forward(self, x):
        out_conv = self.subnet_conv.rpc_sync().forward(x)
        out_fc = self.subnet_fc.rpc_sync().forward(out_conv)
        return out_fc
    
    def parameter_rrefs(self):
        param_rrefs = []
        param_rrefs.extend(self.subnet_conv.rpc_sync().parameter_rrefs())
        param_rrefs.extend(self.subnet_fc.rpc_sync().parameter_rrefs())
        return param_rrefs

def train(model, dataloader, loss_fn, optimizer, num_epochs=2):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.
    model.train()
    dist_utils.init_parameters(model)
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            with dist_autograd.context() as context_id:
                inputs, labels = batch_data
                # Forward pass
                outputs_rref = model(inputs)
                # Compute loss locally
                loss = loss_fn(outputs_rref, labels)
                # Backward pass
                dist_autograd.backward(context_id,[loss])
                # Update parameters
                optimizer.step(context_id)
        loss_total += loss.item()
    
    print("Training Finished!")

def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))


def main():
    args = parse_args()
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    DATA_PATH = "./data"
    if args.rank == 0:
        
        rpc.init_rpc("worker0", rank=args.rank, world_size=args.n_devices)
        # construct the model
        model = ParallelNet(in_channels=1, num_classes=10)
        # construct the dataset
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

        # construct the loss_fn and optimizer
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        dist_optimizer = DistributedOptimizer(torch.optim.SGD, model.parameter_rrefs(), lr=0.01)

        train(model, train_loader, loss_fn, dist_optimizer)
        test(model, test_loader)
    
    elif args.rank == 1:
        rpc.init_rpc("worker1", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker1...")

    elif args.rank == 2:
        rpc.init_rpc("worker2", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker2...")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')
    # Division options
    parser.add_argument('--mode', default='division', type=str,help='The mode of dataset division')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
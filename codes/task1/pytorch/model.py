import os
import sys
sys.path.append("../../../")
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from MyOptimizer import GdOptimizer, AdamOptimizer
from codes.datawriter import getSummaryWriter
from math import sqrt

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)
        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def train(model, dataloader, optimizer, loss_fn, num_epochs=1, batch_size = 1):
    print("Start training ...")
    loss_total = 0.
    model.train()
    writer = getSummaryWriter(num_epochs, False)
    train_cnt = 0
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            # with dist_autograd.context() as context_id:
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_total += loss.item()
            if i % 20 == 19:    
                writer.add_scalar('Train Loss', loss_total / 20, train_cnt)
                train_cnt += batch_size
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0
        print(f"Finished epoch: {epoch + 1:3d} / {num_epochs:3d}")
    
    print("Training Finished!")
    writer.close()

def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def main():
    model = Net(in_channels=1, num_classes=10)
    model.cuda()

    DATA_PATH = "./data"

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    batch_size = 200
    epochs = 1
    lr = 5e-4 * sqrt(batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = GdOptimizer(model.parameters(), lr=lr)
    optimizer = AdamOptimizer(model.parameters(), lr=lr, b1=0.9, b2=0.999)

    train(model, train_loader, optimizer, loss_fn, num_epochs = epochs, batch_size = batch_size)
    test(model, test_loader)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
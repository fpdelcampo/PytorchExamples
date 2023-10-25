import numpy as np
import torch
from torch.fx import symbolic_trace
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # Produces 64 x (28 - 3 + 1) x (28 - 3 + 1) -> 64 x 13 x 13
        x = self.pool(torch.relu(self.conv2(x))) # Produces 32 x (13 - 3 + 1) x (13 - 3 + 1) -> 32 x (11//2) x (11//2)
        x = self.pool(torch.relu(self.conv3(x))) # Produces 32 x (5 - 3 + 1) x (5 - 3 + 1) -> 32 x (3//2) x (3//2)
        x = torch.relu(self.fc1(torch.flatten(x, 1)))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(x, dim=1)
        return x

cnn = CNN()

criterion = nn.NLLLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)

epochs = 10
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = cnn(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss/len(trainloader)}")

test_loss = 0
for image, label in testloader:
    out = cnn(image)
    loss = criterion(out, label)
    test_loss += loss.item()
print(f"Final test loss is {test_loss/len(testloader)}")
#torch.save(cnn.state_dict(), 'cnn.pth')

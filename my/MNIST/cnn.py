import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LOG_INTERVEL = 10
LEARNING_RATE = 0.01
EPOCH_NUM = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train_dataset = datasets.MNIST('data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                               ])
                    )
mnist_test_dataset = datasets.MNIST('data', train=False,
                                transform=transforms.Compose
                               ([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                ])
                    )
train_loader = torch.utils.data.DataLoader(
    mnist_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    mnist_test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True
)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, loss_fn, optimizer, epoch):
    model.train() # 注意！
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 对应的数据放到CPU或GPU进行计算
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVEL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

my_cnn = CNNNet()
optimizer = optim.SGD(my_cnn.parameters(), lr=LEARNING_RATE)
loss_fn = F.nll_loss

for epoch in range(1, EPOCH_NUM+1):
    train(my_cnn, device, train_loader, loss_fn, optimizer, epoch)

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
#             print(output.detach().numpy().shape)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             print(pred.numpy().shape)
#             print(target.numpy().shape)
#             这里的view_as是让target和pred的维度保持一致，target是（1000，）而pred是（1000,1）
#             需要这一个进行调整
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCH_NUM+1):
    test(my_cnn, device, test_loader, loss_fn)
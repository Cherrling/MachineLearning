import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # 新增


batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# fig = plt.figure()
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(train_dataset.targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.savefig('mnist.png')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, 
                            out_channels=10,
                            kernel_size=5,),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
model = Net().to('cuda')


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 设置 TensorBoard 日志记录器
writer = SummaryWriter('logs/mnist_experiment')

def train(epoch):
    running_loss = 0.0  
    running_total = 0
    running_correct = 0
    
    
    progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, data in progress_bar:
        input, target = data
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        
        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred = torch.max(output, 1)
        running_total += target.size(0)
        running_correct += (pred == target).sum().item()
        
        if batch_idx % 300 == 299:
            running_loss = 0.0
        loss = running_loss / (batch_idx+1)
        acc = running_correct / running_total
        progress_bar.set_postfix(loss=  loss, acc= acc)
          # 记录损失和准确率到 TensorBoard
        writer.add_scalar('training loss', loss, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training accuracy', acc, epoch * len(train_loader) + batch_idx)
        
        
        
        
def test(epoch):
    correct = 0 
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            images ,labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            
            output = model(images)
            _, pred = torch.max(output.data, dim = 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
        acc = correct / total
        print(f"Accuracy on test set: {100 * acc:.2f}%")
        writer.add_scalar('test accuracy', acc, epoch)
        return acc
        

if __name__ == "__main__":
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test(epoch)
        acc_list_test.append(acc_test)

    torch.save(model.state_dict(), 'mnist_cnn.pth')
    writer.close()






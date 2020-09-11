import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

download_path = '/Users/cao.yumin/Desktop/DL_fd/assign2'
train_data = torchvision.datasets.CIFAR10(root=download_path+'.data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root=download_path+'.data', train=False, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class VGG16(nn.Module):
    def __init__(self,n_classes):
        super(VGG16,self).__init__()
        # conv layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # max pooling (kernel size, stride)
        self.pool = nn.MaxPool2d(2,2)
        
        # batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 10)
        
    def forward(self, x, training=True):
        x = self.bn1(self.conv1_1(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn1(self.conv1_2(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.1, training=training)
        
        x = self.bn2(self.conv2_1(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn2(self.conv2_2(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.2, training=training)
        
        x = self.bn3(self.conv3_1(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn3(self.conv3_2(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn3(self.conv3_3(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.3, training = training)
        
        x = self.bn4(self.conv4_1(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn4(self.conv4_2(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn4(self.conv4_3(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.4, training = training)
        
        x = self.bn4(self.conv5_1(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn4(self.conv5_2(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.bn4(self.conv5_3(x))
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.5, training = training)
        
        x = x.view(-1, 1*1*512)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = F.leaky_relu(self.fc7(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = self.fc8(x)
        
        return x

class VGG(nn.Module):
    def __init__(self,n_classes):
        super(VGG,self).__init__()
        # conv layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # max pooling (kernel size, stride)
        self.pool = nn.MaxPool2d(2,2)
        
        # batch normalization
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(512)
        
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 10)
        
    def forward(self, x, training=True):
        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.1, training=training)
        
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.2, training=training)
        
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_3(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.3, training = training)
        
        x = F.leaky_relu(self.conv4_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4_3(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.4, training = training)
        
        x = F.leaky_relu(self.conv5_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5_3(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.dropout(x, 0.5, training = training)
        
        x = x.view(-1, 1*1*512)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = F.leaky_relu(self.fc7(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = self.fc8(x)
        
        return x

class VGG1(nn.Module):
    def __init__(self,n_classes):
        super(VGG1,self).__init__()
        # conv layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # max pooling (kernel size, stride)
        self.pool = nn.MaxPool2d(2,2)
        
        # batch normalization
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(512)
        
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 10)
        
    def forward(self, x, training=True):
        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.01)
        x = self.pool(x)
        # x = F.dropout(x, 0.1, training=training)
        
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.01)
        x = self.pool(x)
        # x = F.dropout(x, 0.2, training=training)
        
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_3(x), negative_slope=0.01)
        x = self.pool(x)
        # x = F.dropout(x, 0.3, training = training)
        
        x = F.leaky_relu(self.conv4_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4_3(x), negative_slope=0.01)
        x = self.pool(x)
        # x = F.dropout(x, 0.4, training = training)
        
        x = F.leaky_relu(self.conv5_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5_3(x), negative_slope=0.01)
        x = self.pool(x)
        # x = F.dropout(x, 0.5, training = training)
        
        x = x.view(-1, 1*1*512)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = F.leaky_relu(self.fc7(x), negative_slope=0.01)
        x = F.dropout(x, 0.5, training=training)
        x = self.fc8(x)
        
        return x

net1 = VGG16(10)
net_without_bnorm = VGG(10)
net_without_bnorm_dpout = VGG1(10)

criterion = nn.CrossEntropyLoss()
temp_list = []
for net in [net1, net_without_bnorm, net_without_bnorm_dpout]:
  for lr in [0.001,0.01]:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for k in [2,3,4]:
      for epoch in range(k):  # loop over the dat           aset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          # zero the parameter gradients
          optimizer.zero_grad()                                       

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 2000 == 1999:    # print every 2000 mini-batches
              # print('[%d, %5d] loss: %.3f' %
              #       (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0

      print(f'Finished Training{k}')

      # PATH = download_path+'/cifar_net.pth'
      # torch.save(net.state_dict(), PATH)

      correct = 0
      total = 0
      with torch.no_grad():
          for data in testloader:
              images, labels = data
              outputs = net(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

      print('Accuracy of the network on the 10000 test images: %d %%' % (
          100 * correct / total))

      class_correct = list(0. for i in range(10))
      class_total = list(0. for i in range(10))
      with torch.no_grad():
          for data in testloader:
              images, labels = data
              outputs = net(images)
              _, predicted = torch.max(outputs, 1)
              c = (predicted == labels).squeeze()
              for i in range(4):
                  label = labels[i]
                  class_correct[label] += c[i].item()
                  class_total[label] += 1

      for i in range(10):
        print('Number of %5s : %2d' % (
        classes[i],  class_total[i]))
        
      for i in range(10):
          print('Accuracy of %5s : %2d %%' % (
              classes[i], 100 * class_correct[i] / class_total[i]))



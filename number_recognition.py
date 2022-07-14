import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np

batch_size = 200  # 分批训练数据、每批数据量
learning_rate = 1e-3  # 学习率
num_epoches = 20  # 训练次数
# 预处理
data_tf = transforms.Compose(
    [
        transforms.ToTensor(),  # 对原有数据转成Tensor类型
        transforms.RandomRotation(2),  # 随机旋转
        # transforms.Normalize([0.5], [0.5])  # 用平均值和标准偏差归一化张量图像
    ]
)
DOWNLOAD_MNIST = False  # 如果并未下载，改成True
train_dataset = datasets.MNIST(
    root="./mnist",
    train=True,  # download train data
    transform=data_tf,
    download=DOWNLOAD_MNIST,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle 是否打乱加载数据


class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, kernel_size=3, stride=1, padding=1),
            # input shape(1*28*28),(28+1*2-3)/1+1=28 卷积后输出（6*28*28）
            # 输出图像大小计算公式:(n*n像素的图）(n+2p-k)/s+1
            nn.ReLU(True),  # 激活函数
            nn.MaxPool2d(2, 2),  # 28/2=14 池化后（6*14*14）
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # (14-5)/1+1=10 卷积后（16*10*10）
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 池化后（16*5*5）=400，the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out


net = CNN(1, 10)
print(net)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion = nn.CrossEntropyLoss()  # 多分类用的交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def train():
    for epoch in range(num_epoches):
        print('epoch{}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        # 训练
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            out = net(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / len(train_dataset)
        ))

    torch.save(net.state_dict(), './models/model1.pth')


if __name__ == '__main__':
    train()

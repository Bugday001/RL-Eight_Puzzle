from number_recognition import CNN, data_tf
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWNLOAD_MNIST = False  # 如果并未下载，改成True
test_dataset = datasets.MNIST(
    root='./mnist',
    train=False,        #download test data
    transform=data_tf,
    download=DOWNLOAD_MNIST
)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

model = CNN(1, 10)# 实例化模型
model.load_state_dict(torch.load('./models/model1.pth'))# 加载权重参数

model.to(device)   # GPU模式需要添加
# print(model) #　输出模型信息


# ------------在整个测试集上测试-------------------------------------------
correct = 0 # 测试机中测试正确的个数
total = 0 # 测试集总共的样本个数
count = 0 # 共进行了count个batch = total/batch_size
with torch.no_grad():
    for images, labels in test_loader:

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # forward
        pre_labels = model(images) # 100*10

        _, pred = torch.max(pre_labels, 1) # 100*1
        correct += (pred == labels).sum().item() # 正确的个数
        total += labels.size(0) # 计算测试集中总样本个数
        count += 1 # 记录测试集共分成了多少个batch
        print("在第{0}个batch中的Acc为：{1}" .format(count, correct/total))

# 总体平均 Acc
accuracy = float(correct) / total
print("======================  Result  =============================")
print('测试集上平均Acc = {:.5f}'.format(accuracy))
print("测试集共样本{0}个，分为{1}个batch，预测正确{2}个".format(total, count, correct))

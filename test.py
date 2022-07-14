import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image
from number_recognition import CNN
import matplotlib.pyplot as plt

net = CNN(1, 10)
net.load_state_dict(torch.load('./models/model1.pth'))
input_image = './img/0_7.jpg'

im = Image.open(input_image).resize((28, 28))  # 取图片数据
im = im.convert('L')  # 灰度图
im_data = np.array(im)

im_data = torch.from_numpy(im_data).float()
tf = torchvision.transforms.Normalize([0.5], [0.5])
im_data = im_data.view(1, 1, 28, 28)
im_data = tf(im_data)
out = net(im_data)
_, pred = torch.max(out, 1)
print(out)
print('预测的数字是：{}。'.format(pred))

plt.figure("dog")
plt.imshow(im.convert("RGB"))
plt.show()
# tensor([[-3097.7283,   258.6362,  -677.3108,  1798.7858,   212.7564,  -781.6487,
#          -4395.8403,  4200.6802,   -29.2207,  2058.1138]],
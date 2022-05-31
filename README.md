# m1_AlexNet
这里代码参考于网络流传的博客，原始出处已经无从考证
代码里的网络结构:
```
#网络模型构建
#注意：这里每层与原版参数不一样，输入为65x65，后续计算也不一样，仅仅结构相同
class AlexNet(nn.Module):
    def __init__(self,num_classes=2):                                       #分类数量,原论文为1000
        super(AlexNet, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,48, kernel_size=11),                                #48个卷积核 ,11x11x3，步长1，原论文为96,步长4
            nn.ReLU(inplace=True),                                          #卷积后通过ReLU
            nn.MaxPool2d(kernel_size=3,stride=2),                           #池化3x3,步长2
            nn.Conv2d(48,128, kernel_size=5, padding=2),                    #5x5x48的128个卷积核，步长2
            nn.ReLU(inplace=True),                                          #卷积后通过ReLU
            nn.MaxPool2d(kernel_size=3,stride=2),                           #池化3x3,步长2
            nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),            #3x3x128的192个卷积核
            nn.ReLU(inplace=True),                                          #卷积后通过ReLU
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),                           #池化3x3，步长2，这时的输出已经为6x6x128
        )
        self.classifier=nn.Sequential(
            nn.Linear(6*6*128,2048),                                        #全连接
            nn.ReLU(inplace=True),                                          #通过ReLu
            nn.Dropout(0.5),                                                #随机失活一半
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,num_classes),
        )
 
 
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)                                      #展开，进入全连接层
        x=self.classifier(x)
 
        return x

```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 48, 55, 55]          17,472
              ReLU-2           [-1, 48, 55, 55]               0
         MaxPool2d-3           [-1, 48, 27, 27]               0
            Conv2d-4          [-1, 128, 27, 27]         153,728
              ReLU-5          [-1, 128, 27, 27]               0
         MaxPool2d-6          [-1, 128, 13, 13]               0
            Conv2d-7          [-1, 192, 13, 13]         221,376
              ReLU-8          [-1, 192, 13, 13]               0
            Conv2d-9          [-1, 192, 13, 13]         331,968
             ReLU-10          [-1, 192, 13, 13]               0
           Conv2d-11          [-1, 128, 13, 13]         221,312
             ReLU-12          [-1, 128, 13, 13]               0
        MaxPool2d-13            [-1, 128, 6, 6]               0
           Linear-14                 [-1, 2048]       9,439,232
             ReLU-15                 [-1, 2048]               0
          Dropout-16                 [-1, 2048]               0
           Linear-17                 [-1, 2048]       4,196,352
             ReLU-18                 [-1, 2048]               0
          Dropout-19                 [-1, 2048]               0
           Linear-20                    [-1, 2]           4,098
================================================================
Total params: 14,585,538
Trainable params: 14,585,538
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 5.52
Params size (MB): 55.64
Estimated Total Size (MB): 61.21
----------------------------------------------------------------

```


使用的数据集下载:https://pan.baidu.com/s/1rPZzQTE00r8lnc9Ott9j2Q?pwd=n5im 


# ALexNet结构
![consturcture](https://user-images.githubusercontent.com/74494790/170641650-415ddb18-b9a1-4d9b-a88c-ce7ec1bad009.png)


输入:三通道224x224x3彩色图像，经过11x11x3，步长为4的96个卷积核对224x224x3的图像卷积，输出图像大小为$\frac {227-11} {4}+1=55$（每个卷积核的参数是不同的，处理出的图像也不同，有几个就输出几维）最终输出$55 * 55 * 96$的特征响应图(在上图论文原图中，由于使用两个GPU，拆成了两个48）.

接下来经过3x3,stride=2的池化层,输出大小:$\frac {55-3}{2}+1=27$  特征图为27x27x96

接下来经过256个5x5x96的same卷积核,填充2,输出大小:$\frac {27-5+2*2}1+1=27$，然后通过3x3,s=2的池化，此时输出为13x13x256

接下来通过一次384个3x3x256 same卷积核 输出尺寸$\frac {13-3+2*1}{1}+1=13$

再通过一次384个3x3 x192 same卷积核(论文中第四层，这里有个问题，论文说是384个3x3x192，通道数咋对不上了，我认为这里应该是3x3x384),输出尺寸13

再通过一次256个13x13x192 same卷积核(这里也有点问题，同上)，输出尺寸13

接下来进行最大池化，3x3,s=2,输出尺寸$\frac {13-3} {2}+1=6$

接下来进行全连接9216连到4096再连到4096，然后练到1000,然后使用softmax.



注意：每次卷积后及全连接后需要通过relu，全连接中间有随机失活。

2.ALexNet创新点

概述

1. 成功使用ReLU作为CNN的激活函数，验证了其效果在较深的网络中超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题。
2. 训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合，一般在全连接层使用，在预测的时候是不使用Dropout的，即Dropout为1.
3. 在CNN中使用重叠的最大池化(步长小于卷积核)。此前CNN中普遍使用平均池化，使用最大池化可以避免平均池化的模糊效果。
4. 同时重叠效果可以提升特征的丰富性。 提出LRN（Local Response Normalization，即局部响应归一化）层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
5. 使用CUDA加速神经网络的训练，利用了GPU强大的计算能力。

**线性整流函数（ReLU）激活函数**

1. 在进行深层神经网络的时候，使用sigmoid与tanh函数作为激活函数，由于这两个激活函数都含有饱和区，会导致产生梯度爆炸和梯度消失。
2. 反而而使用ReLU函数作为激活函数会缓解梯度消失和梯度爆炸这两个问题。 ReLU是线性的，且导数始终为1，计算量大大减少，收敛速度会比Sigmoid/tanh快很多。从而加快训练的速度。 但因为表达式中还有各层的权重，所以ReLU没有彻底解决梯度消失问题。

**重叠池化（Overlapping Pooling）**

1. 一般的池化（Pooling）是不重叠的，池化区域的窗口大小与步长相同。
2. 在AlexNet中使用的池化（Pooling）却是可重叠（Overlapping）的，也就是说，在池化的时候，每次移动的步长小于池化的窗口长度。
3. AlexNet池化的大小为3×3的正方形，每次池化移动步长为2，这样就会出现重叠。

**采用重叠池化的优点：**

1. 不仅可以提升预测精度，同时一定程度上可以减缓过拟合。
2. 相比于正常池化（步长s=2，窗口z=2） 重叠池化(步长s=2，窗口z=3) 可以减少top-1, top-5分别为0.4% 和0.3%；
3. 重叠池化可以避免过拟合。

**LRN局部响应归一化**(BN更好用,BN于ResNet中被提出)

局部响应归一化（Local Response Normalization，LRN），提出于2012年的AlexNet中。首先要引入一个神经生物学的概念：侧抑制（lateral inhibitio），即指被激活的神经元抑制相邻的神经元。LRN仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制）。归一化（normaliazation）的目的就是“抑制”,LRN就是借鉴这种侧抑制来实现局部抑制，尤其是我们使用RELU的时候，这种“侧抑制”很有效 ，因而在Alexnet里使用有较好的效果。

优点有以下两点：

1. 归一化有助于快速收敛；
2. 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

LRN一般是在激活、池化后进行的一中处理方法。 LRN函数类似DROPOUT和数据增强作为relu激励之后防止数据过拟合而提出的一种处理方法。这个函数很少使用，基本上被类似DROPOUT这样的方法取代






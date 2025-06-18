import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
torch.manual_seed(10)#固定每次初始化模型的权重
training_step = 500#迭代此时
batch_size = 192#每个批次的大小
n_features = 21#特征数目
M = 10000#生成的数据数目
#生成数据
df_AU = pd.read_excel('./AU_info_obj.xlsx').iloc[:, :].values
df = pd.read_excel('F:dataset\CASMEII\CASME2_preprocessed_Li Xiaobai\CASME2-coding-20190701.xlsx', 
                       usecols=[0, 1, 3, 4, 5, 7, 8])
Label_all = df.iloc[:,6].values
data = []
target = []
for (label_all, label_au) in zip(Label_all, df_AU):

            if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':


                data.append(label_au)
                if label_all == 'happiness':
                    target.append(0)
                elif label_all == 'repression':
                    target.append(1)
                elif label_all == 'disgust':
                    target.append(2)
                elif label_all == 'surprise':
                    target.append(3)
                else:
                    target.append(4)

data = np.array(data)
target = np.array(target)

# 对训练集进行切割，然后进行训练
x_train,x_val,y_train,y_val = train_test_split(data,target,test_size=0.2,shuffle=False)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)
#定义网络结构
class Net(torch.nn.Module):  # 继承 torch 的 Module

    def __init__(self, n_features):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.l1 = nn.Linear(n_features,500)#特征输入
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.ReLU()#激活函数
        self.l3 = nn.BatchNorm1d(500)#批标准化
        self.l4 = nn.Linear(500,250)
        self.l5 = nn.ReLU()
        self.l6 = nn.BatchNorm1d(250)
        self.l7 = nn.Linear(250,5)
        #self.l8 = nn.Sigmoid()
    def forward(self, inputs):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        out = torch.from_numpy(inputs).to(torch.float32)#将输入的numpy格式转换成tensor
        out = self.l1(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.dropout(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        #out = self.l8(out)
        return out


#定义模型
model = Net(n_features=n_features)

#定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
#定义目标损失函数
loss_func = torch.nn.CrossEntropyLoss() #这里采用均方差函数

#开始迭代
for step in range(training_step):
    M_train = len(x_train)
    with tqdm(np.arange(0,M_train,batch_size), desc='Training...') as tbar:
        for index in tbar:
            L = index
            R = min(M_train,index+batch_size)
            #-----------------训练内容------------------
            train_pre = model(x_train[L:R,:])     # 喂给 model训练数据 x, 输出预测值
            _,predicts = torch.max(train_pre,1)
            yt_temp = torch.from_numpy(y_train[L:R])
            yt_temp = yt_temp.type(torch.LongTensor)
            correct_num = torch.eq(predicts,yt_temp).sum().cpu()
            train_acc = correct_num/len(predicts)
            train_loss = loss_func(train_pre, yt_temp).to(torch.float32)
            val_pre = model(x_val)
            _,vpred = torch.max(val_pre,1)
            val_correct = torch.eq(vpred,y_val).sum().cpu()
            val_acc = val_correct/len(vpred)
            val_loss = loss_func(val_pre, y_val).to(torch.float32)
            #-------------------------------------------
            tbar.set_postfix(train_loss=float(train_loss.data),val_loss=float(val_loss.data),train_acc = train_acc,val_acc = val_acc)#打印在进度条上
            tbar.update()  # 默认参数n=1，每update一次，进度+n

            #-----------------反向传播更新---------------
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上



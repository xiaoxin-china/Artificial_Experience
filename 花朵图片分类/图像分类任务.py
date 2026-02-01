#花朵分类任务
import torch
from torch import nn
import torch.optim as optim #优化器
from torchvision import transforms  #数据比较少，用来数据增强
from torchvision import models  #用现成的模型resNet
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import time
import copy
#---------------------------
#超参数
batch_size = 64
model_name = 'resnet50'
feature_extract = True
num_classes = 102#分多少类
LR = 0.001
Step_Size = 7#学习率每几次衰减一次
Gamma = 0.1#学习率衰减比例
Epochs = 20
#----------------------------
#读数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + ('/valid')

#数据预处理
data_transforms = {
    'train':
        #Compose:按顺序去做我接下来这些事
        transforms.Compose([
            transforms.Resize([256,256]),#因为数据的图片长宽都不是统一的，这里让所有数据大小统一
            #由于数据太少，下面开始进行数据增强，生成更多张图片
            transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选，指定个小角度45、30、15都行
            transforms.CenterCrop(128),#从中间开始裁剪，最后输入数据大小为裁剪后的大小128*128
            transforms.RandomHorizontalFlip(p = 0.5),#水平翻转，选取一个概率
            transforms.RandomVerticalFlip(p = 0.5),#随机垂直反转
            transforms.ColorJitter(0.1,0.1,0.1,0.1),#亮度，对比度，饱和度，色相，不过改颜色用的不大多
            transforms.RandomGrayscale(p = 0.025),#概率转换成灰度率，3通道就是RGB，可能转换成RRR、GGG、BBB，不过极不常见，一般不考虑
            transforms.ToTensor(),#转化成张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#标准化，均值，标准差，分别对RGB做标准化求得均值和方差
        ]),
    'valid':
        #验证集不用做数据增强，以实际为准，像考试一样
        transforms.Compose([
            transforms.Resize([128,128]),#训练集裁剪的多么大，测试集就多么大
            transforms.ToTensor(),#转换成张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#验证集标准化的均值和标准差和训练集一样
        ])

}


#dataloader定义加载
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])for x in ['train','valid']}#字典类型,通过现成文件夹来指定
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size = batch_size,shuffle=True)for x in ['train','valid']}
dataset_sizes = {x:len(image_datasets[x])for x in ['train','valid']}
class_names = image_datasets['train'].classes


#-----------------------
#模型定义
#model_name = 'resnet'#可选的比较多，【'resnet','alexnet','vgg','squeezenet','densenet','inception'】
#是否用人家训练好的特征,都冻住，参数权重不更新
#feature_extract = True

#训练设备
train_on_GPU = torch.cuda.is_available()
if train_on_GPU:
    print('Training on GPU')
else:
    print('Training on CPU')
device = torch.device('cuda:0'if train_on_GPU else 'cpu')


#-----------------------
#数据适配预训练模型调整
def set_parameter_requires_grad(model, feature_extracting):#把权重参数冻起来，不改变
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#加载预训练模型
def initialize_model(model_name,num_classes,feature_extract,use_pretrained = True):
    model_ft = getattr(models,model_name)(pretrained = use_pretrained)#用不用人家训练好的权重参数当作我们初始化
    set_parameter_requires_grad(model_ft, feature_extract)#把权重参数冻住，不更新

    #重新定义全连接层，因为和已有模型最后输出是不一样的
    num_ftrs = model_ft.fc.in_features#全连接层的输入是多少
    model_ft.fc = nn.Linear(num_ftrs,num_classes)#输入大小维度还是一样的，输出改成我自己的分102类
    #新加的一层全连接，默认是反向传播的，所以权重会进行更新
    return model_ft



#————————————————————--————
#模型组合
#模型初始化
model_ft = initialize_model(model_name,num_classes,feature_extract,use_pretrained = True)
model_ft.to(device)
#模型保存
Filename = 'my_model.pth'
#是否训练所有层
params_to_update = model_ft.parameters()#保存下所有参数（包括冻住的和没冻住的）
if feature_extract:
    params_to_update = []#它存在的意义是，一会传进去告诉优化器一会要更新哪个参数
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)#它存在的意义是，一会传进去告诉优化器一会要更新哪个参数
            print('\t',name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print('\t',name)
#———————————————————————————————
#优化器
optimizer = optim.Adam(params_to_update, lr=LR)#把要更新的东西传去，此处迁移学习只更新全连接层最后一层，也可以全学
#学习率衰减策略
#学习率衰减策略其中一种方法：StepLR：你告诉我啥时候衰减，我就按照你的比例衰减一次
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Step_Size, gamma=Gamma)#学习率每7个衰减成原先的十分之一


#_______________________________
#损失函数
criterion = nn.CrossEntropyLoss()#损失函数，分类的，用交叉熵



#——————————————————————————————————
#训练模块
def train_model(model,dataloaders,criterion,optimizer,num_epochs=Epochs,filename = Filename):
    #计算时间
    since = time.time()
    #最好的一次准确率
    best_acc = 0
    model.to(device)
    #训练中一些指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    #学习率
    LRs = [optimizer.param_groups[0]['lr']]
    #最好的那次模型，先初始化，后续结果越来越好就再赋值
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()#梯度清零
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                _,preds = torch.max(outputs, 1)
                #训练阶段更新权重
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #计算损失
                running_loss += loss.item() * inputs.size(0)#当前batch损失
                running_corrects += torch.sum(preds == labels.data)#正确的个数

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()  # 学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=20,filename=Filename)
import os, time, copy
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt

"""
阶段1：只训练FC
阶段2：解冻全模型做微调
"""
# ========== 0) 超参数 ==========
data_dir = "./flower_data_dataloader"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")

batch_size = 64
num_classes = 102
model_name = "resnet50"

feature_extract = True      # True=冻结骨干网络，只训练最后的fc层；False=全模型微调
lr = 1e-3
epochs = 20
step_size = 7
gamma = 0.1

save_path = "my_model.pth"

# ========== 1) 设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== 2) 数据预处理（关键：与预训练模型输入保持一致） ==========
# ResNet预训练通常用224x224 + ImageNet均值方差
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),                       # 先把短边缩到256（保持比例）
        transforms.RandomResizedCrop(224),             # 随机裁剪到224（增强）
        transforms.RandomHorizontalFlip(),             # 随机水平翻转
        transforms.RandomRotation(30),                 # 小角度旋转
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),     # 轻微颜色扰动
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
    "valid": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                    # 验证集固定中心裁剪（稳定评估）
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
}

# ========== 3) 数据集与 DataLoader ==========
image_datasets = {
    "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
    "valid": datasets.ImageFolder(valid_dir, data_transforms["valid"]),
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True if x == "train" else False,  # valid 不需要 shuffle
        num_workers=0
    )
    for x in ["train", "valid"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
class_names = image_datasets["train"].classes

print("Train size:", dataset_sizes["train"])
print("Valid size:", dataset_sizes["valid"])
print("Num classes:", len(class_names))

# ========== 4) 冻结参数（迁移学习核心点） ==========
def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# ========== 5) 初始化模型 ==========
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model = getattr(models, model_name)(weights="DEFAULT" if use_pretrained else None)#新写法
    set_parameter_requires_grad(model, feature_extract)

    # 替换最后的分类层（ResNet是 fc）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

model = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model = model.to(device)

# ========== 6) 只把“需要训练”的参数交给优化器 ==========
params_to_update = [p for p in model.parameters() if p.requires_grad]
print("Trainable params:", sum(p.numel() for p in params_to_update))

optimizer = optim.Adam(params_to_update, lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = nn.CrossEntropyLoss()

# ========== 7) 训练函数（标准模板） ==========
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, save_path):
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 只有训练阶段才开梯度
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)                 # [B, num_classes]
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # 保存验证集上最好的模型
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    "state_dict": best_model_wts,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                print(f"✅ Saved best model to {save_path} (best_acc={best_acc:.4f})")

        # 学习率衰减（每个 epoch 结束）
        scheduler.step()
        print("LR now:", optimizer.param_groups[0]["lr"])

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best valid Acc: {best_acc:.4f}")

    # 训练结束加载最优权重
    model.load_state_dict(best_model_wts)
    return model

#model = train_model(model, dataloaders, criterion, optimizer, scheduler, epochs, save_path)


#顺序：先加载权重参数，再解冻，再建立优化器
#加载之前训练好的权重参数（FC层）
checkpoint = torch.load("my_model.pth")
best_acc = checkpoint["best_acc"]
model.load_state_dict(checkpoint["state_dict"])
#当前FC层训练的比较好，把前面的层解冻，开始每一层权重开始更新，继续训练所有层
for param in model.parameters():
    param.requires_grad = True
#优化器
optimizer = optim.Adam(model.parameters(),lr=lr/10)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#损失函数
criterion = nn.CrossEntropyLoss()


model = train_model(model, dataloaders, criterion, optimizer, scheduler, epochs, save_path)











def im_convert(tensor):
    """把 Tensor 反归一化并转成可显示的 numpy(H,W,C)"""
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
    image = image * np.array(imagenet_std) + np.array(imagenet_mean)  # 反Normalize
    image = np.clip(image, 0, 1)
    return image

@torch.no_grad()
def show_batch_predictions(model, dataloaders, class_names, device, phase="valid",
                           rows=2, cols=4):
    """
    从 dataloaders[phase] 取一个 batch，画图并显示预测/真实类别
    绿色=预测正确，红色=预测错误
    """
    model.eval()

    # 取一个 batch
    dataiter = iter(dataloaders[phase])
    images, labels = next(dataiter)

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)                  # [B, num_classes]
    _, preds = torch.max(outputs, 1)         # [B]

    images_cpu = images.cpu()
    labels_cpu = labels.cpu().numpy()
    preds_cpu = preds.cpu().numpy()

    fig = plt.figure(figsize=(20, 10))
    total = rows * cols

    for idx in range(min(total, images_cpu.size(0))):
        ax = fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(images_cpu[idx]))

        pred_name = class_names[preds_cpu[idx]]
        true_name = class_names[labels_cpu[idx]]
        correct = (preds_cpu[idx] == labels_cpu[idx])

        ax.set_title(f"{pred_name} ({true_name})",
                     color=("green" if correct else "red"))

    plt.tight_layout()
    plt.show()



show_batch_predictions(model, dataloaders, class_names, device, phase="valid", rows=2, cols=4)

import os, time, copy
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets, models

# ========== 0) 超参数 ==========
data_dir = "./flower_data"
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
    model = getattr(models, model_name)(pretrained=use_pretrained)
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

model = train_model(model, dataloaders, criterion, optimizer, scheduler, epochs, save_path)

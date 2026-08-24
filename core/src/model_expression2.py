import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import torch.nn.functional as F

# 新增：早停参数
patience = 10
no_improve = 0
best_acc = 0.0

# 新增：历史记录
train_losses = []
val_losses = []
train_accs = []
val_accs = []

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, min_size=64):
        super().__init__(root, transform)
        filtered_samples = []
        for sample in self.samples:
            path, label = sample
            with Image.open(path) as img:
                w, h = img.size
            if w >= min_size and h >= min_size:
                filtered_samples.append(sample)
        self.samples = filtered_samples
        self.targets = [s[1] for s in self.samples]

# --- 数据预处理 ---

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6,1.4)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),  # 必须在数据增强之前转换为张量
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

data_dir = "binary_classified_emotions"
full_dataset = FilteredImageFolder(data_dir, transform=None, min_size=64)
class_names = full_dataset.classes
num_classes = len(class_names)

# 划分数据集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 设置 transform
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True  # 加速GPU数据传输
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out)) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.05):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.sum(dim=-1)
        loss = (1 - self.epsilon) * loss + self.epsilon * smooth_loss / n_classes
        return loss.mean()

class EmotionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        in_channels = self.backbone.classifier[1].in_features
        
        self.cbam = CBAM(in_channels=in_channels, reduction_ratio=16)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 放前面
            nn.Linear(in_channels, num_classes)
        )

    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionModel(num_classes=num_classes).to(device)

full_dataset = ConcatDataset([train_dataset, val_dataset])
class_counts = np.zeros(num_classes)
for _, label in full_dataset:
    class_counts[label] += 1
    
total_samples = class_counts.sum()
# 修改训练参数
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 增加权重衰减
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)  # 动态学习率

# weights = 1. / class_counts.astype(np.float32)
# weights = torch.tensor(weights, device=device)
# criterion = nn.CrossEntropyLoss(weight=weights)
weights = class_counts.sum() / (num_classes * class_counts + 1e-6)  # 避免除0
weights = weights / weights.sum() * num_classes  # 归一化
weights = torch.tensor(weights, device=device).float()
criterion = nn.CrossEntropyLoss(weight=weights)


num_epochs = 30

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

def evaluate(model, dataloader):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            # 显式转换为CPU并转为int类型
            all_preds.extend(preds.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_acc, all_preds, all_labels

def compute_val_loss(model, val_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_dataset)

def plot_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 记录训练指标
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        # 验证阶段
        val_acc, y_pred, y_true = evaluate(model, val_loader)
        val_loss = compute_val_loss(model, val_loader)  # 需要实现该函数
        scheduler.step(val_loss)
        
        # 记录验证指标
        val_losses.append(val_loss)

        val_accs.append(val_acc.item())
        
        # 打印结果
        print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 保存最佳模型
        if val_acc > best_acc:
            no_improve = 0
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_emotion_model2.pt")
            
            # 保存混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            np.save('best_confusion_matrix.npy', cm)
            plot_confusion_matrix(cm, class_names, 'best_confusion_matrix.png')
            
            print("Saved Best Model and Confusion Matrix")
        else:
            no_improve += 1
        
        # 早停机制
        if no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        print("-" * 20)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_curves.png')
    plt.close()
    
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")
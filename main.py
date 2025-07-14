import os
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 确定训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集

class ChestXRayDataset(Dataset):
    def __init__(self, num_classes = 2, transform = None):
        self.image_names = []
        self.labels = []
        self.transform = transform
        label_dict = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }
        # 定义支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for label in os.listdir(r"D:\pycharm\Projects\Data\chest_xray\train"):
            if label in label_dict:
                for image in os.listdir(fr"D:\pycharm\Projects\Data\chest_xray\train\{label}"):
                    # 检查文件扩展名是否为图像格式
                    if any(image.lower().endswith(ext) for ext in image_extensions):
                        self.image_names.append(os.path.join(fr"D:\pycharm\Projects\Data\chest_xray\train\{label}", image))
                        self.labels.append(label_dict[label])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        image = Image.open(self.image_names[idx])
        # 将灰度图像转换为RGB图像
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


# transform
channel_mean = [0.485, 0.456, 0.406]
channel_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(channel_mean, channel_std)
    ]
)

dataset = ChestXRayDataset(transform = train_transform)

# 划分训练集和验证集
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接，处理通道数不一致的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义简化版ResNet模型
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.in_channels = 16
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 残差块层
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 初始化模型、损失函数和优化器
model = SimpleResNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# TensorBoard设置
writer = SummaryWriter('runs/chest_xray_experiment')

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
    
    # 记录到TensorBoard
    writer.add_scalar('Training Loss', epoch_loss, epoch)
    writer.add_scalar('Training Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, val_loader, criterion, epoch, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    print(f'Validation Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
    
    # 记录到TensorBoard
    writer.add_scalar('Validation Loss', epoch_loss, epoch)
    writer.add_scalar('Validation Accuracy', epoch_acc, epoch)
    
    # 学习率调整
    scheduler.step(epoch_loss)
    
    return epoch_loss, epoch_acc

# 训练主循环
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, epoch, device)
    val_loss, val_acc = validate(model, val_dataloader, criterion, epoch, device)
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        print(f'Saving best model with val_loss: {val_loss:.4f}')
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

print('Training completed!')
writer.close()
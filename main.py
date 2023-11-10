import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 检查CUDA是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this code on a machine with CUDA.")
else:
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集和验证集
train_dataset = datasets.ImageFolder(root='cats_and_dogs_train', transform=transform)
valid_dataset = datasets.ImageFolder(root='cats_and_dogs_valid', transform=transform)

# 模型定义
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(2048)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048 * 1 * 1, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))
        x = self.pool(self.bn7(F.relu(self.conv7(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

# 初始化权重
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)

# 设备设置
device = torch.device("cuda:0")

# 早停法设置
early_stopping_patience = 10  # 设置早停的耐心值，例如 10 epochs
no_improvement_count = 0  # 用于跟踪性能没有改善的 epochs 数量

# 训练函数
def train_model(lr, batch_size):
    global no_improvement_count  # 使用全局变量以跟踪状态
    # 数据加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 模型实例化并应用权重初始化
    model = ImprovedNet().to(device)
    model.apply(weights_init)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    criterion = nn.CrossEntropyLoss()

    # 训练和验证过程
    best_valid_acc = 0.0
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        scheduler.step(train_loss)

        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_acc = 100 * correct_valid / total_valid

        print(f"Epoch [{epoch + 1}/70], Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}, Train Accuracy: {train_acc:.2f}%, Valid Accuracy: {valid_acc:.2f}%")

        # 更新最佳验证准确率和计数器
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            no_improvement_count = 0  # 重置计数器
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improvement_count += 1

        # 检查是否达到早停条件
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return best_valid_acc

# 如果是直接运行main.py，则执行一次训练
if __name__ == "__main__":
    train_model(lr=0.001, batch_size=256)
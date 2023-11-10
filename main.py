import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights  # 导入 models
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 检查CUDA是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this code on a machine with CUDA.")
else:
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")

# 数据预处理和增强
# 调整预处理以符合 ResNet 的输入要求
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 训练集和验证集
train_dataset = datasets.ImageFolder(root='cats_and_dogs_train', transform=transform)
valid_dataset = datasets.ImageFolder(root='cats_and_dogs_valid', transform=transform)

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        # 加载预训练的 ResNet，使用新的 weights 参数
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 冻结所有层
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 替换最后一层
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet(x)


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
    model = ResNetModel().to(device)
    model.apply(weights_init)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    criterion = nn.CrossEntropyLoss()

    # 训练和验证过程
    best_valid_acc = 0.0
    for epoch in range(30):
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

        print(f"Epoch [{epoch + 1}/30], Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}, Train Accuracy: {train_acc:.2f}%, Valid Accuracy: {valid_acc:.2f}%")

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
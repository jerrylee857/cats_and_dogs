#后台执行  nohup python hyperparam_tuning.py > result.log 2>&1 &
#终端与日志共同输出  python hyperparam_tuning.py | tee result.log


from main import train_model
import itertools

# 定义超参数搜索空间
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [128, 256, 512]

best_acc = 0
best_params = {}

# 遍历所有的超参数组合
for lr, batch_size in itertools.product(learning_rates, batch_sizes):
    print(f"Training with lr: {lr}, batch_size: {batch_size}")
    valid_acc = train_model(lr, batch_size)
    
    # 更新最佳模型
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_params = {'lr': lr, 'batch_size': batch_size}

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best Parameters: {best_params}")

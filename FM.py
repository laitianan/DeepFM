import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.onnx
import onnx
import onnxruntime as ort

class FactorizationMachine(nn.Module):
    """
    完整的FM模型实现，支持稀疏数据和高维特征
    """
    def __init__(self, num_features, factor_dim=10):
        super(FactorizationMachine, self).__init__()
        self.num_features = num_features
        self.factor_dim = factor_dim
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶线性项权重
        self.linear_weights = nn.Embedding(num_features, 1)
        # 二阶交叉项隐向量
        self.factor_vectors = nn.Embedding(num_features, factor_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.linear_weights.weight)
        nn.init.xavier_uniform_(self.factor_vectors.weight)
        nn.init.constant_(self.bias, 0.0)
    
    def forward(self, feature_indices, feature_values=None):
        """
        前向传播
        Args:
            feature_indices: 非零特征的索引 [batch_size, num_nonzero_features]
            feature_values: 非零特征的值 [batch_size, num_nonzero_features], 如果为None则默认为1
        """
        if feature_values is None:
            feature_values = torch.ones_like(feature_indices, dtype=torch.float32)
        else:
            feature_values = feature_values.float()
        
        # 一阶项计算
        linear_emb = self.linear_weights(feature_indices)  # [batch_size, num_nonzero, 1]
        linear_term = torch.sum(linear_emb.squeeze(-1) * feature_values, dim=1)  # [batch_size]
        
        # 二阶交叉项计算（优化后的公式）
        factor_emb = self.factor_vectors(feature_indices)  # [batch_size, num_nonzero, factor_dim]
        factor_emb_weighted = factor_emb * feature_values.unsqueeze(-1)  # 乘以特征值
        
        # 使用优化公式计算二阶项: 0.5 * sum((sum v_i*x_i)^2 - sum(v_i^2*x_i^2))
        sum_squared = torch.sum(factor_emb_weighted, dim=1) ** 2  # [batch_size, factor_dim]
        squared_sum = torch.sum(factor_emb_weighted ** 2, dim=1)  # [batch_size, factor_dim]
        
        interaction_term = 0.5 * torch.sum(sum_squared - squared_sum, dim=1)  # [batch_size]
        
        # 最终输出
        output = self.bias + linear_term + interaction_term
        return torch.sigmoid(output)  # 适用于二分类

class SparseDataset(Dataset):
    """处理稀疏数据的自定义数据集"""
    def __init__(self, features, labels):
        self.features = features  # 列表的列表，每个内部列表是样本的非零特征索引
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

def create_sample_data(num_samples=1000, num_features=100, sparsity=0.95):
    """创建示例稀疏数据"""
    features = []
    labels = []
    
    for _ in range(num_samples):
        # 随机选择少量非零特征（模拟稀疏性）
        num_nonzero = max(1, int(num_features * (1 - sparsity)))
        nonzero_indices = np.random.choice(num_features, num_nonzero, replace=False)
        features.append(nonzero_indices)
        labels.append(np.random.randint(0, 2))  # 二分类标签
    
    return features, np.array(labels)

def collate_sparse_batch(batch):
    """处理稀疏数据的批处理函数"""
    features, labels = zip(*batch)
    
    # 找到本批次中最长的序列长度
    max_len = max(len(f) for f in features)
    
    # 填充特征索引和创建掩码
    padded_features = []
    feature_values = []
    
    for feature in features:
        padded = torch.cat([feature, torch.zeros(max_len - len(feature), dtype=torch.long)])
        padded_features.append(padded)
        # 创建特征值（稀疏数据中通常为1，但这里保留扩展性）
        values = torch.cat([torch.ones(len(feature)), torch.zeros(max_len - len(feature))])
        feature_values.append(values)
    
    return (torch.stack(padded_features), 
            torch.stack(feature_values), 
            torch.stack(labels))

# 创建示例数据
print("创建训练数据...")
train_features, train_labels = create_sample_data(1000, num_features=500)
test_features, test_labels = create_sample_data(200, num_features=500)

# 创建数据加载器
train_dataset = SparseDataset(train_features, train_labels)
test_dataset = SparseDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_sparse_batch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_sparse_batch)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FactorizationMachine(num_features=500, factor_dim=10).to(device)
print(f"使用设备: {device}")

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.BCELoss()

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for feature_indices, feature_values, labels in dataloader:
        feature_indices = feature_indices.to(device)
        feature_values = feature_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(feature_indices, feature_values)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted.squeeze() == labels.squeeze()).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for feature_indices, feature_values, labels in dataloader:
            feature_indices = feature_indices.to(device)
            feature_values = feature_values.to(device)
            labels = labels.to(device)
            
            outputs = model(feature_indices, feature_values)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted.squeeze() == labels.squeeze()).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

# 训练模型
print("开始训练...")
for epoch in range(20):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    
    if epoch % 5 == 0:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

print("训练完成!")

# 最终评估
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'最终测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}')





def save_model(model, optimizer, epoch, path):
    """
    保存完整的训练状态，包括模型参数、优化器状态和当前epoch[2,6](@ref)
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
    }, path)
    print(f"模型已保存至: {path}")

# 保存最终模型
save_model(model, optimizer, 20, 'fm_model_complete.pth')


# 仅保存模型参数（推荐用于部署）[2,3](@ref)
torch.save(model.state_dict(), 'fm_model_state_dict.pth')
print("模型参数已保存至: fm_model_state_dict.pth")

# -------------------- 模型加载 --------------------
def load_model(model, optimizer, path, device):
    """
    加载完整训练状态，用于继续训练或推理[2,6](@ref)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.to(device)
    model.eval()  # 设置为评估模式[2](@ref)
    
    print(f"加载模型完成: 训练轮数={epoch}, 损失={loss:.4f}")
    return model, optimizer, epoch, loss

# 示例：加载模型用于继续训练
# 首先需要重新初始化模型和优化器
loaded_model = FactorizationMachine(num_features=500, factor_dim=10).to(device)
loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)

# 加载保存的检查点
loaded_model, loaded_optimizer, start_epoch, loaded_loss = load_model(
    loaded_model, loaded_optimizer, 'fm_model_complete.pth', device
)

# 示例：仅加载模型参数用于推理（推荐方式）[3](@ref)
inference_model = FactorizationMachine(num_features=500, factor_dim=10).to(device)
inference_model.load_state_dict(torch.load('fm_model_state_dict.pth', map_location=device))
inference_model.eval()  # 设置为评估模式[2](@ref)
print("模型参数加载完成，可用于推理")
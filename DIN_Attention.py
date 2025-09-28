# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 17:48:09 2024

@author: admin
"""

import torch 
import torch.nn as nn 

import torch.nn.functional as F

class DICE(nn.Module):
    """
    DICE激活函数实现
    特点：
    1. 自适应调整激活阈值
    2. 平滑过渡的激活状态
    3. 适合处理推荐系统中的稀疏特征
    """
    def __init__(self, dim=None, epsilon=1e-8):
        """
        Args:
            dim: 指定输入特征维度（用于参数化版本）
            epsilon: 数值稳定性的小常数
        """
        super(DICE, self).__init__()
        self.epsilon = epsilon
        
        # 如果是参数化版本（推荐）
        if dim is not None:
            self.alpha = nn.Parameter(torch.zeros(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.alpha = None
            self.beta = None
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状可以是任意维度
        Returns:
            经过DICE激活的输出，形状与输入相同
        """
        # 1. 计算均值和方差（沿特征维度）
        if x.dim() <= 2:
            # 对于2D输入 (batch_size, features)
            axes = list(range(x.dim() - 1))
        else:
            # 对于更高维输入（如序列数据）
            axes = list(range(1, x.dim()))
        
        mean = x.mean(dim=axes, keepdim=True)
        var = x.var(dim=axes, keepdim=True)
        
        # 2. 标准化得分
        # 添加epsilon防止除零
        z = (x - mean) / torch.sqrt(var + self.epsilon)
        
        # 3. 计算激活概率
        if self.alpha is not None:
            # 参数化版本：p(z) = sigmoid(beta * (z - alpha))
            p = torch.sigmoid(self.beta * (z - self.alpha))
        else:
            # 原始版本：p(z) = sigmoid(z)
            p = torch.sigmoid(z)
        
        # 4. 应用激活
        return x * p


class DIN_Attention(nn.Module):
    """
    DIN (Deep Interest Network) 注意力机制
    实现对用户历史行为序列的加权聚合
    """
    
    def __init__(self, idim, hdim, odim):
        """
        Args:
            idim: 输入维度 (单个embedding的维度)
            hdim: 隐藏层维度
            odim: 输出维度
        """
        super(DIN_Attention, self).__init__()
        self.idim = idim
        
        # 注意力网络：计算候选商品与历史行为的相关性权重
        # 输入: [source, target, source*target] 拼接，所以是 idim*4
        self.attention_net = nn.Sequential(
            nn.Linear(idim * 3, hdim),
            DICE(),
            nn.Linear(hdim, 1),
        )
        
        # 输出投影层（可选）
        self.output_proj = nn.Linear(idim, odim) if odim != idim else nn.Identity()
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """Xavier初始化所有线性层参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, source, target, mask=None):
        """
        Args:
            source: 用户历史行为序列 [batch_size, seq_len, idim]
            target: 候选商品embedding [batch_size, idim]
            mask: 序列掩码 [batch_size, seq_len] (可选)
        
        Returns:
            weighted_sequence: 加权后的用户表示 [batch_size, odim]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, idim = source.shape
        
        # 1. 扩展target向量以匹配序列长度
        # target: [batch_size, idim] -> [batch_size, seq_len, idim]
        target_expanded = target.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 2. 计算交互特征
        # 元素级乘法：捕捉细粒度交互
        element_wise_product = source * target_expanded
        
        # 3. 拼接特征：源序列 + 目标 + 交互特征
        # 注意：这里使用torch.cat而不是torch.concat（两者功能相同，但cat更常用）
        combined = torch.cat([source, target_expanded, element_wise_product], dim=-1)
        # combined形状: [batch_size, seq_len, idim*3]
        
        # 4. 计算注意力权重
        attention_scores = self.attention_net(combined).squeeze(-1)
        # attention_scores形状: [batch_size, seq_len]
        
        # 5. 应用掩码（如果提供）
        if mask is not None:
            # 将填充位置的注意力权重设置为极小值
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 6. 应用softmax得到归一化权重
        # attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_scores
        # attention_weights形状: [batch_size, seq_len]
        
        # 7. 加权聚合序列
        # [batch_size, seq_len, idim] * [batch_size, seq_len, 1] -> [batch_size, idim]
        weighted_sequence = torch.sum(source * attention_weights.unsqueeze(-1), dim=1)
        
        # 8. 输出投影
        output = self.output_proj(weighted_sequence)
        
        return output, attention_weights

# 测试代码
def test_din_attention():
    # 模拟数据
    batch_size, seq_len, embedding_dim = 32, 10, 64
    output_dim = 32
    
    # 用户历史行为序列：10个历史商品
    source = torch.randn(batch_size, seq_len, embedding_dim)
    # 候选商品
    target = torch.randn(batch_size, embedding_dim)
    # 序列掩码（模拟变长序列）
    mask = torch.ones(batch_size, seq_len)
    mask[:, 7:] = 0  # 后3个位置是填充的
    
    # 初始化模型
    model = DIN_Attention(
        idim=embedding_dim, 
        hdim=128, 
        odim=output_dim
    )
    # model.eval()
    # 前向传播
    weighted_output, attn_weights = model(source, target, mask)
    
    print("输入形状:")
    print(f"source: {source.shape}")
    print(f"target: {target.shape}")
    print(f"输出形状: {weighted_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重示例:\n{attn_weights[0]}")
    
    return weighted_output, attn_weights

# 运行测试
if __name__ == "__main__":
    weighted_output, attn_weights=test_din_attention()
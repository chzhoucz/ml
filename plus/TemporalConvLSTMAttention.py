import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# 常量定义
INPUT_LENGTH = 90    # 输入序列长度
OUTPUT_LENGTH = 90   # 输出序列长度
NUM_EXPERIMENTS = 5  # 实验次数
Learning_Rate = 0.004 # 学习率

# 数据加载    
train_path = "data/pre_data/train.csv"
test_path = "data/pre_data/test.csv"

trainData = pd.read_csv(train_path)
testData = pd.read_csv(test_path)
    
trainData = trainData.drop(["Date"], axis=1)
testData = testData.drop(["Date"], axis=1)

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(trainData)
test_scaled = scaler.transform(testData)

# 创建序列数据
def create_sequences(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i+input_length, :-1])
        y.append(data[i+input_length:i+input_length+output_length, -1])
    return np.array(X), np.array(y)

# 准备训练集和测试集
train_seq_X, train_seq_y = create_sequences(train_scaled, INPUT_LENGTH, OUTPUT_LENGTH)
test_seq_X, test_seq_y = create_sequences(test_scaled, INPUT_LENGTH, OUTPUT_LENGTH)
X_train = torch.FloatTensor(train_seq_X)
y_train = torch.FloatTensor(train_seq_y)
X_test = torch.FloatTensor(test_seq_X)
y_test = torch.FloatTensor(test_seq_y)

# 创建数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TemporalConvLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_len=OUTPUT_LENGTH, conv_kernels=[3,5,7]):
        super().__init__()
        self.output_len = output_len
        
        # 1. 多尺度卷积
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, k, padding=k//2),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim)
            ) for k in conv_kernels
        ])
        
        # 2. 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim*len(conv_kernels),
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # 3. 时间注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 4. 统一输出头（关键修改）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)  # 严格匹配目标长度
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.permute(0, 2, 1)  # [B, D, T]
        
        # 多尺度卷积
        conv_outs = [conv(x) for conv in self.conv_branches]
        conv_merged = torch.cat(conv_outs, dim=1)  # [B, C*len(kernels), T]
        
        # LSTM处理
        lstm_out, _ = self.lstm(conv_merged.permute(0, 2, 1))  # [B, T, 2*hidden_dim]
        
        # 注意力加权
        attn_weights = self.attention(lstm_out)  # [B, T, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 2*hidden_dim]
        
        # 最终输出（不再拼接）
        return self.fc(context)  # [B, output_len]

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TemporalConvLSTMAttention(X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

# 训练模型
num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))  # 更宽视图，更适合时间序列

# 跳过第一个 loss（从第2个 epoch 开始画）
plt.plot(range(2, len(train_losses) + 1), train_losses[1:], label='Training Loss', color='blue', linewidth=2)

# 设置字体大小
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title(f'ConvLSTMAttention Training Loss ({OUTPUT_LENGTH}_day)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# 自动调整布局，防止文字被裁剪
plt.tight_layout()

# 保存高分辨率图像
plt.savefig(f'plus/ConvLSTMAttention_{OUTPUT_LENGTH}_training_loss.png', dpi=300)
# 多次实验评估
mse_scores = []
mae_scores = []

for i in range(NUM_EXPERIMENTS):
    max_idx = len(X_test) - 1
    idx = random.randint(0, max_idx)  # 每次取不同位置的数据
    partial_X_test = X_test[idx:idx+1].to(device)
    partial_y_test = y_test[idx:idx+1].to(device)

    with torch.no_grad():
        y_pred = model(partial_X_test).cpu().numpy()[0]  # 去掉批次维度

    y_true = partial_y_test[0].cpu().numpy()
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'MSE: {mse:.4f} ')
    print(f'MAE: {mae:.4f} ')
    mse_scores.append(mse)
    mae_scores.append(mae)

    # 反归一化预测结果
    y_pred_reshaped = np.zeros((OUTPUT_LENGTH, partial_X_test.shape[2] + 1))  # 修改形状
    y_pred_reshaped[:, :-1] = np.tile(partial_X_test[0, -1, :].cpu().numpy(), (OUTPUT_LENGTH, 1))
    y_pred_reshaped[:, -1] = y_pred  # 使用所有预测值
    y_pred = scaler.inverse_transform(y_pred_reshaped)[:, -1]

    # 反归一化真实值
    y_test_reshaped = np.zeros((OUTPUT_LENGTH, partial_X_test.shape[2] + 1))  # 修改形状
    y_test_reshaped[:, :-1] = np.tile(partial_X_test[0, -1, :].cpu().numpy(), (OUTPUT_LENGTH, 1))
    y_test_reshaped[:, -1] = partial_y_test[0].cpu().numpy()  # 使用所有真实值
    y_test_real = scaler.inverse_transform(y_test_reshaped)[:, -1]

    # 获取输入序列的最后96个点
    input_reshaped = np.zeros((INPUT_LENGTH, partial_X_test.shape[2] + 1))
    input_reshaped[:, :-1] = partial_X_test[0, :, :].cpu().numpy()
    input_reshaped[:, -1] = partial_X_test[0, :, -1].cpu().numpy()
    input_seq = scaler.inverse_transform(input_reshaped)[:, -1]

    # 绘制组合曲线图
    plt.figure(figsize=(15, 6))
    plt.plot(range(OUTPUT_LENGTH), y_pred, 'r-', label='Prediction')
    plt.plot(range(OUTPUT_LENGTH), y_test_real, 'g--', label='Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'ConvLSTMAttention Prediction({OUTPUT_LENGTH}_day)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plus/ConvLSTMAttention{OUTPUT_LENGTH}_prediction_{i+1}.png')
    plt.close()

# 输出最终评估结果
print(f'ConvLSTMAttention {OUTPUT_LENGTH}_day_ouput ')
print(f'MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}')
print(f'MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}')